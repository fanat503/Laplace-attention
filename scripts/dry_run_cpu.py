# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""CPU dry-run of the REAL trainer: stub torch_xla -> torch.device('cpu').

The only thing CI could not tell us before a Kaggle session was "does the
training loop itself run" - --help and unit tests exercise everything except
the loop. This harness runs train_xla.py's actual worker end-to-end on CPU
(single process, tiny config), so integration bugs (CSV wiring, probe calls,
checkpoint paths, resume) die HERE, not in the TPU queue.

Usage:
    python scripts/dry_run_cpu.py --config configs/smoke_hla_s42.json \
        [--override k=v ...]

Exit code 0 = the pilot's code path is alive.
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def install_cpu_xla_stub() -> None:
    if "torch_xla" in sys.modules:
        raise RuntimeError("torch_xla already imported - dry run must own the stub")

    xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    xm = types.ModuleType("torch_xla.core.xla_model")

    xm.REDUCE_SUM = "sum"
    xm.xla_device = lambda: torch.device("cpu")
    xm.get_ordinal = lambda: 0
    xm.get_local_ordinal = lambda: 0
    xm.xrt_world_size = lambda: 1
    xm.is_master_ordinal = lambda local=False: True
    xm.mark_step = lambda: None
    xm.rendezvous = lambda tag: None
    xm.all_reduce = lambda op, value: value
    xm.set_rng_state = lambda seed: torch.manual_seed(int(seed))

    def _reduce_gradients(optimizer):  # single process: nothing to reduce
        return None

    def _optimizer_step(optimizer, **kwargs):
        optimizer.step()

    def _save(payload, path, master_only=True):
        torch.save(payload, path)

    xm.reduce_gradients = _reduce_gradients
    xm.optimizer_step = _optimizer_step
    xm.save = _save

    runtime = types.ModuleType("torch_xla.runtime")
    runtime.global_ordinal = lambda: 0
    runtime.local_ordinal = lambda: 0
    runtime.world_size = lambda: 1

    dist = types.ModuleType("torch_xla.distributed")
    pl = types.ModuleType("torch_xla.distributed.parallel_loader")

    class MpDeviceLoader:  # CPU tensors are already "on device"
        def __init__(self, loader, device, **kwargs):
            self._loader = loader

        def __iter__(self):
            return iter(self._loader)

        def __len__(self):
            return len(self._loader)

    pl.MpDeviceLoader = MpDeviceLoader

    xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")

    def _spawn(fn, args=(), nprocs=None, start_method="fork"):
        fn(0, *args)

    xmp.spawn = _spawn

    debug = types.ModuleType("torch_xla.debug")
    metrics = types.ModuleType("torch_xla.debug.metrics")
    metrics.metrics_report = lambda: "(cpu dry run: no XLA metrics)"

    xla.core = core
    core.xla_model = xm
    xla.runtime = runtime
    xla.distributed = dist
    dist.parallel_loader = pl
    dist.xla_multiprocessing = xmp
    xla.debug = debug
    debug.metrics = metrics

    for name, mod in {
        "torch_xla": xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xm,
        "torch_xla.runtime": runtime,
        "torch_xla.distributed": dist,
        "torch_xla.distributed.parallel_loader": pl,
        "torch_xla.distributed.xla_multiprocessing": xmp,
        "torch_xla.debug": debug,
        "torch_xla.debug.metrics": metrics,
    }.items():
        sys.modules[name] = mod


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU dry-run of the real XLA trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    install_cpu_xla_stub()

    from src.train_xla import apply_override, load_config, train_worker_xla  # noqa: E402

    config = load_config(args.config)
    for item in args.override:
        apply_override(config, item)

    # Single CPU process: the world-size guard would (correctly) refuse
    # num_cores=8 with world=1, so a dry run must always be single-core.
    config["num_cores"] = 1
    config["num_workers"] = 0  # keep the dry run single-process end to end

    train_worker_xla(config)
    print("DRY RUN COMPLETE: trainer code path is alive on CPU")


if __name__ == "__main__":
    main()
