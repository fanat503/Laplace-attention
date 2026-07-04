"""Unit tests for pure trainer helpers (no TPU needed).

train_xla.py imports torch_xla at module level; on CPU-only CI we inject a
minimal stub so the pure-Python helpers (lr schedule, sharded samplers,
config validation, overrides) can be tested. The stub is NOT used by the
functions under test.
"""
from __future__ import annotations

import math
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _install_torch_xla_stub() -> None:
    if "torch_xla" in sys.modules:
        return
    xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    xm = types.ModuleType("torch_xla.core.xla_model")
    xm.get_ordinal = lambda: 0
    xm.get_local_ordinal = lambda: 0
    xm.xrt_world_size = lambda: 1
    xm.mark_step = lambda: None
    xm.rendezvous = lambda tag: None
    dist = types.ModuleType("torch_xla.distributed")
    pl = types.ModuleType("torch_xla.distributed.parallel_loader")
    pl.MpDeviceLoader = object
    xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn = lambda *a, **k: None
    debug = types.ModuleType("torch_xla.debug")
    metrics = types.ModuleType("torch_xla.debug.metrics")
    metrics.metrics_report = lambda: ""
    xla.core = core
    core.xla_model = xm
    xla.distributed = dist
    dist.parallel_loader = pl
    dist.xla_multiprocessing = xmp
    xla.debug = debug
    debug.metrics = metrics
    for name, mod in {
        "torch_xla": xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xm,
        "torch_xla.distributed": dist,
        "torch_xla.distributed.parallel_loader": pl,
        "torch_xla.distributed.xla_multiprocessing": xmp,
        "torch_xla.debug": debug,
        "torch_xla.debug.metrics": metrics,
    }.items():
        sys.modules[name] = mod


_install_torch_xla_stub()

from src.train_xla import (  # noqa: E402
    EvenShardedSequentialSampler,
    ShardedSequentialSampler,
    apply_override,
    get_lr,
    validate_config,
    validate_init_config_compatibility,
    validate_resume_config_compatibility,
)


class TestLrSchedule:
    def test_warmup_starts_above_zero_and_reaches_base(self):
        # step 0 must have non-zero lr (dead first update otherwise)
        assert get_lr(0, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4) > 0
        assert math.isclose(
            get_lr(99, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-3
        )

    def test_cosine_endpoints(self):
        # right at warmup end: base_lr; at max_steps: min_lr
        assert math.isclose(get_lr(100, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-3)
        assert math.isclose(get_lr(1000, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-4)

    def test_monotone_decay_after_warmup(self):
        prev = float("inf")
        for s in range(100, 1001, 50):
            lr = get_lr(s, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4)
            assert lr <= prev + 1e-12
            prev = lr

    def test_never_below_min_lr(self):
        for s in range(0, 1100, 37):
            lr = get_lr(s, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4)
            if s >= 100:
                assert lr >= 1e-4 - 1e-12


class TestEvenSampler:
    def test_no_duplicates_no_out_of_range(self):
        for n, ws, bs in [(103, 4, 2), (64, 8, 1), (1000, 8, 4), (17, 2, 3)]:
            seen = []
            for r in range(ws):
                s = EvenShardedSequentialSampler(n, rank=r, world_size=ws, batch_size=bs)
                seen.extend(list(s))
            assert len(seen) == len(set(seen)), f"dup at n={n},ws={ws},bs={bs}"
            assert all(0 <= i < n for i in seen)

    def test_equal_length_across_ranks(self):
        for r in range(8):
            s = EvenShardedSequentialSampler(1001, rank=r, world_size=8, batch_size=2)
            assert len(s) == len(EvenShardedSequentialSampler(1001, rank=0, world_size=8, batch_size=2))

    def test_resume_is_exact_suffix(self):
        full = list(EvenShardedSequentialSampler(103, rank=1, world_size=4, batch_size=2))
        resumed = list(EvenShardedSequentialSampler(103, rank=1, world_size=4, batch_size=2,
                                                    start_local_sample=6))
        assert full[6:] == resumed

    def test_resume_beyond_end_is_empty(self):
        s = EvenShardedSequentialSampler(100, rank=0, world_size=4, batch_size=2,
                                         start_local_sample=10_000)
        assert len(list(s)) == 0

    def test_length_multiple_of_batch(self):
        for start in (0, 1, 3, 7):
            s = EvenShardedSequentialSampler(103, rank=0, world_size=4, batch_size=2,
                                             start_local_sample=start)
            assert len(s) % 2 == 0


class TestValSampler:
    def test_covers_everything_once(self):
        n, ws = 101, 8
        seen = []
        for r in range(ws):
            seen.extend(list(ShardedSequentialSampler(n, r, ws)))
        assert sorted(seen) == list(range(n))

    def test_len_matches_iter(self):
        for r in range(8):
            s = ShardedSequentialSampler(101, r, 8)
            assert len(list(s)) == len(s)


class TestApplyOverride:
    def test_nested_json_types(self):
        cfg = {"model": {"n_layer": 2}, "lr": 1e-3}
        apply_override(cfg, "model.n_layer=24")
        apply_override(cfg, "lr=0.0005")
        apply_override(cfg, "model.use_rope=true")
        apply_override(cfg, "run_name=test_run")
        assert cfg["model"]["n_layer"] == 24
        assert cfg["lr"] == 0.0005
        assert cfg["model"]["use_rope"] is True
        assert cfg["run_name"] == "test_run"

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError):
            apply_override({}, "no_equals_sign")


class TestOptimizerGroups:
    """HLA params must be excluded from weight decay: zero IS their identity
    state, so decay would be a constant force against the mechanisms."""

    def test_hla_params_in_nodecay_group(self):
        from src.model import GPT, GPTConfig
        from src.train_xla import make_optimizer

        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        model = GPT(cfg)
        opt = make_optimizer(model, {"lr": 1e-3, "weight_decay": 0.1})
        decay_params = {id(p) for p in opt.param_groups[0]["params"]}
        hla_names = ("W_phase", "W_gate", "W_range", "W_layer_temp")
        leaked = [n for n, p in model.named_parameters()
                  if any(m in n for m in hla_names) and id(p) in decay_params]
        assert leaked == [], f"HLA params leaked into decay group: {leaked}"

    def test_backbone_matrices_still_decayed(self):
        from src.model import GPT, GPTConfig
        from src.train_xla import make_optimizer

        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        model = GPT(cfg)
        opt = make_optimizer(model, {"lr": 1e-3, "weight_decay": 0.1})
        decay_params = {id(p) for p in opt.param_groups[0]["params"]}
        attn_w = dict(model.named_parameters())["transformer.h.0.attn.c_attn.weight"]
        assert id(attn_w) in decay_params


class TestValidateConfig:
    def _minimal(self, tmp_path):
        p = str(tmp_path / "t.pt")
        torch.save(torch.ones(10, dtype=torch.int32), p)
        return {
            "seed": 42, "save_dir": "/tmp/x", "train_path": p, "val_path": p,
            "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
            "grad_accum": 1, "max_steps": 10, "lr": 1e-3, "min_lr": 1e-4,
            "warmup": 2, "model": {"block_size": 64, "vocab_size": 256},
        }

    def test_valid_passes(self, tmp_path):
        validate_config(self._minimal(tmp_path))

    def test_missing_key_rejected(self, tmp_path):
        cfg = self._minimal(tmp_path)
        del cfg["lr"]
        with pytest.raises(KeyError):
            validate_config(cfg)

    def test_nonexistent_data_rejected(self, tmp_path):
        cfg = self._minimal(tmp_path)
        cfg["train_path"] = "/nonexistent/file.bin"
        with pytest.raises(FileNotFoundError):
            validate_config(cfg)


class TestCompatChecks:
    def test_resume_mismatch_rejected(self):
        cur = {"seed": 42, "model": {"n_layer": 24}}
        saved = {"seed": 43, "model": {"n_layer": 24}}
        with pytest.raises(ValueError):
            validate_resume_config_compatibility(cur, saved)

    def test_resume_match_ok(self):
        cur = {"seed": 42, "model": {"n_layer": 24}}
        validate_resume_config_compatibility(cur, dict(cur))

    def test_init_shape_mismatch_rejected(self):
        cur = {"model": {"n_embd": 1024}}
        saved = {"model": {"n_embd": 1408}}
        with pytest.raises(ValueError):
            validate_init_config_compatibility(cur, saved)

    def test_none_saved_is_ok(self):
        validate_resume_config_compatibility({"seed": 1}, None)
        validate_init_config_compatibility({"model": {}}, None)
