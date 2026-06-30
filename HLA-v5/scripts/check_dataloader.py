"""Check FixedDataset/DataLoader batching invariants before TPU training.

This is a CPU-side structural check: it verifies fixed shapes, dtype, no short
batches in the checked prefix, and enough dataset length for the configured run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset, fixed_token_collate, worker_init_fn  # noqa: E402


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    block = int(cfg["model"]["block_size"])
    batch = int(cfg["batch_size_per_device"])
    workers = int(cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers)
    vocab = cfg.get("expected_vocab_size", cfg["model"].get("vocab_size"))
    ds = FixedDataset(cfg["train_path"], block, expected_vocab_size=vocab, cast_to_long=False)
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        collate_fn=fixed_token_collate,
        persistent_workers=bool(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
    )
    required_sequences = int(cfg["batch_size_per_device"]) * int(cfg["num_cores"]) * int(cfg["grad_accum"]) * int(cfg["max_steps"])
    if len(ds) < required_sequences and not bool(cfg.get("allow_multiple_epochs", False)):
        raise RuntimeError(f"dataset too short: have {len(ds):,} sequences, need {required_sequences:,}")
    for i, item in enumerate(loader):
        if i >= args.batches:
            break
        x = item["input_ids"]
        expected = (batch, block + 1)
        if tuple(x.shape) != expected:
            raise RuntimeError(f"batch {i} shape mismatch: got {tuple(x.shape)}, expected {expected}")
        if str(x.dtype) != "torch.int64":
            raise RuntimeError(f"batch {i} dtype mismatch: got {x.dtype}, expected torch.int64")
        if int(x.min()) < 0 or int(x.max()) >= int(vocab):
            raise RuntimeError(f"batch {i} token range invalid: min={int(x.min())}, max={int(x.max())}, vocab={vocab}")
    print(f"DATALOADER VALID: checked_batches={min(args.batches, len(loader))} batch_shape=({batch}, {block + 1}) workers={workers}")


if __name__ == "__main__":
    main()
