"""Validate train/validation fixed-token datasets for a config.

This is intentionally lightweight: it validates tensor format/range and compares
sample fingerprints to catch accidental train/val aliasing or path mistakes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset  # noqa: E402


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def training_requirements(config):
    batch = int(config["batch_size_per_device"])
    cores = int(config["num_cores"])
    block = int(config["model"]["block_size"])
    grad_accum = int(config["grad_accum"])
    max_steps = int(config["max_steps"])
    required_sequences = batch * cores * grad_accum * max_steps
    required_effective_tokens = required_sequences * block
    required_stored_tokens = required_sequences * (block + 1)
    return {
        "required_sequences": required_sequences,
        "required_effective_tokens": required_effective_tokens,
        "required_stored_tokens": required_stored_tokens,
    }


def validation_requirements(config):
    batch = int(config["eval_batch_size_per_device"])
    cores = int(config["num_cores"])
    block = int(config["model"]["block_size"])
    val_batches = int(config.get("val_batches", 0))
    required_sequences = batch * cores * val_batches
    return {
        "required_sequences_for_full_val_batches": required_sequences,
        "required_effective_tokens_for_full_val_batches": required_sequences * block,
        "required_stored_tokens_for_full_val_batches": required_sequences * (block + 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--full", action="store_true", help="Scan full tensors for min/max; can be slow")
    args = ap.parse_args()

    cfg = load_config(args.config)
    block = int(cfg["model"]["block_size"])
    vocab = cfg.get("expected_vocab_size", cfg["model"].get("vocab_size"))
    train = FixedDataset(cfg["train_path"], block, expected_vocab_size=vocab, validate_full=args.full)
    val = FixedDataset(cfg["val_path"], block, expected_vocab_size=vocab, validate_full=args.full)

    print("TRAIN")
    print(train.summary())
    print("VAL")
    print(val.summary())

    if os.path.abspath(cfg["train_path"]) == os.path.abspath(cfg["val_path"]):
        raise RuntimeError("train_path and val_path point to the same file")
    if train.info.sample_fingerprint == val.info.sample_fingerprint:
        raise RuntimeError(
            "train/val sample fingerprints are identical; this may indicate leakage or duplicated files"
        )

    train_req = training_requirements(cfg)
    val_req = validation_requirements(cfg)
    print("TRAINING REQUIREMENTS")
    for k, v in train_req.items():
        print(f"  {k}: {v:,}")
    print("VALIDATION REQUIREMENTS")
    for k, v in val_req.items():
        print(f"  {k}: {v:,}")

    allow_multiple_epochs = bool(cfg.get("allow_multiple_epochs", False))
    if train.info.n_sequences < train_req["required_sequences"] and not allow_multiple_epochs:
        raise RuntimeError(
            "train dataset is too small for max_steps with allow_multiple_epochs=false: "
            f"have_sequences={train.info.n_sequences:,}, "
            f"required_sequences={train_req['required_sequences']:,}. "
            f"Required stored token ids={train_req['required_stored_tokens']:,}."
        )
    if val.info.n_sequences < val_req["required_sequences_for_full_val_batches"]:
        raise RuntimeError(
            "val dataset is too small for requested val_batches: "
            f"have_sequences={val.info.n_sequences:,}, "
            f"required_sequences={val_req['required_sequences_for_full_val_batches']:,}."
        )

    print("DATA PAIR VALID")


if __name__ == "__main__":
    main()
