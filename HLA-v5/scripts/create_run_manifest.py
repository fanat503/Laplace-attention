"""Create an auditable manifest for a run config.

The manifest records config hash, file metadata, optional full file hashes,
environment snapshot, and token budget. Use it before TPU launch and keep it with
run artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.manifest import atomic_write_json, build_experiment_manifest, load_json  # noqa: E402


def token_budget(config):
    return {
        "tokens_per_update": int(config["batch_size_per_device"])
        * int(config["num_cores"])
        * int(config["model"]["block_size"])
        * int(config["grad_accum"]),
        "max_steps": int(config["max_steps"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash-large-files", action="store_true", help="Hash train/val/init files fully; can be slow")
    args = ap.parse_args()

    cfg = load_json(args.config)
    budget = token_budget(cfg)
    budget["total_tokens"] = budget["tokens_per_update"] * budget["max_steps"]
    manifest = build_experiment_manifest(
        config=cfg,
        config_path=args.config,
        hash_large_files=args.hash_large_files,
        extra={"token_budget": budget},
    )
    atomic_write_json(args.out, manifest)
    print(f"wrote {args.out}")
    print(f"manifest_hash={manifest['manifest_hash']} config_hash={manifest['config_hash']}")


if __name__ == "__main__":
    main()
