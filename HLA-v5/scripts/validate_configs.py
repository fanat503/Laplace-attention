"""Validate base/HLA experiment configs for sterile matched runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALLOWED_MAIN_DIFFS = {
    "variant",
    "run_name",
    "save_dir",
    "init_ckpt",
    "model.phase_mult",
    "model.laplace_alpha",
    "model.distance_laplace_alpha",
    "model.baseline_type",
}


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    fa, fb = flatten(a), flatten(b)
    keys = sorted(set(fa) | set(fb))
    return [(k, fa.get(k), fb.get(k)) for k in keys if fa.get(k) != fb.get(k)]


def tokens_per_update(cfg: Dict[str, Any]) -> int:
    return int(cfg["batch_size_per_device"]) * int(cfg["num_cores"]) * int(cfg["model"]["block_size"]) * int(cfg["grad_accum"])


def validate_pair(base_path: str, hla_path: str) -> None:
    base, hla = load(base_path), load(hla_path)
    diffs = diff(base, hla)
    illegal = [x for x in diffs if x[0] not in ALLOWED_MAIN_DIFFS]
    if illegal:
        text = "\n".join(f"  {k}: base={a!r}, hla={b!r}" for k, a, b in illegal)
        raise RuntimeError("Illegal base/HLA config differences:\n" + text)

    for cfg, name in [(base, "base"), (hla, "hla")]:
        for k in ["train_path", "val_path", "init_ckpt", "save_dir", "run_name"]:
            if not cfg.get(k):
                raise RuntimeError(f"{name}: missing/empty {k}")
        if int(cfg["model"]["block_size"]) <= 0:
            raise RuntimeError(f"{name}: invalid block_size")
        if int(cfg["grad_accum"]) <= 0:
            raise RuntimeError(f"{name}: invalid grad_accum")

    print("CONFIG PAIR VALID")
    for label, cfg in [("base", base), ("hla", hla)]:
        tpu = tokens_per_update(cfg)
        total = tpu * int(cfg["max_steps"])
        print(f"{label}: tokens/update={tpu:,} total_tokens={total:,} max_steps={cfg['max_steps']:,}")
    print("allowed diffs:")
    for k, a, b in diffs:
        print(f"  {k}: {a!r} -> {b!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hla", required=True)
    args = ap.parse_args()
    validate_pair(args.base, args.hla)


if __name__ == "__main__":
    main()
