"""Attention KL comparison between trained baseline and HLA checkpoints.

Computes D_KL(attn_base || attn_hla) per layer/head on identical validation inputs.
Uses attention capture, so run with modest --seq-len/--batches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402


def resolve_device(name: str):
    if name == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return torch.device(name)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(ckpt: str, config: str | None, device: str) -> tuple[GPT, Dict[str, Any]]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = load_json(config) if config else payload["config"]
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = GPT(GPTConfig(**cfg["model"]))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.set_diagnostics(enabled=True, capture_attention=True)
    return model, cfg


def iter_val(cfg: Dict[str, Any], seq_len: int, batch_size: int, batches: int, device: str):
    ds = FixedDataset(cfg["val_path"], cfg["model"]["block_size"], expected_vocab_size=cfg.get("expected_vocab_size", cfg["model"].get("vocab_size")))
    for i in range(min(batches, len(ds) // batch_size)):
        xs = [ds[i * batch_size + j]["input_ids"][:seq_len] for j in range(batch_size)]
        yield torch.stack(xs).to(device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--hla-checkpoint", required=True)
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--hla-config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-10)
    args = ap.parse_args()

    device = resolve_device(args.device)
    base, base_cfg = load_model(args.base_checkpoint, args.base_config, device)
    hla, hla_cfg = load_model(args.hla_checkpoint, args.hla_config, device)
    if base_cfg["model"]["n_layer"] != hla_cfg["model"]["n_layer"] or base_cfg["model"]["n_head"] != hla_cfg["model"]["n_head"]:
        raise RuntimeError("base/HLA layer/head mismatch")
    L, H = base_cfg["model"]["n_layer"], base_cfg["model"]["n_head"]
    kl_sum = torch.zeros(L, H, dtype=torch.float64)
    n = 0
    for x in iter_val(base_cfg, args.seq_len, args.batch_size, args.batches, device):
        _ = base(x)
        _ = hla(x)
        for li in range(L):
            p = base.transformer.h[li].attn.last_attn
            q = hla.transformer.h[li].attn.last_attn
            if p is None or q is None:
                raise RuntimeError("attention capture missing")
            p = p.float().clamp_min(args.eps)
            q = q.float().clamp_min(args.eps)
            kl = (p * (p.log() - q.log())).sum(dim=-1).mean(dim=(0, 2))
            kl_sum[li] += kl.detach().cpu().double()
        n += 1
    kl = (kl_sum / max(1, n)).numpy()
    out = {
        "base_checkpoint": args.base_checkpoint,
        "hla_checkpoint": args.hla_checkpoint,
        "seq_len": args.seq_len,
        "batches": n,
        "kl_base_to_hla_per_layer_head": kl.tolist(),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
