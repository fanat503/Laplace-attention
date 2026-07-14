# Copyright 2026 Slyatski Ilya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Base-vs-HLA divergence over training (reviewer: "how fast do they separate?").

For step-matched checkpoint pairs computes, on IDENTICAL validation tokens:
  - cosine similarity of flattened logits (global picture)
  - mean per-token KL( base || hla ) over next-token distributions
    (distribution-level divergence; cosine alone can hide calibrated shifts)
  - val loss of each model on the probe batch (sanity anchor)

Works with both checkpoint layouts produced by this repo:
  - resume payloads: {"model": state_dict, "config": {...}, "step": N}
  - bf16 weight dumps (best_val_*/final_*): bare state_dict (config required)

Reads token data via src.data.FixedDataset - the SAME loader as training
(int16/int32/int64 .pt or .bin+sidecar; never assumes a dtype).

Usage:
  python scripts/plot_divergence.py \
      --base-config configs/200m_base_s42.json \
      --hla-config  configs/200m_hla_s42.json \
      --val-data    data/val_fixed_tokens.bin \
      --pairs step_5000_base.pt:step_5000_hla.pt step_10000_base.pt:step_10000_hla.pt \
      --out runs/divergence.csv --plot runs/divergence.png

Emits CSV always; PNG only if matplotlib is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402


def load_model(ckpt_path: str, config_path: str) -> tuple[GPT, int]:
    """Load either a resume payload or a bare state_dict; return (model, step)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = GPT(GPTConfig(**cfg["model"]))
    # Security (reviewer attack K4): this is a public analysis script - try the
    # safe deserializer first; fall back to full unpickling ONLY for our own
    # resume payloads (which embed a config dict and need it).
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = -1
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
        step = int(payload.get("step", -1))
    else:
        state = payload
    if step < 0:
        m = re.search(r"step_(\d+)", os.path.basename(ckpt_path))
        step = int(m.group(1)) if m else -1
    # bf16 weight dumps load fine into fp32 modules via strict load + cast
    state = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, step


@torch.no_grad()
def divergence_metrics(base: GPT, hla: GPT, x: torch.Tensor, y: torch.Tensor) -> dict:
    lb, loss_b = base(x, y)
    lh, loss_h = hla(x, y)
    lb32, lh32 = lb.float(), lh.float()
    if base.config.vocab_size != hla.config.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: base={base.config.vocab_size}, "
            f"hla={hla.config.vocab_size} - KL over different supports is meaningless"
        )
    if lb32.shape != lh32.shape:
        raise ValueError(f"logit shape mismatch: {tuple(lb32.shape)} vs {tuple(lh32.shape)}")
    cos = F.cosine_similarity(lb32.reshape(1, -1), lh32.reshape(1, -1)).item()
    # mean per-token KL(base || hla) over the true vocab slice
    V = base.config.vocab_size
    logp_b = F.log_softmax(lb32[..., :V], dim=-1)
    logp_h = F.log_softmax(lh32[..., :V], dim=-1)
    kl = (logp_b.exp() * (logp_b - logp_h)).sum(-1).mean().item()
    return {
        "cosine_sim": cos,
        "kl_base_hla": kl,
        "val_loss_base": float(loss_b),
        "val_loss_hla": float(loss_h),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--hla-config", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="base_ckpt:hla_ckpt per training step, colon-separated")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="default: model block_size")
    ap.add_argument("--out", default="divergence.csv")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        seq_len = args.seq_len or int(json.load(f)["model"]["block_size"])
    ds = FixedDataset(args.val_data, seq_len, return_dict=False, cast_to_long=True)
    chunks = torch.stack([ds[i] for i in range(min(args.batch_size, len(ds)))])
    x, y = chunks[:, :-1], chunks[:, 1:]

    rows = []
    for pair in args.pairs:
        bc, hc = pair.split(":")
        base, sb = load_model(bc, args.base_config)
        hla, sh = load_model(hc, args.hla_config)
        if sb >= 0 and sh >= 0 and sb != sh:
            print(f"[WARN] step mismatch in pair: base={sb}, hla={sh}")
        m = divergence_metrics(base, hla, x, y)
        m["step"] = max(sb, sh)
        rows.append(m)
        print(f"step={m['step']:>7} cos={m['cosine_sim']:.6f} "
              f"KL={m['kl_base_hla']:.6f} "
              f"loss_b={m['val_loss_base']:.4f} loss_h={m['val_loss_hla']:.4f}")

    rows.sort(key=lambda r: r["step"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "cosine_sim", "kl_base_hla",
                                          "val_loss_base", "val_loss_hla"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - CSV written, plot skipped")
            return
        steps = [r["step"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(steps, [r["cosine_sim"] for r in rows], "o-", color="tab:blue",
                 label="cosine(logits)")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("cosine similarity", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(steps, [r["kl_base_hla"] for r in rows], "s--", color="tab:red",
                 label="KL(base||hla)")
        ax2.set_ylabel("mean per-token KL", color="tab:red")
        ax1.set_title("Base vs HLA divergence over training")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
