"""Inspect checkpoint/init files without starting training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HLA_MARKERS = ("W_phase_q", "W_phase_k", "W_range_k", "W_range_v", "W_gate_k", "W_gate_v")


def load(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def state_from_payload(payload: Any) -> Dict[str, torch.Tensor]:
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    payload = load(args.path)
    state = state_from_payload(payload)
    state_numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    unique_param_estimate = state_numel
    # state_dict contains both transformer.wte.weight and lm_head.weight even when
    # tied. Parameter-count scripts use model.named_parameters(); here we avoid
    # double-counting the tied LM head when inspecting raw checkpoints.
    if "transformer.wte.weight" in state and "lm_head.weight" in state:
        a, b = state["transformer.wte.weight"], state["lm_head.weight"]
        if torch.is_tensor(a) and torch.is_tensor(b) and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b):
            unique_param_estimate -= b.numel()
    hla = sum(v.numel() for k, v in state.items() if any(m in k for m in HLA_MARKERS) and torch.is_tensor(v))
    hla_max_abs = 0.0
    for k, v in state.items():
        if any(m in k for m in HLA_MARKERS) and torch.is_tensor(v) and v.numel() > 0:
            hla_max_abs = max(hla_max_abs, float(v.float().abs().max().item()))
    out = {
        "path": args.path,
        "num_tensors": len(state),
        "state_dict_numel": int(state_numel),
        "unique_param_estimate": int(unique_param_estimate),
        "hla_params": int(hla),
        "hla_identity_max_abs": hla_max_abs,
        "has_optimizer": isinstance(payload, dict) and payload.get("optimizer") is not None,
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "best_val": payload.get("best_val") if isinstance(payload, dict) else None,
        "config_hash": payload.get("config_hash") if isinstance(payload, dict) else None,
        "role": payload.get("role") if isinstance(payload, dict) else None,
        "manifest_mode": payload.get("manifest", {}).get("mode") if isinstance(payload, dict) else None,
    }
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        for k, v in out.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
