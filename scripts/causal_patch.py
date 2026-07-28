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


"""Causal patching: transplant HLA's RETRIEVAL geometry into the base twin.

THE Oral experiment (Type 5: architecture -> mechanism -> behavior). The
sterile pair guarantees base and HLA share init and data; after training they
differ only through the mechanisms' influence. This script builds a FRANKEN
model: HLA's retrieval side (Q/K projections, phases, K-gate, score biases,
q-temp) spliced onto base's transmission side (V projection, output proj,
MLPs, embeddings), then measures retrieval behavior (induction, distractor
margin, positional recall).

Causal logic, pre-registered in EXPERIMENT_CARD (H5):
  - franken ~ HLA on retrieval probes  => retrieval geometry CARRIES the gain
    (the mechanistic claim becomes causal, not correlational);
  - franken ~ base                     => the gain lives in the V/MLP path,
    and the "cleaner retrieval" story is NOT the cause - report honestly.

Transplant sets (--transplant):
  qk        : Q,K rows of every c_attn only (pure geometry, no mechanisms)
  phase     : qk + W_phase_q/k + W_phase_scale
  retrieval : phase + K-side gate/range + salience + distance + forget + qtemp
              (everything inside the softmax; V-side stays base)
  full      : every weight from HLA (sanity: franken == HLA exactly)

Note W_layer_temp is deliberately NOT transplanted in 'retrieval' (it scales
both K- and V-side envelopes - mixed allegiance; documented limitation).

Works on CPU. Ground-truth tests in tests/test_eval.py::TestCausalPatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402
from src.eval import (  # noqa: E402
    evaluate_induction,
    evaluate_distractor_induction,
    positional_recall_curve,
)

RETRIEVAL_MECH_KEYS = (
    ".W_phase_q", ".W_phase_k", ".W_phase_scale",
    ".W_gate_k.weight", ".W_range_k",
    ".W_gate_sal.weight", ".W_gate_d.weight",
    ".W_gate_f.weight", ".W_range_f",
    ".W_qtemp.weight",
)
PHASE_KEYS = (".W_phase_q", ".W_phase_k", ".W_phase_scale")


def load_state(path: str) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    return {k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
            for k, v in state.items()}


def build_franken(base_state: Dict[str, torch.Tensor],
                  hla_state: Dict[str, torch.Tensor],
                  n_embd: int,
                  transplant: str) -> Dict[str, torch.Tensor]:
    """Start from BASE everywhere, splice HLA's retrieval side in."""
    if set(base_state) != set(hla_state):
        raise ValueError("state dict key sets differ - not a sterile pair")
    if transplant == "full":
        return dict(hla_state)

    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in base_state.items()}
    for key in hla_state:
        if ".c_attn.weight" in key:
            # rows [0:2C] = Q,K (retrieval); rows [2C:3C] = V (transmission)
            spliced = out[key].clone()
            spliced[: 2 * n_embd] = hla_state[key][: 2 * n_embd]
            out[key] = spliced
        elif transplant in ("phase", "retrieval") and any(p in key for p in PHASE_KEYS):
            out[key] = hla_state[key].clone()
        elif transplant == "retrieval" and any(p in key for p in RETRIEVAL_MECH_KEYS):
            out[key] = hla_state[key].clone()
    return out


def probe(model: GPT, device: str = "cpu") -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["induction"] = float(evaluate_induction(model, device=device, batch_size=8))
    try:
        d = evaluate_distractor_induction(model, device=device, batch_size=8)
        out.update({f"distractor_{k}": float(v) for k, v in d.items()})
    except Exception as e:
        out["distractor_error"] = str(e)  # type: ignore[assignment]
    try:
        out.update({f"posrec_{k}": float(v)
                    for k, v in positional_recall_curve(model, device=device, batch_size=4).items()})
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Transplant HLA retrieval geometry into the base twin")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--hla-checkpoint", required=True)
    ap.add_argument("--hla-config", required=True,
                    help="HLA run config (franken runs with mechanisms ACTIVE)")
    ap.add_argument("--transplant", default="retrieval",
                    choices=["qk", "phase", "retrieval", "full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = json.load(open(args.hla_config, encoding="utf-8"))["model"]
    base_state = load_state(args.base_checkpoint)
    hla_state = load_state(args.hla_checkpoint)
    franken_state = build_franken(base_state, hla_state, int(cfg["n_embd"]), args.transplant)

    results: Dict[str, Dict[str, float]] = {}
    for name, state in (("base", base_state), ("hla", hla_state), ("franken", franken_state)):
        model = GPT(GPTConfig(**cfg)).eval()
        model.load_state_dict(state, strict=True)
        results[name] = probe(model, device=args.device)
        print(f"{name:8s}: " + "  ".join(f"{k}={v:.5f}" for k, v in results[name].items()
                                         if isinstance(v, float) and "pos_" not in k))

    results["meta"] = {"transplant": args.transplant,  # type: ignore[assignment]
                       "base_checkpoint": args.base_checkpoint,
                       "hla_checkpoint": args.hla_checkpoint}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
