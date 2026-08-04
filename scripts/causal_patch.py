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


def probe(model: GPT, device: str = "cpu", seed: int = 42,
          batch_size: int = 8) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["induction"] = float(evaluate_induction(
        model, device=device, seed=seed, batch_size=batch_size))
    try:
        d = evaluate_distractor_induction(
            model, device=device, seed=seed, batch_size=batch_size)
        # Keys already carry the distractor_ prefix - do NOT re-prefix
        # (v1 wrote distractor_distractor_induction; regression-tested now).
        out.update({str(k): float(v) for k, v in d.items()})
    except Exception as e:
        out["distractor_error"] = str(e)  # type: ignore[assignment]
    try:
        out.update({f"posrec_{k}": float(v)
                    for k, v in positional_recall_curve(
                        model, device=device, seed=seed,
                        batch_size=max(2, batch_size // 2)).items()})
    except Exception:
        pass
    return out


def probe_multi(model: GPT, device: str = "cpu", seeds=(42, 43, 44, 45, 46),
                batch_size: int = 8) -> Dict[str, float]:
    """Probe under several seeds -> mean and std per metric.

    The H5 decision number needs an uncertainty: a gap-closure fraction
    without error bars is exactly the kind of single-number claim Reviewer 2
    strikes down. Seeds vary the synthetic probe content, not the model.
    """
    import statistics
    per_seed = [probe(model, device=device, seed=s, batch_size=batch_size)
                for s in seeds]
    keys = [k for k in per_seed[0]
            if all(isinstance(r.get(k), float) for r in per_seed)]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in per_seed if r[k] == r[k]]  # drop NaN
        if not vals:
            out[k] = float("nan")
            out[f"{k}_std"] = float("nan")
            continue
        out[k] = float(statistics.fmean(vals))
        out[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return out


GAP_METRICS = ("induction", "distractor_induction", "distractor_margin",
               "posrec_litm_worst_frac")
# Pre-registered (EXPERIMENT_CARD H5): >50% closure on retrieval probes =>
# retrieval geometry CAUSES the gain; <20% => story is NOT causal - report so.
MIN_MEANINGFUL_GAP = 1e-4


def gap_closure(base: Dict[str, float], hla: Dict[str, float],
                franken: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """closure = (franken - base) / (hla - base), guarded.

    Guards (each has a test):
      - |hla-base| < MIN_MEANINGFUL_GAP -> closure undefined (NaN), flagged:
        dividing by a near-zero gap manufactures arbitrary percentages;
      - propagated std via first-order bounds from per-metric stds.
    """
    out: Dict[str, Dict[str, float]] = {}
    for m in GAP_METRICS:
        if m not in base or m not in hla or m not in franken:
            continue
        gap = hla[m] - base[m]
        rec = {"base": base[m], "hla": hla[m], "franken": franken[m],
               "gap": gap}
        if not (gap == gap) or abs(gap) < MIN_MEANINGFUL_GAP:
            rec["closure"] = float("nan")
            rec["closure_note"] = 1.0  # gap too small to attribute
        else:
            rec["closure"] = (franken[m] - base[m]) / gap
            noise = max(base.get(f"{m}_std", 0.0), hla.get(f"{m}_std", 0.0),
                        franken.get(f"{m}_std", 0.0))
            rec["closure_std_bound"] = 3.0 * noise / abs(gap)
        out[m] = rec
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
    ap.add_argument("--probe-seeds", default="42,43,44,45,46",
                    help="comma-separated probe seeds (error bars for H5)")
    ap.add_argument("--probe-batch", type=int, default=8)
    args = ap.parse_args()

    cfg = json.load(open(args.hla_config, encoding="utf-8"))["model"]
    base_state = load_state(args.base_checkpoint)
    hla_state = load_state(args.hla_checkpoint)
    franken_state = build_franken(base_state, hla_state, int(cfg["n_embd"]), args.transplant)

    results: Dict[str, Dict[str, float]] = {}
    seeds = tuple(int(s) for s in args.probe_seeds.split(","))
    for name, state in (("base", base_state), ("hla", hla_state), ("franken", franken_state)):
        model = GPT(GPTConfig(**cfg)).eval()
        model.load_state_dict(state, strict=True)
        results[name] = probe_multi(model, device=args.device, seeds=seeds,
                                    batch_size=args.probe_batch)
        print(f"{name:8s}: " + "  ".join(f"{k}={v:.5f}" for k, v in results[name].items()
                                         if isinstance(v, float) and "pos_" not in k
                                         and not k.endswith("_std")))

    closure = gap_closure(results["base"], results["hla"], results["franken"])
    results["gap_closure"] = closure  # type: ignore[assignment]
    print("\n=== H5 gap closure (pre-registered: >0.50 causal, <0.20 not) ===")
    for m, rec in closure.items():
        if rec.get("closure_note"):
            print(f"  {m:24s}: gap={rec['gap']:+.6f} TOO SMALL to attribute (no claim)")
        else:
            print(f"  {m:24s}: closure={rec['closure']:+.3f} "
                  f"(±{rec.get('closure_std_bound', float('nan')):.3f}) "
                  f"gap={rec['gap']:+.6f}")

    results["meta"] = {"transplant": args.transplant,  # type: ignore[assignment]
                       "base_checkpoint": args.base_checkpoint,
                       "hla_checkpoint": args.hla_checkpoint,
                       "probe_seeds": list(seeds),
                       "probe_batch": args.probe_batch}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
