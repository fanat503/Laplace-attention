"""Pure-Python parameter/token/time budget estimator for current GPT/HLA configs.

Does not import torch. Useful on machines where TPU dependencies are not installed.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def estimate_params(cfg: Dict[str, Any]) -> Dict[str, int]:
    m = cfg["model"]
    d = int(m["n_embd"])
    h = int(m["n_head"])
    L = int(m["n_layer"])
    V = int(m.get("padded_vocab_size") or m["vocab_size"])
    T = int(m["block_size"])
    hd = d // h
    if d % h != 0 or hd % 2 != 0:
        raise ValueError("invalid d/head config")

    token_embedding = V * d
    position_embedding = T * d
    final_norm = d
    attention_base_per_layer = 4 * d * d
    hidden = round_up(int(8 * d / 3), int(m.get("ffn_hidden_multiple_of", 64)))
    mlp_per_layer = 3 * d * hidden
    norms_per_layer = 2 * d
    hla_per_layer = 2 * (h * d * (hd // 2)) + 2 * (d * h) + 2 * h
    per_layer = attention_base_per_layer + mlp_per_layer + norms_per_layer + hla_per_layer
    total = token_embedding + position_embedding + L * per_layer + final_norm
    return {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "attention_base_per_layer": attention_base_per_layer,
        "mlp_hidden": hidden,
        "mlp_per_layer": mlp_per_layer,
        "norms_per_layer": norms_per_layer,
        "hla_per_layer": hla_per_layer,
        "hla_total": hla_per_layer * L,
        "per_layer_total": per_layer,
        "final_norm": final_norm,
        "total_unique_parameters": total,
    }


def estimate_tokens(cfg: Dict[str, Any]) -> Dict[str, int]:
    batch = int(cfg["batch_size_per_device"])
    cores = int(cfg["num_cores"])
    block = int(cfg["model"]["block_size"])
    grad_accum = int(cfg["grad_accum"])
    max_steps = int(cfg["max_steps"])
    seqs_per_update = batch * cores * grad_accum
    tokens_per_update = seqs_per_update * block
    stored_ids_per_update = seqs_per_update * (block + 1)
    required_sequences = seqs_per_update * max_steps
    return {
        "sequences_per_update": seqs_per_update,
        "tokens_per_update_effective": tokens_per_update,
        "stored_ids_per_update": stored_ids_per_update,
        "max_steps": max_steps,
        "total_effective_tokens": tokens_per_update * max_steps,
        "required_sequences": required_sequences,
        "required_stored_token_ids": required_sequences * (block + 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens-per-sec", type=float, default=None, help="Optional measured throughput for ETA")
    args = ap.parse_args()
    cfg = load(args.config)
    out = {"params": estimate_params(cfg), "tokens": estimate_tokens(cfg)}
    if args.tokens_per_sec:
        seconds = out["tokens"]["total_effective_tokens"] / args.tokens_per_sec
        out["time_estimate"] = {
            "tokens_per_sec": args.tokens_per_sec,
            "hours": seconds / 3600,
            "days": seconds / 86400,
        }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
