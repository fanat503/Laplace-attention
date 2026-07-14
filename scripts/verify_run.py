"""Preflight verifier for HLA-v4 experiments.

Runs on CPU by default and checks the parts that should be correct before using
TPU time: config, dataset, model construction, HLA identity, forward/backward,
optimizer step, init checkpoint loading, and shared-backbone equality if two init
checkpoints are provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402
from src.data import FixedDataset  # noqa: E402

HLA_MARKERS = ("W_phase_q", "W_phase_k", "W_range_k", "W_range_v", "W_gate_k", "W_gate_v")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hla_identity_error(model: GPT) -> float:
    err = 0.0
    for name, tensor in model.state_dict().items():
        if any(m in name for m in HLA_MARKERS):
            err = max(err, float(tensor.detach().float().abs().max().item()))
    return err


def load_model_state(model: GPT, path: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)


def verify_config(config: Dict[str, Any]) -> None:
    print("[1/7] config")
    for key in ["model", "train_path", "val_path", "batch_size_per_device"]:
        if key not in config:
            raise KeyError(f"missing config key: {key}")
    _ = GPTConfig(**config["model"])
    print("  ok")


def verify_dataset(config: Dict[str, Any], max_items: int = 2) -> None:
    print("[2/7] dataset")
    vocab = config["model"].get("vocab_size")
    ds = FixedDataset(config["train_path"], config["model"]["block_size"], expected_vocab_size=vocab)
    print(ds.summary())
    for i in range(min(max_items, len(ds))):
        item = ds[i]["input_ids"]
        expected = config["model"]["block_size"] + 1
        if item.ndim != 1 or item.numel() != expected:
            raise RuntimeError(f"dataset item shape mismatch: got shape={tuple(item.shape)}, expected=({expected},)")
        if item.dtype != torch.long:
            raise RuntimeError(f"dataset item dtype mismatch: got {item.dtype}, expected torch.long")
    print("  ok")


def verify_model(config: Dict[str, Any]) -> GPT:
    print("[3/7] model/init")
    torch.manual_seed(int(config.get("seed", 42)))
    model = GPT(GPTConfig(**config["model"]))
    err = hla_identity_error(model)
    if err != 0.0:
        raise RuntimeError(f"HLA identity error after init: {err}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,} hla_identity_error={err}")
    return model


def verify_forward_backward(model: GPT, config: Dict[str, Any], *, skip: bool = False) -> None:
    print("[4/7] forward/backward/step")
    if skip:
        print("  skipped")
        return
    model.train()
    B = min(2, int(config.get("batch_size_per_device", 1)))
    T = min(16, int(config["model"]["block_size"]))
    V = int(config["model"]["vocab_size"])
    V_out = int(config["model"].get("padded_vocab_size") or V)
    idx = torch.randint(0, V, (B, T), dtype=torch.long)
    logits, loss = model(idx, idx)
    if logits.shape != (B, T, V_out):
        raise RuntimeError(f"logits shape mismatch: got {tuple(logits.shape)}, expected {(B, T, V_out)}")
    if loss is None or not bool(torch.isfinite(loss).detach().cpu().item()):
        raise RuntimeError(f"invalid loss: {loss}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"  loss={float(loss.item()):.6f} ok")


def verify_init_load(config: Dict[str, Any], init_path: Optional[str]) -> None:
    print("[5/7] init checkpoint")
    if not init_path:
        print("  skipped")
        return
    model = GPT(GPTConfig(**config["model"]))
    load_model_state(model, init_path)
    err = hla_identity_error(model)
    if err != 0.0:
        raise RuntimeError(f"init checkpoint violates HLA identity: max_abs={err}")
    print(f"  loaded {init_path}; hla_identity_error={err}")


def verify_shared(base_config_path: Optional[str], hla_config_path: Optional[str], base_init: Optional[str], hla_init: Optional[str]) -> None:
    print("[6/7] shared-backbone equality")
    if not (base_config_path and hla_config_path and base_init and hla_init):
        print("  skipped")
        return
    base_cfg = load_json(base_config_path)
    hla_cfg = load_json(hla_config_path)
    base = GPT(GPTConfig(**base_cfg["model"]))
    hla = GPT(GPTConfig(**hla_cfg["model"]))
    load_model_state(base, base_init)
    load_model_state(hla, hla_init)
    mismatches = []
    hla_state = hla.state_dict()
    for k, v in base.state_dict().items():
        if any(m in k for m in HLA_MARKERS):
            continue
        if k in hla_state and tuple(v.shape) == tuple(hla_state[k].shape):
            if not torch.equal(v.cpu(), hla_state[k].cpu()):
                mismatches.append(k)
    if mismatches:
        raise RuntimeError(f"shared backbone mismatch keys: {mismatches[:16]}")
    print("  ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--hla-config", default=None)
    ap.add_argument("--base-init", default=None)
    ap.add_argument("--hla-init", default=None)
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-forward-backward", action="store_true", help="Do not run expensive full-model CPU forward/backward")
    args = ap.parse_args()

    cfg = load_json(args.config)
    verify_config(cfg)
    if not args.skip_dataset:
        verify_dataset(cfg)
    else:
        print("[2/7] dataset skipped")
    model = verify_model(cfg)
    verify_forward_backward(model, cfg, skip=args.skip_forward_backward)
    verify_init_load(cfg, args.init)
    verify_shared(args.base_config, args.hla_config, args.base_init, args.hla_init)
    print("[7/7] complete: OK")


if __name__ == "__main__":
    main()
