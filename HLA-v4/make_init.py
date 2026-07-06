"""
make_init.py

Creates a sterile init_state.pt — a shared starting point
for Base and v4 experiments.

IMPORTANT:
  block_size, vocab_size, n_layer, n_head, n_embd
  MUST exactly match modal_app.py / training config.

This script also:
  - validates architecture keys
  - checks zero-initialization of phase/gate/range params
  - verifies weight tying
  - verifies no NaN/Inf in parameters
  - saves a metadata manifest
  - computes SHA256 checksum for reproducibility
"""

import os
import json
import time
import hashlib
from dataclasses import asdict

import torch

from train1 import GPT, GPTConfig

# =========================================================
# CONFIG — MUST match modal_app.py COMMON_MODEL shape
# =========================================================

INIT_CFG = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=16,
    n_head=12,
    n_embd=768,

    # Neutral settings: feature disabled, but all params exist
    phase_mult=0.0,
    use_laplace=True,
    laplace_alpha=0.0,
    laplace_range_k=0.45,
    laplace_range_v=0.18,
    beta_k=0.55,
    beta_v=0.18,

    gradient_checkpointing=False,  # not needed for init creation
)

SEED = 42
OUT_PATH = "init_state.pt"
MANIFEST_PATH = "init_state_manifest.json"


# =========================================================
# HELPERS
# =========================================================

def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def tensor_has_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def pretty_bool(ok: bool) -> str:
    return "✅" if ok else "❌"


def model_shape_summary(cfg: GPTConfig) -> str:
    head_dim = cfg.n_embd // cfg.n_head
    return (
        f"layers={cfg.n_layer}, heads={cfg.n_head}, embd={cfg.n_embd}, "
        f"block={cfg.block_size}, vocab={cfg.vocab_size}, head_dim={head_dim}"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print("make_init.py — creating sterile init_state.pt")
    print("=" * 72)

    # -----------------------------------------------------
    # 1) Validate config shape
    # -----------------------------------------------------
    assert INIT_CFG.n_embd % INIT_CFG.n_head == 0, (
        f"n_embd={INIT_CFG.n_embd} must be divisible by n_head={INIT_CFG.n_head}"
    )
    head_dim = INIT_CFG.n_embd // INIT_CFG.n_head
    assert head_dim % 2 == 0, (
        f"head_dim={head_dim} must be even for pairwise complex rotation"
    )

    print(f"\nSeed: {SEED}")
    print(f"Shape: {model_shape_summary(INIT_CFG)}")

    # -----------------------------------------------------
    # 2) Reproducibility
    # -----------------------------------------------------
    torch.manual_seed(SEED)

    # -----------------------------------------------------
    # 3) Create model
    # -----------------------------------------------------
    print("\nCreating model on CPU...")
    model = GPT(INIT_CFG)
    model.eval()

    # -----------------------------------------------------
    # 4) Parameter counts
    # -----------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())

    print(f"\nParameter counts:")
    print(f"  total params (with shared refs counted once by module list): {total_params:,}")
    print(f"  unique params:                                              {unique_params:,}")
    print(f"  fp32 size estimate:                                         {unique_params * 4 / 1024**2:.1f} MB")

    # -----------------------------------------------------
    # 5) Check weight tying properly (on model, not state_dict)
    # -----------------------------------------------------
    tied_ok = model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print(f"\nWeight tying:")
    print(f"  {pretty_bool(tied_ok)} transformer.wte.weight is lm_head.weight")
    if not tied_ok:
        raise RuntimeError("Weight tying check failed: wte and lm_head are not tied.")

    # -----------------------------------------------------
    # 6) Get state_dict
    # -----------------------------------------------------
    sd = model.state_dict()
    print(f"\nstate_dict:")
    print(f"  keys: {len(sd)}")

    # -----------------------------------------------------
    # 7) Check required architecture keys across ALL layers
    # -----------------------------------------------------
    print("\nChecking required architecture keys across all layers...")
    required_suffixes = [
        "attn.W_phase_q",
        "attn.W_phase_k",
        "attn.W_gate_k.weight",
        "attn.W_gate_v.weight",
        "attn.W_range_k",
        "attn.W_range_v",
    ]

    missing_keys = []
    for layer_idx in range(INIT_CFG.n_layer):
        for suffix in required_suffixes:
            key = f"transformer.h.{layer_idx}.{suffix}"
            if key not in sd:
                missing_keys.append(key)

    if missing_keys:
        print(f"  {pretty_bool(False)} missing {len(missing_keys)} keys")
        for k in missing_keys[:20]:
            print(f"    MISSING: {k}")
        raise RuntimeError(
            "Some required architecture keys are missing from state_dict. "
            "Check CausalSelfAttention / Block / GPT construction."
        )
    else:
        print(f"  {pretty_bool(True)} all required keys exist in all {INIT_CFG.n_layer} layers")

    # -----------------------------------------------------
    # 8) Check zero-init of phase/gate/range params across ALL layers
    # -----------------------------------------------------
    print("\nChecking zero initialization across all layers...")
    zero_specs = [
        ("W_phase_q", "attn.W_phase_q"),
        ("W_phase_k", "attn.W_phase_k"),
        ("W_gate_k",  "attn.W_gate_k.weight"),
        ("W_gate_v",  "attn.W_gate_v.weight"),
        ("W_range_k", "attn.W_range_k"),
        ("W_range_v", "attn.W_range_v"),
    ]

    zero_init_failures = []
    for layer_idx in range(INIT_CFG.n_layer):
        for short_name, suffix in zero_specs:
            key = f"transformer.h.{layer_idx}.{suffix}"
            max_abs = tensor_max_abs(sd[key])
            ok = (max_abs == 0.0)
            print(f"  {pretty_bool(ok)} layer={layer_idx:02d} {short_name:<9} max_abs={max_abs:.6f}")
            if not ok:
                zero_init_failures.append((key, max_abs))

    if zero_init_failures:
        raise RuntimeError(
            "Zero initialization check failed for some phase/gate/range parameters."
        )

    # -----------------------------------------------------
    # 9) Check all tensors are finite
    # -----------------------------------------------------
    print("\nChecking all parameters are finite...")
    non_finite_keys = []
    for k, v in sd.items():
        if torch.is_floating_point(v) and not tensor_has_finite(v):
            non_finite_keys.append(k)

    if non_finite_keys:
        print(f"  {pretty_bool(False)} found non-finite tensors:")
        for k in non_finite_keys[:20]:
            print(f"    {k}")
        raise RuntimeError("Found NaN/Inf in initial parameters.")
    else:
        print(f"  {pretty_bool(True)} all floating tensors are finite")

    # -----------------------------------------------------
    # 10) Save state_dict
    # -----------------------------------------------------
    print(f"\nSaving init state to: {OUT_PATH}")
    torch.save(sd, OUT_PATH)

    size_mb = os.path.getsize(OUT_PATH) / 1024**2
    print(f"Saved file size: {size_mb:.1f} MB")

    # -----------------------------------------------------
    # 11) Round-trip check
    # -----------------------------------------------------
    print("\nRound-trip load check...")
    loaded = torch.load(OUT_PATH, map_location="cpu", weights_only=True)

    if set(sd.keys()) != set(loaded.keys()):
        missing = set(sd.keys()) - set(loaded.keys())
        extra = set(loaded.keys()) - set(sd.keys())
        raise RuntimeError(
            f"Round-trip key mismatch.\nMissing: {list(missing)[:10]}\nExtra: {list(extra)[:10]}"
        )

    for key in sd:
        if sd[key].shape != loaded[key].shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: {sd[key].shape} vs {loaded[key].shape}"
            )
        if sd[key].dtype != loaded[key].dtype:
            raise RuntimeError(
                f"Dtype mismatch for {key}: {sd[key].dtype} vs {loaded[key].dtype}"
            )

    print(f"  {pretty_bool(True)} round-trip keys/shapes/dtypes all match")

    # -----------------------------------------------------
    # 12) SHA256 for reproducibility
    # -----------------------------------------------------
    sha256 = sha256_of_file(OUT_PATH)
    print(f"\nSHA256: {sha256}")

    # -----------------------------------------------------
    # 13) Save manifest
    # -----------------------------------------------------
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "out_path": OUT_PATH,
        "sha256": sha256,
        "file_size_mb": round(size_mb, 2),
        "total_params": total_params,
        "unique_params": unique_params,
        "shape_summary": model_shape_summary(INIT_CFG),
        "init_cfg": asdict(INIT_CFG),
        "checks": {
            "weight_tying": tied_ok,
            "required_keys_present": True,
            "zero_init_phase_gate_range": True,
            "all_finite": True,
            "round_trip_ok": True,
        },
        "notes": [
            "This init_state.pt is intended as a shared initialization for Base and v4.",
            "Config shape fields must exactly match modal_app.py / training config.",
            "phase_mult and laplace_alpha are config scalars, not stored in state_dict.",
        ],
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {MANIFEST_PATH}")

    # -----------------------------------------------------
    # 14) Final instructions
    # -----------------------------------------------------
    print("\n" + "=" * 72)
    print("init_state.pt is ready for upload to Modal Volume")
    print("=" * 72)
    print("\nNext steps:")
    print("  1) Put init_state.pt into ./data/init_state.pt")
    print("  2) modal run modal_app.py --mode upload --use-init true")
    print("  3) modal run modal_app.py --mode check")
    print("  4) modal run modal_app.py --mode train --variant base --preset smoke")


if __name__ == "__main__":
    main()