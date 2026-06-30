"""Profile FLOPs for both base and HLA models to compute matched compute."""
import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig


def count_forward_flops(model: GPT, batch_size: int = 1, seq_len: int = 2048) -> int:
    """Count forward FLOPs per call (excluding backward).

    Uses 6N rule for total training: forward + backward = 6N where N = params * tokens.
    Here we count forward only (N = 2 * params * tokens).
    """
    config = model.config
    B = batch_size
    T = seq_len
    C = config.n_embd
    H = config.n_head
    hs = config.n_embd // config.n_head
    L = config.n_layer

    # Attention QKV projection.
    qkv_flops = L * (2 * B * T * C * 3 * C)

    # Holographic phase rotation.
    if config.phase_mult != 0:
        # W_phase projection: einsum over C dim.
        phase_flops = L * (4 * B * T * C * H * (hs // 2))  # Q + K
        # rotate_pairwise: 4 mul-add per element.
        rotate_flops = L * (B * T * H * hs * 4)
    else:
        phase_flops = 0
        rotate_flops = 0

    # Laplace gating.
    if config.use_laplace and config.laplace_alpha != 0:
        # W_gate: 2 * C * H per layer (no batch dim).
        gate_flops = L * (4 * B * T * C * H)
        # mix computation: per-element mul + add.
        mix_flops = L * (B * T * H * hs * 4)
    else:
        gate_flops = 0
        mix_flops = 0

    # Attention scores (Q @ K^T).
    attn_score_flops = L * (2 * B * H * T * T * hs)
    # Attention values (attn @ V).
    attn_value_flops = L * (2 * B * H * T * T * hs)

    # Distance-Laplace bias.
    if config.use_distance_laplace and config.distance_laplace_alpha != 0:
        # Distance matrix (T*T) + bias computation.
        dist_flops = L * (2 * B * H * T * T)
    else:
        dist_flops = 0

    # Output projection.
    proj_flops = L * (2 * B * T * C * C)

    # MLP (SwiGLU).
    hidden = L * (8 * C // 3)  # approximate
    hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
    if config.fused_swiglu:
        mlp_flops = L * (2 * B * T * C * (2 * hidden) + 2 * B * T * hidden * C)
    else:
        mlp_flops = L * (4 * B * T * C * hidden + 2 * B * T * hidden * C)

    # LayerNorm (negligible but include for completeness).
    ln_flops = L * (4 * B * T * C) * 2  # 2 LN per block

    # Final LN.
    final_ln_flops = 4 * B * T * C

    total = (
            qkv_flops + phase_flops + rotate_flops +
            gate_flops + mix_flops +
            attn_score_flops + attn_value_flops + dist_flops +
            proj_flops + mlp_flops + ln_flops + final_ln_flops
    )

    return {
        "qkv": qkv_flops,
        "phase": phase_flops,
        "rotate": rotate_flops,
        "gate": gate_flops,
        "mix": mix_flops,
        "attn_score": attn_score_flops,
        "attn_value": attn_value_flops,
        "distance_bias": dist_flops,
        "proj": proj_flops,
        "mlp": mlp_flops,
        "ln": ln_flops + final_ln_flops,
        "total": total,
    }


def main():
    # Load configs.
    base_path = "configs/200m_base_s42.json"
    hla_path = "configs/200m_hla_s42.json"

    with open(base_path) as f:
        base_config = json.load(f)
    with open(hla_path) as f:
        hla_config = json.load(f)

    # Create models.
    print("Creating base model...")
    base_model = GPT(GPTConfig(**base_config["model"]))
    base_model.eval()

    print("Creating HLA model...")
    hla_model = GPT(GPTConfig(**hla_config["model"]))
    hla_model.eval()
    hla_model.reset_hla_identity()

    # Profile.
    print("\nProfiling FLOPs...")
    base_flops = count_forward_flops(base_model, batch_size=1, seq_len=2048)
    hla_flops = count_forward_flops(hla_model, batch_size=1, seq_len=2048)

    # Print breakdown.
    print(f"\n{'Component':25s}  {'Base':>15s}  {'HLA':>15s}  {'Ratio':>10s}")
    print(f"{'-' * 70}")
    for k in base_flops:
        ratio = hla_flops[k] / base_flops[k] if base_flops[k] > 0 else float("inf")
        print(f"{k:25s}  {base_flops[k]:>15,}  {hla_flops[k]:>15,}  {ratio:>10.3f}")

    # Total ratio.
    total_ratio = hla_flops["total"] / base_flops["total"]
    print(f"\nTotal FLOPs ratio (HLA/Base): {total_ratio:.4f}")
    print(f"HLA overhead: {(total_ratio - 1) * 100:.2f}%")

    # Recommendation.
    print("\n--- Recommendation ---")
    base_max_steps = base_config["max_steps"]
    print(f"Base max_steps: {base_max_steps}")

    # For matched training FLOPs: multiply base steps by inverse ratio.
    # Training FLOPs ≈ 6 × params × tokens = 6 × params × (steps × batch × seq_len).
    # If we want same total compute, HLA needs fewer steps.
    matched_hla_steps = int(base_max_steps / total_ratio)
    print(f"For matched compute: HLA max_steps = {matched_hla_steps}")
    print(f"  (vs current {hla_config['max_steps']})")
    print(
        f"  Difference: {hla_config['max_steps'] - matched_hla_steps} steps ({100 * (hla_config['max_steps'] - matched_hla_steps) / base_max_steps:.2f}%)")

    # Save profile results.
    results = {
        "base_flops": base_flops,
        "hla_flops": hla_flops,
        "total_ratio": total_ratio,
        "base_max_steps": base_max_steps,
        "matched_hla_max_steps": matched_hla_steps,
    }
    with open("flops_profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: flops_profile.json")


if __name__ == "__main__":
    main()