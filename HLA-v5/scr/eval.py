"""Evaluation probes and diagnostics for base/HLA GPT models."""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch


@torch.no_grad()
def evaluate_induction(model, device, seed: int = 42, batch_size: int = 32) -> float:
    was_training = model.training
    model.eval()
    try:
        T = min(256, model.config.block_size)
        if T < 8:
            return float("nan")

        # Use the constants defined at module top.
        required_vocab = INDUCTION_TOK_B_OFFSET + batch_size
        if model.config.vocab_size < required_vocab:
            return float("nan")

        pos_a1 = T // 3
        pos_b1 = pos_a1 + 1
        pos_a2 = (2 * T) // 3
        if pos_a2 >= T - 1:
            return float("nan")

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(
            low=100,
            high=min(20000, model.config.vocab_size - 1),
            size=(batch_size, T),
            generator=g,
        ).to(device)

        # Vectorized induction-token placement.
        i = torch.arange(batch_size, device=device)
        tokens[:, pos_a1] = INDUCTION_TOK_A_OFFSET + i
        tokens[:, pos_b1] = INDUCTION_TOK_B_OFFSET + i
        tokens[:, pos_a2] = INDUCTION_TOK_A_OFFSET + i
        target_ids = INDUCTION_TOK_B_OFFSET + i

        logits, _ = model(tokens)
        if not bool(torch.isfinite(logits).all().detach().cpu().item()):
            return float("nan")
        probs = torch.softmax(logits[:, pos_a2, :].float(), dim=-1)
        return float(probs.gather(1, target_ids[:, None]).mean().detach().cpu().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def measure_attention_entropy(model, device, seed: int = 42, batch_size: int = 4) -> float:
    was_training = model.training
    model.eval()
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        _ = model(tokens)
        entropies = []
        for block in model.transformer.h:
            e = getattr(block.attn, "last_entropy", None)
            if e is not None:
                entropies.append(e.float().mean())
        if not entropies:
            return float("nan")
        return float(torch.stack(entropies).mean().detach().cpu().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def phase_statistics(model) -> Tuple[Optional[torch.Tensor], float]:
    all_norms = []
    for block in model.transformer.h:
        attn = block.attn
        if not hasattr(attn, "W_phase_q") or attn.W_phase_q is None:
            return None, float("nan")
        Wq = attn.W_phase_q.detach().float()
        Wk = attn.W_phase_k.detach().float()
        norm_q = torch.linalg.norm(Wq, dim=(1, 2))
        norm_k = torch.linalg.norm(Wk, dim=(1, 2))
        all_norms.append(0.5 * (norm_q + norm_k))
    if not all_norms:
        return None, float("nan")
    stacked = torch.stack(all_norms, dim=0)
    mean_per_head = stacked.mean(dim=0).detach().cpu()
    return mean_per_head, float(stacked.mean().detach().cpu().item())


@torch.no_grad()
def hla_statistics(model) -> Dict[str, float]:
    vals: Dict[str, list] = {
        "layer_gate_multiplier": [],
        "angle_q_abs_mean": [],
        "angle_k_abs_mean": [],
        "gate_k_mean": [],
        "gate_v_mean": [],
        "mix_k_mean": [],
        "mix_v_mean": [],
        "distance_bias_mean": [],
        "distance_bias_abs_mean": [],
    }
    for block in model.transformer.h:
        attn = block.attn
        mapping = {
            "layer_gate_multiplier": torch.tensor(float(getattr(attn, "layer_gate_multiplier", 1.0))),
            "angle_q_abs_mean": getattr(attn, "last_angle_q_abs_mean", None),
            "angle_k_abs_mean": getattr(attn, "last_angle_k_abs_mean", None),
            "gate_k_mean": getattr(attn, "last_gate_k_mean", None),
            "gate_v_mean": getattr(attn, "last_gate_v_mean", None),
            "mix_k_mean": getattr(attn, "last_mix_k_mean", None),
            "mix_v_mean": getattr(attn, "last_mix_v_mean", None),
            "distance_bias_mean": getattr(attn, "last_distance_bias_mean", None),
            "distance_bias_abs_mean": getattr(attn, "last_distance_bias_abs_mean", None),
        }
        for k, v in mapping.items():
            if v is not None:
                vals[k].append(v.detach().float().mean())
    out: Dict[str, float] = {}
    for k, xs in vals.items():
        if xs:
            out[k] = float(torch.stack(xs).mean().detach().cpu().item())
    return out


__all__ = ["evaluate_induction", "measure_attention_entropy", "phase_statistics", "hla_statistics"]
