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



"""Evaluation probes and diagnostics for base/HLA GPT models."""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Token id offsets used to build synthetic A->B induction pairs. They must be
# below the smallest GPT-2 BPE ids used for random filler (100..20000) minus
# batch_size, so that induction tokens never collide with filler tokens.
INDUCTION_TOK_A_OFFSET = 10
INDUCTION_TOK_B_OFFSET = 50


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
def depth_profile_statistics(model) -> Dict[str, float]:
    """Readouts of the learned depth/head profiles (interpretability for free).

    - layer_temp_l{i}: effective learned multiplier at layer i (only when
      learnable_layer_temp is on); equals the static heuristic at init.
    - phase_budget_mean/min/max: distribution of per-head phase budgets
      (1 + tanh(W_phase_scale)) in [0, 2]; 1.0 = init value for every head.
    """
    import math as _m

    out: Dict[str, float] = {}
    temps = []
    budgets = []
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        if getattr(attn, "layer_dependent_gate", False) and getattr(attn, "learnable_layer_temp", False):
            depth = float(attn.layer_idx) / float(max(1, attn.n_layer))
            t = float(torch.nn.functional.softplus(attn.W_layer_temp.detach().float()) / _m.log(2.0))
            temps.append(1.0 + depth * t)
        if getattr(attn, "per_head_phase", False):
            budgets.append(1.0 + torch.tanh(attn.W_phase_scale.detach().float()))
    if temps:
        out["layer_temp_first"] = temps[0]
        out["layer_temp_last"] = temps[-1]
        out["layer_temp_mean"] = float(sum(temps) / len(temps))
    if budgets:
        all_b = torch.cat(budgets)
        out["phase_budget_mean"] = float(all_b.mean())
        out["phase_budget_min"] = float(all_b.min())
        out["phase_budget_max"] = float(all_b.max())
    return out


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
        "angle_q_sat_frac": [],
        "angle_k_sat_frac": [],
        "gate_k_sat_frac": [],
        "gate_k_abs_mean": [],
        "gate_v_abs_mean": [],
        "gate_v_sat_frac": [],
        "salience_bias_abs_mean": [],
        "salience_sat_frac": [],
        "forget_bias_abs_mean": [],
        "forget_sat_frac": [],
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
            "angle_q_sat_frac": getattr(attn, "last_angle_q_sat_frac", None),
            "angle_k_sat_frac": getattr(attn, "last_angle_k_sat_frac", None),
            "gate_k_sat_frac": getattr(attn, "last_gate_k_sat_frac", None),
            "gate_k_abs_mean": getattr(attn, "last_gate_k_abs_mean", None),
            "gate_v_abs_mean": getattr(attn, "last_gate_v_abs_mean", None),
            "gate_v_sat_frac": getattr(attn, "last_gate_v_sat_frac", None),
            "salience_bias_abs_mean": getattr(attn, "last_salience_bias_abs_mean", None),
            "salience_sat_frac": getattr(attn, "last_salience_sat_frac", None),
            "forget_bias_abs_mean": getattr(attn, "last_forget_bias_abs_mean", None),
            "forget_sat_frac": getattr(attn, "last_forget_sat_frac", None),
        }
        for k, v in mapping.items():
            if v is not None:
                vals[k].append(v.detach().float().mean())
    out: Dict[str, float] = {}
    for k, xs in vals.items():
        if xs:
            out[k] = float(torch.stack(xs).mean().detach().cpu().item())
    return out


@torch.no_grad()
def svd_statistics(model, max_layers: Optional[int] = None) -> Dict[str, float]:
    """Spectral (SVD) diagnostics of attention weights. CPU-only, master-only.

    Metrics (averaged over layers/heads):
      - phase_erank:     effective rank of W_phase_q/k per head
                         (erank = exp(entropy of normalized singular values);
                          1.0 = rank-1 collapse, head_dim/2 = full use of rotation subspace)
      - phase_top1:      share of top singular value in W_phase (concentration; 1.0 = rank-1)
      - qk_stable_rank:  stable rank ||A||_F^2 / s_max^2 of the Q and K blocks of c_attn
      - v_stable_rank:   same for the V block
      - gate_erank:      effective rank of W_gate_k/v (H x C matrices)

    Why: HLA claims to *decouple* retrieval from content transmission. If true,
    the spectra of Q/K blocks in the HLA run should diverge from base over
    training (retrieval subspace specializes) while V spectra stay comparable.
    Tracking eranks during training gives that evidence for free.
    """

    def _svals(mat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(mat.detach().float().cpu())

    def _erank(s: torch.Tensor, eps: float = 1e-12) -> float:
        total = float(s.sum())
        if not (total > 0):
            return 0.0
        p = (s / total).clamp_min(eps)
        return float(torch.exp(-(p * p.log()).sum()))

    def _stable_rank(s: torch.Tensor) -> float:
        if s.numel() == 0 or float(s[0]) == 0.0:
            return 0.0
        return float((s * s).sum() / (s[0] * s[0]))

    phase_eranks: list = []
    phase_top1: list = []
    qk_stable: list = []
    v_stable: list = []
    gate_eranks: list = []

    blocks = list(model.transformer.h)
    if max_layers is not None:
        blocks = blocks[: int(max_layers)]

    for block in blocks:
        attn = block.attn
        C = attn.n_embd

        # Phase matrices: (H, C, head_dim//2), per-head spectra.
        for W in (attn.W_phase_q, attn.W_phase_k):
            for h in range(attn.n_head):
                s = _svals(W[h])
                if float(s.sum()) > 0:
                    phase_eranks.append(_erank(s))
                    phase_top1.append(float(s[0] / s.sum()))

        # c_attn blocks: rows [0:C] = Q, [C:2C] = K, [2C:3C] = V.
        w = attn.c_attn.weight  # (3C, C)
        s_q = _svals(w[:C, :])
        s_k = _svals(w[C : 2 * C, :])
        s_v = _svals(w[2 * C :, :])
        qk_stable.append(0.5 * (_stable_rank(s_q) + _stable_rank(s_k)))
        v_stable.append(_stable_rank(s_v))

        # Gate matrices: (H, C) -> rank <= H.
        for G in (attn.W_gate_k.weight, attn.W_gate_v.weight):
            s = _svals(G)
            if float(s.sum()) > 0:
                gate_eranks.append(_erank(s))

    out: Dict[str, float] = {
        "qk_stable_rank": float(torch.tensor(qk_stable).mean()) if qk_stable else float("nan"),
        "v_stable_rank": float(torch.tensor(v_stable).mean()) if v_stable else float("nan"),
    }
    # Phase/gate spectra are all-zero at identity init: report NaN (not 0) so
    # plots clearly show "not yet active" instead of a fake rank-0 reading.
    out["phase_erank"] = float(torch.tensor(phase_eranks).mean()) if phase_eranks else float("nan")
    out["phase_top1"] = float(torch.tensor(phase_top1).mean()) if phase_top1 else float("nan")
    out["gate_erank"] = float(torch.tensor(gate_eranks).mean()) if gate_eranks else float("nan")
    return out


@torch.no_grad()
def evaluate_distractor_induction(
    model,
    device,
    seed: int = 42,
    batch_size: int = 32,
    n_distractors: int = 24,
) -> Dict[str, float]:
    """Induction under noise: A->B recall with repeated distractor pairs.

    Standard induction ([A][B] ... [A] -> [B]?) measures retrieval. This probe
    measures retrieval *under interference*: between the first [A][B] and the
    second [A] we insert `n_distractors` REPEATED spurious pairs [C_i][D_i]
    (repetition makes them attractive to induction heads - realistic noise,
    not just random filler).

    Returns:
      distractor_induction : P(correct B | query A) with distractors present
      distractor_margin    : P(B) - max_i P(D_i)  (positive = the true target
                             beats the strongest distractor)

    Why it exists: val loss can hide selective-attention gains. The salience
    and Laplace gating mechanisms specifically claim to SUPPRESS irrelevant
    keys; this is the direct behavioral readout of that claim. Compare
    base-vs-HLA trajectories of distractor_margin during training.
    """
    was_training = model.training
    model.eval()
    try:
        T = min(512, model.config.block_size)
        if model.config.vocab_size < 20000 or T < 8 * (n_distractors + 2):
            return {"distractor_induction": float("nan"), "distractor_margin": float("nan")}

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(100, min(20000, model.config.vocab_size - 1),
                               (batch_size, T), generator=g)

        i = torch.arange(batch_size)
        tok_a = INDUCTION_TOK_A_OFFSET + i          # (B,)
        tok_b = INDUCTION_TOK_B_OFFSET + i

        # Layout: ... [A][B] ... {[C_j][D_j] x n, each pair repeated twice} ... [A] -> ?
        pos_a1 = T // 8
        tokens[:, pos_a1] = tok_a
        tokens[:, pos_a1 + 1] = tok_b

        # Distractor pairs: unique per batch row, repeated twice to form
        # competing induction patterns.
        span = T - pos_a1 - 4
        d_tokens = torch.randint(20000, min(40000, model.config.vocab_size - 1),
                                 (batch_size, n_distractors, 2), generator=g)
        d_positions = []
        for j in range(n_distractors):
            p1 = pos_a1 + 2 + (span * (2 * j) // (2 * n_distractors + 1))
            p2 = pos_a1 + 2 + (span * (2 * j + 1) // (2 * n_distractors + 1))
            for p in (p1, p2):
                if p + 1 < T - 2:
                    tokens[:, p] = d_tokens[:, j, 0]
                    tokens[:, p + 1] = d_tokens[:, j, 1]
            d_positions.append(j)

        pos_a2 = T - 2
        tokens[:, pos_a2] = tok_a

        logits, _ = model(tokens.to(device))
        if not bool(torch.isfinite(logits).all().detach().cpu().item()):
            return {"distractor_induction": float("nan"), "distractor_margin": float("nan")}
        probs = torch.softmax(logits[:, pos_a2, :].float(), dim=-1).cpu()  # (B, V)

        p_b = probs.gather(1, tok_b[:, None]).squeeze(1)                  # (B,)
        p_d = probs.gather(1, d_tokens[:, :, 1])                          # (B, n_distr): P(D_j)
        margin = (p_b - p_d.max(dim=1).values).mean()
        return {
            "distractor_induction": float(p_b.mean()),
            "distractor_margin": float(margin),
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def head_interference_statistics(
    model,
    max_layers: Optional[int] = None,
    topk: int = 8,
) -> Dict[str, float]:
    """Cross-head subspace interference, after Anthropic's Transformer Circuits
    (Elhage et al. 2021: composition/interference via W_OV and W_QK products).

    For every layer we build per-head subspaces (rank<=head_dim, top-`topk`
    singular directions in the residual stream R^C):

      write_i = column space of W_O[:, head i]           (what head i WRITES)
      qk_read_j = row space of [W_Q; W_K][head j]        (what head j's QK READS)
      v_read_j  = row space of W_V[head j]               (what head j's OV READS)

    Metrics (mean over layers; i != j pairs within a layer):

      qk_interference : mean overlap(write_i, qk_read_j)
          "how much what one head writes lands in another head's RETRIEVAL input"
          -> the noise floor for Q/K matching. HLA's claim predicts THIS drops
             vs base: Laplace gating suppresses irrelevant key content, phase
             rotation moves matching into rotated subspaces.
      ov_interference : mean overlap(write_i, v_read_j)
          "how much one head's output feeds another head's CONTENT input"
          -> legitimate composition channel (virtual attention heads / induction
             circuits). Should NOT collapse; if it did, we'd be destroying
             composition, not removing noise.
      qk_self_overlap / ov_self_overlap : i == j diagonal, for reference.
      qk_ov_separation : ov_interference - qk_interference (higher = cleaner
          decoupling of retrieval from content transmission; the paper's headline
          mechanistic number).

    Overlap metric: ||A^T B||_F^2 / min(dim_A, dim_B) in [0, 1] for orthonormal
    bases A, B (average squared cosine of principal angles).

    Cost: one SVD per head-matrix, O(L*H) small SVDs; run at svd_every cadence.
    """

    def _col_basis(M: torch.Tensor, k: int) -> torch.Tensor:
        # Basis for column space of M (C x hd), directions live in R^C.
        U, S, _ = torch.linalg.svd(M.detach().float().cpu(), full_matrices=False)
        r = int((S > 1e-6 * float(S.max().clamp_min(1e-12))).sum())
        return U[:, : min(k, r)].contiguous()

    def _row_basis(M: torch.Tensor, k: int) -> torch.Tensor:
        # Basis for row space of M (hd x C), directions live in R^C.
        _, S, Vh = torch.linalg.svd(M.detach().float().cpu(), full_matrices=False)
        r = int((S > 1e-6 * float(S.max().clamp_min(1e-12))).sum())
        return Vh[: min(k, r), :].T.contiguous()

    def _overlap(A: torch.Tensor, B: torch.Tensor) -> float:
        if A.numel() == 0 or B.numel() == 0:
            return float("nan")
        d = float(min(A.shape[1], B.shape[1]))
        if d <= 0:
            return float("nan")
        return float((A.T @ B).pow(2).sum() / d)

    blocks = list(model.transformer.h)
    if max_layers is not None:
        blocks = blocks[: int(max_layers)]

    qk_cross: list = []
    ov_cross: list = []
    qk_self: list = []
    ov_self: list = []

    for block in blocks:
        attn = block.attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        W = attn.c_attn.weight  # (3C, C): rows [0:C]=Q, [C:2C]=K, [2C:3C]=V
        Wo = attn.c_proj.weight  # (C, C): columns grouped by head

        writes, qk_reads, v_reads = [], [], []
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            writes.append(_col_basis(Wo[:, sl], topk))
            # QK read: what the residual stream must contain for this head to match.
            qk_reads.append(_row_basis(torch.cat([W[sl, :], W[C + h * hd : C + (h + 1) * hd, :]], dim=0), topk))
            v_reads.append(_row_basis(W[2 * C + h * hd : 2 * C + (h + 1) * hd, :], topk))

        for i in range(H):
            for j in range(H):
                oqk = _overlap(writes[i], qk_reads[j])
                oov = _overlap(writes[i], v_reads[j])
                if i == j:
                    qk_self.append(oqk)
                    ov_self.append(oov)
                else:
                    qk_cross.append(oqk)
                    ov_cross.append(oov)

    def _mean(xs: list) -> float:
        xs = [x for x in xs if x == x]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    qk_i = _mean(qk_cross)
    ov_i = _mean(ov_cross)
    return {
        "qk_interference": qk_i,
        "ov_interference": ov_i,
        "qk_self_overlap": _mean(qk_self),
        "ov_self_overlap": _mean(ov_self),
        "qk_ov_separation": (ov_i - qk_i) if (qk_i == qk_i and ov_i == ov_i) else float("nan"),
    }


__all__ = [
    "evaluate_induction",
    "evaluate_distractor_induction",
    "measure_attention_entropy",
    "phase_statistics",
    "hla_statistics",
    "svd_statistics",
    "head_interference_statistics",
    "depth_profile_statistics",
]
