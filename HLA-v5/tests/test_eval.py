"""Eval probe tests: induction metric, entropy, HLA statistics."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval import (  # noqa: E402
    depth_profile_statistics,
    evaluate_distractor_induction,
    evaluate_induction,
    head_interference_statistics,
    hla_statistics,
    measure_attention_entropy,
    phase_statistics,
    svd_statistics,
)
from src.model import GPT, GPTConfig  # noqa: E402


def make_model(vocab_size=50257, **kw):
    cfg = GPTConfig(block_size=64, vocab_size=vocab_size, n_layer=1, n_head=2,
                    n_embd=32, gradient_checkpointing=False, **kw)
    return GPT(cfg)


class TestInduction:
    def test_returns_probability(self):
        model = make_model()
        r = evaluate_induction(model, device="cpu", batch_size=4)
        assert 0.0 <= r <= 1.0

    def test_small_vocab_returns_nan(self):
        model = make_model(vocab_size=40)  # < INDUCTION_TOK_B_OFFSET + batch
        r = evaluate_induction(model, device="cpu", batch_size=32)
        assert r != r  # NaN

    def test_deterministic(self):
        model = make_model()
        r1 = evaluate_induction(model, device="cpu", seed=42)
        r2 = evaluate_induction(model, device="cpu", seed=42)
        assert r1 == r2

    def test_restores_training_mode(self):
        model = make_model().train()
        evaluate_induction(model, device="cpu")
        assert model.training


class TestDistractorInduction:
    # NOTE: memory-conscious settings. Logits are (B, T, V); with V=50257 keep
    # B and T small so CI/CPU boxes don't OOM: (2, 512, 50257) fp32 ~ 200 MB.
    def _model(self):
        cfg = GPTConfig(block_size=512, vocab_size=50257, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        return GPT(cfg)

    def test_returns_valid_metrics_and_deterministic(self):
        m = self._model()
        r1 = evaluate_distractor_induction(m, device="cpu", seed=42, batch_size=2,
                                           n_distractors=8)
        assert 0.0 <= r1["distractor_induction"] <= 1.0
        assert -1.0 <= r1["distractor_margin"] <= 1.0
        r2 = evaluate_distractor_induction(m, device="cpu", seed=42, batch_size=2,
                                           n_distractors=8)
        assert r1 == r2

    def test_small_context_returns_nan(self):
        cfg = GPTConfig(block_size=64, vocab_size=50257, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        r = evaluate_distractor_induction(GPT(cfg), device="cpu")
        assert r["distractor_induction"] != r["distractor_induction"]  # NaN

    def test_restores_training_mode(self):
        m = self._model().train()
        evaluate_distractor_induction(m, device="cpu", batch_size=2, n_distractors=8)
        assert m.training


class TestDepthProfile:
    def test_empty_when_disabled(self):
        m = make_model()
        assert depth_profile_statistics(m) == {}

    def test_layer_temp_readout_at_init(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=4, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        layer_dependent_gate=True, learnable_layer_temp=True)
        s = depth_profile_statistics(GPT(cfg))
        # at init: temp softplus(0)/log2 = 1 -> mult = 1 + depth
        assert abs(s["layer_temp_first"] - 1.0) < 1e-6
        assert abs(s["layer_temp_last"] - (1.0 + 3.0 / 4.0)) < 1e-6

    def test_phase_budget_readout_at_init(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, per_head_phase=True)
        s = depth_profile_statistics(GPT(cfg))
        assert abs(s["phase_budget_mean"] - 1.0) < 1e-6
        assert abs(s["phase_budget_min"] - 1.0) < 1e-6


class TestEntropy:
    def test_positive_bounded(self):
        model = make_model()
        e = measure_attention_entropy(model, device="cpu")
        import math
        assert 0.0 < e < math.log(64) + 1e-6  # entropy <= log(T)


class TestStatistics:
    def test_phase_statistics_at_identity(self):
        model = make_model(phase_mult=0.15)
        per_head, mean = phase_statistics(model)
        assert per_head is not None
        assert mean == 0.0  # identity init => zero phase norms

    def test_svd_statistics_at_identity(self):
        """Phase/gate are zero at init: erank must be NaN (inactive), not 0."""
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        s = svd_statistics(model)
        assert s["phase_erank"] != s["phase_erank"]  # NaN
        assert s["gate_erank"] != s["gate_erank"]    # NaN
        # backbone Q/K/V blocks are randomly initialized -> positive stable rank
        assert s["qk_stable_rank"] > 1.0
        assert s["v_stable_rank"] > 1.0

    def test_svd_statistics_after_perturbation(self):
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.02)
                blk.attn.W_phase_k.normal_(0, 0.02)
                blk.attn.W_gate_k.weight.normal_(0, 0.02)
                blk.attn.W_gate_v.weight.normal_(0, 0.02)
        s = svd_statistics(model)
        head_half = model.config.n_embd // model.config.n_head // 2
        assert 1.0 <= s["phase_erank"] <= head_half + 1e-6
        assert 0.0 < s["phase_top1"] <= 1.0
        assert s["gate_erank"] > 0.0

    def test_svd_rank1_collapse_detected(self):
        """A rank-1 phase matrix must give erank ~= 1 and top1 ~= 1."""
        model = make_model(phase_mult=0.15)
        with torch.no_grad():
            for blk in model.transformer.h:
                H, C, K = blk.attn.W_phase_q.shape
                u = torch.randn(C, 1)
                v = torch.randn(1, K)
                blk.attn.W_phase_q.copy_((u @ v).unsqueeze(0).expand(H, C, K))
        s = svd_statistics(model)
        assert abs(s["phase_erank"] - 1.0) < 0.05
        assert s["phase_top1"] > 0.95

    def test_interference_keys_and_ranges(self):
        model = make_model()
        s = head_interference_statistics(model)
        for k in ("qk_interference", "ov_interference", "qk_self_overlap",
                  "ov_self_overlap", "qk_ov_separation"):
            assert k in s
        for k in ("qk_interference", "ov_interference"):
            assert 0.0 <= s[k] <= 1.0, f"{k}={s[k]} out of [0,1]"

    def test_interference_orthogonal_heads_near_zero(self):
        """Ground truth: heads writing/reading in disjoint coordinate blocks
        must have ~zero cross-head interference."""
        model = make_model()
        attn = model.transformer.h[0].attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        with torch.no_grad():
            attn.c_attn.weight.zero_()
            attn.c_proj.weight.zero_()
            for h in range(H):
                rows = slice(h * hd, (h + 1) * hd)
                cols = slice(h * (C // H), (h + 1) * (C // H))
                # Q, K, V of head h read ONLY from its own block of coords.
                for base in (0, C, 2 * C):
                    blk = attn.c_attn.weight[base + h * hd : base + (h + 1) * hd, cols]
                    blk.copy_(torch.randn_like(blk))
                # head h writes ONLY into its own block of coords.
                attn.c_proj.weight[cols, rows] = torch.randn(C // H, hd)
        s = head_interference_statistics(model, max_layers=1)
        assert s["qk_interference"] < 1e-6
        assert s["ov_interference"] < 1e-6
        # Self overlap: random topk-dim subspaces inside the head's own
        # (C//H)-dim coordinate block overlap ~ topk/(C//H) in expectation
        # (0.5 here); must be clearly nonzero, in contrast to cross ~ 0.
        assert s["qk_self_overlap"] > 0.2

    def test_interference_identical_heads_near_one(self):
        """Ground truth: all heads identical => cross overlap == self overlap ~ 1."""
        model = make_model()
        attn = model.transformer.h[0].attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        with torch.no_grad():
            q0 = torch.randn(hd, C)
            v0 = torch.randn(hd, C)
            o0 = torch.randn(C, hd)
            for h in range(H):
                attn.c_attn.weight[h * hd : (h + 1) * hd, :] = q0
                attn.c_attn.weight[C + h * hd : C + (h + 1) * hd, :] = q0
                attn.c_attn.weight[2 * C + h * hd : 2 * C + (h + 1) * hd, :] = v0
                attn.c_proj.weight[:, h * hd : (h + 1) * hd] = o0
        s = head_interference_statistics(model, max_layers=1)
        assert abs(s["qk_interference"] - s["qk_self_overlap"]) < 1e-5
        assert abs(s["ov_interference"] - s["ov_self_overlap"]) < 1e-5

    def test_hla_statistics_keys(self):
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        x = torch.randint(0, 50257, (1, 16))
        model.eval()
        with torch.no_grad():
            model(x)
        stats = hla_statistics(model)
        assert "mix_k_mean" in stats and "mix_v_mean" in stats
        # identity init => mix must be exactly 1
        assert abs(stats["mix_k_mean"] - 1.0) < 1e-6
        assert abs(stats["mix_v_mean"] - 1.0) < 1e-6
