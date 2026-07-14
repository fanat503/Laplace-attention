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
    perturbation_bounds,
    per_layer_gate_analysis,
    mechanism_gradient_statistics,
    attention_head_similarity,
    mechanism_knockout,
    prefix_matching_score,
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
        # batch_size=8: default 32 materializes (32,256,50257) fp32 ~ 1.6 GB
        # logits - fine on TPU, OOM on small CI hosts (found in panel sweep).
        model = make_model()
        r1 = evaluate_induction(model, device="cpu", seed=42, batch_size=8)
        r2 = evaluate_induction(model, device="cpu", seed=42, batch_size=8)
        assert r1 == r2

    def test_restores_training_mode(self):
        model = make_model().train()
        evaluate_induction(model, device="cpu", batch_size=8)
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


class TestPerturbationBounds:
    def _hla(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        return GPT(cfg)

    def test_actual_within_theoretical(self):
        """Core Theorem-4 check: captured mix never exceeds the envelope,
        even with saturated gates."""
        m = self._hla()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 100.0)  # saturate
                blk.attn.W_range_k.normal_(0, 100.0)
        out = perturbation_bounds(m)
        for i in range(2):
            assert out[f"L{i:02d}_mix_k_max"] <= out[f"L{i:02d}_theo_mix_k_max"] + 1e-4
            assert out[f"L{i:02d}_mix_k_min"] >= out[f"L{i:02d}_theo_mix_k_min"] - 1e-4
            assert 0.0 <= out[f"L{i:02d}_util_k_up"] <= 1.0 + 1e-6

    def test_saturated_gates_reach_utilization_one(self):
        m = self._hla()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
        out = perturbation_bounds(m)
        assert out["L00_util_k_up"] > 0.99, "saturation must reach the envelope"

    def test_identity_init_dormant(self):
        out = perturbation_bounds(self._hla())
        assert abs(out["L00_util_k_up"]) < 1e-6, "identity init must show ~0 utilization"

    def test_restores_model_state(self):
        m = self._hla().train()
        perturbation_bounds(m)
        assert m.training
        assert m.transformer.h[0].attn.capture_diagnostics is False


class TestPerLayerGateAnalysis:
    def test_per_layer_keys_and_depth_scaling(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=4, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        layer_dependent_gate=True)
        m = GPT(cfg)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
        out = per_layer_gate_analysis(m)
        for i in range(4):
            assert f"L{i:02d}_mix_k_mean" in out
            assert f"L{i:02d}_attn_entropy" in out
        # deeper layer => wider envelope => larger mix at saturation
        assert out["L03_mix_k_mean"] > out["L00_mix_k_mean"]


class TestMechanismGradientStatistics:
    def test_active_nonzero_inactive_zero_and_summary(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        use_salience_bias=True, salience_alpha=0.0)  # salience OFF
        m = GPT(cfg)
        x = torch.randint(0, 256, (2, 32))
        _, loss = m(x, x)
        loss.backward()
        m.capture_mechanism_grad_norms()
        out = mechanism_gradient_statistics(m)
        assert out["L00_grad_phase_q"] > 0.0, "active mechanism must have gradient (Theorem 5)"
        assert out["L00_grad_gate_sal"] == 0.0, "inactive mechanism must record exact 0"
        assert out["mech_grad_min"] > 0.0, "summary must exclude inactive zeros"

    def test_empty_before_capture(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        assert mechanism_gradient_statistics(GPT(cfg)) == {}


class TestProbeNonInterference:
    """Review attack R-E: probes and captures must NEVER alter training.

    Sterility invariant I5 extends to instrumentation: enabling metrics
    cannot change the trajectory by a single bit."""

    def test_grad_capture_does_not_change_weights(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, phase_mult=0.15)

        def train3(capture):
            torch.manual_seed(0)
            m = GPT(cfg)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            torch.manual_seed(1)
            for _ in range(3):
                x = torch.randint(0, 128, (2, 16))
                _, l = m(x, x)
                l.backward()
                opt.step()
                if capture:
                    m.capture_mechanism_grad_norms()
                opt.zero_grad(set_to_none=True)
            return [p.detach().clone() for p in m.parameters()]

        pa, pb = train3(False), train3(True)
        assert all(torch.equal(a, b) for a, b in zip(pa, pb)), \
            "grad capture altered the training trajectory!"

    def test_perturbation_bounds_does_not_change_weights(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        before = [p.detach().clone() for p in m.parameters()]
        perturbation_bounds(m)
        after = list(m.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_captured_norms_survive_zero_grad(self):
        """R-C: snapshot must be a copy, not a view of .grad."""
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, phase_mult=0.15)
        m = GPT(cfg)
        x = torch.randint(0, 128, (2, 16))
        _, loss = m(x, x)
        loss.backward()
        m.capture_mechanism_grad_norms()
        v1 = float(m.transformer.h[0].attn.last_mechanism_grad_norms["phase_q"])
        for p in m.parameters():
            p.grad = None
        v2 = float(m.transformer.h[0].attn.last_mechanism_grad_norms["phase_q"])
        assert v1 == v2 and v1 > 0.0


class TestProbeEdgeCases:
    """Review attack R-A/R-G: probes under unusual but legal configs."""

    def test_bounds_track_learned_layer_temp(self):
        """R-A: with theta != 0 the envelope must use the LIVE lambda."""
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0,
                        layer_dependent_gate=True, learnable_layer_temp=True)
        m = GPT(cfg)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
                blk.attn.W_layer_temp.fill_(2.0)  # learned, non-default
        out = perturbation_bounds(m)
        assert out["L01_util_k_up"] > 0.99, \
            "envelope must track live softplus(theta), not the static heuristic"
        assert out["L01_mix_k_max"] <= out["L01_theo_mix_k_max"] + 1e-4

    def test_bounds_empty_when_laplace_off(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, use_laplace=False)
        assert perturbation_bounds(GPT(cfg)) == {}

    def test_per_layer_analysis_no_crash_minimal_model(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, use_laplace=False)
        out = per_layer_gate_analysis(GPT(cfg))
        assert "L00_attn_entropy" in out


class TestProbeStateRestoration:
    """Review attack P1: probes must restore the CALLER's diagnostics state,
    not blindly reset it."""

    def test_perturbation_bounds_restores_enabled_diagnostics(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        m.set_diagnostics(enabled=True, capture_attention=True)
        perturbation_bounds(m)
        assert m.transformer.h[0].attn.capture_diagnostics is True
        assert m.transformer.h[0].attn.capture_attention is True

    def test_perturbation_bounds_restores_disabled_diagnostics(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        perturbation_bounds(m)
        assert m.transformer.h[0].attn.capture_diagnostics is False

    def test_per_layer_analysis_restores_state(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False)
        m = GPT(cfg)
        m.set_diagnostics(enabled=True)
        per_layer_gate_analysis(m)
        assert m.transformer.h[0].attn.capture_diagnostics is True


class TestAttentionHeadSimilarity:
    """Nanda attack: activation-space head redundancy (JS between heads)."""

    def _model(self, **kw):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False, **kw)
        return GPT(cfg)

    def test_identical_heads_zero_js(self):
        m = self._model()
        attn = m.transformer.h[0].attn
        with torch.no_grad():
            C, hd = attn.n_embd, attn.head_dim
            w = attn.c_attn.weight
            # head 1 copies head 0 for Q and K -> identical attention maps
            w[hd:2*hd, :] = w[0:hd, :]
            w[C+hd:C+2*hd, :] = w[C:C+hd, :]
        out = attention_head_similarity(m)
        assert out["head_js_mean"] < 1e-6, "identical heads must give JS ~ 0"

    def test_js_bounded_and_state_restored(self):
        import math
        m = self._model(phase_mult=0.15)
        m.set_diagnostics(enabled=True, capture_attention=True)
        out = attention_head_similarity(m)
        assert 0.0 <= out["head_js_mean"] <= math.log(2.0) + 1e-6
        assert m.transformer.h[0].attn.capture_attention is True  # restored

    def test_deterministic(self):
        m = self._model(phase_mult=0.15)
        assert attention_head_similarity(m, seed=1) == attention_head_similarity(m, seed=1)


class TestAngleStd:
    """Nanda attack: mean |angle| hides bimodality - std must be captured."""

    def test_std_zero_at_identity_nonzero_when_active(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False, phase_mult=0.15)
        m = GPT(cfg).eval()
        x = torch.randint(0, 256, (1, 32))
        m(x)
        assert float(m.transformer.h[0].attn.last_angle_q_std) == 0.0
        with torch.no_grad():
            m.transformer.h[0].attn.W_phase_q.normal_(0, 0.5)
        m(x)
        assert float(m.transformer.h[0].attn.last_angle_q_std) > 0.0


class TestMechanismKnockout:
    def _model(self):
        c = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                      n_embd=32, gradient_checkpointing=False,
                      phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        m = GPT(c)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.3)
                blk.attn.W_gate_k.weight.normal_(0, 0.3)
        return m

    def test_restoration_exact(self):
        m = self._model().eval()
        x = torch.randint(0, 256, (2, 32))
        l_before, _ = m(x)
        mechanism_knockout(m, x, x)
        l_after, _ = m(x)
        assert torch.equal(l_before, l_after), "knockout must restore exactly"

    def test_active_mechanism_has_effect(self):
        m = self._model()
        x = torch.randint(0, 256, (2, 32))
        out = mechanism_knockout(m, x, x)
        assert out["ko_phase_delta"] != 0.0, "randomized phase must carry load"

    def test_absent_mechanism_zero_delta(self):
        m = self._model()
        x = torch.randint(0, 256, (2, 32))
        out = mechanism_knockout(m, x, x)
        assert out["ko_salience_delta"] == 0.0  # salience_alpha=0 in config


class TestPrefixMatching:
    # Memory note: logits are (B,T,50257); keep B=1, block=128 so tiny CI
    # hosts survive ((1,128,50257) fp32 ~ 25 MB).
    def test_keys_and_range(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=2, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = prefix_matching_score(GPT(c), batch_size=1)
        assert "prefix_match_global_max" in out
        assert 0.0 <= out["prefix_match_global_max"] <= 1.0
        assert "L00_prefix_match_max" in out and "L01_prefix_match_mean" in out

    def test_state_restored_and_deterministic(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        m = GPT(c)
        r1 = prefix_matching_score(m, seed=7, batch_size=1)
        r2 = prefix_matching_score(m, seed=7, batch_size=1)
        assert r1 == r2
        assert m.transformer.h[0].attn.capture_attention is False


class TestRangeFlexConsistency:
    """Final polish sweep: range_flex became a config knob, but
    perturbation_bounds still hardcoded the historical 1.25 factor -
    with range_flex != 0.25 the probe reported a silently WRONG envelope."""

    def _hla_flex(self, flex):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        range_flex=flex)
        return GPT(cfg)

    def test_envelope_tracks_range_flex(self):
        import math as _m
        for flex in (0.25, 0.5, 1.0):
            m = self._hla_flex(flex)
            out = perturbation_bounds(m)
            attn = m.transformer.h[0].attn
            lam = float(attn.layer_gate_multiplier)
            eff = lam * min(attn.laplace_alpha * attn.laplace_range_k * (1.0 + flex),
                            attn.k_log_clip)
            expected_hi = (1 - attn.beta_k) + attn.beta_k * _m.exp(eff)
            got = out["L00_theo_mix_k_max"]
            assert abs(got - expected_hi) < 1e-9, (
                f"range_flex={flex}: probe envelope {got} != model envelope "
                f"{expected_hi} (hardcoded 1.25 regression)")

    def test_saturated_mix_stays_inside_flex_envelope(self):
        """With range_flex=1.0 the true envelope is WIDER than the 1.25 one:
        a saturated gate must still be inside the probe's reported bound."""
        m = self._hla_flex(1.0)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
        out = perturbation_bounds(m)
        assert out["L00_mix_k_max"] <= out["L00_theo_mix_k_max"] + 1e-4
        assert out["L00_util_k_up"] > 0.99
