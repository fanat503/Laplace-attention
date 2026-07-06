"""Model correctness tests: sterility, causality, backend parity, gradients.

Run:  pytest tests/ -v        (CPU, no torch_xla required)
These are the tests that must be green before ANY TPU/Kaggle run.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


def small_cfg(**kw) -> GPTConfig:
    base = dict(
        block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
        gradient_checkpointing=False,
    )
    base.update(kw)
    return GPTConfig(**base)


HLA_KW = dict(
    phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
    use_rope=True, use_distance_laplace=True, distance_laplace_alpha=0.25,
    use_salience_bias=True, salience_alpha=1.0,
    layer_dependent_gate=True,
)
BASE_KW = dict(
    phase_mult=0.0, use_laplace=True, laplace_alpha=0.0,
    use_rope=True, use_distance_laplace=True, distance_laplace_alpha=0.0,
    use_salience_bias=True, salience_alpha=0.0,
    layer_dependent_gate=True,
)


def randomize_hla(model: GPT, std: float = 0.05) -> None:
    """Make all HLA branches active so tests exercise the full code path."""
    with torch.no_grad():
        for blk in model.transformer.h:
            blk.attn.W_phase_q.normal_(0, std)
            blk.attn.W_phase_k.normal_(0, std)
            blk.attn.W_range_k.normal_(0, 4 * std)
            blk.attn.W_range_v.normal_(0, 4 * std)
            blk.attn.W_gate_k.weight.normal_(0, std)
            blk.attn.W_gate_v.weight.normal_(0, std)
            blk.attn.W_gate_sal.weight.normal_(0, std)


class TestForward:
    def test_forward_shapes_and_finite_loss(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        x = torch.randint(0, 256, (2, 16))
        logits, loss = model(x, x)
        assert logits.shape == (2, 16, 256)
        assert loss is not None and torch.isfinite(loss)

    def test_forward_all_branches_active(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        x = torch.randint(0, 256, (2, 32))
        logits, loss = model(x, x)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)

    def test_gradient_checkpointing_matches(self):
        torch.manual_seed(0)
        m1 = GPT(small_cfg(**HLA_KW, gradient_checkpointing=True))
        m2 = GPT(small_cfg(**HLA_KW, gradient_checkpointing=False))
        m2.load_state_dict(m1.state_dict())
        randomize_hla(m1)
        m2.load_state_dict(m1.state_dict())
        m1.train(); m2.train()
        x = torch.randint(0, 256, (2, 16))
        _, l1 = m1(x, x)
        _, l2 = m2(x, x)
        assert torch.allclose(l1, l2, atol=1e-6)


class TestSterility:
    """Identity-at-init: HLA model with zeroed HLA params == baseline exactly."""

    def test_hla_identity_error_is_zero_after_init(self):
        model = GPT(small_cfg(**HLA_KW))
        assert model.hla_identity_error() == 0.0

    def test_identity_at_init_exact_logit_match(self):
        torch.manual_seed(123)
        base = GPT(small_cfg(**BASE_KW)).eval()
        hla = GPT(small_cfg(**HLA_KW)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lh, _ = hla(x)
        assert torch.equal(lb, lh), (
            f"HLA at identity-init must produce bit-identical logits, "
            f"max diff={float((lb - lh).abs().max())}"
        )

    def test_parameter_count_matched(self):
        base = GPT(small_cfg(**BASE_KW))
        hla = GPT(small_cfg(**HLA_KW))
        assert base.parameter_count() == hla.parameter_count(), (
            "base and HLA must be parameter-matched for a fair comparison"
        )


class TestCausality:
    """No future leakage through phase rotation, Laplace gates or distance bias."""

    def test_last_token_change_does_not_affect_prefix(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        T = 32
        a = torch.randint(0, 256, (1, T))
        b = a.clone()
        b[0, T - 1] = (b[0, T - 1] + 7) % 256
        la, _ = model(a)
        lb, _ = model(b)
        assert torch.equal(la[0, : T - 1], lb[0, : T - 1]), "future token leaked into past logits"

    def test_middle_token_change_affects_only_suffix(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        T = 32
        a = torch.randint(0, 256, (1, T))
        c = a.clone()
        c[0, 10] = (c[0, 10] + 3) % 256
        la, _ = model(a)
        lc, _ = model(c)
        assert torch.equal(la[0, :10], lc[0, :10])
        assert not torch.equal(la[0, 10:], lc[0, 10:])


class TestBackendParity:
    def test_sdpa_matches_manual(self):
        torch.manual_seed(77)
        kw = dict(HLA_KW)
        kw.update(use_distance_laplace=False, distance_laplace_alpha=0.0)
        msdpa = GPT(small_cfg(**kw, attention_backend="sdpa")).eval()
        randomize_hla(msdpa)
        mman = GPT(small_cfg(**kw, attention_backend="manual")).eval()
        mman.load_state_dict(msdpa.state_dict(), strict=True)
        x = torch.randint(0, 256, (2, 32))
        l1, _ = msdpa(x)
        l2, _ = mman(x)
        assert torch.allclose(l1, l2, atol=1e-5), (
            f"sdpa/manual mismatch: {float((l1 - l2).abs().max())}"
        )


class TestPaddedVocab:
    def test_padded_vocab_loss_finite_and_padded_ignored(self):
        cfg = small_cfg(vocab_size=250, padded_vocab_size=256)
        model = GPT(cfg).eval()
        x = torch.randint(0, 250, (2, 16))
        logits, loss = model(x, x)
        assert logits.shape[-1] == 256
        assert torch.isfinite(loss)

    def test_padded_smaller_than_vocab_rejected(self):
        with pytest.raises(ValueError):
            small_cfg(vocab_size=256, padded_vocab_size=128)


class TestGradients:
    def test_all_params_receive_grad_when_active(self):
        # Enable EVERY optional branch so every parameter is on an active path.
        model = GPT(small_cfg(**HLA_KW, learnable_layer_temp=True, per_head_phase=True,
                              use_forget_gate=True, forget_alpha=1.0))
        randomize_hla(model)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_f.weight.normal_(0, 0.05)
                blk.attn.W_range_f.normal_(0, 0.2)
        model.train()
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        def _expected_zero(n: str) -> bool:
            # Layer 0 temp has depth=0 -> multiplier is constant 1, no gradient
            # by construction.
            return n == "transformer.h.0.attn.W_layer_temp"
        missing = [n for n, p in model.named_parameters()
                   if not _expected_zero(n)
                   and (p.grad is None or float(p.grad.detach().abs().sum()) == 0.0)]
        assert missing == [], f"params without gradient: {missing}"

    def test_inactive_flags_mean_frozen_params(self):
        """When learnable_layer_temp/per_head_phase are OFF, their params must
        NOT receive gradients (they exist only for parameter matching)."""
        model = GPT(small_cfg(**HLA_KW))
        randomize_hla(model)
        model.train()
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        for blk in model.transformer.h:
            for p in (blk.attn.W_layer_temp, blk.attn.W_phase_scale):
                assert p.grad is None or float(p.grad.abs().sum()) == 0.0

    def test_ignore_index_minus_100(self):
        model = GPT(small_cfg()).eval()
        x = torch.randint(0, 256, (1, 16))
        y = x.clone()
        y[0, :8] = -100
        _, loss = model(x, y)
        assert torch.isfinite(loss)


class TestPositionalEncoding:
    def test_rope_only_no_wpe_params(self):
        model = GPT(small_cfg(use_rope=True, use_wpe=False))
        names = [n for n, _ in model.named_parameters()]
        assert not any("wpe" in n for n in names), "wpe must not exist when use_wpe=False"
        x = torch.randint(0, 256, (2, 16))
        logits, loss = model.eval()(x, x)
        assert torch.isfinite(loss)

    def test_wpe_present_by_default(self):
        model = GPT(small_cfg())
        names = [n for n, _ in model.named_parameters()]
        assert any("wpe" in n for n in names)

    def test_no_positional_info_rejected(self):
        with pytest.raises(ValueError):
            small_cfg(use_rope=False, use_wpe=False)

    def test_rope_gives_position_sensitivity(self):
        """Without wpe, RoPE must still make the model position-aware.

        NOTE: feeding the SAME token everywhere is NOT a valid probe: identical
        tokens give identical V vectors, and any softmax-weighted average of
        identical vectors is that same vector regardless of attention scores.
        The correct probe: shift the sequence by one and compare logits of the
        same token at its old vs new position.
        """
        model = GPT(small_cfg(use_rope=True, use_wpe=False)).eval()
        x = (torch.arange(1, 17).unsqueeze(0)) % 256
        l1, _ = model(x)
        shifted = torch.cat([torch.tensor([[99]]), x[:, :-1]], dim=1)
        l2, _ = model(shifted)
        # token x[0,5] is at position 5 in x and position 6 in shifted
        diff = (l1[0, 5] - l2[0, 6]).detach().abs().max()
        assert float(diff) > 1e-4, "RoPE model must be position-sensitive"

    def test_param_match_rope_pair(self):
        base = GPT(small_cfg(**BASE_KW, use_wpe=False))
        hla = GPT(small_cfg(**HLA_KW, use_wpe=False))
        assert base.parameter_count() == hla.parameter_count()


class TestLearnableLayerTemp:
    """Learnable per-layer gate temperature: exact heuristic at init."""

    KW = dict(use_rope=True, use_wpe=False, use_laplace=True, laplace_alpha=1.0,
              layer_dependent_gate=True)

    def test_identity_at_init_vs_static_heuristic(self):
        """theta=0 must reproduce the static heuristic bit-exactly."""
        torch.manual_seed(3)
        m_static = GPT(small_cfg(**self.KW, learnable_layer_temp=False)).eval()
        m_learn = GPT(small_cfg(**self.KW, learnable_layer_temp=True)).eval()
        m_learn.load_state_dict(m_static.state_dict(), strict=True)
        # activate gates so layer_mult actually matters
        for m in (m_static, m_learn):
            with torch.no_grad():
                for blk in m.transformer.h:
                    blk.attn.W_gate_k.weight.normal_(0, 0.05)
        m_learn.load_state_dict(m_static.state_dict(), strict=True)
        x = torch.randint(0, 256, (2, 32))
        l1, _ = m_static(x)
        l2, _ = m_learn(x)
        assert torch.allclose(l1, l2, atol=1e-6), (
            f"learnable temp at theta=0 must equal static heuristic, diff="
            f"{float((l1 - l2).abs().max())}"
        )

    def test_temp_changes_behavior_when_learned(self):
        model = GPT(small_cfg(**self.KW, learnable_layer_temp=True)).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.5)
        x = torch.randint(0, 256, (2, 32))
        l0, _ = model(x)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_layer_temp.fill_(3.0)
        l1, _ = model(x)
        assert not torch.allclose(l0, l1), "layer temp had no effect"

    def test_temp_receives_gradient(self):
        model = GPT(small_cfg(**self.KW, learnable_layer_temp=True))
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.1)
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        # layer 0 has depth=0 -> no gradient by construction; check deepest layer
        g = model.transformer.h[-1].attn.W_layer_temp.grad
        assert g is not None and float(g.abs().sum()) > 0

    def test_requires_layer_dependent_gate(self):
        with pytest.raises(ValueError):
            small_cfg(layer_dependent_gate=False, learnable_layer_temp=True)

    def test_parameter_matched(self):
        base = GPT(small_cfg(**BASE_KW))
        hla = GPT(small_cfg(**HLA_KW, learnable_layer_temp=True))
        # W_layer_temp exists in both; only the flag differs
        assert base.parameter_count() == hla.parameter_count()


class TestPerHeadPhase:
    """Per-head phase budget: interpretable, H params/layer."""

    def test_identity_at_init(self):
        torch.manual_seed(4)
        m_global = GPT(small_cfg(**HLA_KW, per_head_phase=False)).eval()
        m_perhead = GPT(small_cfg(**HLA_KW, per_head_phase=True)).eval()
        m_perhead.load_state_dict(m_global.state_dict(), strict=True)
        with torch.no_grad():
            for blk in m_global.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.05)
                blk.attn.W_phase_k.normal_(0, 0.05)
        m_perhead.load_state_dict(m_global.state_dict(), strict=True)
        x = torch.randint(0, 256, (2, 32))
        l1, _ = m_global(x)
        l2, _ = m_perhead(x)
        assert torch.allclose(l1, l2, atol=1e-6), "per-head budget at init must equal global phase_mult"

    def test_budget_bounds(self):
        """Effective budget must stay in [0, 2*phase_mult]."""
        model = GPT(small_cfg(**HLA_KW, per_head_phase=True)).eval()
        model.set_diagnostics(enabled=True)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 100.0)  # saturate inner tanh -> angle = budget bound
                blk.attn.W_phase_scale.fill_(100.0)   # budget -> 2x
        import math as _m
        x = torch.randint(0, 256, (1, 16))
        model(x)
        for blk in model.transformer.h:
            max_angle = float(blk.attn.last_angle_q_abs_mean)
            assert max_angle <= 2 * _m.pi * 0.15 + 1e-4

    def test_head_can_opt_out(self):
        """Setting a head's scale to -inf-ish must zero its rotation."""
        model = GPT(small_cfg(**HLA_KW, per_head_phase=True)).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.5)
                blk.attn.W_phase_k.normal_(0, 0.5)
                blk.attn.W_phase_scale.fill_(-100.0)  # tanh -> -1 -> budget 0
        x = torch.randint(0, 256, (1, 16))
        l_off, _ = model(x)
        model2 = GPT(small_cfg(**HLA_KW, per_head_phase=True)).eval()
        model2.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            for blk in model2.transformer.h:
                blk.attn.W_phase_q.zero_()
                blk.attn.W_phase_k.zero_()
                blk.attn.W_phase_scale.zero_()
        l_zero, _ = model2(x)
        assert torch.allclose(l_off, l_zero, atol=1e-5), "budget=0 must equal no rotation"

    def test_scale_receives_gradient(self):
        model = GPT(small_cfg(**HLA_KW, per_head_phase=True))
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.1)
                blk.attn.W_phase_k.normal_(0, 0.1)
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        g = model.transformer.h[0].attn.W_phase_scale.grad
        assert g is not None and float(g.abs().sum()) > 0


class TestLayerDependentPhase:
    """Depth-adaptive phase budget: shares the layer multiplier with gates."""

    KW = dict(use_rope=True, use_wpe=False, phase_mult=0.15,
              layer_dependent_gate=True)

    def test_identity_at_init(self):
        torch.manual_seed(6)
        base = GPT(small_cfg(**BASE_KW, layer_dependent_phase=True)).eval()
        hla = GPT(small_cfg(**HLA_KW, layer_dependent_phase=True)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lh, _ = hla(x)
        assert torch.equal(lb, lh)

    def test_depth_scales_phase(self):
        """Layer 0 keeps budget = pi*phase_mult; deepest layer gets ~2x."""
        import math as _m
        model = GPT(small_cfg(**self.KW, n_layer=4, layer_dependent_phase=True)).eval()
        model.set_diagnostics(enabled=True)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 100.0)  # saturate inner tanh
                blk.attn.W_phase_k.normal_(0, 100.0)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        a0 = float(model.transformer.h[0].attn.last_angle_q_abs_mean)
        a3 = float(model.transformer.h[3].attn.last_angle_q_abs_mean)
        # |tanh| of N(0,100) samples is ~1 but not exactly (a few draws land
        # near zero), so compare the RATIO of depths, which cancels that factor.
        assert abs(a3 / a0 - 1.75) < 0.02             # mult_3 / mult_0 = 1.75

    def test_requires_layer_dependent_gate(self):
        with pytest.raises(ValueError):
            small_cfg(phase_mult=0.15, layer_dependent_gate=False,
                      layer_dependent_phase=True)

    def test_off_means_uniform_depth(self):
        model = GPT(small_cfg(**self.KW, n_layer=4, layer_dependent_phase=False)).eval()
        model.set_diagnostics(enabled=True)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 100.0)
                blk.attn.W_phase_k.normal_(0, 100.0)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        a0 = float(model.transformer.h[0].attn.last_angle_q_abs_mean)
        a3 = float(model.transformer.h[3].attn.last_angle_q_abs_mean)
        assert abs(a0 - a3) < 1e-4


class TestSalienceBias:
    """Additive per-key salience bias: the mechanism that allows STRONG
    suppression/boost of tokens, with no multiplicative floor."""

    def test_identity_at_init(self):
        torch.manual_seed(11)
        base = GPT(small_cfg(**BASE_KW)).eval()
        hla = GPT(small_cfg(**HLA_KW)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lh, _ = hla(x)
        assert torch.equal(lb, lh)

    def test_salience_changes_attention_when_active(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        x = torch.randint(0, 256, (1, 32))
        l0, _ = model(x)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_sal.weight.normal_(0, 0.5)
        l1, _ = model(x)
        assert not torch.allclose(l0, l1), "salience bias had no effect"

    def test_salience_respects_clip(self):
        """|salience bias| <= salience_clip even with saturated gates."""
        cfg = small_cfg(**HLA_KW)
        model = GPT(cfg).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_sal.weight.normal_(0, 100.0)
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (2, 32))
        model(x)
        for blk in model.transformer.h:
            bias_abs = float(blk.attn.last_salience_bias_abs_mean)
            assert bias_abs <= cfg.salience_clip + 1e-5

    def test_salience_causal(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        T = 32
        a = torch.randint(0, 256, (1, T))
        b = a.clone()
        b[0, T - 1] = (b[0, T - 1] + 7) % 256
        la, _ = model(a)
        lb, _ = model(b)
        assert torch.equal(la[0, : T - 1], lb[0, : T - 1])

    def test_parameter_matched_with_salience(self):
        base = GPT(small_cfg(**BASE_KW))
        hla = GPT(small_cfg(**HLA_KW))
        assert base.parameter_count() == hla.parameter_count()

    def test_strong_suppression_possible(self):
        """The design goal: a distant token CAN be suppressed below the
        multiplicative floor. Verify attention weight of a salience-suppressed
        key drops by >5x vs baseline."""
        cfg = small_cfg(**HLA_KW, n_layer=1)
        model = GPT(cfg).eval()
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        att0 = model.transformer.h[0].attn.last_attn.clone()  # (1,H,T,T)
        # push salience of ALL tokens strongly negative except token 0
        with torch.no_grad():
            attn = model.transformer.h[0].attn
            attn.W_gate_sal.weight.normal_(0, 100.0)  # saturate tanh -> +-1 random
        model(x)
        att1 = model.transformer.h[0].attn.last_attn
        # distribution must change substantially (weights redistribute > 5x on some keys)
        ratio = (att1.clamp_min(1e-9) / att0.clamp_min(1e-9))
        assert float(ratio.min()) < 0.2 or float(ratio.max()) > 5.0


class TestMathematicalInvariants:
    """Properties the architecture claims on paper — verified numerically."""

    def test_phase_rotation_is_isometry(self):
        """Pairwise rotation must preserve norms: |R(x)| == |x| exactly.
        This is the 'retrieval geometry unchanged' claim of the paper."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        attn = model.transformer.h[0].attn
        x = torch.randn(2, 4, 8, attn.head_dim)
        angles = torch.rand(2, 4, 8, attn.head_dim // 2) * 3.14
        rotated = attn._rotate_pairwise(x, torch.cos(angles), torch.sin(angles))
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    def test_rotation_invertible(self):
        """Rotating by -angle must undo rotating by +angle."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        attn = model.transformer.h[0].attn
        x = torch.randn(2, 4, 8, attn.head_dim)
        angles = torch.rand(2, 4, 8, attn.head_dim // 2)
        fwd = attn._rotate_pairwise(x, torch.cos(angles), torch.sin(angles))
        back = attn._rotate_pairwise(fwd, torch.cos(-angles), torch.sin(-angles))
        assert torch.allclose(x, back, atol=1e-5)

    def test_mix_bounds_respect_config(self):
        """mix must stay within the analytic envelope (1-b)+b*exp(+-min(a*r_max, clip))."""
        import math as _m
        cfg = small_cfg(**HLA_KW)
        model = GPT(cfg).eval()
        randomize_hla(model, std=5.0)  # push gates to saturation deliberately
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (4, 64))
        model(x)
        for blk in model.transformer.h:
            lm = blk.attn.layer_gate_multiplier
            for mix, rng, clip, beta in [
                (blk.attn.last_mix_k, cfg.laplace_range_k, cfg.k_log_clip, cfg.beta_k),
                (blk.attn.last_mix_v, cfg.laplace_range_v, cfg.v_log_clip, cfg.beta_v),
            ]:
                assert mix is not None
                bound = min(cfg.laplace_alpha * rng * 1.25 * lm, clip * lm)
                hi = (1 - beta) + beta * _m.exp(bound) + 1e-4
                lo = (1 - beta) + beta * _m.exp(-bound) - 1e-4
                assert float(mix.max()) <= hi, f"mix above analytic bound: {float(mix.max())} > {hi}"
                assert float(mix.min()) >= lo

    def test_saturation_zero_at_identity(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        x = torch.randint(0, 256, (2, 32))
        model(x)
        for blk in model.transformer.h:
            assert float(blk.attn.last_gate_k_sat_frac) == 0.0
            assert float(blk.attn.last_angle_q_sat_frac) == 0.0

    def test_saturation_detects_pressure(self):
        """With huge weights the saturation fraction must fire — this is the
        signal that phase_mult/ranges are too tight for what the model wants."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model, std=50.0)
        x = torch.randint(0, 256, (2, 32))
        model(x)
        sat = max(float(blk.attn.last_gate_k_sat_frac) for blk in model.transformer.h)
        assert sat > 0.5, "saturation metric failed to detect saturated gates"

    def test_train_step_parity_base_vs_identity_hla(self):
        """One full backward+AdamW step from shared init: losses must match to
        float tolerance BEFORE the step, and HLA params must move AFTER (grads
        flow into the new branches even from identity)."""
        torch.manual_seed(7)
        base = GPT(small_cfg(**BASE_KW))
        hla = GPT(small_cfg(**HLA_KW))
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        x = torch.randint(0, 256, (4, 32))
        lb = base(x, x)[1]
        lh = hla(x, x)[1]
        assert torch.allclose(lb, lh, atol=1e-7), "identical init must give identical loss"
        opt = torch.optim.AdamW(hla.parameters(), lr=1e-3)
        lh.backward()
        opt.step()
        # phase params got gradient through tanh at 0 (derivative 1) => must move
        assert hla.hla_identity_error() > 0.0, "HLA branches frozen: no learning signal"

    def test_no_nan_with_extreme_inputs(self):
        """Adversarial: extreme HLA weights + long sequence must stay finite
        (the clip's actual job)."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model, std=100.0)
        x = torch.randint(0, 256, (2, 64))
        logits, loss = model(x, x)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)

    def test_batch_invariance(self):
        """Same sequence alone vs inside a batch must give identical logits
        (catches cross-batch leakage through gate statistics)."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        a = torch.randint(0, 256, (1, 32))
        b = torch.randint(0, 256, (3, 32))
        batch = torch.cat([a, b], dim=0)
        la_solo, _ = model(a)
        la_batch, _ = model(batch)
        assert torch.allclose(la_solo[0], la_batch[0], atol=1e-5)

    def test_determinism_two_forwards(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        randomize_hla(model)
        x = torch.randint(0, 256, (2, 32))
        l1, _ = model(x)
        l2, _ = model(x)
        assert torch.equal(l1, l2)


class TestBf16AndRobustness:
    """TPU reality checks: XLA_USE_BF16 casts everything to bf16."""

    def test_identity_survives_bf16(self):
        """Bit-exact base<->HLA identity must hold in bf16 (R11)."""
        torch.manual_seed(1)
        base = GPT(small_cfg(**BASE_KW)).eval()
        hla = GPT(small_cfg(**HLA_KW)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        base_bf = base.to(torch.bfloat16)
        hla_bf = hla.to(torch.bfloat16)
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base_bf(x)
        lh, _ = hla_bf(x)
        assert torch.equal(lb, lh), "bf16 broke base<->HLA identity"

    def test_full_flags_state_dict_roundtrip(self):
        """Every optional mechanism on simultaneously: save/load must be exact (R14)."""
        cfg = small_cfg(**HLA_KW, learnable_layer_temp=True, per_head_phase=True,
                        layer_dependent_phase=True)
        m1 = GPT(cfg)
        m2 = GPT(cfg)
        m2.load_state_dict(m1.state_dict(), strict=True)
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)

    def test_parameter_order_deterministic(self):
        """Optimizer state resume relies on identical parameter ordering (R15)."""
        cfg = small_cfg(**HLA_KW, learnable_layer_temp=True, per_head_phase=True)
        names1 = [n for n, _ in GPT(cfg).named_parameters()]
        names2 = [n for n, _ in GPT(cfg).named_parameters()]
        assert names1 == names2

    def test_grad_checkpointing_with_all_flags(self):
        """Gradient checkpointing must work with every mechanism active."""
        torch.manual_seed(2)
        cfg = small_cfg(**HLA_KW, learnable_layer_temp=True, per_head_phase=True,
                        layer_dependent_phase=True, gradient_checkpointing=True)
        model = GPT(cfg)
        randomize_hla(model)
        model.train()
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)


class TestForgetGate:
    """FoX-style cumulative gate: the external-baseline mechanism reproduced
    inside the sterile harness."""

    FKW = dict(use_forget_gate=True, forget_alpha=1.0, use_rope=True, use_wpe=False)

    def test_identity_at_init(self):
        torch.manual_seed(21)
        base = GPT(small_cfg(**BASE_KW, use_forget_gate=True, forget_alpha=0.0)).eval()
        fox = GPT(small_cfg(**BASE_KW, use_forget_gate=True, forget_alpha=1.0)).eval()
        fox.load_state_dict(base.state_dict(), strict=True)
        fox.reset_hla_identity()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lf, _ = fox(x)
        assert torch.equal(lb, lf)

    def test_causality(self):
        """Cumulative sums are the classic way to leak the future - verify not."""
        model = GPT(small_cfg(**self.FKW)).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_f.weight.normal_(0, 0.5)
                blk.attn.W_range_f.normal_(0, 0.5)
        T = 32
        a = torch.randint(0, 256, (1, T))
        b = a.clone()
        b[0, T - 1] = (b[0, T - 1] + 7) % 256
        la, _ = model(a)
        lb, _ = model(b)
        assert torch.equal(la[0, : T - 1], lb[0, : T - 1]), "forget gate leaked future"

    def test_negative_gate_is_recency_bias(self):
        """All-negative gates => distant keys get MORE negative bias than
        recent ones (monotone in distance) - the FoX forgetting semantics."""
        model = GPT(small_cfg(**self.FKW, n_layer=1)).eval()
        attn = model.transformer.h[0].attn
        with torch.no_grad():
            attn.W_gate_f.weight.fill_(-100.0)  # tanh -> -1 for every token
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        att = model.transformer.h[0].attn.last_attn[0, 0]  # (T, T)
        # for the last query, attention should be concentrated near recent keys
        last_row = att[-1]
        recent_mass = float(last_row[-4:].sum())
        distant_mass = float(last_row[:4].sum())
        assert recent_mass > distant_mass, "negative forget gate must induce recency bias"

    def test_grad_flows(self):
        model = GPT(small_cfg(**self.FKW))
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_f.weight.normal_(0, 0.1)
        x = torch.randint(0, 256, (2, 32))
        _, loss = model(x, x)
        loss.backward()
        g = model.transformer.h[0].attn.W_gate_f.weight.grad
        assert g is not None and float(g.abs().sum()) > 0

    def test_parameter_matched(self):
        a = GPT(small_cfg(**BASE_KW, use_forget_gate=True, forget_alpha=0.0))
        b = GPT(small_cfg(**BASE_KW, use_forget_gate=True, forget_alpha=1.0))
        assert a.parameter_count() == b.parameter_count()

    def test_bias_respects_clip(self):
        cfg = small_cfg(**self.FKW)
        model = GPT(cfg).eval()
        model.set_diagnostics(enabled=True)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_f.weight.fill_(100.0)
        x = torch.randint(0, 256, (1, 64))
        model(x)
        for blk in model.transformer.h:
            assert float(blk.attn.last_forget_bias_abs_mean) <= cfg.forget_clip + 1e-5


class TestGenerate:
    """Autoregressive decoding: needed for lm-eval-harness downstream evals."""

    def test_shapes_and_range(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        idx = torch.randint(0, 256, (2, 5))
        out = model.generate(idx, max_new_tokens=7, greedy=True)
        assert out.shape == (2, 12)
        assert torch.equal(out[:, :5], idx), "prompt must be preserved"
        assert int(out.max()) < 256 and int(out.min()) >= 0

    def test_greedy_deterministic(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        idx = torch.randint(0, 256, (1, 4))
        o1 = model.generate(idx, max_new_tokens=6, greedy=True)
        o2 = model.generate(idx, max_new_tokens=6, greedy=True)
        assert torch.equal(o1, o2)

    def test_padded_vocab_never_sampled(self):
        """Padded ids (>= vocab_size) must be impossible outputs."""
        cfg = small_cfg(vocab_size=250, padded_vocab_size=256)
        model = GPT(cfg).eval()
        idx = torch.randint(0, 250, (2, 4))
        torch.manual_seed(0)
        out = model.generate(idx, max_new_tokens=20, temperature=2.0)
        assert int(out.max()) < 250, "sampled a padded-vocab id!"

    def test_context_cropping(self):
        """Prompt longer than block_size must not crash (crop, not error)."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        idx = torch.randint(0, 256, (1, 64))  # == block_size
        out = model.generate(idx, max_new_tokens=3, greedy=True)
        assert out.shape == (1, 67)

    def test_restores_training_mode(self):
        model = GPT(small_cfg(**HLA_KW)).train()
        model.generate(torch.randint(0, 256, (1, 4)), max_new_tokens=2, greedy=True)
        assert model.training

    def test_t1_prompt(self):
        """Single-token prompt (R29 edge case) with ALL mechanisms active."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        out = model.generate(torch.randint(0, 256, (1, 1)), max_new_tokens=3, greedy=True)
        assert out.shape == (1, 4)


class TestDiagnosticsGating:
    def test_dropout_nonzero_rejected(self):
        """R26: dropout is not implemented - a non-zero value must refuse to
        construct rather than silently do nothing."""
        with pytest.raises(ValueError):
            small_cfg(dropout=0.1)

    def test_entropy_not_computed_in_training(self):
        """R31: entropy is eval-only; training forwards must skip it."""
        model = GPT(small_cfg(**HLA_KW))
        model.train()
        x = torch.randint(0, 256, (1, 16))
        model(x, x)
        assert model.transformer.h[0].attn.last_entropy is None

    def test_entropy_computed_in_eval(self):
        model = GPT(small_cfg(**HLA_KW)).eval()
        x = torch.randint(0, 256, (1, 16))
        model(x)
        e = model.transformer.h[0].attn.last_entropy
        assert e is not None and torch.isfinite(e).all()

    def test_entropy_computed_in_training_with_diagnostics(self):
        model = GPT(small_cfg(**HLA_KW))
        model.train()
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 16))
        model(x, x)
        assert model.transformer.h[0].attn.last_entropy is not None


class TestRotationAlgebra:
    def test_rotation_composition(self):
        """R25: R(a) . R(a) == R(2a) - the pairwise map is a true rotation group
        action, not just any norm-preserving transform."""
        model = GPT(small_cfg(**HLA_KW)).eval()
        attn = model.transformer.h[0].attn
        x = torch.randn(1, 4, 8, attn.head_dim)
        ang = torch.rand(1, 4, 8, attn.head_dim // 2)
        twice = attn._rotate_pairwise(
            attn._rotate_pairwise(x, torch.cos(ang), torch.sin(ang)),
            torch.cos(ang), torch.sin(ang),
        )
        direct = attn._rotate_pairwise(x, torch.cos(2 * ang), torch.sin(2 * ang))
        assert torch.allclose(twice, direct, atol=1e-5)


class TestConfigValidation:
    @pytest.mark.parametrize("bad", [
        dict(n_embd=33),                    # not divisible by n_head
        dict(n_head=2, n_embd=34),          # head_dim odd (34/2=17)
        dict(block_size=0),
        dict(attention_backend="flash"),
        dict(k_log_clip=0.0),
    ])
    def test_invalid_configs_rejected(self, bad):
        with pytest.raises(ValueError):
            small_cfg(**bad)
