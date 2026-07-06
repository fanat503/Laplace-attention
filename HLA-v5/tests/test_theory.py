"""Numerical verification of every theorem in docs/THEORY.md.

Each test cites its theorem. If a refactor breaks a mathematical guarantee,
this file is where CI catches it.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


def cfg(**kw):
    base = dict(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
                gradient_checkpointing=False)
    base.update(kw)
    return GPTConfig(**base)


ALL_ON = dict(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
              use_salience_bias=True, salience_alpha=1.0,
              use_distance_laplace=True, distance_laplace_alpha=0.5,
              use_forget_gate=True, forget_alpha=1.0,
              use_rope=True, use_wpe=False)
ALL_OFF = dict(phase_mult=0.0, use_laplace=True, laplace_alpha=0.0,
               use_salience_bias=True, salience_alpha=0.0,
               use_distance_laplace=True, distance_laplace_alpha=0.0,
               use_forget_gate=True, forget_alpha=0.0,
               use_rope=True, use_wpe=False)


class TestTheorem1Identity:
    """HLA(x; backbone, 0) == GPT(x; backbone), exactly."""

    def test_bit_exact_all_mechanisms(self):
        torch.manual_seed(0)
        base = GPT(cfg(**ALL_OFF)).eval()
        hla = GPT(cfg(**ALL_ON)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        for _ in range(3):
            x = torch.randint(0, 256, (2, 32))
            lb, _ = base(x)
            lh, _ = hla(x)
            assert torch.equal(lb, lh)


class TestTheorem2StrictInclusion:
    """F_base is a strict subset of F_HLA: the salience witness."""

    def test_base_with_zero_wq_is_uniform(self):
        model = GPT(cfg(use_salience_bias=True, salience_alpha=1.0)).eval()
        attn = model.transformer.h[0].attn
        with torch.no_grad():
            attn.c_attn.weight[: attn.n_embd, :].zero_()  # W_Q = 0
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        row = model.transformer.h[0].attn.last_attn[0, 0][10, :11]
        assert float(row.max() - row.min()) < 1e-6, "W_Q=0 must give uniform attention in base"

    def test_salience_escapes_base_class(self):
        model = GPT(cfg(use_salience_bias=True, salience_alpha=1.0, salience_range=2.0)).eval()
        attn = model.transformer.h[0].attn
        with torch.no_grad():
            attn.c_attn.weight[: attn.n_embd, :].zero_()
            attn.W_gate_sal.weight.normal_(0, 1.0)
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        row = model.transformer.h[0].attn.last_attn[0, 0][10, :11]
        assert float(row.max() - row.min()) > 0.01, "salience must produce key-dependent attention at W_Q=0"


class TestTheorem3Isometry:
    def test_norm_preservation_random_angles(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        for _ in range(3):
            z = torch.randn(2, 4, 8, attn.head_dim)
            th = torch.randn(2, 4, 8, attn.head_dim // 2) * 3.0
            r = attn._rotate_pairwise(z, torch.cos(th), torch.sin(th))
            assert torch.allclose(z.norm(dim=-1), r.norm(dim=-1), atol=1e-5)

    def test_group_action(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        z = torch.randn(1, 2, 4, attn.head_dim)
        a = torch.randn(1, 2, 4, attn.head_dim // 2)
        b = torch.randn(1, 2, 4, attn.head_dim // 2)
        ab = attn._rotate_pairwise(attn._rotate_pairwise(z, torch.cos(a), torch.sin(a)),
                                   torch.cos(b), torch.sin(b))
        direct = attn._rotate_pairwise(z, torch.cos(a + b), torch.sin(a + b))
        assert torch.allclose(ab, direct, atol=1e-5)


class TestTheorem4Envelopes:
    """Score perturbation bounded regardless of weights/inputs."""

    def test_total_bias_bounded_at_saturation(self):
        c = cfg(**ALL_ON)
        model = GPT(c).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_sal.weight.fill_(1000.0)
                blk.attn.W_gate_f.weight.fill_(1000.0)
                blk.attn.W_gate_k.weight.fill_(1000.0)
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 64))
        logits, _ = model(x)
        assert torch.isfinite(logits).all()
        lm = 1.0  # layer_dependent_gate off in this cfg
        bound = c.salience_clip + c.distance_laplace_clip * lm + c.forget_clip
        for blk in model.transformer.h:
            total = (float(blk.attn.last_salience_bias_abs_mean)
                     + float(blk.attn.last_distance_bias_abs_mean)
                     + float(blk.attn.last_forget_bias_abs_mean))
            assert total <= bound + 1e-4

    def test_mix_envelope_exact_endpoints(self):
        c = cfg(**ALL_ON)
        model = GPT(c).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)   # tanh -> sign(sum x): +-1
                blk.attn.W_range_k.fill_(1000.0)          # range at +25%
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 32))
        model(x)
        eff = min(c.laplace_alpha * c.laplace_range_k * 1.25, c.k_log_clip)
        expected_hi = (1 - c.beta_k) + c.beta_k * math.exp(eff)
        expected_lo = (1 - c.beta_k) + c.beta_k * math.exp(-eff)
        # gate = tanh(1000 * sum(x)) = exactly +1 or -1 per position (sign of
        # the input projection), so the mix tensor must sit exactly at the two
        # envelope endpoints - and never beyond.
        mix = model.transformer.h[0].attn.last_mix_k.float()
        assert abs(float(mix.max()) - expected_hi) < 1e-3
        assert abs(float(mix.min()) - expected_lo) < 1e-3


class TestTheorem5FirstOrderSignal:
    def test_mechanism_gradients_nonzero_at_identity(self):
        torch.manual_seed(3)
        model = GPT(cfg(**ALL_ON))
        model.train()
        x = torch.randint(0, 256, (4, 32))
        _, loss = model(x, x)
        loss.backward()
        mech_sq = 0.0
        for n, p in model.named_parameters():
            if p.grad is not None and any(m in n for m in ("W_phase", "W_gate", "W_range")):
                mech_sq += float(p.grad.pow(2).sum())
        assert mech_sq > 0.0, "identity point must not be a saddle for the mechanisms"

    def test_tanh_derivative_at_zero_is_one(self):
        """The reason Theorem 5 works: gradient passes tanh(0) undamped."""
        z = torch.zeros(5, requires_grad=True)
        torch.tanh(z).sum().backward()
        assert torch.allclose(z.grad, torch.ones(5))
