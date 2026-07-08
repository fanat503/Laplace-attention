# Copyright 2026 Ivan Ivanov
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



"""make_init.py tests: sterile shared-backbone init generation."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.make_init import (  # noqa: E402
    assert_hla_identity,
    copy_shared_backbone,
    is_hla_key,
    parameter_report,
    stable_hash,
)
from src.model import GPT, GPTConfig  # noqa: E402


def make_pair():
    base_cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                         n_embd=32, phase_mult=0.0, use_laplace=True,
                         laplace_alpha=0.0, gradient_checkpointing=False)
    hla_cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, phase_mult=0.15, use_laplace=True,
                        laplace_alpha=1.0, gradient_checkpointing=False)
    torch.manual_seed(1)
    base = GPT(base_cfg)
    torch.manual_seed(2)  # deliberately different seed
    hla = GPT(hla_cfg)
    return base, hla


class TestIsHlaKey:
    @pytest.mark.parametrize("key,expected", [
        ("transformer.h.0.attn.W_phase_q", True),
        ("transformer.h.11.attn.W_gate_k.weight", True),
        ("transformer.h.0.attn.c_attn.weight", False),
        ("transformer.wte.weight", False),
        ("lm_head.weight", False),
    ])
    def test_patterns(self, key, expected):
        assert is_hla_key(key) is expected


class TestSharedBackbone:
    def test_backbone_identical_after_copy(self):
        base, hla = make_pair()
        manifest = copy_shared_backbone(base, hla)
        assert manifest["num_copied_non_hla_keys"] > 0
        base_sd, hla_sd = base.state_dict(), hla.state_dict()
        for k in manifest["copied_non_hla_keys"]:
            assert torch.equal(base_sd[k], hla_sd[k]), k

    def test_hla_identity_after_copy(self):
        base, hla = make_pair()
        copy_shared_backbone(base, hla)
        assert hla.hla_identity_error() == 0.0
        assert base.hla_identity_error() == 0.0

    def test_logits_identical_after_shared_init(self):
        base, hla = make_pair()
        copy_shared_backbone(base, hla)
        base.eval(); hla.eval()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lh, _ = hla(x)
        assert torch.equal(lb, lh), "shared-backbone init must give identical outputs"


class TestAssertIdentity:
    def test_passes_on_fresh_model(self):
        base, _ = make_pair()
        assert_hla_identity(base)

    def test_fails_on_perturbed_model(self):
        _, hla = make_pair()
        with torch.no_grad():
            hla.transformer.h[0].attn.W_phase_q.add_(0.01)
        with pytest.raises(RuntimeError):
            assert_hla_identity(hla)


class TestParameterReport:
    def test_groups_sum_to_total(self):
        base, _ = make_pair()
        rep = parameter_report(base)
        parts = rep["embedding"] + rep["attention_base"] + rep["mlp"] + rep["norm"] + rep["hla"] + rep["other"]
        assert parts == rep["total"]
        assert rep["hla"] > 0  # W_phase/W_range/W_gate present even in base (ablated)


class TestStableHash:
    def test_deterministic_and_order_independent(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})
