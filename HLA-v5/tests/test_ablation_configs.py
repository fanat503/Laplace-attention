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



"""Tests for the ablation-matrix generator: single-factor discipline."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_ablation_configs import ARMS, ZERO_KEYS, make_arm  # noqa: E402


@pytest.fixture()
def templates():
    with open(os.path.join(ROOT, "configs/200m_base_v2_s42.json")) as f:
        base = json.load(f)
    with open(os.path.join(ROOT, "configs/200m_hla_v2_s42.json")) as f:
        hla = json.load(f)
    return base, hla


class TestSingleFactor:
    def test_base_arm_all_zero(self, templates):
        base, hla = templates
        _, cfg = make_arm(base, hla, "base", 42, "/tmp/x")
        for k in ZERO_KEYS:
            assert cfg["model"][k] == 0.0, f"{k} must be off in base arm"

    @pytest.mark.parametrize("arm,active_key", [
        ("phase", "phase_mult"),
        ("gates", "laplace_alpha"),
        ("salience", "salience_alpha"),
        ("distance", "distance_laplace_alpha"),
        ("forget", "forget_alpha"),
    ])
    def test_single_mechanism_arms(self, templates, arm, active_key):
        """Each single arm must activate EXACTLY one alpha."""
        base, hla = templates
        _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
        m = cfg["model"]
        for k in ZERO_KEYS:
            if k == active_key:
                assert m[k] != 0.0, f"{arm}: {k} should be active"
            else:
                assert m[k] == 0.0, f"{arm}: {k} leaked into single-factor arm"

    def test_full_matches_template(self, templates):
        base, hla = templates
        _, cfg = make_arm(base, hla, "full", 42, "/tmp/x")
        for k in ZERO_KEYS:
            if k in hla["model"]:
                assert cfg["model"][k] == hla["model"][k]

    def test_shared_init_per_seed(self, templates):
        """All arms of one seed must point to the SAME init checkpoint."""
        base, hla = templates
        inits = set()
        for arm in ARMS:
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            inits.add(cfg["init_ckpt"])
        assert len(inits) == 1, "arms must share one sterile init per seed"

    def test_seeds_get_distinct_names(self, templates):
        base, hla = templates
        n42, _ = make_arm(base, hla, "phase", 42, "/tmp/x")
        n43, _ = make_arm(base, hla, "phase", 43, "/tmp/x")
        assert n42 != n43

    def test_structural_flags_uniform(self, templates):
        """use_* switches identical in every arm => parameter matching holds
        across the whole matrix."""
        base, hla = templates
        flags = ("use_laplace", "use_distance_laplace", "use_salience_bias", "use_forget_gate")
        reference = None
        for arm in ARMS:
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            vals = tuple(cfg["model"][f] for f in flags)
            if reference is None:
                reference = vals
            assert vals == reference, f"{arm}: structural flags differ"


class TestCLI:
    def test_end_to_end_generation(self, tmp_path, templates):
        out = str(tmp_path / "abl")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/make_ablation_configs.py"),
             "--base", os.path.join(ROOT, "configs/200m_base_v2_s42.json"),
             "--hla", os.path.join(ROOT, "configs/200m_hla_v2_s42.json"),
             "--outdir", out, "--seeds", "42", "43"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        manifest = json.load(open(os.path.join(out, "MANIFEST.json")))
        assert len(manifest["configs"]) == 2 * len(ARMS)
        # every emitted config must construct a valid GPTConfig
        from src.model import GPTConfig
        for p in manifest["configs"]:
            cfg = json.load(open(p))
            GPTConfig(**cfg["model"])


class TestLeaveOneOut:
    @pytest.mark.parametrize("arm,dropped", [
        ("no_phase", "phase_mult"),
        ("no_gates", "laplace_alpha"),
        ("no_salience", "salience_alpha"),
        ("no_distance", "distance_laplace_alpha"),
    ])
    def test_drops_exactly_one(self, templates, arm, dropped):
        base, hla = templates
        _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
        m = cfg["model"]
        assert m[dropped] == 0.0, f"{arm}: {dropped} must be OFF"
        for k in ZERO_KEYS:
            if k == dropped or k == "forget_alpha":
                continue  # forget not in HLA template -> stays 0
            if k in hla["model"] and hla["model"][k] != 0.0:
                assert m[k] == hla["model"][k], f"{arm}: {k} must keep template value"


class TestArmConfigValidation:
    """Review attack P2: every generated arm pair must pass validate_configs
    (forget_alpha was missing from the allow-list before this test existed)."""

    def test_all_arms_pass_pair_validator(self, templates, tmp_path):
        import subprocess
        base, hla = templates
        base_name, base_cfg = make_arm(base, hla, "base", 42, str(tmp_path))
        base_path = tmp_path / f"{base_name}.json"
        with open(base_path, "w") as f:
            json.dump(base_cfg, f)
        for arm in ARMS:
            if arm == "base":
                continue
            name, cfg = make_arm(base, hla, arm, 42, str(tmp_path))
            p = tmp_path / f"{name}.json"
            with open(p, "w") as f:
                json.dump(cfg, f)
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts/validate_configs.py"),
                 "--base", str(base_path), "--hla", str(p)],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, f"arm '{arm}' rejected by validator:\n{r.stdout}{r.stderr}"


class TestSharedInitCompatibility:
    """Review attack N6/N7: one init per seed must strict-load into EVERY arm."""

    def test_base_arm_init_loads_into_all_arms(self, templates):
        import torch
        from src.model import GPT, GPTConfig
        base, hla = templates
        small = dict(n_layer=2, n_head=2, n_embd=32, block_size=64,
                     vocab_size=256, gradient_checkpointing=False)

        def small_model(arm):
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            mc = dict(cfg["model"]); mc.update(small)
            return GPT(GPTConfig(**mc))

        donor = small_model("base").state_dict()
        for arm in ARMS:
            m = small_model(arm)
            m.load_state_dict(donor, strict=True)  # raises on mismatch

    def test_manifest_contains_init_commands(self, templates, tmp_path):
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/make_ablation_configs.py"),
             "--base", os.path.join(ROOT, "configs/200m_base_v2_s42.json"),
             "--hla", os.path.join(ROOT, "configs/200m_hla_v2_s42.json"),
             "--outdir", str(tmp_path), "--seeds", "42", "43"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        man = json.load(open(tmp_path / "MANIFEST.json"))
        assert "init_commands" in man
        assert "42" in man["init_commands"] and "43" in man["init_commands"]
        assert "make_init" in man["init_commands"]["42"]


class TestDocsConsistency:
    """Review attack K1 (Karpathy-style 'the README must not lie'):
    documentation numbers are cross-checked against reality by CI."""

    def test_readme_test_count_matches_collected(self):
        import re, subprocess
        readme = open(os.path.join(ROOT, "..", "README.md"), encoding="utf-8").read()
        # Only TOTAL-count claims (badge, layout total, quick-start echo,
        # section header) - NOT per-file mentions like "test_theory.py (9 tests)".
        claimed = set(int(m) for m in re.findall(r"tests-(\d+)%20passing", readme))
        claimed |= set(int(m) for m in re.findall(r"·\s*(\d+) tests", readme))
        claimed |= set(int(m) for m in re.findall(r"(\d+) CPU tests", readme))
        claimed |= set(int(m) for m in re.findall(r"→ (\d+) passed", readme))
        # \D{0,4} (non-digits!) so backtracking can't split "188" into 18+8
        claimed |= set(int(m) for m in re.findall(r"## \D{0,4}(\d+) tests", readme))
        r = subprocess.run(
            [sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"),
             "--collect-only", "-q", "-p", "no:cacheprovider"],
            capture_output=True, text=True)
        # pytest -q --collect-only prints per-file lines "path: N"
        per_file = re.findall(r": (\d+)\s*$", r.stdout, re.M)
        actual = sum(int(n) for n in per_file) if per_file else -1
        assert actual > 0, f"collection failed: {r.stdout[-300:]}"
        wrong = {c for c in claimed if c != actual}
        assert not wrong, (
            f"README claims test counts {sorted(wrong)} but actual is {actual}. "
            f"Update README badges/sections."
        )
