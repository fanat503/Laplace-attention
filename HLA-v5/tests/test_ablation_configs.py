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
