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
        flags = ("use_laplace", "use_distance_laplace", "use_salience_bias", "use_forget_gate", "use_qtemp")
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


class TestToolingIntegrity:
    """Final review sweep: helper scripts must work on the SHIPPED files."""

    def test_check_environment_parses_range_specs(self):
        """F1: parse_pins must understand range specs, not only pkg==x.y -
        previously requirements.txt parsed to {} and the check passed vacuously."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        ce = importlib.import_module("check_environment")
        specs = ce.parse_pins(os.path.join(ROOT, "requirements.txt"))
        assert "torch" in specs and specs["torch"], "torch range spec must be parsed"
        assert ce.spec_satisfied("2.5.0", [(">=", "2.4"), ("<", "2.15")])
        assert not ce.spec_satisfied("2.15.1", [(">=", "2.4"), ("<", "2.15")])

    def test_check_environment_rejects_empty_specfile(self, tmp_path):
        import subprocess
        p = tmp_path / "empty.txt"
        p.write_text("# only comments\n")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/check_environment.py"),
             "--requirements", str(p)], capture_output=True, text=True)
        assert r.returncode != 0, "empty spec file must fail, not pass vacuously"

    def test_preflight_references_existing_files(self):
        """F2: every path preflight.sh invokes must exist in the repo."""
        sh = open(os.path.join(ROOT, "scripts/preflight.sh")).read()
        assert "requirements_tpu.txt" not in sh, "preflight references a non-existent file"
        assert "python src/make_init.py" in sh, "make_init lives in src/"

    def test_configs_readme_matches_validator(self):
        """F3: configs/README allow-list must be a superset-consistent view of
        the actual validator (script is the source of truth)."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        vc = importlib.import_module("validate_configs")
        readme = open(os.path.join(ROOT, "configs/README.md")).read()
        for key in sorted(vc.ALLOWED_MAIN_DIFFS):
            assert f"`{key}`" in readme, f"configs/README.md missing allowed key {key}"

    def test_profile_flops_analytic_no_model_instantiation(self):
        """OOM fix: profiling must not allocate model weights."""
        src = open(os.path.join(ROOT, "scripts/profile_flops.py")).read()
        assert "no weights allocated" in src
        assert "hidden = int(8 * C / 3)" in src, "MLP hidden must not be scaled by L (old bug)"


class TestFinalPolishSweep:
    """Regression tests for the final polish round: every doc/tool must track
    the range_flex knob and the qtemp/W_gate_d mechanisms (no silent drift)."""

    def test_audit_uses_range_flex_not_hardcoded(self):
        """audit_config_values must derive the envelope from the config's
        range_flex, not the historical hardcoded 1.25."""
        src = open(os.path.join(ROOT, "scripts/audit_config_values.py")).read()
        assert 'm.get("range_flex", 0.25)' in src, "audit must read range_flex from config"
        assert "alpha * rng * 1.25" not in src, "hardcoded 1.25 must be gone from envelope math"
        assert "alpha * rng_k * 1.25" not in src, "hardcoded 1.25 must be gone from E8 too"

    def test_perturbation_bounds_uses_model_range_flex(self):
        src = open(os.path.join(ROOT, "src/eval.py")).read()
        assert 'getattr(attn, "range_flex", 0.25)' in src, (
            "perturbation_bounds must read range_flex from the model")

    def test_profile_flops_counts_qtemp_and_distance_gate(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        pf = importlib.import_module("profile_flops")
        from src.model import GPTConfig
        common = dict(block_size=128, vocab_size=512, n_layer=2, n_head=2,
                      n_embd=64, gradient_checkpointing=False)
        base = type("M", (), {"config": GPTConfig(**common)})()
        qt = type("M", (), {"config": GPTConfig(use_qtemp=True, qtemp_alpha=1.0, **common)})()
        fb = pf.count_forward_flops(base, batch_size=1, seq_len=64)
        fq = pf.count_forward_flops(qt, batch_size=1, seq_len=64)
        assert fb["qtemp"] == 0 and fq["qtemp"] > 0, "qtemp FLOPs must be counted when active"
        assert fq["total"] > fb["total"]
        # distance must include its own W_gate_d projection (C->H), not just T*T
        dm = type("M", (), {"config": GPTConfig(use_distance_laplace=True,
                                                distance_laplace_alpha=0.5, **common)})()
        fd = pf.count_forward_flops(dm, batch_size=1, seq_len=64)
        B, T, C, H, L = 1, 64, 64, 2, 2
        assert fd["distance_bias"] >= L * (2 * B * T * C * H + 2 * B * H * T * T), (
            "distance FLOPs must include the dedicated W_gate_d projection")

    def test_trainer_csv_logs_qtemp_columns(self):
        """qtemp diagnostics existed in hla_statistics but never reached the
        CSV - the qtemp ablation arm would have produced no mechanism curve."""
        src = open(os.path.join(ROOT, "src/train_xla.py")).read()
        assert '"qtemp_mean"' in src, "CSV header must include qtemp_mean"
        assert '"qtemp_sat_frac"' in src, "CSV header must include qtemp_sat_frac"
        assert 'metrics.get("qtemp_mean"' in src, "CSV row must write qtemp_mean"
        assert 'metrics.get("qtemp_sat_frac"' in src, "CSV row must write qtemp_sat_frac"

    def test_trainer_csv_header_row_same_length(self):
        """Header list and row list in train_xla.py must stay in lockstep."""
        import re as _re
        src = open(os.path.join(ROOT, "src/train_xla.py")).read()
        hdr_m = _re.search(r'writer\.writerow\(\[\s*"step",(.*?)\]\)', src, _re.S)
        assert hdr_m, "CSV header writerow not found"
        n_header = len(_re.findall(r'"[a-z_0-9]+"', hdr_m.group(0)))
        row_m = _re.search(r'writer\.writerow\(\[\s*completed_step,(.*?)\]\)', src, _re.S)
        assert row_m, "CSV data writerow not found"
        body = row_m.group(0)
        n_row = body.count("\n") - 1  # one entry per line in the literal
        assert n_header == n_row, (
            f"CSV header has {n_header} columns but data row writes {n_row}")

    def test_audit_has_qtemp_check(self, templates, tmp_path):
        """E11: the qtemp arm's values must be auditable, not invisible."""
        import subprocess
        base, hla = templates
        _, cfg = make_arm(base, hla, "qtemp", 42, str(tmp_path))
        p = tmp_path / "qtemp_arm.json"
        with open(p, "w") as f:
            json.dump(cfg, f)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/audit_config_values.py"),
             "--config", str(p)], capture_output=True, text=True)
        assert "E11" in r.stdout, "audit must report E11 for an active qtemp config"
        assert "FAIL" not in r.stdout.replace("0 fail", ""), r.stdout

    def test_docs_arm_count_matches_code(self):
        """Anti-drift: every 'N arms' claim in README/EXPERIMENT_CARD must
        equal len(ARMS). (Caught a real 15-vs-14 drift during review.)"""
        import re as _re
        n = len(ARMS)
        for rel in ("../README.md", "docs/EXPERIMENT_CARD.md"):
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
            claims = [int(m) for m in _re.findall(r"(\d+) arms", text)]
            for c in claims:
                assert c == n, f"{rel} claims {c} arms but code has {n}"

    def test_structural_flags_cover_all_use_switches(self):
        """The uniformity test above is only as good as its flag list: it must
        include every use_* mechanism switch GPTConfig knows about."""
        import dataclasses
        from src.model import GPTConfig
        cfg_flags = {f.name for f in dataclasses.fields(GPTConfig)
                     if f.name.startswith("use_") and f.name not in ("use_wpe", "use_rope")}
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "test_ablation_configs.py"), encoding="utf-8").read()
        for flag in cfg_flags:
            assert f'"{flag}"' in src, (
                f"structural-flags test does not check {flag}; add it to the tuple")
