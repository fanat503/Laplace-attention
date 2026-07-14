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
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
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
        for rel in ("README.md", "docs/EXPERIMENT_CARD.md"):
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


class TestExternalReviewRound2:
    """Regression tests for the second external review (B1/B4/B6/B7)."""

    def test_dual_positional_encoding_rejected(self):
        """B4: use_wpe=True + use_rope=True must raise, not silently dual-encode."""
        from src.model import GPTConfig
        with pytest.raises(ValueError, match="exactly one positional scheme"):
            GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                      n_embd=32, use_wpe=True, use_rope=True)

    def test_trainer_importable_and_helpful_without_xla(self):
        """B1: on a CPU-only host the trainer must import cleanly and give a
        clear SystemExit from main(), not ModuleNotFoundError at line 1."""
        import subprocess
        code = (
            "import sys; sys.modules['torch_xla'] = None\n"  # simulate absence even if installed
            "import importlib.util, os\n"
            "spec = importlib.util.spec_from_file_location('train_xla', os.path.join(%r, 'src', 'train_xla.py'))\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "assert m.xm is None or m.xm is not None  # import survived\n"
            "print('IMPORT_OK')\n"
        ) % ROOT
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert "IMPORT_OK" in r.stdout, f"trainer import must survive missing torch_xla:\n{r.stderr[-500:]}"

    def test_forget_arm_rejects_global_bf16_env(self, monkeypatch):
        """B7: forget_alpha != 0 + XLA_USE_BF16=1 must fail fast (bf16 cumsum
        silently swallows forgetting steps near |S|~200)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_xla_b7", os.path.join(ROOT, "src", "train_xla.py"))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pytest.skip("trainer refused to import in this env")
        cfg = {"seed": 1, "save_dir": "/tmp/x", "train_path": "a", "val_path": "b",
               "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
               "grad_accum": 1, "max_steps": 1, "lr": 1e-4, "min_lr": 1e-5,
               "warmup": 0, "model": {"forget_alpha": 1.0}}
        monkeypatch.setenv("XLA_USE_BF16", "1")
        with pytest.raises(ValueError, match="fp32 cumsum"):
            mod.validate_config(cfg)
        monkeypatch.delenv("XLA_USE_BF16")
        monkeypatch.setenv("XLA_DOWNCAST_BF16", "1")
        with pytest.raises(ValueError, match="fp32 cumsum"):
            mod.validate_config(cfg)

    def test_experiment_card_declares_mechanism_sets(self):
        """B6: EXPERIMENT_CARD must state which mechanisms each shipped config
        actually activates (capacity != default arm)."""
        card = open(os.path.join(ROOT, "docs", "EXPERIMENT_CARD.md"), encoding="utf-8").read()
        assert "Active mechanism sets per config" in card
        assert "capacity, not the" in card.replace("\n", " ")

    def test_theory_has_rope_phase_commutation(self):
        """A4 (Nanda): the before/after-RoPE ordering question is settled by
        Corollary 7.1, and the numeric claim in it must be true."""
        theory = open(os.path.join(ROOT, "docs", "THEORY.md"), encoding="utf-8").read()
        assert "Corollary 7.1" in theory
        import math, torch
        from src.model import GPT, GPTConfig
        torch.manual_seed(0)
        cfg = GPTConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=32,
                        gradient_checkpointing=False, use_rope=True, use_wpe=False,
                        phase_mult=0.3)
        a = GPT(cfg).transformer.h[0].attn
        B, H, T, hs = 2, 2, 16, 16
        q = torch.randn(B, H, T, hs)
        pos = torch.arange(T, dtype=torch.float32)
        rope_ang = torch.einsum("t,k->tk", pos, a.rope_inv_freq).view(1, 1, T, hs // 2)
        ph_ang = 0.3 * math.pi * torch.randn(B, H, T, hs // 2)

        def rot(x, ang):
            return a._rotate_pairwise(x, torch.cos(ang), torch.sin(ang))

        q_rope_then_phase = rot(rot(q, rope_ang), ph_ang)
        q_phase_then_rope = rot(rot(q, ph_ang), rope_ang)
        q_sum = rot(q, rope_ang + ph_ang)
        assert (q_rope_then_phase - q_phase_then_rope).abs().max() < 1e-5
        assert (q_rope_then_phase - q_sum).abs().max() < 1e-5


class TestFinalHardening:
    """Final 'fix it so it stays fixed' round: every analysis entrypoint must
    work on a CPU-only, memory-constrained host (the post-hoc analysis
    environment), and artifact dirs must be untrackable."""

    def test_prepare_c4_help_works_without_data_deps(self):
        """--help must not require tiktoken/datasets/tqdm (same class as B1)."""
        import subprocess
        code = ("import sys\n"
                "for m in ('tiktoken','datasets','tqdm'):\n"
                "    sys.modules[m] = None\n"  # simulate absence
                "import runpy, sys as s\n"
                "s.argv = ['prepare_c4_data.py', '--help']\n"
                "try:\n"
                "    runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n"
                "    raise SystemExit(0 if e.code in (0, None) else 1)\n"
                % os.path.join(ROOT, "scripts", "prepare_c4_data.py"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, f"--help must survive missing data deps:\n{r.stderr[-400:]}"

    def test_analysis_scripts_default_device_is_auto(self):
        """post-hoc analysis runs on CPU hosts: default --device xla crashed
        with ModuleNotFoundError before this round."""
        for rel in ("scripts/analyze_checkpoint.py", "scripts/compare_attention_kl.py"):
            src = open(os.path.join(ROOT, rel)).read()
            assert 'default="auto"' in src, f"{rel}: --device default must be auto"
            assert 'if name == "auto"' in src, f"{rel}: resolve_device must handle auto"

    def test_compare_kl_clamps_seq_len(self):
        src = open(os.path.join(ROOT, "scripts/compare_attention_kl.py")).read()
        assert 'min(int(seq_len), int(cfg["model"]["block_size"]))' in src, (
            "unclamped --seq-len crashed on block_size+1 sequences")

    def test_divergence_micro_batches(self):
        """plot_divergence held two (B,T,vocab) fp32 logit tensors -> OOM 137
        on small hosts; must accumulate per-row."""
        src = open(os.path.join(ROOT, "scripts/plot_divergence.py")).read()
        assert "for i in range(n_rows)" in src
        assert "kl_sum" in src and "nb2" in src, "cosine must be accumulated, not materialized"

    def test_induction_by_distance_micro_chunks(self):
        src = open(os.path.join(ROOT, "scripts/analyze_checkpoint.py")).read()
        assert "for s in range(0, batch_size, chunk)" in src, (
            "batch=16 forward materializes (16,T,vocab) logits -> OOM")

    def test_analyze_checkpoint_has_knockout_curve(self):
        """The long-context evidence figure must be produced by the standard
        checkpoint analysis, not require custom code at paper time."""
        src = open(os.path.join(ROOT, "scripts/analyze_checkpoint.py")).read()
        assert "knockout_by_context_length" in src
        assert "prefix_matching" in src

    def test_make_plots_has_mechanism_dashboard(self):
        """Every diagnostic CSV column family must be plottable out of the box."""
        src = open(os.path.join(ROOT, "scripts/make_plots.py")).read()
        assert "mechanism_dashboard" in src
        for col in ("qk_interference", "distractor_margin", "mech_grad_mean",
                    "qtemp_sat_frac", "layer_temp_last", "svd_phase_erank"):
            assert col in src, f"dashboard must cover {col}"

    def test_gitignore_blocks_experiment_artifacts(self):
        """Token files are ~21 GB; a stray `git add -A` after data prep must
        not be able to commit them."""
        gi = open(os.path.join(ROOT, ".gitignore")).read()
        for pat in ("data/", "runs/", "inits/", "*.pt", "*.bin"):
            assert f"\n{pat}\n" in gi or gi.endswith(f"\n{pat}") or f"\n{pat}\n" in gi + "\n", (
                f".gitignore must contain {pat}")

    def test_citation_version_matches_pyproject(self):
        import re as _re
        cff = open(os.path.join(ROOT, "CITATION.cff")).read()
        pyp = open(os.path.join(ROOT, "pyproject.toml")).read()
        cff_v = _re.search(r'^version: "([\d.]+)"', cff, _re.M).group(1)
        pyp_v = _re.search(r'^version = "([\d.]+)"', pyp, _re.M).group(1)
        assert pyp_v.startswith(cff_v), (
            f"CITATION.cff version {cff_v} vs pyproject {pyp_v} - keep in sync")


class TestRound5Hardening:
    """Fifth sweep: config typo guard, CRLF hygiene, CI e2e helpers."""

    def _load_trainer(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_xla_r5", os.path.join(ROOT, "src", "train_xla.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_unknown_top_level_key_rejected(self):
        """A typo like warmup_steps (vs warmup) silently fell back to defaults
        and trained with wrong hyperparameters. Must raise, checked BEFORE
        file-existence so it fires on any host."""
        mod = self._load_trainer()
        cfg = {"seed": 1, "save_dir": "s", "train_path": "t", "val_path": "v",
               "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
               "grad_accum": 1, "max_steps": 1, "lr": 1e-4, "min_lr": 1e-5,
               "warmup": 0, "model": {}, "warmup_steps": 500}
        with pytest.raises(KeyError, match="Unknown top-level config keys"):
            mod.validate_config(cfg)

    def test_unknown_key_escape_hatch(self):
        mod = self._load_trainer()
        assert "allow_unknown_config_keys" in mod.KNOWN_TOP_LEVEL_KEYS

    def test_every_known_key_is_actually_used(self):
        """The allow-list must not drift: every KNOWN key (minus meta) must
        appear in the trainer source, else it's a stale entry."""
        mod = self._load_trainer()
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        meta = {"allow_unknown_config_keys", "_doc", "variant"}
        for key in mod.KNOWN_TOP_LEVEL_KEYS - meta:
            assert f'"{key}"' in src, f"stale KNOWN_TOP_LEVEL_KEYS entry: {key}"

    def test_shipped_configs_have_no_unknown_keys(self):
        """Every shipped config must pass the new guard (keys subset check
        only - paths may not exist on CI)."""
        import glob as _glob
        mod = self._load_trainer()
        for path in sorted(_glob.glob(os.path.join(ROOT, "configs", "*.json"))):
            cfg = json.load(open(path))
            unknown = set(cfg) - mod.KNOWN_TOP_LEVEL_KEYS
            assert not unknown, f"{os.path.basename(path)}: unknown keys {unknown}"

    def test_no_crlf_in_tracked_text_files(self):
        """CRLF snuck in via web uploads (prepare_data.py, utils.py) - breaks
        patches and produces noisy diffs. .gitattributes now enforces LF; this
        test keeps the tree itself clean."""
        import glob as _glob
        bad = []
        for pattern in ("src/*.py", "scripts/*.py", "tests/*.py", "docs/*.md",
                        "*.md", "*.toml", "*.txt", ".github/workflows/*.yml"):
            for f in _glob.glob(os.path.join(ROOT, pattern)):
                if b"\r\n" in open(f, "rb").read():
                    bad.append(os.path.relpath(f, ROOT))
        assert not bad, f"CRLF line endings in: {bad}"

    def test_gitattributes_enforces_lf(self):
        ga = os.path.join(ROOT, ".gitattributes")
        assert os.path.exists(ga), ".gitattributes must exist (LF enforcement)"
        content = open(ga).read()
        assert "eol=lf" in content

    def test_ci_check_analysis_script(self, tmp_path):
        """CI e2e gate helper: accepts a good analysis JSON, rejects broken."""
        import subprocess
        good = {"val_loss": 10.9, "induction_by_distance": {}, "prefix_matching": {},
                "attention_entropy": [], "knockout_by_context_length": {"64": {"loss_full": 10.9}}}
        p = tmp_path / "good.json"
        p.write_text(json.dumps(good))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ci_check_analysis.py"), str(p)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        bad = dict(good); del bad["knockout_by_context_length"]
        p2 = tmp_path / "bad.json"
        p2.write_text(json.dumps(bad))
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ci_check_analysis.py"), str(p2)],
                            capture_output=True, text=True)
        assert r2.returncode != 0

    def test_ci_workflow_has_cli_and_e2e_smoke(self):
        wf = open(os.path.join(ROOT, ".github", "workflows", "tests.yml")).read()
        assert "CLI smoke" in wf, "CI must gate every script's --help"
        assert "E2E pipeline smoke" in wf, "CI must run the dummy-data pipeline"
        assert "ci_check_analysis.py" in wf
