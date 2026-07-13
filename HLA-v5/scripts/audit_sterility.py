"""Static sterility audit for the HLA-v4 repository.

This script does not replace runtime verification. It checks repository-level
invariants that should hold before a large TPU/TRC run.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "src/model.py",
    "src/data.py",
    "src/eval.py",
    "src/train_xla.py",
    "src/make_init.py",
    "src/manifest.py",
    "src/utils.py",
    "src/__init__.py",
    "scripts/verify_run.py",
    "scripts/analyze_checkpoint.py",
    "scripts/analyze_subspaces.py",
    "scripts/audit_config_values.py",
    "scripts/check_dataloader.py",
    "scripts/check_environment.py",
    "scripts/compare_attention_kl.py",
    "scripts/count_params.py",
    "scripts/create_run_manifest.py",
    "scripts/estimate_budget.py",
    "scripts/compare_inits.py",
    "scripts/inspect_checkpoint.py",
    "scripts/make_ablation_configs.py",
    "scripts/plot_divergence.py",
    "scripts/make_dummy_data.py",
    "scripts/make_plots.py",
    "scripts/prepare_c4_data.py",
    "scripts/validate_log.py",
    "scripts/validate_configs.py",
    "scripts/validate_data_pair.py",
    "configs/200m_base_s42.json",
    "configs/200m_hla_s42.json",
    "configs/200m_base_v2_s42.json",
    "configs/200m_hla_v2_s42.json",
    "configs/700m_base_14b_s42.json",
    "configs/700m_hla_14b_s42.json",
    "configs/800m_base_s42.json",
    "configs/800m_hla_s42.json",
    "tests/test_model.py",
    "tests/test_data.py",
    "tests/test_eval.py",
    "tests/test_make_init.py",
    "tests/test_train_utils.py",
    "tests/test_theory.py",
    "tests/test_ablation_configs.py",
    "requirements.txt",
    "pyproject.toml",
    "docs/METRICS.md",
    "docs/STERILITY.md",
    "docs/THEORY.md",
    "docs/EXPERIMENT_CARD.md",
    "docs/DATA_CARD.md",
]

FORBIDDEN_TRAIN_STRINGS = [
    "from accelerate import",
    # NOTE: bare "torch.cuda" is allowed in seed_everything (harmless,
    # guarded by is_available); forbid actual GPU placement/config instead.
    ".to('cuda')",
    '.to("cuda")',
    'device="cuda"',
    "device='cuda'",
    "PYTORCH_CUDA_ALLOC_CONF",
]

REQUIRED_MODEL_STRINGS = [
    "def reset_hla_identity",
    "def hla_identity_error",
    "self.reset_hla_identity()",
    "self.lm_head.weight = self.transformer.wte.weight",
    'torch.einsum("btd,hdk->bhtk"',
    "attention_backend",
    "layer_dependent_gate",
    "layer_gate_multiplier",
    "use_distance_laplace",
    "distance_laplace_alpha",
]

REQUIRED_TRAIN_STRINGS = [
    "EvenShardedSequentialSampler",
    "validation_loss_xla",
    "validate_resume_config_compatibility",
    "validate_init_config_compatibility",
    "check_init_hla_identity",
    "atomic_xm_save",
    "crash_",
    "run_state_",
]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def check_required_files() -> None:
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        fail(f"missing required files: {missing}")


def check_python_parse() -> None:
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_model() -> None:
    s = read("src/model.py")
    for needle in REQUIRED_MODEL_STRINGS:
        if needle not in s:
            fail(f"src/model.py missing required invariant string: {needle}")
    if "expm1" in s:
        fail("src/model.py contains expm1, but current policy is to preserve provided torch.exp logic")


def check_train() -> None:
    s = read("src/train_xla.py")
    for bad in FORBIDDEN_TRAIN_STRINGS:
        if bad in s:
            fail(f"train_xla.py contains forbidden GPU/Accelerate string: {bad}")
    for needle in REQUIRED_TRAIN_STRINGS:
        if needle not in s:
            fail(f"train_xla.py missing required invariant string: {needle}")


def check_data() -> None:
    s = read("src/data.py")
    for needle in ["FixedDataset", "sample_fingerprint", "validate_full", "expected_vocab_size", "mmap"]:
        if needle not in s:
            fail(f"src/data.py missing required invariant: {needle}")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def check_main_configs() -> None:
    allowed = {
        "variant",
        "run_name",
        "save_dir",
        "init_ckpt",
        "_doc",
        "model.phase_mult",
        "model.laplace_alpha",
        "model.distance_laplace_alpha",
        "model.salience_alpha",
        "model.forget_alpha",
        "model.qtemp_alpha",
        "model.layer_dependent_gate",
        "model.learnable_layer_temp",
        "model.per_head_phase",
        "model.layer_dependent_phase",
        "model.baseline_type",
    }
    base = __import__("json").loads(read("configs/800m_base_s42.json"))
    hla = __import__("json").loads(read("configs/800m_hla_s42.json"))
    fb, fh = _flatten(base), _flatten(hla)
    illegal = []
    for k in sorted(set(fb) | set(fh)):
        if fb.get(k) != fh.get(k) and k not in allowed:
            illegal.append((k, fb.get(k), fh.get(k)))
    if illegal:
        text = "\n".join(f"  {k}: base={a!r}, hla={b!r}" for k, a, b in illegal)
        fail("Illegal differences between main base/HLA configs:\n" + text)
    for cfg_name, cfg in [("base", base), ("hla", hla)]:
        required = ["seed", "train_path", "val_path", "init_ckpt", "max_steps", "grad_accum", "num_cores"]
        missing = [k for k in required if k not in cfg]
        if missing:
            fail(f"{cfg_name} config missing keys: {missing}")


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()

    check_required_files()
    check_python_parse()
    check_model()
    check_train()
    check_data()
    check_main_configs()
    print("STERILITY AUDIT PASSED")


if __name__ == "__main__":
    main()
