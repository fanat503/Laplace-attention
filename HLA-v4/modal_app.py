"""
modal_app.py — Holographic Laplace Attention (HLA-v4)
NeurIPS-grade experiment orchestration for Modal cloud.

Usage:
  modal run modal_app.py --mode upload --use-init true
  modal run modal_app.py --mode check
  modal run modal_app.py --mode train --variant base --preset smoke
  modal run modal_app.py --mode train --variant v4   --preset article --seed 42
  modal run modal_app.py --mode train --variant v4   --preset aggressive --seed 42
  modal run modal_app.py --mode resume --variant v4  --resume-from /checkpoints/v4_article_590m_s42_abc12345/latest_v4_article_590m_s42_abc12345_resume.pt
  modal run modal_app.py --mode download --run-name v4_article_590m_s42_abc12345
"""

import os
import json
import time
import hashlib
from pathlib import Path

import modal

# =========================================================
# GLOBALS
# =========================================================

APP_NAME        = "hla-v4"
PYTHON_VERSION  = "3.11"
GPU_TYPE        = "A100-80GB"
GPU_COST_PER_HR = 3.70  # approximate Modal A100 cost

DATA_VOLUME_NAME = "hla-data"
CKPT_VOLUME_NAME = "hla-checkpoints"

LOCAL_DATA_DIR = Path("data")
LOCAL_TRAIN    = LOCAL_DATA_DIR / "train_fixed_tokens.pt"
LOCAL_VAL      = LOCAL_DATA_DIR / "val_fixed_tokens.pt"
LOCAL_INIT     = LOCAL_DATA_DIR / "init_state.pt"

REMOTE_TRAIN = "/data/train_fixed_tokens.pt"
REMOTE_VAL   = "/data/val_fixed_tokens.pt"
REMOTE_INIT  = "/data/init_state.pt"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",
        "accelerate==0.34.2",
    )
    .add_local_python_source("train1")
)

data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_SHAPE = {
    "block_size": 1024,
    "vocab_size": 50257,
    "n_layer":    16,
    "n_head":     12,
    "n_embd":     768,
}

COMMON_MODEL = {
    **MODEL_SHAPE,
    "use_laplace":     True,
    "laplace_range_k": 0.45,
    "laplace_range_v": 0.18,
    "beta_k":          0.55,
    "beta_v":          0.18,
    "gradient_checkpointing": True,
}

VARIANT_OVERRIDES = {
    "base": {
        "phase_mult":    0.0,
        "laplace_alpha": 0.0,
        "baseline_type": "parameter_matched_ablated",
    },
    "v4": {
        "phase_mult":    0.18,
        "laplace_alpha": 1.25,
        "baseline_type": "active_hla_v4",
    },
}


# =========================================================
# TRAINING PRESETS
# =========================================================

PRESETS = {
    # ----- Smoke test: ~2.6M tokens, ~2 min, ~$0.15 -----
    "smoke": {
        "batch_size_per_device":      32,
        "eval_batch_size_per_device":  8,
        "grad_accum":                  4,
        "lr":       3e-4,
        "min_lr":   3e-5,
        "warmup":   5,
        "max_steps": 20,
        "log_every":  5,
        "val_batches": 2,
        "save_every": 9999,
        "resume_every": 9999,
        "timeout_sec": 20 * 60,
        "estimated_hours": 0.05,
    },

    # ----- Article: ~590M tokens, ~3.5h, ~$13 -----
    "article": {
        "batch_size_per_device":      32,
        "eval_batch_size_per_device": 32,
        "grad_accum":                  4,
        "lr":       3e-4,
        "min_lr":   3e-5,
        "warmup":   300,
        "max_steps": 4500,
        "log_every":  250,
        "val_batches": 50,
        "save_every":  1500,
        "resume_every": 500,
        "timeout_sec": int(4.5 * 3600),
        "estimated_hours": 3.5,
    },

    # ----- Aggressive: ~590M tokens, bigger LR, riskier -----
    "aggressive": {
        "batch_size_per_device":      32,
        "eval_batch_size_per_device": 32,
        "grad_accum":                  4,
        "lr":       6e-4,
        "min_lr":   6e-5,
        "warmup":   200,
        "max_steps": 4500,
        "log_every":  250,
        "val_batches": 50,
        "save_every":  1500,
        "resume_every": 500,
        "timeout_sec": int(4.5 * 3600),
        "estimated_hours": 3.5,
    },

    # ----- Long: ~1.05B tokens, ~6h, ~$22 -----
    "long": {
        "batch_size_per_device":      32,
        "eval_batch_size_per_device": 32,
        "grad_accum":                  4,
        "lr":       3e-4,
        "min_lr":   3e-5,
        "warmup":   400,
        "max_steps": 8000,
        "log_every":  500,
        "val_batches": 50,
        "save_every":  2000,
        "resume_every": 1000,
        "timeout_sec": int(8 * 3600),
        "estimated_hours": 6.0,
    },

    # ----- CPT (Continued Pretraining): softer LR -----
    "cpt": {
        "batch_size_per_device":      32,
        "eval_batch_size_per_device": 32,
        "grad_accum":                  4,
        "lr":       1e-4,
        "min_lr":   1e-5,
        "warmup":   100,
        "max_steps": 3000,
        "log_every":  100,
        "val_batches": 50,
        "save_every":  1000,
        "resume_every": 500,
        "timeout_sec": int(3 * 3600),
        "estimated_hours": 2.5,
    },
}

COMMON_TRAIN = {
    "mixed_precision": "bf16",
    "train_path":      REMOTE_TRAIN,
    "val_path":        REMOTE_VAL,
    "min_free_gb_best":  1.0,
    "min_free_gb_final": 2.0,
    "grad_clip":         1.0,
    "fused_adamw":       False,
}


# =========================================================
# HELPERS
# =========================================================

def validate_model_shape(model_cfg: dict):
    assert model_cfg["n_embd"] % model_cfg["n_head"] == 0, \
        f"n_embd ({model_cfg['n_embd']}) must be divisible by n_head ({model_cfg['n_head']})"
    head_dim = model_cfg["n_embd"] // model_cfg["n_head"]
    assert head_dim % 2 == 0, \
        f"head_dim ({head_dim}) must be even for pairwise complex rotation"


def tokens_per_update(cfg: dict) -> int:
    return (
        cfg["batch_size_per_device"]
        * cfg["grad_accum"]
        * cfg["model"]["block_size"]
    )


def planned_tokens(cfg: dict) -> int:
    return tokens_per_update(cfg) * cfg["max_steps"]


def config_hash(cfg: dict) -> str:
    # Exclude non-deterministic / meta fields from hash
    hashable = {k: v for k, v in cfg.items()
                if k not in ("run_name", "save_dir", "config_hash",
                             "timeout_sec", "estimated_hours")}
    payload = json.dumps(hashable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def auto_run_name(variant: str, preset: str, seed: int, cfg: dict) -> str:
    toks_m = round(planned_tokens(cfg) / 1_000_000)
    return f"{variant}_{preset}_{toks_m}m_s{seed}_{config_hash(cfg)}"


def estimate_cost(cfg: dict) -> float:
    hours = cfg.get("estimated_hours", cfg["timeout_sec"] / 3600)
    return hours * GPU_COST_PER_HR


def make_config(
    variant: str,
    preset: str,
    seed: int,
    run_name: str | None = None,
    use_init: bool = True,
    resume_from: str | None = None,
) -> dict:
    if variant not in VARIANT_OVERRIDES:
        valid = ", ".join(VARIANT_OVERRIDES.keys())
        raise ValueError(f"Unknown variant: '{variant}'. Valid: {valid}")
    if preset not in PRESETS:
        valid = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: '{preset}'. Valid: {valid}")

    model = {**COMMON_MODEL, **VARIANT_OVERRIDES[variant]}
    validate_model_shape(model)

    cfg = {
        **COMMON_TRAIN,
        **PRESETS[preset],
        "seed":              seed,
        "variant":           variant,
        "preset":            preset,
        "experiment_family": "hla-v4",
        "model":             model,
    }

    # Resume takes priority over init
    if resume_from:
        cfg["resume_ckpt"] = resume_from
        cfg.pop("init_ckpt", None)
    elif use_init:
        cfg["init_ckpt"] = REMOTE_INIT

    if not run_name:
        run_name = auto_run_name(variant, preset, seed, cfg)

    cfg["run_name"]    = run_name
    cfg["save_dir"]    = f"/checkpoints/{run_name}"
    cfg["config_hash"] = config_hash(cfg)

    return cfg


def print_config_summary(cfg: dict):
    tpu   = tokens_per_update(cfg)
    total = planned_tokens(cfg)
    cost  = estimate_cost(cfg)

    print("=" * 72)
    print(f"  HLA-v4 Experiment Config")
    print("=" * 72)
    print(f"  Run name       : {cfg['run_name']}")
    print(f"  Variant        : {cfg['variant']}")
    print(f"  Preset         : {cfg['preset']}")
    print(f"  Seed           : {cfg['seed']}")
    print(f"  Config hash    : {cfg['config_hash']}")
    print(f"  Init ckpt      : {cfg.get('init_ckpt', '(none)')}")
    print(f"  Resume ckpt    : {cfg.get('resume_ckpt', '(none)')}")
    print()
    print(f"  Model:")
    print(f"    layers={cfg['model']['n_layer']}, heads={cfg['model']['n_head']}, "
          f"embd={cfg['model']['n_embd']}, block={cfg['model']['block_size']}")
    print(f"    phase_mult={cfg['model']['phase_mult']}, "
          f"laplace_alpha={cfg['model']['laplace_alpha']}")
    print(f"    baseline_type={cfg['model'].get('baseline_type', 'unknown')}")
    print()
    print(f"  Training:")
    print(f"    batch/device   = {cfg['batch_size_per_device']}")
    print(f"    grad_accum     = {cfg['grad_accum']}")
    print(f"    lr             = {cfg['lr']}")
    print(f"    warmup         = {cfg['warmup']}")
    print(f"    max_steps      = {cfg['max_steps']:,}")
    print(f"    tokens/update  = {tpu:,}")
    print(f"    planned tokens = {total:,} ({total / 1e9:.3f}B)")
    print()
    print(f"  Budget:")
    print(f"    timeout        = {cfg['timeout_sec'] / 3600:.2f}h")
    print(f"    est. cost      = ${cost:.2f}")
    print(f"    save_dir       = {cfg['save_dir']}")
    print("=" * 72)


def validate_local_files_for_upload(require_init: bool):
    required = [LOCAL_TRAIN, LOCAL_VAL]
    if require_init:
        required.append(LOCAL_INIT)

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local files for upload:\n  - " + "\n  - ".join(missing)
            + "\n\nMake sure you have these files in the 'data/' directory."
        )

    print("Local files ready for upload:")
    for p in required:
        size_mb = p.stat().st_size / 1024 ** 2
        print(f"  ✅ {p}  ({size_mb:.1f} MB)")


# =========================================================
# REMOTE FUNCTIONS
# =========================================================

@app.function(
    image=image,
    timeout=5 * 60,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def inspect_remote_files():
    import glob

    result = {"data_files": {}, "checkpoint_dirs": []}

    for p in [REMOTE_TRAIN, REMOTE_VAL, REMOTE_INIT]:
        result["data_files"][p] = {
            "exists":  os.path.exists(p),
            "size_mb": round(os.path.getsize(p) / 1024 ** 2, 2) if os.path.exists(p) else None,
        }

    ckpt_dirs = sorted(glob.glob("/checkpoints/*/"))
    for d in ckpt_dirs:
        files = os.listdir(d)
        total_mb = sum(
            os.path.getsize(os.path.join(d, f)) / 1024 ** 2
            for f in files if os.path.isfile(os.path.join(d, f))
        )
        result["checkpoint_dirs"].append({
            "path":     d,
            "n_files":  len(files),
            "total_mb": round(total_mb, 1),
            "files":    sorted(files),
        })

    return result


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=14 * 60 * 60,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote(config: dict):
    from train1 import train_worker

    start = time.time()
    try:
        train_worker(config)
    finally:
        elapsed = time.time() - start
        cost = (elapsed / 3600) * GPU_COST_PER_HR
        print(f"\n{'=' * 40}")
        print(f"  GPU time:  {elapsed / 3600:.2f}h")
        print(f"  Est. cost: ${cost:.2f}")
        print(f"{'=' * 40}")
        ckpt_vol.commit()


@app.function(
    image=image,
    timeout=10 * 60,
    volumes={"/checkpoints": ckpt_vol},
)
def list_run_files(run_name: str) -> dict:
    run_dir = f"/checkpoints/{run_name}"
    if not os.path.exists(run_dir):
        return {"error": f"Run directory not found: {run_dir}"}

    files = {}
    for f in sorted(os.listdir(run_dir)):
        fp = os.path.join(run_dir, f)
        if os.path.isfile(fp):
            files[f] = {
                "size_mb": round(os.path.getsize(fp) / 1024 ** 2, 2),
            }
    return {"run_name": run_name, "path": run_dir, "files": files}


# =========================================================
# LOCAL ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main(
    mode: str       = "check",
    variant: str    = "v4",
    preset: str     = "smoke",
    seed: int       = 42,
    run_name: str   = "",
    use_init: bool  = True,
    resume_from: str = "",
):
    """
    Modes:
      upload  — Upload data/init to Modal volume
      check   — Inspect remote volumes
      train   — Start training from scratch or init
      resume  — Continue training from a full checkpoint
      list    — List files in a completed run
    """

    # ---- UPLOAD ----
    if mode == "upload":
        validate_local_files_for_upload(require_init=use_init)

        with data_vol.batch_upload() as batch:
            batch.put_file(str(LOCAL_TRAIN), "/train_fixed_tokens.pt")
            batch.put_file(str(LOCAL_VAL), "/val_fixed_tokens.pt")
            if LOCAL_INIT.exists():
                batch.put_file(str(LOCAL_INIT), "/init_state.pt")
            elif use_init:
                raise FileNotFoundError("use_init=True but data/init_state.pt not found.")

        print(f"\n✅ Uploaded to Modal volume '{DATA_VOLUME_NAME}'")
        print("Next: modal run modal_app.py --mode check")
        return

    # ---- CHECK ----
    if mode == "check":
        info = inspect_remote_files.remote()

        print("\n📦 Data files:")
        for path, meta in info["data_files"].items():
            status = "✅" if meta["exists"] else "❌"
            size   = f"{meta['size_mb']} MB" if meta["size_mb"] else "N/A"
            print(f"  {status} {path}  ({size})")

        print(f"\n📁 Checkpoint directories ({len(info['checkpoint_dirs'])}):")
        if not info["checkpoint_dirs"]:
            print("  (none)")
        for d in info["checkpoint_dirs"]:
            print(f"  {d['path']}  ({d['n_files']} files, {d['total_mb']} MB)")
        return

    # ---- LIST ----
    if mode == "list":
        if not run_name:
            raise ValueError("--run-name is required for list mode")
        info = list_run_files.remote(run_name)
        if "error" in info:
            print(f"❌ {info['error']}")
            return
        print(f"\n📁 Run: {info['run_name']}")
        print(f"   Path: {info['path']}")
        for fname, meta in info["files"].items():
            print(f"   {fname}  ({meta['size_mb']} MB)")
        return

    # ---- TRAIN / RESUME ----
    if mode not in ("train", "resume"):
        valid = "upload, check, train, resume, list"
        raise ValueError(f"Unknown mode: '{mode}'. Valid: {valid}")

    resume_path = resume_from if resume_from else None

    if mode == "resume" and not resume_path:
        raise ValueError(
            "--resume-from is required for resume mode.\n"
            "Example: --resume-from /checkpoints/v4_article_.../latest_..._resume.pt"
        )

    cfg = make_config(
        variant=variant,
        preset=preset,
        seed=seed,
        run_name=run_name if run_name else None,
        use_init=use_init and (mode != "resume"),
        resume_from=resume_path,
    )

    print_config_summary(cfg)

    cost = estimate_cost(cfg)
    print(f"\n💰 Estimated cost for this run: ${cost:.2f}")

    if preset == "smoke":
        print("🔬 [SMOKE] Quick sanity check. Cheap and fast.")
    elif preset == "article":
        print("📄 [ARTICLE] Paper-quality matched comparison.")
    elif preset == "aggressive":
        print("⚡ [AGGRESSIVE] Higher LR, same tokens. Risk/reward.")
    elif preset == "long":
        print("🏋️ [LONG] Extended run. Make sure article preset passed first.")
    elif preset == "cpt":
        print("🔄 [CPT] Continued pretraining with softer LR.")

    if mode == "resume":
        print(f"🔁 [RESUME] Continuing from: {resume_path}")

    print("\n🚀 Launching training on Modal...")
    train_remote.remote(cfg)