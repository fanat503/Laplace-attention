<div align="center">

# HLA · Holographic Laplace Attention

**Separating finding tokens from transmitting them in softmax attention**

[![tests](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml/badge.svg)](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml)
[![Theory](https://img.shields.io/badge/theory-9%20verified%20theorems-blue)](docs/THEORY.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

</div>

Every attention head does two interconnected jobs with one set of vectors: **retrieval** (which tokens matter) and **transmission** (what they say). Because both share the residual stream, they interfere: one head's *output* lands in other heads' *retrieval* inputs as noise. HLA gives each job its own learned, channel.

This repository contains:

1. **The mechanism** — one modified attention equation, seven identity-initialized components ([`src/model.py`](src/model.py), single file);
2. **A sterile comparison harness** — base and HLA train from the *same initial weights* on the *same data in the same order*, with parameter matching and config validation enforced by tests;
3. **Theory and diagnostics** — 9 numerically verified theorems ([`docs/THEORY.md`](docs/THEORY.md)), causal knockout probes, interference and spectral metrics logged during training ([`docs/METRICS.md`](docs/METRICS.md)).

**Status**: infrastructure complete and tested; sterile training runs are the current roadmap item. Earlier iterations showed a 0.09 validation loss gap at 100M — reproducing that inside the harness is the first experiment, not a claim.

## The idea in one formula

```
score_ij = ( τ_i · R(θ_i)·q_i ) · ( m_j · R(φ_j)·k_j ) / √d  +  B_ij
out_i    = Σ_{j≤i} softmax_j(score_ij) · ( u_j · v_j )
```

Phase carries *where to look*; magnitude carries *what is said* — like a hologram, where the image lives in phase interference, not intensity.

| Component | Role | At init |
|---|---|---|
| `R(θ_i)`, `R(φ_j)` | content-conditioned rotation of Q, K — learned matching geometry, composes with RoPE | `I` |
| `m_j`, `u_j` | multiplicative K, V gates: key loudness and content volume | `1` |
| `B_ij` | additive biases: salience  + content-conditioned distance decay + FoX-style gate (baseline) | `0` |
| `τ_i` | per-query softmax temperature | `1` |

Everything is per-head, tanh-bounded, causally safe, and **zero at initialization**: an HLA model *is* a standard Transformer at step 0 (in fp32 and bf16). Full derivation, envelopes, and proofs: [`docs/THEORY.md`](docs/THEORY.md).

## Getting started

```bash
git clone https://github.com/fanat503/Laplace-attention.git
cd Laplace-attention
pip install -r requirements.txt        # yoou can use CPU

python -m pytest tests/ -q             # 256 passed
python scripts/audit_sterility.py      # STERILITY AUDIT PASSED
```

Train a sterile base/HLA pair (TPU/XLA; every step below also runs on CPU for smoke-testing):

```bash
# 1 · Validate configs and then make init
python scripts/validate_configs.py --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json
python src/make_init.py --shared-backbone \
    --base-config configs/200m_base_s42.json --hla-config configs/200m_hla_s42.json \
    --out-base inits/init_200m_base_s42.pt --out-hla inits/init_200m_hla_s42.pt

# 2 · Tokenize
python scripts/prepare_c4_data.py --train-tokens 5400000000 --val-tokens 20000000 --out-dir data

# 3 · Train both
python src/train_xla.py --config configs/200m_base_s42.json
python src/train_xla.py --config configs/200m_hla_s42.json

# 4 · Ablation matrix
python scripts/make_ablation_configs.py \
    --base configs/200m_base_v2_s42.json --hla configs/200m_hla_v2_s42.json \
    --outdir configs/ablations_200m --seeds 42 43 44
```

Recommended ladder before any long run: smoke (10 steps), then pilot (1000 steps) and then full runs. We pre-registered our experimental design on [`docs/EXPERIMENT_CARD.md`](docs/EXPERIMENT_CARD.md).

## Why trust the comparison

The methodology is designed so that cheating is hard — 256 tests ([protocol & threat model](docs/STERILITY.md)):

- **Same start** — shared backbone weights (tensor-equal), HLA params zeroed;
- **Same size** — every mechanism module exists in the base too (α = 0, frozen, counted);
- **Same data** — deterministic fixed-token pipeline, fingerprints, even sharding, sample-exact resume;
- **Same knobs** — config validator rejects any undeclared difference; hyperparameters tuned on base only; HLA params excluded from weight decay (zero *is* their identity state);
- **Measured honestly** — parameter-matched and FLOPs-matched pairs; mechanism compute overhead ~7% at 200M.

Training logs record interference metrics (specifically, whether retrieval clarity improves while compositional capabilities are preserved), distractor induction margins, saturation fractions, spectral ranks, and per-mechanism gradient norms. Crucially, the figures presented in this paper are generated directly from these raw CSV logs rather than via post-hoc analysis. Checkpoint level causal probes, including mechanism knockouts and prefix matching scores, are documented in [`src/eval.py`](src/eval.py).

## Repository layout

```
├── src/          model.py (GPT + mechanisms), train_xla.py (TPU trainer), eval.py (probes)
│                 make_init.py, data.py, manifest.py, utils.py
├── configs/      20 paired base/HLA JSONs, FLOPs-matched, v2 recipe, pilot, smoke
├── scripts/      validation (validate_configs, audit_*), experiment (make_ablation_configs,
│                 prepare_c4_data), analysis (profile_flops, make_plots, analyze_*)
├── docs/         THEORY, METRICS, STERILITY, EXPERIMENT_CARD, DATA_CARD
└── tests/        256 tests
```

CI runs the full suite plus three audits on every push.

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

**Apache-2.0** · Independent research; contributions, replications, and compute support welcome — open an issue.
