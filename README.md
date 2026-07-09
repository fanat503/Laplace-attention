<div align="center">

# HLA · Holographic Laplace Attention

### What if attention heads didn't have to whisper and listen through the same wire?

**A drop-in attention mechanism that separates *finding* tokens from *transmitting* them —<br>starting as a bit-exact standard Transformer and learning how much separation it needs.**

[![tests](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml/badge.svg)](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml) [![Sterile](https://img.shields.io/badge/comparisons-bit--exact%20sterile-blueviolet)](#sterile-by-construction) [![Theory](https://img.shields.io/badge/theory-5%20verified%20theorems-blue)](HLA-v5/docs/THEORY.md) [![Metrics](https://img.shields.io/badge/metrics-documented-blue)](HLA-v5/docs/METRICS.md) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20XLA%2FTPU-orange)](HLA-v5/requirements.txt) [![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

[The problem](#the-problem) · [The idea](#the-idea) · [One formula](#one-formula-six-mechanisms) · [Mechanisms](#the-mechanisms) · [Sterility](#sterile-by-construction) · [Diagnostics](#measure-everything) · [Quick start](#quick-start) · [Layout](#repository-layout) · [Roadmap](#status--roadmap) · [FAQ](#faq)

</div>

---

## The problem

Every attention head in a Transformer does **two unrelated jobs with one set of vectors**:

1. **Retrieval** — figure out *which* tokens matter (Query·Key matching)
2. **Transmission** — deliver *what* those tokens say (Value pathway)

Because both signals share the residual stream, they interfere: what one head *writes* as content lands in other heads' *retrieval* inputs as noise. Irrelevant tokens attract attention they shouldn't; important distant tokens fade when they shouldn't.

Recent work attacks pieces of this — Differential Transformer subtracts noise *after* matching, Forgetting Transformer decays distant scores, Selective Attention masks tokens out. **HLA goes after the root cause: give each job its own learned, controllable channel.**

## The idea

> **Phase carries "where to look". Magnitude carries "what is said."**
> Like a hologram — the image lives in interference patterns of phase, not in intensity.

Every HLA mechanism is per-head, content-conditioned, causally safe — and **exactly zero at initialization**. An HLA model *is* a standard Transformer at step 0 (bit-exact in fp32 **and** bf16, verified by tests) and learns how much decoupling it needs. Any gain over a parameter-matched baseline trained from the *same initial weights* on the *same data in the same order* is attributable to the mechanisms — not to initialization luck, parameter count, or data shuffling.

Earlier iterations (v3/v4) showed a **−0.09 validation-loss gap** at 100M params. v5 is the infrastructure to find out — rigorously — whether that survives scale.

## One formula, six mechanisms

The entire architecture is one modified attention equation (full derivation and five formally proved theorems: [`docs/THEORY.md`](HLA-v5/docs/THEORY.md)):

```
score_ij = ( R(θ_i)·q_i ) · ( m_j · R(φ_j)·k_j ) / √d  +  B_ij
out_i    = Σ_{j≤i} softmax_j(score_ij) · ( u_j · v_j )
```

| Symbol | Role | Config switch | At init |
|---|---|---|---|
| `R(θ_i)`, `R(φ_j)` | rotate Q/K into learned matching geometry | `phase_mult` | identity `I` |
| `m_j` | multiplicative key salience (floor `1−β_k`) | `use_laplace` + `laplace_alpha` | `1` |
| `u_j` | multiplicative content volume | same pair (V side) | `1` |
| `B_ij` | additive score bias = salience + distance + forget | `use_salience_bias` / `use_distance_laplace` / `use_forget_gate` | `0` |

Retrieval (inside softmax) and transmission (`u_j·v_j`) share **no learned scalars** — that is the decoupling, stated syntactically. Proven and numerically verified in `tests/test_theory.py` (9 tests): exact identity at init (T1), strictly larger function class than the baseline (T2), rotation is an isometry and a group action (T3), all perturbations analytically bounded (T4), mechanisms receive non-zero gradient from step one (T5).

## The mechanisms

*Every formula below is the actual code (`HLA-v5/src/model.py`, 838 lines, single file), not a simplification.*

| # | Mechanism | Acts on | One-line intuition |
|---|---|---|---|
| 1 | **Phase rotation** | Q, K before matching | rotate retrieval into subspaces noise doesn't occupy |
| 2 | **Laplace gating** | K, V multiplicative | smooth per-token volume control for keys & values |
| 3 | **Salience bias** | scores, additive | silence distractors ×0.135, amplify targets ×7.4 — no floor |
| 4 | **Distance bias** | scores, additive | each key's content decides how far it reaches |
| 5 | **Forget gate** (FoX-family, **baseline arm — OFF in all HLA configs**) | scores, cumulative | the Forgetting-Transformer hypothesis, reproduced in-harness for fair comparison (Lin et al., ICLR 2025) |
| 6 | **Adaptivity axes** | budgets of 1–4 | per-head (`per_head_phase`, learned `W_range_*`) and per-depth (`layer_dependent_gate`/`_phase`, `learnable_layer_temp`) |

### 1 · Content-conditioned phase rotation

```
angles = π · phase_mult · tanh(W_phase · x)          # per head, per token
q, k   = rotate_pairwise(q, angles_q), rotate_pairwise(k, angles_k)
```

An **isometry**: norms untouched (tested), only matching geometry changes. Composes with RoPE — position stays RoPE's job, alignment becomes learnable. Optional: per-head budgets `phase_mult·(1+tanh(s_h)) ∈ [0, 2·phase_mult]` (heads can opt out entirely) and depth scaling.

### 2 · Residual Laplace gating (K & V)

```
gate  = tanh(W_gate · x)                              # per-key content
range = base_range · (1 + 0.25 · tanh(W_range))       # learned per-head reach
mix   = (1−β) + β · exp(clamp(α · gate · range, ±clip))
k, v  = k · mix_k,  v · mix_v
```

Analytic envelope `mix ∈ [(1−β)+β·e^(−c), (1−β)+β·e^(c)]`; the clip is a numerical guard that never binds in shipped configs (checked by `audit_config_values.py`). The deliberate **floor** (1−β) means gating whispers — it cannot silence. Silencing is salience's job:

### 3 · Additive salience bias — no floor

```
score += clamp(α_s · range_s · tanh(W_sal · x_key), ±clip_s)
```

Log-space and additive ⇒ unbounded suppression: at ±2 nats a distracting key drops to **×0.135** attention weight, an important key gains **×7.4** — independent of distance.

### 4 · Distance-aware Laplace bias

```
score += clamp(α_d · range_d · dist(t,s) · gate_k(x_s), ±clip_d)
```

**Bidirectional**, unlike a pure forget gate: negative-gate keys decay with distance, positive-gate keys *survive* at long range.

### 5 · FoX-style cumulative forget gate (baseline arm)

```
S_t    = cumsum( α_f · range_f · tanh(W_f · x_t) )
score += clamp(S_i − S_j, ±clip_f)
```

**Not part of HLA** — a competitor mechanism (Forgetting Transformer, Lin et al., ICLR 2025) re-implemented inside the sterile harness, identity-initialized and parameter-matched, activated only by the `forget` ablation arm. Enables the cleanest `base | FoX-style | HLA` three-way comparison in the literature: same init, same data order, same trainer.

### 6 · Learned depth profile

```
mult_l = 1 + (l/L) · softplus(θ_l) / softplus(0)      # one scalar per layer
```

At θ=0 this **equals** the static heuristic `1 + l/L` exactly — the model then calibrates its own depth curve, logged live (`layer_temp_*`) as a free interpretability figure.

## Sterile by construction

**The comparison methodology is a contribution in itself** (formal protocol with invariants I1–I5 and a threat model: [`docs/STERILITY.md`](HLA-v5/docs/STERILITY.md)). Base and HLA differ in *nothing* except the active mechanisms:

| Guarantee | Enforcement |
|---|---|
| Identical backbone weights | `make_init.py --shared-backbone` copies every non-HLA tensor, verifies tensor equality |
| Bit-exact identity at init | all HLA params zeroed; `hla_identity_error() == 0.0` asserted at init creation **and** trainer load; `torch.equal` logits test (fp32 & bf16) |
| Exact parameter matching | every HLA module exists in base too (α=0, params frozen — tested); counts equal per pair |
| Identical data order | `FixedDataset`: deterministic non-overlapping chunks, fingerprints, even sharding, O(1) exact resume |
| Config discipline | `validate_configs.py` rejects any diff outside the allow-list; `--flops-matched` may only *reduce* HLA steps |
| Optimizer fairness | HLA params excluded from weight decay (zero *is* their identity state); all hyperparameters tuned on base only |
| Zero future leakage | permutation causality tests across every mechanism (incl. the cumulative forget gate) |
| Full provenance | config hashes, env snapshots, dataset manifests stored with every run |

Ships with **parameter-matched** *and* **FLOPs-matched** config pairs, plus a **single-factor ablation matrix generator** (`make_ablation_configs.py`: 9 arms × N seeds, shared init per seed ⇒ paired statistics by construction).

## Measure everything

Full math for every metric: [`docs/METRICS.md`](HLA-v5/docs/METRICS.md) · theory: [`docs/THEORY.md`](HLA-v5/docs/THEORY.md) · pre-registered plan: [`docs/EXPERIMENT_CARD.md`](HLA-v5/docs/EXPERIMENT_CARD.md) · data provenance: [`docs/DATA_CARD.md`](HLA-v5/docs/DATA_CARD.md).

| Question | Metric (logged to the training CSV) |
|---|---|
| Is retrieval getting *cleaner* while composition survives? | `qk_interference` ↓, `ov_interference` ≈, `qk_ov_separation` ↑ — Transformer-Circuits-style subspace overlaps |
| Does the model resist *attractive* noise? | `distractor_margin` = P(target) − max P(distractor), induction probes with repeated competing patterns |
| Are my envelopes too tight? | `*_sat_frac` — fraction of \|tanh\| > 0.99. ≈0 ⇒ bounds fine; >10–20% persistent ⇒ widen and re-run |
| Did phase collapse to something trivial? | `svd_phase_erank`, `svd_phase_top1` |
| Do Q/K specialize while V stays put? | `svd_qk_stable_rank` vs `svd_v_stable_rank` |
| What did the model *choose*? | `layer_temp_*` (learned depth profile), `phase_budget_*` (per-head rotation appetite) |

Spectral & interference metrics run master-only every `svd_every` steps (~15 s @ 300M): **training is never slowed** (attention-entropy computation is likewise gated out of training forwards).

## Quick start

```bash
git clone https://github.com/fanat503/Laplace-attention.git
cd Laplace-attention/HLA-v5
pip install -r requirements.txt             # torch (CPU is enough), numpy, pytest

# 0 · Trust nothing, verify (always, before any TPU time)
python -m pytest tests/ -q                  # → 162 passed
python scripts/audit_sterility.py           # → STERILITY AUDIT PASSED

# 1 · Sterility gate + numeric sanity audit for the config pair
python scripts/validate_configs.py \
    --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json
python scripts/audit_config_values.py --all # envelopes, clip slack, grad liveness — PASS/WARN/FAIL

# 2 · Prepare tokenized data (downloads + tokenizes C4 in one pass; CPU session)
python scripts/prepare_c4_data.py \
    --train-tokens 5400000000 --val-tokens 20000000 --out-dir data

# 3 · Shared sterile init (one backbone, two roles)
python src/make_init.py --shared-backbone \
    --base-config configs/200m_base_s42.json --hla-config configs/200m_hla_s42.json \
    --out-base inits/init_200m_base_s42.pt  --out-hla inits/init_200m_hla_s42.pt

# 4 · Train both from the same weights (TPU/XLA; sequential — one run uses all 8 cores)
python src/train_xla.py --config configs/200m_base_s42.json
python src/train_xla.py --config configs/200m_hla_s42.json

# 5 · Full ablation matrix for the paper (9 arms × N seeds, incl. the FoX arm)
python scripts/make_ablation_configs.py \
    --base configs/200m_base_v2_s42.json --hla configs/200m_hla_v2_s42.json \
    --outdir configs/ablations_200m --seeds 42 43 44

# Override anything without touching configs (incl. resuming an interrupted run):
python src/train_xla.py --config configs/200m_hla_s42.json \
    --override resume_ckpt=runs/200m_hla_s42/latest_200m_hla_s42_resume.pt
```

> **Kaggle TPU v5e-8**: configs ship with `num_cores: 8` and `/kaggle/...` paths. One 200M run ≈ 6–8 h, so a base+HLA pair spans sessions — the trainer checkpoints every 500 steps and resumes exactly (O(1), sample-precise).
> Ladder: `smoke` (10 steps) → `pilot` (1 000) → full runs.

## Repository layout

*(generated from the actual tree — every count is real)*

```
Laplace-attention/
├── README.md · LICENSE · CITATION.cff · CONTRIBUTING.md · .gitignore
├── .github/workflows/tests.yml     # CI: 162 tests + 3 audits on every push/PR, warnings-as-errors
├── HLA-v4/                         # archived predecessor (3 files; identity-init bug fixed retroactively)
│       make_init.py · modal_app.py · train1.py
└── HLA-v5/                         # ← current version, all development here
    ├── src/                        # 8 files, ~3 950 lines
    │   ├── model.py                #   838 · GPT + all six mechanisms, generate(), single file
    │   ├── train_xla.py            # 1 648 · TPU trainer: sharded determinism, O(1) resume,
    │   │                           #         token-weighted validation, crash-safe checkpoints
    │   ├── eval.py                 #   472 · induction / distractor / SVD / interference / depth probes
    │   ├── make_init.py            #   422 · shared-backbone sterile init generator
    │   ├── data.py                 #   362 · FixedDataset: fixed-token pipeline (.pt / .bin+sidecar)
    │   ├── manifest.py             #   143 · provenance & hashing
    │   ├── utils.py                #    63 · seeding, atomic IO
    │   └── __init__.py
    ├── configs/                    # 20 JSONs + README: paired base/HLA at 200m·300m·600m·700m·800m,
    │                               #   FLOPs-matched 300m pair, batch-shape ablations (700m b2g16/b4g8/b8g4),
    │                               #   v2 recipe (aggressive envelope + salience), pilot, smoke
    ├── scripts/                    # 22 Python tools + 2 shell helpers
    │   ├── validation:   validate_configs · audit_config_values · audit_sterility ·
    │   │                 validate_data_pair · validate_log · verify_run · preflight.sh
    │   ├── experiment:   make_ablation_configs · prepare_c4_data · prepare_data ·
    │   │                 make_dummy_data · create_run_manifest · download_data_gcs.sh
    │   ├── analysis:     analyze_checkpoint · analyze_subspaces · compare_attention_kl ·
    │   │                 compare_inits · inspect_checkpoint · make_plots · profile_flops
    │   └── utility:      check_dataloader · check_environment · count_params · estimate_budget
    ├── docs/                       # 5 documents
    │   ├── THEORY.md               #   unified formula + 5 proved & numerically verified theorems
    │   ├── STERILITY.md            #   formal protocol: invariants I1–I5 + threat model
    │   ├── METRICS.md              #   exact math of every logged metric
    │   ├── EXPERIMENT_CARD.md      #   pre-registered hypotheses, run ladder, exclusion rules
    │   └── DATA_CARD.md            #   corpus, tokenizer, processing guarantees
    ├── tests/                      # 7 files · 162 tests · CPU-only, ~7 s
    │   ├── test_model.py           #   73 · mechanisms, sterility, causality, bf16, generate
    │   ├── test_train_utils.py     #   24 · LR schedule, sharded samplers, config compat, optimizer groups
    │   ├── test_eval.py            #   19 · probes with ground-truth witnesses
    │   ├── test_data.py            #   14 · determinism, sharding, .bin sidecars
    │   ├── test_make_init.py       #   12 · shared-backbone equality, identity rejection
    │   ├── test_ablation_configs.py#   11 · single-factor discipline, paired seeds
    │   └── test_theory.py          #    9 · one test per theorem claim
    ├── requirements.txt · pyproject.toml
```

## 162 tests = the paper's claims, executable

Sterility (bit-exact identity in fp32/bf16, parameter matching, corrupted-init rejection) · causality (permutation tests, every mechanism incl. cumulative forget) · theorems (identity, strict-inclusion witness, isometry & group action, envelope endpoints under saturation, non-vanishing gradients) · training (one-step parity from shared init, gradient flow to every *active* param, frozen *inactive* params, NaN-robustness at extreme weights, grad-checkpointing equivalence) · backends (SDPA ↔ manual parity; SDPA refuses to silently drop active score biases) · metric ground truth (interference = 0 for orthogonal heads, = self for identical heads; rank-1 collapse detection) · generation (greedy determinism, padded-vocab never sampled, context cropping) · data & trainer (determinism, sharding without duplicates, exact-suffix resume, tiny-dataset loud failure, LR schedule endpoints) · ablation tooling (single-factor discipline, shared-init pairing, structural-flag uniformity).

## Status & roadmap

- [x] v3/v4: −0.09 val-loss gap @ 100M (pre-sterile infrastructure; v4 identity-init bug found & fixed retroactively)
- [x] v5 infrastructure: 6 mechanisms, sterility protocol, theory, diagnostics, 162 tests, CI
- [x] Pre-registered experimental plan ([EXPERIMENT_CARD](HLA-v5/docs/EXPERIMENT_CARD.md))
- [ ] Smoke + pilot @ Kaggle TPU v5e-8 ← **here**
- [ ] 200M pairs (v1 soft & v2 aggressive recipes), seed 42 — reproduce the gap sterile
- [ ] Ablation matrix: 9 arms × 3 seeds (incl. FoX baseline arm)
- [ ] 300M FLOPs-matched → 700M scaling trend, ≥3 seeds, paired statistics
- [ ] Downstream evals (lm-eval-harness) · headline scale (compute grants welcome)

## FAQ

<details>
<summary><b>Isn't this what PoPE did?</b></summary>

Nearly the opposite. Both works identify the same object: in complex form the QK product contains a **content-dependent phase term** that interferes with positional rotation. **PoPE deletes it** (strips phases, keeps magnitudes) so position becomes clean. **HLA domesticates it**: the phase is learned explicitly (tanh-bounded, zero-init), giving the model an extra *retrieval coordinate* — while RoPE keeps position and gates keep salience. If HLA's phase channel yields gains, content phase is a *resource*, not a nuisance — a directly testable disagreement between the two papers.
</details>

<details>
<summary><b>Why is a Forgetting-Transformer-style gate in the codebase — isn't that their idea?</b></summary>

It is their idea, clearly credited (Lin et al., ICLR 2025), and that is exactly why it is here: **as a baseline**, OFF in every HLA config, activated only by the `forget` ablation arm. Re-implementing a competitor's mechanism inside your own controlled harness is standard practice — it is the only way to compare fairly (same init, same data order, same trainer). It also enables the composition experiment (HLA + forget), which no one has run.
</details>

<details>
<summary><b>Why "holographic"?</b></summary>

In a hologram the image is stored in *phase interference patterns*, not intensity. In HLA, retrieval information lives in learned *phase alignment* between Q and K, while content travels in *magnitudes* through V. Phase = where to look; magnitude = what is said.
</details>

<details>
<summary><b>Why should I trust a −0.09 gap?</b></summary>

Because the infrastructure makes cheating structurally hard: same initial weights (verified tensor-equal), same data order (fingerprinted), equal parameters (counted, frozen when inactive), bit-exact identity at init (torch.equal, fp32 & bf16), a config validator that rejects any undeclared difference, and a pre-registered plan committed before the headline runs. The remaining honest caveat: v3/v4 numbers predate this harness — reproducing them *inside* it is the first roadmap item.
</details>

---

<div align="center">

**Independent research** by a 13-year-old researcher · mentorship & compute from a multimodal-ML collaborator
Contributions, replications, and compute support welcome — open an issue.

**Apache-2.0 License** · [`CITATION.cff`](CITATION.cff)

</div>
