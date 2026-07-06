<div align="center">

# HLA · Holographic Laplace Attention

### What if attention heads didn't have to whisper and listen through the same wire?

**A drop-in attention mechanism that separates *finding* tokens from *transmitting* them —<br>starting as a bit-exact standard Transformer and learning how much separation it needs.**

[![tests](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml/badge.svg)](https://github.com/fanat503/Laplace-attention/actions/workflows/tests.yml) [![Sterile](https://img.shields.io/badge/comparisons-bit--exact%20sterile-blueviolet)](#sterile-by-construction) [![Metrics](https://img.shields.io/badge/metrics-documented-blue)](HLA-v5/docs/METRICS.md) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20XLA%2FTPU-orange)](HLA-v5/requirements.txt) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[The problem](#the-problem) · [The idea](#the-idea) · [Mechanisms](#four-mechanisms-one-principle) · [Sterility](#sterile-by-construction) · [Diagnostics](#measure-everything) · [Quick start](#quick-start) · [Results](#status--roadmap) · [FAQ](#faq)

</div>

---

## The problem

Every attention head in a Transformer does **two unrelated jobs with one set of vectors**:

1. **Retrieval** — figure out *which* tokens matter (Query·Key matching)
2. **Transmission** — deliver *what* those tokens say (Value pathway)

Because both signals share the residual stream, they interfere: what one head *writes* as content lands in other heads' *retrieval* inputs as noise. The model pays a tax on every attention operation — irrelevant tokens attract attention they shouldn't, and important distant tokens fade when they shouldn't.

Recent work attacks pieces of this: Differential Transformer subtracts the noise *after* matching, Forgetting Transformer decays distant scores, Selective Attention masks tokens out. **HLA goes after the root cause: give each job its own learned, controllable channel.**

## The idea

> **Phase carries "where to look". Magnitude carries "what is said."**
> Like a hologram — where the image lives in interference patterns of phase, not in intensity.

HLA adds four mechanisms to causal self-attention. Each is per-head, content-conditioned, causally safe — and **exactly zero at initialization**. An HLA model *is* a standard Transformer at step 0 (bit-exact, verified by tests) and learns how much decoupling it needs. Any gain over a parameter-matched baseline trained from the *same initial weights* on the *same data in the same order* is attributable to the mechanisms — not to initialization luck, parameter count, or data shuffling.

Earlier iterations (v3/v4) showed a **−0.09 validation-loss gap** at 100M params. v5 is the infrastructure to find out — rigorously — whether that survives scale.

## Four mechanisms, one principle

*Every formula below is the actual code (`src/model.py`), not a simplification.*

| | Mechanism | Acts on | One-line intuition |
|---|---|---|---|
| | **Phase rotation** | Q, K before matching | rotate retrieval into subspaces noise doesn't occupy |
| | **Laplace gating** | K, V multiplicative | smooth per-token volume control for keys & values |
| | **Salience bias** | scores, additive | silence distractors ×0.135, amplify targets ×7.4 |
| | **Distance bias** | scores, additive | let each key decide how far it should reach |

### Content-conditioned phase rotation

```
angles = π · phase_mult · tanh(W_phase · x)          # per head, per token
q, k   = rotate_pairwise(q, angles_q), rotate_pairwise(k, angles_k)
```

An **isometry**: norms untouched (tested), only matching geometry changes. Composes with RoPE — position stays RoPE's job, alignment becomes learnable.

**Per-head budgets** (`per_head_phase`): each head learns its own rotation allowance `phase_mult · (1 + tanh(s_h)) ∈ [0, 2·phase_mult]` — H scalars per layer, directly readable as *"how much rotation did this head want?"*. Heads can opt out entirely.

**Depth-adaptive** (`layer_dependent_phase`): the budget scales with the same depth profile as the gates — deeper, more semantic layers may rotate more.

### Residual Laplace gating (K & V)

```
gate  = tanh(W_gate · x)                              # per-key content
range = base_range · (1 + 0.25 · tanh(W_range))       # learned per-head reach
mix   = (1−β) + β · exp(clamp(α · gate · range, ±clip))
k, v  = k · mix_k,  v · mix_v
```

Analytic envelope `mix ∈ [(1−β)+β·e^(−c), (1−β)+β·e^(c)]`; the clip is a numerical guard that never binds in shipped configs (verified). Note the deliberate **floor** (1−β): gating whispers, it cannot silence. Silencing is salience's job:

### Additive salience bias — no floor

```
score += clamp(α_s · range_s · tanh(W_sal · x_key), ±clip_s)
```

Log-space and additive ⇒ unbounded suppression: at ±2 nats a distracting key drops to **×0.135** attention weight, an important key gains **×7.4** — independent of distance.

### Distance-aware Laplace bias

```
score += clamp(α_d · range_d · dist(t,s) · gate_k(x_s), ±clip_d)
```

**Bidirectional**, unlike a pure forget gate: negative-gate keys decay with distance, positive-gate keys *survive* at long range — each key's content decides.

### FoX-style cumulative forget gate (baseline mechanism, built-in)

```
S_t     = cumsum( α_f · range_f · tanh(W_f · x_t) )
score  += clamp(S_i − S_j, ±clip_f)
```

The Forgetting-Transformer hypothesis (cumulative data-dependent decay), reproduced **inside the sterile harness**: identity-init, parameter-matched, same data — enabling the cleanest FoX-vs-HLA comparison possible. Use it as an ablation arm or as an external-baseline run.

### Learned depth profile (optional everywhere)

```
mult_l = 1 + (l/L) · softplus(θ_l) / softplus(0)      # one scalar per layer
```

At θ=0 this **equals** the static heuristic `1 + l/L` exactly — then the model calibrates its own depth curve, logged live (`layer_temp_*`) as a free interpretability figure.

## Sterile by construction

**The comparison methodology is a contribution in itself.** Base and HLA differ in *nothing* except the active mechanisms:

| Guarantee | Enforcement |
|---|---|
| Identical backbone weights | `make_init.py --shared-backbone` copies every non-HLA tensor, verifies tensor equality |
| Bit-exact identity at init | all HLA params zeroed; `hla_identity_error() == 0.0` asserted at init creation **and** trainer load; `torch.equal` logits test |
| Exact parameter matching | every HLA module exists in base too (disabled via α=0, params frozen — tested); counts equal per pair |
| Identical data order | `FixedDataset`: deterministic non-overlapping chunks, fingerprints, even sharding, O(1) exact resume |
| Config discipline | `validate_configs.py` rejects any diff outside the allow-list; `--flops-matched` may only *reduce* HLA steps |
| Zero future leakage | permutation causality tests across every mechanism |
| Full provenance | config hashes, env snapshots, dataset manifests stored with every run |

Ships with **parameter-matched** *and* **FLOPs-matched** config pairs (200M → 800M).

## Measure everything

Full math for every metric: [`docs/METRICS.md`](HLA-v5/docs/METRICS.md) · unified formula & theorems: [`docs/THEORY.md`](HLA-v5/docs/THEORY.md) · formal sterility protocol: [`docs/STERILITY.md`](HLA-v5/docs/STERILITY.md) · pre-registration: [`docs/EXPERIMENT_CARD.md`](HLA-v5/docs/EXPERIMENT_CARD.md). The highlights:

| Question | Metric (logged to CSV during training) |
|---|---|
| Is retrieval getting *cleaner* while composition survives? | `qk_interference` ↓, `ov_interference` ≈, `qk_ov_separation` ↑ — Transformer-Circuits-style subspace overlaps between what heads write and what other heads' QK/OV circuits read |
| Does the model resist *attractive* noise? | `distractor_margin` = P(target) − max P(distractor) on induction probes with **repeated competing patterns** injected |
| Are my envelopes too tight? | `*_sat_frac` — fraction of \|tanh\| > 0.99. ≈0 ⇒ bounds fine; >10–20% persistent ⇒ widen and re-run |
| Did phase collapse to something trivial? | `svd_phase_erank` (effective rank of learned rotations), `svd_phase_top1` |
| Do Q/K specialize while V stays put? | `svd_qk_stable_rank` vs `svd_v_stable_rank` — the decoupling picture in two curves |
| What did the model *choose*? | `layer_temp_*` (learned depth profile), `phase_budget_*` (per-head rotation appetite) |

Spectral & interference metrics run master-only every `svd_every` steps (~15 s @ 300M): **training is never slowed**.

## Quick start

```bash
pip install torch pytest                    # CPU is enough for tests & init
cd HLA-v5

# 0 · Trust nothing, verify (always, before any TPU time)
python -m pytest tests/ -q                  # → 162 passed

# 1 · Sterility gate + numeric sanity audit for the config pair
python scripts/validate_configs.py \
    --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json
python scripts/audit_config_values.py --all # envelopes, clip slack, grad
                                            #   liveness, LR/width band — PASS/WARN/FAIL

# 2 · Shared sterile init (one backbone, two roles)
python src/make_init.py --shared-backbone \
    --base-config configs/200m_base_s42.json --hla-config configs/200m_hla_s42.json \
    --out-base inits/init_200m_base_s42.pt  --out-hla inits/init_200m_hla_s42.pt

# 3 · Train both from the same weights (TPU/XLA)
python src/train_xla.py --config configs/200m_base_s42.json
python src/train_xla.py --config configs/200m_hla_s42.json

# Override anything without touching configs:
python src/train_xla.py --config configs/200m_hla_s42.json \
    --override max_steps=100 --override run_name=smoke
```

> **Kaggle TPU v5e-8**: configs ship with `num_cores: 8` and `/kaggle/...` paths.
> Ladder: `smoke` (10 steps) → `pilot` (1 000) → full runs.

## Repository layout

```
HLA-v5/
├── src/
│   ├── model.py        # GPT + all HLA mechanisms (single file, no framework magic)
│   ├── train_xla.py    # TPU trainer: sharded determinism, O(1) resume,
│   │                   #   token-weighted validation, crash-safe checkpoints
│   ├── data.py         # FixedDataset — sterile fixed-token pipeline
│   ├── eval.py         # induction / distractor / SVD / interference probes
│   ├── make_init.py    # shared-backbone sterile init generator
│   └── manifest.py     # provenance & hashing
├── configs/            # paired base/HLA JSONs: 200m…800m, FLOPs-matched pairs,
│                       #   v2 recipe (aggressive envelope + salience), pilot, smoke
├── scripts/            # validate_configs, sterility audit, data prep, analysis
├── docs/METRICS.md     # exact math of every metric
└── tests/              # 162 CPU tests — run anywhere, no TPU needed
```

## 162 tests = the paper's claims, executable

Sterility (bit-exact identity, parameter matching, corrupted-init rejection) · causality (permutation tests, every mechanism) · math invariants (rotation isometry & invertibility, envelope bounds under saturation, budget bounds, batch invariance) · training (one-step parity from shared init, gradient flow to every *active* param, frozen *inactive* params, NaN-robustness at extreme weights) · backends (SDPA ↔ manual parity; SDPA refuses to silently drop active biases) · metric ground truth (interference = 0 for orthogonal heads, = self for identical heads; rank-1 collapse detection) · data & trainer (determinism, sharding without duplicates, exact-suffix resume, LR schedule endpoints).

## Status & roadmap

- [x] v3/v4: −0.09 val-loss gap @ 100M (pre-sterile-infrastructure)
- [x] v5 infrastructure: mechanisms, sterility protocol, diagnostics, 162 tests
- [ ] Smoke + pilot @ Kaggle TPU v5e-8 ← **here**
- [ ] 200M pairs: v1 (soft gates) & v2 (aggressive + salience), reproduce the gap sterile
- [ ] Component ablations: phase / gates / salience / distance / learned-temp / per-head
- [ ] ≥3 seeds, error bars → 300M FLOPs-matched → 7B scaling trend
- [ ] Downstream evals (lm-eval-harness) + FoX baseline in the same sterile harness
- [ ] Multi-B headline run (compute grants — reach out if you can help)

## FAQ

<details>
<summary><b>Isn't this what PoPE did?</b></summary>

Nearly the opposite. Both works identify the same object: in complex form the QK product contains a **content-dependent phase term** that interferes with positional rotation. **PoPE deletes it** (strips phases, keeps magnitudes) so position becomes clean. **HLA domesticates it**: the phase is learned explicitly (tanh-bounded, zero-init), giving the model an extra *retrieval coordinate* — while RoPE keeps position and gates keep salience. If HLA's phase channel yields gains, content phase is a *resource*, not a nuisance — a directly testable disagreement between the two papers.
</details>

<details>
<summary><b>How is this different from the Forgetting Transformer?</b></summary>

FoX applies a data-dependent *decay* to attention scores — one direction, forget. HLA's distance bias is **bidirectional** (each key's content decides whether to fade or to *persist* at long range), and FoX has no phase-space retrieval and no V-pathway gating. FoX is our strongest external baseline, and the sterile harness can host a forget-gate variant for the cleanest FoX-vs-HLA comparison in the literature: same init, same data order.
</details>

<details>
<summary><b>Why "holographic"?</b></summary>

In a hologram the image is stored in *phase interference patterns*, not intensity. In HLA, retrieval information lives in learned *phase alignment* between Q and K, while content travels in *magnitudes* through V. Phase = where to look; magnitude = what is said.
</details>

<details>
<summary><b>Why should I trust a −0.09 gap?</b></summary>

Because the infrastructure makes cheating structurally hard: same initial weights (verified tensor-equal), same data order (fingerprinted), equal parameters (counted, frozen when inactive), bit-exact identity at init (torch.equal test), and a config validator that rejects any undeclared difference. The remaining honest caveat: v3/v4 numbers predate this harness — reproducing them *inside* it is the first roadmap item.
</details>

---

<div align="center">

**Independent research** by a 13-year-old researcher · mentorship & compute from a multimodal-ML collaborator
Contributions, replications, and compute support welcome — open an issue.

**MIT License**

</div>
