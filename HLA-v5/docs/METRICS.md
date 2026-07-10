# Metrics reference — mathematics and interpretation

Every metric logged by the trainer, its exact formula, and how to read it.
This doubles as the paper appendix draft.

---

## 1. Core training metrics

| Metric | Formula | Notes |
|---|---|---|
| `train_loss` | mean CE over update window, averaged across replicas | smoothed over `log_every` steps |
| `val_loss` | Σ(loss·tokens) / Σ(tokens) over all replicas | **token-weighted**: replicas with fewer batches don't bias the mean |
| `val_ppl` | exp(val_loss) | capped at val_loss < 20 to avoid inf |
| `grad_norm` | ‖g‖₂ after cross-replica reduction, before clipping | reduce → clip → step ordering |

## 2. Behavioral probes

### `induction` — classic induction score
Synthetic sequence per batch row *i*: random filler tokens, then
`[Aᵢ][Bᵢ] … [Aᵢ] → ?` where Aᵢ, Bᵢ are unique token ids.

**Formula**: `induction = mean_i P(Bᵢ | context ending in second Aᵢ)` — the softmax probability the model assigns to the correct continuation.

**Read**: 1/vocab ≈ 0.00002 at random; induction heads typically push this to 0.1–0.9 during training. Measures *retrieval in silence* — filler is random and unattractive.

### `distractor_induction` and `distractor_margin` — retrieval under interference

Layout of each sequence:

```
...random... [A][B] ...  [C₁][D₁] [C₁][D₁]  [C₂][D₂] [C₂][D₂] ... [Cₙ][Dₙ] [Cₙ][Dₙ] ... [A] → ?
              target            n repeated distractor pairs                        query
```

Each distractor pair [Cⱼ][Dⱼ] appears **twice**. Why twice? Induction heads fire on *repeated* patterns: "I saw X before, what followed X?" A once-off random pair is ignored; a repeated pair becomes a genuine competing induction pattern — attractive noise, which is what your suppression mechanisms claim to handle.

At the query position (the second [A]) take the softmax distribution p(·):

- **`distractor_induction`** = `mean_i p(Bᵢ)` — probability of the *correct* answer with distractors present. Compare with plain `induction`: the gap between them is the damage done by interference.
- **`distractor_margin`** = `mean_i [ p(Bᵢ) − maxⱼ p(Dⱼ) ]` — the correct answer's probability minus the probability of the **strongest wrong candidate** (the distractor answer the model is most tempted by).

**Read**:
- margin > 0 — the true target beats every distractor (model retrieves B *despite* the noise);
- margin ≈ 0 — the model is confused between B and some Dⱼ (a greedy decode would be a coin flip);
- margin < 0 — a distractor *wins*: the model would answer Dⱼ instead of B. Retrieval hijacked by noise.

**Why it exists**: val loss averages over everything and can hide selective-attention gains. If HLA's salience/K-gates really suppress irrelevant-but-attractive keys, the HLA `distractor_margin` curve must rise faster than base. It is the *behavioral* half of the decoupling evidence; interference metrics (below) are the *structural* half.

## 3. Saturation metrics — "is the model hitting the walls?"

Every bounded nonlinearity in HLA is a `tanh`. For each one we log the fraction of positions where it is essentially pinned:

`X_sat_frac = mean( 𝟙[ |tanh(raw)| > 0.99 ] )`

| Metric | Bounded quantity |
|---|---|
| `angle_q_sat_frac`, `angle_k_sat_frac` | phase angles vs the π·phase_mult budget |
| `gate_k_sat_frac`, `gate_v_sat_frac` | K/V gates vs the range envelope |
| `salience_sat_frac` | salience gate vs ±salience_clip |

**Read**: ≈ 0 — the envelope is not the limiting factor (bounds are fine). Persistently > 0.1–0.2 — the model is pressing against the wall: widen `phase_mult` / ranges in the next run. This converts "did I limit the model too much?" from an opinion into a measurement.

## 4. Spectral (SVD) metrics — shape of the learned weights

For a matrix W with singular values s₁ ≥ s₂ ≥ … ≥ sₙ:

- **Effective rank** (`erank`): `exp( −Σ pᵢ log pᵢ )` where `pᵢ = sᵢ / Σs`. It is exp(entropy of the normalized spectrum): 1.0 if one direction dominates (rank-1 collapse), n if all directions contribute equally. *"How many directions does this matrix really use?"*
- **Top-1 share** (`top1`): `s₁ / Σs` ∈ (0,1]. Concentration; > 0.95 ⇒ effectively rank-1.
- **Stable rank**: `‖W‖²_F / s₁² = Σsᵢ² / s₁²`. A robust, differentiable lower bound on rank; insensitive to tiny singular values.

| Metric | Matrix | Question |
|---|---|---|
| `svd_phase_erank` | W_phase per head (C × hd/2) | Did the learned rotation collapse to a trivial rank-1 map, or does it use the full rotation subspace? |
| `svd_phase_top1` | same | concentration check for the above |
| `svd_qk_stable_rank` | Q and K blocks of c_attn (C × C each, averaged) | Does the **retrieval** circuit specialize (rank drop / divergence from base) during training? |
| `svd_v_stable_rank` | V block of c_attn | ...while the **content** circuit stays comparable to base? The pair of curves *is* the decoupling picture |
| `svd_gate_erank` | W_gate (H × C) | Do heads develop diverse gating directions or share one? |

Phase/gate metrics are **NaN at identity init** (all-zero matrices have no spectrum) — deliberately NaN, not 0, so plots show "not yet active" rather than a fake reading.

## 5. Head-interference metrics — Transformer-Circuits-style

Framework: each head *reads from* and *writes to* the shared residual stream (Elhage et al., 2021). Per layer, per head *h*, build orthonormal bases of the top-k singular directions (k = 8), all living in ℝ^C:

- `write_h` = column space of W_O[:, h] — *what head h writes into the stream*
- `qk_read_h` = row space of [W_Q; W_K] head block — *what h's retrieval circuit reads*
- `v_read_h` = row space of W_V head block — *what h's content circuit reads*

**Overlap** between two subspaces with orthonormal bases A (dim a) and B (dim b):

`overlap(A, B) = ‖AᵀB‖²_F / min(a, b) ∈ [0, 1]`

This equals the mean of cos²(principal angles) between the subspaces — the exact generalization of cosine similarity from vectors to subspaces: 0 = orthogonal (no interaction possible), 1 = one contains the other.

| Metric | Definition | Meaning |
|---|---|---|
| `qk_interference` | mean over i≠j of overlap(write_i, qk_read_j) | how much one head's *output* lands in another head's *retrieval input* = **noise floor for matching**. HLA predicts ↓ vs base |
| `ov_interference` | mean over i≠j of overlap(write_i, v_read_j) | write→content coupling = **legitimate composition channel** (virtual heads, induction circuits). Must NOT collapse — if it did we'd be destroying composition, not denoising |
| `qk_self_overlap`, `ov_self_overlap` | i = j diagonal | reference points |
| `qk_ov_separation` | ov_interference − qk_interference | **the headline number**: ↑ means retrieval got cleaner while composition survived |

Ground-truth tests: heads reading/writing disjoint coordinate blocks give interference ≈ 1e-15; identical heads give cross = self.

## 6. Learned-profile readouts (interpretability)

- `layer_temp_first/last/mean`: effective depth multiplier `1 + (l/L)·softplus(θ_l)/softplus(0)` at the first/last layer and its mean. At init: 1.0 and 2−1/L (the static heuristic). Divergence from the linear profile = the model disagrees with the heuristic.
- `phase_budget_mean/min/max`: distribution of per-head budgets `1 + tanh(s_h)` ∈ [0, 2]. All 1.0 at init. min → 0 means some head *opted out* of rotation entirely; max → 2 means some head doubled its budget. Scatter these against per-head interference for the mechanistic figure.

### Why softplus in the layer temperature?

`softplus(θ) = ln(1 + e^θ)` — a smooth, always-positive function: softplus(0) = ln 2 ≈ 0.693.

The multiplier is `mult_l = 1 + (l/L) · softplus(θ_l)/softplus(0)`:

1. **(l/L)** — depth fraction: 0 for the first layer, →1 for the last. This is the *shape* of the static heuristic (deeper ⇒ wider envelope).
2. **softplus(θ_l)/softplus(0)** — learned *amplitude* of that shape. At θ=0 the ratio is exactly **1**, so `mult = 1 + l/L` — bit-identical to the static heuristic ⇒ identity-init preserved.
3. softplus (not exp, not raw θ) because it is positive (multiplier can never flip sign or hit 0 catastrophically), smooth at 0 (well-behaved gradients from the first step), and grows gently.

So: the *heuristic provides the prior, θ_l learns how strongly each layer follows it* — including "less than the heuristic" (θ < 0 ⇒ ratio < 1).

## 7. Adaptivity map — what adapts along which axis

| Mechanism | Per-token (content) | Per-head | Per-layer (depth) |
|---|---|---|---|
| Phase rotation | ✓ W_phase·x | ✓ per_head_phase (budget) | ✓ layer_dependent_phase (**new**) |
| K/V Laplace gates | ✓ W_gate·x | ✓ W_range (learned range) | ✓ layer_dependent_gate (+ learnable_layer_temp) |
| Salience bias | ✓ W_sal·x | ✓ H outputs of W_sal | — (deliberate: salience is "what", not "where in depth") |
| Distance bias | ✓ via gate_k | ✓ via gate_k heads | ✓ layer multiplier |

Both the gates *and* the phase are now depth-adaptive; the phase reuses the **same** layer multiplier (including the learned temperature) so depth behavior stays consistent across mechanisms and adds zero extra parameters.

## 8. Perturbation-bound utilization (Theorem 4, live)

`perturbation_bounds(model)` - post-hoc probe on any checkpoint.

| Metric | Formula | Read |
|---|---|---|
| `L*_mix_k_max/min` | actual extremes of the captured K-mix tensor | must sit inside the envelope |
| `L*_theo_mix_k_max/min` | (1-b) + b*exp(+-eff), eff = lam*min(alpha*range*1.25, clip) | Theorem-4 envelope. NOTE: the binding term is alpha*range*1.25, not the clip - in every shipped config the clip has >=1.2x slack (audit E3) |
| `L*_util_k_up/down` | (max-1)/(theo_max-1) and (1-min)/(1-theo_min), in [0,1] | ~0 = dormant/loose; ->1 = pressing the envelope, widen ranges |

With `learnable_layer_temp` the effective lam is read from the live
softplus(theta) value, so the bound tracks the learned depth profile.

## 9. Mechanism gradient norms (Theorem 5, live)

Captured in-training at `svd_every` cadence: `GPT.capture_mechanism_grad_norms()`
snapshots ||grad W||_2 for all 11 mechanism parameters per layer AFTER
cross-replica reduction and clipping, BEFORE zero_grad (device tensors, no
host sync at capture time).

| Metric | Formula | Read |
|---|---|---|
| `L*_grad_{name}` | per-layer, per-param L2 norm | inactive mechanisms record exact 0 (grad is None) |
| `mech_grad_mean/min` (CSV) | mean/min over ACTIVE (>0) entries | a sustained slide of `mech_grad_min` toward 0 while train loss still moves = a mechanism is going silent; cross-check its `*_sat_frac` |

No target band is prescribed: mechanism/backbone gradient ratios depend on
parameter counts and layer depth; the paper reports trajectories, not thresholds.
