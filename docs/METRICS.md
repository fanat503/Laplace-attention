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

## 2b. Mechanism activity means (per-eval CSV columns)

Cheap per-layer means captured on the diagnostics forward and averaged over
layers — the "is it alive and how loud" counterpart to the saturation
fractions of §3:

| Metric | Quantity | Read |
|---|---|---|
| `angle_q_abs_mean`, `angle_k_abs_mean` | mean \|angle\| of the content phase rotation (radians) | 0 at identity; growth = the phase channel is being used. Compare against the π·phase_mult budget |
| `gate_k_mean`, `gate_v_mean` | mean raw tanh gate value in [−1, 1] | sign = net amplify/suppress tendency; magnitude = how decisively the gate votes |
| `mix_k_mean`, `mix_v_mean` | mean multiplicative mix applied to K/V | 1.0 at identity; sustained drift = net loudness change. Bounded by the Theorem-4 envelope (§8) |

## 3. Saturation metrics — "is the model hitting the walls?"

Every bounded nonlinearity in HLA is a `tanh`. For each one we log the fraction of positions where it is essentially pinned:

`X_sat_frac = mean( 𝟙[ |tanh(raw)| > 0.99 ] )`

| Metric | Bounded quantity |
|---|---|
| `angle_q_sat_frac`, `angle_k_sat_frac` | phase angles vs the π·phase_mult budget |
| `gate_k_sat_frac`, `gate_v_sat_frac` | K/V gates vs the range envelope |
| `salience_sat_frac` | salience gate vs ±salience_clip |
| `forget_sat_frac` | forget gate vs ±forget_clip (FoX arm only) |
| `qtemp_sat_frac` | Q-temperature gate vs ±qtemp_clip |

Related non-saturation readouts: `angle_q_std` / `angle_k_std` (spread of the
learned rotation angles — 0 at identity, growth = the phase channel actually
differentiates tokens) and `qtemp_mean` (mean per-query temperature multiplier;
1.0 at identity, drift ≠ 1 = queries learn individual sharpness).

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
| Distance bias | ✓ via its own gate W_gate_d (deconfounded from the K-gate) | ✓ H outputs of W_gate_d | ✓ layer multiplier |
| Q-temperature | ✓ W_qtemp·x (per query) | ✓ H outputs of W_qtemp | — |

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
snapshots ||grad W||_2 for all 13 mechanism parameters per layer (phase_q/k,
phase_scale, range_k/v/f, gate_k/v/sal/f/d, qtemp, layer_temp) AFTER
cross-replica reduction and clipping, BEFORE zero_grad (device tensors, no
host sync at capture time).

| Metric | Formula | Read |
|---|---|---|
| `L*_grad_{name}` | per-layer, per-param L2 norm | inactive mechanisms record exact 0 (grad is None) |
| `mech_grad_mean/min` (CSV) | mean/min over ACTIVE (>0) entries | a sustained slide of `mech_grad_min` toward 0 while train loss still moves = a mechanism is going silent; cross-check its `*_sat_frac` |

No target band is prescribed: mechanism/backbone gradient ratios depend on
parameter counts and layer depth; the paper reports trajectories, not thresholds.

## 10. Post-hoc causal & circuit probes (checkpoint-time, not logged in CSV)

All three are side-effect-free (verified by tests: weights, training mode and
diagnostics flags are restored bit-exactly).

### `mechanism_knockout(model, batch, targets)`

Causal attribution by inference-time silencing (Nanda-style ablation).
**Measurement tool only — never a deployment mode.** For each mechanism
(phase, gates, salience, distance, forget, qtemp) it temporarily sets the
alpha to 0 on a held-out batch, records `ko_<mech>_delta = loss_ko − loss_full`,
then restores the alpha.

- `delta > 0` — the mechanism carries trained load; `delta ≈ 0` — decorative
  or redundantly encoded.
- The **Δloss vs context-length curve** for the long-range mechanisms
  (distance / salience / forget) is the quantitative evidence for the
  long-context claim — that is precisely why one measures by silencing.
- Complements the training-time ablation arms: arms answer *"what if never
  trained with X"*; knockout answers *"what does X do in THIS trained model"*.

### `prefix_matching_score(model)`

Canonical induction-head detector (Olsson et al., 2022): attention weight
from the second `[A]` back to the token AFTER the first `[A]`, per head.
`induction` (§2) measures the *behavior* (P(B) at the output); this measures
the *mechanism* (attention pattern). Reported per-layer max/mean over heads +
global max — the standard way to chart emergence of induction circuitry.

### `gate_redundancy_statistics(model)`

Pairwise Pearson correlation between the four per-token content gates
(K, V, salience, distance), flattened over batch x tokens x heads, averaged
over layers: `gate_corr_kv`, `gate_corr_ksal`, ... plus the scalar
`gate_redundancy_mean_abs` (mean |off-diagonal|).

- ~0 - gates specialize (four mechanisms earn their parameters);
- |corr| -> 1 - redundant parameterization; the pre-registered rule R-B
  (EXPERIMENT_CARD) merges any pair with |corr| > 0.9 across seeds in v6.
- NaN at identity init (zero gates have no correlation structure) -
  deliberately NaN, like the SVD metrics.

This is the measurement-first answer to "why not an orthogonality loss?":
an aux loss would change the objective (breaking the same-loss sterility
invariant) and presume the answer. Ground-truth tested: cloned gates give
corr = 1.0, independent random gates stay below 0.9.

### `positional_recall_curve(model)`

Direct Lost-in-the-Middle probe (Liu et al., 2023): a needle pair [A][B] is
planted at depth fraction 10/30/50/70/90% of the context; the query [A] sits
at the end; report P(B) per depth. Scalars: `litm_middle_drop`
(mean-edge − worst-middle; 0 = flat, positive = the classic U-sag) and
`litm_worst_frac` (worst-middle / best-edge; 1.0 = no sag). The pre-registered
long-context reading: HLA's distance/salience mechanisms should FLATTEN this
curve relative to the sterile base twin. NaN for vocabularies too small for
the needle token block (deliberate, like the SVD metrics).

### `attention_head_similarity(model)`

Mean pairwise Jensen–Shannon divergence between per-head attention
distributions at matched positions (bounded by ln 2). Low JS = redundant
heads attending alike; rising JS in the HLA run = heads differentiating —
the attention-space counterpart of the weight-space interference metrics (§5).
