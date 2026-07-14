# Experiment card — pre-registration

This file is the pre-registered experimental plan. It is committed BEFORE the
headline runs; any deviation must be documented in the "Deviations" section
with a date and reason. Purpose: protection against post-hoc configuration
selection (see docs/STERILITY.md, "Selection effects").

## Primary hypothesis
H1: The full HLA mechanism set (phase + K/V gates + salience + distance),
trained from a sterile shared init against a parameter-matched ablated base,
achieves lower validation loss at matched tokens, and the gap does not shrink
with model scale over 200M -> 300M -> 700M.
(The Q-temperature mechanism ships OFF in the primary HLA recipe; it is
evaluated as its own single-factor arm `qtemp` in the ablation matrix and
joins the full recipe only if that arm shows a positive effect — a
pre-registered decision, not post-hoc selection.)

Secondary (mechanistic) hypotheses:
H2: qk_interference decreases vs base while ov_interference is preserved
    (qk_ov_separation increases).
H3: distractor_margin improves faster than base during training.

## Active mechanism sets per config (B6: capacity vs default)

The codebase implements SEVEN mechanisms; shipped training configs activate a
deliberate SUBSET. "Seven mechanisms" describes model capacity, not the
default treatment arm - stated here explicitly so no reader infers that the
headline number used everything at once.

| Config family | Active in HLA arm | Off (identity, parameter-matched) |
|---|---|---|
| 200m v1 (`200m_hla_s42`) | phase, K/V gates, distance | salience, forget, qtemp, adaptivity extras |
| 200m v2 (`200m_hla_v2_s42`, primary) | phase, K/V gates, distance, salience | forget, qtemp, adaptivity extras |
| Ablation matrix arms | exactly one mechanism per single-factor arm | everything else |
| `forget` arm | forget only (FoX baseline) | all HLA mechanisms |
| `qtemp` arm | qtemp only | all others |

## Pre-registered decisions (locked before headline runs)
| Decision | Value | Rationale |
|---|---|---|
| Primary config | `200m_hla_v2_s42` recipe (aggressive envelope + salience) | v1 recipe's multiplicative floor caps suppression at x0.77 (see CONFIG_AUDIT) |
| Primary metric | token-weighted final val loss; tie-breaker: val loss at matched wall-clock | |
| Seeds | 42, 43, 44 (add 45, 46 if gap < 5x seed std) | seed-noise band measured at ~0.002 init loss |
| Statistical test | paired t-test across seeds (same seed = same init pair) | paired by construction |
| Ablation matrix | 14 arms x 3 seeds via `make_ablation_configs.py` | single-factor discipline enforced by tests |
| Hyperparameters | tuned on BASE only (standard recipes); never adjusted per-arm | fairness invariant I5 |
| Exclusion rule | a run is excluded only for infrastructure failure (crash, data corruption), never for its result; exclusions logged here | |

## Run ladder (in order; each gates the next)
1. smoke (10 steps, both variants) — infrastructure alive
2. pilot (1000 steps) — loss curves sane, diagnostics populated
3. 200M v1 pair, seed 42 — reproduce the historical v3/v4 gap in the sterile harness
4. 200M v2 pair, seed 42 — primary recipe first reading
5. 200M ablation matrix (14 arms x 3 seeds)
6. 300M FLOPs-matched pair (3 seeds)
7. 700M pair (>= 2 seeds, budget permitting)
8. Downstream evals (lm-eval-harness) on best checkpoints
9. Headline scale (compute-dependent; only after 5-7 confirm the trend)

## Reporting commitments
- Report parameter-matched AND FLOPs-matched comparisons where both exist.
- Report wall-clock tokens/sec for both variants (mechanism overhead honesty).
- Report all seeds (no seed selection), mean +- std, and the paired test.
- Negative/flat results at any rung are reported, not hidden.

## Deviations
| Date | Deviation | Reason |
|---|---|---|
| (none yet) | | |
