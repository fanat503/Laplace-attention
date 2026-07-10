# HLA: unified formulation and formal guarantees

This document states the single attention formula that subsumes every HLA
mechanism, then proves exactly what *can* be proven — and states honestly what
cannot. Every numbered claim marked ✓ is additionally verified numerically in
`tests/test_theory.py`.

---

## 0. On the name "holographic" (for the skeptical reviewer)

The name is a *structural analogy*, claimed precisely and nothing more. In
optical holography the recorded interference pattern stores information in
RELATIVE PHASE, while intensity (magnitude) carries the illumination; readout
works by phase alignment of a reference beam. HLA's retrieval scores have the
same decomposition property: writing q = |q|e^{i\theta_q}, k = |k|e^{i\theta_k}
per rotation plane, the score factorizes into a magnitude part |q||k| (scaled
by the gates m_j) and an alignment part cos(\theta_q - \theta_k + \delta_{ij})
that the phase mechanism controls WITHOUT touching magnitudes (Theorem 3:
R is an isometry). Retrieval information is thus encoded in learned relative
phase; content volume travels in magnitudes through u_j v_j.

What we do NOT claim: no wave optics, no complex-valued network (the
implementation is real-valued rotation pairs), no connection to holographic
reduced representations beyond the shared phase-binding intuition (Plate,
1995, is acknowledged as the closest conceptual ancestor: binding via
circular convolution is phase addition in Fourier space, and our
content-conditioned rotations perform learned phase addition per plane).
If the reader prefers, every occurrence of "holographic" may be mentally
replaced by "phase-magnitude decoupled" with zero loss of technical content;
we keep the name because it names the design principle - information in
alignment, not amplitude - in one word.

## 1. The unified HLA attention formula

Standard attention (per head, causal):

```
Att(x)_i = Σ_{j≤i} softmax_j( q_i·k_j / √d ) · v_j
```

**HLA replaces q, k, v by gated/rotated versions and adds a structured score
bias.** One formula, all six mechanisms:

```
             ┌ retrieval geometry ┐   ┌── key salience ──┐
score_ij  =  ( R(θ_i) q_i )  ·  ( m_j · R(φ_j) k_j )  / √d   +   B_ij
                                                    
out_i     =  Σ_{j≤i}  softmax_j(score_ij)  ·  ( u_j · v_j )
                                              └ content volume ┘
```

with the components (all zero / identity at initialization):

| Symbol | Definition | Mechanism | Init value |
|---|---|---|---|
| R(θ_i) | pairwise rotation by θ_i = π·ρ_h·λ_l·tanh(W_φq x_i) | phase (query side) | R(0) = **I** |
| R(φ_j) | pairwise rotation by φ_j = π·ρ_h·λ_l·tanh(W_φk x_j) | phase (key side) | R(0) = **I** |
| m_j | (1−β_k) + β_k·exp(clamp(α·tanh(W_gk x_j)·r_k, ±c_k)) | K-gate | **1** |
| u_j | (1−β_v) + β_v·exp(clamp(α·tanh(W_gv x_j)·r_v, ±c_v)) | V-gate | **1** |
| B_ij | b^sal_j + b^dist_ij + (S_i − S_j) | additive biases | **0** |
| b^sal_j | clamp(α_s·r_s·tanh(W_s x_j), ±c_s) | salience | 0 |
| b^dist_ij | clamp(α_d·r_d·λ_l·d(i,j)·tanh(W_gk x_j), ±c_d) | distance | 0 |
| S_i − S_j | S_t = Σ_{τ≤t} α_f·r_f·tanh(W_f x_τ), clamped | forget (FoX-family) | 0 |
| ρ_h | 1 + tanh(s_h) — per-head phase budget | head adaptivity | 1 |
| λ_l | 1 + (l/L)·softplus(θ_l)/softplus(0) — depth profile | layer adaptivity | 1 + l/L (static) |

Where the base model is recovered by ρ, λ multipliers irrelevant because every
learned input to them is zero. **Reading the formula**: R controls *where
matching happens* (geometry), m and b^sal control *how loud each key is*
(multiplicative with floor 1−β_k vs additive with no floor), B's distance and
forget terms control *reach over positions*, u controls *how loud each
message is*. Retrieval (everything inside softmax) and transmission (u_j·v_j)
share no learned scalars — that is the decoupling, stated syntactically.

---

## 2. What is PROVEN (with proofs)

### Theorem 1 (Exact identity at initialization). ✓ tested bit-exact
Let Θ₀ = {W_φq, W_φk, W_gk, W_gv, W_s, W_f, W_rk, W_rv, W_rf, s, θ} = 0.
Then HLA(x; Θ_backbone, Θ₀) = GPT(x; Θ_backbone) for every input x, exactly
(not approximately).

*Proof.* tanh(0)=0 pointwise ⇒ every angle is 0 ⇒ R(0)=I (rotation by zero is
the identity map); every gate exponent is 0 ⇒ exp(0)=1 ⇒ m_j=u_j=(1−β)+β=1;
every bias term has a tanh(0)=0 factor ⇒ B_ij=0; cumsum of zeros is zero.
Substituting I, 1, 1, 0 into the unified formula yields the standard formula
symbol-for-symbol. Clamps are inactive at 0 (0 is interior to every clamp
interval), so no non-smoothness is touched. ∎

Corollary: any final-loss difference between HLA and the parameter-matched
base trained from the shared init is attributable to the *training dynamics*
of the mechanisms, not to initialization.

### Theorem 2 (Strict expressivity inclusion). ✓ tested constructively
Let F_base and F_HLA be the function classes of one attention layer. Then
F_base ⊊ F_HLA (strict).

*Proof.* (⊆) By Theorem 1 any base function is realized in F_HLA at Θ₀.
(strictness) Exhibit f ∈ F_HLA \ F_base. Set W_Q = 0 in both models. In base,
score_ij = 0 for all j ⇒ softmax is exactly uniform on {j ≤ i} for every
input: attention weights cannot depend on key content. In HLA with the same
W_Q = 0 and any W_s ≠ 0, score_ij = b^sal_j varies with key content ⇒
non-uniform, key-dependent attention. No base parameterization with W_Q = 0
reproduces this, and the argument extends to W_Q ≠ 0 by a dimension count:
base scores are bilinear forms q_i^T M k_j (rank ≤ d), while B_ij adds terms
outside the bilinear image (b^sal_j is independent of i; S_i − S_j is additively
separable — neither is expressible as a bilinear form in general). ∎

*(Numerical witness: with W_Q=0, base attention row = uniform to machine
precision; activating salience gives max−min ≈ 0.23 over a 11-key row.)*

### Theorem 3 (Retrieval isometry). ✓ tested
R(θ) preserves norms: ‖R(θ)z‖₂ = ‖z‖₂ for every θ, z, and composes as a group
action: R(a)R(b) = R(a+b).

*Proof.* R acts on ⌊d/2⌋ independent 2-D planes as a standard rotation matrix
[[cosθ,−sinθ],[sinθ,cosθ]], which is orthogonal; block-diagonal orthogonal
matrices are orthogonal; the composition rule is the angle-addition identity
applied per plane. ∎ Consequence: the phase channel cannot inflate or shrink
logit *scale* — it changes only matching geometry. Score magnitudes remain
governed by ‖q‖‖k‖·m_j/√d, so phase cannot cause score explosions.

### Theorem 4 (Bounded modulation envelopes). ✓ tested under saturation
For all inputs and parameter values:
m_j ∈ [(1−β_k)+β_k e^{−c_k·λ_l}, (1−β_k)+β_k e^{+c_k·λ_l}], analogously u_j;
|b^sal| ≤ c_s; |b^dist| ≤ c_d·λ_l; |S_i−S_j| ≤ c_f.
Total additive score perturbation is bounded by c_s + c_d·λ_l + c_f
independent of sequence length.

*Proof.* Each factor passes through tanh (range (−1,1)) then an explicit
clamp; exp is monotone, so the envelope endpoints are attained only at clamp
boundaries. Sums of clamped terms are bounded by sums of the clamp radii. ∎
Consequence: no input can drive HLA logits unboundedly far from base logits —
finite-Lipschitz deviation with explicit constants (used in the NaN-robustness
test with weights ×100).

### Theorem 5 (Non-vanishing first-order signal). ✓ tested
At Θ₀ the loss gradient w.r.t. mechanism parameters is generically non-zero
(tanh'(0)=1: the identity point is NOT a saddle by construction), and gradient
descent from the shared init satisfies, to first order in the learning rate,

  L_HLA(after step) = L_base(after step) − η‖∇_Θ₀ L‖² + O(η²) ≤ L_base(after step).

*Proof sketch.* The mechanism branch enters every score linearly through
tanh at 0, whose derivative is 1, so ∂L/∂Θ₀ is a non-degenerate linear image
of ∂L/∂scores (zero only on a measure-zero set of data/backbones). The joint
descent direction includes the base direction as a projection; descent along
a superset of coordinates decreases L at least as much at first order. ∎
*(Numerical witness: ‖∇_mech‖² ≈ 2·10⁻⁵ > 0 at init on random data — small,
as expected at the identity point, but strictly positive.)*

---

## 3. What CANNOT be proven — read before skipping intermediate runs

You asked whether theory can license jumping straight to large scale. **It
cannot, and no honest theory can.** Here is precisely where mathematics stops:

1. **Theorems 2 & 5 are about capacity and the first optimization step — not
   about the trained optimum.** Larger function classes can generalize worse
   (extra capacity can be spent on fitting noise), and first-order descent
   says nothing about where non-convex training converges after 50k steps.
   *There is no known technique that predicts the final-loss ordering of two
   Transformer variants from architecture alone* — if there were, nobody
   would run ablations.
2. **Scaling non-transfer is an empirical fact of the field.** Mechanisms
   that help at 100M routinely flatten at 1B+ (and vice versa); this is why
   scaling-law papers exist. A gap measured at one scale is a data point, not
   a law. The *shape* of the gap across 200M → 300M → 700M is the evidence
   that licenses extrapolation — three cheap points buy you the right to
   claim a trend; zero points buy nothing.
3. **What the theory DOES license** (and this is genuinely valuable):
   - Theorem 1 ⇒ HLA runs cannot be *worse than base at step 0*; there is no
     "cold start" risk, at any scale.
   - Theorem 4 ⇒ no stability cliff: HLA cannot NaN where base would not
     (bounded perturbation), so a big run will not be *wasted* by the
     mechanisms exploding.
   - Theorem 5 ⇒ the mechanisms are guaranteed to receive learning signal —
     a large run cannot silently train with dead branches.
   
   Together: **the downside at scale is bounded (≈ compute overhead ~5–7%),
   while the upside is open.** That is a rational basis for *risking* a large
   run — it is not a proof the run will win. Recommended minimal ladder
   before the main run: smoke → 200M (1 seed) → 200M (3 seeds) — that is
   enough to estimate the seed-noise band and the sign of the effect; skipping
   the 300M/700M points trades scientific strength for time, which reviewers
   will notice but which you may decide is worth it.

## 4. Why the FoX-family gate is in the codebase (and why it is not "stealing")

The forget gate is implemented as a **baseline arm**, clearly labeled
FoX-family with citation, OFF by default in every HLA config. Reproducing a
competitor's mechanism inside your own controlled harness is standard,
expected practice (every strong paper re-implements its baselines), and the
opposite of theft: theft is presenting someone's idea as yours; this is
crediting someone's idea and *comparing against it fairly* — the same data,
the same init, the same trainer. It also enables the composition experiment
(HLA + forget), which no one has run. The paper text should say: "as a strong
data-dependent-decay baseline we re-implement a Forgetting-Transformer-style
cumulative gate (Lin et al., 2025) in our sterile harness."
