# Lean 4 formalization of the paper's theoretical claims

Machine-checked Lean 4 (`mathlib`, pinned to `v4.31.0`) formalizations of the
theoretical statements of *Amortized quadrature for posterior expectations in
inverse problems* (`paper/main_v9.tex`, §3–§5 and the two formal results the
appendix carries in full).

```bash
cd formal
lake exe cache get     # prebuilt mathlib oleans for the pinned toolchain
lake build             # kernel-checks every module (no sorry, no admit, no native_decide)
lake env lean Axioms.lean   # prints the axioms of every main theorem
```

Every theorem below is kernel-verified with **no `sorry`**: `Axioms.lean` prints,
for each, only Lean's three standard axioms (`propext`, `Classical.choice`,
`Quot.sound`). Nothing domain-specific is axiomatized. The one imported result
of the paper, the empirical Bernstein inequality of Maurer & Pontil (2009),
enters Theorem 1(ii) as an **explicit hypothesis on the concentration events**
(`certified_selection_of_events`), exactly as the paper imports it; an
unconditional instance with Hoeffding's inequality (from mathlib) is proved
beside it (`hoeffding_pair_event`).

## The setting

`Setting.lean` models the RKHS abstractly: a real Hilbert space `H`, a feature
map `Φ : X → H` (so `k(x,y) = ⟪Φ x, Φ y⟫`, which is every positive-semidefinite
kernel), a quadrature as nodes `z : Fin M → X` with signed weights
`w : Fin M → ℝ`, its embedding `∑ wⱼ Φ(zⱼ)`, and the squared discrepancy to a
reference with kernel mean `m ∈ H` as `‖emb − m‖²`. When a measure enters, the
kernel mean is the Bochner integral `∫ Φ dρ` (`Floor.lean`), and `M` i.i.d.
samples are the coordinates of the product measure `ρ^M` (`ProductSpace.lean`).
Coincident nodes are allowed everywhere; no distinctness or invertibility
hypothesis appears anywhere.

## Paper statement → Lean theorem

| Paper | Lean (file : theorem) | Hypotheses as proved | Notes |
|---|---|---|---|
| Display (1), the discrepancy expansion | `Setting : mmdSq_expand` | any feature-map kernel | Gram term, kernel-mean term, self-affinity |
| "worst case over the unit ball is the discrepancy" (cited) | `Setting : worstCase_error` | — | attained, as an `IsGreatest` |
| `c_ρ` cancels; the criterion equals `MMD² − c_ρ` under the exact kernel mean | `Setting : crit_exact` | — | |
| Remark 1 (reference fence) | `Setting : mmd_triangle` | — | triangle inequality |
| Display (2), the floor identity | `Floor : floor_identity_unitDiag` | `Φ` strongly measurable, `‖Φ‖ = 1`, `ξᵢ` i.i.d. `ρ`, `M ≥ 1` | **general form** `floor_identity`: `(E k(X,X) − c_ρ)/M` with no unit diagonal |
| `c_ρ ≤ 1` | `Floor : selfAffinity_le` | bounded `Φ` | `c_ρ ≤ E k(X,X)` |
| §4.2 the closed-form weights (two solves, unit sum, the minimizer) | `Weights : deployed_weights`, `closedForm_isMin`, `ridgeMin_unique` | `ε > 0`, any Gram, any estimand | KKT certificate `quadObj_min_of_kkt` |
| §4.4 "the samples never strictly win the criterion against their reweighting" | `Weights : twoArm_chain` | **any** `K`, **any** estimand, `ε ≥ 0` | `Ĵ(w_rw) ≤ Ĵ(w_eq) − ε(‖w_rw‖² − 1/M) ≤ Ĵ(w_eq)` |
| Theorem A.1's ridged chain (exact kernel mean) | `Weights : mmdSq_reweight_le` | exact kernel mean | |
| Theorem 1(i), exact selection never worse than the samples | `ExactSelection : exact_selection`, `exact_selection_isLeast`, `neverWorse_exact` | exact kernel mean; **any** nodes, any weights, any candidate family | no hypothesis beyond positive semidefiniteness; no unit diagonal, no i.i.d. sampling |
| the untrained-network degenerate case of 1(i) | `ExactSelection : neverWorse_exact_untrained` | | |
| the engine identity `cs-eq-engine` | `Certification : engine_identity`, `paired_diff` | any split | |
| the witness bound `cs-eq-supg` | `Certification : witness_bound_of_bounded` | `sup ‖Φ‖ ≤ κ` | the bounded-kernel hypothesis of §7 |
| `D²` as the cross-Gram form | `Certification : distSq_gram_form` | | |
| Theorem 1(ii)(b): one-slack excess and certified sign | `Certification : selection_excess_of_conc`, `certified_sign_of_conc`, `certified_selection_exact` | on the concentration event | |
| the slack is identically zero for coincident candidates | `Certification : slack_zero_of_coincide` | | the untrained-network comment |
| Theorem 1(ii) with three candidates at `δ/3` (`cs-cor-menu`(ii)), both displays | `CertificationProb : certified_selection_of_events`, `certified_selection_min` | **the concentration events are a hypothesis** (supplied by Maurer–Pontil in the paper) | union bound and selection logic checked |
| Theorem 1(ii), unconditional instance | `CertificationProb : hoeffding_pair_event` | i.i.d. split, bounded kernel | slack `2Dκ√(2 ln(2/δ)/L)` from mathlib's Hoeffding |
| Lemma S2, the one-pair gain | `PairGain : pair_gain`, `pair_gain_paper` | `ε ≥ 0`; `k ≥ 0` for the constant `2 + 2ε` | general constant `b + 2ε` for any `‖Φ zᵢ − Φ zⱼ‖² ≤ b` |
| Lemmas S3–S4 | `StrictGainMoments : inner_second_moment`; `StrictGain : triple_moment`, `pair_mean_moment` | | |
| **Proposition 1** (sharper form, `M ≥ 2`) | `StrictGain : strict_gain_sharper` | `k` PSD, unit diagonal, `k ≥ 0`; `ε ≥ 0`; i.i.d. seeds; exact kernel mean; `z ↦ MMD²(Q_wε(z))` a.e. strongly measurable | `(4 Var_ρ(μ_ρ) + (M−2)(1−c_ρ)r_ρ)/((1+ε)M²)` |
| **Proposition 1** as printed (`M ≥ 3`) | `StrictGain : strict_gain` | same | `β_M(ε) r_ρ (1−c_ρ)/M` with `β_M(ε) = (M−2)/((1+ε)M)` |
| `r_ρ ≤ 1 − c_ρ` | `SpectralRatio : spectralRatio_le` | `c_ρ < 1` | `rTimes ≤ (1 − c_ρ)²` |
| `r_ρ > 0` | `SpectralRatioPos : spectralRatio_pos` | `c_ρ < 1`, `Φ` strongly measurable | **no separability of `H` needed** (the fragment assumed it) |
| Proposition 1, second half: no target-free constant | `NoTargetFree : no_target_free_constant`, `expected_gain_le` | the discrete kernel `k(i,j) = 1{i=j}` on `ℕ`, `ρ_n` uniform on `n` atoms | see the note below |

`r_ρ` is defined in its moment form `(1 − c_ρ) r_ρ = ∬ k̄² dρ dρ` (`rTimes`), the
form the proof uses; the operator form `tr Σ²/tr Σ` is the same number by the Hilbert–Schmidt
identity `tr(a ⊗ b) = ⟪b, a⟫` for the covariance operator, and is not needed
anywhere.

## What is checked more generally than the paper states

* **The floor identity travels.** `floor_identity` gives
  `E MMD²(Q₀, ρ) = (E k(X,X) − c_ρ)/M` for any bounded feature map; the paper's
  display is the unit-diagonal case. This is the form a Stein kernel needs.
* **Theorem 1(i) needs nothing about the kernel** beyond positive
  semidefiniteness (being a feature-map kernel): no unit diagonal, no `k ≥ 0`, no
  i.i.d. sampling, any finite candidate family, any nodes.
* **The two-arm inequality holds for any matrix `K` and any estimand vector**,
  not only Grams and kernel means.
* **`r_ρ > 0`** uses only the strong measurability of `Φ`, not a separable `H`.

## What is not machine-checked, and why

* **The empirical Bernstein inequality** (Maurer & Pontil 2009, Theorem 4) is
  the paper's imported result. Theorem 1(ii) is checked modulo it: the events
  it supplies are a hypothesis of `certified_selection_of_events`. Everything the
  paper adds — the engine identity, the pairing, the witness bounds, the union
  bound, the selection logic, the degenerate cases — is checked. The Hoeffding
  instance `hoeffding_pair_event` is checked unconditionally.
* **The squared-exponential instance of the impossibility half.** The paper's
  sentence names the squared-exponential kernel; the fragment spreads atoms
  until the SE features are nearly orthogonal. What is checked here is the same
  collapse for the kernel of the class with exactly orthogonal atom features
  (the discrete kernel on `ℕ`): it is positive semidefinite, has unit diagonal
  and is nonnegative, so it lies in Proposition 1's class, and it shows that no
  constant depending only on `M` and the kernel can replace `r_ρ`. Extending
  the machine check to the SE kernel needs an explicit SE feature map into
  `ℓ²` and the lattice Gershgorin bounds of the fragment; it is not done.
* **Measurability of the deployed minimizer map** `z ↦ w_ε(z)` is assumed
  (`hmeas` in `strict_gain`, `hint` in `expected_gain_le`) rather than derived
  from the closed form's continuity; every other hypothesis of Proposition 1 is
  the paper's.

## Module map

| Module | Content |
|---|---|
| `Setting` | features, kernel, quadratures, display (1), worst case, `crit_exact`, Remark 1 |
| `Weights` | ridged constrained minimizer, two-arm inequality, KKT, closed form, uniqueness |
| `ExactSelection` | Theorem 1(i) |
| `Floor` | kernel mean as a Bochner integral, the floor identity (general and unit-diagonal), `c_ρ ≤ E k(X,X)` |
| `Certification` | engine identity, witness bounds, selection logic on the concentration event, degenerate case |
| `CertificationProb` | Theorem 1(ii) modulo the concentration inequality; the Hoeffding instance |
| `PairGain` | Lemma S2 |
| `ProductSpace` | the i.i.d. sample as a product measure: coordinate laws, pair laws, peeling |
| `StrictGainAlgebra`, `StrictGainMoments`, `StrictGain` | Lemmas S3–S4 and Proposition 1 |
| `SpectralRatio`, `SpectralRatioPos` | `0 < r_ρ ≤ 1 − c_ρ` |
| `NoTargetFree` | the impossibility half, discrete kernel |
| `Axioms.lean` | the axiom audit |

## Where each theorem appears in the paper

Appendix A of `paper/main_v9.tex` (`paper/v9_appendix_proofs.tex`) states and
proves every result in the form checked here, and each appendix statement
carries a `% lean: …` source comment naming its theorem; in print the paper says
only that the formal results are machine-checked in Lean 4 (`tests/test_lean_tags.py`
keeps the comments and `Axioms.lean` in agreement both ways). The correspondence: A.1 the
setting (`Setting`), A.2 the floor identity (`Floor`), A.3 the weights and the
two-arm inequality (`Weights`), A.4 Theorem 1(i) (`ExactSelection`), A.5
Proposition 1 with the range of `r_ρ` and the no-target-free constant
(`PairGain`, `StrictGainAlgebra`, `StrictGainMoments`, `StrictGain`,
`SpectralRatio`, `SpectralRatioPos`, `NoTargetFree`), A.6 Theorem 1(ii)
(`Certification`, `CertificationProb`). The body's §3.2 carries the general
floor identity as a sentence, Proposition 1 defines `r_ρ` in the moment form
used here and states the impossibility half for the discrete kernel, and §5.2's
comments carry the Hoeffding form of the slack.
