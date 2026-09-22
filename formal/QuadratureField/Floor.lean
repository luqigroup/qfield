import QuadratureField.Setting

/-!
# Display (2): the floor identity, and its generalization

Paper §3.2: for `M` independent samples of the reference and equal weights,
`E MMD²(Q₀, ρ) = (1 − c_ρ)/M` when `k(x,x) = 1`.

Here the identity is proved in a **lighter** form (`floor_identity`):
`E MMD²(Q₀, ρ) = (E k(X,X) − c_ρ)/M` for any bounded, strongly measurable
feature map, with no unit diagonal; the paper's display is the corollary
`floor_identity_unitDiag`. The general form is what the paper's §7 would need
for a Stein kernel (`k(x,x)` not constant): the floor identity does travel,
with `E k(X,X)` in place of `1`.

The sample is modelled as `M` measurable maps `ξ i : Ω → X` that are jointly
independent (`iIndepFun`) and identically distributed with law `ρ`; the kernel
mean is the Bochner integral `m = ∫ Φ dρ` and `c_ρ = ‖m‖² = ∬ k dρ dρ`.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] {H : Type*} [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]

/-- The kernel mean of `ρ` as a Bochner integral. -/
def kernelMean (Φ : X → H) (ρ : Measure X) : H := ∫ x, Φ x ∂ρ

/-- Bounded strongly measurable features are integrable on a finite measure. -/
lemma integrable_feature (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    (ρ : Measure X) [IsFiniteMeasure ρ] : Integrable Φ ρ :=
  Integrable.of_bound hΦ.aestronglyMeasurable B (ae_of_all _ hB)

/-- The kernel-mean function `μ_ρ(x) = ∫ k(x, y) dρ(y)` is `⟪Φ x, m⟫`: the
reproducing property of the mean element. -/
lemma inner_kernelMean (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    (ρ : Measure X) [IsFiniteMeasure ρ] (v : H) :
    ∫ y, ⟪v, Φ y⟫_ℝ ∂ρ = ⟪v, kernelMean Φ ρ⟫_ℝ :=
  integral_inner (integrable_feature Φ hΦ hB ρ) v

/-- `c_ρ = ‖m‖² = ∬ k dρ dρ`. -/
lemma selfAffinity_eq (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    (ρ : Measure X) [IsFiniteMeasure ρ] :
    ∫ x, ∫ y, kernel Φ x y ∂ρ ∂ρ = ‖kernelMean Φ ρ‖ ^ 2 := by
  have h1 : ∀ x, ∫ y, kernel Φ x y ∂ρ = ⟪Φ x, kernelMean Φ ρ⟫_ℝ := fun x =>
    inner_kernelMean Φ hΦ hB ρ (Φ x)
  simp_rw [h1]
  have h2 : ∫ x, ⟪Φ x, kernelMean Φ ρ⟫_ℝ ∂ρ = ∫ x, ⟪kernelMean Φ ρ, Φ x⟫_ℝ ∂ρ := by
    simp_rw [real_inner_comm]
  rw [h2, inner_kernelMean Φ hΦ hB ρ, real_inner_self_eq_norm_sq]

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- A bounded strongly measurable function of a measurable map is integrable. -/
lemma integrable_comp_of_bound {Y : Type*} [NormedAddCommGroup Y] {ξ : Ω → X} (hξ : Measurable ξ)
    {f : X → Y} (hf : StronglyMeasurable f) {C : ℝ} (hC : ∀ x, ‖f x‖ ≤ C) :
    Integrable (fun ω => f (ξ ω)) P :=
  Integrable.of_bound (hf.comp_measurable hξ).aestronglyMeasurable C (ae_of_all _ fun ω => hC (ξ ω))

/-- Change of variables to the law of one sample. -/
lemma integral_comp_law {Y : Type*} [NormedAddCommGroup Y] [NormedSpace ℝ Y] {ξ : Ω → X}
    (hξ : Measurable ξ) {ρ : Measure X} (hlaw : P.map ξ = ρ) {f : X → Y}
    (hf : StronglyMeasurable f) : ∫ ω, f (ξ ω) ∂P = ∫ x, f x ∂ρ := by
  rw [← hlaw, integral_map hξ.aemeasurable hf.aestronglyMeasurable]

/-- **Independent samples decorrelate the features:** for `i ≠ j`,
`E ⟪Φ(X_i), Φ(X_j)⟫ = ‖m‖² = c_ρ`. -/
lemma integral_inner_of_indep (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ}
    (hB : ∀ x, ‖Φ x‖ ≤ B) {ρ : Measure X} [IsProbabilityMeasure ρ] {ξ ξ' : Ω → X}
    (hξ : Measurable ξ) (hξ' : Measurable ξ') (hind : IndepFun ξ ξ' P)
    (hlaw : P.map ξ = ρ) (hlaw' : P.map ξ' = ρ) :
    ∫ ω, ⟪Φ (ξ ω), Φ (ξ' ω)⟫_ℝ ∂P = ‖kernelMean Φ ρ‖ ^ 2 := by
  set f : X × X → ℝ := fun p => ⟪Φ p.1, Φ p.2⟫_ℝ with hf_def
  have hf : StronglyMeasurable f :=
    (hΦ.comp_measurable measurable_fst).inner (hΦ.comp_measurable measurable_snd)
  have hpair : Measurable fun ω => (ξ ω, ξ' ω) := hξ.prodMk hξ'
  have hmap : P.map (fun ω => (ξ ω, ξ' ω)) = (P.map ξ).prod (P.map ξ') :=
    (indepFun_iff_map_prod_eq_prod_map_map hξ.aemeasurable hξ'.aemeasurable).mp hind
  have h1 : ∫ ω, ⟪Φ (ξ ω), Φ (ξ' ω)⟫_ℝ ∂P = ∫ p, f p ∂(P.map fun ω => (ξ ω, ξ' ω)) := by
    rw [integral_map hpair.aemeasurable hf.aestronglyMeasurable]
  rw [h1, hmap, hlaw, hlaw']
  have hint : Integrable f (ρ.prod ρ) := by
    refine Integrable.of_bound hf.aestronglyMeasurable (B * B) (ae_of_all _ fun p => ?_)
    calc ‖f p‖ = |⟪Φ p.1, Φ p.2⟫_ℝ| := rfl
      _ ≤ ‖Φ p.1‖ * ‖Φ p.2‖ := abs_real_inner_le_norm _ _
      _ ≤ B * B :=
        mul_le_mul (hB p.1) (hB p.2) (norm_nonneg _) ((norm_nonneg (Φ p.1)).trans (hB p.1))
  rw [integral_prod f hint]
  have hinner : ∀ x, ∫ y, f (x, y) ∂ρ = ⟪Φ x, kernelMean Φ ρ⟫_ℝ := fun x =>
    inner_kernelMean Φ hΦ hB ρ (Φ x)
  simp_rw [hinner]
  have h2 : ∫ x, ⟪Φ x, kernelMean Φ ρ⟫_ℝ ∂ρ = ∫ x, ⟪kernelMean Φ ρ, Φ x⟫_ℝ ∂ρ := by
    simp_rw [real_inner_comm]
  rw [h2, inner_kernelMean Φ hΦ hB ρ, real_inner_self_eq_norm_sq]

/-- `E ⟪Φ(X_j), m⟫ = ‖m‖²`. -/
lemma integral_inner_mean (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ}
    (hB : ∀ x, ‖Φ x‖ ≤ B) {ρ : Measure X} [IsProbabilityMeasure ρ] {ξ : Ω → X}
    (hξ : Measurable ξ) (hlaw : P.map ξ = ρ) :
    ∫ ω, ⟪Φ (ξ ω), kernelMean Φ ρ⟫_ℝ ∂P = ‖kernelMean Φ ρ‖ ^ 2 := by
  have hg : StronglyMeasurable fun x => ⟪Φ x, kernelMean Φ ρ⟫_ℝ :=
    hΦ.inner stronglyMeasurable_const
  rw [integral_comp_law hξ hlaw hg]
  have h2 : ∫ x, ⟪Φ x, kernelMean Φ ρ⟫_ℝ ∂ρ = ∫ x, ⟪kernelMean Φ ρ, Φ x⟫_ℝ ∂ρ := by
    simp_rw [real_inner_comm]
  rw [h2, inner_kernelMean Φ hΦ hB ρ, real_inner_self_eq_norm_sq]

/-- `E ‖Φ(X_j)‖² = ∫ k(x,x) dρ`. -/
lemma integral_norm_sq_feature (Φ : X → H) (hΦ : StronglyMeasurable Φ) {ρ : Measure X}
    {ξ : Ω → X} (hξ : Measurable ξ) (hlaw : P.map ξ = ρ) :
    ∫ ω, ‖Φ (ξ ω)‖ ^ 2 ∂P = ∫ x, ‖Φ x‖ ^ 2 ∂ρ := by
  have hg : StronglyMeasurable fun x => ‖Φ x‖ ^ 2 := hΦ.norm.pow 2
  exact integral_comp_law hξ hlaw hg

/-- The pairwise feature moments of an i.i.d. sample: `E k(X_i, X_j)` is `E k(X,X)` on
the diagonal and `c_ρ` off it. -/
lemma pair_moment (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    {ρ : Measure X} [IsProbabilityMeasure ρ] {M : ℕ} {ξ : Fin M → Ω → X}
    (hξ : ∀ i, Measurable (ξ i)) (hind : iIndepFun ξ P) (hlaw : ∀ i, P.map (ξ i) = ρ)
    (i j : Fin M) :
    ∫ ω, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ ∂P =
      ‖kernelMean Φ ρ‖ ^ 2 +
        if i = j then (∫ x, ‖Φ x‖ ^ 2 ∂ρ) - ‖kernelMean Φ ρ‖ ^ 2 else 0 := by
  by_cases hij : i = j
  · subst hij
    simp only [if_true]
    have : ∀ ω, ⟪Φ (ξ i ω), Φ (ξ i ω)⟫_ℝ = ‖Φ (ξ i ω)‖ ^ 2 := fun ω =>
      real_inner_self_eq_norm_sq _
    simp_rw [this]
    rw [integral_norm_sq_feature Φ hΦ (hξ i) (hlaw i)]
    ring
  · simp only [hij, if_false, add_zero]
    exact integral_inner_of_indep Φ hΦ hB (hξ i) (hξ j) (hind.indepFun hij) (hlaw i) (hlaw j)

/-- **The floor identity, general form.** For `M ≥ 1` i.i.d. samples and equal weights,
`E MMD²(Q₀, ρ) = (E k(X,X) − c_ρ)/M`. No unit diagonal is assumed. -/
theorem floor_identity (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    {ρ : Measure X} [IsProbabilityMeasure ρ] {M : ℕ} (hM : 0 < M) {ξ : Fin M → Ω → X}
    (hξ : ∀ i, Measurable (ξ i)) (hind : iIndepFun ξ P) (hlaw : ∀ i, P.map (ξ i) = ρ) :
    ∫ ω, mmdSq Φ (fun j => ξ j ω) (wEq M) (kernelMean Φ ρ) ∂P =
      ((∫ x, ‖Φ x‖ ^ 2 ∂ρ) - ‖kernelMean Φ ρ‖ ^ 2) / M := by
  set m := kernelMean Φ ρ with hm
  set κ := ∫ x, ‖Φ x‖ ^ 2 ∂ρ with hκ
  set c := ‖m‖ ^ 2 with hc
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  -- pointwise expansion
  have hpt : ∀ ω, mmdSq Φ (fun j => ξ j ω) (wEq M) m =
      (1 / (M : ℝ)) ^ 2 * ∑ i, ∑ j, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ
        - 2 * (1 / (M : ℝ)) * ∑ j, ⟪Φ (ξ j ω), m⟫_ℝ + c := by
    intro ω
    rw [mmdSq_expand]
    simp only [wEq, kernel]
    have e1 : ∑ i, ∑ j, 1 / (M : ℝ) * (1 / (M : ℝ)) * ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ
        = (1 / (M : ℝ)) ^ 2 * ∑ i, ∑ j, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      ring
    have e2 : 2 * ∑ j, 1 / (M : ℝ) * ⟪Φ (ξ j ω), m⟫_ℝ
        = 2 * (1 / (M : ℝ)) * ∑ j, ⟪Φ (ξ j ω), m⟫_ℝ := by
      simp only [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      ring
    rw [e1, e2]
  -- integrability of the pieces
  have hint_pair : ∀ i j, Integrable (fun ω => ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ) P := by
    intro i j
    refine Integrable.of_bound ((hΦ.comp_measurable (hξ i)).inner
      (hΦ.comp_measurable (hξ j))).aestronglyMeasurable (B * B) (ae_of_all _ fun ω => ?_)
    calc ‖⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ‖ = |⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ| := rfl
      _ ≤ ‖Φ (ξ i ω)‖ * ‖Φ (ξ j ω)‖ := abs_real_inner_le_norm _ _
      _ ≤ B * B := mul_le_mul (hB (ξ i ω)) (hB (ξ j ω)) (norm_nonneg _)
          ((norm_nonneg (Φ (ξ i ω))).trans (hB (ξ i ω)))
  have hint_mean : ∀ j, Integrable (fun ω => ⟪Φ (ξ j ω), m⟫_ℝ) P := by
    intro j
    refine Integrable.of_bound ((hΦ.comp_measurable (hξ j)).inner
      stronglyMeasurable_const).aestronglyMeasurable (B * ‖m‖) (ae_of_all _ fun ω => ?_)
    calc ‖⟪Φ (ξ j ω), m⟫_ℝ‖ = |⟪Φ (ξ j ω), m⟫_ℝ| := rfl
      _ ≤ ‖Φ (ξ j ω)‖ * ‖m‖ := abs_real_inner_le_norm _ _
      _ ≤ B * ‖m‖ := mul_le_mul_of_nonneg_right (hB _) (norm_nonneg _)
  have hint_dsum : Integrable (fun ω => ∑ i, ∑ j, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ) P :=
    integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hint_pair i j
  have hint_msum : Integrable (fun ω => ∑ j, ⟪Φ (ξ j ω), m⟫_ℝ) P :=
    integrable_finsetSum _ fun j _ => hint_mean j
  have hf1 : Integrable (fun ω => (1 / (M : ℝ)) ^ 2 * ∑ i, ∑ j, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ) P :=
    hint_dsum.const_mul _
  have hf2 : Integrable (fun ω => 2 * (1 / (M : ℝ)) * ∑ j, ⟪Φ (ξ j ω), m⟫_ℝ) P :=
    hint_msum.const_mul _
  have hdouble : ∫ ω, ∑ i, ∑ j, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ ∂P
      = ∑ i, ∑ j, ∫ ω, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ ∂P := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hint_pair i j]
    refine Finset.sum_congr rfl fun i _ => ?_
    exact integral_finsetSum _ fun j _ => hint_pair i j
  have hsingle : ∫ ω, ∑ j, ⟪Φ (ξ j ω), m⟫_ℝ ∂P = ∑ j, ∫ ω, ⟪Φ (ξ j ω), m⟫_ℝ ∂P :=
    integral_finsetSum _ fun j _ => hint_mean j
  -- integrate the expansion
  simp_rw [hpt]
  have e := integral_add (hf1.sub hf2) (integrable_const c) (μ := P)
  simp only [Pi.sub_apply] at e
  rw [e, integral_sub hf1 hf2, integral_const_mul, integral_const_mul, hdouble, hsingle]
  simp only [integral_const, probReal_univ, smul_eq_mul, one_mul]
  -- the moments
  have hpair : ∀ i j, ∫ ω, ⟪Φ (ξ i ω), Φ (ξ j ω)⟫_ℝ ∂P = c + if i = j then κ - c else 0 :=
    fun i j => pair_moment Φ hΦ hB hξ hind hlaw i j
  have hmean : ∀ j, ∫ ω, ⟪Φ (ξ j ω), m⟫_ℝ ∂P = c := fun j =>
    integral_inner_mean Φ hΦ hB (hξ j) (hlaw j)
  simp_rw [hpair, hmean]
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  field_simp
  ring

/-- **Display (2) of the paper.** With a unit-diagonal kernel, `E MMD²(Q₀, ρ) = (1 − c_ρ)/M`. -/
theorem floor_identity_unitDiag (Φ : X → H) (hΦ : StronglyMeasurable Φ)
    (hunit : ∀ x, ‖Φ x‖ = 1) {ρ : Measure X} [IsProbabilityMeasure ρ] {M : ℕ} (hM : 0 < M)
    {ξ : Fin M → Ω → X} (hξ : ∀ i, Measurable (ξ i)) (hind : iIndepFun ξ P)
    (hlaw : ∀ i, P.map (ξ i) = ρ) :
    ∫ ω, mmdSq Φ (fun j => ξ j ω) (wEq M) (kernelMean Φ ρ) ∂P =
      (1 - ‖kernelMean Φ ρ‖ ^ 2) / M := by
  have hB : ∀ x, ‖Φ x‖ ≤ 1 := fun x => (hunit x).le
  rw [floor_identity Φ hΦ hB hM hξ hind hlaw]
  have : ∫ x, ‖Φ x‖ ^ 2 ∂ρ = 1 := by
    simp_rw [hunit, one_pow]
    simp
  rw [this]

/-- The self-affinity is at most `E k(X,X)` (the variance of `‖Φ(X)‖` is nonnegative); with a
unit diagonal, `c_ρ ≤ 1`, so the floor is nonnegative. -/
theorem selfAffinity_le (Φ : X → H) (hΦ : StronglyMeasurable Φ) {B : ℝ} (hB : ∀ x, ‖Φ x‖ ≤ B)
    (ρ : Measure X) [IsProbabilityMeasure ρ] :
    ‖kernelMean Φ ρ‖ ^ 2 ≤ ∫ x, ‖Φ x‖ ^ 2 ∂ρ := by
  have hint := integrable_feature Φ hΦ hB ρ
  have hnorm : Integrable (fun x => ‖Φ x‖) ρ := hint.norm
  have hsq : Integrable (fun x => ‖Φ x‖ ^ 2) ρ :=
    Integrable.of_bound (hΦ.norm.pow 2).aestronglyMeasurable (B ^ 2)
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
        exact pow_le_pow_left₀ (norm_nonneg _) (hB x) 2)
  set a := ∫ x, ‖Φ x‖ ∂ρ with ha
  have h1 : ‖kernelMean Φ ρ‖ ≤ a := norm_integral_le_integral_norm _
  have h2 : a ^ 2 ≤ ∫ x, ‖Φ x‖ ^ 2 ∂ρ := by
    have hvar : 0 ≤ ∫ x, (‖Φ x‖ - a) ^ 2 ∂ρ := integral_nonneg fun x => sq_nonneg _
    have hexp : ∫ x, (‖Φ x‖ - a) ^ 2 ∂ρ = (∫ x, ‖Φ x‖ ^ 2 ∂ρ) - a ^ 2 := by
      have : ∀ x, (‖Φ x‖ - a) ^ 2 = ‖Φ x‖ ^ 2 - 2 * a * ‖Φ x‖ + a ^ 2 := fun x => by ring
      simp_rw [this]
      have hf2 : Integrable (fun x => 2 * a * ‖Φ x‖) ρ := hnorm.const_mul (2 * a)
      have e := integral_add (hsq.sub hf2) (integrable_const (a ^ 2)) (μ := ρ)
      simp only [Pi.sub_apply] at e
      rw [e, integral_sub hsq hf2, integral_const_mul, integral_const, probReal_univ,
        smul_eq_mul, one_mul]
      ring
    linarith
  calc ‖kernelMean Φ ρ‖ ^ 2 ≤ a ^ 2 := by gcongr
    _ ≤ _ := h2

end QuadratureField
