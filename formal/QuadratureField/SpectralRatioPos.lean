import QuadratureField.SpectralRatio

/-!
# Strict positivity of the spectral-concentration ratio

Proposition 1 states `r_ρ ∈ (0, 1 − c_ρ]`; `SpectralRatio.lean` gave the upper
half. Here: `rTimes = ∬ k̄² dρ dρ > 0` whenever `c_ρ < 1`, so `r_ρ > 0`. The
fragment proves this through the Hilbert–Schmidt norm of the covariance
operator in a separable RKHS. The proof below needs **no separability of `H`**,
only the strong measurability of `Φ` already assumed everywhere (its range is
then separable): with `C h = ∫ ⟪h, Φ̄⟫ Φ̄ dρ` the covariance operator,
`rTimes = 0` forces `C Φ̄(x) = 0` for a.e. `x`, hence `C h = 0` for every `h`,
hence `⟪h, Φ̄(x)⟫ = 0` a.e. for every `h`; running `h` over a countable dense
subset of the range gives `Φ̄ = 0` a.e., i.e. `1 − c_ρ = ∫ ‖Φ̄‖² = 0`.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory TopologicalSpace

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] {H : Type*} [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]
variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ] (hΦ : StronglyMeasurable Φ)
  (hunit : ∀ x, ‖Φ x‖ = 1)

/-- The covariance operator of the centred feature, applied to `h`. -/
def covOp (h : H) : H := ∫ y, ⟪h, Φ y - kernelMean Φ ρ⟫_ℝ • (Φ y - kernelMean Φ ρ) ∂ρ

include hΦ hunit

lemma integrable_cov_integrand (h : H) :
    Integrable (fun y => ⟪h, Φ y - kernelMean Φ ρ⟫_ℝ • (Φ y - kernelMean Φ ρ)) ρ := by
  refine Integrable.of_bound ((stronglyMeasurable_const.inner (hΦ.sub stronglyMeasurable_const)).smul
    (hΦ.sub stronglyMeasurable_const)).aestronglyMeasurable (‖h‖ * 2 * 2) (ae_of_all _ fun y => ?_)
  rw [norm_smul, Real.norm_eq_abs]
  calc |⟪h, Φ y - kernelMean Φ ρ⟫_ℝ| * ‖Φ y - kernelMean Φ ρ‖
      ≤ (‖h‖ * ‖Φ y - kernelMean Φ ρ‖) * ‖Φ y - kernelMean Φ ρ‖ :=
        mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
    _ ≤ (‖h‖ * 2) * 2 := by
        gcongr
        · exact norm_centered_le_two Φ ρ hunit y
        · exact norm_centered_le_two Φ ρ hunit y

lemma integrable_inner_mul (a h : H) :
    Integrable (fun y => ⟪h, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪a, Φ y - kernelMean Φ ρ⟫_ℝ) ρ := by
  refine Integrable.of_bound ((stronglyMeasurable_const.inner (hΦ.sub stronglyMeasurable_const)).mul
    (stronglyMeasurable_const.inner (hΦ.sub stronglyMeasurable_const))).aestronglyMeasurable
    ((‖h‖ * 2) * (‖a‖ * 2)) (ae_of_all _ fun y => ?_)
  rw [Real.norm_eq_abs, abs_mul]
  refine mul_le_mul ?_ ?_ (abs_nonneg _) (by positivity)
  · exact (abs_real_inner_le_norm _ _).trans
      (mul_le_mul_of_nonneg_left (norm_centered_le_two Φ ρ hunit y) (norm_nonneg _))
  · exact (abs_real_inner_le_norm _ _).trans
      (mul_le_mul_of_nonneg_left (norm_centered_le_two Φ ρ hunit y) (norm_nonneg _))

/-- `⟪a, C h⟫ = ∫ ⟪h, Φ̄⟫ ⟪a, Φ̄⟫ dρ`. -/
lemma inner_covOp (a h : H) :
    ⟪a, covOp Φ ρ h⟫_ℝ = ∫ y, ⟪h, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪a, Φ y - kernelMean Φ ρ⟫_ℝ ∂ρ := by
  unfold covOp
  rw [← integral_inner (integrable_cov_integrand Φ ρ hΦ hunit h) a]
  simp_rw [real_inner_smul_right]

lemma covOp_symm (a h : H) : ⟪a, covOp Φ ρ h⟫_ℝ = ⟪h, covOp Φ ρ a⟫_ℝ := by
  rw [inner_covOp Φ ρ hΦ hunit, inner_covOp Φ ρ hΦ hunit]
  congr 1
  funext y
  ring

lemma covOp_nonneg (h : H) : 0 ≤ ⟪h, covOp Φ ρ h⟫_ℝ := by
  rw [inner_covOp Φ ρ hΦ hunit]
  exact integral_nonneg fun y => mul_self_nonneg _

/-- The quadratic form of `C` along a line. -/
lemma quad_line (a v : H) (t : ℝ) :
    ⟪a + t • v, covOp Φ ρ (a + t • v)⟫_ℝ =
      ⟪a, covOp Φ ρ a⟫_ℝ + 2 * t * ⟪a, covOp Φ ρ v⟫_ℝ + t ^ 2 * ⟪v, covOp Φ ρ v⟫_ℝ := by
  simp only [inner_covOp Φ ρ hΦ hunit]
  have hpt : ∀ y, ⟪a + t • v, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪a + t • v, Φ y - kernelMean Φ ρ⟫_ℝ
      = ⟪a, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪a, Φ y - kernelMean Φ ρ⟫_ℝ
        + (2 * t) * (⟪v, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪a, Φ y - kernelMean Φ ρ⟫_ℝ)
        + t ^ 2 * (⟪v, Φ y - kernelMean Φ ρ⟫_ℝ * ⟪v, Φ y - kernelMean Φ ρ⟫_ℝ) := by
    intro y
    rw [inner_add_left, real_inner_smul_left]
    ring
  simp_rw [hpt]
  have e1 := integral_add (μ := ρ) ((integrable_inner_mul Φ ρ hΦ hunit a a).add
      ((integrable_inner_mul Φ ρ hΦ hunit a v).const_mul (2 * t)))
      ((integrable_inner_mul Φ ρ hΦ hunit v v).const_mul (t ^ 2))
  try simp only [Pi.add_apply] at e1
  have e2 := integral_add (μ := ρ) (integrable_inner_mul Φ ρ hΦ hunit a a)
      ((integrable_inner_mul Φ ρ hΦ hunit a v).const_mul (2 * t))
  try simp only [Pi.add_apply] at e2
  rw [e1, e2, integral_const_mul, integral_const_mul]

/-- A vector with vanishing quadratic form is in the kernel of `C`. -/
lemma covOp_eq_zero_of_quad_zero (v : H) (hv : ⟪v, covOp Φ ρ v⟫_ℝ = 0) : covOp Φ ρ v = 0 := by
  have key : ∀ a : H, ⟪a, covOp Φ ρ v⟫_ℝ = 0 := by
    intro a
    by_contra hne
    set s := ⟪a, covOp Φ ρ v⟫_ℝ with hs
    set qa := ⟪a, covOp Φ ρ a⟫_ℝ with hqa
    -- choose t so that qa + 2 t s = -1
    have hs : s ≠ 0 := hne
    have h := covOp_nonneg Φ ρ hΦ hunit (a + (-(qa + 1) / (2 * s)) • v)
    rw [quad_line Φ ρ hΦ hunit, hv] at h
    have hval : qa + 2 * (-(qa + 1) / (2 * s)) * s + (-(qa + 1) / (2 * s)) ^ 2 * 0 = -1 := by
      rw [mul_zero, add_zero]
      field_simp
      ring
    linarith
  have h := key (covOp Φ ρ v)
  rw [real_inner_self_eq_norm_sq] at h
  exact norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp h)

/-- **`r_ρ > 0` when `c_ρ < 1`** (the lower half of the range in Proposition 1). -/
theorem rTimes_pos (hc : ‖kernelMean Φ ρ‖ ^ 2 < 1) : 0 < rTimes Φ ρ := by
  set m := kernelMean Φ ρ with hm
  by_contra hle
  push_neg at hle
  have hz : rTimes Φ ρ = 0 := le_antisymm hle (rTimes_nonneg Φ ρ hΦ hunit)
  have hγm := stronglyMeasurable_gamma Φ ρ hΦ hunit
  have hγb := abs_gamma_le Φ ρ hΦ hunit
  -- Step 1: the inner second moment vanishes a.e.
  have hq_int : Integrable (fun x => ∫ y, ⟪Φ x - m, Φ y - m⟫_ℝ ^ 2 ∂ρ) ρ := by
    have hFm : StronglyMeasurable (fun x => ∫ y, ⟪Φ x - m, Φ y - m⟫_ℝ ^ 2 ∂ρ) :=
      (hγm.pow 2).integral_prod_right'
    refine Integrable.of_bound hFm.aestronglyMeasurable 16 (ae_of_all _ fun x => ?_)
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 16)
      (f := fun y => ⟪Φ x - m, Φ y - m⟫_ℝ ^ 2) (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow]
        calc |⟪Φ x - m, Φ y - m⟫_ℝ| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) (hγb x y) 2
          _ = 16 := by norm_num)
    simpa [probReal_univ] using h
  have h1 : ∀ᵐ x ∂ρ, ∫ y, ⟪Φ x - m, Φ y - m⟫_ℝ ^ 2 ∂ρ = 0 := by
    have := (integral_eq_zero_iff_of_nonneg (fun x => integral_nonneg fun y => sq_nonneg _)
      hq_int).mp hz
    filter_upwards [this] with x hx using hx
  -- Step 2: `C Φ̄(x) = 0` a.e.
  have h2 : ∀ᵐ x ∂ρ, covOp Φ ρ (Φ x - m) = 0 := by
    filter_upwards [h1] with x hx
    refine covOp_eq_zero_of_quad_zero Φ ρ hΦ hunit _ ?_
    rw [inner_covOp Φ ρ hΦ hunit]
    simpa [sq] using hx
  -- Step 3: `C h = 0` for every `h`
  have h3 : ∀ h : H, covOp Φ ρ h = 0 := by
    intro h
    have hself : ⟪covOp Φ ρ h, covOp Φ ρ h⟫_ℝ =
        ∫ x, ⟪h, Φ x - m⟫_ℝ * ⟪covOp Φ ρ h, Φ x - m⟫_ℝ ∂ρ := inner_covOp Φ ρ hΦ hunit _ h
    have hzero : ∫ x, ⟪h, Φ x - m⟫_ℝ * ⟪covOp Φ ρ h, Φ x - m⟫_ℝ ∂ρ = 0 := by
      apply integral_eq_zero_of_ae
      filter_upwards [h2] with x hx
      rw [real_inner_comm (Φ x - m) (covOp Φ ρ h), covOp_symm Φ ρ hΦ hunit, hx, inner_zero_right,
        mul_zero, Pi.zero_apply]
    rw [hzero, real_inner_self_eq_norm_sq] at hself
    exact norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp hself)
  -- Step 4: every `h` is orthogonal to `Φ̄(x)` for a.e. `x`
  have h4 : ∀ h : H, ∀ᵐ x ∂ρ, ⟪h, Φ x - m⟫_ℝ = 0 := by
    intro h
    have hq : ∫ x, ⟪h, Φ x - m⟫_ℝ * ⟪h, Φ x - m⟫_ℝ ∂ρ = 0 := by
      rw [← inner_covOp Φ ρ hΦ hunit, h3, inner_zero_right]
    have := (integral_eq_zero_iff_of_nonneg (fun x => mul_self_nonneg _)
      (integrable_inner_mul Φ ρ hΦ hunit h h)).mp hq
    filter_upwards [this] with x hx
    exact mul_self_eq_zero.mp hx
  -- Step 5: a countable dense subset of the range of `Φ`
  obtain ⟨T, hTc, hT⟩ := hΦ.isSeparable_range
  have h5 : ∀ᵐ x ∂ρ, ∀ t ∈ T, ⟪t - m, Φ x - m⟫_ℝ = 0 :=
    (ae_ball_iff hTc).mpr fun t _ => h4 (t - m)
  -- Step 6: hence `Φ̄(x) = 0` a.e.
  have h6 : ∀ᵐ x ∂ρ, ‖Φ x - m‖ ^ 2 = 0 := by
    filter_upwards [h5] with x hx
    have hclosed : IsClosed {v : H | ⟪v - m, Φ x - m⟫_ℝ = 0} :=
      isClosed_eq ((continuous_id.sub continuous_const).inner continuous_const) continuous_const
    have hsub : T ⊆ {v : H | ⟪v - m, Φ x - m⟫_ℝ = 0} := fun t ht => hx t ht
    have hmem : Φ x ∈ closure T := hT ⟨x, rfl⟩
    have := closure_minimal hsub hclosed hmem
    simp only [Set.mem_setOf_eq] at this
    rwa [real_inner_self_eq_norm_sq] at this
  -- Step 7: contradiction with `1 − c_ρ = ∫ ‖Φ̄‖² > 0`
  have : ∫ x, ‖Φ x - m‖ ^ 2 ∂ρ = 0 := integral_eq_zero_of_ae h6
  rw [hm, integral_norm_centered_sq Φ ρ hΦ hunit] at this
  linarith

/-- **`r_ρ ∈ (0, 1 − c_ρ]`** whenever `c_ρ < 1` — Proposition 1's range, in full. -/
theorem spectralRatio_pos (hc : ‖kernelMean Φ ρ‖ ^ 2 < 1) : 0 < spectralRatio Φ ρ :=
  div_pos (rTimes_pos Φ ρ hΦ hunit hc) (by linarith)

end QuadratureField
