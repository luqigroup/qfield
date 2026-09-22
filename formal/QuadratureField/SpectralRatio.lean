import QuadratureField.StrictGain

/-!
# The spectral-concentration ratio `r_ρ`: range

Proposition 1 states `r_ρ = tr Σ²/tr Σ ∈ (0, 1 − c_ρ]`, with `Σ` the covariance
operator of the centred feature. In moment form (`bf-def-rrho`, the form used
in the proof) `(1 − c_ρ) r_ρ = ∬ k̄² dρ dρ =: rTimes` and `1 − c_ρ = ∫ ‖Φ̄‖² dρ`
(`tr Σ`). This file proves the range: `0 ≤ rTimes ≤ (1 − c_ρ)²`, so
`0 ≤ r_ρ ≤ 1 − c_ρ`, and `1 − c_ρ = ∫ ‖Φ̄‖²`. Strict positivity of `r_ρ` is in
`SpectralRatioPos.lean`, with no separability assumption on `H`.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] {H : Type*} [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]
variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ] (hΦ : StronglyMeasurable Φ)
  (hunit : ∀ x, ‖Φ x‖ = 1)
include hΦ hunit

/-- `tr Σ = ∫ ‖Φ̄‖² dρ = 1 − c_ρ` under a unit diagonal. -/
theorem integral_norm_centered_sq :
    ∫ x, ‖Φ x - kernelMean Φ ρ‖ ^ 2 ∂ρ = 1 - ‖kernelMean Φ ρ‖ ^ 2 := by
  set m := kernelMean Φ ρ with hm
  have hint : Integrable Φ ρ := integrable_feature Φ hΦ (fun x => (hunit x).le) ρ
  have h1 : ∀ x, ‖Φ x - m‖ ^ 2 = 1 - 2 * ⟪m, Φ x⟫_ℝ + ‖m‖ ^ 2 := by
    intro x
    rw [norm_sub_sq_real, hunit, real_inner_comm]
    ring
  simp_rw [h1]
  have hi : Integrable (fun x => 2 * ⟪m, Φ x⟫_ℝ) ρ :=
    (Integrable.of_bound (stronglyMeasurable_const.inner hΦ).aestronglyMeasurable (‖m‖)
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs]
        calc |⟪m, Φ x⟫_ℝ| ≤ ‖m‖ * ‖Φ x‖ := abs_real_inner_le_norm _ _
          _ = ‖m‖ := by rw [hunit, mul_one])).const_mul 2
  have e := integral_add ((integrable_const (1 : ℝ)).sub hi) (integrable_const (‖m‖ ^ 2)) (μ := ρ)
  simp only [Pi.sub_apply] at e
  rw [e, integral_sub (integrable_const _) hi, integral_const_mul, integral_inner hint m,
    integral_const, integral_const, probReal_univ, smul_eq_mul, one_mul, smul_eq_mul, one_mul]
  rw [hm]
  unfold kernelMean
  rw [real_inner_self_eq_norm_sq]
  ring

/-- `0 ≤ rTimes`. -/
theorem rTimes_nonneg : 0 ≤ rTimes Φ ρ :=
  integral_nonneg fun y => integral_nonneg fun x => sq_nonneg _

/-- `rTimes ≤ (1 − c_ρ)²`: Cauchy–Schwarz on the centred kernel. -/
theorem rTimes_le : rTimes Φ ρ ≤ (1 - ‖kernelMean Φ ρ‖ ^ 2) ^ 2 := by
  set m := kernelMean Φ ρ with hm
  have hcs : ∀ y x, ⟪Φ y - m, Φ x - m⟫_ℝ ^ 2 ≤ ‖Φ y - m‖ ^ 2 * ‖Φ x - m‖ ^ 2 := by
    intro y x
    have := abs_real_inner_le_norm (Φ y - m) (Φ x - m)
    calc ⟪Φ y - m, Φ x - m⟫_ℝ ^ 2 = |⟪Φ y - m, Φ x - m⟫_ℝ| ^ 2 := (sq_abs _).symm
      _ ≤ (‖Φ y - m‖ * ‖Φ x - m‖) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
      _ = ‖Φ y - m‖ ^ 2 * ‖Φ x - m‖ ^ 2 := by ring
  have hn2 : StronglyMeasurable (fun x => ‖Φ x - m‖ ^ 2) :=
    (hΦ.sub stronglyMeasurable_const).norm.pow 2
  have hn2b : ∀ x, |‖Φ x - m‖ ^ 2| ≤ 4 := by
    intro x
    rw [abs_of_nonneg (by positivity)]
    calc ‖Φ x - m‖ ^ 2 ≤ 2 ^ 2 := pow_le_pow_left₀ (norm_nonneg _) (norm_centered_le_two Φ ρ hunit x) 2
      _ = 4 := by norm_num
  have hn2i : Integrable (fun x => ‖Φ x - m‖ ^ 2) ρ :=
    Integrable.of_bound hn2.aestronglyMeasurable 4 (ae_of_all _ fun x => by
      rw [Real.norm_eq_abs]; exact hn2b x)
  have hγm := stronglyMeasurable_gamma Φ ρ hΦ hunit
  have hγb := abs_gamma_le Φ ρ hΦ hunit
  -- inner bound, for each y
  have hinner : ∀ y, ∫ x, ⟪Φ y - m, Φ x - m⟫_ℝ ^ 2 ∂ρ ≤ ‖Φ y - m‖ ^ 2 * ∫ x, ‖Φ x - m‖ ^ 2 ∂ρ := by
    intro y
    rw [← integral_const_mul]
    refine integral_mono ?_ (hn2i.const_mul _) (fun x => hcs y x)
    refine Integrable.of_bound ((hγm.comp_measurable (measurable_const.prodMk measurable_id)).pow 2).aestronglyMeasurable
      16 (ae_of_all _ fun x => ?_)
    rw [Real.norm_eq_abs, abs_pow]
    calc |⟪Φ y - m, Φ x - m⟫_ℝ| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) (hγb y x) 2
      _ = 16 := by norm_num
  -- outer
  have houter : rTimes Φ ρ ≤ ∫ y, ‖Φ y - m‖ ^ 2 * ∫ x, ‖Φ x - m‖ ^ 2 ∂ρ ∂ρ := by
    unfold rTimes
    refine integral_mono ?_ (hn2i.mul_const _) hinner
    have hFm : StronglyMeasurable (fun y => ∫ x, ⟪Φ y - m, Φ x - m⟫_ℝ ^ 2 ∂ρ) :=
      (hγm.pow 2).integral_prod_right'
    refine Integrable.of_bound hFm.aestronglyMeasurable 16 (ae_of_all _ fun y => ?_)
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 16)
      (f := fun x => ⟪Φ y - m, Φ x - m⟫_ℝ ^ 2) (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_pow]
        calc |⟪Φ y - m, Φ x - m⟫_ℝ| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) (hγb y x) 2
          _ = 16 := by norm_num)
    simpa [probReal_univ] using h
  rw [integral_mul_const, integral_norm_centered_sq Φ ρ hΦ hunit] at houter
  calc rTimes Φ ρ ≤ _ := houter
    _ = (1 - ‖kernelMean Φ ρ‖ ^ 2) ^ 2 := by ring

/-- The spectral-concentration ratio in moment form: `r_ρ = rTimes / (1 − c_ρ)`. -/
def spectralRatio : ℝ := rTimes Φ ρ / (1 - ‖kernelMean Φ ρ‖ ^ 2)

/-- **`r_ρ ≤ 1 − c_ρ`** (the upper half of the range in Proposition 1), when `c_ρ < 1`. -/
theorem spectralRatio_le (hc : ‖kernelMean Φ ρ‖ ^ 2 < 1) :
    spectralRatio Φ ρ ≤ 1 - ‖kernelMean Φ ρ‖ ^ 2 := by
  unfold spectralRatio
  rw [div_le_iff₀ (by linarith)]
  calc rTimes Φ ρ ≤ (1 - ‖kernelMean Φ ρ‖ ^ 2) ^ 2 := rTimes_le Φ ρ hΦ hunit
    _ = (1 - ‖kernelMean Φ ρ‖ ^ 2) * (1 - ‖kernelMean Φ ρ‖ ^ 2) := by ring

theorem spectralRatio_nonneg (hc : ‖kernelMean Φ ρ‖ ^ 2 < 1) : 0 ≤ spectralRatio Φ ρ :=
  div_nonneg (rTimes_nonneg Φ ρ hΦ hunit) (by linarith)

/-- Proposition 1's right side in the paper's variables: `β_M(ε) r_ρ (1 − c_ρ)/M` equals the
Lean statement's `(M−2)/((1+ε)M) · rTimes / M`. -/
theorem gain_bound_eq (hc : ‖kernelMean Φ ρ‖ ^ 2 < 1) (β M : ℝ) :
    β * spectralRatio Φ ρ * ((1 - ‖kernelMean Φ ρ‖ ^ 2) / M) = β * (rTimes Φ ρ / M) := by
  unfold spectralRatio
  have : (1 - ‖kernelMean Φ ρ‖ ^ 2) ≠ 0 := by linarith
  field_simp

end QuadratureField
