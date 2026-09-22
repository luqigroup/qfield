import QuadratureField.ProductSpace
import QuadratureField.StrictGainAlgebra

/-!
# The second moments behind Proposition 1 (Lemmas S3–S4)

Three moment computations over the i.i.d. seed law, all by Fubini on the
product measure:

* `inner_second_moment` — for a bounded `β` with `∫ β dρ = 0`,
  `E[((1/M) ∑_j β(w_j) − 2Δ/M)²] = (n/M²) ∫ β² + (4/M²) Δ²` over `w ~ ρ^n`
  (the conditional second-moment identity, given the pair);
* `pair_diff_sq_integral` — `E[(α(x₀) − α(x₁))²] = 2 ∫ α² − 2 (∫ α)²`;
* `integral_swap_of_bound` — Fubini for bounded measurable integrands, used to
  move the third sample outside.

These are pure measure-theoretic identities; nothing about kernels enters
beyond boundedness and measurability.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] (ρ : Measure X) [IsProbabilityMeasure ρ]

/-- Fubini for bounded strongly measurable integrands on `ρ ⊗ ρ`. -/
lemma integral_swap_of_bound {G : X → X → ℝ} (hG : StronglyMeasurable (Function.uncurry G))
    {C : ℝ} (hC : ∀ x y, |G x y| ≤ C) :
    ∫ x, ∫ y, G x y ∂ρ ∂ρ = ∫ y, ∫ x, G x y ∂ρ ∂ρ := by
  apply integral_integral_swap
  exact Integrable.of_bound hG.aestronglyMeasurable C (ae_of_all _ fun p => hC p.1 p.2)

/-- `E[(α(x₀) − α(x₁))²] = 2 ∫ α² − 2 (∫ α)²` for a bounded measurable `α`. -/
lemma pair_diff_sq_integral {α : X → ℝ} (hα : StronglyMeasurable α) {C : ℝ}
    (hC : ∀ x, |α x| ≤ C) :
    ∫ x₀, ∫ x₁, (α x₀ - α x₁) ^ 2 ∂ρ ∂ρ = 2 * (∫ x, α x ^ 2 ∂ρ) - 2 * (∫ x, α x ∂ρ) ^ 2 := by
  have hint : Integrable α ρ := Integrable.of_bound hα.aestronglyMeasurable C (ae_of_all _ hC)
  have hsq : Integrable (fun x => α x ^ 2) ρ :=
    Integrable.of_bound (hα.pow 2).aestronglyMeasurable (C ^ 2) (ae_of_all _ fun x => by
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_pow_left₀ (abs_nonneg _) (hC x) 2)
  have hinner : ∀ x₀, ∫ x₁, (α x₀ - α x₁) ^ 2 ∂ρ
      = α x₀ ^ 2 - 2 * α x₀ * (∫ x, α x ∂ρ) + ∫ x, α x ^ 2 ∂ρ := by
    intro x₀
    have : ∀ x₁, (α x₀ - α x₁) ^ 2 = α x₀ ^ 2 - 2 * α x₀ * α x₁ + α x₁ ^ 2 := fun x₁ => by ring
    simp_rw [this]
    have h1 : Integrable (fun x₁ => 2 * α x₀ * α x₁) ρ := hint.const_mul _
    have e := integral_add ((integrable_const (α x₀ ^ 2)).sub h1) hsq (μ := ρ)
    simp only [Pi.sub_apply] at e
    rw [e, integral_sub (integrable_const _) h1, integral_const_mul, integral_const,
      probReal_univ, smul_eq_mul, one_mul]
  simp_rw [hinner]
  have h2 : Integrable (fun x₀ => 2 * α x₀ * (∫ x, α x ∂ρ)) ρ := by
    have := hint.const_mul (2 * ∫ x, α x ∂ρ)
    refine this.congr (ae_of_all _ fun x => ?_)
    ring
  have e := integral_add (hsq.sub h2) (integrable_const (∫ x, α x ^ 2 ∂ρ)) (μ := ρ)
  simp only [Pi.sub_apply] at e
  rw [e, integral_sub hsq h2, integral_const, probReal_univ, smul_eq_mul, one_mul]
  have h3 : ∫ x₀, 2 * α x₀ * (∫ x, α x ∂ρ) ∂ρ = 2 * (∫ x, α x ∂ρ) ^ 2 := by
    have : (fun x₀ => 2 * α x₀ * (∫ x, α x ∂ρ)) = fun x₀ => (2 * ∫ x, α x ∂ρ) * α x₀ := by
      funext x₀; ring
    rw [this, integral_const_mul]; ring
  rw [h3]; ring

/-- **The conditional second moment** (Lemma S3): over `w ~ ρ^n`, for a bounded
measurable `β` with `∫ β dρ = 0` and constants `M ≠ 0`, `Δ`,
`E[((1/M) ∑_j β(w_j) − (2/M) Δ)²] = (n/M²) ∫ β² + (4/M²) Δ²`. -/
theorem inner_second_moment {n : ℕ} {β : X → ℝ} (hβ : StronglyMeasurable β) {C : ℝ}
    (hC : ∀ x, |β x| ≤ C) (hzero : ∫ x, β x ∂ρ = 0) {M : ℝ} (hM : M ≠ 0) (Δ : ℝ) :
    ∫ w, ((1 / M) * ∑ j, β (w j) - (2 / M) * Δ) ^ 2 ∂(piρ ρ n) =
      (n / M ^ 2) * (∫ x, β x ^ 2 ∂ρ) + (4 / M ^ 2) * Δ ^ 2 := by
  -- integrability of the pieces
  have hβj : ∀ j : Fin n, Integrable (fun w => β (w j)) (piρ ρ n) :=
    fun j => integrable_coord ρ hβ hC j
  have hβjl : ∀ j l : Fin n, Integrable (fun w => β (w j) * β (w l)) (piρ ρ n) := by
    intro j l
    refine Integrable.of_bound ((hβ.comp_measurable (measurable_pi_apply j)).mul
      (hβ.comp_measurable (measurable_pi_apply l))).aestronglyMeasurable (C * C)
      (ae_of_all _ fun w => ?_)
    rw [Real.norm_eq_abs, abs_mul]
    exact mul_le_mul (hC (w j)) (hC (w l)) (abs_nonneg _) ((abs_nonneg (β (w j))).trans (hC (w j)))
  have hS : Integrable (fun w => ∑ j, β (w j)) (piρ ρ n) := integrable_finsetSum _ fun j _ => hβj j
  have hSS : Integrable (fun w => ∑ j, ∑ l, β (w j) * β (w l)) (piρ ρ n) :=
    integrable_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ => hβjl j l
  -- the pointwise expansion
  have hpt : ∀ w : Fin n → X, ((1 / M) * ∑ j, β (w j) - (2 / M) * Δ) ^ 2 =
      (1 / M ^ 2) * ∑ j, ∑ l, β (w j) * β (w l) - (4 * Δ / M ^ 2) * ∑ j, β (w j)
        + (4 / M ^ 2) * Δ ^ 2 := by
    intro w
    have : (∑ j, β (w j)) ^ 2 = ∑ j, ∑ l, β (w j) * β (w l) := by
      rw [sq, Finset.sum_mul_sum]
    rw [← this]
    field_simp
    ring
  simp_rw [hpt]
  have hf1 : Integrable (fun w => (1 / M ^ 2) * ∑ j, ∑ l, β (w j) * β (w l)) (piρ ρ n) :=
    hSS.const_mul _
  have hf2 : Integrable (fun w => (4 * Δ / M ^ 2) * ∑ j, β (w j)) (piρ ρ n) := hS.const_mul _
  have e := integral_add (hf1.sub hf2) (integrable_const ((4 / M ^ 2) * Δ ^ 2)) (μ := piρ ρ n)
  simp only [Pi.sub_apply] at e
  rw [e, integral_sub hf1 hf2, integral_const_mul, integral_const_mul, integral_const,
    probReal_univ, smul_eq_mul, one_mul]
  -- the first moment vanishes
  have hS0 : ∫ w, ∑ j, β (w j) ∂(piρ ρ n) = 0 := by
    rw [integral_finsetSum _ fun j _ => hβj j]
    simp [integral_coord ρ hβ, hzero]
  -- the second moment
  have hSS' : ∫ w, ∑ j, ∑ l, β (w j) * β (w l) ∂(piρ ρ n) = n * ∫ x, β x ^ 2 ∂ρ := by
    rw [integral_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ => hβjl j l]
    have : ∀ j : Fin n, ∫ w, ∑ l, β (w j) * β (w l) ∂(piρ ρ n) = ∫ x, β x ^ 2 ∂ρ := by
      intro j
      rw [integral_finsetSum _ fun l _ => hβjl j l]
      have hterm : ∀ l : Fin n, ∫ w, β (w j) * β (w l) ∂(piρ ρ n) =
          if j = l then ∫ x, β x ^ 2 ∂ρ else 0 := by
        intro l
        by_cases hjl : j = l
        · subst hjl
          simp only [if_true]
          have hβ2 : StronglyMeasurable (fun x => β x ^ 2) := hβ.pow 2
          have : (fun w : Fin n → X => β (w j) * β (w j)) = fun w => (fun x => β x ^ 2) (w j) := by
            funext w; ring
          rw [this, integral_coord ρ hβ2 j]
        · simp only [hjl, if_false]
          rw [integral_coord_mul ρ hβ hβ hjl, hzero, zero_mul]
      simp_rw [hterm]
      simp
    simp_rw [this]
    simp
  rw [hS0, hSS']
  ring

end QuadratureField
