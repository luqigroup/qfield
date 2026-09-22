import QuadratureField.PairGain

/-!
# The pathwise decomposition behind Lemma S3 of the strict-gain proof

With the sample written as `v = cons x₀ (cons x₁ w)` (`M = n + 2`), the
pair inner product `⟪D, u⟫`, `D = emb(Q_eq) − m`, `u = Φ x₀ − Φ x₁`, is

`⟪D, u⟫ = (1/M) ∑_j β(w_j) − (2/M) Δ`,   `β(y) = ⟪Φ y − m, u⟫`,   `Δ = ⟪m, u⟫`,

provided `‖Φ x₀‖ = ‖Φ x₁‖` (the unit diagonal). The two seed features that
define the pair drop out exactly, which is why the conditional mean of
`⟪D, u⟫` is `−2Δ/M` and its conditional variance is that of a sum of `M − 2`
i.i.d. terms.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The equal-weight embedding of `cons x₀ (cons x₁ w)` is the average of the features. -/
lemma emb_wEq_cons_cons (Φ : X → H) {n : ℕ} (x₀ x₁ : X) (w : Fin n → X) :
    emb Φ (Fin.cons x₀ (Fin.cons x₁ w)) (wEq (n + 2)) =
      (1 / ((n : ℝ) + 2)) • (Φ x₀ + Φ x₁ + ∑ j, Φ (w j)) := by
  simp only [emb, wEq]
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]
  rw [← Finset.smul_sum, ← smul_add, ← smul_add, ← add_assoc]
  congr 1
  all_goals first | rfl | (push_cast; ring)

/-- **The pathwise decomposition.** -/
theorem pair_inner_decomp (Φ : X → H) {n : ℕ} (x₀ x₁ : X) (hdiag : ‖Φ x₀‖ = ‖Φ x₁‖)
    (w : Fin n → X) (m : H) :
    ⟪emb Φ (Fin.cons x₀ (Fin.cons x₁ w)) (wEq (n + 2)) - m, Φ x₀ - Φ x₁⟫_ℝ =
      (1 / ((n : ℝ) + 2)) * ∑ j, ⟪Φ (w j) - m, Φ x₀ - Φ x₁⟫_ℝ
        - (2 / ((n : ℝ) + 2)) * ⟪m, Φ x₀ - Φ x₁⟫_ℝ := by
  set u := Φ x₀ - Φ x₁ with hu
  set M : ℝ := (n : ℝ) + 2 with hM
  have hMpos : M ≠ 0 := by rw [hM]; positivity
  rw [emb_wEq_cons_cons, inner_sub_left, real_inner_smul_left, inner_add_left, inner_add_left,
    sum_inner]
  -- ⟪Φ x₀, u⟫ + ⟪Φ x₁, u⟫ = ‖Φ x₀‖² − ‖Φ x₁‖² = 0
  have hpair : ⟪Φ x₀, u⟫_ℝ + ⟪Φ x₁, u⟫_ℝ = 0 := by
    rw [hu, inner_sub_right, inner_sub_right, real_inner_self_eq_norm_sq,
      real_inner_self_eq_norm_sq, real_inner_comm (Φ x₁) (Φ x₀), hdiag]
    ring
  have hβ : ∑ j, ⟪Φ (w j) - m, u⟫_ℝ = ∑ j, ⟪Φ (w j), u⟫_ℝ - n * ⟪m, u⟫_ℝ := by
    simp only [inner_sub_left, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
  rw [hβ]
  have : ⟪Φ x₀, u⟫_ℝ + ⟪Φ x₁, u⟫_ℝ + ∑ j, ⟪Φ (w j), u⟫_ℝ = ∑ j, ⟪Φ (w j), u⟫_ℝ := by
    rw [hpair, zero_add]
  rw [this]
  field_simp
  ring

/-- The deviation `β` has `ρ`-mean zero: stated algebraically, `⟪∫Φ − m, u⟫ = 0`
when `∫ Φ dρ = m`; the integral form is in `StrictGain.lean`. -/
lemma beta_mean_zero (m u : H) : ⟪m - m, u⟫_ℝ = 0 := by simp

end QuadratureField
