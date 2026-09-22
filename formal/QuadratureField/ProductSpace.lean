import QuadratureField.Floor

/-!
# The law of an i.i.d. sample: the product measure on `Fin n → X`

Infrastructure for the expectation step of Proposition 1. The seed draw
`z₁, …, z_M ~ ρ` i.i.d. is the product measure `piρ ρ M` on `Fin M → X`;
its coordinates are jointly independent with law `ρ`. We record: the
integral of a function of one coordinate, of a product of functions of two
distinct coordinates, and the peeling of the first coordinate
(`integral_pi_cons`), which reduces an `(n+1)`-fold integral to an outer
`ρ`-integral of an inner `n`-fold one.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] (ρ : Measure X) [IsProbabilityMeasure ρ]

/-- The law of `n` i.i.d. samples of `ρ`. -/
abbrev piρ (n : ℕ) : Measure (Fin n → X) := Measure.pi fun _ : Fin n => ρ

/-- The coordinates of the product space are jointly independent. -/
lemma coord_indep (n : ℕ) : iIndepFun (fun (i : Fin n) (v : Fin n → X) => v i) (piρ ρ n) := by
  have h := iIndepFun_pi (μ := fun _ : Fin n => ρ) (X := fun _ : Fin n => (id : X → X))
    (fun _ => aemeasurable_id)
  simpa using h

/-- Each coordinate has law `ρ`. -/
lemma coord_law (n : ℕ) (j : Fin n) : (piρ ρ n).map (fun v => v j) = ρ :=
  (measurePreserving_eval (fun _ : Fin n => ρ) j).map_eq

/-- The integral of a function of one coordinate. -/
lemma integral_coord {n : ℕ} {β : X → ℝ} (hβ : StronglyMeasurable β) (j : Fin n) :
    ∫ v, β (v j) ∂(piρ ρ n) = ∫ x, β x ∂ρ := by
  conv_rhs => rw [← coord_law ρ n j]
  rw [integral_map (measurable_pi_apply j).aemeasurable hβ.aestronglyMeasurable]

/-- The integral of a product of functions of two distinct coordinates factorizes. -/
lemma integral_coord_mul {n : ℕ} {β γ : X → ℝ} (hβ : StronglyMeasurable β)
    (hγ : StronglyMeasurable γ) {j l : Fin n} (hjl : j ≠ l) :
    ∫ v, β (v j) * γ (v l) ∂(piρ ρ n) = (∫ x, β x ∂ρ) * ∫ x, γ x ∂ρ := by
  have hind := (coord_indep ρ n).indepFun hjl
  rw [hind.integral_fun_comp_mul_comp (measurable_pi_apply j).aemeasurable
    (measurable_pi_apply l).aemeasurable hβ.aestronglyMeasurable hγ.aestronglyMeasurable,
    integral_coord ρ hβ, integral_coord ρ hγ]

/-- The integral of a bounded measurable function of two distinct coordinates is the
iterated `ρ`-integral (the pair has law `ρ ⊗ ρ`). -/
lemma integral_coord_pair {n : ℕ} {F : X → X → ℝ} (hF : StronglyMeasurable (Function.uncurry F))
    {C : ℝ} (hC : ∀ a b, |F a b| ≤ C) {i j : Fin n} (hij : i ≠ j) :
    ∫ v, F (v i) (v j) ∂(piρ ρ n) = ∫ a, ∫ b, F a b ∂ρ ∂ρ := by
  have hind := (coord_indep ρ n).indepFun hij
  have hpair : Measurable fun v : Fin n → X => (v i, v j) :=
    (measurable_pi_apply i).prodMk (measurable_pi_apply j)
  have hmap : (piρ ρ n).map (fun v => (v i, v j)) =
      ((piρ ρ n).map (fun v => v i)).prod ((piρ ρ n).map (fun v => v j)) :=
    (indepFun_iff_map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
      (measurable_pi_apply j).aemeasurable).mp hind
  have h1 : ∫ v, F (v i) (v j) ∂(piρ ρ n) =
      ∫ p, Function.uncurry F p ∂((piρ ρ n).map fun v => (v i, v j)) := by
    rw [integral_map hpair.aemeasurable hF.aestronglyMeasurable]; rfl
  rw [h1, hmap, coord_law, coord_law]
  have hint : Integrable (Function.uncurry F) (ρ.prod ρ) :=
    Integrable.of_bound hF.aestronglyMeasurable C (ae_of_all _ fun p => by
      rw [Real.norm_eq_abs]; exact hC p.1 p.2)
  rw [integral_prod _ hint]
  rfl

/-- A bounded strongly measurable function of one coordinate is integrable. -/
lemma integrable_coord {n : ℕ} {β : X → ℝ} (hβ : StronglyMeasurable β) {C : ℝ}
    (hC : ∀ x, |β x| ≤ C) (j : Fin n) : Integrable (fun v => β (v j)) (piρ ρ n) :=
  Integrable.of_bound (hβ.comp_measurable (measurable_pi_apply j)).aestronglyMeasurable C
    (ae_of_all _ fun v => hC (v j))

/-- **Peeling the first coordinate.** -/
theorem integral_pi_cons {n : ℕ} (F : (Fin (n + 1) → X) → ℝ) (hF : Integrable F (piρ ρ (n + 1))) :
    ∫ v, F v ∂(piρ ρ (n + 1)) = ∫ x, ∫ w, F (Fin.cons x w) ∂(piρ ρ n) ∂ρ := by
  have h := (measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => ρ) 0).symm
  have hF' : Integrable (fun p => F ((MeasurableEquiv.piFinSuccAbove (fun _ => X) 0).symm p))
      (ρ.prod (piρ ρ n)) := by
    have := (h.integrable_comp_emb (MeasurableEquiv.piFinSuccAbove (fun _ => X) 0).symm.measurableEmbedding).mpr hF
    exact this
  rw [← h.integral_comp' F]
  change ∫ p, F ((MeasurableEquiv.piFinSuccAbove (fun _ => X) 0).symm p) ∂(ρ.prod (piρ ρ n)) = _
  rw [integral_prod _ hF']
  congr 1
  ext x
  congr 1
  ext w
  congr 1
  ext i
  simp [MeasurableEquiv.piFinSuccAbove]

end QuadratureField
