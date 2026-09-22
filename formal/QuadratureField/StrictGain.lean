import QuadratureField.StrictGainMoments

/-!
# Proposition 1: the strict expected gain of reweighting

Paper §5.1, Proposition 1 (`bf-thm-betafrac`, main inequality and sharper form).
Seeds `z ~ ρ^M` i.i.d., `M = n + 2`, exact kernel mean `m = ∫ Φ dρ`, kernel
positive semidefinite with unit diagonal and `k ≥ 0`, ridge `ε ≥ 0`, and
`wε(z)` any ridged constrained minimizer at the seeds. Then

`E[MMD²(Q_eq) − MMD²(Q_wε)] ≥ (4 Var_ρ(μ_ρ) + (M−2)(1−c_ρ) r_ρ) / ((1+ε) M²)`

(`strict_gain_sharper`, valid for every `M ≥ 2`), hence for `M ≥ 3`

`E[MMD²(Q_eq) − MMD²(Q_wε)] ≥ ((M−2)/((1+ε)M)) · (1−c_ρ) r_ρ / M`

(`strict_gain`, the paper's display with `β_M(ε) = (M−2)/((1+ε)M)`), where
`(1−c_ρ) r_ρ = ∬ k̄² dρ dρ` is `rTimes` and `Var_ρ(μ_ρ) = ∫ ⟪m, Φ̄⟫² dρ` is
`varMean`. The only hypothesis beyond the paper's is that the map
`z ↦ MMD²(Q_wε(z))` is (a.e.) strongly measurable, which the deployed closed
form satisfies; the minimizer itself is otherwise arbitrary.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] {H : Type*} [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]

/-- `(1 − c_ρ) r_ρ`: the second moment of the centred kernel `k̄(y,x) = ⟪Φ̄ y, Φ̄ x⟫`. -/
def rTimes (Φ : X → H) (ρ : Measure X) : ℝ :=
  ∫ y, ∫ x, ⟪Φ y - kernelMean Φ ρ, Φ x - kernelMean Φ ρ⟫_ℝ ^ 2 ∂ρ ∂ρ

/-- `Var_ρ(μ_ρ) = ∫ ⟪m, Φ̄ x⟫² dρ`. -/
def varMean (Φ : X → H) (ρ : Measure X) : ℝ :=
  ∫ x, ⟪kernelMean Φ ρ, Φ x - kernelMean Φ ρ⟫_ℝ ^ 2 ∂ρ

section Setup
variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ]

lemma norm_kernelMean_le_one (hunit : ∀ x, ‖Φ x‖ = 1) : ‖kernelMean Φ ρ‖ ≤ 1 := by
  unfold kernelMean
  have h := norm_integral_le_of_norm_le_const (μ := ρ) (f := Φ) (C := 1)
    (ae_of_all _ fun x => (hunit x).le)
  simpa [probReal_univ] using h

lemma norm_centered_le_two (hunit : ∀ x, ‖Φ x‖ = 1) (y : X) : ‖Φ y - kernelMean Φ ρ‖ ≤ 2 := by
  calc ‖Φ y - kernelMean Φ ρ‖ ≤ ‖Φ y‖ + ‖kernelMean Φ ρ‖ := norm_sub_le _ _
    _ ≤ 1 + 1 := by gcongr; exact (hunit y).le; exact norm_kernelMean_le_one Φ ρ hunit
    _ = 2 := by norm_num

lemma integrable_centered (hΦ : StronglyMeasurable Φ) (hunit : ∀ x, ‖Φ x‖ = 1) :
    Integrable (fun y => Φ y - kernelMean Φ ρ) ρ :=
  (integrable_feature Φ hΦ (fun x => (hunit x).le) ρ).sub (integrable_const _)

lemma integral_centered (hΦ : StronglyMeasurable Φ) (hunit : ∀ x, ‖Φ x‖ = 1) :
    ∫ y, (Φ y - kernelMean Φ ρ) ∂ρ = 0 := by
  rw [integral_sub (integrable_feature Φ hΦ (fun x => (hunit x).le) ρ) (integrable_const _),
    integral_const, probReal_univ, one_smul]
  exact sub_self _

lemma integral_inner_centered (hΦ : StronglyMeasurable Φ) (hunit : ∀ x, ‖Φ x‖ = 1) (u : H) :
    ∫ y, ⟪Φ y - kernelMean Φ ρ, u⟫_ℝ ∂ρ = 0 := by
  have : ∀ y, ⟪Φ y - kernelMean Φ ρ, u⟫_ℝ = ⟪u, Φ y - kernelMean Φ ρ⟫_ℝ := fun y =>
    real_inner_comm _ _
  simp_rw [this]
  rw [integral_inner (integrable_centered Φ ρ hΦ hunit) u, integral_centered Φ ρ hΦ hunit,
    inner_zero_right]

lemma stronglyMeasurable_emb_wEq (hΦ : StronglyMeasurable Φ) (M : ℕ) :
    StronglyMeasurable (fun v : Fin M → X => emb Φ v (wEq M)) := by
  unfold emb
  have h : StronglyMeasurable (∑ j : Fin M, fun v : Fin M → X => wEq M j • Φ (v j)) :=
    Finset.stronglyMeasurable_sum _ fun j _ =>
      StronglyMeasurable.const_smul (𝕜 := ℝ) (hΦ.comp_measurable (measurable_pi_apply j)) (wEq M j)
  convert h using 1
  funext v
  simp [Finset.sum_apply]

lemma norm_emb_wEq_le_one (hunit : ∀ x, ‖Φ x‖ = 1) {M : ℕ} (hM : 0 < M) (v : Fin M → X) :
    ‖emb Φ v (wEq M)‖ ≤ 1 := by
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  calc ‖emb Φ v (wEq M)‖ ≤ ∑ j, ‖wEq M j • Φ (v j)‖ := norm_sum_le _ _
    _ = ∑ j : Fin M, (1 / (M : ℝ)) := by
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [norm_smul, hunit, mul_one, wEq, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    _ = 1 := by simp [hM']

lemma measurable_cons {n : ℕ} (x₀ : X) :
    Measurable (fun v : Fin n → X => (Fin.cons x₀ v : Fin (n + 1) → X)) := by
  refine measurable_pi_iff.mpr fun i => ?_
  refine Fin.cases ?_ (fun j => ?_) i
  · simp only [Fin.cons_zero]; exact measurable_const
  · simp only [Fin.cons_succ]; exact measurable_pi_apply j

/-- Peel the first two coordinates of a bounded measurable integrand. -/
lemma integral_pi_cons_cons {n : ℕ} (F : (Fin (n + 2) → X) → ℝ) (hFm : StronglyMeasurable F)
    {C : ℝ} (hC : ∀ v, |F v| ≤ C) :
    ∫ v, F v ∂(piρ ρ (n + 2)) =
      ∫ x₀, ∫ x₁, ∫ w, F (Fin.cons x₀ (Fin.cons x₁ w)) ∂(piρ ρ n) ∂ρ ∂ρ := by
  have hF : Integrable F (piρ ρ (n + 2)) :=
    Integrable.of_bound hFm.aestronglyMeasurable C (ae_of_all _ fun v => by
      rw [Real.norm_eq_abs]; exact hC v)
  rw [integral_pi_cons ρ F hF]
  congr 1
  funext x₀
  have hF1 : Integrable (fun v' => F (Fin.cons x₀ v')) (piρ ρ (n + 1)) :=
    Integrable.of_bound (hFm.comp_measurable (measurable_cons x₀)).aestronglyMeasurable C
      (ae_of_all _ fun v' => by rw [Real.norm_eq_abs]; exact hC _)
  exact integral_pi_cons ρ (fun v' => F (Fin.cons x₀ v')) hF1

end Setup

section Moments
variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ] (hΦ : StronglyMeasurable Φ)
  (hunit : ∀ x, ‖Φ x‖ = 1)
include hΦ hunit

/-- The centred kernel `γ y x = ⟪Φ̄ y, Φ̄ x⟫`, measurable and bounded by `4`. -/
lemma stronglyMeasurable_gamma :
    StronglyMeasurable (fun p : X × X => ⟪Φ p.1 - kernelMean Φ ρ, Φ p.2 - kernelMean Φ ρ⟫_ℝ) :=
  ((hΦ.comp_measurable measurable_fst).sub stronglyMeasurable_const).inner
    ((hΦ.comp_measurable measurable_snd).sub stronglyMeasurable_const)

lemma abs_gamma_le (y x : X) : |⟪Φ y - kernelMean Φ ρ, Φ x - kernelMean Φ ρ⟫_ℝ| ≤ 4 := by
  calc |⟪Φ y - kernelMean Φ ρ, Φ x - kernelMean Φ ρ⟫_ℝ|
      ≤ ‖Φ y - kernelMean Φ ρ‖ * ‖Φ x - kernelMean Φ ρ‖ := abs_real_inner_le_norm _ _
    _ ≤ 2 * 2 := mul_le_mul (norm_centered_le_two Φ ρ hunit y) (norm_centered_le_two Φ ρ hunit x)
        (norm_nonneg _) (by norm_num)
    _ = 4 := by norm_num

/-- `∫ γ y x dρ(x) = 0`. -/
lemma integral_gamma (y : X) :
    ∫ x, ⟪Φ y - kernelMean Φ ρ, Φ x - kernelMean Φ ρ⟫_ℝ ∂ρ = 0 := by
  rw [integral_inner (integrable_centered Φ ρ hΦ hunit), integral_centered Φ ρ hΦ hunit,
    inner_zero_right]

/-- **Lemma S4, the variance term:** `E_{x₀,x₁} ∫_y ⟪Φ̄ y, Φ x₀ − Φ x₁⟫² = 2 (1 − c_ρ) r_ρ`. -/
theorem triple_moment :
    ∫ x₀, ∫ x₁, ∫ y, ⟪Φ y - kernelMean Φ ρ, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ ∂ρ ∂ρ = 2 * rTimes Φ ρ := by
  set m := kernelMean Φ ρ with hm
  set γ : X → X → ℝ := fun y x => ⟪Φ y - m, Φ x - m⟫_ℝ with hγ
  have hsplit : ∀ y x₀ x₁, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ = γ y x₀ - γ y x₁ := by
    intro y x₀ x₁
    have : Φ x₀ - Φ x₁ = (Φ x₀ - m) - (Φ x₁ - m) := by abel
    rw [this, inner_sub_right]
  simp_rw [hsplit]
  -- measurability and bounds of the integrands
  have hγm : StronglyMeasurable (fun p : X × X => γ p.1 p.2) := stronglyMeasurable_gamma Φ ρ hΦ hunit
  have hγb : ∀ y x, |γ y x| ≤ 4 := abs_gamma_le Φ ρ hΦ hunit
  have hGb : ∀ y x₀ x₁, |(γ y x₀ - γ y x₁) ^ 2| ≤ 64 := by
    intro y x₀ x₁
    rw [abs_pow]
    have : |γ y x₀ - γ y x₁| ≤ 8 := by
      calc |γ y x₀ - γ y x₁| ≤ |γ y x₀| + |γ y x₁| := abs_sub _ _
        _ ≤ 4 + 4 := add_le_add (hγb _ _) (hγb _ _)
        _ = 8 := by norm_num
    calc |γ y x₀ - γ y x₁| ^ 2 ≤ 8 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
      _ = 64 := by norm_num
  -- swap the third sample past x₁, for each fixed x₀
  have hswap1 : ∀ x₀, ∫ x₁, ∫ y, (γ y x₀ - γ y x₁) ^ 2 ∂ρ ∂ρ
      = ∫ y, ∫ x₁, (γ y x₀ - γ y x₁) ^ 2 ∂ρ ∂ρ := by
    intro x₀
    refine integral_swap_of_bound ρ (G := fun x₁ y => (γ y x₀ - γ y x₁) ^ 2) ?_ (fun x₁ y => hGb y x₀ x₁)
    have h1 : StronglyMeasurable (fun p : X × X => γ p.2 x₀) :=
      hγm.comp_measurable (measurable_snd.prodMk measurable_const)
    have h2 : StronglyMeasurable (fun p : X × X => γ p.2 p.1) :=
      hγm.comp_measurable (measurable_snd.prodMk measurable_fst)
    exact (h1.sub h2).pow 2
  simp_rw [hswap1]
  -- swap the third sample past x₀
  have hFm : StronglyMeasurable (fun p : X × X => ∫ x₁, (γ p.2 p.1 - γ p.2 x₁) ^ 2 ∂ρ) := by
    have h : StronglyMeasurable (fun q : (X × X) × X => (γ q.1.2 q.1.1 - γ q.1.2 q.2) ^ 2) := by
      have h1 : StronglyMeasurable (fun q : (X × X) × X => γ q.1.2 q.1.1) :=
        hγm.comp_measurable ((measurable_snd.comp measurable_fst).prodMk
          (measurable_fst.comp measurable_fst))
      have h2 : StronglyMeasurable (fun q : (X × X) × X => γ q.1.2 q.2) :=
        hγm.comp_measurable ((measurable_snd.comp measurable_fst).prodMk measurable_snd)
      exact (h1.sub h2).pow 2
    exact h.integral_prod_right'
  have hFb : ∀ x₀ y, |∫ x₁, (γ y x₀ - γ y x₁) ^ 2 ∂ρ| ≤ 64 := by
    intro x₀ y
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 64)
      (f := fun x₁ => (γ y x₀ - γ y x₁) ^ 2) (ae_of_all _ fun x₁ => by
        rw [Real.norm_eq_abs]; exact hGb y x₀ x₁)
    simpa [probReal_univ] using h
  rw [integral_swap_of_bound ρ (G := fun x₀ y => ∫ x₁, (γ y x₀ - γ y x₁) ^ 2 ∂ρ) hFm hFb]
  -- inside, for each y: E[(γ_y(x₀) − γ_y(x₁))²] = 2 ∫ γ_y² − 2 (∫ γ_y)² = 2 ∫ γ_y²
  have hinner : ∀ y, ∫ x₀, ∫ x₁, (γ y x₀ - γ y x₁) ^ 2 ∂ρ ∂ρ = 2 * ∫ x, γ y x ^ 2 ∂ρ := by
    intro y
    have hαm : StronglyMeasurable (γ y) := hγm.comp_measurable (measurable_const.prodMk measurable_id)
    rw [pair_diff_sq_integral ρ hαm (hγb y), integral_gamma Φ ρ hΦ hunit y]
    ring
  simp_rw [hinner]
  rw [integral_const_mul]
  rfl

/-- **Lemma S4, the mean term:** `E_{x₀,x₁} ⟪m, Φ x₀ − Φ x₁⟫² = 2 Var_ρ(μ_ρ)`. -/
theorem pair_mean_moment :
    ∫ x₀, ∫ x₁, ⟪kernelMean Φ ρ, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ ∂ρ = 2 * varMean Φ ρ := by
  set m := kernelMean Φ ρ with hm
  set α : X → ℝ := fun x => ⟪m, Φ x - m⟫_ℝ with hα
  have hsplit : ∀ x₀ x₁, ⟪m, Φ x₀ - Φ x₁⟫_ℝ = α x₀ - α x₁ := by
    intro x₀ x₁
    have : Φ x₀ - Φ x₁ = (Φ x₀ - m) - (Φ x₁ - m) := by abel
    rw [this, inner_sub_right]
  simp_rw [hsplit]
  have hαm : StronglyMeasurable α :=
    stronglyMeasurable_const.inner (hΦ.sub stronglyMeasurable_const)
  have hαb : ∀ x, |α x| ≤ 2 := by
    intro x
    calc |α x| ≤ ‖m‖ * ‖Φ x - m‖ := abs_real_inner_le_norm _ _
      _ ≤ 1 * 2 := mul_le_mul (norm_kernelMean_le_one Φ ρ hunit) (norm_centered_le_two Φ ρ hunit x)
          (norm_nonneg _) (by norm_num)
      _ = 2 := by norm_num
  have hα0 : ∫ x, α x ∂ρ = 0 := by
    simp only [hα]
    rw [integral_inner (integrable_centered Φ ρ hΦ hunit), integral_centered Φ ρ hΦ hunit,
      inner_zero_right]
  rw [pair_diff_sq_integral ρ hαm hαb, hα0]
  simp only [varMean]
  ring_nf
  rfl

end Moments

/-- The pair inner product `a(z) = ⟪emb(Q_eq) − m, Φ(z_0) − Φ(z_1)⟫`. -/
def pairInner (Φ : X → H) (m : H) {M : ℕ} (v : Fin (M + 2) → X) : ℝ :=
  ⟪emb Φ v (wEq (M + 2)) - m, Φ (v 0) - Φ (v 1)⟫_ℝ

section Main
variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ] (hΦ : StronglyMeasurable Φ)
  (hunit : ∀ x, ‖Φ x‖ = 1)
include hΦ hunit

/-- **Lemma S3–S4 assembled:** `E a² = (2/M²) ((M−2)(1−c_ρ) r_ρ + 4 Var_ρ(μ_ρ))`, `M = n + 2`. -/
theorem integral_pairInner_sq (n : ℕ) :
    ∫ v, pairInner Φ (kernelMean Φ ρ) v ^ 2 ∂(piρ ρ (n + 2)) =
      (2 / ((n : ℝ) + 2) ^ 2) * (n * rTimes Φ ρ + 4 * varMean Φ ρ) := by
  set m := kernelMean Φ ρ with hm
  set M : ℝ := (n : ℝ) + 2 with hM
  have hMpos : 0 < M := by rw [hM]; positivity
  have hM0 : M ≠ 0 := hMpos.ne'
  have hMnat : ((n + 2 : ℕ) : ℝ) = M := by push_cast; rfl
  -- measurability and bound of a²
  have ham : StronglyMeasurable (fun v : Fin (n + 2) → X => pairInner Φ m v) := by
    unfold pairInner
    exact ((stronglyMeasurable_emb_wEq Φ hΦ (n + 2)).sub stronglyMeasurable_const).inner
      ((hΦ.comp_measurable (measurable_pi_apply 0)).sub (hΦ.comp_measurable (measurable_pi_apply 1)))
  have hab : ∀ v : Fin (n + 2) → X, |pairInner Φ m v| ≤ 4 := by
    intro v
    unfold pairInner
    calc |⟪emb Φ v (wEq (n + 2)) - m, Φ (v 0) - Φ (v 1)⟫_ℝ|
        ≤ ‖emb Φ v (wEq (n + 2)) - m‖ * ‖Φ (v 0) - Φ (v 1)‖ := abs_real_inner_le_norm _ _
      _ ≤ 2 * 2 := by
        gcongr
        · calc ‖emb Φ v (wEq (n + 2)) - m‖ ≤ ‖emb Φ v (wEq (n + 2))‖ + ‖m‖ := norm_sub_le _ _
            _ ≤ 1 + 1 := add_le_add (norm_emb_wEq_le_one Φ hunit (by omega) v)
                (norm_kernelMean_le_one Φ ρ hunit)
            _ = 2 := by norm_num
        · calc ‖Φ (v 0) - Φ (v 1)‖ ≤ ‖Φ (v 0)‖ + ‖Φ (v 1)‖ := norm_sub_le _ _
            _ = 2 := by rw [hunit, hunit]; norm_num
      _ = 4 := by norm_num
  have hab2 : ∀ v : Fin (n + 2) → X, |pairInner Φ m v ^ 2| ≤ 16 := by
    intro v
    rw [abs_pow]
    calc |pairInner Φ m v| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) (hab v) 2
      _ = 16 := by norm_num
  -- peel the two pair coordinates
  rw [integral_pi_cons_cons ρ (fun v => pairInner Φ m v ^ 2) (ham.pow 2) hab2]
  -- inner integral: the conditional second moment
  have hinner : ∀ x₀ x₁, ∫ w, pairInner Φ m (Fin.cons x₀ (Fin.cons x₁ w)) ^ 2 ∂(piρ ρ n)
      = (n / M ^ 2) * (∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ)
        + (4 / M ^ 2) * ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 := by
    intro x₀ x₁
    have hdecomp : ∀ w : Fin n → X, pairInner Φ m (Fin.cons x₀ (Fin.cons x₁ w))
        = (1 / M) * ∑ j, ⟪Φ (w j) - m, Φ x₀ - Φ x₁⟫_ℝ - (2 / M) * ⟪m, Φ x₀ - Φ x₁⟫_ℝ := by
      intro w
      unfold pairInner
      simp only [Fin.cons_zero, Fin.cons_one]
      rw [pair_inner_decomp Φ x₀ x₁ (by rw [hunit, hunit]) w m]
    simp_rw [hdecomp]
    have hβm : StronglyMeasurable (fun y => ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ) :=
      (hΦ.sub stronglyMeasurable_const).inner stronglyMeasurable_const
    have hβb : ∀ y, |⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ| ≤ 4 := by
      intro y
      calc |⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ| ≤ ‖Φ y - m‖ * ‖Φ x₀ - Φ x₁‖ := abs_real_inner_le_norm _ _
        _ ≤ 2 * 2 := by
          gcongr
          · exact norm_centered_le_two Φ ρ hunit y
          · calc ‖Φ x₀ - Φ x₁‖ ≤ ‖Φ x₀‖ + ‖Φ x₁‖ := norm_sub_le _ _
              _ = 2 := by rw [hunit, hunit]; norm_num
        _ = 4 := by norm_num
    exact inner_second_moment ρ hβm hβb (integral_inner_centered Φ ρ hΦ hunit _) hM0 _
  simp_rw [hinner]
  -- outer integrals: split and evaluate
  have hVm : StronglyMeasurable (fun p : X × X => ∫ y, ⟪Φ y - m, Φ p.1 - Φ p.2⟫_ℝ ^ 2 ∂ρ) := by
    have h : StronglyMeasurable (fun q : (X × X) × X => ⟪Φ q.2 - m, Φ q.1.1 - Φ q.1.2⟫_ℝ ^ 2) := by
      exact (((hΦ.comp_measurable measurable_snd).sub stronglyMeasurable_const).inner
        ((hΦ.comp_measurable (measurable_fst.comp measurable_fst)).sub
          (hΦ.comp_measurable (measurable_snd.comp measurable_fst)))).pow 2
    exact h.integral_prod_right'
  have hVb : ∀ x₀ x₁, |∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ| ≤ 16 := by
    intro x₀ x₁
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 16)
      (f := fun y => ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2) (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow]
        have : |⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ| ≤ 4 := by
          calc |⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ| ≤ ‖Φ y - m‖ * ‖Φ x₀ - Φ x₁‖ := abs_real_inner_le_norm _ _
            _ ≤ 2 * 2 := by
              gcongr
              · exact norm_centered_le_two Φ ρ hunit y
              · calc ‖Φ x₀ - Φ x₁‖ ≤ ‖Φ x₀‖ + ‖Φ x₁‖ := norm_sub_le _ _
                  _ = 2 := by rw [hunit, hunit]; norm_num
            _ = 4 := by norm_num
        calc |⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
          _ = 16 := by norm_num)
    simpa [probReal_univ] using h
  have hΔm : StronglyMeasurable (fun p : X × X => ⟪m, Φ p.1 - Φ p.2⟫_ℝ ^ 2) :=
    (stronglyMeasurable_const.inner ((hΦ.comp_measurable measurable_fst).sub
      (hΦ.comp_measurable measurable_snd))).pow 2
  have hΔb : ∀ x₀ x₁, |⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2| ≤ 4 := by
    intro x₀ x₁
    rw [abs_pow]
    have : |⟪m, Φ x₀ - Φ x₁⟫_ℝ| ≤ 2 := by
      calc |⟪m, Φ x₀ - Φ x₁⟫_ℝ| ≤ ‖m‖ * ‖Φ x₀ - Φ x₁‖ := abs_real_inner_le_norm _ _
        _ ≤ 1 * 2 := by
          gcongr
          · exact norm_kernelMean_le_one Φ ρ hunit
          · calc ‖Φ x₀ - Φ x₁‖ ≤ ‖Φ x₀‖ + ‖Φ x₁‖ := norm_sub_le _ _
              _ = 2 := by rw [hunit, hunit]; norm_num
        _ = 2 := by norm_num
    calc |⟪m, Φ x₀ - Φ x₁⟫_ℝ| ^ 2 ≤ 2 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
      _ = 4 := by norm_num
  -- integrability of the two pieces in x₁ (for fixed x₀) and in x₀
  have hV1 : ∀ x₀, Integrable (fun x₁ => (n / M ^ 2) * ∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ) ρ := by
    intro x₀
    refine (Integrable.of_bound (hVm.comp_measurable (measurable_const.prodMk measurable_id)).aestronglyMeasurable
      16 (ae_of_all _ fun x₁ => by rw [Real.norm_eq_abs]; exact hVb x₀ x₁)).const_mul _
  have hΔ1 : ∀ x₀, Integrable (fun x₁ => (4 / M ^ 2) * ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2) ρ := by
    intro x₀
    refine (Integrable.of_bound (hΔm.comp_measurable (measurable_const.prodMk measurable_id)).aestronglyMeasurable
      4 (ae_of_all _ fun x₁ => by rw [Real.norm_eq_abs]; exact hΔb x₀ x₁)).const_mul _
  have hstep1 : ∀ x₀, ∫ x₁, ((n / M ^ 2) * (∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ)
      + (4 / M ^ 2) * ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2) ∂ρ
      = (n / M ^ 2) * (∫ x₁, ∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ ∂ρ)
        + (4 / M ^ 2) * ∫ x₁, ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ := by
    intro x₀
    rw [integral_add (hV1 x₀) (hΔ1 x₀), integral_const_mul, integral_const_mul]
  simp_rw [hstep1]
  have hV0 : Integrable (fun x₀ => (n / M ^ 2) * ∫ x₁, ∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ ∂ρ) ρ := by
    refine (Integrable.of_bound hVm.integral_prod_right'.aestronglyMeasurable 16
      (ae_of_all _ fun x₀ => ?_)).const_mul _
    rw [Real.norm_eq_abs]
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 16)
      (f := fun x₁ => ∫ y, ⟪Φ y - m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ) (ae_of_all _ fun x₁ => by
        rw [Real.norm_eq_abs]; exact hVb x₀ x₁)
    simpa [probReal_univ] using h
  have hΔ0 : Integrable (fun x₀ => (4 / M ^ 2) * ∫ x₁, ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2 ∂ρ) ρ := by
    refine (Integrable.of_bound hΔm.integral_prod_right'.aestronglyMeasurable 4
      (ae_of_all _ fun x₀ => ?_)).const_mul _
    rw [Real.norm_eq_abs]
    have h := norm_integral_le_of_norm_le_const (μ := ρ) (C := 4)
      (f := fun x₁ => ⟪m, Φ x₀ - Φ x₁⟫_ℝ ^ 2) (ae_of_all _ fun x₁ => by
        rw [Real.norm_eq_abs]; exact hΔb x₀ x₁)
    simpa [probReal_univ] using h
  rw [integral_add hV0 hΔ0, integral_const_mul, integral_const_mul, triple_moment Φ ρ hΦ hunit,
    pair_mean_moment Φ ρ hΦ hunit]
  field_simp
  try ring

/-- **Proposition 1, sharper form** (`bf-eq-sharper`), valid for every `M = n + 2 ≥ 2`:
`E[MMD²(Q_eq) − MMD²(Q_wε)] ≥ (4 Var_ρ(μ_ρ) + (M−2)(1−c_ρ) r_ρ) / ((1+ε) M²)`. -/
theorem strict_gain_sharper (hnn : ∀ x y, 0 ≤ kernel Φ x y) {n : ℕ} {ε : ℝ} (hε : 0 ≤ ε)
    (wε : (Fin (n + 2) → X) → (Fin (n + 2) → ℝ))
    (hw : ∀ v, IsRidgeMin (gram Φ v) (meanVec Φ v (kernelMean Φ ρ)) ε (wε v))
    (hmeas : AEStronglyMeasurable (fun v => mmdSq Φ v (wε v) (kernelMean Φ ρ)) (piρ ρ (n + 2))) :
    (4 * varMean Φ ρ + n * rTimes Φ ρ) / ((1 + ε) * ((n : ℝ) + 2) ^ 2) ≤
      ∫ v, (mmdSq Φ v (wEq (n + 2)) (kernelMean Φ ρ) - mmdSq Φ v (wε v) (kernelMean Φ ρ))
        ∂(piρ ρ (n + 2)) := by
  set m := kernelMean Φ ρ with hm
  have hMpos : 0 < n + 2 := by omega
  -- pathwise pair gain with the pair (0, 1)
  have h01 : (0 : Fin (n + 2)) ≠ 1 := Fin.zero_ne_one
  have hpath : ∀ v, pairInner Φ m v ^ 2 / (2 + 2 * ε)
      ≤ mmdSq Φ v (wEq (n + 2)) m - mmdSq Φ v (wε v) m := fun v =>
    pair_gain_paper Φ hunit hnn hMpos v m hε (hw v) h01
  -- integrability
  have hgain_nonneg : ∀ v, 0 ≤ mmdSq Φ v (wEq (n + 2)) m - mmdSq Φ v (wε v) m := fun v =>
    sub_nonneg.mpr (mmdSq_reweight_le Φ hMpos v m hε (hw v)).2
  have hgain_le : ∀ v, mmdSq Φ v (wEq (n + 2)) m - mmdSq Φ v (wε v) m ≤ 4 := by
    intro v
    have h0 : 0 ≤ mmdSq Φ v (wε v) m := by unfold mmdSq; positivity
    have h4 : mmdSq Φ v (wEq (n + 2)) m ≤ 4 := by
      unfold mmdSq
      have : ‖emb Φ v (wEq (n + 2)) - m‖ ≤ 2 := by
        calc ‖emb Φ v (wEq (n + 2)) - m‖ ≤ ‖emb Φ v (wEq (n + 2))‖ + ‖m‖ := norm_sub_le _ _
          _ ≤ 1 + 1 := add_le_add (norm_emb_wEq_le_one Φ hunit hMpos v)
              (norm_kernelMean_le_one Φ ρ hunit)
          _ = 2 := by norm_num
      calc ‖emb Φ v (wEq (n + 2)) - m‖ ^ 2 ≤ 2 ^ 2 := pow_le_pow_left₀ (norm_nonneg _) this 2
        _ = 4 := by norm_num
    linarith
  have hgain_meas : AEStronglyMeasurable
      (fun v => mmdSq Φ v (wEq (n + 2)) m - mmdSq Φ v (wε v) m) (piρ ρ (n + 2)) := by
    have : StronglyMeasurable (fun v : Fin (n + 2) → X => mmdSq Φ v (wEq (n + 2)) m) := by
      unfold mmdSq
      exact ((stronglyMeasurable_emb_wEq Φ hΦ (n + 2)).sub stronglyMeasurable_const).norm.pow 2
    exact this.aestronglyMeasurable.sub hmeas
  have hgain_int : Integrable (fun v => mmdSq Φ v (wEq (n + 2)) m - mmdSq Φ v (wε v) m)
      (piρ ρ (n + 2)) :=
    Integrable.of_bound hgain_meas 4 (ae_of_all _ fun v => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hgain_nonneg v)]; exact hgain_le v)
  have ham : StronglyMeasurable (fun v : Fin (n + 2) → X => pairInner Φ m v) := by
    unfold pairInner
    exact ((stronglyMeasurable_emb_wEq Φ hΦ (n + 2)).sub stronglyMeasurable_const).inner
      ((hΦ.comp_measurable (measurable_pi_apply 0)).sub (hΦ.comp_measurable (measurable_pi_apply 1)))
  have hab : ∀ v : Fin (n + 2) → X, |pairInner Φ m v ^ 2| ≤ 16 := by
    intro v
    rw [abs_pow]
    have : |pairInner Φ m v| ≤ 4 := by
      unfold pairInner
      calc |⟪emb Φ v (wEq (n + 2)) - m, Φ (v 0) - Φ (v 1)⟫_ℝ|
          ≤ ‖emb Φ v (wEq (n + 2)) - m‖ * ‖Φ (v 0) - Φ (v 1)‖ := abs_real_inner_le_norm _ _
        _ ≤ 2 * 2 := by
          gcongr
          · calc ‖emb Φ v (wEq (n + 2)) - m‖ ≤ ‖emb Φ v (wEq (n + 2))‖ + ‖m‖ := norm_sub_le _ _
              _ ≤ 1 + 1 := add_le_add (norm_emb_wEq_le_one Φ hunit hMpos v)
                  (norm_kernelMean_le_one Φ ρ hunit)
              _ = 2 := by norm_num
          · calc ‖Φ (v 0) - Φ (v 1)‖ ≤ ‖Φ (v 0)‖ + ‖Φ (v 1)‖ := norm_sub_le _ _
              _ = 2 := by rw [hunit, hunit]; norm_num
        _ = 4 := by norm_num
    calc |pairInner Φ m v| ^ 2 ≤ 4 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
      _ = 16 := by norm_num
  have ha_int : Integrable (fun v => pairInner Φ m v ^ 2 / (2 + 2 * ε)) (piρ ρ (n + 2)) :=
    (Integrable.of_bound (ham.pow 2).aestronglyMeasurable 16 (ae_of_all _ fun v => by
      rw [Real.norm_eq_abs]; exact hab v)).div_const _
  -- integrate the pathwise bound
  have hmono := integral_mono ha_int hgain_int hpath
  rw [integral_div, integral_pairInner_sq Φ ρ hΦ hunit n] at hmono
  refine le_trans (le_of_eq ?_) hmono
  have hε' : (1 + ε) ≠ 0 := by positivity
  field_simp
  ring

/-- **Proposition 1 as printed** (`bf-eq-main`): for `M = n + 2 ≥ 3`,
`E[MMD²(Q_eq) − MMD²(Q_wε)] ≥ β_M(ε) · (1−c_ρ) r_ρ / M` with
`β_M(ε) = (M−2)/((1+ε)M)`. -/
theorem strict_gain (hnn : ∀ x y, 0 ≤ kernel Φ x y) {n : ℕ} {ε : ℝ} (hε : 0 ≤ ε)
    (wε : (Fin (n + 2) → X) → (Fin (n + 2) → ℝ))
    (hw : ∀ v, IsRidgeMin (gram Φ v) (meanVec Φ v (kernelMean Φ ρ)) ε (wε v))
    (hmeas : AEStronglyMeasurable (fun v => mmdSq Φ v (wε v) (kernelMean Φ ρ)) (piρ ρ (n + 2))) :
    ((n : ℝ) / ((1 + ε) * ((n : ℝ) + 2))) * (rTimes Φ ρ / ((n : ℝ) + 2)) ≤
      ∫ v, (mmdSq Φ v (wEq (n + 2)) (kernelMean Φ ρ) - mmdSq Φ v (wε v) (kernelMean Φ ρ))
        ∂(piρ ρ (n + 2)) := by
  have h := strict_gain_sharper Φ ρ hΦ hunit hnn hε wε hw hmeas
  have hvar : 0 ≤ varMean Φ ρ := integral_nonneg fun x => sq_nonneg _
  have hε' : 0 < 1 + ε := by positivity
  have hM : 0 < (n : ℝ) + 2 := by positivity
  calc ((n : ℝ) / ((1 + ε) * ((n : ℝ) + 2))) * (rTimes Φ ρ / ((n : ℝ) + 2))
      = (n * rTimes Φ ρ) / ((1 + ε) * ((n : ℝ) + 2) ^ 2) := by field_simp; try ring
    _ ≤ (4 * varMean Φ ρ + n * rTimes Φ ρ) / ((1 + ε) * ((n : ℝ) + 2) ^ 2) := by
        gcongr
        linarith
    _ ≤ _ := h

end Main

end QuadratureField
