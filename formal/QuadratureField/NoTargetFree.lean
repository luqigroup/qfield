import QuadratureField.SpectralRatio
import QuadratureField.ProductSpace

/-!
# No target-free constant: the second half of Proposition 1

Proposition 1's constant is `β_M(ε) r_ρ` with `r_ρ` depending on the reference.
The paper states that the dependence cannot be removed: no positive constant
depending only on `M` and the kernel bounds the expected gain below by a
fraction of the floor for every reference (not even for every reference with
`c_ρ ≤ 1/2`). The fragment shows this for the squared-exponential kernel by
spreading atoms until their features are nearly orthogonal.

Here the same collapse is machine-checked for the kernel in the paper's class
with *exactly* orthogonal atom features, `k(i, j) = 1{i = j}` on `ℕ` (positive
semidefinite, unit diagonal, nonnegative): with `ρ_n` uniform on `n` atoms,
`c_{ρ_n} = 1/n ≤ 1/2`, and for every ridged constrained minimizer the expected
gain is at most `4M²/n`, while the floor is at least `1/(2M)`; so the ratio
gain/floor is at most `8M³/n → 0`. The mechanism is the fragment's: on the
event that the `M` seeds land on distinct atoms the constrained reweighting
cannot improve on equal weights at all (the gap is exactly zero), and the
complementary event has probability at most `M²/n`.
-/

noncomputable section
open scoped BigOperators InnerProductSpace ENNReal
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField
namespace NoTargetFree

/-- `ℓ²(ℕ)`. -/
abbrev H₀ := lp (fun _ : ℕ => ℝ) 2

/-- Orthonormal atom features. -/
def Φ₀ (i : ℕ) : H₀ := lp.single 2 i (1 : ℝ)

lemma inner_Φ₀ (i j : ℕ) : ⟪Φ₀ i, Φ₀ j⟫_ℝ = if i = j then 1 else 0 := by
  unfold Φ₀
  rw [lp.inner_single_left]
  by_cases h : i = j
  · subst h; simp [lp.single_apply_self]
  · simp [lp.single_apply_ne _ _ _ (Ne.symm h), h]

lemma norm_Φ₀ (i : ℕ) : ‖Φ₀ i‖ = 1 := by
  unfold Φ₀; rw [lp.norm_single (by norm_num)]; simp

lemma kernel_Φ₀_nonneg (i j : ℕ) : 0 ≤ kernel Φ₀ i j := by
  unfold kernel; rw [inner_Φ₀]; split_ifs <;> norm_num

lemma stronglyMeasurable_Φ₀ : StronglyMeasurable Φ₀ := StronglyMeasurable.of_discrete

/-- The uniform reference on the first `n` atoms. -/
def ρn (n : ℕ) : Measure ℕ := (n : ℝ≥0∞)⁻¹ • ∑ i ∈ Finset.range n, Measure.dirac i

lemma ρn_apply (n : ℕ) (s : Set ℕ) :
    ρn n s = (n : ℝ≥0∞)⁻¹ * ∑ i ∈ Finset.range n, s.indicator 1 i := by
  simp [ρn, Measure.smul_apply, Measure.finset_sum_apply, Measure.dirac_apply]

instance ρn_isProbabilityMeasure (n : ℕ) [NeZero n] : IsProbabilityMeasure (ρn n) := by
  constructor
  rw [ρn_apply]
  simp [Set.indicator_univ, Finset.sum_const, Finset.card_range]
  rw [ENNReal.inv_mul_cancel] <;> simp [NeZero.ne n]

/-- Integrals against `ρ_n` are averages over the atoms. -/
lemma integral_ρn {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] (n : ℕ)
    (f : ℕ → E) :
    ∫ x, f x ∂(ρn n) = (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, f i := by
  unfold ρn
  rw [integral_smul_measure, integral_finset_sum_measure (fun i _ => integrable_dirac (by simp))]
  simp only [ENNReal.toReal_inv, ENNReal.toReal_natCast]
  congr 1
  exact Finset.sum_congr rfl fun i _ => integral_dirac f i

/-- Mass of the atoms beyond `n` is zero. -/
lemma ρn_ge_n (n : ℕ) : ρn n {k | n ≤ k} = 0 := by
  rw [ρn_apply]
  have : ∀ i ∈ Finset.range n, ({k : ℕ | n ≤ k}).indicator (1 : ℕ → ℝ≥0∞) i = 0 := by
    intro i hi
    rw [Finset.mem_range] at hi
    simp [Set.indicator_apply, not_le.mpr hi]
  rw [Finset.sum_eq_zero this, mul_zero]


/-! ### The kernel mean and the pathwise no-gain identity -/

lemma kernelMean_ρn (n : ℕ) [NeZero n] :
    kernelMean Φ₀ (ρn n) = (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Φ₀ i :=
  integral_ρn n Φ₀

lemma inner_Φ₀_sum (n k : ℕ) : ⟪Φ₀ k, ∑ i ∈ Finset.range n, Φ₀ i⟫_ℝ = if k < n then 1 else 0 := by
  rw [inner_sum]
  simp_rw [inner_Φ₀]
  rw [Finset.sum_ite_eq]
  simp [Finset.mem_range]

lemma inner_Φ₀_kernelMean (n : ℕ) [NeZero n] (k : ℕ) :
    ⟪Φ₀ k, kernelMean Φ₀ (ρn n)⟫_ℝ = (n : ℝ)⁻¹ * (if k < n then 1 else 0) := by
  rw [kernelMean_ρn, real_inner_smul_right, inner_Φ₀_sum]

lemma norm_kernelMean_ρn_sq (n : ℕ) [NeZero n] : ‖kernelMean Φ₀ (ρn n)‖ ^ 2 = (n : ℝ)⁻¹ := by
  have hn : (n : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne n
  rw [← real_inner_self_eq_norm_sq, kernelMean_ρn, real_inner_smul_left, real_inner_smul_right,
    sum_inner]
  simp_rw [inner_Φ₀_sum]
  have : ∑ i ∈ Finset.range n, (if i < n then (1 : ℝ) else 0) = n := by
    rw [Finset.sum_congr rfl (fun i hi => if_pos (Finset.mem_range.mp hi))]
    simp
  rw [this]
  field_simp

/-- On distinct in-range atoms, the discrepancy of any unit-sum weighting is `∑ w_j² − 1/n`. -/
lemma mmdSq_distinct {M n : ℕ} [NeZero n] (v : Fin M → ℕ) (hinj : Function.Injective v)
    (hrange : ∀ j, v j < n) (w : Fin M → ℝ) (hw : unitSum w) :
    mmdSq Φ₀ v w (kernelMean Φ₀ (ρn n)) = ∑ j, w j ^ 2 - (n : ℝ)⁻¹ := by
  rw [mmdSq_expand, norm_kernelMean_ρn_sq]
  have hgram : ∀ i j : Fin M, w i * w j * kernel Φ₀ (v i) (v j) = if i = j then w i ^ 2 else 0 := by
    intro i j
    unfold kernel
    rw [inner_Φ₀]
    by_cases h : i = j
    · subst h; simp [sq]
    · have : v i ≠ v j := fun h' => h (hinj h')
      simp [this, h]
  simp_rw [hgram]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  have hmean : ∀ j : Fin M, w j * ⟪Φ₀ (v j), kernelMean Φ₀ (ρn n)⟫_ℝ = (n : ℝ)⁻¹ * w j := by
    intro j
    rw [inner_Φ₀_kernelMean, if_pos (hrange j)]
    ring
  simp_rw [hmean]
  rw [← Finset.mul_sum]
  have : ∑ j, w j = 1 := hw
  rw [this]
  ring

/-- On distinct in-range atoms the reweighting gain is zero: equal weights are already
optimal, so any constrained minimizer does no better. -/
lemma gain_nonpos_distinct {M n : ℕ} [NeZero n] (hM : 0 < M) (v : Fin M → ℕ)
    (hinj : Function.Injective v) (hrange : ∀ j, v j < n) {wε : Fin M → ℝ} (hws : unitSum wε) :
    mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) - mmdSq Φ₀ v wε (kernelMean Φ₀ (ρn n)) ≤ 0 := by
  rw [mmdSq_distinct v hinj hrange _ (wEq_unitSum hM), mmdSq_distinct v hinj hrange wε hws]
  have h1 := one_div_le_dot_self wε hws hM
  rw [dot_self_eq_sum_sq] at h1
  have h2 : ∑ j : Fin M, wEq M j ^ 2 = 1 / M := by
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
    simp [wEq, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    field_simp
  rw [h2]
  linarith

/-- The gain never exceeds `4`. -/
lemma gain_le_four {M n : ℕ} [NeZero n] (hM : 0 < M) (v : Fin M → ℕ) (wε : Fin M → ℝ) :
    mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) - mmdSq Φ₀ v wε (kernelMean Φ₀ (ρn n)) ≤ 4 := by
  have h0 : 0 ≤ mmdSq Φ₀ v wε (kernelMean Φ₀ (ρn n)) := by unfold mmdSq; positivity
  have h4 : mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) ≤ 4 := by
    unfold mmdSq
    have : ‖emb Φ₀ v (wEq M) - kernelMean Φ₀ (ρn n)‖ ≤ 2 := by
      calc ‖emb Φ₀ v (wEq M) - kernelMean Φ₀ (ρn n)‖
          ≤ ‖emb Φ₀ v (wEq M)‖ + ‖kernelMean Φ₀ (ρn n)‖ := norm_sub_le _ _
        _ ≤ 1 + 1 := add_le_add (norm_emb_wEq_le_one Φ₀ norm_Φ₀ hM v)
            (norm_kernelMean_le_one Φ₀ (ρn n) norm_Φ₀)
        _ = 2 := by norm_num
    calc ‖emb Φ₀ v (wEq M) - kernelMean Φ₀ (ρn n)‖ ^ 2 ≤ 2 ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) this 2
      _ = 4 := by norm_num
  linarith

/-! ### The collision bound -/

/-- The bad-event counter: collisions among the seeds, and seeds outside the atoms. -/
def bad (M n : ℕ) (v : Fin M → ℕ) : ℝ :=
  (∑ p ∈ (Finset.univ : Finset (Fin M)).offDiag, if v p.1 = v p.2 then (1 : ℝ) else 0)
    + ∑ j, if n ≤ v j then (1 : ℝ) else 0

lemma bad_nonneg (M n : ℕ) (v : Fin M → ℕ) : 0 ≤ bad M n v := by
  unfold bad
  refine add_nonneg (Finset.sum_nonneg fun p _ => ?_) (Finset.sum_nonneg fun j _ => ?_) <;>
    split_ifs <;> norm_num

/-- If the seeds are not distinct in-range atoms, the counter is at least one. -/
lemma one_le_bad {M n : ℕ} (v : Fin M → ℕ) (h : ¬ (Function.Injective v ∧ ∀ j, v j < n)) :
    1 ≤ bad M n v := by
  unfold bad
  rw [not_and_or] at h
  rcases h with h | h
  · obtain ⟨i, j, hij, hne⟩ : ∃ i j, v i = v j ∧ i ≠ j := by
      simp only [Function.Injective, not_forall] at h
      obtain ⟨i, j, hij, hne⟩ := h
      exact ⟨i, j, hij, hne⟩
    have hmem : (i, j) ∈ (Finset.univ : Finset (Fin M)).offDiag := by
      simp [Finset.mem_offDiag, hne]
    have h1 : (1 : ℝ) ≤ ∑ p ∈ (Finset.univ : Finset (Fin M)).offDiag,
        if v p.1 = v p.2 then (1 : ℝ) else 0 := by
      have := Finset.single_le_sum (f := fun p : Fin M × Fin M => if v p.1 = v p.2 then (1 : ℝ) else 0)
        (fun p _ => by split_ifs <;> norm_num) hmem
      simpa [hij] using this
    have h2 : (0 : ℝ) ≤ ∑ j, if n ≤ v j then (1 : ℝ) else 0 :=
      Finset.sum_nonneg fun j _ => by split_ifs <;> norm_num
    linarith
  · push_neg at h
    obtain ⟨j, hj⟩ := h
    have h1 : (0 : ℝ) ≤ ∑ p ∈ (Finset.univ : Finset (Fin M)).offDiag,
        if v p.1 = v p.2 then (1 : ℝ) else 0 :=
      Finset.sum_nonneg fun p _ => by split_ifs <;> norm_num
    have h2 : (1 : ℝ) ≤ ∑ j, if n ≤ v j then (1 : ℝ) else 0 := by
      have := Finset.single_le_sum (f := fun j : Fin M => if n ≤ v j then (1 : ℝ) else 0)
        (fun j _ => by split_ifs <;> norm_num) (Finset.mem_univ j)
      simpa [hj] using this
    linarith

/-- Pointwise: the gain is at most `4` times the counter. -/
lemma gain_le_bad {M n : ℕ} [NeZero n] (hM : 0 < M) (v : Fin M → ℕ) {wε : Fin M → ℝ}
    (hws : unitSum wε) :
    mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) - mmdSq Φ₀ v wε (kernelMean Φ₀ (ρn n))
      ≤ 4 * bad M n v := by
  by_cases h : Function.Injective v ∧ ∀ j, v j < n
  · have := gain_nonpos_distinct hM v h.1 h.2 hws
    have := bad_nonneg M n v
    linarith
  · have := gain_le_four (n := n) hM v wε
    have := one_le_bad v h
    linarith

/-- A collision between two given coordinates has probability `1/n`. -/
lemma integral_collision {M n : ℕ} [NeZero n] {i j : Fin M} (hij : i ≠ j) :
    ∫ v, (if v i = v j then (1 : ℝ) else 0) ∂(piρ (ρn n) M) = (n : ℝ)⁻¹ := by
  have hn : (n : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne n
  have hF : StronglyMeasurable (Function.uncurry fun a b : ℕ => if a = b then (1 : ℝ) else 0) :=
    StronglyMeasurable.of_discrete
  have hC : ∀ a b : ℕ, |if a = b then (1 : ℝ) else 0| ≤ 1 := by
    intro a b; split_ifs <;> norm_num
  rw [integral_coord_pair (ρn n) hF hC hij]
  rw [integral_ρn]
  have hinner : ∀ a ∈ Finset.range n,
      ∫ b, (if a = b then (1 : ℝ) else 0) ∂(ρn n) = (n : ℝ)⁻¹ := by
    intro a ha
    rw [integral_ρn]
    simp only [smul_eq_mul]
    rw [Finset.sum_ite_eq, if_pos ha, mul_one]
  rw [Finset.sum_congr rfl hinner]
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, smul_eq_mul]
  field_simp

/-- A seed outside the atoms has probability zero. -/
lemma integral_outside {M n : ℕ} [NeZero n] (j : Fin M) :
    ∫ v, (if n ≤ v j then (1 : ℝ) else 0) ∂(piρ (ρn n) M) = 0 := by
  have hβ : StronglyMeasurable (fun a : ℕ => if n ≤ a then (1 : ℝ) else 0) :=
    StronglyMeasurable.of_discrete
  rw [integral_coord (ρn n) hβ j, integral_ρn]
  have : ∀ a ∈ Finset.range n, (if n ≤ a then (1 : ℝ) else 0) = 0 := by
    intro a ha; rw [if_neg (not_le.mpr (Finset.mem_range.mp ha))]
  rw [Finset.sum_congr rfl this]
  simp

/-- `E[bad] ≤ M²/n`. -/
lemma integral_bad_le {M n : ℕ} [NeZero n] :
    ∫ v, bad M n v ∂(piρ (ρn n) M) ≤ (M : ℝ) ^ 2 / n := by
  have hpair : ∀ p : Fin M × Fin M, Integrable (fun v : Fin M → ℕ => if v p.1 = v p.2 then (1 : ℝ) else 0)
      (piρ (ρn n) M) := by
    intro p
    refine Integrable.of_bound (StronglyMeasurable.of_discrete).aestronglyMeasurable 1
      (ae_of_all _ fun v => ?_)
    rw [Real.norm_eq_abs]; split_ifs <;> norm_num
  have hout : ∀ j : Fin M, Integrable (fun v : Fin M → ℕ => if n ≤ v j then (1 : ℝ) else 0)
      (piρ (ρn n) M) := by
    intro j
    refine Integrable.of_bound (StronglyMeasurable.of_discrete).aestronglyMeasurable 1
      (ae_of_all _ fun v => ?_)
    rw [Real.norm_eq_abs]; split_ifs <;> norm_num
  unfold bad
  rw [integral_add (integrable_finsetSum _ fun p _ => hpair p) (integrable_finsetSum _ fun j _ => hout j),
    integral_finsetSum _ fun p _ => hpair p, integral_finsetSum _ fun j _ => hout j]
  have h1 : ∀ p ∈ (Finset.univ : Finset (Fin M)).offDiag,
      ∫ v, (if v p.1 = v p.2 then (1 : ℝ) else 0) ∂(piρ (ρn n) M) = (n : ℝ)⁻¹ := by
    intro p hp
    have hne : p.1 ≠ p.2 := (Finset.mem_offDiag.mp hp).2.2
    exact integral_collision hne
  rw [Finset.sum_congr rfl h1]
  simp only [integral_outside, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul,
    Finset.offDiag_card, Finset.card_univ, Fintype.card_fin]
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  rw [div_eq_mul_inv]
  gcongr
  have : ((M * M - M : ℕ) : ℝ) ≤ (M : ℝ) ^ 2 := by
    have : M * M - M ≤ M * M := Nat.sub_le _ _
    calc ((M * M - M : ℕ) : ℝ) ≤ ((M * M : ℕ) : ℝ) := by exact_mod_cast this
      _ = (M : ℝ) ^ 2 := by push_cast; ring
  exact this

/-- **No target-free constant: the expected gain collapses.** For the discrete kernel and
the uniform reference on `n` atoms, every ridged constrained minimizer with integrable gain
has `E[MMD²(Q_eq) − MMD²(Q_wε)] ≤ 4M²/n`. -/
theorem expected_gain_le {M n : ℕ} [NeZero n] (hM : 0 < M) {ε : ℝ}
    (wε : (Fin M → ℕ) → (Fin M → ℝ))
    (hw : ∀ v, IsRidgeMin (gram Φ₀ v) (meanVec Φ₀ v (kernelMean Φ₀ (ρn n))) ε (wε v))
    (hint : Integrable (fun v => mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n))
      - mmdSq Φ₀ v (wε v) (kernelMean Φ₀ (ρn n))) (piρ (ρn n) M)) :
    ∫ v, (mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) - mmdSq Φ₀ v (wε v) (kernelMean Φ₀ (ρn n)))
        ∂(piρ (ρn n) M) ≤ 4 * (M : ℝ) ^ 2 / n := by
  have hbad_int : Integrable (fun v => 4 * bad M n v) (piρ (ρn n) M) := by
    refine Integrable.of_bound (StronglyMeasurable.of_discrete).aestronglyMeasurable
      (4 * ((M : ℝ) ^ 2 + M)) (ae_of_all _ fun v => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (by linarith [bad_nonneg M n v])]
    unfold bad
    have h1 : ∑ p ∈ (Finset.univ : Finset (Fin M)).offDiag,
        (if v p.1 = v p.2 then (1 : ℝ) else 0) ≤ (M : ℝ) ^ 2 := by
      calc _ ≤ ∑ p ∈ (Finset.univ : Finset (Fin M)).offDiag, (1 : ℝ) :=
            Finset.sum_le_sum fun p _ => by split_ifs <;> norm_num
        _ = ((M * M - M : ℕ) : ℝ) := by
            simp [Finset.sum_const, Finset.offDiag_card, Finset.card_univ]
        _ ≤ (M : ℝ) ^ 2 := by
            have : M * M - M ≤ M * M := Nat.sub_le _ _
            calc ((M * M - M : ℕ) : ℝ) ≤ ((M * M : ℕ) : ℝ) := by exact_mod_cast this
              _ = (M : ℝ) ^ 2 := by push_cast; ring
    have h2 : ∑ j : Fin M, (if n ≤ v j then (1 : ℝ) else 0) ≤ M := by
      calc _ ≤ ∑ j : Fin M, (1 : ℝ) := Finset.sum_le_sum fun j _ => by split_ifs <;> norm_num
        _ = M := by simp
    nlinarith
  calc ∫ v, (mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n)) - mmdSq Φ₀ v (wε v) (kernelMean Φ₀ (ρn n)))
        ∂(piρ (ρn n) M)
      ≤ ∫ v, 4 * bad M n v ∂(piρ (ρn n) M) :=
        integral_mono hint hbad_int fun v => gain_le_bad hM v (hw v).1
    _ = 4 * ∫ v, bad M n v ∂(piρ (ρn n) M) := integral_const_mul _ _
    _ ≤ 4 * ((M : ℝ) ^ 2 / n) := by gcongr; exact integral_bad_le
    _ = 4 * (M : ℝ) ^ 2 / n := by ring

/-- The floor of `ρ_n` is at least `1/(2M)` for `n ≥ 2`, and `c_{ρ_n} = 1/n ≤ 1/2`. -/
lemma floor_ρn_ge {M n : ℕ} [NeZero n] (hM : 0 < M) (hn : 2 ≤ n) :
    ‖kernelMean Φ₀ (ρn n)‖ ^ 2 ≤ 1 / 2 ∧
      1 / (2 * (M : ℝ)) ≤ (1 - ‖kernelMean Φ₀ (ρn n)‖ ^ 2) / M := by
  rw [norm_kernelMean_ρn_sq]
  have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hM' : (0 : ℝ) < M := by exact_mod_cast hM
  have hinv : (n : ℝ)⁻¹ ≤ 1 / 2 := by
    rw [inv_eq_one_div, div_le_div_iff₀ (by linarith) (by norm_num)]; linarith
  refine ⟨hinv, ?_⟩
  rw [div_le_div_iff₀ (by positivity) hM']
  nlinarith

/-- **The impossibility, as the paper states it.** For every `β > 0` and every node count
`M`, there is a reference in the class with `c_ρ ≤ 1/2` (here `ρ_n` for `n` large) on which
every ridged constrained minimizer's expected gain is below `β` times the floor: no positive
constant depending only on `M` and the kernel can replace `r_ρ`. -/
theorem no_target_free_constant {M : ℕ} (hM : 0 < M) {β : ℝ} (hβ : 0 < β) :
    ∃ n : ℕ, 2 ≤ n ∧ ∀ (_ : NeZero n), ‖kernelMean Φ₀ (ρn n)‖ ^ 2 ≤ 1 / 2 ∧
      ∀ (ε : ℝ) (wε : (Fin M → ℕ) → (Fin M → ℝ)),
        (∀ v, IsRidgeMin (gram Φ₀ v) (meanVec Φ₀ v (kernelMean Φ₀ (ρn n))) ε (wε v)) →
        Integrable (fun v => mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n))
          - mmdSq Φ₀ v (wε v) (kernelMean Φ₀ (ρn n))) (piρ (ρn n) M) →
        ∫ v, (mmdSq Φ₀ v (wEq M) (kernelMean Φ₀ (ρn n))
            - mmdSq Φ₀ v (wε v) (kernelMean Φ₀ (ρn n))) ∂(piρ (ρn n) M)
          < β * ((1 - ‖kernelMean Φ₀ (ρn n)‖ ^ 2) / M) := by
  obtain ⟨n₀, hn₀⟩ := exists_nat_gt (8 * (M : ℝ) ^ 3 / β)
  refine ⟨n₀ + 2, by omega, fun _ => ?_⟩
  have hM' : (0 : ℝ) < M := by exact_mod_cast hM
  have hn : (8 * (M : ℝ) ^ 3 / β) < ((n₀ + 2 : ℕ) : ℝ) := by push_cast; linarith
  have hnpos : (0 : ℝ) < ((n₀ + 2 : ℕ) : ℝ) := by positivity
  obtain ⟨hc, hfloor⟩ := floor_ρn_ge (n := n₀ + 2) hM (by omega)
  refine ⟨hc, fun ε wε hw hint => ?_⟩
  have hgain := expected_gain_le hM wε hw hint
  have hkey : 4 * (M : ℝ) ^ 2 / ((n₀ + 2 : ℕ) : ℝ) < β * (1 / (2 * (M : ℝ))) := by
    rw [div_lt_iff₀ hnpos]
    rw [div_lt_iff₀ hβ] at hn
    have : β * (1 / (2 * (M : ℝ))) * ((n₀ + 2 : ℕ) : ℝ) = β * ((n₀ + 2 : ℕ) : ℝ) / (2 * M) := by
      field_simp
    rw [this, lt_div_iff₀ (by positivity)]
    nlinarith
  calc _ ≤ 4 * (M : ℝ) ^ 2 / ((n₀ + 2 : ℕ) : ℝ) := hgain
    _ < β * (1 / (2 * (M : ℝ))) := hkey
    _ ≤ β * ((1 - ‖kernelMean Φ₀ (ρn (n₀ + 2))‖ ^ 2) / M) := by gcongr

end NoTargetFree
end QuadratureField
