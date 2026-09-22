import QuadratureField.Weights

/-!
# The one-pair gain (Lemma S2 of the strict-gain proof), pathwise

The pathwise core of Proposition 1 (`bf-lem-pairgain`). At any node
configuration (coincident nodes allowed), for the ridged constrained
minimizer `wε` under the exact kernel mean, moving weight along one pair of
nodes shows

`MMD²(Q_eq) − MMD²(Q_wε) ≥ ⟪D, u⟫² / (b + 2ε)`

for every `b ≥ ‖u‖²` with `b + 2ε > 0`, where `D = emb(Q_eq) − m` and
`u = Φ(z_i) − Φ(z_j)`. With `k ≥ 0` and a unit diagonal one may take `b = 2`
(the paper's constant); with a unit diagonal alone, `b = 4` (the fragment's
remark on dropping `k ≥ 0`). The lemma holds for every pair `i ≠ j`.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset Matrix

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

lemma emb_add (Φ : X → H) {M : ℕ} (z : Fin M → X) (v v' : Fin M → ℝ) :
    emb Φ z (v + v') = emb Φ z v + emb Φ z v' := by
  simp [emb, add_smul, Finset.sum_add_distrib]

lemma emb_smul (Φ : X → H) {M : ℕ} (z : Fin M → X) (c : ℝ) (v : Fin M → ℝ) :
    emb Φ z (c • v) = c • emb Φ z v := by
  simp [emb, Finset.smul_sum, smul_smul]

lemma emb_single (Φ : X → H) {M : ℕ} (z : Fin M → X) (i : Fin M) :
    emb Φ z (Pi.single i (1 : ℝ)) = Φ (z i) := by
  simp [emb, Pi.single_apply]

lemma emb_sub (Φ : X → H) {M : ℕ} (z : Fin M → X) (v v' : Fin M → ℝ) :
    emb Φ z (v - v') = emb Φ z v - emb Φ z v' := by
  simp [emb, sub_smul, Finset.sum_sub_distrib]

/-- The pair direction in weight space, `e_i − e_j`. -/
def pairDir {M : ℕ} (i j : Fin M) : Fin M → ℝ := Pi.single i 1 - Pi.single j 1

lemma sum_pairDir {M : ℕ} (i j : Fin M) : ∑ l, pairDir i j l = 0 := by
  simp [pairDir, Finset.sum_sub_distrib]

lemma pairDir_dot_self {M : ℕ} {i j : Fin M} (hij : i ≠ j) : pairDir i j ⬝ᵥ pairDir i j = 2 := by
  simp only [pairDir, dotProduct_sub, dotProduct_single, Pi.sub_apply, Pi.single_eq_same,
    Pi.single_eq_of_ne hij, Pi.single_eq_of_ne hij.symm]
  norm_num

lemma wEq_dot_pairDir {M : ℕ} (i j : Fin M) : wEq M ⬝ᵥ pairDir i j = 0 := by
  simp only [dotProduct, wEq, ← Finset.mul_sum, sum_pairDir, mul_zero]

lemma emb_pairDir (Φ : X → H) {M : ℕ} (z : Fin M → X) (i j : Fin M) :
    emb Φ z (pairDir i j) = Φ (z i) - Φ (z j) := by
  rw [pairDir, emb_sub, emb_single, emb_single]

/-- The test weight `w_t = w_eq + t (e_i − e_j)`: unit sum, embedding `emb(w_eq) + t u`,
squared norm `1/M + 2t²`. -/
lemma test_weight {M : ℕ} (hM : 0 < M) {i j : Fin M} (hij : i ≠ j) (t : ℝ) :
    unitSum (wEq M + t • pairDir i j) ∧
      (wEq M + t • pairDir i j) ⬝ᵥ (wEq M + t • pairDir i j) = 1 / M + 2 * t ^ 2 := by
  constructor
  · unfold unitSum
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib,
      ← Finset.mul_sum, sum_pairDir, mul_zero, add_zero]
    exact wEq_unitSum hM
  · rw [add_dotProduct, dotProduct_add, dotProduct_add, wEq_dot hM, dotProduct_smul,
      smul_dotProduct, smul_dotProduct, dotProduct_smul, wEq_dot_pairDir,
      dotProduct_comm (pairDir i j) (wEq M), wEq_dot_pairDir, pairDir_dot_self hij]
    simp only [smul_eq_mul]
    ring

/-- **The one-pair gain, pathwise** (`bf-lem-pairgain`). -/
theorem pair_gain (Φ : X → H) {M : ℕ} (hM : 0 < M) (z : Fin M → X) (m : H) {ε : ℝ} (hε : 0 ≤ ε)
    {wε : Fin M → ℝ} (hw : IsRidgeMin (gram Φ z) (meanVec Φ z m) ε wε) {i j : Fin M} (hij : i ≠ j)
    {b : ℝ} (hb : ‖Φ (z i) - Φ (z j)‖ ^ 2 ≤ b) (hpos : 0 < b + 2 * ε) :
    ⟪emb Φ z (wEq M) - m, Φ (z i) - Φ (z j)⟫_ℝ ^ 2 / (b + 2 * ε) ≤
      mmdSq Φ z (wEq M) m - mmdSq Φ z wε m := by
  set D := emb Φ z (wEq M) - m with hD
  set u := Φ (z i) - Φ (z j) with hu
  set a := ⟪D, u⟫_ℝ with ha
  set c := b + 2 * ε with hc
  obtain ⟨hws, hmin⟩ := hw
  -- the test point at the optimal step t = -a/c
  set t := -a / c with ht
  obtain ⟨hunit, hnorm⟩ := test_weight hM hij t
  have hopt := hmin _ hunit
  unfold ridgeCrit at hopt
  rw [crit_exact, crit_exact, hnorm] at hopt
  -- MMD² at the test point
  have hembt : emb Φ z (wEq M + t • pairDir i j) = D + m + t • u := by
    rw [emb_add, emb_smul, emb_pairDir, hD, hu]; abel
  have hmmdt : mmdSq Φ z (wEq M + t • pairDir i j) m = ‖D‖ ^ 2 + 2 * t * a + t ^ 2 * ‖u‖ ^ 2 := by
    unfold mmdSq
    rw [hembt]
    have : D + m + t • u - m = D + t • u := by abel
    rw [this, norm_add_sq_real, real_inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs,
      sq_abs]
    ring
  have hmmd_eq : mmdSq Φ z (wEq M) m = ‖D‖ ^ 2 := by unfold mmdSq; rfl
  have hwε := one_div_le_dot_self wε hws hM
  -- gain ≥ -2 t a - t² (‖u‖² + 2ε)
  have hgain : -2 * t * a - t ^ 2 * (‖u‖ ^ 2 + 2 * ε) ≤ mmdSq Φ z (wEq M) m - mmdSq Φ z wε m := by
    rw [hmmd_eq]
    nlinarith [mul_nonneg hε (sub_nonneg.mpr hwε)]
  -- and the right side of that is at least a²/c at t = -a/c
  have hstep : a ^ 2 / c ≤ -2 * t * a - t ^ 2 * (‖u‖ ^ 2 + 2 * ε) := by
    have hcpos : 0 < c := hpos
    have hu_le : ‖u‖ ^ 2 + 2 * ε ≤ c := by rw [hc]; linarith
    have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
    have h1 : t ^ 2 * (‖u‖ ^ 2 + 2 * ε) ≤ t ^ 2 * c := mul_le_mul_of_nonneg_left hu_le ht2
    have h2 : -2 * t * a - t ^ 2 * c = a ^ 2 / c := by
      rw [ht]; field_simp; ring
    linarith
  exact hstep.trans hgain

/-- With `k ≥ 0` and a unit diagonal, `‖Φ(z_i) − Φ(z_j)‖² = 2 − 2k(z_i, z_j) ≤ 2`. -/
lemma pair_norm_sq_le_two (Φ : X → H) (hunit : ∀ x, ‖Φ x‖ = 1) (hnn : ∀ x y, 0 ≤ kernel Φ x y)
    (x y : X) : ‖Φ x - Φ y‖ ^ 2 ≤ 2 := by
  rw [norm_sub_sq_real, hunit, hunit]
  have := hnn x y
  simp only [kernel] at this
  nlinarith

/-- With a unit diagonal alone, `‖Φ(z_i) − Φ(z_j)‖² ≤ 4`. -/
lemma pair_norm_sq_le_four (Φ : X → H) (hunit : ∀ x, ‖Φ x‖ = 1) (x y : X) :
    ‖Φ x - Φ y‖ ^ 2 ≤ 4 := by
  have h := norm_sub_le (Φ x) (Φ y)
  rw [hunit, hunit] at h
  nlinarith [norm_nonneg (Φ x - Φ y)]

/-- **The paper's form** of the pair gain: `k ≥ 0`, unit diagonal, so
`gain ≥ ⟪D,u⟫²/(2 + 2ε)`. -/
theorem pair_gain_paper (Φ : X → H) (hunit : ∀ x, ‖Φ x‖ = 1) (hnn : ∀ x y, 0 ≤ kernel Φ x y)
    {M : ℕ} (hM : 0 < M) (z : Fin M → X) (m : H) {ε : ℝ} (hε : 0 ≤ ε)
    {wε : Fin M → ℝ} (hw : IsRidgeMin (gram Φ z) (meanVec Φ z m) ε wε) {i j : Fin M} (hij : i ≠ j) :
    ⟪emb Φ z (wEq M) - m, Φ (z i) - Φ (z j)⟫_ℝ ^ 2 / (2 + 2 * ε) ≤
      mmdSq Φ z (wEq M) m - mmdSq Φ z wε m :=
  pair_gain Φ hM z m hε hw hij (pair_norm_sq_le_two Φ hunit hnn _ _) (by linarith)

end QuadratureField
