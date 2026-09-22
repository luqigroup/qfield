import QuadratureField.Setting

/-!
# The closed-form weights and the two-arm inequality

Paper §4.2 and §4.4 (and Theorem A.1's ridged chain). For a Gram `K`, an
estimand vector `μh` (exact or estimated) and a ridge `ε ≥ 0`, the deployed
weights minimize the ridged criterion `wᵀ(K + εI)w − 2wᵀμh` over unit-sum
vectors.

* `twoArm_chain` — **the samples never strictly win the criterion against
  their own reweighting**, for ANY estimand, ANY ridge `ε ≥ 0` and ANY matrix
  `K` (no positive-semidefiniteness, no unit diagonal is used):
  `Ĵ(w_rw) ≤ Ĵ(w_eq) − ε(‖w_rw‖² − 1/M) ≤ Ĵ(w_eq)`.
* `mmdSq_reweight_le` — the same chain in the discrepancy under the exact
  kernel mean (the middle part of Theorem A.1's sharper chain).
* `isRidgeMin_of_kkt` — the KKT certificate: a unit-sum `w` with
  `A w − μh` constant is a minimizer whenever `A` is symmetric positive
  semidefinite.
* `closedForm_isRidgeMin` — the closed form of §4.2, two linear solves
  `A⁻¹μh` and `A⁻¹1`, is a minimizer for every positive definite `A`
  (so for `K + εI`, `ε > 0`, at every Gram, coincident nodes included), and
  `ridgeMin_unique` — it is the only one.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset Matrix

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The ridged criterion `Ĵ(w) + ε‖w‖²`. -/
def ridgeCrit {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) (ε : ℝ)
    (w : Fin M → ℝ) : ℝ :=
  crit K μh w + ε * (w ⬝ᵥ w)

/-- `w` is a ridged constrained minimizer: unit-sum, and no unit-sum vector does
better on the ridged criterion. -/
def IsRidgeMin {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) (ε : ℝ)
    (w : Fin M → ℝ) : Prop :=
  unitSum w ∧ ∀ v, unitSum v → ridgeCrit K μh ε w ≤ ridgeCrit K μh ε v

lemma wEq_unitSum {M : ℕ} (hM : 0 < M) : unitSum (wEq M) := by
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  simp [unitSum, wEq, hM']

lemma wEq_dot {M : ℕ} (hM : 0 < M) : wEq M ⬝ᵥ wEq M = 1 / M := by
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  simp [dotProduct, wEq]
  field_simp

lemma dot_self_eq_sum_sq {M : ℕ} (w : Fin M → ℝ) : w ⬝ᵥ w = ∑ j, w j ^ 2 := by
  simp [dotProduct, sq]

/-- Cauchy–Schwarz on the unit-sum constraint: `‖w‖² ≥ 1/M`. -/
lemma one_div_le_dot_self {M : ℕ} (w : Fin M → ℝ) (hw : unitSum w) (hM : 0 < M) :
    (1 : ℝ) / M ≤ w ⬝ᵥ w := by
  have h := sq_sum_le_card_mul_sum_sq (s := (univ : Finset (Fin M))) (f := w)
  have hs : ∑ j, w j = 1 := hw
  rw [hs, one_pow, card_univ, Fintype.card_fin] at h
  rw [dot_self_eq_sum_sq, div_le_iff₀ (by exact_mod_cast hM)]
  linarith [h]

/-- **The samples never strictly win the criterion against their reweighting**
(§4.4; `sg-prop-twoarm`): for any estimand, any ridge and any `K`. -/
theorem twoArm_chain {M : ℕ} (hM : 0 < M) (K : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ)
    {ε : ℝ} (hε : 0 ≤ ε) {w : Fin M → ℝ} (hw : IsRidgeMin K μh ε w) :
    crit K μh w ≤ crit K μh (wEq M) - ε * (w ⬝ᵥ w - 1 / M) ∧
      crit K μh w ≤ crit K μh (wEq M) := by
  obtain ⟨hws, hmin⟩ := hw
  have h1 := hmin (wEq M) (wEq_unitSum hM)
  unfold ridgeCrit at h1
  rw [wEq_dot hM] at h1
  have h2 := one_div_le_dot_self w hws hM
  refine ⟨by linarith, ?_⟩
  nlinarith [mul_nonneg hε (sub_nonneg.mpr h2)]

/-- The ridged floor chain in the discrepancy, under the exact kernel mean
(Theorem A.1, middle of the sharper chain): reweighting the nodes is never worse
than equal weights, with the exact ridge slack. -/
theorem mmdSq_reweight_le (Φ : X → H) {M : ℕ} (hM : 0 < M) (z : Fin M → X) (m : H)
    {ε : ℝ} (hε : 0 ≤ ε) {w : Fin M → ℝ}
    (hw : IsRidgeMin (gram Φ z) (meanVec Φ z m) ε w) :
    mmdSq Φ z w m ≤ mmdSq Φ z (wEq M) m - ε * (w ⬝ᵥ w - 1 / M) ∧
      mmdSq Φ z w m ≤ mmdSq Φ z (wEq M) m := by
  have h := twoArm_chain hM (gram Φ z) (meanVec Φ z m) hε hw
  rw [crit_exact, crit_exact] at h
  constructor <;> linarith [h.1, h.2]

/-! ### The KKT certificate and the closed form -/

/-- The unridged quadratic objective of a symmetric matrix `A`. -/
def quadObj {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) (v : Fin M → ℝ) : ℝ :=
  v ⬝ᵥ (A *ᵥ v) - 2 * (v ⬝ᵥ μh)

/-- `(ε • 1) *ᵥ u = ε • u`. -/
lemma smul_one_mulVec {M : ℕ} (ε : ℝ) (u : Fin M → ℝ) :
    (ε • (1 : Matrix (Fin M) (Fin M) ℝ)) *ᵥ u = ε • u := by
  ext i
  simp [Matrix.mulVec, dotProduct, Matrix.one_apply]

lemma dot_smul_one_mulVec {M : ℕ} (ε : ℝ) (u : Fin M → ℝ) :
    u ⬝ᵥ ((ε • (1 : Matrix (Fin M) (Fin M) ℝ)) *ᵥ u) = ε * (u ⬝ᵥ u) := by
  rw [smul_one_mulVec, dotProduct_smul, smul_eq_mul]

lemma ridgeCrit_eq_quadObj {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) (ε : ℝ)
    (w : Fin M → ℝ) :
    ridgeCrit K μh ε w = quadObj (K + ε • (1 : Matrix (Fin M) (Fin M) ℝ)) μh w := by
  simp only [ridgeCrit, crit, quadObj, Matrix.add_mulVec, dotProduct_add, dot_smul_one_mulVec]
  ring

lemma dot_mulVec_symm {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (hA : A.IsSymm) (a b : Fin M → ℝ) :
    a ⬝ᵥ (A *ᵥ b) = b ⬝ᵥ (A *ᵥ a) := by
  rw [dotProduct_mulVec, dotProduct_comm]
  congr 1
  conv_lhs => rw [← hA.eq]
  exact vecMul_transpose A a

/-- The quadratic expansion around `w` for a symmetric `A`. -/
lemma quadObj_sub {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (hA : A.IsSymm) (μh : Fin M → ℝ)
    (v w : Fin M → ℝ) :
    quadObj A μh v - quadObj A μh w =
      (v - w) ⬝ᵥ (A *ᵥ (v - w)) + 2 * ((v - w) ⬝ᵥ (A *ᵥ w - μh)) := by
  simp only [quadObj, Matrix.mulVec_sub, dotProduct_sub, sub_dotProduct]
  have := dot_mulVec_symm A hA v w
  ring_nf
  linarith

lemma sum_sub_eq_zero_of_unitSum {M : ℕ} {v w : Fin M → ℝ} (hv : unitSum v) (hw : unitSum w) :
    ∑ j, (v j - w j) = 0 := by
  rw [Finset.sum_sub_distrib]
  have h1 : ∑ j, v j = 1 := hv
  have h2 : ∑ j, w j = 1 := hw
  rw [h1, h2]; ring

lemma dot_const_of_unitSum {M : ℕ} {v w : Fin M → ℝ} (hv : unitSum v) (hw : unitSum w) (c : ℝ) :
    (v - w) ⬝ᵥ (fun _ => c) = 0 := by
  simp only [dotProduct, Pi.sub_apply, ← Finset.sum_mul]
  rw [sum_sub_eq_zero_of_unitSum hv hw, zero_mul]

/-- **KKT sufficiency.** If `A` is symmetric positive semidefinite (as a quadratic
form), `w` has unit sum and `A w − μh` is a constant vector, then `w` minimizes
the quadratic objective over unit-sum vectors. -/
theorem quadObj_min_of_kkt {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (hA : A.IsSymm)
    (hpsd : ∀ u : Fin M → ℝ, 0 ≤ u ⬝ᵥ (A *ᵥ u)) (μh : Fin M → ℝ) {w : Fin M → ℝ}
    (hw : unitSum w) {c : ℝ} (hkkt : A *ᵥ w - μh = fun _ => c) :
    ∀ v, unitSum v → quadObj A μh w ≤ quadObj A μh v := by
  intro v hv
  have h := quadObj_sub A hA μh v w
  have hcross : (v - w) ⬝ᵥ (A *ᵥ w - μh) = 0 := by
    rw [hkkt]; exact dot_const_of_unitSum hv hw c
  have := hpsd (v - w)
  linarith

lemma dot_self_nonneg' {M : ℕ} (u : Fin M → ℝ) : 0 ≤ u ⬝ᵥ u := by
  rw [dot_self_eq_sum_sq]; positivity

lemma dot_self_pos_of_ne_zero {M : ℕ} {u : Fin M → ℝ} (hu : u ≠ 0) : 0 < u ⬝ᵥ u := by
  rw [dot_self_eq_sum_sq]
  obtain ⟨i, hi⟩ : ∃ i, u i ≠ 0 := by
    by_contra h
    push_neg at h
    exact hu (funext h)
  have h1 : 0 < u i ^ 2 := by positivity
  have h2 : u i ^ 2 ≤ ∑ j, u j ^ 2 :=
    Finset.single_le_sum (fun j _ => sq_nonneg (u j)) (Finset.mem_univ i)
  linarith

lemma isSymm_add_smul_one {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (hK : K.IsSymm) (ε : ℝ) :
    (K + ε • (1 : Matrix (Fin M) (Fin M) ℝ)).IsSymm := by
  rw [Matrix.IsSymm, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_one, hK.eq]

/-- KKT sufficiency for the ridged criterion. -/
theorem isRidgeMin_of_kkt {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (hK : K.IsSymm)
    (hpsd : ∀ u : Fin M → ℝ, 0 ≤ u ⬝ᵥ (K *ᵥ u)) (μh : Fin M → ℝ) {ε : ℝ} (hε : 0 ≤ ε)
    {w : Fin M → ℝ} (hw : unitSum w) {c : ℝ}
    (hkkt : (K + ε • (1 : Matrix (Fin M) (Fin M) ℝ)) *ᵥ w - μh = fun _ => c) :
    IsRidgeMin K μh ε w := by
  refine ⟨hw, fun v hv => ?_⟩
  rw [ridgeCrit_eq_quadObj, ridgeCrit_eq_quadObj]
  refine quadObj_min_of_kkt _ (isSymm_add_smul_one K hK ε) ?_ μh hw hkkt v hv
  intro u
  rw [Matrix.add_mulVec, dotProduct_add, dot_smul_one_mulVec]
  have := hpsd u
  nlinarith [mul_nonneg hε (dot_self_nonneg' u)]

/-- Gram matrices are symmetric. -/
lemma gram_isSymm (Φ : X → H) {M : ℕ} (z : Fin M → X) : (gram Φ z).IsSymm := by
  ext i j; simp [gram, Matrix.transpose_apply, kernel_comm]

/-- Gram quadratic forms are the squared norm of the embedding, hence nonnegative. -/
lemma dot_gram_mulVec (Φ : X → H) {M : ℕ} (z : Fin M → X) (v : Fin M → ℝ) :
    v ⬝ᵥ (gram Φ z *ᵥ v) = ‖emb Φ z v‖ ^ 2 := by
  rw [norm_emb_sq]
  simp only [dotProduct, Matrix.mulVec, gram, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

lemma gram_psd (Φ : X → H) {M : ℕ} (z : Fin M → X) (v : Fin M → ℝ) :
    0 ≤ v ⬝ᵥ (gram Φ z *ᵥ v) := by
  rw [dot_gram_mulVec]; positivity

/-! ### The closed form: two linear solves -/

/-- The all-ones vector. -/
def ones (M : ℕ) : Fin M → ℝ := fun _ => 1

/-- The closed-form constrained weights of §4.2 from the two solves `A⁻¹ μh` and
`A⁻¹ 1`: `w = A⁻¹ μh − λ A⁻¹ 1` with `λ = (1ᵀA⁻¹μh − 1)/(1ᵀA⁻¹1)`. -/
def closedForm {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) : Fin M → ℝ :=
  A⁻¹ *ᵥ μh - (((ones M) ⬝ᵥ (A⁻¹ *ᵥ μh) - 1) / ((ones M) ⬝ᵥ (A⁻¹ *ᵥ ones M))) • (A⁻¹ *ᵥ ones M)

lemma dot_ones {M : ℕ} (v : Fin M → ℝ) : ones M ⬝ᵥ v = ∑ j, v j := by
  simp [dotProduct, ones]

/-- The closed form has unit sum whenever `1ᵀ A⁻¹ 1 ≠ 0`. -/
lemma closedForm_unitSum {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ)
    (h : ones M ⬝ᵥ (A⁻¹ *ᵥ ones M) ≠ 0) : unitSum (closedForm A μh) := by
  unfold unitSum
  rw [← dot_ones, closedForm, dotProduct_sub, dotProduct_smul, smul_eq_mul]
  field_simp
  ring

/-- The closed form satisfies the KKT condition when `A` is invertible. -/
lemma closedForm_kkt {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (hA : IsUnit A.det) (μh : Fin M → ℝ) :
    A *ᵥ closedForm A μh - μh =
      fun _ => -(((ones M) ⬝ᵥ (A⁻¹ *ᵥ μh) - 1) / ((ones M) ⬝ᵥ (A⁻¹ *ᵥ ones M))) := by
  have hinv : ∀ v : Fin M → ℝ, A *ᵥ (A⁻¹ *ᵥ v) = v := by
    intro v
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv A hA, Matrix.one_mulVec]
  unfold closedForm
  rw [Matrix.mulVec_sub, Matrix.mulVec_smul, hinv, hinv]
  ext i
  simp [ones]

/-- A positive definite quadratic form makes `A` invertible. -/
lemma isUnit_det_of_quad_pos {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ)
    (hpd : ∀ u : Fin M → ℝ, u ≠ 0 → 0 < u ⬝ᵥ (A *ᵥ u)) : IsUnit A.det := by
  rw [← Matrix.isUnit_iff_isUnit_det, ← Matrix.mulVec_injective_iff_isUnit]
  intro u v huv
  by_contra hne
  have h0 : A *ᵥ (u - v) = 0 := by
    rw [Matrix.mulVec_sub]
    change A *ᵥ u - A *ᵥ v = 0
    rw [huv, sub_self]
  have := hpd (u - v) (sub_ne_zero.mpr hne)
  rw [h0, dotProduct_zero] at this
  exact lt_irrefl _ this

/-- The inverse of a positive definite quadratic form is positive definite. -/
lemma inv_quad_pos {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ)
    (hpd : ∀ u : Fin M → ℝ, u ≠ 0 → 0 < u ⬝ᵥ (A *ᵥ u)) {v : Fin M → ℝ} (hv : v ≠ 0) :
    0 < v ⬝ᵥ (A⁻¹ *ᵥ v) := by
  have hdet := isUnit_det_of_quad_pos A hpd
  have hAu : A *ᵥ (A⁻¹ *ᵥ v) = v := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv A hdet, Matrix.one_mulVec]
  have hu0 : A⁻¹ *ᵥ v ≠ 0 := by
    intro h; rw [h, Matrix.mulVec_zero] at hAu; exact hv hAu.symm
  have := hpd _ hu0
  calc 0 < (A⁻¹ *ᵥ v) ⬝ᵥ (A *ᵥ (A⁻¹ *ᵥ v)) := this
    _ = v ⬝ᵥ (A⁻¹ *ᵥ v) := by rw [hAu, dotProduct_comm]

lemma ones_ne_zero {M : ℕ} (hM : 0 < M) : ones M ≠ 0 := by
  intro h
  have := congrFun h ⟨0, hM⟩
  simp [ones] at this

/-- **The closed form is the constrained minimizer** for every symmetric `A` with a
positive definite quadratic form (in particular `A = K + εI` with `ε > 0` at every
Gram, coincident nodes included). -/
theorem closedForm_isMin {M : ℕ} (hM : 0 < M) (A : Matrix (Fin M) (Fin M) ℝ) (hA : A.IsSymm)
    (hpd : ∀ u : Fin M → ℝ, u ≠ 0 → 0 < u ⬝ᵥ (A *ᵥ u)) (μh : Fin M → ℝ) :
    unitSum (closedForm A μh) ∧
      ∀ v, unitSum v → quadObj A μh (closedForm A μh) ≤ quadObj A μh v := by
  have hdet := isUnit_det_of_quad_pos A hpd
  have hpos : 0 < ones M ⬝ᵥ (A⁻¹ *ᵥ ones M) := inv_quad_pos A hpd (ones_ne_zero hM)
  have hsum := closedForm_unitSum A μh hpos.ne'
  refine ⟨hsum, quadObj_min_of_kkt A hA ?_ μh hsum (closedForm_kkt A hdet μh)⟩
  intro u
  by_cases hu : u = 0
  · simp [hu]
  · exact (hpd u hu).le

/-- The quadratic form of `K + εI` is positive definite for `ε > 0` and `K` PSD. -/
lemma quad_pos_add_smul_one {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ)
    (hpsd : ∀ u : Fin M → ℝ, 0 ≤ u ⬝ᵥ (K *ᵥ u)) {ε : ℝ} (hε : 0 < ε) :
    ∀ u : Fin M → ℝ, u ≠ 0 → 0 < u ⬝ᵥ ((K + ε • (1 : Matrix (Fin M) (Fin M) ℝ)) *ᵥ u) := by
  intro u hu
  rw [Matrix.add_mulVec, dotProduct_add, dot_smul_one_mulVec]
  have := hpsd u
  have := dot_self_pos_of_ne_zero hu
  nlinarith

/-- **§4.2, the deployed weights.** At every Gram (any node configuration, coincident
nodes included) and every estimand, the closed form with `A = K + εI`, `ε > 0`, is
the ridged constrained minimizer: two linear solves, unit sum, and no unit-sum
vector does better. -/
theorem deployed_weights (Φ : X → H) {M : ℕ} (hM : 0 < M) (z : Fin M → X) (μh : Fin M → ℝ)
    {ε : ℝ} (hε : 0 < ε) :
    IsRidgeMin (gram Φ z) μh ε (closedForm (gram Φ z + ε • (1 : Matrix (Fin M) (Fin M) ℝ)) μh) := by
  have hA := isSymm_add_smul_one (gram Φ z) (gram_isSymm Φ z) ε
  have hpd := quad_pos_add_smul_one (gram Φ z) (gram_psd Φ z) hε
  obtain ⟨hsum, hmin⟩ := closedForm_isMin hM _ hA hpd μh
  refine ⟨hsum, fun v hv => ?_⟩
  rw [ridgeCrit_eq_quadObj, ridgeCrit_eq_quadObj]
  exact hmin v hv

/-- Uniqueness of the minimizer for a symmetric positive definite quadratic form. -/
theorem ridgeMin_unique {M : ℕ} (A : Matrix (Fin M) (Fin M) ℝ) (hA : A.IsSymm)
    (hpd : ∀ u : Fin M → ℝ, u ≠ 0 → 0 < u ⬝ᵥ (A *ᵥ u))
    (μh : Fin M → ℝ) {w w' : Fin M → ℝ} (hw : unitSum w) (hw' : unitSum w')
    (hmin : ∀ v, unitSum v → quadObj A μh w ≤ quadObj A μh v)
    (hmin' : ∀ v, unitSum v → quadObj A μh w' ≤ quadObj A μh v)
    {c : ℝ} (hkkt : A *ᵥ w - μh = fun _ => c) : w' = w := by
  have hexp := quadObj_sub A hA μh w' w
  have hcross : (w' - w) ⬝ᵥ (A *ᵥ w - μh) = 0 := by
    rw [hkkt]; exact dot_const_of_unitSum hw' hw c
  have h1 := hmin w' hw'
  have h2 := hmin' w hw
  have hq : (w' - w) ⬝ᵥ (A *ᵥ (w' - w)) = 0 := by linarith
  by_contra hne
  have := hpd (w' - w) (sub_ne_zero.mpr hne)
  linarith

end QuadratureField
