import Mathlib

/-!
# The setting: features, kernel, quadratures, and the worst-case error

The abstract model of §3.1 of the paper. `H` is a real Hilbert space (the
RKHS), `Φ : X → H` a feature map, so the kernel is `k x y = ⟪Φ x, Φ y⟫`. A
quadrature is a node vector `z : Fin M → X` (coincident nodes allowed) with
signed weights `w : Fin M → ℝ`; its embedding is `∑ j, w j • Φ (z j)`, and
its squared discrepancy to a reference whose kernel mean is the point
`m : H` is `‖emb − m‖²`.

Statements proved here (all pure Hilbert-space algebra, no hypothesis on
`k` beyond being induced by a feature map, i.e. positive semidefinite):

* `mmdSq_expand` — display (1) of the paper: the Gram term, the kernel-mean
  term and the self-affinity `‖m‖²`.
* `worstCase_isGreatest` — the worst case of the integration error over the
  unit ball of `H` is the discrepancy (the cited fact behind display (1)).
* `crit_exact` — the criterion `Ĵ = wᵀKw − 2wᵀμ` equals `MMD² − c_ρ` under the
  exact kernel mean: the self-affinity cancels from every comparison.
* `mmd_triangle` — Remark 1: the error against the posterior behind the
  reference is bounded by the sum of the two distances.
* `integrates_constants` — unit-sum weights integrate constants exactly.
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset Matrix

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The kernel induced by the feature map `Φ`. -/
def kernel (Φ : X → H) (x y : X) : ℝ := ⟪Φ x, Φ y⟫_ℝ

/-- Embedding of the quadrature `∑ j, w j δ_{z j}`. -/
def emb (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) : H :=
  ∑ j, w j • Φ (z j)

/-- Squared discrepancy of the quadrature to a reference with kernel mean `m`. -/
def mmdSq (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) : ℝ :=
  ‖emb Φ z w - m‖ ^ 2

/-- The Gram matrix at the nodes. -/
def gram (Φ : X → H) {M : ℕ} (z : Fin M → X) : Matrix (Fin M) (Fin M) ℝ :=
  fun i j => kernel Φ (z i) (z j)

/-- The exact estimand: the kernel mean of the reference read at the nodes. -/
def meanVec (Φ : X → H) {M : ℕ} (z : Fin M → X) (m : H) : Fin M → ℝ :=
  fun j => ⟪Φ (z j), m⟫_ℝ

/-- The criterion `Ĵ(w) = wᵀ K w − 2 wᵀ μh` for a Gram `K` and an estimand vector `μh`
(exact or estimated). -/
def crit {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (μh : Fin M → ℝ) (w : Fin M → ℝ) : ℝ :=
  w ⬝ᵥ (K *ᵥ w) - 2 * (w ⬝ᵥ μh)

/-- Unit-sum (signed) weights. -/
def unitSum {M : ℕ} (w : Fin M → ℝ) : Prop := ∑ j, w j = 1

/-- Equal weights. -/
def wEq (M : ℕ) : Fin M → ℝ := fun _ => (1 : ℝ) / M

lemma kernel_comm (Φ : X → H) (x y : X) : kernel Φ x y = kernel Φ y x := by
  simp [kernel, real_inner_comm]

lemma kernel_self (Φ : X → H) (x : X) : kernel Φ x x = ‖Φ x‖ ^ 2 := by
  simp [kernel]

lemma inner_emb_left (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (v : H) :
    ⟪emb Φ z w, v⟫_ℝ = ∑ j, w j * ⟪Φ (z j), v⟫_ℝ := by
  simp [emb, sum_inner, real_inner_smul_left]

lemma inner_emb_right (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (v : H) :
    ⟪v, emb Φ z w⟫_ℝ = ∑ j, w j * ⟪v, Φ (z j)⟫_ℝ := by
  simp [emb, inner_sum, real_inner_smul_right]

/-- The Gram term: `‖∑ w_j Φ(z_j)‖² = ∑_{i,j} w_i w_j k(z_i, z_j)`. -/
lemma norm_emb_sq (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) :
    ‖emb Φ z w‖ ^ 2 = ∑ i, ∑ j, w i * w j * kernel Φ (z i) (z j) := by
  rw [← real_inner_self_eq_norm_sq, inner_emb_left]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [inner_emb_right, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [kernel]; ring

/-- **Display (1).** -/
theorem mmdSq_expand (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) :
    mmdSq Φ z w m =
      ∑ i, ∑ j, w i * w j * kernel Φ (z i) (z j)
        - 2 * ∑ j, w j * ⟪Φ (z j), m⟫_ℝ + ‖m‖ ^ 2 := by
  unfold mmdSq
  rw [norm_sub_sq_real, norm_emb_sq, inner_emb_left]

/-- The criterion in matrix form is the same double sum. -/
lemma crit_eq_sums {M : ℕ} (K : Matrix (Fin M) (Fin M) ℝ) (μh w : Fin M → ℝ) :
    crit K μh w = ∑ i, ∑ j, w i * w j * K i j - 2 * ∑ j, w j * μh j := by
  unfold crit
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

/-- **The self-affinity cancels.** Under the exact kernel mean the criterion is the
discrepancy minus `c_ρ = ‖m‖²`, for every node configuration and every weight
vector; hence the criterion ranks candidates exactly as the discrepancy does. -/
theorem crit_exact (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) :
    crit (gram Φ z) (meanVec Φ z m) w = mmdSq Φ z w m - ‖m‖ ^ 2 := by
  rw [crit_eq_sums, mmdSq_expand]
  simp only [gram, meanVec]
  ring

/-- The integration error of `f ∈ H` (with `f(x) = ⟪f, Φ x⟫`, the reproducing
property, and `∫ f dρ = ⟪f, m⟫`) is the inner product with `emb − m`. -/
lemma error_eq_inner (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) (f : H) :
    (∑ j, w j * ⟪f, Φ (z j)⟫_ℝ) - ⟪f, m⟫_ℝ = ⟪f, emb Φ z w - m⟫_ℝ := by
  rw [inner_sub_right, inner_emb_right]

/-- **The worst case over the unit ball is the discrepancy** (the cited identity
behind display (1)): `sup_{‖f‖ ≤ 1} |⟪f, v⟫| = ‖v‖`, attained. -/
theorem worstCase_isGreatest (v : H) :
    IsGreatest ((fun f : H => |⟪f, v⟫_ℝ|) '' Metric.closedBall (0 : H) 1) ‖v‖ := by
  constructor
  · by_cases hv : v = 0
    · refine ⟨0, by simp, ?_⟩
      simp [hv]
    · refine ⟨(‖v‖⁻¹) • v, ?_, ?_⟩
      · simp only [Metric.mem_closedBall, dist_zero_right, norm_smul, norm_inv, norm_norm]
        rw [inv_mul_cancel₀ (norm_ne_zero_iff.mpr hv)]
      · simp only [real_inner_smul_left, real_inner_self_eq_norm_sq]
        rw [abs_of_nonneg (by positivity)]
        field_simp
  · rintro _ ⟨f, hf, rfl⟩
    simp only [Metric.mem_closedBall, dist_zero_right] at hf
    calc |⟪f, v⟫_ℝ| ≤ ‖f‖ * ‖v‖ := abs_real_inner_le_norm f v
      _ ≤ 1 * ‖v‖ := by gcongr
      _ = ‖v‖ := one_mul _

/-- Worst-case form of the error of a quadrature: for every `f` in the unit ball
the error is at most `√(MMD²)`, with equality attained. -/
theorem worstCase_error (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) :
    IsGreatest ((fun f : H => |(∑ j, w j * ⟪f, Φ (z j)⟫_ℝ) - ⟪f, m⟫_ℝ|) ''
      Metric.closedBall (0 : H) 1) (Real.sqrt (mmdSq Φ z w m)) := by
  have h := worstCase_isGreatest (emb Φ z w - m)
  have hsq : Real.sqrt (mmdSq Φ z w m) = ‖emb Φ z w - m‖ := by
    unfold mmdSq; exact Real.sqrt_sq (norm_nonneg _)
  have hfun : (fun f : H => |(∑ j, w j * ⟪f, Φ (z j)⟫_ℝ) - ⟪f, m⟫_ℝ|) =
      fun f : H => |⟪f, emb Φ z w - m⟫_ℝ| := by
    funext f; rw [error_eq_inner]
  rw [hfun, hsq]; exact h

/-- **Remark 1.** With `mρ` the kernel mean of the reference and `mπ` that of the
posterior behind it, the error against the posterior is at most the error against
the reference plus the distance between the two. -/
theorem mmd_triangle (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (mρ mπ : H) :
    ‖emb Φ z w - mπ‖ ≤ ‖emb Φ z w - mρ‖ + ‖mρ - mπ‖ :=
  norm_sub_le_norm_sub_add_norm_sub _ _ _

/-- Unit-sum weights integrate constants exactly. -/
theorem integrates_constants {M : ℕ} (w : Fin M → ℝ) (hw : unitSum w) (c : ℝ) :
    ∑ j, w j * c = c := by
  rw [← Finset.sum_mul, hw, one_mul]

end QuadratureField
