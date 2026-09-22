import QuadratureField.Setting

/-!
# The certification split, deterministic part

Paper §5.2, Theorem 1(ii), and the engine lemma `cs-lem-engine` behind it.
The split points `x : Fin L → X` are fixed here (the theorem's probability
statement conditions on the fitting data and is added in
`CertificationProb.lean`); everything in this file is an identity or an
inequality that holds at every realization of the split.

* `witness_eq` — `h_Q(y) = ⟪emb Q, Φ y⟫ = ∑ w_j k(z_j, y)`.
* `engine_identity` — `Ĵ_split(Q) − J(Q) = −(2/L) ∑ᵢ (h_Q(xᵢ) − E h_Q)`: the Gram
  term cancels and the criterion error is a centred mean of one bounded
  function.
* `paired_diff` — for two candidates scored on the same split,
  `Δ̂ − Δ = −(2/L) ∑ᵢ (g(xᵢ) − E g)` with `g = h_a − h_b`.
* `witness_bound` — `|g(y)| ≤ D ‖Φ y‖` with `D = MMD(Q_a, Q_b)`; under a bounded
  kernel `sup k(x,x) ≤ κ²` this is `D κ`, and `κ = 1` for the paper's kernels.
  (This is where the bounded kernel enters: the "lighter assumption".)
* `distSq_gram_form` — `D²` as the cross-Gram form of the paper.
* `selection_excess_of_conc`, `certified_sign_of_conc` — the selection logic on
  the concentration event `|Δ̂ − Δ| ≤ γ`: one slack, and exactness when the
  observed gap exceeds the slack.
* `slack_zero_of_coincide` — if the two candidates have the same embedding, the
  paired witness vanishes identically, so the sample variance and the slack
  are identically zero (the untrained-network degenerate case).
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The plug-in estimand from the split: `μ̂(z)_j = (1/L) ∑ᵢ k(z_j, xᵢ)`. -/
def muHat (Φ : X → H) {M L : ℕ} (z : Fin M → X) (x : Fin L → X) : Fin M → ℝ :=
  fun j => (1 / (L : ℝ)) * ∑ i, kernel Φ (z j) (x i)

/-- The witness function of a quadrature: `h_Q(y) = ∑ w_j k(z_j, y)`. -/
def witness (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (y : X) : ℝ :=
  ∑ j, w j * kernel Φ (z j) y

lemma witness_eq (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (y : X) :
    witness Φ z w y = ⟪emb Φ z w, Φ y⟫_ℝ := by
  rw [inner_emb_left]; rfl

/-- The exact criterion `J(Q) = MMD² − c_ρ`, written with the kernel mean `m`. -/
def critExact (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) : ℝ :=
  mmdSq Φ z w m - ‖m‖ ^ 2

/-- `E h_Q(X) = ⟪emb Q, m⟫` is the exact kernel-mean term. -/
lemma meanWitness_eq (Φ : X → H) {M : ℕ} (z : Fin M → X) (w : Fin M → ℝ) (m : H) :
    ∑ j, w j * meanVec Φ z m j = ⟪emb Φ z w, m⟫_ℝ := by
  rw [inner_emb_left]; rfl

/-- **The engine identity** (`cs-eq-engine`): the split criterion minus the exact one is
`−(2/L) ∑ᵢ (h_Q(xᵢ) − ⟪emb Q, m⟫)`. -/
theorem engine_identity (Φ : X → H) {M L : ℕ} (hL : 0 < L) (z : Fin M → X) (w : Fin M → ℝ)
    (x : Fin L → X) (m : H) :
    crit (gram Φ z) (muHat Φ z x) w - critExact Φ z w m =
      -(2 / (L : ℝ)) * ∑ i, (witness Φ z w (x i) - ⟪emb Φ z w, m⟫_ℝ) := by
  have hL' : (L : ℝ) ≠ 0 := by exact_mod_cast hL.ne'
  unfold critExact
  rw [← crit_exact, crit_eq_sums, crit_eq_sums]
  simp only [muHat, meanVec, gram, witness]
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  rw [← inner_emb_left]
  simp only [kernel]
  -- both sides are linear in the finite sums; rearrange
  have hswap : ∑ j, w j * ((1 / (L : ℝ)) * ∑ i, ⟪Φ (z j), Φ (x i)⟫_ℝ)
      = (1 / (L : ℝ)) * ∑ i, ∑ j, w j * ⟪Φ (z j), Φ (x i)⟫_ℝ := by
    simp_rw [Finset.mul_sum]
    conv_lhs => rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [hswap]
  field_simp
  ring

/-- The paired difference of two candidates scored on the same split
(`cs-eq-diff`): `Δ̂ − Δ = −(2/L) ∑ᵢ (g(xᵢ) − E g)`, `g = h_a − h_b`. -/
theorem paired_diff (Φ : X → H) {M L : ℕ} (hL : 0 < L) (za zb : Fin M → X) (wa wb : Fin M → ℝ)
    (x : Fin L → X) (m : H) :
    (crit (gram Φ za) (muHat Φ za x) wa - crit (gram Φ zb) (muHat Φ zb x) wb)
      - (critExact Φ za wa m - critExact Φ zb wb m) =
      -(2 / (L : ℝ)) * ∑ i,
        ((witness Φ za wa (x i) - witness Φ zb wb (x i)) - ⟪emb Φ za wa - emb Φ zb wb, m⟫_ℝ) := by
  have ha := engine_identity Φ hL za wa x m
  have hb := engine_identity Φ hL zb wb x m
  have : ∑ i, ((witness Φ za wa (x i) - witness Φ zb wb (x i))
      - ⟪emb Φ za wa - emb Φ zb wb, m⟫_ℝ)
      = ∑ i, (witness Φ za wa (x i) - ⟪emb Φ za wa, m⟫_ℝ)
        - ∑ i, (witness Φ zb wb (x i) - ⟪emb Φ zb wb, m⟫_ℝ) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [inner_sub_left]; ring
  rw [this]
  linarith

/-- The paired witness `g = h_a − h_b` is the inner product with the embedding difference. -/
lemma pairedWitness_eq (Φ : X → H) {M : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ) (y : X) :
    witness Φ za wa y - witness Φ zb wb y = ⟪emb Φ za wa - emb Φ zb wb, Φ y⟫_ℝ := by
  rw [witness_eq, witness_eq, inner_sub_left]

/-- **The witness bound** (`cs-eq-supg`): `|g(y)| ≤ MMD(Q_a, Q_b) ‖Φ y‖`. -/
theorem witness_bound (Φ : X → H) {M : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ) (y : X) :
    |witness Φ za wa y - witness Φ zb wb y| ≤ ‖emb Φ za wa - emb Φ zb wb‖ * ‖Φ y‖ := by
  rw [pairedWitness_eq]
  exact abs_real_inner_le_norm _ _

/-- Under a bounded kernel `k(y,y) ≤ κ²`, `|g(y)| ≤ D κ`; for the paper's unit-diagonal
kernels `κ = 1`. -/
theorem witness_bound_of_bounded (Φ : X → H) {M : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ)
    {κ : ℝ} (hκ : ∀ y, ‖Φ y‖ ≤ κ) (y : X) :
    |witness Φ za wa y - witness Φ zb wb y| ≤ ‖emb Φ za wa - emb Φ zb wb‖ * κ :=
  (witness_bound Φ za zb wa wb y).trans
    (mul_le_mul_of_nonneg_left (hκ y) (norm_nonneg _))

/-- `D² = MMD²(Q_a, Q_b)` in its cross-Gram form. -/
theorem distSq_gram_form (Φ : X → H) {M : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ) :
    ‖emb Φ za wa - emb Φ zb wb‖ ^ 2 =
      ∑ i, ∑ j, wa i * wa j * kernel Φ (za i) (za j)
        - 2 * ∑ i, ∑ j, wa i * wb j * kernel Φ (za i) (zb j)
        + ∑ i, ∑ j, wb i * wb j * kernel Φ (zb i) (zb j) := by
  rw [norm_sub_sq_real, norm_emb_sq, norm_emb_sq, inner_emb_left]
  have hmid : ∑ i, wa i * ⟪Φ (za i), emb Φ zb wb⟫_ℝ
      = ∑ i, ∑ j, wa i * wb j * kernel Φ (za i) (zb j) := by
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [inner_emb_right, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [kernel]; ring
  rw [hmid]

/-! ### The selection logic on the concentration event -/

/-- **One-slack selection excess** (`cs-eq-selexcess`). If `|Δ̂ − Δ| ≤ γ` and candidate `a`
is selected (`Ĵ_a ≤ Ĵ_b`), then `J_a ≤ J_b + γ`; and the excess of the selected candidate
over the better of the two is at most `γ`, and zero when the selection is correct. -/
theorem selection_excess_of_conc {Ja Jb Ĵa Ĵb γ : ℝ}
    (hconc : |(Ĵa - Ĵb) - (Ja - Jb)| ≤ γ) (hsel : Ĵa ≤ Ĵb) :
    Ja ≤ Jb + γ ∧ Ja - min Ja Jb ≤ γ ∧ (Ja ≤ Jb → Ja - min Ja Jb = 0) := by
  have h1 := (abs_le.mp hconc).1
  have hγ : 0 ≤ γ := (abs_nonneg _).trans hconc
  refine ⟨by linarith, ?_, fun h => by rw [min_eq_left h]; ring⟩
  rcases le_total Ja Jb with h | h
  · rw [min_eq_left h]; linarith
  · rw [min_eq_right h]; linarith

/-- **Certified sign** (`cs-thm-certsel`(3)). If `|Δ̂ − Δ| ≤ γ` and `|Δ̂| > γ`, then `Δ` has
the sign of `Δ̂`: the selection is exactly correct. -/
theorem certified_sign_of_conc {Δ Dh γ : ℝ} (hconc : |Dh - Δ| ≤ γ) (hgap : γ < |Dh|) :
    (0 < Dh → 0 < Δ) ∧ (Dh < 0 → Δ < 0) := by
  have h := abs_le.mp hconc
  constructor
  · intro hpos
    rw [abs_of_pos hpos] at hgap
    linarith
  · intro hneg
    rw [abs_of_neg hneg] at hgap
    linarith

/-- Exact selection between two candidates on the certified event: the one with the
smaller split criterion has the smaller true discrepancy. -/
theorem certified_selection_exact {Ja Jb Ĵa Ĵb γ : ℝ}
    (hconc : |(Ĵa - Ĵb) - (Ja - Jb)| ≤ γ) (hgap : γ < |Ĵa - Ĵb|) :
    (Ĵa ≤ Ĵb → Ja ≤ Jb) ∧ (Ĵb ≤ Ĵa → Jb ≤ Ja) := by
  have h := certified_sign_of_conc hconc hgap
  constructor
  · intro hab
    rcases lt_or_eq_of_le hab with hlt | heq
    · have := h.2 (by linarith); linarith
    · rw [heq, sub_self, abs_zero] at hgap
      linarith [abs_nonneg ((Ĵa - Ĵb) - (Ja - Jb))]
  · intro hba
    rcases lt_or_eq_of_le hba with hlt | heq
    · have := h.1 (by linarith); linarith
    · rw [heq, sub_self, abs_zero] at hgap
      linarith [abs_nonneg ((Ĵa - Ĵb) - (Ja - Jb))]

/-- The unbiased sample variance of the paired witness over the split. -/
def sampleVar {L : ℕ} (g : Fin L → ℝ) : ℝ :=
  (1 / ((L : ℝ) - 1)) * ∑ i, (g i - (1 / (L : ℝ)) * ∑ i', g i') ^ 2

/-- The paper's slack `γ̂(δ) = 2√(2 V̂ ln(4/δ)/L) + (28/3) D ln(4/δ)/(L − 1)`. -/
def slack (L : ℕ) (V D δ : ℝ) : ℝ :=
  2 * Real.sqrt (2 * V * Real.log (4 / δ) / L) + (28 / 3) * D * Real.log (4 / δ) / ((L : ℝ) - 1)

/-- **The degenerate case.** If the two candidates have the same embedding (an untrained
network: moved and reweighted candidates coincide), the paired witness vanishes at every
split point, so `D = 0`, `V̂ = 0` and the slack is identically zero. -/
theorem slack_zero_of_coincide (Φ : X → H) {M L : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ)
    (hcoin : emb Φ za wa = emb Φ zb wb) (x : Fin L → X) (δ : ℝ) :
    (∀ i, witness Φ za wa (x i) - witness Φ zb wb (x i) = 0) ∧
      ‖emb Φ za wa - emb Φ zb wb‖ = 0 ∧
      sampleVar (fun i => witness Φ za wa (x i) - witness Φ zb wb (x i)) = 0 ∧
      slack L (sampleVar (fun i => witness Φ za wa (x i) - witness Φ zb wb (x i)))
        ‖emb Φ za wa - emb Φ zb wb‖ δ = 0 := by
  have hg : ∀ i, witness Φ za wa (x i) - witness Φ zb wb (x i) = 0 := by
    intro i; rw [pairedWitness_eq, hcoin, sub_self, inner_zero_left]
  have hD : ‖emb Φ za wa - emb Φ zb wb‖ = 0 := by rw [hcoin, sub_self, norm_zero]
  have hV : sampleVar (fun i => witness Φ za wa (x i) - witness Φ zb wb (x i)) = 0 := by
    simp [sampleVar, hg]
  refine ⟨hg, hD, hV, ?_⟩
  rw [hV, hD]
  simp [slack]

end QuadratureField
