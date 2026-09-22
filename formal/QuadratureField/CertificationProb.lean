import QuadratureField.Certification
import QuadratureField.ExactSelection
import QuadratureField.ProductSpace

/-!
# Theorem 1(ii): certified selection under an estimated kernel mean

The certification split `x ~ ρ^L` is drawn independently of everything used to
build the candidates, which are therefore fixed here (the paper's "condition on
the fitting data"). Two results:

* `certified_selection_of_events` — **Theorem 1(ii) modulo the concentration
  inequality**: given events `E_ab` of probability at least `1 − δ/3` on which
  the paired criterion error `|Δ̂_ab − Δ_ab|` is at most a slack `γ_ab(x)`, then
  on one event of probability at least `1 − δ` the returned candidate is within
  `γ_{â,0}` of the samples, and the choice between the two solved candidates is
  exact when its observed gap exceeds the slack and within the slack otherwise.
  The paper supplies the events by the empirical Bernstein inequality of
  Maurer & Pontil (2009) with `γ_ab = γ̂_ab(δ/3)`, which enters the Lean
  statement as the hypothesis on the events and is not re-proved here.

* `hoeffding_pair_event` — an **unconditional** instance: Hoeffding's inequality
  (from mathlib) certifies the same statements with the non-adaptive slack
  `2 D κ √(2 ln(2/δ)/L)`, where `D = MMD(Q_a, Q_b)` and `κ ≥ sup ‖Φ‖`. This is a
  fully machine-checked never-worse-within-slack guarantee under the estimated
  kernel mean; it is weaker than the paper's only in that its slack does not
  shrink with the empirical variance.
-/

noncomputable section
open scoped BigOperators InnerProductSpace NNReal
open Finset MeasureTheory ProbabilityTheory

namespace QuadratureField

variable {X : Type*} [MeasurableSpace X] {H : Type*} [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]

section Events

/-- The union bound for three events. -/
lemma three_events_union {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (A B C : Set Ω) {δ : ℝ} (hA : P.real Aᶜ ≤ δ / 3) (hB : P.real Bᶜ ≤ δ / 3)
    (hC : P.real Cᶜ ≤ δ / 3) : P.real (A ∩ B ∩ C)ᶜ ≤ δ := by
  rw [Set.compl_inter, Set.compl_inter]
  calc P.real (Aᶜ ∪ Bᶜ ∪ Cᶜ) ≤ P.real (Aᶜ ∪ Bᶜ) + P.real Cᶜ := measureReal_union_le _ _
    _ ≤ P.real Aᶜ + P.real Bᶜ + P.real Cᶜ := by gcongr; exact measureReal_union_le _ _
    _ ≤ δ / 3 + δ / 3 + δ / 3 := by gcongr
    _ = δ := by ring

variable (Φ : X → H) {M : ℕ}

/-- The three candidates: the samples, the reweighted samples, the moved quadrature. -/
def threeCand (z0 zhat : Fin M → X) (wrw wmv : Fin M → ℝ) : Fin 3 → Cand X M :=
  ![⟨z0, wEq M⟩, ⟨z0, wrw⟩, ⟨zhat, wmv⟩]

/-- **Theorem 1(ii), modulo the concentration inequality.** Let `Ĵ` be the split
criterion and `J` the exact one for the three candidates, `â` the split argmin, and
let `E₀ₐ, E_rw,mv` be events on which the paired criterion errors are within their
slacks. On the intersection (probability at least `1 − δ` when each event has
probability at least `1 − δ/3`):
(a) `J(â) ≤ J(Q₀) + γ_{â,0}`, and (b) between the two solved candidates the choice is
exact when the observed gap exceeds `γ_{rw,mv}`, and otherwise the excess over the
better of them is at most `γ_{rw,mv}`. -/
theorem certified_selection_of_events {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] {δ : ℝ}
    (Ĵ : Fin 3 → Ω → ℝ) (J : Fin 3 → ℝ) (γ : Fin 3 → Fin 3 → Ω → ℝ)
    (E : Fin 3 → Fin 3 → Set Ω)
    (hE : ∀ a b, ∀ ω ∈ E a b, |(Ĵ a ω - Ĵ b ω) - (J a - J b)| ≤ γ a b ω)
    (hP : ∀ a b, a ≠ b → P.real (E a b)ᶜ ≤ δ / 3)
    (ahat : Ω → Fin 3) (hsel : ∀ ω a, Ĵ (ahat ω) ω ≤ Ĵ a ω) :
    P.real (E 1 0 ∩ E 2 0 ∩ E 1 2)ᶜ ≤ δ ∧
      ∀ ω ∈ E 1 0 ∩ E 2 0 ∩ E 1 2,
        J (ahat ω) ≤ J 0 + (if ahat ω = 0 then 0 else γ (ahat ω) 0 ω) ∧
        ((γ 1 2 ω < |Ĵ 1 ω - Ĵ 2 ω| →
            ((Ĵ 1 ω ≤ Ĵ 2 ω → J 1 ≤ J 2) ∧ (Ĵ 2 ω ≤ Ĵ 1 ω → J 2 ≤ J 1))) ∧
          (Ĵ 1 ω ≤ Ĵ 2 ω → J 1 - min (J 1) (J 2) ≤ γ 1 2 ω) ∧
          (Ĵ 2 ω ≤ Ĵ 1 ω → J 2 - min (J 2) (J 1) ≤ γ 1 2 ω)) := by
  refine ⟨three_events_union P _ _ _ (hP 1 0 (by decide)) (hP 2 0 (by decide))
    (hP 1 2 (by decide)), ?_⟩
  rintro ω ⟨⟨h10, h20⟩, h12⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- against the samples
    by_cases h0 : ahat ω = 0
    · simp [h0]
    · have hconc : |(Ĵ (ahat ω) ω - Ĵ 0 ω) - (J (ahat ω) - J 0)| ≤ γ (ahat ω) 0 ω := by
        have : ahat ω = 1 ∨ ahat ω = 2 := by
          revert h0; generalize ahat ω = a; revert a; decide
        rcases this with h | h <;> rw [h] at * <;> [exact hE 1 0 ω h10; exact hE 2 0 ω h20]
      have := (selection_excess_of_conc hconc (hsel ω 0)).1
      simp [h0, this]
  · intro hgap
    exact certified_selection_exact (hE 1 2 ω h12) hgap
  · intro h
    exact (selection_excess_of_conc (hE 1 2 ω h12) h).2.1
  · intro h
    have hconc : |(Ĵ 2 ω - Ĵ 1 ω) - (J 2 - J 1)| ≤ γ 1 2 ω := by
      have h' := hE 1 2 ω h12
      rw [show (Ĵ 2 ω - Ĵ 1 ω) - (J 2 - J 1) = -((Ĵ 1 ω - Ĵ 2 ω) - (J 1 - J 2)) by ring, abs_neg]
      exact h'
    exact (selection_excess_of_conc hconc h).2.1


/-- The pairwise concentration bound is symmetric in the pair. -/
lemma conc_symm {Ĵa Ĵb Ja Jb γ : ℝ} (h : |(Ĵa - Ĵb) - (Ja - Jb)| ≤ γ) :
    |(Ĵb - Ĵa) - (Jb - Ja)| ≤ γ := by
  rw [show (Ĵb - Ĵa) - (Jb - Ja) = -((Ĵa - Ĵb) - (Ja - Jb)) by ring, abs_neg]; exact h

/-- **The excess over the best candidate** (`cs-cor-menu`(ii), first display): on the same
event, `J(â) ≤ min_a J(a) + max_{pairs} γ`. -/
theorem certified_selection_min {Ω : Type*} [MeasurableSpace Ω]
    (Ĵ : Fin 3 → Ω → ℝ) (J : Fin 3 → ℝ) (γ : Fin 3 → Fin 3 → Ω → ℝ)
    (E : Fin 3 → Fin 3 → Set Ω)
    (hE : ∀ a b, ∀ ω ∈ E a b, |(Ĵ a ω - Ĵ b ω) - (J a - J b)| ≤ γ a b ω)
    (ahat : Ω → Fin 3) (hsel : ∀ ω a, Ĵ (ahat ω) ω ≤ Ĵ a ω) :
    ∀ ω ∈ E 1 0 ∩ E 2 0 ∩ E 1 2, ∀ a,
      J (ahat ω) ≤ J a + max (γ 1 0 ω) (max (γ 2 0 ω) (γ 1 2 ω)) := by
  rintro ω ⟨⟨h10, h20⟩, h12⟩ a
  set G := max (γ 1 0 ω) (max (γ 2 0 ω) (γ 1 2 ω)) with hG
  have hγ10 : γ 1 0 ω ≤ G := le_max_left _ _
  have hγ20 : γ 2 0 ω ≤ G := le_trans (le_max_left _ _) (le_max_right _ _)
  have hγ12 : γ 1 2 ω ≤ G := le_trans (le_max_right _ _) (le_max_right _ _)
  have hG0 : 0 ≤ G := le_trans (abs_nonneg _) (le_trans (hE 1 0 ω h10) hγ10)
  -- the three pairwise bounds, in both orientations
  have b10 := hE 1 0 ω h10
  have b20 := hE 2 0 ω h20
  have b12 := hE 1 2 ω h12
  have b01 := conc_symm b10
  have b02 := conc_symm b20
  have b21 := conc_symm b12
  -- a generic step: from a concentration bound for the pair (x, a) and the argmin
  have step : ∀ (x : Fin 3) {γab : ℝ}, ahat ω = x →
      |(Ĵ x ω - Ĵ a ω) - (J x - J a)| ≤ γab → γab ≤ G → J (ahat ω) ≤ J a + G := by
    intro x γab hx hc hle
    subst hx
    have := (selection_excess_of_conc hc (hsel ω a)).1
    linarith
  -- case analysis on the pair (â, a)
  have key : ∀ x a : Fin 3, x = a ∨ (x = 0 ∧ a = 1) ∨ (x = 0 ∧ a = 2) ∨ (x = 1 ∧ a = 0) ∨
      (x = 1 ∧ a = 2) ∨ (x = 2 ∧ a = 0) ∨ (x = 2 ∧ a = 1) := by decide
  rcases key (ahat ω) a with h | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h]; linarith
  · subst h2; exact step 0 h1 b01 hγ10
  · subst h2; exact step 0 h1 b02 hγ20
  · subst h2; exact step 1 h1 b10 hγ10
  · subst h2; exact step 1 h1 b12 hγ12
  · subst h2; exact step 2 h1 b20 hγ20
  · subst h2; exact step 2 h1 b21 hγ12

end Events

section Hoeffding

variable (Φ : X → H) (ρ : Measure X) [IsProbabilityMeasure ρ] (hΦ : StronglyMeasurable Φ)
  {κ : ℝ} (hκ : ∀ x, ‖Φ x‖ ≤ κ)
include hΦ hκ

/-- The paired witness `g = h_a − h_b` integrates to `⟪emb_a − emb_b, m⟫`. -/
lemma integral_pairedWitness {M : ℕ} (za zb : Fin M → X) (wa wb : Fin M → ℝ) :
    ∫ y, (witness Φ za wa y - witness Φ zb wb y) ∂ρ =
      ⟪emb Φ za wa - emb Φ zb wb, kernelMean Φ ρ⟫_ℝ := by
  simp_rw [pairedWitness_eq]
  exact integral_inner (integrable_feature Φ hΦ hκ ρ) _


/-- **Theorem 1(ii), unconditional Hoeffding instance.** For fixed candidates `a, b` and a
split of `L` i.i.d. samples of `ρ`, with `D = MMD(Q_a, Q_b)` and `κ ≥ sup ‖Φ‖`, the paired
criterion error exceeds `2 D κ √(2 ln(2/δ)/L)` with probability at most `δ`. -/
theorem hoeffding_pair_event {M L : ℕ} (hL : 0 < L) (hκ0 : 0 ≤ κ) (za zb : Fin M → X)
    (wa wb : Fin M → ℝ) {δ : ℝ} (hδ : 0 < δ) (hδ1 : δ ≤ 1) :
    (piρ ρ L).real {x | 2 * ‖emb Φ za wa - emb Φ zb wb‖ * κ * Real.sqrt (2 * Real.log (2 / δ) / L)
        < |(crit (gram Φ za) (muHat Φ za x) wa - crit (gram Φ zb) (muHat Φ zb x) wb)
            - (critExact Φ za wa (kernelMean Φ ρ) - critExact Φ zb wb (kernelMean Φ ρ))|} ≤ δ := by
  set m := kernelMean Φ ρ with hm
  set g : X → ℝ := fun y => witness Φ za wa y - witness Φ zb wb y with hg
  set Eg : ℝ := ⟪emb Φ za wa - emb Φ zb wb, m⟫_ℝ with hEg
  set D : ℝ := ‖emb Φ za wa - emb Φ zb wb‖ with hD
  set b : ℝ := D * κ with hb
  set P := piρ ρ L with hP
  have hLpos : (0 : ℝ) < L := by exact_mod_cast hL
  have hL' : (L : ℝ) ≠ 0 := hLpos.ne'
  have hb0' : 0 ≤ b := mul_nonneg (norm_nonneg _) hκ0
  -- the paired witness: bounded, measurable, with mean `Eg`
  have hgb : ∀ y, |g y| ≤ b := fun y => witness_bound_of_bounded Φ za zb wa wb hκ y
  have hgm : StronglyMeasurable g := by
    have : g = fun y => ⟪emb Φ za wa - emb Φ zb wb, Φ y⟫_ℝ :=
      funext (pairedWitness_eq Φ za zb wa wb)
    rw [this]; exact stronglyMeasurable_const.inner hΦ
  have hEg' : ∫ y, g y ∂ρ = Eg := integral_pairedWitness Φ ρ hΦ hκ za zb wa wb
  -- the paired-difference identity
  set Y : Fin L → (Fin L → X) → ℝ := fun i x => g (x i) - Eg with hY
  have habs : ∀ x : Fin L → X,
      |(crit (gram Φ za) (muHat Φ za x) wa - crit (gram Φ zb) (muHat Φ zb x) wb)
        - (critExact Φ za wa m - critExact Φ zb wb m)| = (2 / L) * |∑ i, Y i x| := by
    intro x
    rw [paired_diff Φ hL za zb wa wb x m, abs_mul, abs_neg, abs_of_pos (by positivity)]
  set t : ℝ := b * Real.sqrt (2 * Real.log (2 / δ) / L) with ht
  have ht0 : 0 ≤ t := mul_nonneg hb0' (Real.sqrt_nonneg _)
  have hslack : 2 * D * κ * Real.sqrt (2 * Real.log (2 / δ) / L) = 2 * t := by rw [ht, hb]; ring
  -- reduce the event to the two tails of the sum
  have hsub : {x : Fin L → X | 2 * D * κ * Real.sqrt (2 * Real.log (2 / δ) / L)
      < |(crit (gram Φ za) (muHat Φ za x) wa - crit (gram Φ zb) (muHat Φ zb x) wb)
          - (critExact Φ za wa m - critExact Φ zb wb m)|} ⊆
      {x | (L : ℝ) * t ≤ ∑ i, Y i x} ∪ {x | (L : ℝ) * t ≤ ∑ i, -Y i x} := by
    intro x hx
    simp only [Set.mem_setOf_eq, Set.mem_union] at hx ⊢
    rw [habs, hslack] at hx
    have hlt : (L : ℝ) * t < |∑ i, Y i x| := by
      have := mul_lt_mul_of_pos_left hx (by positivity : (0 : ℝ) < L / 2)
      calc (L : ℝ) * t = L / 2 * (2 * t) := by ring
        _ < L / 2 * (2 / L * |∑ i, Y i x|) := this
        _ = |∑ i, Y i x| := by field_simp
    rcases le_or_gt 0 (∑ i, Y i x) with h | h
    · left; rw [abs_of_nonneg h] at hlt; exact hlt.le
    · right; rw [abs_of_neg h] at hlt; rw [Finset.sum_neg_distrib]; exact hlt.le
  by_cases hb0 : b = 0
  · -- the two candidates have the same witness: nothing to certify
    have hg0 : ∀ y, g y = 0 := fun y => abs_nonpos_iff.mp (by simpa [hb0] using hgb y)
    have hEg0 : Eg = 0 := by rw [← hEg']; simp [hg0]
    have hY0 : ∀ i x, Y i x = 0 := fun i x => by simp [hY, hg0, hEg0]
    have hempty : {x : Fin L → X | 2 * D * κ * Real.sqrt (2 * Real.log (2 / δ) / L)
        < |(crit (gram Φ za) (muHat Φ za x) wa - crit (gram Φ zb) (muHat Φ zb x) wb)
            - (critExact Φ za wa m - critExact Φ zb wb m)|} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_lt]
      rw [habs, hslack]
      simp [hY0, ht, hb0]
    rw [hempty]
    simp only [measureReal_empty]
    exact hδ.le
  · have hbpos : 0 < b := lt_of_le_of_ne hb0' (Ne.symm hb0)
    -- independence and sub-Gaussianity of the centred summands
    have hind : iIndepFun Y P := by
      have h := (coord_indep ρ L).comp (fun _ : Fin L => fun y : X => g y - Eg)
        (fun _ => (hgm.sub stronglyMeasurable_const).measurable)
      exact h
    set c : ℝ≥0 := (‖b - (-b)‖₊ / 2) ^ 2 with hc
    have hcR : (c : ℝ) = b ^ 2 := by
      rw [hc]
      push_cast
      rw [Real.norm_eq_abs, abs_of_pos (by linarith)]
      ring
    have hsubG : ∀ i ∈ (univ : Finset (Fin L)), HasSubgaussianMGF (Y i) c P := by
      intro i _
      have h := hasSubgaussianMGF_of_mem_Icc (μ := P) (X := fun x => g (x i)) (a := -b) (b := b)
        (hgm.comp_measurable (measurable_pi_apply i)).aemeasurable
        (ae_of_all _ fun x => abs_le.mp (hgb (x i)))
      have hmean : ∫ x, g (x i) ∂P = Eg := by rw [integral_coord ρ hgm i]; exact hEg'
      rw [hmean] at h
      exact h
    have hsubGneg : ∀ i ∈ (univ : Finset (Fin L)), HasSubgaussianMGF (fun x => -Y i x) c P :=
      fun i hi => (hsubG i hi).neg
    have hindneg : iIndepFun (fun i x => -Y i x) P :=
      hind.comp (fun _ => fun r : ℝ => -r) (fun _ => measurable_neg)
    have hε : 0 ≤ (L : ℝ) * t := by positivity
    have hup := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hind hsubG hε
    have hdown := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindneg hsubGneg hε
    -- the exponent
    have hlog : 0 ≤ Real.log (2 / δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
    have hsq : Real.sqrt (2 * Real.log (2 / δ) / L) ^ 2 = 2 * Real.log (2 / δ) / L :=
      Real.sq_sqrt (by positivity)
    have hexp : Real.exp (-((L : ℝ) * t) ^ 2 / (2 * ((L : ℝ) * b ^ 2))) = δ / 2 := by
      have : -((L : ℝ) * t) ^ 2 / (2 * ((L : ℝ) * b ^ 2)) = -Real.log (2 / δ) := by
        rw [ht, mul_pow, mul_pow, hsq]
        field_simp
        try ring
      rw [this, Real.exp_neg, Real.exp_log (by positivity), inv_div]
    have hsum : ((∑ _i ∈ (univ : Finset (Fin L)), c : ℝ≥0) : ℝ) = (L : ℝ) * b ^ 2 := by
      rw [NNReal.coe_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hcR]
    have hup' : P.real {x | (L : ℝ) * t ≤ ∑ i, Y i x} ≤ δ / 2 := by
      refine hup.trans (le_of_eq ?_)
      rw [hsum]
      exact hexp
    have hdown' : P.real {x | (L : ℝ) * t ≤ ∑ i, -Y i x} ≤ δ / 2 := by
      refine hdown.trans (le_of_eq ?_)
      rw [hsum]
      exact hexp
    calc P.real _ ≤ P.real ({x | (L : ℝ) * t ≤ ∑ i, Y i x} ∪ {x | (L : ℝ) * t ≤ ∑ i, -Y i x}) :=
          measureReal_mono hsub
      _ ≤ P.real {x | (L : ℝ) * t ≤ ∑ i, Y i x} + P.real {x | (L : ℝ) * t ≤ ∑ i, -Y i x} :=
          measureReal_union_le _ _
      _ ≤ δ / 2 + δ / 2 := add_le_add hup' hdown'
      _ = δ := by ring

end Hoeffding


end QuadratureField
