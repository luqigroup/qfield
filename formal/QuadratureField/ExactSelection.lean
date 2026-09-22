import QuadratureField.Setting

/-!
# Theorem 1(i): exact selection is never worse than the samples

Paper §4.4 and Theorem 1(i) (`sg-thm-safeguard`(1), `sg-eq-selchain`). A finite
family of candidate quadratures is scored by the criterion `Ĵ` under the exact
kernel mean; the candidate with the smallest score is returned. Because
`Ĵ = MMD² − c_ρ` for every candidate (`crit_exact`), the returned candidate has
the smallest discrepancy, and in particular is no worse than the samples when
the samples are a candidate.

Nothing is assumed about how the candidates were produced: the moved nodes may
come from any network, trained, untrained or adversarial, at any observation and
any node count; nodes may coincide; the kernel is any feature-map kernel (no unit
diagonal is used).
-/

noncomputable section
open scoped BigOperators InnerProductSpace
open Finset

namespace QuadratureField

variable {X : Type*} {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- A candidate quadrature with `M` nodes. -/
structure Cand (X : Type*) (M : ℕ) where
  z : Fin M → X
  w : Fin M → ℝ

/-- The criterion of a candidate under the exact kernel mean `m`. -/
def score (Φ : X → H) (m : H) {M : ℕ} (c : Cand X M) : ℝ :=
  crit (gram Φ c.z) (meanVec Φ c.z m) c.w

/-- The discrepancy of a candidate. -/
def mmdSqC (Φ : X → H) (m : H) {M : ℕ} (c : Cand X M) : ℝ := mmdSq Φ c.z c.w m

theorem score_eq (Φ : X → H) (m : H) {M : ℕ} (c : Cand X M) :
    score Φ m c = mmdSqC Φ m c - ‖m‖ ^ 2 :=
  crit_exact Φ c.z c.w m

/-- The argmin of the criterion exists over a finite nonempty candidate family. -/
theorem exists_selection {ι : Type*} [Fintype ι] [Nonempty ι] (Φ : X → H) (m : H) {M : ℕ}
    (cand : ι → Cand X M) : ∃ a_hat, ∀ a, score Φ m (cand a_hat) ≤ score Φ m (cand a) := by
  obtain ⟨a, -, ha⟩ := Finset.exists_min_image univ (fun a => score Φ m (cand a)) univ_nonempty
  exact ⟨a, fun b => ha b (mem_univ b)⟩

/-- **Theorem 1(i), general form.** The criterion's argmin is the discrepancy's argmin:
`MMD²(Q_â) ≤ MMD²(Q_a)` for every candidate `a`. -/
theorem exact_selection {ι : Type*} (Φ : X → H) (m : H) {M : ℕ} (cand : ι → Cand X M)
    (a_hat : ι) (hsel : ∀ a, score Φ m (cand a_hat) ≤ score Φ m (cand a)) :
    ∀ a, mmdSqC Φ m (cand a_hat) ≤ mmdSqC Φ m (cand a) := by
  intro a
  have h := hsel a
  rw [score_eq, score_eq] at h
  linarith

/-- The returned discrepancy is the least over the candidates. -/
theorem exact_selection_isLeast {ι : Type*} (Φ : X → H) (m : H) {M : ℕ} (cand : ι → Cand X M)
    (a_hat : ι) (hsel : ∀ a, score Φ m (cand a_hat) ≤ score Φ m (cand a)) :
    IsLeast (Set.range fun a => mmdSqC Φ m (cand a)) (mmdSqC Φ m (cand a_hat)) :=
  ⟨⟨a_hat, rfl⟩, by rintro _ ⟨a, rfl⟩; exact exact_selection Φ m cand a_hat hsel a⟩

/-- **Theorem 1(i) as printed: never worse than the samples.** With the samples
`Q_0 = (z⁰, 1/M)` a candidate, the returned candidate has discrepancy at most that of
the samples, for any nodes `ẑ` (any network) and any weights on them. -/
theorem neverWorse_exact (Φ : X → H) (m : H) {M : ℕ} (z0 zhat : Fin M → X)
    (wrw wmv : Fin M → ℝ) :
    let cand : Fin 3 → Cand X M := ![⟨z0, wEq M⟩, ⟨z0, wrw⟩, ⟨zhat, wmv⟩]
    ∀ a_hat : Fin 3, (∀ a, score Φ m (cand a_hat) ≤ score Φ m (cand a)) →
      mmdSqC Φ m (cand a_hat) ≤ mmdSq Φ z0 (wEq M) m ∧
        ∀ a, mmdSqC Φ m (cand a_hat) ≤ mmdSqC Φ m (cand a) := by
  intro cand a_hat hsel
  refine ⟨?_, exact_selection Φ m cand a_hat hsel⟩
  have := exact_selection Φ m cand a_hat hsel 0
  simpa [cand, mmdSqC] using this

/-- The degenerate case of an untrained network: the moved candidate coincides with the
reweighted one, and the selection is no worse than the reweighted samples. -/
theorem neverWorse_exact_untrained (Φ : X → H) (m : H) {M : ℕ} (z0 : Fin M → X)
    (wrw : Fin M → ℝ) :
    let cand : Fin 3 → Cand X M := ![⟨z0, wEq M⟩, ⟨z0, wrw⟩, ⟨z0, wrw⟩]
    ∀ a_hat : Fin 3, (∀ a, score Φ m (cand a_hat) ≤ score Φ m (cand a)) →
      mmdSqC Φ m (cand a_hat) ≤ mmdSq Φ z0 wrw m := by
  intro cand a_hat hsel
  have := exact_selection Φ m cand a_hat hsel 1
  simpa [cand, mmdSqC] using this

end QuadratureField
