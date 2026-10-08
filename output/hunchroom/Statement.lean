∀ (n : ℕ), 2 ≤ n →
∀ (Θ : Type)
  (C : List (Fin n) → Set (ℕ → Fin n))
  [m : MeasurableSpace (ℕ → Fin n)],
  (∀ h, C h = {ω | ∀ k : Fin h.length, ω k.val = h.get k}) →
  m = MeasurableSpace.generateFrom (Set.range C) →
∀ (p : Θ → List (Fin n) → Fin n → ℝ)
  (w : Θ → List (Fin n) → Fin n)
  (P : Θ → MeasureTheory.Measure (ℕ → Fin n)),
  (∀ β h i, 0 < p β h i) →
  (∀ β h, Finset.sum Finset.univ (fun i : Fin n => p β h i) = 1) →
  (∀ β h i, p β h i ≤ p β h (w β h)) →
  (∀ β, P β Set.univ = 1) →
  (∀ β h i, P β (C (h ++ [i])) =
    P β (C h) * ENNReal.ofReal (p β h i)) →
  (∀ β β', ∃ K : ℝ, 0 ≤ K ∧ ∀ h,
    Finset.sum Finset.univ
      (fun i : Fin n => (Real.sqrt (p β h i) - Real.sqrt (p β' h i)) ^ 2) ≤
    K * (1 - p β h (w β h))) →
  (∀ β, P β {ω | ¬ Summable (fun t : ℕ =>
    1 - p β (List.ofFn (fun k : Fin t => ω k.val))
      (w β (List.ofFn (fun k : Fin t => ω k.val))))} = 0) →
  (∀ β β', (P β).AbsolutelyContinuous (P β') ∧
    (P β').AbsolutelyContinuous (P β)) ∧
  (∀ F : Θ → ℝ, (∃ β β', F β ≠ F β') →
    ¬ ∃ est : ℕ → List (Fin n) → ℝ,
      ∀ (β : Θ) (ε : ℝ), 0 < ε →
        Filter.Tendsto
          (fun t : ℕ => P β {ω |
            ε ≤ abs (est t (List.ofFn (fun k : Fin t => ω k.val)) - F β)})
          Filter.atTop (nhds 0)) ∧
  (∀ β β', ¬ ∃ test : ℕ → List (Fin n) → Bool,
    Filter.Tendsto
      (fun t : ℕ => P β {ω |
        test t (List.ofFn (fun k : Fin t => ω k.val)) = true})
      Filter.atTop (nhds 0) ∧
    Filter.Tendsto
      (fun t : ℕ => P β' {ω |
        test t (List.ofFn (fun k : Fin t => ω k.val)) = false})
      Filter.atTop (nhds 0)) ∧
  (∀ F : Θ → ℝ, (∃ β β', F β ≠ F β') →
    ¬ ∃ conf : ℕ → List (Fin n) → Set ℝ,
      (∀ β, Filter.Tendsto
        (fun t : ℕ => P β {ω |
          F β ∉ conf t (List.ofFn (fun k : Fin t => ω k.val))})
        Filter.atTop (nhds 0)) ∧
      (∀ (β : Θ) (ε : ℝ), 0 < ε →
        Filter.Tendsto
          (fun t : ℕ => P β {ω |
            ∃ a ∈ conf t (List.ofFn (fun k : Fin t => ω k.val)),
            ∃ b ∈ conf t (List.ofFn (fun k : Fin t => ω k.val)),
              ε ≤ abs (a - b)})
          Filter.atTop (nhds 0)))