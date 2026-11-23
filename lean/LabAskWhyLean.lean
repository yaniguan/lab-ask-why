/-!
# LabAskWhyLean: Formal L6 Epistemic Symmetry, NOINFO Abstention & Characterization Guard

Self-contained Lean 4 (`v4.36.0`) formalization for `lab-ask-why` (`L6`), synchronized
with the central 14-repository Auto-Lab Trust Regime (`AutoLabLean`):

1. **Symmetric Epistemic Retrieval & Socratic Abstention (`EpistemicProbe`, `checkAskWhy`)**:
   - If maximum retrieval score `< noinfoThreshPermille`, the co-scientist MUST abstain
     (`noinfo_abstain`) rather than hallucinating unsupported mechanisms.
   - At least one rationale snippet must be retrieved (`0 < supportHits + contradictHits`).
   - Whenever both supporting and contradicting evidence exist (`0 < supportHits` and
     `0 < contradictHits`), proposing an action requires an explicit falsifiable
     discriminating experiment (`contested_claim_requires_experiment`).
2. **Precursor-Overlap Characterization Guard (`AssayTelemetry`, `attributeAssayOutcome`)**:
   - Separates hardware/control faults (`.executionArtifact`) and analytical
     non-identifiability such as precursor ghost-peak overlap (`.ambiguousCharacterization`)
     from proof-carrying scientific truth (`.certifiedPositive` / `.certifiedNegative`).
   - Formally guarantees `unidentifiable_spectrum_never_certified`.
-/
namespace LabAskWhyLean

/-- Structured boundary violation reporting the repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

def RegimeViolation.format (v : RegimeViolation) : String :=
  s!"[{v.repo} :: {v.rule}] {v.detail}"

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Epistemic Symmetry & Contested-Claim Crux Requirement -/

/-- Fixed-point summary of an L6 retrieval and Socratic disagreement probe. -/
structure EpistemicProbe where
  topScorePermille            : Nat
  noinfoThreshPermille        : Nat
  supportHits                 : Nat
  contradictHits              : Nat
  hasDiscriminatingExperiment : Bool
  deriving DecidableEq, Repr

/-- Declarative specification of a valid L6 epistemic probe. -/
structure EpistemicProbeOK (p : EpistemicProbe) : Prop where
  enoughEvidence    : p.noinfoThreshPermille ≤ p.topScorePermille
  nonEmptyRationale : 0 < p.supportHits + p.contradictHits
  contestedResolved : (0 < p.supportHits ∧ 0 < p.contradictHits) →
                        p.hasDiscriminatingExperiment = true

/-- Decidable guardrail checker for `lab-ask-why` probes. -/
def checkAskWhy (p : EpistemicProbe) : Option RegimeViolation :=
  firstViolation
    [ (decide (p.noinfoThreshPermille ≤ p.topScorePermille),
        { repo := "lab-ask-why", rule := "noinfo_abstain",
          detail := "retrieval evidence below NOINFO threshold; must abstain" }),
      (decide (0 < p.supportHits + p.contradictHits),
        { repo := "lab-ask-why", rule := "nonempty_rationale",
          detail := "zero supporting or contradicting rationale sentences retrieved" }),
      (decide (¬(0 < p.supportHits ∧ 0 < p.contradictHits) ∨ p.hasDiscriminatingExperiment = true),
        { repo := "lab-ask-why", rule := "contested_requires_experiment",
          detail := "both support and contradict evidence present without a discriminating experiment" }) ]

/-- Soundness and completeness of `checkAskWhy` with respect to `EpistemicProbeOK`. -/
theorem checkAskWhy_ok_iff (p : EpistemicProbe) :
    checkAskWhy p = none ↔ EpistemicProbeOK p := by
  unfold checkAskWhy
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq]
  constructor
  · rintro ⟨h1, h2, h3⟩
    refine ⟨h1, h2, ?_⟩
    intro hcont
    rcases h3 with hneg | hexp
    · exact absurd hcont hneg
    · exact hexp
  · rintro ⟨h1, h2, h3⟩
    refine ⟨h1, h2, ?_⟩
    by_cases hc : 0 < p.supportHits ∧ 0 < p.contradictHits
    · exact Or.inr (h3 hc)
    · exact Or.inl hc

/-- Anti-confirmation theorem: a contested claim with opposing evidence cannot pass
    `checkAskWhy` unless it carries a discriminating experiment. -/
theorem contested_claim_requires_experiment (p : EpistemicProbe)
    (hok : checkAskWhy p = none)
    (hsup : 0 < p.supportHits) (hcon : 0 < p.contradictHits) :
    p.hasDiscriminatingExperiment = true :=
  ((checkAskWhy_ok_iff p).mp hok).contestedResolved ⟨hsup, hcon⟩

/-- Abstention boundary theorem: any probe below the NOINFO retrieval floor is rejected. -/
theorem below_noinfo_always_abstains (p : EpistemicProbe)
    (hlow : p.topScorePermille < p.noinfoThreshPermille) :
    checkAskWhy p ≠ none := by
  intro hok
  have hev := ((checkAskWhy_ok_iff p).mp hok).enoughEvidence
  omega

/-- Empty-rationale boundary theorem: zero retrieved rationale sentences cannot pass. -/
theorem empty_rationale_always_rejected (p : EpistemicProbe)
    (hsup : p.supportHits = 0) (hcon : p.contradictHits = 0) :
    checkAskWhy p ≠ none := by
  intro hok
  have hne := ((checkAskWhy_ok_iff p).mp hok).nonEmptyRationale
  omega

/-! ## 2. Analytical Precursor-Overlap Characterization Guard (`A-Lab` Safeguard) -/

/-- Post-execution telemetry for diagnostic epistemic attribution. -/
structure AssayTelemetry where
  preGateCertified          : Bool
  acousticNormal            : Bool
  scheduleWindowsRespected  : Bool
  tactileSeatedCleanly      : Bool
  posControlPassed          : Bool
  negControlBlankClean      : Bool
  spectralResidualPermille  : Nat
  maxResidualPermille       : Nat
  precursorOverlapPermille  : Nat
  maxOverlapPermille        : Nat
  signalAboveActivityThresh : Bool
  deriving DecidableEq, Repr

/-- Four-way causal attribution of an experimental wet-lab readout. -/
inductive AttributionOutcome where
  | executionArtifact
  | ambiguousCharacterization
  | certifiedPositive
  | certifiedNegative
  deriving DecidableEq, Repr

def executionClean (t : AssayTelemetry) : Bool :=
  t.preGateCertified &&
  t.acousticNormal &&
  t.scheduleWindowsRespected &&
  t.tactileSeatedCleanly &&
  t.posControlPassed &&
  t.negControlBlankClean

def characterizationIdentifiable (t : AssayTelemetry) : Bool :=
  decide (t.spectralResidualPermille ≤ t.maxResidualPermille ∧
          t.precursorOverlapPermille ≤ t.maxOverlapPermille)

/-- Exact 4-way causal attribution of an experimental wet-lab readout. -/
def attributeAssayOutcome (t : AssayTelemetry) : AttributionOutcome :=
  if !executionClean t then
    .executionArtifact
  else if !characterizationIdentifiable t then
    .ambiguousCharacterization
  else if t.signalAboveActivityThresh then
    .certifiedPositive
  else
    .certifiedNegative

/-- Whether an experimental readout is safe to admit into scientific memory. -/
def admitToScientificMemory (o : AttributionOutcome) : Bool :=
  match o with
  | .certifiedPositive => true
  | .certifiedNegative => true
  | _                  => false

/-- Execution/control failures are isolated as `.executionArtifact` and never poison memory. -/
theorem corrupted_run_never_poisons_memory (t : AssayTelemetry)
    (hfault : t.preGateCertified = false ∨ t.acousticNormal = false ∨
              t.scheduleWindowsRespected = false ∨ t.tactileSeatedCleanly = false ∨
              t.posControlPassed = false ∨ t.negControlBlankClean = false) :
    attributeAssayOutcome t = .executionArtifact ∧
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  have hex : executionClean t = false := by
    unfold executionClean
    rcases hfault with h | h | h | h | h | h <;> simp [h]
  unfold attributeAssayOutcome
  simp [hex, admitToScientificMemory]

/-- Formal `A-Lab` Characterization Safeguard: whenever precursor ghost-peak overlap or
    profile fit residual exceeds its identifiability ceiling, the readout is NEVER certified
    as positive or negative. -/
theorem unidentifiable_spectrum_never_certified (t : AssayTelemetry)
    (ham : t.maxResidualPermille < t.spectralResidualPermille ∨
           t.maxOverlapPermille < t.precursorOverlapPermille) :
    attributeAssayOutcome t ≠ .certifiedPositive ∧
    attributeAssayOutcome t ≠ .certifiedNegative := by
  have hid : characterizationIdentifiable t = false := by
    unfold characterizationIdentifiable
    apply decide_eq_false
    rcases ham with hr | ho <;> omega
  unfold attributeAssayOutcome
  cases executionClean t <;> simp [hid]

/-- Soundness of proof-carrying negative results. -/
theorem certified_negative_sound (t : AssayTelemetry)
    (hneg : attributeAssayOutcome t = .certifiedNegative) :
    executionClean t = true ∧
    t.spectralResidualPermille ≤ t.maxResidualPermille ∧
    t.precursorOverlapPermille ≤ t.maxOverlapPermille ∧
    t.signalAboveActivityThresh = false := by
  unfold attributeAssayOutcome at hneg
  cases hex : executionClean t with
  | false => simp [hex] at hneg
  | true =>
    simp only [hex] at hneg
    cases hid : characterizationIdentifiable t with
    | false => simp [hid] at hneg
    | true =>
      simp only [hid] at hneg
      cases hsig : t.signalAboveActivityThresh with
      | true => simp [hsig] at hneg
      | false =>
        unfold characterizationIdentifiable at hid
        have ⟨hr, ho⟩ := of_decide_eq_true hid
        exact ⟨rfl, hr, ho, rfl⟩

/-- Soundness of proof-carrying positive discoveries. -/
theorem certified_positive_sound (t : AssayTelemetry)
    (hpos : attributeAssayOutcome t = .certifiedPositive) :
    executionClean t = true ∧
    t.spectralResidualPermille ≤ t.maxResidualPermille ∧
    t.precursorOverlapPermille ≤ t.maxOverlapPermille ∧
    t.signalAboveActivityThresh = true := by
  unfold attributeAssayOutcome at hpos
  cases hex : executionClean t with
  | false => simp [hex] at hpos
  | true =>
    simp only [hex] at hpos
    cases hid : characterizationIdentifiable t with
    | false => simp [hid] at hpos
    | true =>
      simp only [hid] at hpos
      cases hsig : t.signalAboveActivityThresh with
      | false => simp [hsig] at hpos
      | true =>
        unfold characterizationIdentifiable at hid
        have ⟨hr, ho⟩ := of_decide_eq_true hid
        exact ⟨rfl, hr, ho, rfl⟩

/-! ## 3. Compile-Time `#guard` Witnesses -/

#guard (checkAskWhy
         { topScorePermille := 789, noinfoThreshPermille := 350,
           supportHits := 2, contradictHits := 1,
           hasDiscriminatingExperiment := true }).isNone == true

#guard checkAskWhy
         { topScorePermille := 789, noinfoThreshPermille := 350,
           supportHits := 2, contradictHits := 1,
           hasDiscriminatingExperiment := false }
       == some { repo := "lab-ask-why", rule := "contested_requires_experiment",
                 detail := "both support and contradict evidence present without a discriminating experiment" }

#guard (checkAskWhy
         { topScorePermille := 220, noinfoThreshPermille := 350,
           supportHits := 1, contradictHits := 0,
           hasDiscriminatingExperiment := false }).isSome == true

#guard attributeAssayOutcome
         { preGateCertified := true, acousticNormal := false, scheduleWindowsRespected := true,
           tactileSeatedCleanly := true, posControlPassed := true, negControlBlankClean := true,
           spectralResidualPermille := 40, maxResidualPermille := 100,
           precursorOverlapPermille := 30, maxOverlapPermille := 150,
           signalAboveActivityThresh := false }
       == .executionArtifact

#guard attributeAssayOutcome
         { preGateCertified := true, acousticNormal := true, scheduleWindowsRespected := true,
           tactileSeatedCleanly := true, posControlPassed := true, negControlBlankClean := true,
           spectralResidualPermille := 80, maxResidualPermille := 100,
           precursorOverlapPermille := 420, maxOverlapPermille := 150,
           signalAboveActivityThresh := true }
       == .ambiguousCharacterization

#guard attributeAssayOutcome
         { preGateCertified := true, acousticNormal := true, scheduleWindowsRespected := true,
           tactileSeatedCleanly := true, posControlPassed := true, negControlBlankClean := true,
           spectralResidualPermille := 45, maxResidualPermille := 100,
           precursorOverlapPermille := 35, maxOverlapPermille := 150,
           signalAboveActivityThresh := false }
       == .certifiedNegative

#guard attributeAssayOutcome
         { preGateCertified := true, acousticNormal := true, scheduleWindowsRespected := true,
           tactileSeatedCleanly := true, posControlPassed := true, negControlBlankClean := true,
           spectralResidualPermille := 50, maxResidualPermille := 100,
           precursorOverlapPermille := 40, maxOverlapPermille := 150,
           signalAboveActivityThresh := true }
       == .certifiedPositive

end LabAskWhyLean
