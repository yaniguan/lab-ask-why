/-!
# LabAskWhyLean: Formal L6 Epistemic Symmetry, Multi-Claim Consistency & Characterization Guard

Self-contained Lean 4 (`v4.36.0`) formalization for `lab-ask-why` (`L6`), synchronized
with the central Auto-Lab Trust Regime (`AutoLabLean`):

1. **Symmetric Epistemic Retrieval & Socratic Abstention (`EpistemicProbe`, `checkAskWhy`)**:
   - If maximum retrieval score `< noinfoThreshPermille`, the co-scientist MUST abstain
     (`noinfo_abstain`) rather than hallucinating unsupported mechanisms.
   - At least one rationale snippet must be retrieved (`0 < supportHits + contradictHits`).
   - Whenever both supporting and contradicting evidence exist (`0 < supportHits` and
     `0 < contradictHits`), proposing an action requires an explicit falsifiable
     discriminating experiment (`contested_claim_requires_experiment`).
2. **Analytical Precursor-Overlap Characterization Guard (`AssayTelemetry`, `attributeAssayOutcome`)**:
   - Separates hardware/control faults (`.executionArtifact`) and analytical
     non-identifiability such as precursor ghost-peak overlap (`.ambiguousCharacterization`)
     from proof-carrying scientific truth (`.certifiedPositive` / `.certifiedNegative`).
   - Formally guarantees `unidentifiable_spectrum_never_certified` and conservative
     ceiling permille quantization soundness (`ceil_permille_preserves_violation`).
3. **Multi-Claim Entailment & Mutual-Exclusion Consistency (`EpistemicGraph`, `checkEpistemicGraph`)**:
   - Enforces cross-claim mechanistic prerequisite consistency (`entailment_conflict_requires_experiment`)
     and mutual-exclusion consistency (`mutual_exclusion_conflict_requires_experiment`) across
     campaign probe graphs even when individual probes appear locally uncontested.
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

/-- Conservative floor quantization (`num / denom`) never falsely clears `noinfoThreshPermille`
    when the underlying rational ratio `num / denom` is strictly below `noinfoThreshPermille`. -/
theorem floor_permille_preserves_abstention (num denom thresh : Nat)
    (_hdenom : 0 < denom)
    (hbelow : num < thresh * denom) :
    num / denom < thresh := by
  exact Nat.div_lt_of_lt_mul (Nat.mul_comm thresh denom ▸ hbelow)

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

/-- Conservative ceiling permille conversion `(num + denom - 1) / denom` strictly preserves
    any continuous ceiling violation `maxPermille * denom < num` (e.g., 100.0001 or 100.5 permille),
    eliminating round-half-to-even boundary soundness holes. -/
theorem ceil_permille_preserves_violation (num denom maxPermille : Nat)
    (hdenom : 0 < denom)
    (hviol : maxPermille * denom < num) :
    maxPermille < (num + denom - 1) / denom := by
  have hle : (maxPermille + 1) * denom ≤ num + denom - 1 := by
    have hmul : (maxPermille + 1) * denom = maxPermille * denom + denom := Nat.succ_mul maxPermille denom
    omega
  have hdiv : maxPermille + 1 ≤ (num + denom - 1) / denom :=
    (Nat.le_div_iff_mul_le hdenom).mpr hle
  omega

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

/-! ## 3. Multi-Claim Entailment & Mutual-Exclusion Consistency (`EpistemicGraph`) -/

/-- Directed mechanistic prerequisite edge: `antecedent` entails `consequent`. -/
structure ImplicationEdge where
  antecedent : Nat
  consequent : Nat
  deriving DecidableEq, Repr

/-- Symmetric mutual-exclusion edge: `claimA` and `claimB` cannot jointly hold. -/
structure MutualExclusionEdge where
  claimA : Nat
  claimB : Nat
  deriving DecidableEq, Repr

/-- Multi-claim epistemic campaign graph over local probes and cross-claim constraints. -/
structure EpistemicGraph where
  probes       : List EpistemicProbe
  implications : List ImplicationEdge
  exclusions   : List MutualExclusionEdge := []
  deriving DecidableEq, Repr

/-- Total zero-indexed lookup avoiding standard-library version-specific indexing syntax. -/
def probeAt? : List EpistemicProbe → Nat → Option EpistemicProbe
  | [], _ => none
  | p :: _, 0 => some p
  | _ :: ps, n + 1 => probeAt? ps n

/-- Sequential validation of any list of items via a violation checker. -/
def firstMapViolation {α : Type} (f : α → Option RegimeViolation) : List α → Option RegimeViolation
  | [] => none
  | x :: xs =>
    match f x with
    | some v => some v
    | none => firstMapViolation f xs

theorem firstMapViolation_none_iff {α : Type} (f : α → Option RegimeViolation) (xs : List α) :
    firstMapViolation f xs = none ↔ ∀ x ∈ xs, f x = none := by
  induction xs with
  | nil => simp [firstMapViolation]
  | cons hd tl ih =>
    unfold firstMapViolation
    cases h : f hd with
    | some v =>
      constructor
      · intro hnone
        contradiction
      · intro hall
        have hhd := hall hd List.mem_cons_self
        rw [h] at hhd
        contradiction
    | none =>
      constructor
      · intro htl x hx
        rcases List.mem_cons.mp hx with rfl | hmem
        · exact h
        · exact (ih.mp htl) x hmem
      · intro hall
        exact ih.mpr (fun x hx => hall x (List.mem_cons_of_mem hd hx))

/-- Check a single implication edge `antecedent -> consequent` against `probes`. -/
def checkImplicationEdge (probes : List EpistemicProbe) (e : ImplicationEdge) : Option RegimeViolation :=
  match probeAt? probes e.antecedent, probeAt? probes e.consequent with
  | some pA, some pB =>
    if decide (¬(0 < pA.supportHits ∧ 0 < pB.contradictHits) ∨ pA.hasDiscriminatingExperiment = true) then
      none
    else
      some { repo := "lab-ask-why",
             rule := "entailment_conflict_requires_experiment",
             detail := "supported claim entails a contradicted prerequisite without a discriminating experiment" }
  | _, _ =>
    some { repo := "lab-ask-why",
           rule := "implication_index_bounds",
           detail := "implication edge index out of bounds for probe graph" }

/-- Check a single mutual-exclusion edge `(claimA, claimB)` against `probes`. -/
def checkExclusionEdge (probes : List EpistemicProbe) (mx : MutualExclusionEdge) : Option RegimeViolation :=
  match probeAt? probes mx.claimA, probeAt? probes mx.claimB with
  | some pA, some pB =>
    if decide (¬(0 < pA.supportHits ∧ 0 < pB.supportHits) ∨
               (pA.hasDiscriminatingExperiment = true ∧ pB.hasDiscriminatingExperiment = true)) then
      none
    else
      some { repo := "lab-ask-why",
             rule := "mutual_exclusion_conflict_requires_experiment",
             detail := "mutually exclusive claims both carry supporting evidence without discriminating experiments" }
  | _, _ =>
    some { repo := "lab-ask-why",
           rule := "exclusion_index_bounds",
           detail := "mutual-exclusion edge index out of bounds for probe graph" }

/-- Decidable multi-claim guardrail checker over an entire `EpistemicGraph`. -/
def checkEpistemicGraph (g : EpistemicGraph) : Option RegimeViolation :=
  match firstMapViolation checkAskWhy g.probes with
  | some v => some v
  | none =>
    match firstMapViolation (checkImplicationEdge g.probes) g.implications with
    | some v => some v
    | none => firstMapViolation (checkExclusionEdge g.probes) g.exclusions

/-- Decomposition lemma for `checkEpistemicGraph = none`. -/
theorem checkEpistemicGraph_none_iff (g : EpistemicGraph) :
    checkEpistemicGraph g = none ↔
      (∀ p ∈ g.probes, checkAskWhy p = none) ∧
      (∀ e ∈ g.implications, checkImplicationEdge g.probes e = none) ∧
      (∀ mx ∈ g.exclusions, checkExclusionEdge g.probes mx = none) := by
  unfold checkEpistemicGraph
  cases hp : firstMapViolation checkAskWhy g.probes with
  | some v =>
    constructor
    · intro hnone
      contradiction
    · rintro ⟨hp_all, _, _⟩
      have hnone := (firstMapViolation_none_iff checkAskWhy g.probes).mpr hp_all
      rw [hp] at hnone
      contradiction
  | none =>
    have hp_all := (firstMapViolation_none_iff checkAskWhy g.probes).mp hp
    cases hi : firstMapViolation (checkImplicationEdge g.probes) g.implications with
    | some v =>
      constructor
      · intro hnone
        contradiction
      · rintro ⟨_, hi_all, _⟩
        have hnone := (firstMapViolation_none_iff (checkImplicationEdge g.probes) g.implications).mpr hi_all
        rw [hi] at hnone
        contradiction
    | none =>
      have hi_all := (firstMapViolation_none_iff (checkImplicationEdge g.probes) g.implications).mp hi
      constructor
      · intro hx
        exact ⟨hp_all, hi_all, (firstMapViolation_none_iff (checkExclusionEdge g.probes) g.exclusions).mp hx⟩
      · rintro ⟨_, _, hx_all⟩
        exact (firstMapViolation_none_iff (checkExclusionEdge g.probes) g.exclusions).mpr hx_all

/-- Multi-claim entailment theorem: if claim `A` entails prerequisite `B`, `A` has supporting
    evidence, and `B` has contradicting evidence, a valid campaign MUST attach a discriminating
    experiment to `A` even if neither `A` nor `B` is locally contested on its own. -/
theorem entailment_conflict_requires_experiment (g : EpistemicGraph) (e : ImplicationEdge)
    (pA pB : EpistemicProbe)
    (hok : checkEpistemicGraph g = none)
    (he : e ∈ g.implications)
    (hA : probeAt? g.probes e.antecedent = some pA)
    (hB : probeAt? g.probes e.consequent = some pB)
    (hsup : 0 < pA.supportHits)
    (hcon : 0 < pB.contradictHits) :
    pA.hasDiscriminatingExperiment = true := by
  have ⟨_, himp, _⟩ := (checkEpistemicGraph_none_iff g).mp hok
  have hedge := himp e he
  unfold checkImplicationEdge at hedge
  rw [hA, hB] at hedge
  simp at hedge
  exact hedge (by omega) (by omega)

/-- Contrapositive rejection corollary for untested entailment conflicts. -/
theorem entailment_conflict_without_experiment_rejected (g : EpistemicGraph) (e : ImplicationEdge)
    (pA pB : EpistemicProbe)
    (he : e ∈ g.implications)
    (hA : probeAt? g.probes e.antecedent = some pA)
    (hB : probeAt? g.probes e.consequent = some pB)
    (hsup : 0 < pA.supportHits)
    (hcon : 0 < pB.contradictHits)
    (hnoexp : pA.hasDiscriminatingExperiment = false) :
    checkEpistemicGraph g ≠ none := by
  intro hok
  have hexp := entailment_conflict_requires_experiment g e pA pB hok he hA hB hsup hcon
  rw [hnoexp] at hexp
  contradiction

/-- Multi-claim mutual-exclusion theorem: if mutually exclusive claims `A` and `B` both have
    supporting evidence in a valid campaign, both must carry discriminating experiments. -/
theorem mutual_exclusion_conflict_requires_experiment (g : EpistemicGraph) (mx : MutualExclusionEdge)
    (pA pB : EpistemicProbe)
    (hok : checkEpistemicGraph g = none)
    (hmx : mx ∈ g.exclusions)
    (hA : probeAt? g.probes mx.claimA = some pA)
    (hB : probeAt? g.probes mx.claimB = some pB)
    (hsupA : 0 < pA.supportHits)
    (hsupB : 0 < pB.supportHits) :
    pA.hasDiscriminatingExperiment = true ∧ pB.hasDiscriminatingExperiment = true := by
  have ⟨_, _, hexc⟩ := (checkEpistemicGraph_none_iff g).mp hok
  have hedge := hexc mx hmx
  unfold checkExclusionEdge at hedge
  rw [hA, hB] at hedge
  simp at hedge
  exact hedge (by omega) (by omega)

/-! ## 4. Split-Conformal 3-Way Stance Set Soundness (`SUPPORT` / `CONTRADICT` / `NEI`) -/

/-- Three-way epistemic stance label space. -/
inductive StanceLabel where
  | support
  | contradict
  | nei
  deriving DecidableEq, Repr

/-- Decidable 3-way Split-Conformal prediction set in permille coordinates (`[0, 1000]`). -/
structure ConformalStanceSet where
  supportIn    : Bool
  contradictIn : Bool
  neiIn        : Bool
  deriving DecidableEq, Repr

def ConformalStanceSet.contains (s : ConformalStanceSet) : StanceLabel → Bool
  | .support    => s.supportIn
  | .contradict => s.contradictIn
  | .nei        => s.neiIn

def ConformalStanceSet.isContestedConflict (s : ConformalStanceSet) : Bool :=
  s.supportIn && s.contradictIn

/-- Construct the Split-Conformal set `{y : 1000 - p_y <= qPermille}`, with argmax fallback
    so extreme low-quantile boundary ties never emit an empty epistemic set. -/
def conformalStanceSetPermille (pSupport pContradict pNei qPermille : Nat) : ConformalStanceSet :=
  let sIn := decide (1000 ≤ pSupport + qPermille)
  let cIn := decide (1000 ≤ pContradict + qPermille)
  let nIn := decide (1000 ≤ pNei + qPermille)
  if sIn || cIn || nIn then
    { supportIn := sIn, contradictIn := cIn, neiIn := nIn }
  else if decide (pContradict ≤ pSupport ∧ pNei ≤ pSupport) then
    { supportIn := true, contradictIn := false, neiIn := false }
  else if decide (pNei ≤ pContradict) then
    { supportIn := false, contradictIn := true, neiIn := false }
  else
    { supportIn := false, contradictIn := false, neiIn := true }

/-- Non-emptiness invariant: `conformalStanceSetPermille` always retains at least one stance. -/
theorem conformalStanceSet_nonempty (pS pC pN q : Nat) :
    let cs := conformalStanceSetPermille pS pC pN q
    cs.supportIn = true ∨ cs.contradictIn = true ∨ cs.neiIn = true := by
  unfold conformalStanceSetPermille
  by_cases hany : (decide (1000 ≤ pS + q) || decide (1000 ≤ pC + q) || decide (1000 ≤ pN + q)) = true
  · simp only [hany, ite_true]
    rcases Bool.or_eq_true_iff.mp hany with hsc | hn
    · rcases Bool.or_eq_true_iff.mp hsc with hs | hc
      · exact Or.inl hs
      · exact Or.inr (Or.inl hc)
    · exact Or.inr (Or.inr hn)
  · simp only [hany]
    by_cases h1 : decide (pC ≤ pS ∧ pN ≤ pS) = true
    · simp [h1]
    · by_cases h2 : decide (pN ≤ pC) = true
      · simp [h1, h2]
      · simp [h1, h2]

/-- Coverage soundness: any stance whose nonconformity `1000 - p_y` is bounded by `qPermille`
    is guaranteed to be included in `conformalStanceSetPermille`. -/
theorem conformal_quantile_covers_support (pS pC pN q : Nat)
    (hcov : 1000 ≤ pS + q) :
    (conformalStanceSetPermille pS pC pN q).contains .support = true := by
  unfold conformalStanceSetPermille ConformalStanceSet.contains
  have hs : decide (1000 ≤ pS + q) = true := decide_eq_true hcov
  simp [hs]

/-- Anti-collapse conflict theorem: whenever both `SUPPORT` and `CONTRADICT` clear the
    Split-Conformal threshold `1000 <= p + qPermille`, the emitted set retains BOTH labels
    (`isContestedConflict = true`) rather than collapsing onto a forced binary verdict. -/
theorem dual_conformal_evidence_emits_conflict_set (pS pC pN q : Nat)
    (hsup : 1000 ≤ pS + q) (hcon : 1000 ≤ pC + q) :
    (conformalStanceSetPermille pS pC pN q).isContestedConflict = true := by
  unfold conformalStanceSetPermille ConformalStanceSet.isContestedConflict
  have hs : decide (1000 ≤ pS + q) = true := decide_eq_true hsup
  have hc : decide (1000 ≤ pC + q) = true := decide_eq_true hcon
  simp [hs, hc]

/-! ## 5. Compile-Time `#guard` Witnesses -/

#guard (conformalStanceSetPermille 480 470 50 700).isContestedConflict == true
#guard (conformalStanceSetPermille 860 80 60 680) ==
       { supportIn := true, contradictIn := false, neiIn := false }


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

-- Multi-claim prerequisite conflict witness: each local probe passes `checkAskWhy` individually,
-- yet `checkEpistemicGraph` catches the contradicted prerequisite unless an experiment is attached.
def aspirinParentProbe : EpistemicProbe :=
  { topScorePermille := 820, noinfoThreshPermille := 350,
    supportHits := 2, contradictHits := 0,
    hasDiscriminatingExperiment := false }

def aspirinPrerequisiteProbe : EpistemicProbe :=
  { topScorePermille := 790, noinfoThreshPermille := 350,
    supportHits := 0, contradictHits := 2,
    hasDiscriminatingExperiment := false }

#guard (checkAskWhy aspirinParentProbe).isNone == true
#guard (checkAskWhy aspirinPrerequisiteProbe).isNone == true

#guard checkEpistemicGraph
         { probes := [aspirinParentProbe, aspirinPrerequisiteProbe],
           implications := [{ antecedent := 0, consequent := 1 }] }
       == some { repo := "lab-ask-why",
                 rule := "entailment_conflict_requires_experiment",
                 detail := "supported claim entails a contradicted prerequisite without a discriminating experiment" }

#guard (checkEpistemicGraph
         { probes := [{ aspirinParentProbe with hasDiscriminatingExperiment := true },
                      aspirinPrerequisiteProbe],
           implications := [{ antecedent := 0, consequent := 1 }] }).isNone == true

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
