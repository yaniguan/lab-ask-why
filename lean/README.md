# `LabAskWhyLean` -- Formal L6 Epistemic, Split-Conformal & Characterization Guardrails in Lean 4

Self-contained Lean 4 (`leanprover/lean4:v4.36.0`, zero external dependencies, `0 sorry`, no custom axioms -- standard Lean axioms `propext`, `Classical.choice`, and `Quot.sound` only) formalization of the `L6` epistemic invariants for `lab-ask-why`, synchronized with the central 15-repository formal specification in [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean).

## Proved Theorems & Guardrails

| Component | Definition / Theorem | Guarantee |
| :--- | :--- | :--- |
| Epistemic Probe Gate | `checkAskWhy_ok_iff` | Exact soundness and completeness of `checkAskWhy` wrt `EpistemicProbeOK` |
| Anti-Confirmation | `contested_claim_requires_experiment` | Opposing support and contradict evidence requires an explicit discriminating experiment |
| `NOINFO` Abstention | `below_noinfo_always_abstains`, `floor_permille_preserves_abstention` | Sub-threshold retrieval scores never pass `checkAskWhy` under downward floor quantization |
| Non-Empty Rationale | `empty_rationale_always_rejected` | Probes with zero retrieved rationale snippets are rejected |
| Multi-Claim Entailment | `entailment_conflict_requires_experiment`, `entailment_conflict_without_experiment_rejected` | Supported claims entailing contradicted mechanistic prerequisites require a discriminating experiment |
| Multi-Claim Exclusion | `mutual_exclusion_conflict_requires_experiment` | Mutually exclusive claims with joint supporting hits require discriminating experiments |
| Split-Conformal Stance Sets | `conformalStanceSet_nonempty`, `conformal_quantile_covers_support`, `dual_conformal_evidence_emits_conflict_set` | Split-conformal permille prediction sets are non-empty, cover true stances within quantile threshold, and emit multi-label `{"SUPPORT", "CONTRADICT"}` sets on opposing evidence |
| Execution Isolation | `corrupted_run_never_poisons_memory` | Hardware, acoustic, schedule, or control faults map to `.executionArtifact` |
| Precursor-Overlap Guard | `unidentifiable_spectrum_never_certified`, `ceil_permille_preserves_violation` | High spectral residual or precursor ghost-peak overlap blocks certification under upward ceiling quantization |
| Proof-Carrying Outcomes | `certified_negative_sound`, `certified_positive_sound` | Certified outcomes satisfy all execution and spectral identifiability bounds |

## Quickstart & Differential Parity Verification

```bash
cd lean && lake build
python ../scripts/verify_lean_parity.py --clean
```
