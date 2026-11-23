# `LabAskWhyLean` -- Formal L6 Epistemic & Characterization Guardrails in Lean 4

Self-contained Lean 4 (`leanprover/lean4:v4.36.0`, zero external dependencies, `0 sorry`, `0 axiom`) formalization of the `L6` epistemic invariants for `lab-ask-why`, synchronized with the central 14-repository formal specification in [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean).

## Proved Theorems & Guardrails

| Component | Definition / Theorem | Guarantee |
| :--- | :--- | :--- |
| Epistemic Probe Gate | `checkAskWhy_ok_iff` | Exact soundness and completeness of `checkAskWhy` wrt `EpistemicProbeOK` |
| Anti-Confirmation | `contested_claim_requires_experiment` | Opposing support and contradict evidence requires an explicit discriminating experiment |
| `NOINFO` Abstention | `below_noinfo_always_abstains` | Sub-threshold retrieval scores never pass `checkAskWhy` |
| Non-Empty Rationale | `empty_rationale_always_rejected` | Probes with zero retrieved rationale snippets are rejected |
| Execution Isolation | `corrupted_run_never_poisons_memory` | Hardware, acoustic, schedule, or control faults map to `.executionArtifact` |
| Precursor-Overlap Guard | `unidentifiable_spectrum_never_certified` | High spectral residual or precursor ghost-peak overlap blocks `.certifiedPositive` and `.certifiedNegative` |
| Proof-Carrying Outcomes | `certified_negative_sound`, `certified_positive_sound` | Certified outcomes satisfy all execution and spectral identifiability bounds |

## Quickstart

```bash
cd lean
lake build
```
