# Architecture and History -- `lab-ask-why`

> **Auto-Lab Ecosystem Role**: Layer `L6 · Metacognitive & Epistemic Brain` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Standard scientific question-answering and automated co-scientist systems optimize to declare a single winning hypothesis, hiding competing evidence, unstated mechanistic assumptions, and substantial inter-model disagreement (even models with comparable benchmark accuracy disagree on 16-66% of scientific items). When human researchers or multi-agent scientist ensembles disagree on a claim, premature verdict collapse causes wasted wet-lab runs and confirmation bias.

`lab-ask-why` implements an **epistemic Socratic co-scientist** that refuses forced binary verdict collapse. Instead, it asks *"why do you think so?"*, retrieves verbatim rationale sentences supporting or contradicting each side across primary literature, quantifies epistemic ambiguity via Split-Conformal 3-way stance prediction sets (`{"SUPPORT", "CONTRADICT", "NEI"}`), surfaces latent assumptions, and synthesizes structured falsifiable control-well specifications (`ControlWellSpec`) for the discriminating experiment that would change minds.

### Role in the Auto-Lab `L0`-`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`-`L6`), `lab-ask-why` sits at **`L6 · Metacognitive & Epistemic Brain`** alongside [`lab-decide`](https://github.com/yaniguan/lab-decide):
- **Peer (`L6` Metacognitive Brain)**: Pairs directly with [`lab-decide`](https://github.com/yaniguan/lab-decide). While `lab-ask-why` decomposes competing scientific hypotheses into supporting/contradicting rationales, Split-Conformal stance sets, and proposed discriminating experiments, `lab-decide` quantifies whether candidate operating points lie inside a calibrated conformal trust region or require active exploration.
- **Upstream (`L0`-`L3` Sensory, Literature & Post-Trained Reasoners)**: Consumes speech-stripped protocol spans from [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`) and connects to [`YLM`](https://github.com/yaniguan/YLM) (`L0`), [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3`), [`WignerFlow`](https://github.com/yaniguan/WignerFlow), and [`assayfold`](https://github.com/yaniguan/assayfold).
- **Downstream (`L5` / `L4` Planning, Verification & Attribution)**: Hands off `EpistemicProbeResult` and `ControlWellSpec` packets to [`labshift`](https://github.com/yaniguan/labshift), [`buy-or-make`](https://github.com/yaniguan/buy-or-make), and [`lab-automation-atlas`](https://github.com/yaniguan/lab-automation-atlas) (`L5`), audited by [`lab-verify`](https://github.com/yaniguan/lab-verify) and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`), with post-execution SHA-256 causal attribution certificates closed via [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`).

---

## 2. Data

### Active Training & Held-Out Benchmark Corpus: SciFact (Wadden et al., EMNLP 2020)
Downloaded on demand into gitignored `data/` (`CC-BY-NC-2.0`, SHA-256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`, zero redistribution):

| File | Records | Active Role in `lab-ask-why` | Schema Highlights |
|---|---:|---|---|
| `corpus.jsonl` | `5,183` abstracts | Indexed for BM25, subword/word TF-IDF Truncated-SVD (`LSA`), and entity/numeric overlap (`93` dev-exclusive docs excluded from IDF/SVD fitting) | `doc_id: int`, `title: str`, `abstract: list[str]` |
| `claims_train.jsonl` | `809` claims | Strictly partitioned into `607` `train_fit` claims (fitting `CrossFeatureReranker` + 3-way stance weights) and `202` `train_calib` claims (`1 - alpha = 0.90` Split Conformal calibration) | `id: int`, `claim: str`, `evidence`, `cited_doc_ids` |
| `claims_dev.jsonl` | `300` claims | Held-out evaluation (`188` evidence claims yielding `209` pairs [`138` `SUPPORT`, `71` `CONTRADICT`] + `112` `NOINFO` claims; `0` claim-ID overlap with `train`) | `id: int`, `claim: str`, `evidence`, `cited_doc_ids` |
| `claims_test.jsonl` | `300` claims | Blind test split without gold labels (unused) | `id: int`, `claim: str`, `cited_doc_ids` |

### Open-Source Dataset Registry (`src/lab_ask_why/datasets.py`, 2019-2026)
`OPEN_DATASET_REGISTRY` registers `14` scientific claim verification, conflicting-evidence, and literature/protocol QA datasets (`labbench2`, `peerqa_xt`, `sciver`, `claimcheck`, `chembench`, `litqa2`, `lab_bench`, `conflictbank`, `faitheval_inconsistent`, `sciriff`, `bioasq`, `scifact_open`, `scifact`, `pubmedqa`), enforcing `PermissionError` whenever `purpose != "eval"` on `eval_only=True` benchmarks (`LAB-Bench`, `LitQA2`, `LABBench2`, `ChemBench`) or canary-bearing records.

---

## 3. Model

1. **Zero-Parameter Okapi BM25 Lexical Baseline (`src/lab_ask_why/bm25.py`)**:
   - Biomedical tokenizer (`normalize_biomedical_text`, `tokenize`) mapping Greek characters (`alpha`, `beta`, `delta`, `micro`), preserving `+/-` allele markers (`alpha+`, `cd4-`, `trp53+/-`) and decimal/comma numbers, expanding negative contractions (`cannot`, `does not`), and retaining negation cues (`not`, `no`, `neither`, `without`, `never`).
   - Query-TF-saturated Okapi BM25 ($k_1 = 1.2$, $b = 0.75$, $k_3 = 1.2$) with background-only IDF (`exclude_doc_ids`) and query-budget score normalization in `[0, 1]`.
2. **Frontier 2-Stage Hybrid Retrieval, Reranker & Split-Conformal Stance Verifier (`src/lab_ask_why/frontier_verifier.py`)**:
   - **Stage 1 (`Reciprocal Rank Fusion`)**: Fuses (a) Okapi `BM25`, (b) subword character (`3`/`4`-gram) + biomedical stem/concept TF-IDF `TruncatedSVDSemanticIndex` (`96`-dim `LSA` subspace via sign-canonicalized Halko-Martinsson-Tropp randomized SVD fit strictly on background documents), and (c) `EntityNumericNegationIndex` (acronyms, gene/allele symbols, numeric literals, negations).
   - **Stage 2 (`CrossFeatureReranker`)**: 26-feature damped Newton-IRLS L2 logistic reranker scoring candidate sentences on within-pool BM25 & LSA z-scores, lexical-dense agreement, IDF-weighted rare-entity coverage, scoped negation & quantitative directionality (`increase`/`decrease`/`inhibit`/`activate`), abstract title anaphora rescue, and discourse section position.
   - **Stage 2 (`SplitConformalStanceClassifier`)**: 28-feature Nesterov-accelerated L2 multinomial 3-way classifier over `("SUPPORT", "CONTRADICT", "NEI")` with Mondrian Split Conformal quantiles $\hat{q}_{1-\alpha}^{(y)} = s_{(\lceil (n_y + 1)(1 - \alpha) \rceil)}$ calibrated at $1 - \alpha = 0.90$ on held-out `SciFact train_calib`, preserving opposing evidence mass so conflicting multi-document claims emit `{"SUPPORT", "CONTRADICT"}`.

---

## 4. Training & Leak-Free Multi-Dataset Benchmarks (`results/open_benchmarks_2026.json`)

All supervised parameters (`CrossFeatureReranker` and `SplitConformalStanceClassifier`) and conformal quantiles are fit **strictly once on `SciFact train` (`809` claims)** on CPU (`numpy` only, zero LLM API calls) and evaluated zero-leakage across `SciFact dev` and `7` additional downloaded public corpora (`scripts/benchmark_open_datasets.py`, verified against drift by `tests/test_readme_drift.py`):

<!-- SCIFACT_FRONTIER_TABLE_START -->
| Method (`SciFact dev`, `n=300` claims / `209` pairs) | Hit@1 | Hit@3 | MRR | Set-Recall@3 | Open `doc_recall@5` | 3-Way Stance Macro-F1 | 90% Conformal Coverage (Avg Size) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Random` (20 seeds) | `0.197` | `0.528` | `0.419` | -- | -- | -- | -- |
| `Okapi BM25` (zero-param lexical) | `0.483` | `0.818` | `0.660` | `0.775` | `0.885` | `0.353` | -- |
| `Dense-SVD` (`LSA` char/word TF-IDF) | `0.411` | `0.761` | `0.615` | `0.732` | `0.490` | `0.281` | -- |
| `Stage-1 Hybrid RRF` (`BM25+SVD+Entity`) | `0.502` | `0.804` | `0.673` | `0.770` | `0.883` | -- | -- |
| **`Hybrid + Cross-Feature Reranker`** (`SciFact train`) | **`0.660`** | **`0.909`** | **`0.794`** | **`0.876`** | **`0.883`** | **`0.512`** | **`0.907`** (`2.36`) |
<!-- SCIFACT_FRONTIER_TABLE_END -->

Zero-shot cross-benchmark evaluation (`data_source = "real_downloaded"`, SHA-256 verified in gitignored `data/open_benchmarks/`):

<!-- OPEN_BENCHMARKS_2026_TABLE_START -->
| Dataset (`data_source`) | Evaluated ($n$) | License (`eval_only`) | Baseline (`BM25`) | Frontier (`Hybrid+Reranker`) | Conformal / Conflict Headline |
|---|---:|---|---|---|---|
| `SciFact` (`real_downloaded`) | `300` claims (`209` pairs) | `CC-BY-NC-2.0` (`False`) | Hit@1 `0.483`, Hit@3 `0.818` | **Hit@1 `0.660`, Hit@3 `0.909`, MRR `0.794`** | 90% set coverage **`0.907`** (size `2.36`) |
| `PubMedQA` (`real_downloaded`) | `1000` (`pqa_labeled`) | `MIT` (`False`) | Yes/No Macro-F1 `0.483` | **Yes/No Macro-F1 `0.558`** | 3-way 90% coverage **`0.867`** (`LONG_ANSWER` excluded) |
| `ConflictBank` (`real_downloaded`) | `200` QA + `200` pairs | `CC-BY-SA-4.0` (`False`) | Conflict detect `0.075` (binary `0.000`) | **Conflict detect `1.000`**, stance F1 **`0.257`** | Multi-label `{SUPPORT, CONTRADICT}` conflict sets |
| `FaithEval-Inconsistent` (`real_downloaded`) | `300` multi-doc | `CC-BY-NC-SA-4.0` (`False`) | Polarity conflict `0.177` (forced `0.000`) | **Multi-answer conflict detect `0.967`** | Opposing-document grounding + Socratic crux |
| `CLAIMCHECK` (`real_downloaded`) | `108` critiques | `NOASSERTION` (`False`) | Critique Hit@1 `0.306`, MRR `0.489` | **Critique Hit@1 `0.407`, Hit@3 `0.787`, MRR `0.611`** | 90% coverage **`0.556`** |
| `SciVer` (`real_downloaded`) | `1000` claims | `CC-BY-4.0` (`False`) | Stance Macro-F1 `0.475` | **Stance Macro-F1 `0.429`** (acc `0.492`) | 90% coverage **`0.588`** (leak-free evidence) |
| `ChemBench` (`real_downloaded`) | `167` MCQ (`249` raw) | `MIT` (**`True`**) | Option Hit@1 `0.263`, MRR `0.527` | **Option Hit@1 `0.286`, MRR `0.543`** | Canary-guarded `eval_only=True` verified |
| `LitQA2` (`real_downloaded`) | `194` MCQ (`199` raw) | `CC-BY-SA-4.0` (**`True`**) | Option Hit@1 `0.504`, MRR `0.693` | **Option Hit@1 `0.553`, MRR `0.728`** | Canary-guarded `eval_only=True` verified |
<!-- OPEN_BENCHMARKS_2026_TABLE_END -->

For reference, the standalone zero-parameter lexical demo (`python demo/why_probe.py`, verified against `demo/output.txt` in CI) achieves `0.483` `Hit@1`, `0.813` `Hit@3` (`0.826` `SUPPORT`, `0.789` `CONTRADICT`), `0.658` `Sentence-Recall@3`, `0.775` `Rationale-Set-Recall@3`, `0.765` `NOINFO` AUROC, and `0.880` open-corpus `doc_recall@5` (`0.596` `sentence_hit@3`).

---

## 5. History

| Milestone | Ref / PR | Summary |
|---|---|---|
| Initial Baseline | `6236123` (`2026-11-18`) | Added `demo/why_probe.py`, `demo/output.txt`, `.github/workflows/demo.yml`, and curated datasets/repos/papers. |
| Audit Hardening & Typed Schemas | PR `#23` (`e8a033b`) | Resolved Issues `#1..#22`: safe tar extraction, SHA-256 downloads, three-tier rationale metrics, background IDF, `TwoStageRetriever`, `EpistemicProbeResult`, and `LAB-Bench` canary adapter. |
| Phase-2 Audit, Lean 4 & Wire Loop | PR `#30` (`40af81d`) | Resolved Issues `#24..#29`: floor/ceil permille soundness, multi-claim `EpistemicGraph` proofs in `LabAskWhyLean.lean`, `ControlWellSpec` synthesis, and `autolab_wire.py` closed-loop `lab-attrib` certificates. |
| 2019-2026 Open Dataset Registry | PR `#31` (`b8a9d51`) | Added `14`-dataset `OPEN_DATASET_REGISTRY` (`datasets.py`) and `scripts/fetch_open_datasets.py` with SPDX/`eval_only` guards. |
| Frontier Hybrid Verifier & Lean CI | `feat/frontier-hybrid-reranker-conformal-and-benchmarks` | Added `FrontierHybridVerifier` (`BM25 + LSA + Entity RRF` + `CrossFeatureReranker` + `90%` Split-Conformal stance sets), 8-corpus SHA-256 benchmark suite (`results/open_benchmarks_2026.json`), fail-closed `autolab_wire.py` input/certificate checks, Lean 4.36.0 `#print axioms` + `160`-case `#eval` parity CI, and `tests/test_readme_drift.py`. |

---

## 6. Formal Lean 4.36.0 Verification (`lean/LabAskWhyLean.lean`)

`lean/LabAskWhyLean.lean` (`21` theorems, `0 sorry`, standard Lean axioms `propext`, `Classical.choice`, `Quot.sound` only -- audited in CI by `scripts/verify_lean_parity.py` across `160` `#eval` Python parity test cases) formally verifies:
1. **`checkAskWhy` & Abstention/Crux Invariants**: `checkAskWhy_ok_iff`, `contested_claim_requires_experiment`, `below_noinfo_always_abstains`, `empty_rationale_always_rejected`, and `floor_permille_preserves_abstention`.
2. **Precursor Ghost-Peak & Characterization Safeguard**: `corrupted_run_never_poisons_memory`, `unidentifiable_spectrum_never_certified`, `ceil_permille_preserves_violation`, `certified_negative_sound`, and `certified_positive_sound`.
3. **Multi-Claim Campaign Consistency (`EpistemicGraph`)**: `checkEpistemicGraph_none_iff`, `entailment_conflict_requires_experiment`, `entailment_conflict_without_experiment_rejected`, and `mutual_exclusion_conflict_requires_experiment`.
4. **Split-Conformal 3-Way Stance Sets (`ConformalStanceSet`)**: `conformalStanceSet_nonempty`, `conformal_quantile_covers_support`, and `dual_conformal_evidence_emits_conflict_set`.

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ask-why"`, `LAYER = "L6"`) implements `process_packet(packet: dict) -> dict` without network I/O:
1. **Strict Fail-Closed Validation**: Rejects non-mapping envelopes (`L6_INVALID_PACKET_TYPE`), non-boolean flags (`L6_INVALID_BOOLEAN_FLAG`), non-string/empty claims (`L6_INVALID_CLAIM`), invalid evidence lists (`L6_INVALID_EVIDENCE_SENTENCES`), and malformed or unsound `CERTIFIED_NEGATIVE` certificates (`L6_MALFORMED_NEGATIVE_CERTIFICATE`) cleanly with `halted=True` and `status="BLOCK"`.
2. **Provenance & Split-Conformal Output**: Stamps `data_source` (`"synthetic_fixture"`, `"caller_provided"`, or `"real_downloaded"`), `conformal_stance_set`, `conformal_alpha`, and `stance_probabilities` into `epistemic_probe` alongside `support_top`, `contradict_top`, `why_questions`, `discriminating_observable`, and `control_wells`.
3. **Formal `contested_claim_requires_experiment` Gate**: Blocks (`L6_CONTESTED_WITHOUT_EXPERIMENT` or `L6_AMBIGUOUS_WITHOUT_ORTHOGONAL_ASSAY`) whenever `suppress_discriminating_experiment=True` coincides with opposing evidence or `AMBIGUOUS_CHARACTERIZATION`.
