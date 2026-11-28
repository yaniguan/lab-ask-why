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
| `corpus.jsonl` | `5,183` abstracts | Indexed for BM25, subword/word TF-IDF Truncated-SVD (`LSA`), dense BGE-small, and entity/numeric overlap (`181` dev-exclusive and calibration-exclusive docs excluded from Stage-1 IDF/SVD fitting) | `doc_id: int`, `title: str`, `abstract: list[str]` |
| `claims_train.jsonl` | `809` claims | Strictly partitioned into `607` `train_fit` claims (`75%`, fitting reranker & 3-way stance heads) and `202` `train_calib` claims (`25%`, `1 - alpha = 0.90` Mondrian Split Conformal & bilateral conflict threshold calibration) | `id: int`, `claim: str`, `evidence`, `cited_doc_ids` |
| `claims_dev.jsonl` | `300` claims | Held-out evaluation (`188` evidence claims yielding `209` pairs [`138` `SUPPORT`, `71` `CONTRADICT`] + `112` `NOINFO` claims; `0` claim-ID overlap with `train`) | `id: int`, `claim: str`, `evidence`, `cited_doc_ids` |
| `claims_test.jsonl` | `300` claims | Blind test split without gold labels (unused) | `id: int`, `claim: str`, `cited_doc_ids` |

### Open-Source Dataset Registry (`src/lab_ask_why/datasets.py`, 2019-2026)
`OPEN_DATASET_REGISTRY` registers `14` scientific claim verification, conflicting-evidence, and literature/protocol QA datasets (`labbench2`, `peerqa_xt`, `sciver`, `claimcheck`, `chembench`, `litqa2`, `lab_bench`, `conflictbank`, `faitheval_inconsistent`, `sciriff`, `bioasq`, `scifact_open`, `scifact`, `pubmedqa`), enforcing `PermissionError` whenever `purpose != "eval"` on `eval_only=True` benchmarks (`LAB-Bench`, `LitQA2`, `LABBench2`, `ChemBench`) or canary-bearing records. `probe_normalized_record` never reads `record.label` (`NEUTRAL` rationales are excluded from both support and contradict pools), and `build_conflictbank_claim_evidence_pairs` maps self-paired `CB_claim_evidence` rows (`200`) to `SUPPORT` while constructing `146` cross-subject entity-relation `CONTRADICT` pairs.

---

## 3. Model

1. **Zero-Parameter Okapi BM25 Lexical Baseline (`src/lab_ask_why/bm25.py`)**:
   - Biomedical tokenizer (`normalize_biomedical_text`, `tokenize`) mapping Greek characters (`alpha`, `beta`, `delta`, `micro`), preserving `+/-` allele markers (`alpha+`, `cd4-`, `trp53+/-`) and decimal/comma numbers, expanding negative contractions (`cannot`, `does not`), and retaining negation cues (`not`, `no`, `neither`, `without`, `never`).
   - Query-TF-saturated Okapi BM25 ($k_1 = 1.2$, $b = 0.75$, $k_3 = 1.2$) with background-only IDF (`exclude_doc_ids`) and query-budget score normalization in `[0, 1]`.
2. **Sparse Hybrid Retrieval, Reranker & Mondrian Split-Conformal Stance Verifier (`src/lab_ask_why/frontier_verifier.py`, `numpy`-only core `< 1.5 GB` RSS, optional `scipy` `[frontier]` extra)**:
   - **Stage 1 (`Reciprocal Rank Fusion`)**: Fuses (a) Okapi `BM25`, (b) subword character (`3`/`4`-gram) + biomedical stem/concept TF-IDF `TruncatedSVDSemanticIndex` (`96`-dim `LSA` subspace via sign-canonicalized Halko-Martinsson-Tropp randomized SVD using chunked `262K`-nnz `reduceat` pure-NumPy CSR matmuls matching `scipy.sparse` to `< 1e-15` in `1.43 GB` pure-NumPy / `1.12 GB` `scipy` RSS), and (c) `EntityNumericNegationIndex`.
   - **Stage 2 (`CrossFeatureReranker` + `SplitConformalStanceClassifier` + Bilateral Conflict)**: 26-feature damped Newton-IRLS L2 logistic reranker + separate open-corpus and candidate-abstract 28-feature Nesterov L2 multinomial 3-way classifiers over `("SUPPORT", "CONTRADICT", "NEI")` calibrated via Mondrian Split Conformal quantiles $\hat{q}_{1-\alpha}^{(y)} = s_{(\lceil (n_y + 1)(1 - \alpha) \rceil)}$ at $1 - \alpha = 0.90$ on `SciFact train_calib`, plus a dedicated bilateral cross-passage conflict score calibrated on non-conflicting single-passage controls (`src/lab_ask_why/resources/scifact_wire_calibrator.json`).
3. **Optional Pretrained Neural Bi-Encoder + DeBERTa-v3 NLI Verifier (`src/lab_ask_why/neural_verifier.py`, `[neural]` extra: `torch`, `transformers`)**:
   - Fuses `BAAI/bge-small-en-v1.5` dense bi-encoder retrieval with `BM25` and `EntityNumericNegationIndex` via RRF (`0.900` `doc_recall@5`) and scores candidate rationales with `cross-encoder/nli-deberta-v3-small` in length-sorted CPU batches, calibrated via Mondrian `1 - alpha = 0.90` Split Conformal sets strictly on `SciFact train` (`607` fit + `202` cal).

---

## 4. Training & Leak-Free Multi-Dataset Benchmarks (`results/open_benchmarks_2026.json`)

All supervised parameters and Mondrian conformal quantiles are fit **strictly once on `SciFact train` (`809` claims: `607` fit [`75%`] + `202` cal [`25%`])** on CPU (zero external LLM API calls) and evaluated zero-leakage across `SciFact dev` and `7` additional downloaded public corpora (`scripts/benchmark_open_datasets.py`, verified against drift and undeserved bolding by `tests/test_readme_drift.py`):

<!-- SCIFACT_FRONTIER_TABLE_START -->
| Metric (`SciFact dev`, `n=300` claims / `209` pairs) | Best Trivial Baseline | `Okapi BM25` | `Dense-SVD` | `Sparse Hybrid` (`numpy`) | `Neural BGE + DeBERTa-v3 NLI` |
|---|---:|---:|---:|---:|---:|
| Oracle Rationale `Hit@1` (`209` pairs) | `0.197` (random) | `0.483` | `0.445` | **`0.660`** | `0.574` |
| Oracle Rationale `Hit@3` (`209` pairs) | `0.528` (random) | `0.818` | `0.785` | **`0.909`** | `0.828` |
| Oracle Rationale `MRR` (`209` pairs) | `0.419` (random) | `0.660` | `0.633` | **`0.793`** | `0.716` |
| Oracle `Set-Recall@3` (`209` pairs) | -- | `0.775` | `0.737` | **`0.880`** | `0.785` |
| Open-Corpus `doc_recall@5` (`5,183` docs) | `0.001` (random) | `0.885` | `0.473` | `0.889` | **`0.900`** |
| Official Abstract `Label-Only F1` (`300` claims) | `0.000` (abstain) | `0.231` | -- | **`0.340`** | `0.285` |
| Official Abstract `Label+Rationales F1` (`300` claims) | `0.000` (abstain) | `0.182` | -- | **`0.309`** | `0.210` |
| Official Sentence `Selection-Only F1` (`300` claims) | `0.000` (abstain) | `0.152` | -- | **`0.308`** | `0.153` |
| Official Sentence `Selection+Label F1` (`300` claims) | `0.000` (abstain) | `0.123` | -- | **`0.211`** | `0.135` |
| 3-Way Stance `Accuracy` (`300` claims) | `0.413` (always SUPPORT) | `0.457` | -- | `0.560` | **`0.623`** |
| 3-Way Stance `Macro-F1` (`300` claims) | `0.195` (always SUPPORT) | `0.353` | `0.258` | `0.524` | **`0.590`** |

- **Sparse Hybrid 90% Mondrian Conformal (`SciFact dev` open-corpus, `n=300`)**: cov `0.917` (size `2.41`, 1-lbl `0.06`; S `0.97` [`0.92`,`0.99`], C `0.89` [`0.79`,`0.95`], N `0.88` [`0.80`,`0.92`])
- **Sparse Hybrid 90% Mondrian Conformal (`SciFact dev` per-abstract, `n=323`)**: cov `0.904` (size `2.09`, 1-lbl `0.17`; S `0.97` [`0.93`,`0.99`], C `0.79` [`0.68`,`0.87`], N `0.89` [`0.82`,`0.94`])
- **Neural BGE + DeBERTa-v3 NLI 90% Mondrian Conformal (`SciFact dev` open-corpus, `n=300`)**: cov `0.903` (size `2.08`, 1-lbl `0.22`; S `0.91` [`0.85`,`0.95`], C `0.94` [`0.85`,`0.98`], N `0.88` [`0.80`,`0.92`])
<!-- SCIFACT_FRONTIER_TABLE_END -->

Zero-shot cross-benchmark evaluation (`data_source = "real_downloaded"`, SHA-256 verified in gitignored `data/open_benchmarks/`):

<!-- OPEN_BENCHMARKS_2026_TABLE_START -->
| Dataset (`data_source`) | Evaluated ($n$) | License (`eval_only`) | Metric | Best Trivial Baseline | `Okapi BM25` | `Sparse Hybrid` (`numpy`) | `Neural BGE + DeBERTa-v3` | Conformal Coverage (Size, 1-Lbl, Per-Class [95% CI]) / Negative-Control Audit |
|---|---:|---|---|---:|---:|---:|---:|---|
| `SciFact` (`real_downloaded`) | `300` (`209` pairs) | `CC-BY-NC-2.0` (`False`) | Oracle `Hit@1` | `0.197` (random) | `0.483` | **`0.660`** | `0.574` | cov `0.917` (size `2.41`, 1-lbl `0.06`; S `0.97` [`0.92`,`0.99`], C `0.89` [`0.79`,`0.95`], N `0.88` [`0.80`,`0.92`]) |
| `PubMedQA` 3-way (`real_downloaded`) | `1000` (`pqa_labeled`) | `MIT` (`False`) | 3-Way `Accuracy` | `0.552` (always yes) | `0.259` | `0.461` | `0.476` | cov `0.880` (size `2.18`, 1-lbl `0.04`; S `0.97` [`0.95`,`0.98`], C `0.93` [`0.89`,`0.95`], N `0.30` [`0.22`,`0.39`]) |
| `PubMedQA` binary (`real_downloaded`) | `890` (`yes`/`no`) | `MIT` (`False`) | Binary `Macro-F1` | `0.383` (maj acc `0.620`) | `0.483` | **`0.563`** | `0.474` | `LONG_ANSWER` withheld |
| `ConflictBank` `CB_qa` (`real_downloaded`) | `200` pos + `200` neg | `CC-BY-SA-4.0` (`False`) | Conflict `Balanced Acc` | `0.500` (constant) | `0.497` | **`0.775`** | `0.525` | Sparse TPR `0.550` / neg-ctrl FPR `0.000` (AUROC `0.975`); Neural TPR `0.645` / FPR `0.595` (AUROC `0.510`) |
| `ConflictBank` pairs (`real_downloaded`) | `346` (`200` self + `146` cross) | `CC-BY-SA-4.0` (`False`) | Stance `Macro-F1` | `0.366` (self-200 acc `0.785`) | `0.364` | `0.474` | **`0.782`** | cov `0.965` (size `2.16`, 1-lbl `0.02`; S `0.99` [`0.97`,`1.00`], C `0.92` [`0.87`,`0.96`]) |
| `FaithEval-Inconsistent` (`real_downloaded`) | `300` pos + `287` neg | `CC-BY-NC-SA-4.0` (`False`) | Conflict `Balanced Acc` | `0.500` (constant) | `0.466` | **`0.572`** | `0.506` | Sparse TPR `0.143` / neg-ctrl FPR `0.000` (AUROC `0.840`); Neural TPR `0.047` / FPR `0.035` (AUROC `0.472`) |
| `CLAIMCHECK` critique IR (`real_downloaded`) | `108` critiques | `NOASSERTION` (`False`) | Critique `Hit@1` | `0.188` (random) | `0.306` | **`0.389`** | `0.324` | Sparse MRR `0.603`, Neural MRR `0.528` |
| `CLAIMCHECK` stance (`real_downloaded`) | `108` claims | `NOASSERTION` (`False`) | 3-Way `Macro-F1` | `0.228` (maj acc `0.519`) | `0.278` | `0.229` | **`0.377`** | cov `0.583` (size `1.48`, 1-lbl `0.66`; S `0.24` [`0.12`,`0.40`], C `0.33` [`0.16`,`0.56`], N `0.88` [`0.76`,`0.94`]) |
| `SciVer` (`real_downloaded`) | `1000` claims | `CC-BY-4.0` (`False`) | Stance `Macro-F1` | `0.336` (maj acc `0.505`) | `0.475` | `0.444` | **`0.657`** | cov `0.738` (size `2.19`, 1-lbl `0.16`; S `0.65` [`0.61`,`0.69`], C `0.83` [`0.79`,`0.86`]) |
| `ChemBench` -- option string matching against task text (`real_downloaded`) | `167` MCQ (`249` raw) | `MIT` (**`True`**) | Option `Hit@1` | `0.248` (chance) | `0.263` | **`0.303`** | `0.246` | Chance MRR `0.518`; `eval_only=True` verified |
| `LitQA2` -- with gold passage (`real_downloaded`) | `194` MCQ (`199` raw) | `CC-BY-SA-4.0` (**`True`**) | Option `Hit@1` | `0.244` (chance) | `0.504` | `0.564` | **`0.593`** | Chance MRR `0.507`; `eval_only=True` verified |
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
| Frontier Hybrid Verifier & Lean CI | PR `#32` (`d2570e7`) | Added `FrontierHybridVerifier` (`BM25 + LSA + Entity RRF` + `CrossFeatureReranker` + `90%` Split-Conformal stance sets), 8-corpus SHA-256 benchmark suite (`results/open_benchmarks_2026.json`), Lean 4.36.0 `#print axioms` + `160`-case `#eval` parity CI, and `tests/test_readme_drift.py`. |
| Audit `#33`-`#36` & Neural NLI Verifier | `fix/audit-33-36-neural-nli-and-conformal` | Gated `autolab_wire.py` with `check_ask_why_permille` (`0.35` NOINFO + content-word overlap) & `scifact_wire_calibrator.json`, added negative-control bilateral conflict detection, Official SciFact F1, chunked `< 1.5 GB` RSS pure-NumPy CSR SVD, and `NeuralBiEncoderNliVerifier` (`BAAI/bge-small-en-v1.5` + `cross-encoder/nli-deberta-v3-small`). |

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
2. **`check_ask_why_permille` Abstention & Calibrated Conformal Provenance**: Requires normalized max retrieval confidence `>= 0.35` (`350` permille) and `>= 1` shared non-stopword content token before emitting `SUPPORTED`/`CONTRADICTED`/`CONTESTED` (never falling back to synthetic aspirin sentences unless `allow_synthetic_fixture=True` or `use_demo_corpus=True`); otherwise emits `INSUFFICIENT_EVIDENCE`, `status="ABSTAIN"`, `conformal_stance_set=None`, `conformal_alpha=None`, and `stance_calibrated=False`. Grounded claims load `src/lab_ask_why/resources/scifact_wire_calibrator.json` (`stance_calibrated=True`, `conformal_alpha=0.10`).
3. **Formal `contested_claim_requires_experiment` Gate**: Blocks (`L6_CONTESTED_WITHOUT_EXPERIMENT` or `L6_AMBIGUOUS_WITHOUT_ORTHOGONAL_ASSAY`) whenever `suppress_discriminating_experiment=True` coincides with opposing evidence or `AMBIGUOUS_CHARACTERIZATION`.
