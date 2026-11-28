# lab-ask-why

A co-scientist that does not pick a winner. It asks *"why do you think so?"*, finds the evidence on each side, and names the result that would change minds.

> **Auto-Lab Ecosystem**: Layer `L6 · Metacognitive & Epistemic Brain` in [auto-lab](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)

## Where it fits

In the [Auto-Lab](https://github.com/yaniguan/auto-lab) agent ([yaniguan.github.io/auto-lab](https://yaniguan.github.io/auto-lab/) · [yaniguan.github.io](https://yaniguan.github.io/)), the co-scientist proposes the next simulation or experiment. Here it is a partner in a disagreement. It surfaces the evidence and assumptions behind each claim, then suggests the test that tells them apart. The first attempt finds the sentences behind a claim in real abstracts.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full architectural documentation, dataset breakdowns, retrieval equations, commit lineage, and sibling-repository contracts are maintained in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

### Object
Epistemic Socratic co-scientist (`L6`) that decomposes scientific disagreements without premature verdict collapse—retrieving verbatim supporting and contradicting rationale sentences from primary abstracts, surfacing required assumptions, and framing the discriminating experimental outcome that would change minds.

### Data
- **Active Training & Held-Out Benchmark (`SciFact`, Wadden et al. 2020)**: Downloaded on demand to gitignored `data/` (`CC BY-NC 2.0`, SHA-256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`, zero redistribution): `5,183` abstracts (`corpus.jsonl`; mean `9.7`, median `8` sentences/abstract), `809` training claims (`claims_train.jsonl`: `607` `train_fit` [`75%`] + `202` `train_calib` [`25%`] for Mondrian Split Conformal calibration; `181` dev-only and calibration-only abstracts excluded from Stage-1 IDF/SVD), and `300` held-out dev claims (`claims_dev.jsonl`, `0` claim-ID overlap with `train`), of which `188` evidence-bearing claims yield **`209` `(claim, doc_id)` pairs** (`138` `SUPPORT`, `71` `CONTRADICT`; `112` `NOINFO` claims).
- **8-Corpus Real Downloaded Benchmark Suite (`scripts/benchmark_open_datasets.py` -> `results/open_benchmarks_2026.json`)**: Downloads and SHA-256-verifies **`8` real public scientific corpora** (`SciFact`, `PubMedQA`, `ConflictBank`, `FaithEval-Inconsistent`, `CLAIMCHECK`, `SciVer`, `ChemBench`, and `LitQA2`) into gitignored `data/open_benchmarks/`, evaluating matched single-passage negative controls for conflict detection, trivial/majority/chance baselines, per-class conformal coverage with 95% Wilson CIs, and Official Wadden et al. (2020) SciFact abstract/sentence F1. Labels every output dict and CLI line with `data_source` (`"real_downloaded"` vs `"synthetic_fixture"`) and fails non-zero under `--offline` without verified cache unless `--allow-synthetic` is explicitly passed.
- **Open-Source Dataset Registry & Schema Adapters (`src/lab_ask_why/datasets.py`, 2019-2026)**: Typed `OpenDatasetSpec` registry and leak-free schema normalizers (`NormalizedClaimRecord`: `claim`, `evidence_sentences`, `label` in `SUPPORT`/`CONTRADICT`/`NOINFO`, with `probe_normalized_record` strictly invariant to `record.label` and `ConflictBank` `CB_claim_evidence` pairing `200` self-paired `SUPPORT` rows + `146` cross-subject `CONTRADICT` pairs) covering **`14` open-source scientific claim verification, conflicting-evidence, and literature/protocol QA datasets** (`LABBench2`, `PeerQA-XT`, `SciVer`, `CLAIMCHECK`, `ChemBench`, `LitQA2`, `LAB-Bench`, `ConflictBank`, `FaithEval-Inconsistent`, `SciRIFF`, `BioASQ`, `SciFact-Open`, `SciFact`, and `PubMedQA`), with strict `eval_only` canary guards on `LAB-Bench`, `LitQA2`, `LABBench2`, and `ChemBench`.
- **Curated Catalog**: Indexes `14` open scientific datasets, `9` open-source co-scientist / multi-agent debate / RAG repositories, and `13` foundational & 2026 causal-evidence / disagreement papers.

### Model
1. **Zero-Parameter Okapi `BM25` Lexical Baseline (`src/lab_ask_why/bm25.py`)**: Fast interpretable sentence and open-corpus ranker ($k_1 = 1.2$, $b = 0.75$, $k_3 = 1.2$) over biomedical-normalized tokens (Greek letters mapped to ASCII names, `+/-` allele markers and decimal/comma numbers preserved, negations `not`/`no`/`neither`/`without` retained) with background-document IDF (`exclude_doc_ids`) and smooth Laplace OOV IDF.
2. **Sparse Hybrid Retrieval + Cross-Feature Reranker + Mondrian Split-Conformal Stance Verifier (`src/lab_ask_why/frontier_verifier.py`, `numpy`-only core `< 1.5 GB` RSS, optional `scipy` `[frontier]` extra)**:
   - **Stage 1 (`Reciprocal Rank Fusion`)**: Fuses biomedical Okapi `BM25` + subword character/word TF-IDF Truncated-SVD (`LSA` `96`-dim dense semantic subspace via chunked `262K`-nnz `reduceat` pure-NumPy CSR linear algebra matching `scipy.sparse` to `< 1e-15` in `1.43 GB` pure-NumPy / `1.12 GB` `scipy` RSS) + `EntityNumericNegationIndex` (acronyms, gene/allele markers, numeric literals, negations).
   - **Stage 2 (`CrossFeatureReranker` + `SplitConformalStanceClassifier` + Bilateral Conflict Score)**: 26-feature pairwise claim-sentence interaction reranker + separate open-corpus and per-abstract 28-feature 3-way stance heads over `("SUPPORT", "CONTRADICT", "NEI")` calibrated via Mondrian **Split Conformal Prediction (`1 - alpha = 0.90`)** + dedicated **Bilateral Cross-Passage Conflict Score** calibrated strictly on non-conflicting `SciFact train_calib` single-passage controls (`save_calibrator` / `load_calibrator` persisted in `< 12 KB` at `src/lab_ask_why/resources/scifact_wire_calibrator.json`).
3. **Optional Pretrained Neural Bi-Encoder + DeBERTa-v3 Cross-Encoder NLI Verifier (`src/lab_ask_why/neural_verifier.py`, `[neural]` extra: `torch`, `transformers`)**:
   - **Stage 1 (`Dense + Lexical RRF`)**: Fuses `BAAI/bge-small-en-v1.5` (`384`-dim CLS embeddings) + `BM25` + `EntityNumericNegationIndex` via RRF (`0.900` open-corpus `doc_recall@5` on `SciFact dev`).
   - **Stage 2 (`DeBERTa-v3 3-Way NLI` + Mondrian Split Conformal)**: Scores `(premise, claim)` pairs with `cross-encoder/nli-deberta-v3-small` in length-sorted `torch.no_grad()` CPU batches and calibrates Mondrian `1 - alpha = 0.90` Split Conformal sets strictly on `SciFact train` (`607` fit + `202` cal).

### Training & Multi-Dataset Benchmarks (`results/open_benchmarks_2026.json`)
Both `FrontierHybridVerifier` (`numpy` core; optional `[frontier]` `scipy` acceleration) and `NeuralBiEncoderNliVerifier` (`[neural]` extra) are fit/calibrated **strictly once on `SciFact train` (`809` claims: `607` fit [`75%`] + `202` Split-Conformal calibration [`25%`]; `181` dev-only and calibration-only abstracts excluded from Stage-1 IDF/SVD)** on CPU (zero external LLM API calls) and evaluated zero-leakage across `SciFact dev` and `7` additional real public benchmarks. Cells are bolded dynamically by `tests/test_readme_drift.py` **only** when beating `Best Trivial Baseline`, `Okapi BM25`, and every competing column in the row:

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

Zero-shot evaluation across all `8` real SHA-256-verified public corpora (`data_source = "real_downloaded"`):

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

### History
Initialized in `6236123` (`2026-11-18`) with `demo/why_probe.py`, reference output (`demo/output.txt`), and curated datasets/repos/papers (`2018`-`2026`); hardened across PRs `#23`, `#30`, `#31`, and `#32` with typed schemas, Lean 4 proofs, `autolab-wire/v1`, `FrontierHybridVerifier`, and the `14`-dataset registry; audited and upgraded (`#33`-`#36`) with `check_ask_why_permille` wire abstention & calibrated `scifact_wire_calibrator.json`, bilateral cross-passage conflict detection with matched single-passage negative controls, chunked `< 1.5 GB` RSS pure-NumPy CSR SVD, Official Wadden et al. (2020) SciFact metrics, and `NeuralBiEncoderNliVerifier` (`BAAI/bge-small-en-v1.5` + `cross-encoder/nli-deberta-v3-small`).

### Unique Properties
- **Negative-Control-Audited Bilateral Conflict Detection & Mondrian Conformal Sets**: Dedicated bilateral cross-passage conflict scoring achieves **`0.550` TPR at `0.000` single-passage negative-control FPR (`0.975` AUROC, `0.775` balanced accuracy)** on `ConflictBank CB_qa` (`n=200` pos + `200` neg) and **`0.143` TPR at `0.000` negative-control FPR (`0.840` AUROC)** on `FaithEval-Inconsistent` (`n=300` pos + `287` neg), while `1 - alpha = 0.90` Mondrian Split Conformal sets reach `0.903` (`Neural`, avg size `2.08`, single-label `0.22`) and `0.917` (`Sparse`, avg size `2.41`) coverage on `SciFact dev`.
- **Three-Tier Auditable Stack (`BM25` + `FrontierHybridVerifier` + `NeuralBiEncoderNliVerifier` + Lean 4.36.0)**: Pairs zero-parameter Okapi `BM25` (`0.483` Hit@1, `0.813` Hit@3 on `demo/why_probe.py`) with `numpy`-only `FrontierHybridVerifier` (`0.660` Hit@1, `0.909` Hit@3, `0.793` MRR, `0.340` Official Abstract F1 in `< 1.5 GB` RSS), optional CPU `NeuralBiEncoderNliVerifier` (`0.900` `doc_recall@5`, `0.590` 3-way Stance Macro-F1 on `SciFact`, `0.782` on `ConflictBank` pairs, `0.657` on `SciVer`, `0.593` on `LitQA2`), and `21` Lean 4.36.0 theorems (`0 sorryAx`, `160/160` `#eval` Python parity cases).
- **Ecosystem Interfaces**: Pairs at `L6` with [`lab-decide`](https://github.com/yaniguan/lab-decide) (conformal trust-region gating & BO), connects upstream to [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`), [`YLM`](https://github.com/yaniguan/YLM) (`L0`), and [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` `LitQA2`/`ProtocolQA` & calibrated rewards), feeds discriminating tests downstream to [`labshift`](https://github.com/yaniguan/labshift) (`L5`), [`lab-verify`](https://github.com/yaniguan/lab-verify), and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`), and closes post-run causal attribution through [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`).

## Repository Layout

```text
lab-ask-why/
├── .github/workflows/
│   └── demo.yml                      # CI: Python 3.12 pytest + demo diff + Lean 4.36.0 axiom & parity job
├── community_extensions/             # Elo/EVSI crux synthesizer & precursor ghost-peak deconvolver
├── demo/
│   ├── output.txt                    # Deterministic reference stdout capture on SciFact dev (CPU)
│   └── why_probe.py                  # CLI & zero-parameter BM25 benchmark entrypoint
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── lean/
│   └── LabAskWhyLean.lean            # Lean 4.36.0 formalization (21 theorems, 0 sorry, conformal sets)
├── results/
│   └── open_benchmarks_2026.json     # Committed 8-corpus SHA-256 benchmark report (real_downloaded)
├── scripts/
│   ├── benchmark_open_datasets.py    # Leak-free 8-corpus downloader, SHA-256 verifier & evaluator
│   ├── fetch_open_datasets.py        # On-demand fetcher & schema normalizer into gitignored data/
│   └── verify_lean_parity.py         # Lean 4.36.0 #print axioms audit & 160-case #eval parity test
├── src/lab_ask_why/                  # Package (bm25, frontier_verifier, neural_verifier, datasets, resources/)
├── tests/
│   ├── test_autolab_wire.py          # Fail-closed wire contract & conformal stance tests
│   ├── test_community_extensions.py  # Crux tournament & precursor deconvolution tests
│   ├── test_frontier_verifier.py     # Hybrid RRF, reranker, Split Conformal & Lean parity tests
│   ├── test_open_datasets_2026.py    # Offline tests for 2019-2026 dataset registry, guards & adapters
│   ├── test_readme_drift.py          # Zero-drift check across README, ARCHITECTURE & benchmark JSON
│   └── test_why_probe.py             # Offline pytest regression suite (#1..#29)
├── autolab_wire.py                   # 15-repo Auto-Lab L6 wire node (process_packet)
├── .gitignore                        # Excludes downloaded data/, lean/.lake/, and caches
├── LICENSE                           # MIT License (code)
├── pyproject.toml                    # Package metadata & `lab-ask-why` CLI console script
├── README.md                         # Overview, benchmark tables, and curated co-scientist catalog
└── requirements.txt                  # Bounded dependency (numpy>=1.26,<3.0)
```

## Quickstart & CLI

```bash
pip install -r requirements.txt -e .          # Core numpy-only runtime (< 1.5 GB RSS)
pip install -e ".[frontier]"                  # Optional scipy / scikit-learn sparse acceleration
pip install -e ".[neural]"                    # Optional CPU BGE + DeBERTa-v3 NLI verifier
python demo/why_probe.py
python scripts/benchmark_open_datasets.py --offline
python scripts/verify_lean_parity.py --clean
lab-ask-why --list-datasets
lab-ask-why --claim "GWAS identifies rare synthetic associations." --offline --json
lab-ask-why --export-schema epistemic_probe.schema.json
python scripts/fetch_open_datasets.py --dataset pubmedqa --max-rows 20 --probe
```

Zero-parameter lexical baseline reference output (`demo/output.txt`, CPU only):

```
SciFact dev: rationale sentence retrieval & open-corpus epistemic probing
total claims: 300   with evidence: 188   NOINFO: 112   claim-doc pairs: 209   labels: {'SUPPORT': 138, 'CONTRADICT': 71}
sentences per abstract: mean 9.7, median 8   (IDF scope: background)

method                     hit@1     hit@3  sent_rec@1  sent_rec@3  set_rec@1  set_rec@3
BM25                       0.483     0.813       0.341       0.658      0.455      0.775
lead-k (position)          0.057     0.263
conclusion-first           0.249     0.679
random (20 seeds)          0.197     0.528
  random std               0.024     0.029
BM25 SUPPORT               0.514     0.826       0.376       0.685      0.486      0.775   (n=138)
BM25 CONTRADICT            0.423     0.789       0.274       0.605      0.394      0.775   (n=71)

Epistemic abstention (SUPPORT/CONTRADICT vs NOINFO cited abstracts, n_noinfo_pairs=114): max-BM25 mean 23.99 vs 13.34 | AUROC = 0.765
Open-corpus 2-stage BM25 (all 5,183 abstracts, top-5 docs -> top-3 sents): doc_recall@5 = 0.880 | sentence_hit@3 = 0.596 | NOINFO FDR = 0.321
```

Plain BM25 finds a supporting or contradicting reason in the top 3 for `0.813` of claim-abstract pairs (`0.826` `SUPPORT` vs `0.789` `CONTRADICT`), against `0.528` for random ranking, while `FrontierHybridVerifier` (`BM25 + LSA + Entity RRF` + `CrossFeatureReranker` fit strictly on `SciFact train`) lifts `SciFact dev` rationale retrieval to **`0.660` Hit@1**, **`0.909` Hit@3**, and **`0.793` MRR** (`0.917` open-corpus / `0.904` per-abstract `90%` Mondrian conformal coverage) and `NeuralBiEncoderNliVerifier` lifts open-corpus `doc_recall@5` to **`0.900`** and 3-way Stance Macro-F1 to **`0.590`** (`0.903` Mondrian conformal coverage, avg set size `2.08`).

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ask-why"`, `LAYER = "L6"`) connects `lab-ask-why` into the deterministic 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) execution and post-training loop via `process_packet(packet: dict) -> dict`:

- **Fail-Closed Input & Certificate Validation**: Rejects non-dict packets (`L6_INVALID_PACKET_TYPE`), non-boolean flags (`L6_INVALID_BOOLEAN_FLAG`), non-string/empty claims (`L6_INVALID_CLAIM`), malformed evidence lists (`L6_INVALID_EVIDENCE_SENTENCES`), and malformed or unsound `CERTIFIED_NEGATIVE` certificates (`L6_MALFORMED_NEGATIVE_CERTIFICATE`) cleanly (`halted=True`, `status="BLOCK"`, zero uncaught crashes).
- **Upstream Handoff & Explicit Demo Isolation (`lab-ears` -> `lab-ask-why`)**: Consumes `goal_text` and structured protocol `parsed_spans` extracted by [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`), alongside caller-provided `evidence_sentences` (never substituting a synthetic aspirin corpus unless `allow_synthetic_fixture=True` or `use_demo_corpus=True` is explicitly set, and never padding queries with unrelated anchor text).
- **`check_ask_why_permille` Abstention Gate, Calibrated Conformal Sets & Lean Guardrail**: Requires normalized max retrieval confidence `>= 0.35` (`350` permille) and `>= 1` shared non-stopword content token between claim and top rationale before emitting `SUPPORTED`/`CONTRADICTED`/`CONTESTED`; otherwise emits `epistemic_status="INSUFFICIENT_EVIDENCE"`, `status="ABSTAIN"`, `conformal_stance_set=None`, `conformal_alpha=None`, and `stance_calibrated=False` (`halted=True, status="BLOCK"` under `strict_ask_why=True`). Grounded claims load `src/lab_ask_why/resources/scifact_wire_calibrator.json` (`SciFact train` `607`/`202` Mondrian calibration, `stance_calibrated=True`, `conformal_alpha=0.10`) and enforce `LabAskWhyLean.contested_claim_requires_experiment`.
- **Downstream Gate & Closed-Loop Attribution (`lab-ask-why` -> `lab-decide` ... -> `lab-attrib`)**: Emits `out["epistemic_probe"]` (`data_source`, `conformal_stance_set`, `stance_probabilities`, `support_top`, `contradict_top`, `why_questions`, `discriminating_question`, `control_wells`, `has_discriminating_experiment`) downstream to [`lab-decide`](https://github.com/yaniguan/lab-decide) (`L6`), proceeding through `buy-or-make`, `lab-automation-atlas`, and `labshift` (`L5`), `lab-rehearsal` and `lab-verify` (`L4`), `stepbot` and `lab-hands` (`L1`), and `lab-feet` (`L0`), before post-execution causal blame, NNLS precursor deconvolution, and SHA-256 certified memory feedback via [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`) and `twinstep` / `labfill` / `labbench-posttrain` (`L3`).

## Datasets

All `14` datasets below are registered in `src/lab_ask_why/datasets.py` (`OPEN_DATASET_REGISTRY`) with typed `OpenDatasetSpec` metadata, SPDX / `commercial_use_ok` / `eval_only` guards, and schema adapters normalizing records (`claim`, `evidence_sentences`, `label` in `SUPPORT` / `CONTRADICT` / `NOINFO`) directly into `lab-ask-why`'s Okapi `BM25` + Socratic `EpistemicProbeResult` pipeline (`lab-ask-why --list-datasets`, `scripts/fetch_open_datasets.py`):

| Dataset (`dataset_id`) | Year | What | Hub / Repo ID | SPDX License | Commercial OK | Eval Only |
|---|---:|---|---|---|---|---|
| [LABBench2 (`labbench2`)](https://huggingface.co/datasets/EdisonScientific/labbench2) | 2026 | ~1,900 biology research tasks (`litqa3`, `protocolqa2`, `sourcequality`, `suppqa2`, `trialqa`; [arXiv:2604.09554](https://arxiv.org/abs/2604.09554)) | `EdisonScientific/labbench2` | `CC-BY-SA-4.0` | Yes | **Yes** (canary-guarded) |
| [PeerQA-XT (`peerqa_xt`)](https://huggingface.co/datasets/UKPLab/PeerQA-XT) | 2026 | Cross-domain scientific peer-review QA grounded in biomedical & ML full-text papers ([arXiv:2502.13668](https://arxiv.org/abs/2502.13668)) | `UKPLab/PeerQA-XT` | `CC-BY-NC-SA-4.0` | **No** (**non-commercial**) | No |
| [SciVer (`sciver`)](https://huggingface.co/datasets/chengyewang/SciVer) | 2025 | ACL 2025 multimodal scientific claim verification (3,000 claims across 1,113 papers with rationales; [arXiv:2506.15569](https://arxiv.org/abs/2506.15569)) | `chengyewang/SciVer` | `CC-BY-4.0` | Yes | No |
| [CLAIMCHECK (`claimcheck`)](https://github.com/JHU-CLSP/CLAIMCHECK) | 2025 | EMNLP 2025 NeurIPS peer-review weakness critiques linked to target paper claims ([arXiv:2503.21717](https://arxiv.org/abs/2503.21717)) | `JHU-CLSP/CLAIMCHECK` | `NOASSERTION` | **No** (unasserted) | No |
| [ChemBench (`chembench`)](https://huggingface.co/datasets/jablonkagroup/ChemBench) | 2025 | Chemical reasoning, literature & property QA across chemistry subfields ([arXiv:2411.16955](https://arxiv.org/abs/2411.16955)) | `jablonkagroup/ChemBench` | `MIT` | Yes | **Yes** (canary-guarded) |
| [LitQA2 (`litqa2`)](https://huggingface.co/datasets/futurehouse/lab-bench) | 2024 | Full-text biological literature comprehension & key-passage grounding ([arXiv:2407.10362](https://arxiv.org/abs/2407.10362)) | `futurehouse/lab-bench` | `CC-BY-SA-4.0` | Yes | **Yes** (canary-guarded) |
| [LAB-Bench (`lab_bench`)](https://huggingface.co/datasets/futurehouse/lab-bench) | 2024 | Wet-lab protocol troubleshooting (`ProtocolQA`), `SuppQA`, and `DbQA` ([arXiv:2407.10362](https://arxiv.org/abs/2407.10362)) | `futurehouse/lab-bench` | `CC-BY-SA-4.0` | Yes | **Yes** (canary-guarded) |
| [ConflictBank (`conflictbank`)](https://huggingface.co/datasets/Warrieryes/CB_claim_evidence) | 2024 | NeurIPS 2024 parametric vs. contextual knowledge conflicts (misinformation, temporal, semantic; [arXiv:2408.12076](https://arxiv.org/abs/2408.12076)) | `Warrieryes/CB_claim_evidence` | `CC-BY-SA-4.0` | Yes | No |
| [FaithEval-Inconsistent (`faitheval_inconsistent`)](https://huggingface.co/datasets/Salesforce/FaithEval-inconsistent-v1.0) | 2024 | Multi-document conflicting-context QA & abstention evaluation ([arXiv:2410.03727](https://arxiv.org/abs/2410.03727)) | `Salesforce/FaithEval-inconsistent-v1.0` | `CC-BY-NC-SA-4.0` | **No** (**non-commercial**) | No |
| [SciRIFF (`sciriff`)](https://huggingface.co/datasets/allenai/SciRIFF) | 2024 | NeurIPS 2024 unified 54-task scientific literature instruction & claim verification corpus ([arXiv:2406.07835](https://arxiv.org/abs/2406.07835)) | `allenai/SciRIFF` | `ODC-BY-1.0` | Yes (see per-task terms) | No |
| [BioASQ (`bioasq`)](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) | 2024 | Biomedical literature QA & PubMed passage retrieval (`rag-mini-bioasq`) | `rag-datasets/rag-mini-bioasq` | `CC-BY-2.5` | Yes | No |
| [SciFact-Open (`scifact_open`)](https://github.com/dwadden/scifact-open) | 2022 | Open-domain scientific claim verification scaled to 500K S2ORC abstracts ([arXiv:2210.13777](https://arxiv.org/abs/2210.13777)) | `dwadden/scifact-open` | `CC-BY-NC-2.0` | **No** (**non-commercial**) | No |
| [SciFact (`scifact`)](https://huggingface.co/datasets/allenai/scifact) | 2020 | 1,409 expert biomedical claims with sentence-level `SUPPORT`/`CONTRADICT` rationales ([arXiv:2004.14974](https://arxiv.org/abs/2004.14974)) | `allenai/scifact` | `CC-BY-NC-2.0` | **No** (**non-commercial**) | No |
| [PubMedQA (`pubmedqa`)](https://huggingface.co/datasets/qiaojin/PubMedQA) | 2019 | Biomedical research QA with `yes`/`no`/`maybe` abstract verdicts ([arXiv:1909.06146](https://arxiv.org/abs/1909.06146)) | `qiaojin/PubMedQA` | `MIT` | Yes | No |

Licenses marked **non-commercial** (or `NOASSERTION`) are fine for research evaluation when fetched on demand into gitignored `data/`, and datasets marked **Eval Only** (`LAB-Bench`, `LitQA2`, `LABBench2`, `ChemBench`) enforce `PermissionError` whenever `purpose != "eval"` so benchmark items never enter training, SFT, or RL corpora.

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---:|---|
| [stanford-oval/storm](https://github.com/stanford-oval/storm) | An LLM-powered knowledge curation system that researches a topic and generates a... | 31,907 | MIT |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery | 15,014 | custom (see repo) |
| [Future-House/paper-qa](https://github.com/Future-House/paper-qa) | High accuracy RAG for answering questions from scientific documents with citations | 9,621 | Apache-2.0 |
| [ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public) | Chemcrow | 971 | MIT |
| [Future-House/robin](https://github.com/Future-House/robin) | Robin: A multi-agent system for automating scientific discovery | 760 | Apache-2.0 |
| [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate) | ICML 2024: Improving Factuality and Reasoning in Language Models through Multiagent Debate | 557 | no license file |
| [NoviScl/AI-Researcher](https://github.com/NoviScl/AI-Researcher) | - | 415 | MIT |
| [allenai/scifact](https://github.com/allenai/scifact) | Data and models for the SciFact verification task. | 282 | custom (see repo) |
| [gomesgroup/coscientist](https://github.com/gomesgroup/coscientist) | - | 214 | custom (see repo) |

## Papers

| Paper | Venue | Link |
|---|---|---|
| Accelerating scientific discovery with Co-Scientist | arXiv 2025 | [arXiv:2502.18864](https://arxiv.org/abs/2502.18864) |
| The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery | arXiv 2024 | [arXiv:2408.06292](https://arxiv.org/abs/2408.06292) |
| Robin: A multi-agent system for automating scientific discovery | arXiv 2025 | [arXiv:2505.13400](https://arxiv.org/abs/2505.13400) |
| Emergent autonomous scientific research capabilities of large language models | arXiv 2023 | [arXiv:2304.05332](https://arxiv.org/abs/2304.05332) |
| ChemCrow: Augmenting large-language models with chemistry tools | arXiv 2023 | [arXiv:2304.05376](https://arxiv.org/abs/2304.05376) |
| AI safety via debate | arXiv 2018 | [arXiv:1805.00899](https://arxiv.org/abs/1805.00899) |
| Improving Factuality and Reasoning in Language Models through Multiagent Debate | ICML 2024 | [arXiv:2305.14325](https://arxiv.org/abs/2305.14325) |
| Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers | arXiv 2024 | [arXiv:2409.04109](https://arxiv.org/abs/2409.04109) |
| Fact or Fiction: Verifying Scientific Claims | EMNLP 2020 | [arXiv:2004.14974](https://arxiv.org/abs/2004.14974) |

## Conferences and workshops

- [NeurIPS (main track, Datasets and Benchmarks, workshops)](https://neurips.cc/)
- [ICLR](https://iclr.cc/)
- [ACL 2026](https://2026.aclweb.org/)
- [AI for Science workshop series (NeurIPS / ICML / ICLR)](https://ai4sciencecommunity.github.io/)

## Progress

Recent results, newest first. Date = first arXiv version; each line is a short summary of the abstract.

| Date | Paper | In one line |
|---|---|---|
| 2026-10-09 | [ACED-Bench: Evaluating How LLM Agents Acquire and Act on Causal Evidence](https://arxiv.org/abs/2610.14381) | ACED-Bench: tests whether agents choose experiments and act on causal evidence, with objective ground truth. |
| 2026-08-20 | [Symposium: Trust via Auditable Records for Communities of AI Scientist Agents](https://arxiv.org/abs/2608.19511) | Symposium: auditable records of claims, evidence and assumptions for communities of AI scientist agents. |
| 2026-05-11 | [When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](https://arxiv.org/abs/2605.10171) | When Reviews Disagree: fine-grained contradiction analysis across scientific peer reviews. |
| 2026-02-12 | [Benchmark Illusion: Disagreement among LLMs and Its Scientific Consequences](https://arxiv.org/abs/2602.11898) | Benchmark Illusion: models with similar accuracy still disagree on 16-66% of items. |

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
