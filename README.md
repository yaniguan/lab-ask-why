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
- **Active Benchmark (`SciFact`, Wadden et al. 2020)**: Downloaded on demand to gitignored `data/` (`CC BY-NC 2.0`, zero redistribution): `5,183` abstracts (`corpus.jsonl`; mean `9.7`, median `8` sentences/abstract) and `300` dev claims (`claims_dev.jsonl`), of which `188` evidence-bearing claims yield **`209` `(claim, doc_id)` pairs** (`138` `SUPPORT`, `71` `CONTRADICT`; `112` `NOINFO` claims excluded).
- **Open-Source Dataset Registry & Schema Adapters (`src/lab_ask_why/datasets.py`, 2019–2026)**: Typed `OpenDatasetSpec` registry and schema normalizers (`NormalizedClaimRecord`: `claim`, `evidence_sentences`, `label` in `SUPPORT`/`CONTRADICT`/`NOINFO`) covering **`14` open-source scientific claim verification, conflicting-evidence, and literature/protocol QA datasets** (`LABBench2`, `PeerQA-XT`, `SciVer`, `CLAIMCHECK`, `ChemBench`, `LitQA2`, `LAB-Bench`, `ConflictBank`, `FaithEval-Inconsistent`, `SciRIFF`, `BioASQ`, `SciFact-Open`, `SciFact`, and `PubMedQA`), with strict `eval_only` canary guards on `LAB-Bench`, `LitQA2`, `LABBench2`, and `ChemBench`.
- **Curated Catalog**: Indexes `14` open scientific datasets, `9` open-source co-scientist / multi-agent debate / RAG repositories, and `13` foundational & 2026 causal-evidence / disagreement papers.

### Model
Zero-parameter indexed Okapi `BM25` sentence and open-corpus ranker ($k_1 = 1.2$, $b = 0.75$, $k_3 = 1.2$) over biomedical-normalized tokens (Greek letters mapped to ASCII names, `+/-` allele markers and decimal/comma numbers preserved, negations `not`/`no`/`neither`/`without` retained) with background-document IDF (`exclude_doc_ids`) and smooth Laplace OOV IDF, benchmarked against `lead-k`, `conclusion-first`, and 20-seed random baselines plus rationale-grounded Socratic `why_questions` and typed `EpistemicProbeResult` JSONL output.

### Training
Zero parametric training or LLM API calls (`numpy` + Python standard library on CPU). On SciFact dev (`300` claims: `188` with evidence yielding `209` pairs + `112` `NOINFO` claims), oracle-abstract `BM25` achieves **`0.813` Hit@3**, **`0.658` fractional Sentence-Recall@3**, and **`0.775` complete Rationale-Set-Recall@3** (`0.826` `SUPPORT` vs `0.789` `CONTRADICT` Hit@3), separates `SUPPORT`/`CONTRADICT` from `NOINFO` cited abstracts at **`0.765` AUROC**, and retrieves **`0.880` `doc_recall@5`** (**`0.596` `sentence_hit@3`**, **`0.321` `NOINFO FDR`**) in two-stage hybrid open-corpus retrieval across all `5,183` abstracts.

### History
Initialized in `6236123` (`2026-11-18`) with `demo/why_probe.py`, reference output (`demo/output.txt`), Python 3.12 CI workflow (`.github/workflows/demo.yml`), and curated datasets, repositories, and papers (`2018`–`2026`).

### Unique Properties
- **Polarity-Symmetric Stance Neutrality**: Retrieves supporting and contradicting rationales with near-equal fidelity (`0.797` vs. `0.775` top-3 hit rate) rather than forcing binary adjudication.
- **Zero-Cost Auditable Baseline**: Establishes a fast, deterministic lexical attribution baseline before adding open-corpus retrieval or neural rerankers.
- **Ecosystem Interfaces**: Pairs at `L6` with [`lab-decide`](https://github.com/yaniguan/lab-decide) (conformal trust-region gating & BO), connects upstream to [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`), [`YLM`](https://github.com/yaniguan/YLM) (`L0`), and [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` `LitQA2`/`ProtocolQA` & calibrated rewards), feeds discriminating tests downstream to [`labshift`](https://github.com/yaniguan/labshift) (`L5`), [`lab-verify`](https://github.com/yaniguan/lab-verify), and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`), and closes post-run causal attribution through [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`).

## Repository Layout

```text
lab-ask-why/
├── .github/workflows/
│   └── demo.yml                      # CI: cache S3 archive, run offline pytest, diff demo/output.txt
├── community_extensions/             # Elo/EVSI crux synthesizer & precursor ghost-peak deconvolver
├── demo/
│   ├── output.txt                    # Deterministic reference stdout capture on SciFact dev (CPU)
│   └── why_probe.py                  # CLI & benchmark entrypoint for SciFact / custom JSONL corpora
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── lean/
│   └── LabAskWhyLean.lean            # Lean 4 formalization (contested_claim_requires_experiment)
├── scripts/
│   └── fetch_open_datasets.py        # On-demand fetcher & schema normalizer into gitignored data/
├── src/lab_ask_why/                  # Installable package (BM25, TwoStageRetriever, datasets, LAB-Bench)
├── tests/
│   ├── test_autolab_wire.py          # Wire contract tests (autolab-wire/v1)
│   ├── test_community_extensions.py  # Crux tournament & precursor deconvolution tests
│   ├── test_open_datasets_2026.py    # Offline tests for 2019-2026 dataset registry, guards & adapters
│   └── test_why_probe.py             # Offline pytest regression suite (#1..#22)
├── autolab_wire.py                   # 15-repo Auto-Lab L6 wire node (process_packet)
├── .gitignore                        # Excludes downloaded CC BY-NC 2.0 SciFact data/ & caches
├── LICENSE                           # MIT License (code)
├── pyproject.toml                    # Package metadata & `lab-ask-why` CLI console script
├── README.md                         # Overview, demo results, and curated co-scientist/debate catalog
└── requirements.txt                  # Bounded dependency (numpy>=1.26,<3.0)
```

## Quickstart & CLI

```bash
pip install -r requirements.txt -e .
python demo/why_probe.py
lab-ask-why --list-datasets
lab-ask-why --claim "GWAS identifies rare synthetic associations." --offline --json
lab-ask-why --export-schema epistemic_probe.schema.json
python scripts/fetch_open_datasets.py --dataset pubmedqa --max-rows 20 --probe
```

Output (`demo/output.txt`, CPU only):

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

Plain BM25 finds a supporting or contradicting reason in the top 3 for 0.79 of claim-abstract pairs, against 0.53 for random ranking (abstracts are short, about 10 sentences). It works about equally for SUPPORT and CONTRADICT. Next: retrieve across many papers (not only the gold abstract) and turn retrieved reasons into a proposed discriminating experiment.

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ask-why"`, `LAYER = "L6"`) connects `lab-ask-why` into the deterministic 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) execution and post-training loop via `process_packet(packet: dict) -> dict`:

- **Upstream Handoff (`lab-ears` -> `lab-ask-why`)**: Consumes `goal_text` and structured protocol `parsed_spans` (`volume_ul`, `temp_c`, `duration_min`, `amounts`, `temperatures`, `times`, `speeds`) extracted by [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`), alongside optional `evidence_sentences` (falling back to a built-in 6-sentence offline wet-lab solubility and protocol support/contradict corpus requiring zero network downloads).
- **Symmetric Retrieval & Lean Guardrail**: Ranks supporting and contradicting rationales with Okapi `BM25` and synthesizes Socratic `why_questions`. Enforces `LabAskWhyLean.contested_claim_requires_experiment`: if `suppress_discriminating_experiment=True` while both supporting and contradicting rationales are present, sets `halted=True` and `halt_reason="L6_CONTESTED_WITHOUT_EXPERIMENT"` (`trace` status `"BLOCK"`, with falsifiable control design logged in `bottom_reasoning_traces`).
- **Downstream Gate & Closed-Loop Attribution (`lab-ask-why` -> `lab-decide` ... -> `lab-attrib`)**: Emits `out["epistemic_probe"]` (`support_top`, `contradict_top`, `why_questions`, `discriminating_question`, `has_discriminating_experiment`) downstream to [`lab-decide`](https://github.com/yaniguan/lab-decide) (`L6` split-conformal trust gate), proceeding through `buy-or-make`, `lab-automation-atlas`, and `labshift` (`L5`), `lab-rehearsal` and `lab-verify` (`L4`), `stepbot` and `lab-hands` (`L1`), and `lab-feet` (`L0`), before post-execution causal blame, NNLS precursor deconvolution, and SHA-256 certified memory feedback via [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`) and `twinstep` / `labfill` / `labbench-posttrain` (`L3`).

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
