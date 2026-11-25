# Architecture and History — `lab-ask-why`

> **Auto-Lab Ecosystem Role**: Layer `L6 · Metacognitive & Epistemic Brain` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Standard scientific question-answering and automated co-scientist systems optimize to declare a single winning hypothesis, hiding competing evidence, unstated mechanistic assumptions, and substantial inter-model disagreement (even models with comparable benchmark accuracy disagree on 16–66% of scientific items). When human researchers or multi-agent scientist ensembles disagree on a claim, premature verdict collapse causes wasted wet-lab runs and confirmation bias.

`lab-ask-why` implements an **epistemic Socratic co-scientist** that deliberately refuses to pick a winner. Instead, it asks *"why do you think so?"*, retrieves the concrete rationale sentences supporting or contradicting each side from primary literature, surfaces the latent assumptions required for a claim to hold, and frames the discriminating experimental result that would change minds.

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `lab-ask-why` sits at the top-level epistemic layer **`L6 · Metacognitive & Epistemic Brain`** alongside [`lab-decide`](https://github.com/yaniguan/lab-decide):
- **Peer (`L6` Metacognitive Brain)**: Pairs directly with [`lab-decide`](https://github.com/yaniguan/lab-decide). While `lab-ask-why` decomposes competing scientific hypotheses into supporting/contradicting rationales and proposes *which* discriminating experiment resolves a disagreement, `lab-decide` quantifies whether candidate experiments lie inside a calibrated conformal trust region or require abstention / active data acquisition.
- **Upstream (`L0`–`L3` Literature, Simulation & Post-Trained Reasoners)**: Draws claims, literature context, and candidate mechanistic predictions from [`YLM`](https://github.com/yaniguan/YLM) (`L0` scientific language model), [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` `LitQA2`/`ProtocolQA` evaluators and `CalibratedMCQTemplate` / `FaultDiagnosisRepairTemplate` policies), [`WignerFlow`](https://github.com/yaniguan/WignerFlow), and [`assayfold`](https://github.com/yaniguan/assayfold).
- **Downstream (`L5` / `L4` Experiment Planning & Verification)**: Hands off candidate discriminating test specifications to [`labshift`](https://github.com/yaniguan/labshift) and [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5` workflow/procurement planners), which are audited by [`lab-verify`](https://github.com/yaniguan/lab-verify) and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`) before robotic execution (`stepbot`, `lab-hands`, `lab-feet`).

### Inputs, Outputs, and Scope Boundaries
- **Inputs**:
  - Scientific claims (`claims_dev.jsonl`) paired with candidate literature abstracts (`corpus.jsonl`, sentence-segmented with title and `doc_id`).
- **Outputs**:
  - Ranked sentence indices and verbatim rationale sentences per `(claim, abstract)` pair.
  - Aggregate and label-stratified (`SUPPORT` vs. `CONTRADICT`) top-$k$ rationale hit rates (`k = 1, 3`) benchmarked against a 20-seed random sentence-ranking baseline.
  - Three structured epistemic probes per claim (`why_questions`): (1) supporting-evidence query, (2) necessary-assumption query, and (3) falsifying/mind-changing experimental result query.
- **Scope Boundaries**:
  - The current prototype (`demo/why_probe.py`) evaluates **within-abstract rationale sentence retrieval** given gold evidence documents on the SciFact dev split; open-corpus retrieval across all 5,183 abstracts and handling of `NOINFO` (insufficient-evidence) claims are not yet wired into the script.
  - `why_questions` currently emits deterministic string templates parameterized by the claim text; synthesizing protocol-executable discriminating experiments conditioned on retrieved supporting and contradicting sentences remains the next milestone.
  - Zero neural weights or LLM API calls are required (`numpy` + Python standard library only).

---

## 2. Data

### Active Benchmark Dataset: SciFact (Wadden et al., EMNLP 2020)
`demo/why_probe.py` downloads the official SciFact release archive (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`) on demand into the gitignored `data/` directory (**zero redistribution** in the repository):

| File | Records | Active Usage in `demo/why_probe.py` | Schema Highlights |
|---|---:|---|---|
| `corpus.jsonl` | `5,183` abstracts | All corpus sentences (`~50k` sentences) fit BM25 `idf` and `avgdl`; gold-evidence abstracts supply candidate sentences (`mean 9.7`, `median 8` sentences/abstract) | `doc_id: int`, `title: str`, `abstract: list[str]`, `structured: bool` |
| `claims_dev.jsonl` | `300` claims | `188` claims with non-empty `evidence` yield **`209` `(claim, doc_id)` pairs** (`138` `SUPPORT`, `71` `CONTRADICT`); `112` `NOINFO` claims are filtered out (`if c["evidence"]`) | `id: int`, `claim: str`, `evidence: dict[doc_id, list[{sentences, label}]]`, `cited_doc_ids: list[int]` |
| `claims_train.jsonl` | `809` claims | Present in extracted archive; unused (zero-shot lexical retrieval requires no training split) | Same as `claims_dev.jsonl` |
| `claims_test.jsonl` | `300` claims | Present in extracted archive; unused (blind test set without gold labels) | `id: int`, `claim: str`, `cited_doc_ids: list[int]` |

- **Rationale Structure Nuance**: Among the `209` dev `(claim, doc_id)` pairs, `85` contain multiple independent minimal rationale sets (`len(sets) > 1`) and `25` individual rationale sets require multi-sentence conjunctions (`len(s["sentences"]) > 1`). `build_pairs` unions `s["sentences"]` across all sets for a `(claim, doc_id)` pair into a single `gold` index set.
- **Reference Output (`demo/output.txt`)**: Checked-in 38-line deterministic stdout capture (`2026-11-18`, Linux CPU).

### Curated Ecosystem Catalog & Licensing
`README.md` curates:
- **3 Core Benchmarks**: `SciFact` (`CC BY-NC 2.0`, **non-commercial**), `SciRIFF` (`Apache-2.0` code; per-dataset data licenses), and `LAB-Bench` (`CC BY-SA 4.0`, shared with `labbench-posttrain`).
- **9 Open-Source Co-Scientist / Debate / RAG Repositories**: `stanford-oval/storm`, `SakanaAI/AI-Scientist`, `Future-House/paper-qa`, `ur-whitelab/chemcrow-public`, `Future-House/robin`, `composable-models/llm_multiagent_debate`, `NoviScl/AI-Researcher`, `allenai/scifact`, and `gomesgroup/coscientist`.
- **13 Foundational & 2026 Papers**: Including *AI Safety via Debate* (`arXiv:1805.00899`), *Co-Scientist* (`arXiv:2502.18864`), *Benchmark Illusion* (`arXiv:2602.11898`), *When Reviews Disagree* (`arXiv:2605.10171`), *Symposium* (`arXiv:2608.19511`), and *ACED-Bench* (`arXiv:2610.14381`).
- **License Posture**: Repository code is `MIT`; `SciFact` data is explicitly flagged `CC BY-NC 2.0` (research-only, gitignored).

---

## 3. Model

`lab-ask-why` intentionally starts with a **transparent, zero-parameter lexical retrieval baseline + deterministic Socratic question generator** (`demo/why_probe.py`) to establish an interpretable floor before introducing neural rerankers or LLM debate agents:

1. **Lexical Tokenizer (`tokenize`)**:
   - Lowercases input strings, extracts alphanumeric spans via `re.findall(r"[a-z0-9]+", text.lower())`, and removes a 29-word stopword set (`STOP`, preserving domain negations only outside `not`/`no`).
2. **Corpus-Level Okapi BM25 Sentence Ranker (`BM25`)**:
   - Hyperparameters: $k_1 = 1.2$, $b = 0.75$.
   - Inverse document frequency over $N$ total sentences in `corpus.jsonl`:
     $$\text{IDF}(t) = \ln\!\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$
   - Sentence relevance score for claim $q$ and abstract sentence $s$ with average sentence length $\text{avgdl}$:
     $$\text{score}(q, s) = \sum_{t \in \text{tokenize}(q)} \mathbb{I}[t \in s] \cdot \text{IDF}(t) \cdot \frac{\text{tf}(t, s)\,(k_1 + 1)}{\text{tf}(t, s) + k_1\!\left(1 - b + b\,\frac{|s|}{\text{avgdl}}\right)}$$
   - Sentences within each gold evidence abstract are ordered via stable descending sort (`np.argsort(-scores, kind="stable")`).
3. **20-Seed Monte Carlo Random Baseline**:
   - Evaluates `np.random.default_rng(seed)` (`seed` $\in \{0, \dots, 19\}$) uniform random sentence permutations on the exact same `209` `(claim, doc_id)` pairs, reporting mean and standard deviation across seeds (`0.197 ± 0.024` at $k=1$; `0.528 ± 0.029` at $k=3$).
4. **Epistemic Question Generator (`why_questions`)**:
   - Emits three Socratic prompts per claim targeting (a) empirical evidence, (b) necessary background assumptions, and (c) falsifying/mind-changing experimental outcomes.

---

## 4. Training

- **Zero Parametric Training (Retrieval-Only Baseline)**: `BM25` requires no gradient updates, GPU compute, or labeled rationale supervision; corpus statistics (`idf`, `avgdl`) are computed in a single CPU pass over `corpus.jsonl` in `< 1` second.
- **Verified CPU Evaluation Metrics (`demo/output.txt`)**:

| Subset / Method | Pairs ($n$) | Hit@1 (`recall@1`) | Hit@3 (`recall@3`) |
|---|---:|---:|---:|
| `BM25` (all gold pairs) | `209` | **`0.493`** | **`0.789`** |
| `Random` (20-seed mean $\pm$ std) | `209` | `0.197 ± 0.024` | `0.528 ± 0.029` |
| `BM25` on `SUPPORT` pairs | `138` | `0.507` | `0.797` |
| `BM25` on `CONTRADICT` pairs | `71` | `0.465` | `0.775` |

- **Key Scientific Observation**: Pure lexical overlap retrieves a gold rationale sentence in the top 3 for `78.9%` of claim-abstract pairs (`+26.1` percentage points over random) and performs nearly symmetrically on `SUPPORT` (`0.797`) and `CONTRADICT` (`0.775`) because contradicting sentences share the same entity/process vocabulary as the claim while flipping polarity or directionality.

---

## 5. History

| Commit | Date | Author | Branch / Ref | Summary |
|---|---|---|---|---|
| `6236123` (`62361235e9d3ac87f11eca78957acccad35fb4e7`) | `2026-11-18` | Yani Guan | `main` | Initial commit (`Add curated resources and first demo`): implemented `demo/why_probe.py` (Okapi BM25 rationale sentence retriever, 20-seed random baseline, `SUPPORT`/`CONTRADICT` stratification, and Socratic `why_questions` templates on SciFact dev), recorded `demo/output.txt`, configured `.github/workflows/demo.yml` (Python 3.12 CI), and curated datasets, co-scientist/debate repositories, and 2018–2026 papers in `README.md`. |

---

## 6. Unique Properties

### Epistemic & Architectural Invariants
1. **Stance-Neutral Rationale Retrieval ("Ask Why, Don't Adjudicate")**: Rather than collapsing uncertainty into a binary `True`/`False` classification, `lab-ask-why` retrieves the verbatim sentences underlying both `SUPPORT` and `CONTRADICT` evidence and pairs them with explicit assumption and falsification probes.
2. **Polarity-Symmetric Lexical Baseline**: Demonstrated empirically on SciFact dev (`0.797` `SUPPORT` vs. `0.775` `CONTRADICT` top-3 hit rate), establishing a strong, zero-cost, CPU-only attribution baseline that any downstream neural reranker or debate agent must beat.
3. **Strict Non-Commercial Data Isolation**: Respects SciFact's `CC BY-NC 2.0` license by fetching `data.tar.gz` into gitignored `data/` at runtime while keeping all repository code `MIT`.

### Hardened Statistical & Security Invariants (Audit `#1..#22` Resolved)
- **Three-Tier Rationale Metrics (`hit@k`, `sentence_recall@k`, `rationale_set_recall@k`)**: Preserves minimal rationale-set boundaries (`rationale_sets: list[frozenset[int]]`) alongside `gold` union, reporting binary `Hit@k` (`0.809`), fractional IR `Sentence-Recall@k` (`0.656`), and strict SciFact complete-rationale `Set-Recall@k` (`0.770`) alongside `lead-k` (`0.263`), `conclusion-first` (`0.679`), and 20-seed random (`0.528`) baselines.
- **Background-Only IDF, `NOINFO` Abstention Calibration & Open-Corpus Two-Stage Retrieval**: Fits `BM25` on non-evaluation background documents (`exclude_doc_ids`) with smooth Laplace OOV IDF, evaluates all `112` `NOINFO` claims (`AUROC = 0.767`), and scales via an inverted posting index to two-stage open-corpus retrieval across all `5,183` abstracts (`doc_recall@5 = 0.885`, `sentence_hit@3 = 0.559`).
- **Biomedical Tokenization & Grounded Socratic Probes**: Normalizes Greek letters (`α->alpha`, `β->beta`, `μ->micro`), allele signs (`alpha+`), and decimal/comma numbers (`1,000->1000`, `1.1->1.1`) while retaining negations (`not`, `no`, `neither`, `without`), and grounds `why_questions` in differential rationale mechanisms.
- **Archive & Transport Hardening + Typed `L6` Contracts**: Enforces timeout-bounded atomic `.part` downloads, SHA-256 verification, PEP 706 `filter="data"` + path-traversal/symlink/AppleDouble validation, `EpistemicProbeResult` JSON Schema export, and evaluation-only `CANARY_NEEDLES`-guarded `LAB-Bench` (`ProtocolQA`/`LitQA2`) counterfactual scoring.

### Sibling-Repository Integration Contracts
- **[`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`)**: Supplies speech-stripped acoustic telemetry and parsed wet-lab protocol spans (`parsed_spans`) upstream to `lab-ask-why`.
- **[`lab-decide`](https://github.com/yaniguan/lab-decide) (`L6`)**: Consumes discriminating-experiment candidates (`epistemic_probe`) from `lab-ask-why` and scores whether proposed operating points lie inside the calibrated conformal trust region or require active exploration.
- **[`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`)**: Performs post-execution 4-way causal blame attribution, NNLS precursor ghost-peak deconvolution, and SHA-256 certified ledger updates (`certified_ledger_entries`) feeding back into `L6` epistemic memory and conformal recalibration.
- **[`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3`)**: Supplies `LitQA2` / `SuppQA` / `ProtocolQA` evaluation harnesses (`labeval/`) and calibrated abstention/counterfactual reward templates (`CalibratedMCQTemplate`, `CounterfactualPairScorer`) to post-train neural `why`-probing policies.
- **[`lab-verify`](https://github.com/yaniguan/lab-verify) & [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`)**: Audit structured experimental assumptions and dry-run discriminating protocols before wet-lab dispatch.
- **[`YLM`](https://github.com/yaniguan/YLM) (`L0`) & [`labshift`](https://github.com/yaniguan/labshift) (`L5`)**: Connect literature-grounded Socratic debate (`L6`) to formal `LabLean` specifications and closed-loop campaign scheduling (`L5`).

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ask-why"`, `LAYER = "L6"`) implements `process_packet(packet: dict) -> dict` for the end-to-end 15-repository Auto-Lab wire protocol (`schema_version = "autolab-wire/v1"`), resolving all repository paths relative to `Path(__file__).resolve().parent` without network I/O:

1. **Envelope & Short-Circuit Semantics**: Deep-copies the incoming packet, initializes `schema_version`, `halted`, `halt_reason`, `trace`, and `bottom_reasoning_traces`, and returns immediately unchanged whenever `packet["halted"]` is `True`.
2. **Upstream Grounding (`lab-ears` -> `lab-ask-why`)**: Reads `goal_text`, protocol `parsed_spans` from [`lab-ears`](https://github.com/yaniguan/lab-ears) (`L2`), and optional `evidence_sentences` (defaulting to a deterministic 6-sentence wet-lab solubility and protocol support/contradict mini-corpus). Ranks supporting and contradicting sentences via `lab_ask_why.bm25.BM25` and `classify_rationale_polarity`, then generates grounded Socratic probes via `lab_ask_why.socratic.why_questions`.
3. **Formal Guardrail (`contested_claim_requires_experiment`)**: Mirrors `lean/LabAskWhyLean.lean` (`contested_claim_requires_experiment`). Whenever both supporting and contradicting rationales are retrieved while `suppress_discriminating_experiment` is `True`, `process_packet` sets `halted = True` and `halt_reason = "L6_CONTESTED_WITHOUT_EXPERIMENT"`, recording `"BLOCK"` in `trace` and first-principles falsifiable control design in `bottom_reasoning_traces`.
4. **Downstream Handoff & Closed-Loop Feedback (`lab-decide` & `lab-attrib`)**: Writes `epistemic_probe` (`claim`, `epistemic_status`, `retrieval_confidence`, `abstain_recommended`, `support_top`, `contradict_top`, `why_questions`, `discriminating_question`, `has_discriminating_experiment`, `discriminating_observable`, `control_wells`, `attribution_feedback`) for [`lab-decide`](https://github.com/yaniguan/lab-decide) (`L6`), cascading through `buy-or-make`, `lab-automation-atlas`, `labshift` (`L5`), `lab-rehearsal`, `lab-verify` (`L4`), `stepbot`, `lab-hands` (`L1`), and `lab-feet` (`L0`), with post-execution readout identifiability (`CERTIFIED_POSITIVE`, `CERTIFIED_NEGATIVE`, `AMBIGUOUS_CHARACTERIZATION`, `EXECUTION_ARTIFACT`) and SHA-256 certified ledger feedback returned via [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`) and `twinstep`, `labfill`, and `labbench-posttrain` (`L3`).
