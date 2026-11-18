# lab-ask-why

A co-scientist that does not pick a winner. It asks *"why do you think so?"*, finds the evidence on each side, and names the result that would change minds.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the co-scientist proposes the next simulation or experiment. Here it is a partner in a disagreement. It surfaces the evidence and assumptions behind each claim, then suggests the test that tells them apart. The first attempt finds the sentences behind a claim in real abstracts.

## First attempt

SciFact dev set (claims checked against paper abstracts; gold rationale sentences marked). For each claim and its evidence abstract, BM25 (plain Python and numpy) ranks the abstract's sentences against the claim. A hit means a gold rationale sentence is in the top k. Random sentence ranking (20 seeds) is the baseline. The script also prints three template "why" questions per claim.

```bash
pip install numpy
python demo/why_probe.py
```

Output (Linux, CPU only, 2026-11-18):

```
SciFact dev: rationale sentence retrieval inside gold evidence abstracts
claims with evidence: 188   claim-doc pairs: 209   labels: {'SUPPORT': 138, 'CONTRADICT': 71}
sentences per abstract: mean 9.7, median 8

method                  recall@1  recall@3
BM25                       0.493     0.789
random (20 seeds)          0.197     0.528
  random std               0.024     0.029
BM25 SUPPORT               0.507     0.797   (n=138)
BM25 CONTRADICT            0.465     0.775   (n=71)

======================================================================
Claim: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.
Abstract: Rare Variants Create Synthetic Genome-Wide Associations
Gold label: SUPPORT   gold rationale sentences: [2, 5, 7]
Top-3 BM25 sentences:
  1.*[s7] In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.
  2. [s0] Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.
  3.*[s2] We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.
Why questions:
  - What evidence makes you think that 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants?
  - What are you assuming must be true for '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants' to hold?
  - What result would change your mind about '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants'?

======================================================================
Claim: A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.
Abstract: Increased Microerythrocyte Count in Homozygous α+-Thalassaemia Contributes to Protection against Severe Malarial Anaemia
Gold label: CONTRADICT   gold rationale sentences: [1, 9, 10]
Top-3 BM25 sentences:
  1.*[s1] Individuals homozygous for alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count.
  2.*[s10] CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for alpha(+)-thalassaemia may contribute substantially to their protection against SMA.
  3. [s7] This model predicted that children homozygous for alpha(+)-thalassaemia lose less Hb than children of normal genotype for a reduction in erythrocyte count of >1.1 x 10(12)/l as a result of the reduced mean cell Hb in homozygous alpha(+)-thalassaemia.
Why questions:
  - What evidence makes you think that a high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects?
  - What are you assuming must be true for 'A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects' to hold?
  - What result would change your mind about 'A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects'?

(* = gold rationale sentence)
```

Plain BM25 finds a supporting or contradicting reason in the top 3 for 0.79 of claim-abstract pairs, against 0.53 for random ranking (abstracts are short, about 10 sentences). It works about equally for SUPPORT and CONTRADICT. Next: retrieve across many papers (not only the gold abstract) and turn retrieved reasons into a proposed discriminating experiment.

## Datasets

| Dataset | What | License |
|---|---|---|
| [SciFact](https://github.com/allenai/scifact) | 1.4k expert claims with rationale sentences | CC BY-NC 2.0 (**non-commercial**) |
| [SciRIFF](https://github.com/allenai/SciRIFF) | Instruction data for reading scientific papers | Apache-2.0 (code); data see repo |
| [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) | Biology research tasks (LitQA, ProtocolQA) | CC BY-SA 4.0 |

Licenses marked **non-commercial** are fine for research but not for products.

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
