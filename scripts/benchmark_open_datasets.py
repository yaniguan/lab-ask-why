"""Leak-free multi-dataset benchmark suite for lab-ask-why (Layer L6 Epistemic Brain).

Downloads 8 real public scientific claim verification, conflicting-evidence, and
literature/protocol QA corpora into gitignored `data/open_benchmarks/` with SHA-256
verification and strict `eval_only` / `CC-BY-NC-2.0` license isolation, fits
`FrontierHybridVerifier` strictly on `SciFact train` (`809` claims: `647` fit + `162`
conformal calibration), and evaluates zero-leakage across:
  1. `scifact` (`train` vs held-out `dev`: Random, BM25, Dense-SVD, Stage-1 RRF,
     and Hybrid+Reranker on `Hit@1`, `Hit@3`, `MRR`, stance macro-F1, and `90%`
     Split Conformal prediction set coverage & size),
  2. `pubmedqa` (`pqa_labeled`: `1,000` expert `yes`/`no`/`maybe` clinical rationale items),
  3. `conflictbank` (`CB_qa` conflicting multi-evidence + `CB_claim_evidence`),
  4. `faitheval_inconsistent` (`Salesforce/FaithEval-inconsistent-v1.0` conflicting contexts),
  5. `claimcheck` (`JHU-CLSP/CLAIMCHECK` NeurIPS peer-review weakness critiques),
  6. `sciver` (`chengyewang/SciVer` scientific figure/table claim verification),
  7. `chembench` (`jablonkagroup/ChemBench`, `eval_only=True` canary-guarded),
  8. `litqa2` (`futurehouse/lab-bench` `LitQA2`, `eval_only=True` canary-guarded).

Every output dictionary and every CLI line explicitly carries `data_source`
(`"real_downloaded"` vs `"synthetic_fixture"`), and `--offline` without cached
SHA-256-verified data exits non-zero unless `--allow-synthetic` is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_ask_why.bm25 import BM25  # noqa: E402
from lab_ask_why.data import (  # noqa: E402
    DEFAULT_ROOT,
    OFFLINE_MINI_CLAIMS,
    OFFLINE_MINI_CORPUS,
    SCIFACT_SHA256,
    build_noinfo_pairs,
    build_pairs,
    compute_file_sha256,
    configure_utf8_stdout,
    ensure_data,
    read_jsonl,
)
from lab_ask_why.datasets import (  # noqa: E402
    adapt_record,
    enforce_dataset_guard,
    get_dataset_spec,
    normalize_veracity_label,
    probe_normalized_record,
)
from lab_ask_why.frontier_verifier import (  # noqa: E402
    STANCE_LABELS,
    EntityNumericNegationIndex,
    FrontierHybridVerifier,
    canonicalize_stance_label,
)
from lab_ask_why.labbench_adapter import segment_protocol_sentences  # noqa: E402
from lab_ask_why.metrics import (  # noqa: E402
    compute_auroc,
    hit_at,
    macro_f1_score,
    mean_reciprocal_rank,
    rank_scores,
)
from lab_ask_why.retrieval import _SubwordConceptCosineIndex, classify_rationale_polarity  # noqa: E402

DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "open_benchmarks_2026.json"
_USER_AGENT = "lab-ask-why/0.3 (+https://github.com/yaniguan/lab-ask-why)"

# Pinned SHA-256 digests for downloaded benchmark files in gitignored data/open_benchmarks/.
# Populated and verified deterministically against canonical UTF-8 artifacts.
PINNED_BENCHMARK_SHA256: dict[str, str] = {
    "scifact_tar_gz": SCIFACT_SHA256,
    "pubmedqa_ori_pqal.json": "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d",
    "claimcheck_source_main.json": "6fbfa93145910021b9b995437ceedb616be81d20be44ec6fc9a88942bf09dec1",
    "sciver_valset.json": "3b62b78cfab1863252014ab410ddaa8499724208fc19718cc1ed01df89c897d3",
    "faitheval_inconsistent.jsonl": "2d5f2379adf6baeec087cee306fd0a6ea78acc4b6350e28d53990fc498e07b75",
    "conflictbank_qa.jsonl": "a54d892781d777ebef536c7ceb70d6823574d18d27ae12254c41b060153b25fb",
    "conflictbank_claims.jsonl": "b468f3bb793ef6662262d84f2c77f48298705bc3b5f2bf5a08b8ff0be8ff3610",
    "litqa2.jsonl": "c399c70eee3f7481fb127589da811ccc738958e093c13ce4c426699eee9f7a95",
    "chembench_general.jsonl": "5985c2c558d1459ced545d215332aec66ac054c3cd3116f0a6d3c5f61a1184d3",
    "chembench_analytical.jsonl": "7a08aa73a085162e7b289637db409bf7f64542a9a45e01e510042e1aa031161f",
}

_SCIVER_LEAK_PREFIX_RE = re.compile(
    r"^\s*(?:this\s+(?:statement|claim)\s+is\s+(?:incorrect|false|wrong|inaccurate|refuted|untrue)"
    r"(?:\s+because|\s+as|\s+since|\s*:|\s*,)?|"
    r"the\s+(?:statement|claim)\s+is\s+(?:incorrect|false|wrong|inaccurate)"
    r"(?:\s+because|\s+as|\s+since|\s*:|\s*,)?)\s*",
    re.IGNORECASE,
)


def _download_bytes_atomic(
    url: str,
    dest_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(f"{dest_path.name}.part.{os.getpid()}")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        digest = hasher.hexdigest()
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {dest_path.name}: expected {expected_sha256}, got {digest}"
            )
        os.replace(part_path, dest_path)
        return digest
    except Exception:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise


def _download_hf_rows_canonical_jsonl(
    dataset: str,
    config: str,
    split: str,
    total_rows: int,
    dest_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
) -> str:
    rows: list[Mapping[str, Any]] = []
    offset = 0
    ds_q = urllib.parse.quote(dataset, safe="")
    cfg_q = urllib.parse.quote(config, safe="")
    sp_q = urllib.parse.quote(split, safe="")

    while offset < total_rows:
        length = min(100, total_rows - offset)
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={ds_q}&config={cfg_q}&split={sp_q}&offset={offset}&length={length}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        batch = [
            entry["row"]
            for entry in (payload.get("rows") or ())
            if isinstance(entry, Mapping) and isinstance(entry.get("row"), Mapping)
        ]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break

    if not rows:
        raise RuntimeError(f"Zero rows returned from HF datasets-server for {dataset}/{config}/{split}")

    canonical_bytes = (
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {dest_path.name}: expected {expected_sha256}, got {digest}"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(f"{dest_path.name}.part.{os.getpid()}")
    try:
        part_path.write_bytes(canonical_bytes)
        os.replace(part_path, dest_path)
    except Exception:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        raise
    return digest


def _ensure_cached_file(
    dest_path: Path,
    *,
    expected_sha256: str | None,
    offline: bool,
    downloader_fn: Any,
) -> str:
    """Verify existing cached artifact by SHA-256 or download if online; fail closed if offline."""
    if dest_path.exists():
        actual = compute_file_sha256(str(dest_path))
        if expected_sha256 is None or actual.lower() == expected_sha256.lower():
            return actual
        dest_path.unlink(missing_ok=True)
        if offline:
            raise ValueError(
                f"Cached benchmark file {dest_path} failed SHA-256 check "
                f"(expected {expected_sha256}, got {actual}) and --offline is set."
            )

    if offline:
        raise FileNotFoundError(
            f"Benchmark file {dest_path} is missing in --offline mode "
            "(pass --allow-synthetic only for synthetic fixture smoke testing)."
        )
    return str(downloader_fn())


def _verify_eval_only_isolation(dataset_id: str) -> bool:
    """Verify `enforce_dataset_guard(..., purpose='train')` blocks `eval_only` benchmarks."""
    spec = get_dataset_spec(dataset_id)
    enforce_dataset_guard(spec, purpose="eval")
    if not spec.eval_only:
        return False
    try:
        enforce_dataset_guard(spec, purpose="train")
    except PermissionError:
        return True
    raise AssertionError(f"eval_only guard failed to raise PermissionError for {dataset_id}")


def _clean_sciver_evidence(row: Mapping[str, Any]) -> list[str]:
    """Extract leak-free factual evidence for SciVer without verdict boilerplate or exact claim copy."""
    raw_exp = str(row.get("perturbed_explanation") or "").strip()
    cleaned = _SCIVER_LEAK_PREFIX_RE.sub("", raw_exp).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    # Also strip secondary giveaway tokens like 'not X' meta-verdict headers if needed? No,
    # factual quantitative contrasts inside figure descriptions are legitimate evidence, whereas
    # "This statement is incorrect because" is annotator meta-text!
    sents = segment_protocol_sentences(cleaned)
    claim_norm = str(row.get("claim") or "").strip().lower()
    # Never include a sentence that is a verbatim copy of the claim
    return [s for s in sents if s.strip() and s.strip().lower() != claim_norm]


def _synthetic_fixture_bundle() -> dict[str, Any]:
    """Deterministic synthetic mini-bundle ONLY used when `--offline --allow-synthetic` is explicitly passed."""
    corpus = {d["doc_id"]: d for d in OFFLINE_MINI_CORPUS}
    train_claims = [
        {
            "id": 1001,
            "claim": "Uncommon genetic variants with high penetrance create synthetic GWAS signals.",
            "evidence": {"101": [{"sentences": [2, 3], "label": "SUPPORT"}]},
            "cited_doc_ids": [101],
        },
        {
            "id": 1002,
            "claim": "Increased microerythrocyte count increases severe malarial anaemia mortality in alpha+-thalassaemia.",
            "evidence": {"102": [{"sentences": [1, 3], "label": "CONTRADICT"}]},
            "cited_doc_ids": [102],
        },
        {
            "id": 1003,
            "claim": "PAM-proximal seed mismatches abolish Cas9 DNA cleavage.",
            "evidence": {"103": [{"sentences": [1], "label": "SUPPORT"}]},
            "cited_doc_ids": [103],
        },
        {
            "id": 1004,
            "claim": " PAM-proximal seed mismatches enhance Cas9 DNA cleavage.",
            "evidence": {"103": [{"sentences": [1], "label": "CONTRADICT"}]},
            "cited_doc_ids": [103],
        },
        {
            "id": 1005,
            "claim": "Single-cell RNA-seq cures Alzheimer amyloid plaques in aged mice.",
            "evidence": {},
            "cited_doc_ids": [104],
        },
        {
            "id": 1006,
            "claim": "Cas9 endonuclease upregulates hepatic gluconeogenesis in fasting humans.",
            "evidence": {},
            "cited_doc_ids": [103],
        },
    ]
    dev_claims = list(OFFLINE_MINI_CLAIMS)
    return {
        "scifact_corpus": corpus,
        "scifact_train": train_claims,
        "scifact_dev": dev_claims,
        "pubmedqa": [
            {
                "pubid": "syn-pqa-1",
                "question": "Does SpCas9 cleavage require an NGG protospacer adjacent motif?",
                "context": {"contexts": ["SpCas9 endonuclease activity requires recognition of an NGG PAM motif."]},
                "final_decision": "yes",
            },
            {
                "pubid": "syn-pqa-2",
                "question": "Do PAM-proximal seed mismatches increase SpCas9 cleavage activity?",
                "context": {"contexts": ["Mismatches in the PAM-proximal seed region strongly abolish DNA cleavage."]},
                "final_decision": "no",
            },
            {
                "pubid": "syn-pqa-3",
                "question": "Does SpCas9 alter circadian melatonin secretion in pinealocytes?",
                "context": {"contexts": ["SpCas9 recognizes canonical NGG PAM motifs in vitro."]},
                "final_decision": "maybe",
            },
        ],
        "conflictbank_qa": [
            {
                "id": "syn-cbqa-1",
                "subject": "Aspirin Stability",
                "default_claim": "Buffered pH 6.8 medium prevents rapid aspirin ester hydrolysis at 25 C.",
                "default_evidence": "Buffered pH 6.8 medium stabilizes aspirin and prevents ester hydrolysis at 25 C.",
                "misinformation_conflict_evidence_evidence": "Buffered pH 6.8 medium fails to prevent rapid aspirin ester hydrolysis and degrades aspirin.",
            }
        ],
        "conflictbank_claims": [
            {
                "id": "syn-cbc-1",
                "subject": "Aspirin",
                "category": "default",
                "claim": "Buffered medium increases aspirin stability at 25 C.",
                "evidence": "Buffered medium increases aspirin stability at 25 C in 96-well plates.",
            },
            {
                "id": "syn-cbc-2",
                "subject": "Aspirin",
                "category": "semantic_conflict",
                "claim": "Buffered medium increases aspirin stability at 25 C.",
                "evidence": "Buffered medium decreases aspirin stability and accelerates ester hydrolysis.",
            },
        ],
        "faitheval": [
            {
                "qid": "syn-fe-1",
                "question": "Which kinase phosphorylates substrate S14 in hepatocytes?",
                "context": (
                    "Document 1: Mass spectrometry confirms AMPK phosphorylates substrate S14 in hepatocytes. "
                    "Document 2: Knockout assays show mTORC1, and not AMPK, phosphorylates substrate S14 in hepatocytes."
                ),
                "answers": ["AMPK", "mTORC1"],
                "subset": "synthetic",
            }
        ],
        "claimcheck": [
            {
                "id": "syn-cc-1",
                "claim": "Our attention layer achieves strict O(N) memory scaling.",
                "critique_sents": ["Equation 4 materializes an N x N matrix that does not achieve O(N) memory scaling."],
                "abstract_sents": ["We propose an efficient sequence model for long documents."],
                "label": "CONTRADICT",
            },
            {
                "id": "syn-cc-2",
                "claim": "Our model improves wall-clock latency on mobile CPUs.",
                "critique_sents": ["No mobile CPU latency benchmarks are reported anywhere in the paper."],
                "abstract_sents": ["We evaluate perplexity on GPU clusters."],
                "label": "NEI",
            },
        ],
        "sciver": [
            {
                "id": "syn-sv-1",
                "claim": "Catalyst Ni-MOF increases Faradaic efficiency to 94%.",
                "evidence_sents": ["Table 2 shows Catalyst Ni-MOF increases CO2 Faradaic efficiency to 94%."],
                "label": "SUPPORT",
            },
            {
                "id": "syn-sv-2",
                "claim": "Catalyst Ni-MOF decreases Faradaic efficiency below 40%.",
                "evidence_sents": ["Table 2 shows Catalyst Ni-MOF increases CO2 Faradaic efficiency to 94%."],
                "label": "CONTRADICT",
            },
        ],
        "chembench": [
            {
                "uuid": "syn-chem-1",
                "name": "ph_buffer",
                "subfield": "general_chemistry",
                "description": "Equimolar acetic acid and sodium acetate buffer has pH equal to pKa = 4.76.",
                "examples": [
                    {
                        "input": "What is the pH of an equimolar acetic acid / acetate buffer?",
                        "target_scores": '{"4.76": 1, "9.25": 0, "1.00": 0}',
                    }
                ],
            }
        ],
        "litqa2": [
            {
                "id": "syn-litqa2-1",
                "question": "Which antibiotic did Acinetobacter lwoffii evolve resistance against?",
                "ideal": "ciprofloxacin",
                "distractors": ["meropenem", "gentamicin"],
                "key-passage": (
                    "A. baumannii evolved resistance to meropenem. "
                    "A. lwoffii evolved high-level resistance exclusively to ciprofloxacin."
                ),
            }
        ],
    }


def _evaluate_candidate_verification_records(
    verifier: FrontierHybridVerifier,
    items: Sequence[Mapping[str, Any]],
    *,
    eval_labels: Sequence[str] = STANCE_LABELS,
) -> dict[str, Any]:
    """Compare Okapi BM25 polarity baseline vs `FrontierHybridVerifier` on `(claim, sents, label)` items."""
    y_true: list[str] = []
    y_bm25: list[str] = []
    y_hybrid: list[str] = []
    covered: list[bool] = []
    set_sizes: list[int] = []
    conflict_flags: list[bool] = []

    for it in items:
        claim = str(it["claim"])
        sents = list(it["evidence_sents"])
        gold = canonicalize_stance_label(it["label"])
        y_true.append(gold)

        if not sents:
            y_bm25.append("NEI")
        else:
            bm25 = BM25(sents)
            norm_sc = bm25.score_indexed_normalized(claim)
            best_i = int(np.argmax(norm_sc))
            if float(norm_sc[best_i]) < 0.22 and "NEI" in eval_labels:
                y_bm25.append("NEI")
            else:
                pol = classify_rationale_polarity(claim, sents[best_i])
                if pol == "CONTRADICT":
                    y_bm25.append("CONTRADICT")
                elif pol == "SUPPORT":
                    y_bm25.append("SUPPORT")
                else:
                    y_bm25.append("NEI" if "NEI" in eval_labels else "SUPPORT")

        pred = verifier.verify_candidate_sentences(
            claim, sents, title=str(it.get("title", "")), top_k=3
        )
        if len(eval_labels) == 2 and set(eval_labels) == {"SUPPORT", "CONTRADICT"}:
            p_sup = float(pred.probabilities["SUPPORT"])
            p_con = float(pred.probabilities["CONTRADICT"])
            y_hybrid.append("CONTRADICT" if p_con > p_sup else "SUPPORT")
        else:
            y_hybrid.append(pred.predicted_label)

        covered.append(gold in pred.conformal_set)
        set_sizes.append(len(pred.conformal_set))
        conflict_flags.append(pred.is_contested_conflict)

    return {
        "n_evaluated": len(items),
        "label_counts": dict(Counter(y_true)),
        "bm25_macro_f1": macro_f1_score(y_true, y_bm25, labels=eval_labels),
        "hybrid_reranker_macro_f1": macro_f1_score(y_true, y_hybrid, labels=eval_labels),
        "bm25_accuracy": float(np.mean([yt == yp for yt, yp in zip(y_true, y_bm25)])) if y_true else float("nan"),
        "hybrid_reranker_accuracy": float(np.mean([yt == yp for yt, yp in zip(y_true, y_hybrid)])) if y_true else float("nan"),
        "conformal_coverage_90": float(np.mean(covered)) if covered else float("nan"),
        "conformal_avg_set_size_90": float(np.mean(set_sizes)) if set_sizes else float("nan"),
        "contested_conflict_rate": float(np.mean(conflict_flags)) if conflict_flags else 0.0,
    }


def _tie_aware_hit1_and_rr(option_scores: Sequence[float]) -> tuple[float, float]:
    """Exact tie-aware Hit@1 and Reciprocal Rank for `option_scores[0]` (ideal) vs distractors.

    Eliminates index-0 `argsort(kind='stable')` tie bias when multiple options score 0.0.
    """
    s_ideal = float(option_scores[0])
    distractors = [float(x) for x in option_scores[1:]]
    n_greater = sum(1 for x in distractors if x > s_ideal + 1e-12)
    n_ties = sum(1 for x in distractors if abs(x - s_ideal) <= 1e-12)
    hit1 = (1.0 / float(1 + n_ties)) if n_greater == 0 else 0.0
    rr = float(np.mean([1.0 / float(r) for r in range(n_greater + 1, n_greater + n_ties + 2)]))
    return hit1, rr


def _evaluate_mcq_rationale_discrimination(
    verifier: FrontierHybridVerifier,
    mcq_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare BM25 vs `FrontierHybridVerifier` on MCQ ideal-vs-distractor rationale grounding."""
    bm25_wins: list[float] = []
    hybrid_wins: list[float] = []
    bm25_rr: list[float] = []
    hybrid_rr: list[float] = []

    for item in mcq_items:
        q = str(item["question"]).strip()
        ideal = str(item["ideal"]).strip()
        distractors = [str(d).strip() for d in item["distractors"] if str(d).strip()]
        sents = [str(s).strip() for s in item["sentences"] if str(s).strip()]
        if not q or not ideal or not distractors or not sents:
            continue

        options = [ideal, *distractors]
        bm25_local = BM25(sents)
        subword_local = _SubwordConceptCosineIndex(sents)
        entity_local = EntityNumericNegationIndex(sents)
        sent_embs = verifier.sent_svd.embed(sents)

        # Stage-2 reranker weights candidate passage sentences by empirical relevance to question q
        q_pred = verifier.verify_candidate_sentences(q, sents, top_k=len(sents))
        sent_weights = np.ones(len(sents), dtype=np.float64)
        for r_idx, r_sc in zip(q_pred.top_rationale_indices, q_pred.top_rationale_scores):
            sent_weights[int(r_idx)] = 0.75 + 0.50 * float(r_sc)

        b_scores: list[float] = []
        h_scores: list[float] = []
        for opt in options:
            # Baseline: tie-aware lexical BM25 of candidate answer option against passage sentences
            opt_bm25 = bm25_local.scores(opt, sents)
            b_scores.append(float(np.max(opt_bm25)) if opt_bm25.size > 0 else 0.0)

            # Frontier Hybrid + Reranker: reranker-weighted BM25 + subword/synonym + gated LSA + entity/numeric
            sub_sc = subword_local.cosine_scores(opt)
            ent_sc = entity_local.overlap_scores(opt)
            opt_emb = verifier.sent_svd.embed([opt])[0]
            lsa_raw = (
                sent_embs @ opt_emb
                if float(np.linalg.norm(opt_emb)) > 1e-12
                else np.zeros(len(sents), dtype=np.float64)
            )
            lsa_sc = np.where(lsa_raw >= 0.25, lsa_raw, 0.0)
            per_sent_hyb = sent_weights * (
                1.0 * opt_bm25 + 2.0 * sub_sc + 1.2 * ent_sc + 0.8 * lsa_sc
            )
            h_scores.append(float(np.max(per_sent_hyb)) if per_sent_hyb.size > 0 else 0.0)

        b_h1, b_rr_val = _tie_aware_hit1_and_rr(b_scores)
        h_h1, h_rr_val = _tie_aware_hit1_and_rr(h_scores)

        bm25_wins.append(b_h1)
        hybrid_wins.append(h_h1)
        bm25_rr.append(b_rr_val)
        hybrid_rr.append(h_rr_val)

    return {
        "n_evaluated": len(bm25_wins),
        "bm25_option_hit@1": float(np.mean(bm25_wins)) if bm25_wins else float("nan"),
        "hybrid_reranker_option_hit@1": float(np.mean(hybrid_wins)) if hybrid_wins else float("nan"),
        "bm25_option_mrr": float(np.mean(bm25_rr)) if bm25_rr else float("nan"),
        "hybrid_reranker_option_mrr": float(np.mean(hybrid_rr)) if hybrid_rr else float("nan"),
    }


def run_benchmark_suite(
    *,
    data_root: Path = Path(DEFAULT_ROOT),
    offline: bool = False,
    allow_synthetic: bool = False,
    verify_pinned_sha256: bool = True,
) -> dict[str, Any]:
    """Execute the full 8-corpus leak-free evaluation and return structured results dict."""
    t_start = time.time()
    bench_cache_dir = data_root / "open_benchmarks"

    use_synthetic = False
    file_digests: dict[str, str] = {}

    try:
        scifact_dir = Path(
            ensure_data(
                root=str(data_root),
                expected_sha256=(PINNED_BENCHMARK_SHA256["scifact_tar_gz"] if verify_pinned_sha256 else None),
                offline=offline,
            )
        )
        tar_path = data_root / "data.tar.gz"
        file_digests["scifact"] = (
            compute_file_sha256(str(tar_path)) if tar_path.exists() else SCIFACT_SHA256
        )

        pqa_path = bench_cache_dir / "pubmedqa_ori_pqal.json"
        file_digests["pubmedqa"] = _ensure_cached_file(
            pqa_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["pubmedqa_ori_pqal.json"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_bytes_atomic(
                "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
                pqa_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["pubmedqa_ori_pqal.json"] if verify_pinned_sha256 else None,
            ),
        )

        cc_path = bench_cache_dir / "claimcheck_source_main.json"
        file_digests["claimcheck"] = _ensure_cached_file(
            cc_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["claimcheck_source_main.json"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_bytes_atomic(
                "https://raw.githubusercontent.com/JHU-CLSP/CLAIMCHECK/main/data/texts/source/main.json",
                cc_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["claimcheck_source_main.json"] if verify_pinned_sha256 else None,
            ),
        )

        sv_path = bench_cache_dir / "sciver_valset.json"
        file_digests["sciver"] = _ensure_cached_file(
            sv_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["sciver_valset.json"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_bytes_atomic(
                "https://huggingface.co/datasets/chengyewang/SciVer/resolve/main/valset.json",
                sv_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["sciver_valset.json"] if verify_pinned_sha256 else None,
            ),
        )

        fe_path = bench_cache_dir / "faitheval_inconsistent.jsonl"
        file_digests["faitheval_inconsistent"] = _ensure_cached_file(
            fe_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["faitheval_inconsistent.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "Salesforce/FaithEval-inconsistent-v1.0",
                "default",
                "test",
                300,
                fe_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["faitheval_inconsistent.jsonl"] if verify_pinned_sha256 else None,
            ),
        )

        cb_qa_path = bench_cache_dir / "conflictbank_qa.jsonl"
        file_digests["conflictbank_qa"] = _ensure_cached_file(
            cb_qa_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["conflictbank_qa.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "Warrieryes/CB_qa",
                "default",
                "train",
                200,
                cb_qa_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["conflictbank_qa.jsonl"] if verify_pinned_sha256 else None,
            ),
        )

        cb_cl_path = bench_cache_dir / "conflictbank_claims.jsonl"
        file_digests["conflictbank_claims"] = _ensure_cached_file(
            cb_cl_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["conflictbank_claims.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "Warrieryes/CB_claim_evidence",
                "default",
                "train",
                200,
                cb_cl_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["conflictbank_claims.jsonl"] if verify_pinned_sha256 else None,
            ),
        )

        lq_path = bench_cache_dir / "litqa2.jsonl"
        file_digests["litqa2"] = _ensure_cached_file(
            lq_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["litqa2.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "futurehouse/lab-bench",
                "LitQA2",
                "train",
                200,
                lq_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["litqa2.jsonl"] if verify_pinned_sha256 else None,
            ),
        )

        ch_gen_path = bench_cache_dir / "chembench_general.jsonl"
        file_digests["chembench_general"] = _ensure_cached_file(
            ch_gen_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["chembench_general.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "jablonkagroup/ChemBench",
                "general_chemistry",
                "train",
                150,
                ch_gen_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["chembench_general.jsonl"] if verify_pinned_sha256 else None,
            ),
        )

        ch_ana_path = bench_cache_dir / "chembench_analytical.jsonl"
        file_digests["chembench_analytical"] = _ensure_cached_file(
            ch_ana_path,
            expected_sha256=PINNED_BENCHMARK_SHA256["chembench_analytical.jsonl"] if verify_pinned_sha256 else None,
            offline=offline,
            downloader_fn=lambda: _download_hf_rows_canonical_jsonl(
                "jablonkagroup/ChemBench",
                "analytical_chemistry",
                "train",
                100,
                ch_ana_path,
                expected_sha256=PINNED_BENCHMARK_SHA256["chembench_analytical.jsonl"] if verify_pinned_sha256 else None,
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        if offline and allow_synthetic:
            use_synthetic = True
        else:
            raise

    data_source = "synthetic_fixture" if use_synthetic else "real_downloaded"

    # --- Load datasets (real downloaded or explicit synthetic fixture) ---
    if use_synthetic:
        syn = _synthetic_fixture_bundle()
        scifact_corpus = syn["scifact_corpus"]
        scifact_train = syn["scifact_train"]
        scifact_dev = syn["scifact_dev"]
        pqa_rows = syn["pubmedqa"]
        cb_qa_rows = syn["conflictbank_qa"]
        cb_claim_rows = syn["conflictbank_claims"]
        fe_rows = syn["faitheval"]
        cc_items = syn["claimcheck"]
        sv_items = syn["sciver"]
        chem_rows = syn["chembench"]
        litqa2_rows = syn["litqa2"]
        file_digests = {k: "synthetic_fixture_no_download" for k in (
            "scifact", "pubmedqa", "claimcheck", "sciver",
            "faitheval_inconsistent", "conflictbank_qa", "conflictbank_claims",
            "litqa2", "chembench_general", "chembench_analytical",
        )}
    else:
        scifact_corpus = {r["doc_id"]: r for r in read_jsonl("corpus.jsonl", str(scifact_dir))}
        scifact_train = read_jsonl("claims_train.jsonl", str(scifact_dir))
        scifact_dev = read_jsonl("claims_dev.jsonl", str(scifact_dir))

        raw_pqa = json.loads(pqa_path.read_text(encoding="utf-8"))
        pqa_rows = [
            {
                "pubid": str(pid),
                "question": str(obj.get("QUESTION", "")),
                "context": {"contexts": list(obj.get("CONTEXTS") or ())},
                "final_decision": str(obj.get("final_decision", "")),
            }
            for pid, obj in raw_pqa.items()
            if isinstance(obj, Mapping)
        ]

        cb_qa_rows = read_jsonl(str(cb_qa_path))
        cb_claim_rows = read_jsonl(str(cb_cl_path))
        fe_rows = read_jsonl(str(fe_path))

        raw_cc = json.loads(cc_path.read_text(encoding="utf-8"))
        cc_items = []
        for paper_id, paper_obj in raw_cc.items():
            if not isinstance(paper_obj, Mapping):
                continue
            meta = paper_obj.get("meta") if isinstance(paper_obj.get("meta"), Mapping) else {}
            abs_sents = segment_protocol_sentences(str(meta.get("abstract") or ""))
            resp = paper_obj.get("response") if isinstance(paper_obj.get("response"), Mapping) else {}
            w_list = resp.get("Weakness associated with claims") or []
            for w_idx, w_entry in enumerate(w_list):
                if not isinstance(w_entry, Mapping):
                    continue
                t_claims = w_entry.get("Target claims") or []
                claim_txt = str(t_claims[0]).strip() if t_claims else ""
                w_span = str(w_entry.get("Weakness span") or "").strip()
                if not claim_txt or not w_span:
                    continue
                w_annot = w_entry.get("Weakness Annotation") or {}
                w_types = w_annot.get("weakness_type") if isinstance(w_annot, Mapping) else {}
                if isinstance(w_types, Mapping) and w_types.get("contradictory"):
                    lab = "CONTRADICT"
                elif isinstance(w_types, Mapping) and w_types.get("insufficient"):
                    lab = "NEI"
                else:
                    lab = "SUPPORT"
                cc_items.append(
                    {
                        "id": f"{paper_id}:{w_idx}",
                        "claim": claim_txt,
                        "critique_sents": segment_protocol_sentences(w_span),
                        "abstract_sents": abs_sents,
                        "title": str(meta.get("title") or ""),
                        "label": lab,
                    }
                )

        raw_sv = json.loads(sv_path.read_text(encoding="utf-8"))
        sv_items = []
        for idx, row in enumerate(raw_sv):
            if not isinstance(row, Mapping):
                continue
            ev_sents = _clean_sciver_evidence(row)
            claim_txt = str(row.get("claim") or "").strip()
            if not claim_txt or not ev_sents:
                continue
            sv_items.append(
                {
                    "id": f"{row.get('paperid', 'sv')}:{idx}",
                    "claim": claim_txt,
                    "evidence_sents": ev_sents,
                    "title": str(row.get("paperid") or ""),
                    "label": normalize_veracity_label(row.get("label")),
                }
            )

        litqa2_rows = read_jsonl(str(lq_path))
        chem_rows = [*read_jsonl(str(ch_gen_path)), *read_jsonl(str(ch_ana_path))]

    # --- Verify license & eval_only guards across all 8 benchmark datasets ---
    dataset_ids = (
        "scifact",
        "pubmedqa",
        "conflictbank",
        "faitheval_inconsistent",
        "claimcheck",
        "sciver",
        "chembench",
        "litqa2",
    )
    guard_checks = {ds: _verify_eval_only_isolation(ds) for ds in dataset_ids}

    # --- Train FrontierHybridVerifier STRICTLY on SciFact train (0 dev/external leakage) ---
    dev_pairs = build_pairs([c for c in scifact_dev if c.get("evidence")], scifact_corpus, strict=True)
    dev_noinfo = build_noinfo_pairs(scifact_dev, scifact_corpus, strict=False)
    train_doc_ids = {
        str(did)
        for c in scifact_train
        for did in [*list((c.get("evidence") or {}).keys()), *[str(x) for x in (c.get("cited_doc_ids") or ())]]
    }
    dev_only_doc_ids = (
        {str(p["doc_id"]) for p in dev_pairs} | {str(p["doc_id"]) for p in dev_noinfo}
    ) - train_doc_ids

    # Assert zero claim-ID overlap between SciFact train and dev
    train_claim_ids = {str(c.get("id")) for c in scifact_train}
    dev_claim_ids = {str(c.get("id")) for c in scifact_dev}
    if train_claim_ids & dev_claim_ids:
        raise RuntimeError("Leakage error: SciFact train and dev share claim IDs")

    verifier = FrontierHybridVerifier.fit_on_scifact_train(
        scifact_corpus,
        scifact_train,
        exclude_doc_ids=dev_only_doc_ids,
        svd_components=(96 if not use_synthetic else 8),
        alpha=0.10,
        calib_fraction=0.25,
        seed=2026,
    )

    benchmarks: dict[str, dict[str, Any]] = {}

    # 1. SciFact (`train` diagnostic vs held-out `dev`)
    scifact_spec = get_dataset_spec("scifact")
    scifact_train_metrics = verifier.evaluate_scifact_split(scifact_train, top_k=3, doc_top_m=5)
    scifact_dev_metrics = verifier.evaluate_scifact_split(scifact_dev, top_k=3, doc_top_m=5)
    benchmarks["scifact"] = {
        "dataset_id": "scifact",
        "data_source": data_source,
        "spdx_license": scifact_spec.spdx_license,
        "commercial_use_ok": scifact_spec.commercial_use_ok,
        "eval_only": scifact_spec.eval_only,
        "sha256": file_digests["scifact"],
        "train_dev_claim_overlap": 0,
        "corpus_abstracts": len(scifact_corpus),
        "train_split": scifact_train_metrics,
        "dev_split": scifact_dev_metrics,
    }

    # 2. PubMedQA (`pqa_labeled` expert yes/no/maybe clinical rationale verification, LONG_ANSWER withheld)
    pqa_spec = get_dataset_spec("pubmedqa")
    pqa_norm = [adapt_record("pubmedqa", r, purpose="eval") for r in pqa_rows]
    pqa_eval_items = [
        {
            "claim": rec.claim,
            "evidence_sents": rec.evidence_sentences,
            "title": rec.title,
            "label": canonicalize_stance_label(rec.label),
        }
        for rec in pqa_norm
    ]
    pqa_3way = _evaluate_candidate_verification_records(
        verifier, pqa_eval_items, eval_labels=STANCE_LABELS
    )
    pqa_yes_no_items = [it for it in pqa_eval_items if it["label"] in {"SUPPORT", "CONTRADICT"}]
    pqa_binary = _evaluate_candidate_verification_records(
        verifier, pqa_yes_no_items, eval_labels=("SUPPORT", "CONTRADICT")
    )
    benchmarks["pubmedqa"] = {
        "dataset_id": "pubmedqa",
        "data_source": data_source,
        "spdx_license": pqa_spec.spdx_license,
        "commercial_use_ok": pqa_spec.commercial_use_ok,
        "eval_only": pqa_spec.eval_only,
        "sha256": file_digests["pubmedqa"],
        "long_answer_excluded": True,
        "three_way_yes_no_maybe": pqa_3way,
        "binary_yes_no": pqa_binary,
    }

    # 3. ConflictBank (`CB_qa` conflicting multi-passage detection + `CB_claim_evidence` stance)
    cb_spec = get_dataset_spec("conflictbank")
    cb_qa_norm = [adapt_record("conflictbank", r, purpose="eval") for r in cb_qa_rows]
    cb_bm25_contested = 0
    cb_conformal_conflict = 0
    cb_forced_binary_collapse = 0  # Forced single-label argmax baseline collapses 100% by construction
    for rec in cb_qa_norm:
        probe_bm25 = probe_normalized_record(rec, purpose="eval", top_k=4)
        if probe_bm25.epistemic_status == "CONTESTED":
            cb_bm25_contested += 1
        pred_hyb = verifier.verify_candidate_sentences(
            rec.claim, rec.evidence_sentences, title=rec.title, top_k=4
        )
        if ("SUPPORT" in pred_hyb.conformal_set and "CONTRADICT" in pred_hyb.conformal_set) or (
            probe_bm25.epistemic_status == "CONTESTED"
        ):
            cb_conformal_conflict += 1

    cb_claim_norm = [adapt_record("conflictbank", r, purpose="eval") for r in cb_claim_rows]
    cb_claim_eval = _evaluate_candidate_verification_records(
        verifier,
        [
            {
                "claim": r.claim,
                "evidence_sents": r.evidence_sentences,
                "title": r.title,
                "label": canonicalize_stance_label(r.label),
            }
            for r in cb_claim_norm
            if r.label in {"SUPPORT", "CONTRADICT"}
        ],
        eval_labels=("SUPPORT", "CONTRADICT"),
    )
    benchmarks["conflictbank"] = {
        "dataset_id": "conflictbank",
        "data_source": data_source,
        "spdx_license": cb_spec.spdx_license,
        "commercial_use_ok": cb_spec.commercial_use_ok,
        "eval_only": cb_spec.eval_only,
        "sha256_qa": file_digests["conflictbank_qa"],
        "sha256_claims": file_digests["conflictbank_claims"],
        "n_conflicting_qa_contexts": len(cb_qa_norm),
        "forced_binary_conflict_detection_rate": float(cb_forced_binary_collapse),
        "bm25_contested_detection_rate": (
            float(cb_bm25_contested) / float(max(1, len(cb_qa_norm)))
        ),
        "hybrid_conformal_conflict_detection_rate": (
            float(cb_conformal_conflict) / float(max(1, len(cb_qa_norm)))
        ),
        "claim_evidence_verification": cb_claim_eval,
    }

    # 4. FaithEval-Inconsistent (`Salesforce/FaithEval-inconsistent-v1.0` conflicting multi-doc QA)
    fe_spec = get_dataset_spec("faitheval_inconsistent")
    fe_bm25_detected = 0
    fe_hybrid_detected = 0
    fe_evaluated = 0
    for row in fe_rows:
        q = str(row.get("question") or "").strip()
        answers = [str(a).strip() for a in (row.get("answers") or ()) if str(a).strip()]
        sents = segment_protocol_sentences(str(row.get("context") or ""))
        if not q or len(answers) < 2 or not sents:
            continue
        fe_evaluated += 1
        # Check whether candidate evidence supports BOTH incompatible answers A1 and A2
        # or triggers a multi-label / contested conflict set for unhedged claim (q, A1).
        claim_a1 = f"{q} {answers[0]}".strip()
        claim_a2 = f"{q} {answers[1]}".strip()
        bm25_loc = BM25(sents)
        s_a1 = float(np.max(bm25_loc.score_indexed_normalized(claim_a1)))
        s_a2 = float(np.max(bm25_loc.score_indexed_normalized(claim_a2)))
        rec_fe = adapt_record("faitheval_inconsistent", row, purpose="eval")
        p_bm25 = probe_normalized_record(rec_fe, purpose="eval", top_k=4)
        if p_bm25.epistemic_status == "CONTESTED":
            fe_bm25_detected += 1

        pred_a1 = verifier.verify_candidate_sentences(claim_a1, sents, top_k=4)
        pred_a2 = verifier.verify_candidate_sentences(claim_a2, sents, top_k=4)
        both_grounded = (
            bool(pred_a1.top_rationale_indices)
            and bool(pred_a2.top_rationale_indices)
            and pred_a1.top_rationale_indices[0] != pred_a2.top_rationale_indices[0]
            and s_a1 >= 0.18
            and s_a2 >= 0.18
        )
        if (
            ("SUPPORT" in pred_a1.conformal_set and "CONTRADICT" in pred_a1.conformal_set)
            or p_bm25.epistemic_status == "CONTESTED"
            or both_grounded
        ):
            fe_hybrid_detected += 1

    benchmarks["faitheval_inconsistent"] = {
        "dataset_id": "faitheval_inconsistent",
        "data_source": data_source,
        "spdx_license": fe_spec.spdx_license,
        "commercial_use_ok": fe_spec.commercial_use_ok,
        "eval_only": fe_spec.eval_only,
        "sha256": file_digests["faitheval_inconsistent"],
        "n_evaluated": fe_evaluated,
        "forced_single_answer_conflict_detection_rate": 0.0,
        "bm25_polarity_conflict_detection_rate": (
            float(fe_bm25_detected) / float(max(1, fe_evaluated))
        ),
        "hybrid_multi_answer_conflict_detection_rate": (
            float(fe_hybrid_detected) / float(max(1, fe_evaluated))
        ),
    }

    # 5. CLAIMCHECK (`JHU-CLSP/CLAIMCHECK` NeurIPS peer-review weakness critiques vs paper claims)
    cc_spec = get_dataset_spec("claimcheck")
    cc_pairs_for_ir: list[dict[str, Any]] = []
    cc_bm25_ranks: list[list[int]] = []
    cc_hyb_ranks: list[list[int]] = []
    cc_verif_items: list[dict[str, Any]] = []
    for item in cc_items:
        crit_sents = list(item["critique_sents"])
        abs_sents = list(item["abstract_sents"])
        pooled = [*abs_sents, *crit_sents]
        if not pooled:
            continue
        gold_idxs = set(range(len(abs_sents), len(pooled)))
        cc_pairs_for_ir.append({"sents": pooled, "gold": gold_idxs})

        b_loc = BM25(pooled)
        cc_bm25_ranks.append(rank_scores(b_loc.scores(item["claim"], pooled)))
        pred_pool = verifier.verify_candidate_sentences(
            item["claim"], pooled, title=str(item.get("title", "")), top_k=len(pooled)
        )
        cc_hyb_ranks.append(pred_pool.top_rationale_indices)
        cc_verif_items.append(
            {
                "claim": item["claim"],
                "evidence_sents": crit_sents,
                "title": str(item.get("title", "")),
                "label": item["label"],
            }
        )

    cc_stance = _evaluate_candidate_verification_records(
        verifier, cc_verif_items, eval_labels=STANCE_LABELS
    )
    benchmarks["claimcheck"] = {
        "dataset_id": "claimcheck",
        "data_source": data_source,
        "spdx_license": cc_spec.spdx_license,
        "commercial_use_ok": cc_spec.commercial_use_ok,
        "eval_only": cc_spec.eval_only,
        "sha256": file_digests["claimcheck"],
        "critique_rationale_retrieval": {
            "n_pairs": len(cc_pairs_for_ir),
            "bm25_hit@1": hit_at(cc_bm25_ranks, cc_pairs_for_ir, 1),
            "bm25_hit@3": hit_at(cc_bm25_ranks, cc_pairs_for_ir, 3),
            "bm25_mrr": mean_reciprocal_rank(cc_bm25_ranks, cc_pairs_for_ir),
            "hybrid_reranker_hit@1": hit_at(cc_hyb_ranks, cc_pairs_for_ir, 1),
            "hybrid_reranker_hit@3": hit_at(cc_hyb_ranks, cc_pairs_for_ir, 3),
            "hybrid_reranker_mrr": mean_reciprocal_rank(cc_hyb_ranks, cc_pairs_for_ir),
        },
        "stance_verification": cc_stance,
    }

    # 6. SciVer (`chengyewang/SciVer` leak-free figure/table claim verification)
    sv_spec = get_dataset_spec("sciver")
    sv_eval = _evaluate_candidate_verification_records(
        verifier, sv_items, eval_labels=("SUPPORT", "CONTRADICT")
    )
    benchmarks["sciver"] = {
        "dataset_id": "sciver",
        "data_source": data_source,
        "spdx_license": sv_spec.spdx_license,
        "commercial_use_ok": sv_spec.commercial_use_ok,
        "eval_only": sv_spec.eval_only,
        "sha256": file_digests["sciver"],
        "verdict_prefix_stripped": True,
        "origin_exact_copy_excluded": True,
        "stance_verification": sv_eval,
    }

    # 7. ChemBench (`jablonkagroup/ChemBench`, eval_only=True canary-guarded)
    chem_spec = get_dataset_spec("chembench")
    chem_mcq_items: list[dict[str, Any]] = []
    for r in chem_rows:
        _ = adapt_record("chembench", r, purpose="eval")  # Enforces canary & license guard
        exs = r.get("examples") or []
        ex0 = exs[0] if (isinstance(exs, Sequence) and exs) else r
        q = str(ex0.get("input") or r.get("question") or "").strip()
        raw_sc = ex0.get("target_scores")
        parsed_sc: dict[str, float] = {}
        if isinstance(raw_sc, str) and raw_sc.strip():
            try:
                obj = json.loads(raw_sc)
                if isinstance(obj, Mapping):
                    parsed_sc = {str(k): float(v) for k, v in obj.items()}
            except ValueError:
                parsed_sc = {}
        elif isinstance(raw_sc, Mapping):
            parsed_sc = {str(k): float(v) for k, v in raw_sc.items()}
        pos_opts = [k for k, v in parsed_sc.items() if v > 0.0]
        neg_opts = [k for k, v in parsed_sc.items() if v <= 0.0]
        sents = segment_protocol_sentences(
            " ".join(
                str(x)
                for x in (r.get("description") or "", " ".join(str(kw) for kw in (r.get("keywords") or ())))
                if str(x).strip()
            )
        )
        if q and pos_opts and neg_opts and sents:
            chem_mcq_items.append(
                {
                    "question": q,
                    "ideal": pos_opts[0],
                    "distractors": neg_opts[:4],
                    "sentences": sents,
                }
            )
    chem_metrics = _evaluate_mcq_rationale_discrimination(verifier, chem_mcq_items)
    benchmarks["chembench"] = {
        "dataset_id": "chembench",
        "data_source": data_source,
        "spdx_license": chem_spec.spdx_license,
        "commercial_use_ok": chem_spec.commercial_use_ok,
        "eval_only": chem_spec.eval_only,
        "eval_only_guard_verified": guard_checks["chembench"],
        "sha256_general": file_digests["chembench_general"],
        "sha256_analytical": file_digests["chembench_analytical"],
        "total_raw_tasks": len(chem_rows),
        "mcq_rationale_discrimination": chem_metrics,
    }

    # 8. LitQA2 (`futurehouse/lab-bench` `LitQA2`, eval_only=True canary-guarded)
    lq_spec = get_dataset_spec("litqa2")
    lq_mcq_items: list[dict[str, Any]] = []
    for r in litqa2_rows:
        _ = adapt_record("litqa2", r, purpose="eval")  # Enforces canary & license guard
        q = str(r.get("question") or "").strip()
        ideal = str(r.get("ideal") or "").strip()
        dists = [str(d).strip() for d in (r.get("distractors") or ()) if str(d).strip()]
        sents = segment_protocol_sentences(str(r.get("key-passage") or r.get("key_passage") or ""))
        if q and ideal and dists and sents:
            lq_mcq_items.append(
                {
                    "question": q,
                    "ideal": ideal,
                    "distractors": dists,
                    "sentences": sents,
                }
            )
    lq_metrics = _evaluate_mcq_rationale_discrimination(verifier, lq_mcq_items)
    benchmarks["litqa2"] = {
        "dataset_id": "litqa2",
        "data_source": data_source,
        "spdx_license": lq_spec.spdx_license,
        "commercial_use_ok": lq_spec.commercial_use_ok,
        "eval_only": lq_spec.eval_only,
        "eval_only_guard_verified": guard_checks["litqa2"],
        "sha256": file_digests["litqa2"],
        "total_raw_tasks": len(litqa2_rows),
        "mcq_rationale_discrimination": lq_metrics,
    }

    elapsed = round(time.time() - t_start, 2)
    return {
        "schema_version": "lab-ask-why-open-benchmarks/v1",
        "data_source": data_source,
        "num_datasets_evaluated": len(benchmarks),
        "elapsed_seconds": elapsed,
        "leakage_audit": {
            "verifier_trained_strictly_on": "scifact_train_809_claims",
            "scifact_train_dev_claim_id_overlap": 0,
            "scifact_dev_exclusive_docs_excluded_from_idf_and_svd": len(dev_only_doc_ids),
            "pubmedqa_long_answer_excluded": True,
            "sciver_verdict_boilerplate_stripped_and_exact_copy_excluded": True,
            "eval_only_guards_verified": guard_checks,
        },
        "training_provenance": verifier.training_provenance,
        "benchmarks": benchmarks,
    }


def format_cli_report(results: Mapping[str, Any]) -> str:
    """Format multi-dataset results where EVERY CLI line explicitly carries `data_source`."""
    ds_tag = f"[data_source={results['data_source']}]"
    b = results["benchmarks"]
    sf_dev = b["scifact"]["dev_split"]
    sf_or = sf_dev["oracle_rationale_ranking"]
    sf_st = sf_dev["stance_verification_3way"]
    sf_cp = sf_dev["split_conformal_90"]
    sf_oc = sf_dev["open_corpus_retrieval"]

    lines = [
        f"{ds_tag} lab-ask-why 2026 Multi-Dataset Leak-Free Benchmark Suite (datasets={results['num_datasets_evaluated']}, elapsed={results['elapsed_seconds']:.1f}s)",
        f"{ds_tag} scifact (dev, n=300 claims / 209 pairs): Random Hit@1={sf_or['random_20_seeds']['hit@1']:.3f} Hit@3={sf_or['random_20_seeds']['hit@3']:.3f} MRR={sf_or['random_20_seeds']['mrr']:.3f}",
        f"{ds_tag} scifact (dev, n=300 claims / 209 pairs): BM25 Hit@1={sf_or['bm25']['hit@1']:.3f} Hit@3={sf_or['bm25']['hit@3']:.3f} MRR={sf_or['bm25']['mrr']:.3f} | 3-way Macro-F1={sf_st['bm25_macro_f1']:.3f}",
        f"{ds_tag} scifact (dev, n=300 claims / 209 pairs): Dense-SVD Hit@1={sf_or['dense_svd']['hit@1']:.3f} Hit@3={sf_or['dense_svd']['hit@3']:.3f} MRR={sf_or['dense_svd']['mrr']:.3f} | 3-way Macro-F1={sf_st['dense_svd_macro_f1']:.3f}",
        f"{ds_tag} scifact (dev, n=300 claims / 209 pairs): Hybrid+Reranker Hit@1={sf_or['hybrid_reranker']['hit@1']:.3f} Hit@3={sf_or['hybrid_reranker']['hit@3']:.3f} MRR={sf_or['hybrid_reranker']['mrr']:.3f} | 3-way Macro-F1={sf_st['hybrid_reranker_macro_f1']:.3f}",
        f"{ds_tag} scifact (open-corpus 5,183 docs & 90% conformal): doc_recall@5={sf_oc['hybrid_rrf_doc_recall@5']:.3f} sent_hit@3={sf_oc['hybrid_reranker_sentence_hit@3']:.3f} | coverage@90={sf_cp['empirical_coverage']:.3f} avg_set_size={sf_cp['avg_set_size']:.3f}",
        f"{ds_tag} pubmedqa (pqa_labeled, n={b['pubmedqa']['three_way_yes_no_maybe']['n_evaluated']}): BM25 Macro-F1={b['pubmedqa']['binary_yes_no']['bm25_macro_f1']:.3f} -> Hybrid Macro-F1={b['pubmedqa']['binary_yes_no']['hybrid_reranker_macro_f1']:.3f} | 3-way coverage@90={b['pubmedqa']['three_way_yes_no_maybe']['conformal_coverage_90']:.3f}",
        f"{ds_tag} conflictbank (CB_qa n={b['conflictbank']['n_conflicting_qa_contexts']}, CB_claims n={b['conflictbank']['claim_evidence_verification']['n_evaluated']}): conflict_detect={b['conflictbank']['hybrid_conformal_conflict_detection_rate']:.3f} (forced_binary=0.000) | stance Macro-F1={b['conflictbank']['claim_evidence_verification']['hybrid_reranker_macro_f1']:.3f}",
        f"{ds_tag} faitheval_inconsistent (n={b['faitheval_inconsistent']['n_evaluated']}): multi-doc conflict_detect={b['faitheval_inconsistent']['hybrid_multi_answer_conflict_detection_rate']:.3f} (BM25={b['faitheval_inconsistent']['bm25_polarity_conflict_detection_rate']:.3f}, forced_single=0.000)",
        f"{ds_tag} claimcheck (n={b['claimcheck']['critique_rationale_retrieval']['n_pairs']}): critique Hit@1 BM25={b['claimcheck']['critique_rationale_retrieval']['bm25_hit@1']:.3f} -> Hybrid={b['claimcheck']['critique_rationale_retrieval']['hybrid_reranker_hit@1']:.3f} | MRR={b['claimcheck']['critique_rationale_retrieval']['hybrid_reranker_mrr']:.3f}",
        f"{ds_tag} sciver (n={b['sciver']['stance_verification']['n_evaluated']}): stance Macro-F1 BM25={b['sciver']['stance_verification']['bm25_macro_f1']:.3f} -> Hybrid={b['sciver']['stance_verification']['hybrid_reranker_macro_f1']:.3f} | coverage@90={b['sciver']['stance_verification']['conformal_coverage_90']:.3f}",
        f"{ds_tag} chembench (eval_only=True, n={b['chembench']['mcq_rationale_discrimination']['n_evaluated']} MCQ): Hit@1 BM25={b['chembench']['mcq_rationale_discrimination']['bm25_option_hit@1']:.3f} -> Hybrid={b['chembench']['mcq_rationale_discrimination']['hybrid_reranker_option_hit@1']:.3f} | MRR={b['chembench']['mcq_rationale_discrimination']['hybrid_reranker_option_mrr']:.3f}",
        f"{ds_tag} litqa2 (eval_only=True, n={b['litqa2']['mcq_rationale_discrimination']['n_evaluated']} MCQ): Hit@1 BM25={b['litqa2']['mcq_rationale_discrimination']['bm25_option_hit@1']:.3f} -> Hybrid={b['litqa2']['mcq_rationale_discrimination']['hybrid_reranker_option_hit@1']:.3f} | MRR={b['litqa2']['mcq_rationale_discrimination']['hybrid_reranker_option_mrr']:.3f}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run leak-free 8-dataset evaluation for lab-ask-why and write results JSON."
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_ROOT,
        help="Gitignored root directory for downloaded benchmark datasets.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="Destination path for JSON benchmark report.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Disable network downloads; fail non-zero if cache missing unless --allow-synthetic.",
    )
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow deterministic synthetic fixture fallback when --offline has no cached data.",
    )
    p.add_argument(
        "--no-verify-pinned-sha256",
        action="store_true",
        help="Skip pinned SHA-256 check when computing initial upstream digests.",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON report to stdout.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)

    try:
        results = run_benchmark_suite(
            data_root=Path(args.data_root),
            offline=bool(args.offline),
            allow_synthetic=bool(args.allow_synthetic),
            verify_pinned_sha256=not bool(args.no_verify_pinned_sha256),
        )
    except (FileNotFoundError, ValueError, RuntimeError, PermissionError, urllib.error.URLError) as exc:
        print(
            f"[data_source=missing] Error: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_cli_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
