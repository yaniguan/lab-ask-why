"""Leak-free multi-dataset benchmark suite for lab-ask-why (Layer L6 Epistemic Brain).

Downloads 8 real public scientific claim verification, conflicting-evidence, and
literature/protocol QA corpora into gitignored `data/open_benchmarks/` with SHA-256
verification and strict `eval_only` / `CC-BY-NC-2.0` license isolation, fits
`FrontierHybridVerifier` (and optional `NeuralBiEncoderNliVerifier`: `BAAI/bge-small-en-v1.5`
+ `cross-encoder/nli-deberta-v3-small`) strictly on `SciFact train` (`809` claims: `607`
`train_fit` [`75%`] + `202` `train_calib` [`25%`] conformal calibration), and evaluates
zero-leakage across:
  1. `scifact` (`train` vs held-out `dev`: Random, Majority, BM25, Dense-SVD, Stage-1 RRF,
     Sparse Hybrid+Reranker, and Neural BGE+DeBERTa-v3 NLI on `Hit@1`, `Hit@3`, `MRR`,
     Official Wadden et al. 2020 Abstract & Sentence F1, 3-way stance macro-F1, and `90%`
     Mondrian Split Conformal coverage, set size, single-label rate, and 95% Wilson CIs),
  2. `pubmedqa` (`pqa_labeled`: `1,000` expert `yes`/`no`/`maybe` clinical rationale items),
  3. `conflictbank` (`CB_qa` conflicting multi-evidence vs matched single-passage negative
     control + `CB_claim_evidence` self-paired `SUPPORT` & cross-subject `CONTRADICT`),
  4. `faitheval_inconsistent` (`Salesforce/FaithEval-inconsistent-v1.0` conflicting multi-doc
     contexts vs matched single-document `answers[0]`-only negative control),
  5. `claimcheck` (`JHU-CLSP/CLAIMCHECK` NeurIPS peer-review weakness critiques),
  6. `sciver` (`chengyewang/SciVer` scientific figure/table claim verification),
  7. `chembench` (`jablonkagroup/ChemBench`, option string matching against task text,
     `eval_only=True` canary-guarded),
  8. `litqa2` (`futurehouse/lab-bench` `LitQA2`, with gold passage, `eval_only=True`
     canary-guarded).

Every output dictionary and every CLI line explicitly carries `data_source`
(`"real_downloaded"` vs `"synthetic_fixture"`), and `--offline` without cached
SHA-256-verified data exits non-zero unless `--allow-synthetic` is passed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

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
    build_conflictbank_claim_evidence_pairs,
    enforce_dataset_guard,
    get_dataset_spec,
    normalize_veracity_label,
    probe_normalized_record,
)
import lab_ask_why.frontier_verifier as fv_mod  # noqa: E402
from lab_ask_why.frontier_verifier import (  # noqa: E402
    STANCE_LABELS,
    EntityNumericNegationIndex,
    FrontierHybridVerifier,
    canonicalize_stance_label,
)
from lab_ask_why.labbench_adapter import segment_protocol_sentences  # noqa: E402
from lab_ask_why.metrics import (  # noqa: E402
    chance_mcq_baseline,
    compute_auroc,
    hit_at,
    macro_f1_score,
    majority_class_stance_baseline,
    mean_reciprocal_rank,
    rank_scores,
    wilson_score_interval,
)
from lab_ask_why.neural_verifier import (  # noqa: E402
    NeuralBiEncoderNliVerifier,
    is_neural_backend_available,
)
from lab_ask_why.retrieval import _SubwordConceptCosineIndex, classify_rationale_polarity  # noqa: E402

DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "open_benchmarks_2026.json"
DEFAULT_WIRE_CALIBRATOR_PATH = (
    REPO_ROOT / "src" / "lab_ask_why" / "resources" / "scifact_wire_calibrator.json"
)
_USER_AGENT = "lab-ask-why/0.3 (+https://github.com/yaniguan/lab-ask-why)"
_DOC_SPLIT_RE = re.compile(r"(?=\bDocument(?:\s+\d+)?\s*:)")

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


def measure_peak_rss_mb() -> float:
    """Measure process peak RSS in MiB via `resource.getrusage` (POSIX) or `psapi` (Windows)."""
    if os.name != "nt":
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB
        if sys.platform == "darwin":
            return float(ru) / (1024.0 * 1024.0)
        return float(ru) / 1024.0

    import ctypes
    import ctypes.wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = _PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    get_cur_proc = ctypes.windll.kernel32.GetCurrentProcess
    get_cur_proc.restype = ctypes.c_void_p
    get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_mem_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.wintypes.DWORD,
    ]
    get_mem_info(get_cur_proc(), ctypes.byref(pmc), pmc.cb)
    return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)


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
    sents = segment_protocol_sentences(cleaned)
    claim_norm = str(row.get("claim") or "").strip().lower()
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
            "claim": "PAM-proximal seed mismatches enhance Cas9 DNA cleavage.",
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
                "id": "P108",
                "subject": "Aspirin",
                "category": "default",
                "claim": "Buffered medium increases aspirin stability at 25 C.",
                "evidence": "Buffered medium increases aspirin stability at 25 C in 96-well plates.",
            },
            {
                "id": "P108",
                "subject": "Aspirin",
                "category": "semantic_conflict",
                "claim": "Buffered medium decreases aspirin stability at 80 C.",
                "evidence": "Buffered medium decreases aspirin stability at 80 C and accelerates ester hydrolysis.",
            },
        ],
        "faitheval": [
            {
                "qid": "syn-fe-1",
                "question": "Which kinase phosphorylates substrate S14 in hepatocytes?",
                "context": (
                    "Document 1: Mass spectrometry confirms AMPK phosphorylates substrate S14 in hepatocytes. "
                    "Document 2: Knockout assays show mTORC1 phosphorylates substrate S14 in hepatocytes."
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


def _pack_conformal_stats(
    covered: Sequence[bool],
    set_sizes: Sequence[int],
    by_class: Mapping[str, Sequence[bool]],
    eval_labels: Sequence[str],
) -> dict[str, Any]:
    cov_detail = {
        lab: {
            "n": len(by_class.get(lab, ())),
            "coverage": (
                float(np.mean(by_class[lab])) if by_class.get(lab) else float("nan")
            ),
            "wilson_95_ci": list(
                wilson_score_interval(
                    int(sum(by_class.get(lab, ()))), len(by_class.get(lab, ()))
                )
            ),
        }
        for lab in eval_labels
    }
    return {
        "conformal_coverage_90": float(np.mean(covered)) if covered else float("nan"),
        "conformal_coverage_wilson_95_ci": list(
            wilson_score_interval(int(sum(covered)), len(covered))
        ),
        "conformal_avg_set_size_90": float(np.mean(set_sizes)) if set_sizes else float("nan"),
        "conformal_single_label_rate_90": (
            float(sum(1 for s in set_sizes if s == 1)) / float(len(set_sizes))
            if set_sizes
            else 0.0
        ),
        "conformal_coverage_by_class_90": {
            lab: cov_detail[lab]["coverage"] for lab in eval_labels
        },
        "conformal_coverage_by_class_detail_90": cov_detail,
    }


def _evaluate_candidate_verification_records(
    verifier: FrontierHybridVerifier,
    items: Sequence[Mapping[str, Any]],
    *,
    eval_labels: Sequence[str] = STANCE_LABELS,
    neural_verifier: NeuralBiEncoderNliVerifier | None = None,
) -> dict[str, Any]:
    """Compare Majority, Okapi BM25, `FrontierHybridVerifier`, and optional `NeuralBiEncoderNliVerifier`."""
    y_true: list[str] = []
    y_bm25: list[str] = []
    y_hybrid: list[str] = []
    covered: list[bool] = []
    set_sizes: list[int] = []
    by_cls: dict[str, list[bool]] = {lab: [] for lab in eval_labels}
    conflict_flags: list[bool] = []

    y_neural: list[str] = []
    n_covered: list[bool] = []
    n_set_sizes: list[int] = []
    n_by_cls: dict[str, list[bool]] = {lab: [] for lab in eval_labels}

    if neural_verifier is not None and items:
        neural_verifier.prewarm_candidate_items(items, nli_top_s=4)

    for it in items:
        claim = str(it["claim"])
        sents = list(it["evidence_sents"])
        title = str(it.get("title", ""))
        gold = canonicalize_stance_label(it["label"])
        y_true.append(gold)

        if not sents:
            y_bm25.append("NEI" if "NEI" in eval_labels else "SUPPORT")
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

        pred = verifier.verify_candidate_sentences(claim, sents, title=title, top_k=3)
        if len(eval_labels) == 2 and set(eval_labels) == {"SUPPORT", "CONTRADICT"}:
            p_sup = float(pred.probabilities["SUPPORT"])
            p_con = float(pred.probabilities["CONTRADICT"])
            y_hybrid.append("CONTRADICT" if p_con > p_sup else "SUPPORT")
        else:
            y_hybrid.append(pred.predicted_label)

        is_c = gold in pred.conformal_set
        covered.append(is_c)
        if gold in by_cls:
            by_cls[gold].append(is_c)
        set_sizes.append(len(pred.conformal_set))
        conflict_flags.append(pred.is_contested_conflict)

        if neural_verifier is not None:
            n_pred = neural_verifier.verify_candidate_sentences(
                claim, sents, title=title, top_k=3
            )
            if len(eval_labels) == 2 and set(eval_labels) == {"SUPPORT", "CONTRADICT"}:
                np_sup = float(n_pred.probabilities["SUPPORT"])
                np_con = float(n_pred.probabilities["CONTRADICT"])
                y_neural.append("CONTRADICT" if np_con > np_sup else "SUPPORT")
            else:
                y_neural.append(n_pred.predicted_label)
            nc = gold in n_pred.conformal_set
            n_covered.append(nc)
            if gold in n_by_cls:
                n_by_cls[gold].append(nc)
            n_set_sizes.append(len(n_pred.conformal_set))

    maj = majority_class_stance_baseline(y_true, eval_labels=eval_labels)
    out: dict[str, Any] = {
        "n_evaluated": len(items),
        "label_counts": dict(Counter(y_true)),
        **maj,
        "bm25_macro_f1": macro_f1_score(y_true, y_bm25, labels=eval_labels),
        "hybrid_reranker_macro_f1": macro_f1_score(y_true, y_hybrid, labels=eval_labels),
        "bm25_accuracy": float(np.mean([yt == yp for yt, yp in zip(y_true, y_bm25)])) if y_true else float("nan"),
        "hybrid_reranker_accuracy": float(np.mean([yt == yp for yt, yp in zip(y_true, y_hybrid)])) if y_true else float("nan"),
        **_pack_conformal_stats(covered, set_sizes, by_cls, eval_labels),
        "contested_conflict_rate": float(np.mean(conflict_flags)) if conflict_flags else 0.0,
    }
    if neural_verifier is not None:
        n_stats = _pack_conformal_stats(n_covered, n_set_sizes, n_by_cls, eval_labels)
        out.update(
            {
                "neural_nli_macro_f1": macro_f1_score(y_true, y_neural, labels=eval_labels),
                "neural_nli_accuracy": (
                    float(np.mean([yt == yp for yt, yp in zip(y_true, y_neural)]))
                    if y_true
                    else float("nan")
                ),
                "neural_conformal_coverage_90": n_stats["conformal_coverage_90"],
                "neural_conformal_coverage_wilson_95_ci": n_stats["conformal_coverage_wilson_95_ci"],
                "neural_conformal_avg_set_size_90": n_stats["conformal_avg_set_size_90"],
                "neural_conformal_single_label_rate_90": n_stats["conformal_single_label_rate_90"],
                "neural_conformal_coverage_by_class_90": n_stats["conformal_coverage_by_class_90"],
                "neural_conformal_coverage_by_class_detail_90": n_stats[
                    "conformal_coverage_by_class_detail_90"
                ],
            }
        )
    return out


def _tie_aware_hit1_and_rr(option_scores: Sequence[float]) -> tuple[float, float]:
    """Exact tie-aware Hit@1 and Reciprocal Rank for `option_scores[0]` (ideal) vs distractors."""
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
    *,
    evaluation_mode: str,
    neural_verifier: NeuralBiEncoderNliVerifier | None = None,
) -> dict[str, Any]:
    """Compare Chance, BM25, `FrontierHybridVerifier`, and optional `NeuralBiEncoderNliVerifier` on MCQ."""
    bm25_wins: list[float] = []
    hybrid_wins: list[float] = []
    bm25_rr: list[float] = []
    hybrid_rr: list[float] = []
    neural_wins: list[float] = []
    neural_rr: list[float] = []
    option_counts: list[int] = []

    if neural_verifier is not None and mcq_items:
        neural_verifier.prewarm_mcq_items(mcq_items)

    for item in mcq_items:
        q = str(item["question"]).strip()
        ideal = str(item["ideal"]).strip()
        distractors = [str(d).strip() for d in item["distractors"] if str(d).strip()]
        sents = [str(s).strip() for s in item["sentences"] if str(s).strip()]
        if not q or not ideal or not distractors or not sents:
            continue

        options = [ideal, *distractors]
        option_counts.append(len(options))
        bm25_local = BM25(sents)
        subword_local = _SubwordConceptCosineIndex(sents)
        entity_local = EntityNumericNegationIndex(sents)
        sent_embs = verifier.sent_svd.embed(sents)

        q_pred = verifier.verify_candidate_sentences(q, sents, top_k=len(sents))
        sent_weights = np.ones(len(sents), dtype=np.float64)
        for r_idx, r_sc in zip(q_pred.top_rationale_indices, q_pred.top_rationale_scores):
            sent_weights[int(r_idx)] = 0.75 + 0.50 * float(r_sc)

        b_scores: list[float] = []
        h_scores: list[float] = []
        for opt in options:
            opt_bm25 = bm25_local.scores(opt, sents)
            b_scores.append(float(np.max(opt_bm25)) if opt_bm25.size > 0 else 0.0)

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

        if neural_verifier is not None:
            n_scores = neural_verifier.score_mcq_options(q, options, sents)
            n_h1, n_rr_val = _tie_aware_hit1_and_rr(n_scores)
            neural_wins.append(n_h1)
            neural_rr.append(n_rr_val)

    chance = chance_mcq_baseline(option_counts)
    out: dict[str, Any] = {
        "n_evaluated": len(bm25_wins),
        "evaluation_mode": evaluation_mode,
        **chance,
        "bm25_option_hit@1": float(np.mean(bm25_wins)) if bm25_wins else float("nan"),
        "hybrid_reranker_option_hit@1": float(np.mean(hybrid_wins)) if hybrid_wins else float("nan"),
        "bm25_option_mrr": float(np.mean(bm25_rr)) if bm25_rr else float("nan"),
        "hybrid_reranker_option_mrr": float(np.mean(hybrid_rr)) if hybrid_rr else float("nan"),
    }
    if neural_verifier is not None:
        out["neural_nli_option_hit@1"] = (
            float(np.mean(neural_wins)) if neural_wins else float("nan")
        )
        out["neural_nli_option_mrr"] = float(np.mean(neural_rr)) if neural_rr else float("nan")
    return out


def _pack_conflict_control_metrics(
    pos_flags: Sequence[bool],
    neg_flags: Sequence[bool],
    pos_scores: Sequence[float] | None = None,
    neg_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    n_pos = len(pos_flags)
    n_neg = len(neg_flags)
    tp = int(sum(1 for x in pos_flags if x))
    fp = int(sum(1 for x in neg_flags if x))
    tpr = float(tp) / float(max(1, n_pos))
    fpr = float(fp) / float(max(1, n_neg))
    bal_acc = 0.5 * (tpr + (1.0 - fpr))
    out: dict[str, Any] = {
        "n_positive": n_pos,
        "n_negative_control": n_neg,
        "true_positive_rate": tpr,
        "tpr_wilson_95_ci": list(wilson_score_interval(tp, n_pos)),
        "false_positive_rate": fpr,
        "fpr_wilson_95_ci": list(wilson_score_interval(fp, n_neg)),
        "balanced_accuracy": bal_acc,
    }
    if pos_scores is not None and neg_scores is not None:
        out["conflict_auroc"] = compute_auroc(pos_scores, neg_scores)
    return out


def _evaluate_conflict_pair_with_negative_controls(
    verifier: FrontierHybridVerifier,
    pos_cases: Sequence[tuple[str, list[str], Any]],
    neg_cases: Sequence[tuple[str, list[str], Any]],
    *,
    neural_verifier: NeuralBiEncoderNliVerifier | None = None,
) -> dict[str, Any]:
    """Evaluate BM25, Sparse Hybrid Bilateral, and optional Neural NLI Bilateral WITHOUT ORing."""
    bm25_pos, bm25_neg = [], []
    hyb_pos, hyb_neg = [], []
    hyb_pos_sc, hyb_neg_sc = [], []
    neu_pos, neu_neg = [], []
    neu_pos_sc, neu_neg_sc = [], []

    if neural_verifier is not None:
        neural_verifier.prewarm_conflict_cases([*pos_cases, *neg_cases])

    for claim, passages, norm_rec in pos_cases:
        p_bm25 = probe_normalized_record(norm_rec, purpose="eval", top_k=4)
        bm25_pos.append(p_bm25.epistemic_status == "CONTESTED")

        h_res = verifier.score_bilateral_conflict(claim, passages)
        hyb_pos.append(bool(h_res["is_conflict"]))
        hyb_pos_sc.append(float(h_res["conflict_score"]))

        if neural_verifier is not None:
            n_res = neural_verifier.score_bilateral_conflict(claim, passages)
            neu_pos.append(bool(n_res["is_conflict"]))
            neu_pos_sc.append(float(n_res["conflict_score"]))

    for claim, passages, norm_rec in neg_cases:
        p_bm25 = probe_normalized_record(norm_rec, purpose="eval", top_k=4)
        bm25_neg.append(p_bm25.epistemic_status == "CONTESTED")

        h_res = verifier.score_bilateral_conflict(claim, passages)
        hyb_neg.append(bool(h_res["is_conflict"]))
        hyb_neg_sc.append(float(h_res["conflict_score"]))

        if neural_verifier is not None:
            n_res = neural_verifier.score_bilateral_conflict(claim, passages)
            neu_neg.append(bool(n_res["is_conflict"]))
            neu_neg_sc.append(float(n_res["conflict_score"]))

    bm25_m = _pack_conflict_control_metrics(bm25_pos, bm25_neg)
    hyb_m = _pack_conflict_control_metrics(hyb_pos, hyb_neg, hyb_pos_sc, hyb_neg_sc)
    result: dict[str, Any] = {
        "forced_single_verdict_baseline": {
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "balanced_accuracy": 0.5,
        },
        "bm25": bm25_m,
        "hybrid_bilateral": hyb_m,
        # Top-level convenience fields for hybrid_bilateral (strictly independent of BM25)
        "true_positive_rate": hyb_m["true_positive_rate"],
        "false_positive_rate": hyb_m["false_positive_rate"],
        "balanced_accuracy": hyb_m["balanced_accuracy"],
        "conflict_auroc": hyb_m["conflict_auroc"],
    }
    if neural_verifier is not None:
        result["neural_nli_bilateral"] = _pack_conflict_control_metrics(
            neu_pos, neu_neg, neu_pos_sc, neu_neg_sc
        )
    return result


def run_benchmark_suite(
    *,
    data_root: Path = Path(DEFAULT_ROOT),
    offline: bool = False,
    allow_synthetic: bool = False,
    verify_pinned_sha256: bool = True,
    scifact_only: bool = False,
    pure_numpy: bool = False,
    enable_neural: bool | None = None,
    save_wire_calibrator_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the leak-free benchmark evaluation and return structured results dict."""
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

        if not scifact_only:
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
    except (FileNotFoundError, ValueError, RuntimeError, urllib.error.URLError):
        if offline and allow_synthetic:
            use_synthetic = True
        else:
            raise

    data_source = "synthetic_fixture" if use_synthetic else "real_downloaded"

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
        file_digests = {
            k: "synthetic_fixture_no_download"
            for k in (
                "scifact",
                "pubmedqa",
                "claimcheck",
                "sciver",
                "faitheval_inconsistent",
                "conflictbank_qa",
                "conflictbank_claims",
                "litqa2",
                "chembench_general",
                "chembench_analytical",
            )
        }
    else:
        scifact_corpus = {r["doc_id"]: r for r in read_jsonl("corpus.jsonl", str(scifact_dir))}
        scifact_train = read_jsonl("claims_train.jsonl", str(scifact_dir))
        scifact_dev = read_jsonl("claims_dev.jsonl", str(scifact_dir))
        if not scifact_only:
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

    dataset_ids = (
        ("scifact",)
        if scifact_only
        else (
            "scifact",
            "pubmedqa",
            "conflictbank",
            "faitheval_inconsistent",
            "claimcheck",
            "sciver",
            "chembench",
            "litqa2",
        )
    )
    guard_checks = {ds: _verify_eval_only_isolation(ds) for ds in dataset_ids}

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

    train_claim_ids = {str(c.get("id")) for c in scifact_train}
    dev_claim_ids = {str(c.get("id")) for c in scifact_dev}
    if train_claim_ids & dev_claim_ids:
        raise RuntimeError("Leakage error: SciFact train and dev share claim IDs")

    orig_sp = fv_mod._sp
    try:
        if pure_numpy:
            fv_mod._sp = None
        verifier = FrontierHybridVerifier.fit_on_scifact_train(
            scifact_corpus,
            scifact_train,
            exclude_doc_ids=dev_only_doc_ids,
            svd_components=(96 if not use_synthetic else 8),
            alpha=0.10,
            calib_fraction=0.25,
            seed=2026,
        )
        peak_rss_after_sparse_fit_mb = round(measure_peak_rss_mb(), 2)
    finally:
        fv_mod._sp = orig_sp

    if save_wire_calibrator_path is not None and not use_synthetic:
        verifier.save_calibrator(save_wire_calibrator_path)

    run_neural = (
        bool(enable_neural)
        if enable_neural is not None
        else (not use_synthetic and not scifact_only and is_neural_backend_available())
    )
    neural_verifier: NeuralBiEncoderNliVerifier | None = None
    if run_neural:
        emb_cache = bench_cache_dir / ".neural_cache"
        neural_verifier = NeuralBiEncoderNliVerifier.fit_on_scifact_train(
            scifact_corpus,
            scifact_train,
            exclude_doc_ids=dev_only_doc_ids,
            alpha=0.10,
            calib_fraction=0.25,
            seed=2026,
            embedding_cache_dir=emb_cache,
            local_files_only=offline,
        )

    benchmarks: dict[str, dict[str, Any]] = {}

    # 1. SciFact (`train` diagnostic vs held-out `dev` + Official SciFact F1 + Neural NLI)
    scifact_spec = get_dataset_spec("scifact")
    scifact_train_metrics = verifier.evaluate_scifact_split(scifact_train, top_k=3, doc_top_m=5)
    scifact_dev_metrics = verifier.evaluate_scifact_split(scifact_dev, top_k=3, doc_top_m=5)

    if neural_verifier is not None:
        neural_dev = neural_verifier.evaluate_scifact_split(scifact_dev, top_k=3, doc_top_m=5)
        scifact_dev_metrics["neural_biencoder_nli"] = neural_dev
        scifact_dev_metrics["official_scifact_metrics"]["neural_biencoder_nli"] = neural_dev[
            "official_scifact_metrics"
        ]

    benchmarks["scifact"] = {
        "dataset_id": "scifact",
        "data_source": data_source,
        "spdx_license": scifact_spec.spdx_license,
        "commercial_use_ok": scifact_spec.commercial_use_ok,
        "eval_only": scifact_spec.eval_only,
        "sha256": file_digests["scifact"],
        "train_dev_claim_overlap": 0,
        "corpus_abstracts": len(scifact_corpus),
        "sparse_fit_peak_rss_mb": peak_rss_after_sparse_fit_mb,
        "train_split": scifact_train_metrics,
        "dev_split": scifact_dev_metrics,
    }

    if scifact_only:
        elapsed_sf = round(time.time() - t_start, 2)
        return {
            "schema_version": "lab-ask-why-open-benchmarks/v2",
            "data_source": data_source,
            "num_datasets_evaluated": 1,
            "elapsed_seconds": elapsed_sf,
            "pure_numpy_mode": bool(pure_numpy),
            "peak_rss_mb": round(measure_peak_rss_mb(), 2),
            "training_provenance": verifier.training_provenance,
            "benchmarks": benchmarks,
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
        verifier, pqa_eval_items, eval_labels=STANCE_LABELS, neural_verifier=neural_verifier
    )
    pqa_yes_no_items = [it for it in pqa_eval_items if it["label"] in {"SUPPORT", "CONTRADICT"}]
    pqa_binary = _evaluate_candidate_verification_records(
        verifier,
        pqa_yes_no_items,
        eval_labels=("SUPPORT", "CONTRADICT"),
        neural_verifier=neural_verifier,
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

    # 3. ConflictBank (`CB_qa` with matched single-passage negative control + `CB_claim_evidence`)
    cb_spec = get_dataset_spec("conflictbank")
    cb_pos_cases: list[tuple[str, list[str], Any]] = []
    cb_neg_cases: list[tuple[str, list[str], Any]] = []
    conflict_fields = (
        "misinformation_conflict_evidence_evidence",
        "temporal_conflict_evidence",
        "semantic_conflict_evidence",
    )
    for r in cb_qa_rows:
        rec_pos = adapt_record("conflictbank", r, purpose="eval")
        def_ev = str(r.get("default_evidence") or "").strip()
        conf_evs = [str(r.get(cf) or "").strip() for cf in conflict_fields if str(r.get(cf) or "").strip()]
        if not rec_pos.claim or not def_ev or not conf_evs:
            continue
        cb_pos_cases.append((rec_pos.claim, [def_ev, *conf_evs], rec_pos))
        # Matched negative control: default_evidence ONLY (zero conflict passages)
        neg_row = {
            "id": r.get("id", r.get("relation", "cb-neg")),
            "subject": r.get("subject"),
            "default_claim": r["default_claim"],
            "default_evidence": def_ev,
        }
        rec_neg = adapt_record("conflictbank", neg_row, purpose="eval")
        cb_neg_cases.append((rec_neg.claim, [def_ev], rec_neg))

    cb_conflict_eval = _evaluate_conflict_pair_with_negative_controls(
        verifier, cb_pos_cases, cb_neg_cases, neural_verifier=neural_verifier
    )

    # `CB_claim_evidence`: 200 self-paired SUPPORT + 146 cross-subject CONTRADICT = 346 pairs
    cb_self_only = build_conflictbank_claim_evidence_pairs(
        cb_claim_rows, include_cross_subject_contradictions=False, purpose="eval"
    )
    cb_self_eval = _evaluate_candidate_verification_records(
        verifier,
        [
            {
                "claim": r.claim,
                "evidence_sents": r.evidence_sentences,
                "title": r.title,
                "label": "SUPPORT",
            }
            for r in cb_self_only
        ],
        eval_labels=("SUPPORT", "CONTRADICT"),
        neural_verifier=neural_verifier,
    )

    cb_paired_all = build_conflictbank_claim_evidence_pairs(
        cb_claim_rows, include_cross_subject_contradictions=True, purpose="eval"
    )
    cb_claim_eval = _evaluate_candidate_verification_records(
        verifier,
        [
            {
                "claim": r.claim,
                "evidence_sents": r.evidence_sentences,
                "title": r.title,
                "label": canonicalize_stance_label(r.label),
            }
            for r in cb_paired_all
            if r.label in {"SUPPORT", "CONTRADICT"}
        ],
        eval_labels=("SUPPORT", "CONTRADICT"),
        neural_verifier=neural_verifier,
    )
    benchmarks["conflictbank"] = {
        "dataset_id": "conflictbank",
        "data_source": data_source,
        "spdx_license": cb_spec.spdx_license,
        "commercial_use_ok": cb_spec.commercial_use_ok,
        "eval_only": cb_spec.eval_only,
        "sha256_qa": file_digests["conflictbank_qa"],
        "sha256_claims": file_digests["conflictbank_claims"],
        "n_conflicting_qa_contexts": len(cb_pos_cases),
        "n_negative_control_qa_contexts": len(cb_neg_cases),
        "conflict_detection": cb_conflict_eval,
        "self_paired_support_verification": {
            "n_evaluated": cb_self_eval["n_evaluated"],
            "bm25_accuracy": cb_self_eval["bm25_accuracy"],
            "hybrid_reranker_accuracy": cb_self_eval["hybrid_reranker_accuracy"],
            **(
                {"neural_nli_accuracy": cb_self_eval["neural_nli_accuracy"]}
                if "neural_nli_accuracy" in cb_self_eval
                else {}
            ),
        },
        "claim_evidence_verification": cb_claim_eval,
    }

    # 4. FaithEval-Inconsistent (`n=300` conflicting multi-doc vs matched single-doc `answers[0]` negative control)
    fe_spec = get_dataset_spec("faitheval_inconsistent")
    fe_pos_cases: list[tuple[str, list[str], Any]] = []
    fe_neg_cases: list[tuple[str, list[str], Any]] = []
    for row in fe_rows:
        q = str(row.get("question") or "").strip()
        answers = [str(a).strip() for a in (row.get("answers") or ()) if str(a).strip()]
        raw_ctx = str(row.get("context") or "").strip()
        if not q or len(answers) < 2 or not raw_ctx:
            continue
        docs = [d.strip() for d in _DOC_SPLIT_RE.split(raw_ctx) if d.strip()]
        if not docs:
            docs = [raw_ctx]
        rec_pos = adapt_record("faitheval_inconsistent", row, purpose="eval")
        fe_pos_cases.append((rec_pos.claim, docs, rec_pos))

        # Matched negative control: single document mentioning answers[0] and NOT answers[1]
        a0_low = answers[0].lower()
        a1_low = answers[1].lower()
        only_a0_docs = [d for d in docs if (a0_low in d.lower() and a1_low not in d.lower())]
        if only_a0_docs:
            neg_row = {
                "qid": f"{row.get('qid', 'fe')}:neg",
                "question": q,
                "context": only_a0_docs[0],
                "answers": [answers[0]],
                "subset": row.get("subset"),
            }
            rec_neg = adapt_record("faitheval_inconsistent", neg_row, purpose="eval")
            fe_neg_cases.append((rec_neg.claim, [only_a0_docs[0]], rec_neg))

    fe_conflict_eval = _evaluate_conflict_pair_with_negative_controls(
        verifier, fe_pos_cases, fe_neg_cases, neural_verifier=neural_verifier
    )
    benchmarks["faitheval_inconsistent"] = {
        "dataset_id": "faitheval_inconsistent",
        "data_source": data_source,
        "spdx_license": fe_spec.spdx_license,
        "commercial_use_ok": fe_spec.commercial_use_ok,
        "eval_only": fe_spec.eval_only,
        "sha256": file_digests["faitheval_inconsistent"],
        "n_evaluated": len(fe_pos_cases),
        "n_negative_control": len(fe_neg_cases),
        "conflict_detection": fe_conflict_eval,
    }

    # 5. CLAIMCHECK (`JHU-CLSP/CLAIMCHECK` NeurIPS peer-review weakness critiques vs paper claims)
    cc_spec = get_dataset_spec("claimcheck")
    cc_pairs_for_ir: list[dict[str, Any]] = []
    cc_bm25_ranks: list[list[int]] = []
    cc_hyb_ranks: list[list[int]] = []
    cc_neu_ranks: list[list[int]] = []
    cc_rand_h1: list[float] = []
    cc_rand_h3: list[float] = []
    cc_rand_mrr: list[float] = []
    cc_verif_items: list[dict[str, Any]] = []

    if neural_verifier is not None and cc_items:
        neural_verifier.prewarm_candidate_items(
            [
                {
                    "claim": it["claim"],
                    "evidence_sents": [*it["abstract_sents"], *it["critique_sents"]],
                    "title": str(it.get("title", "")),
                }
                for it in cc_items
            ],
            nli_top_s=6,
        )

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
        if neural_verifier is not None:
            n_pool = neural_verifier.verify_candidate_sentences(
                item["claim"], pooled, title=str(item.get("title", "")), top_k=len(pooled)
            )
            cc_neu_ranks.append(n_pool.top_rationale_indices)

        cc_verif_items.append(
            {
                "claim": item["claim"],
                "evidence_sents": crit_sents,
                "title": str(item.get("title", "")),
                "label": item["label"],
            }
        )

    for seed_i in range(20):
        rng = np.random.default_rng(seed_i)
        r_perm = [list(rng.permutation(len(p["sents"]))) for p in cc_pairs_for_ir]
        cc_rand_h1.append(hit_at(r_perm, cc_pairs_for_ir, 1))
        cc_rand_h3.append(hit_at(r_perm, cc_pairs_for_ir, 3))
        cc_rand_mrr.append(mean_reciprocal_rank(r_perm, cc_pairs_for_ir))

    cc_stance = _evaluate_candidate_verification_records(
        verifier, cc_verif_items, eval_labels=STANCE_LABELS, neural_verifier=neural_verifier
    )
    cc_ir_dict: dict[str, Any] = {
        "n_pairs": len(cc_pairs_for_ir),
        "chance_hit@1": float(np.mean(cc_rand_h1)) if cc_rand_h1 else float("nan"),
        "chance_hit@3": float(np.mean(cc_rand_h3)) if cc_rand_h3 else float("nan"),
        "chance_mrr": float(np.mean(cc_rand_mrr)) if cc_rand_mrr else float("nan"),
        "bm25_hit@1": hit_at(cc_bm25_ranks, cc_pairs_for_ir, 1),
        "bm25_hit@3": hit_at(cc_bm25_ranks, cc_pairs_for_ir, 3),
        "bm25_mrr": mean_reciprocal_rank(cc_bm25_ranks, cc_pairs_for_ir),
        "hybrid_reranker_hit@1": hit_at(cc_hyb_ranks, cc_pairs_for_ir, 1),
        "hybrid_reranker_hit@3": hit_at(cc_hyb_ranks, cc_pairs_for_ir, 3),
        "hybrid_reranker_mrr": mean_reciprocal_rank(cc_hyb_ranks, cc_pairs_for_ir),
    }
    if neural_verifier is not None:
        cc_ir_dict.update(
            {
                "neural_nli_hit@1": hit_at(cc_neu_ranks, cc_pairs_for_ir, 1),
                "neural_nli_hit@3": hit_at(cc_neu_ranks, cc_pairs_for_ir, 3),
                "neural_nli_mrr": mean_reciprocal_rank(cc_neu_ranks, cc_pairs_for_ir),
            }
        )
    benchmarks["claimcheck"] = {
        "dataset_id": "claimcheck",
        "data_source": data_source,
        "spdx_license": cc_spec.spdx_license,
        "commercial_use_ok": cc_spec.commercial_use_ok,
        "eval_only": cc_spec.eval_only,
        "sha256": file_digests["claimcheck"],
        "critique_rationale_retrieval": cc_ir_dict,
        "stance_verification": cc_stance,
    }

    # 6. SciVer (`chengyewang/SciVer` leak-free figure/table claim verification)
    sv_spec = get_dataset_spec("sciver")
    sv_eval = _evaluate_candidate_verification_records(
        verifier,
        sv_items,
        eval_labels=("SUPPORT", "CONTRADICT"),
        neural_verifier=neural_verifier,
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

    # 7. ChemBench (`jablonkagroup/ChemBench`, option string matching against task text)
    chem_spec = get_dataset_spec("chembench")
    chem_mcq_items: list[dict[str, Any]] = []
    for r in chem_rows:
        _ = adapt_record("chembench", r, purpose="eval")
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
    chem_metrics = _evaluate_mcq_rationale_discrimination(
        verifier,
        chem_mcq_items,
        evaluation_mode="option string matching against task text",
        neural_verifier=neural_verifier,
    )
    benchmarks["chembench"] = {
        "dataset_id": "chembench",
        "data_source": data_source,
        "spdx_license": chem_spec.spdx_license,
        "commercial_use_ok": chem_spec.commercial_use_ok,
        "eval_only": chem_spec.eval_only,
        "eval_only_guard_verified": guard_checks["chembench"],
        "evaluation_mode": "option string matching against task text",
        "sha256_general": file_digests["chembench_general"],
        "sha256_analytical": file_digests["chembench_analytical"],
        "total_raw_tasks": len(chem_rows),
        "mcq_rationale_discrimination": chem_metrics,
    }

    # 8. LitQA2 (`futurehouse/lab-bench` `LitQA2`, with gold passage)
    lq_spec = get_dataset_spec("litqa2")
    lq_mcq_items: list[dict[str, Any]] = []
    for r in litqa2_rows:
        _ = adapt_record("litqa2", r, purpose="eval")
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
    lq_metrics = _evaluate_mcq_rationale_discrimination(
        verifier,
        lq_mcq_items,
        evaluation_mode="with gold passage",
        neural_verifier=neural_verifier,
    )
    benchmarks["litqa2"] = {
        "dataset_id": "litqa2",
        "data_source": data_source,
        "spdx_license": lq_spec.spdx_license,
        "commercial_use_ok": lq_spec.commercial_use_ok,
        "eval_only": lq_spec.eval_only,
        "eval_only_guard_verified": guard_checks["litqa2"],
        "evaluation_mode": "with gold passage",
        "sha256": file_digests["litqa2"],
        "total_raw_tasks": len(litqa2_rows),
        "mcq_rationale_discrimination": lq_metrics,
    }

    elapsed = round(time.time() - t_start, 2)
    out_payload: dict[str, Any] = {
        "schema_version": "lab-ask-why-open-benchmarks/v2",
        "data_source": data_source,
        "num_datasets_evaluated": len(benchmarks),
        "neural_verifier_evaluated": neural_verifier is not None,
        "elapsed_seconds": elapsed,
        "leakage_audit": {
            "verifier_trained_strictly_on": "scifact_train_809_claims",
            "scifact_train_fit_calib_split": [607, 202],
            "scifact_train_dev_claim_id_overlap": 0,
            "scifact_dev_exclusive_docs_excluded_from_idf_and_svd": len(dev_only_doc_ids),
            "pubmedqa_long_answer_excluded": True,
            "sciver_verdict_boilerplate_stripped_and_exact_copy_excluded": True,
            "conflict_negative_controls_evaluated": True,
            "eval_only_guards_verified": guard_checks,
        },
        "training_provenance": verifier.training_provenance,
        "benchmarks": benchmarks,
    }
    if neural_verifier is not None:
        out_payload["neural_training_provenance"] = neural_verifier.training_provenance
    return out_payload


def verify_scifact_drift_against_committed(
    live_results: Mapping[str, Any],
    committed_path: Path,
    *,
    tol: float = 1e-6,
) -> None:
    """Assert live SciFact `q_hat_*`, ranking, and conformal metrics match committed JSON within `tol`."""
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    live_cal = live_results["training_provenance"]["conformal_calibration"]
    comm_cal = committed["training_provenance"]["conformal_calibration"]
    for k, exp_v in comm_cal.items():
        got_v = float(live_cal[k])
        if abs(got_v - float(exp_v)) > tol:
            raise AssertionError(
                f"SciFact conformal {k} drift: live={got_v:.9f} vs committed={float(exp_v):.9f}"
            )

    live_dev = live_results["benchmarks"]["scifact"]["dev_split"]
    comm_dev = committed["benchmarks"]["scifact"]["dev_split"]

    for m_key in ("hit@1", "hit@3", "mrr"):
        lv = float(live_dev["oracle_rationale_ranking"]["hybrid_reranker"][m_key])
        cv = float(comm_dev["oracle_rationale_ranking"]["hybrid_reranker"][m_key])
        if abs(lv - cv) > tol:
            raise AssertionError(f"SciFact dev hybrid {m_key} drift: live={lv:.9f} vs committed={cv:.9f}")

    for c_key in ("empirical_coverage", "avg_set_size", "single_label_set_rate"):
        lv = float(live_dev["split_conformal_90"][c_key])
        cv = float(comm_dev["split_conformal_90"][c_key])
        if abs(lv - cv) > tol:
            raise AssertionError(f"SciFact dev conformal {c_key} drift: live={lv:.9f} vs committed={cv:.9f}")


def format_cli_report(results: Mapping[str, Any]) -> str:
    """Format multi-dataset results where EVERY CLI line explicitly carries `data_source`."""
    ds_tag = f"[data_source={results['data_source']}]"
    b = results["benchmarks"]
    sf_dev = b["scifact"]["dev_split"]
    sf_or = sf_dev["oracle_rationale_ranking"]
    sf_st = sf_dev["stance_verification_3way"]
    sf_cp = sf_dev["split_conformal_90"]
    sf_oc = sf_dev["open_corpus_retrieval"]
    sf_off = sf_dev["official_scifact_metrics"]["hybrid_reranker"]

    lines = [
        f"{ds_tag} lab-ask-why 2026 Leak-Free Benchmark Suite (datasets={results['num_datasets_evaluated']}, neural={results.get('neural_verifier_evaluated', False)}, elapsed={results['elapsed_seconds']:.1f}s)",
        f"{ds_tag} scifact (dev, n={sf_dev['total_claims']} claims / {sf_dev['claim_doc_pairs']} pairs): Random Hit@1={sf_or['random_20_seeds']['hit@1']:.3f} | BM25 Hit@1={sf_or['bm25']['hit@1']:.3f} Hit@3={sf_or['bm25']['hit@3']:.3f} | SparseHybrid Hit@1={sf_or['hybrid_reranker']['hit@1']:.3f} Hit@3={sf_or['hybrid_reranker']['hit@3']:.3f} MRR={sf_or['hybrid_reranker']['mrr']:.3f}",
        f"{ds_tag} scifact (official Wadden2020 & conformal): AbsLabelF1={sf_off['abstract_label_only']['f1']:.3f} SentLabelF1={sf_off['sentence_label']['f1']:.3f} | doc_rec@5={sf_oc['hybrid_rrf_doc_recall@5']:.3f} | 3wayF1={sf_st['hybrid_reranker_macro_f1']:.3f} | cov@90={sf_cp['empirical_coverage']:.3f} size={sf_cp['avg_set_size']:.2f} single={sf_cp['single_label_set_rate']:.3f}",
    ]
    if "neural_biencoder_nli" in sf_dev:
        nd = sf_dev["neural_biencoder_nli"]
        noff = nd["official_scifact_metrics"]
        lines.append(
            f"{ds_tag} scifact (Neural BGE+DeBERTa-v3 NLI): Hit@1={nd['oracle_rationale_ranking']['neural_nli_reranker']['hit@1']:.3f} Hit@3={nd['oracle_rationale_ranking']['neural_nli_reranker']['hit@3']:.3f} MRR={nd['oracle_rationale_ranking']['neural_nli_reranker']['mrr']:.3f} | AbsLabelF1={noff['abstract_label_only']['f1']:.3f} SentLabelF1={noff['sentence_label']['f1']:.3f} | 3wayF1={nd['stance_verification_3way']['neural_nli_macro_f1']:.3f} | cov@90={nd['split_conformal_90']['empirical_coverage']:.3f} size={nd['split_conformal_90']['avg_set_size']:.2f}"
        )

    if results["num_datasets_evaluated"] == 1:
        return "\n".join(lines)

    cb_cd = b["conflictbank"]["conflict_detection"]
    fe_cd = b["faitheval_inconsistent"]["conflict_detection"]
    lines.extend(
        [
            f"{ds_tag} pubmedqa (n={b['pubmedqa']['three_way_yes_no_maybe']['n_evaluated']}): MajAcc={b['pubmedqa']['three_way_yes_no_maybe']['majority_class_accuracy']:.3f} | Yes/No F1 BM25={b['pubmedqa']['binary_yes_no']['bm25_macro_f1']:.3f} SparseHybrid={b['pubmedqa']['binary_yes_no']['hybrid_reranker_macro_f1']:.3f} | cov@90={b['pubmedqa']['three_way_yes_no_maybe']['conformal_coverage_90']:.3f}",
            f"{ds_tag} conflictbank (CB_qa pos={cb_cd['hybrid_bilateral']['n_positive']} neg={cb_cd['hybrid_bilateral']['n_negative_control']}): SparseHybrid TPR={cb_cd['true_positive_rate']:.3f} FPR={cb_cd['false_positive_rate']:.3f} AUROC={cb_cd['conflict_auroc']:.3f} | pairs(n={b['conflictbank']['claim_evidence_verification']['n_evaluated']}) F1={b['conflictbank']['claim_evidence_verification']['hybrid_reranker_macro_f1']:.3f}",
            f"{ds_tag} faitheval_inconsistent (pos={fe_cd['hybrid_bilateral']['n_positive']} neg={fe_cd['hybrid_bilateral']['n_negative_control']}): SparseHybrid TPR={fe_cd['true_positive_rate']:.3f} FPR={fe_cd['false_positive_rate']:.3f} AUROC={fe_cd['conflict_auroc']:.3f}",
            f"{ds_tag} claimcheck (n={b['claimcheck']['critique_rationale_retrieval']['n_pairs']}): critique Hit@1 BM25={b['claimcheck']['critique_rationale_retrieval']['bm25_hit@1']:.3f} SparseHybrid={b['claimcheck']['critique_rationale_retrieval']['hybrid_reranker_hit@1']:.3f} | stance MajAcc={b['claimcheck']['stance_verification']['majority_class_accuracy']:.3f}",
            f"{ds_tag} sciver (n={b['sciver']['stance_verification']['n_evaluated']}): stance MajAcc={b['sciver']['stance_verification']['majority_class_accuracy']:.3f} BM25_F1={b['sciver']['stance_verification']['bm25_macro_f1']:.3f} SparseHybrid_F1={b['sciver']['stance_verification']['hybrid_reranker_macro_f1']:.3f}",
            f"{ds_tag} chembench ({b['chembench']['evaluation_mode']}, n={b['chembench']['mcq_rationale_discrimination']['n_evaluated']}): Chance={b['chembench']['mcq_rationale_discrimination']['chance_hit@1']:.3f} BM25={b['chembench']['mcq_rationale_discrimination']['bm25_option_hit@1']:.3f} SparseHybrid={b['chembench']['mcq_rationale_discrimination']['hybrid_reranker_option_hit@1']:.3f}",
            f"{ds_tag} litqa2 ({b['litqa2']['evaluation_mode']}, n={b['litqa2']['mcq_rationale_discrimination']['n_evaluated']}): Chance={b['litqa2']['mcq_rationale_discrimination']['chance_hit@1']:.3f} BM25={b['litqa2']['mcq_rationale_discrimination']['bm25_option_hit@1']:.3f} SparseHybrid={b['litqa2']['mcq_rationale_discrimination']['hybrid_reranker_option_hit@1']:.3f}",
        ]
    )
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
        help="Destination path for JSON benchmark report (pass empty string '' to skip writing).",
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
    p.add_argument(
        "--scifact-only",
        action="store_true",
        help="Run only SciFact fit & dev evaluation (used by CI real-data drift & RSS check).",
    )
    p.add_argument(
        "--pure-numpy",
        action="store_true",
        help="Force pure-NumPy CSR linear algebra path even if scipy is installed.",
    )
    p.add_argument(
        "--max-rss-mb",
        type=float,
        default=None,
        help="Assert process peak RSS stays strictly below this ceiling in MiB.",
    )
    p.add_argument(
        "--check-scifact-drift",
        type=str,
        default=None,
        help="Compare live SciFact q_hat, Hit@1/3, MRR, and conformal stats against committed JSON.",
    )
    p.add_argument(
        "--no-neural",
        action="store_true",
        help="Skip pretrained Neural Bi-Encoder + DeBERTa-v3 NLI evaluation.",
    )
    p.add_argument(
        "--save-wire-calibrator",
        type=str,
        default=None,
        help="Optional path to write compact SciFact wire calibrator JSON.",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON report to stdout.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)

    cal_out = (
        Path(args.save_wire_calibrator)
        if args.save_wire_calibrator
        else (
            DEFAULT_WIRE_CALIBRATOR_PATH
            if (args.output and Path(args.output).resolve() == DEFAULT_RESULTS_PATH.resolve() and not args.scifact_only)
            else None
        )
    )
    enable_neural = False if (args.no_neural or args.scifact_only) else None

    try:
        results = run_benchmark_suite(
            data_root=Path(args.data_root),
            offline=bool(args.offline),
            allow_synthetic=bool(args.allow_synthetic),
            verify_pinned_sha256=not bool(args.no_verify_pinned_sha256),
            scifact_only=bool(args.scifact_only),
            pure_numpy=bool(args.pure_numpy),
            enable_neural=enable_neural,
            save_wire_calibrator_path=cal_out,
        )
        if args.max_rss_mb is not None:
            peak_mb = measure_peak_rss_mb()
            if peak_mb > float(args.max_rss_mb):
                raise RuntimeError(
                    f"Peak RSS {peak_mb:.1f} MiB exceeded --max-rss-mb {float(args.max_rss_mb):.1f} MiB"
                )
        if args.check_scifact_drift:
            verify_scifact_drift_against_committed(results, Path(args.check_scifact_drift))
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        PermissionError,
        AssertionError,
        urllib.error.URLError,
    ) as exc:
        print(
            f"[data_source=missing] Error: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.output and not args.check_scifact_drift:
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
