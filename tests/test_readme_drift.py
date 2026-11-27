"""README and ARCHITECTURE_AND_HISTORY drift guard against results/open_benchmarks_2026.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "open_benchmarks_2026.json"
README_PATH = REPO_ROOT / "README.md"
ARCH_PATH = REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md"
DEMO_OUTPUT_PATH = REPO_ROOT / "demo" / "output.txt"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def _extract_marked_block(doc_text: str, marker_name: str) -> str:
    start_tag = f"<!-- {marker_name}_START -->"
    end_tag = f"<!-- {marker_name}_END -->"
    assert start_tag in doc_text, f"Missing {start_tag}"
    assert end_tag in doc_text, f"Missing {end_tag}"
    s_idx = doc_text.index(start_tag) + len(start_tag)
    e_idx = doc_text.index(end_tag)
    return doc_text[s_idx:e_idx].strip()


def render_expected_scifact_table(data: dict) -> str:
    sf = data["benchmarks"]["scifact"]["dev_split"]
    rk = sf["oracle_rationale_ranking"]
    st = sf["stance_verification_3way"]
    oc = sf["open_corpus_retrieval"]
    cp = sf["split_conformal_90"]

    rnd = rk["random_20_seeds"]
    bm = rk["bm25"]
    svd = rk["dense_svd"]
    rrf = rk["stage1_hybrid_rrf"]
    hyb = rk["hybrid_reranker"]

    lines = [
        "| Method (`SciFact dev`, `n=300` claims / `209` pairs) | Hit@1 | Hit@3 | MRR | Set-Recall@3 | Open `doc_recall@5` | 3-Way Stance Macro-F1 | 90% Conformal Coverage (Avg Size) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `Random` (20 seeds) | `{rnd['hit@1']:.3f}` | `{rnd['hit@3']:.3f}` | `{rnd['mrr']:.3f}` | -- | -- | -- | -- |",
        f"| `Okapi BM25` (zero-param lexical) | `{bm['hit@1']:.3f}` | `{bm['hit@3']:.3f}` | `{bm['mrr']:.3f}` | `{bm['rationale_set_recall@3']:.3f}` | `{oc['bm25_doc_recall@5']:.3f}` | `{st['bm25_macro_f1']:.3f}` | -- |",
        f"| `Dense-SVD` (`LSA` char/word TF-IDF) | `{svd['hit@1']:.3f}` | `{svd['hit@3']:.3f}` | `{svd['mrr']:.3f}` | `{svd['rationale_set_recall@3']:.3f}` | `{oc['dense_svd_doc_recall@5']:.3f}` | `{st['dense_svd_macro_f1']:.3f}` | -- |",
        f"| `Stage-1 Hybrid RRF` (`BM25+SVD+Entity`) | `{rrf['hit@1']:.3f}` | `{rrf['hit@3']:.3f}` | `{rrf['mrr']:.3f}` | `{rrf['rationale_set_recall@3']:.3f}` | `{oc['hybrid_rrf_doc_recall@5']:.3f}` | -- | -- |",
        f"| **`Hybrid + Cross-Feature Reranker`** (`SciFact train`) | **`{hyb['hit@1']:.3f}`** | **`{hyb['hit@3']:.3f}`** | **`{hyb['mrr']:.3f}`** | **`{hyb['rationale_set_recall@3']:.3f}`** | **`{oc['hybrid_rrf_doc_recall@5']:.3f}`** | **`{st['hybrid_reranker_macro_f1']:.3f}`** | **`{cp['empirical_coverage']:.3f}`** (`{cp['avg_set_size']:.2f}`) |",
    ]
    return "\n".join(lines)


def render_expected_multi_dataset_table(data: dict) -> str:
    b = data["benchmarks"]
    sf_dev = b["scifact"]["dev_split"]
    pqa_bin = b["pubmedqa"]["binary_yes_no"]
    pqa_3w = b["pubmedqa"]["three_way_yes_no_maybe"]
    cb = b["conflictbank"]
    fe = b["faitheval_inconsistent"]
    cc_ir = b["claimcheck"]["critique_rationale_retrieval"]
    cc_st = b["claimcheck"]["stance_verification"]
    sv = b["sciver"]["stance_verification"]
    chem = b["chembench"]["mcq_rationale_discrimination"]
    lq = b["litqa2"]["mcq_rationale_discrimination"]

    lines = [
        "| Dataset (`data_source`) | Evaluated ($n$) | License (`eval_only`) | Baseline (`BM25`) | Frontier (`Hybrid+Reranker`) | Conformal / Conflict Headline |",
        "|---|---:|---|---|---|---|",
        f"| `SciFact` (`{b['scifact']['data_source']}`) | `{sf_dev['total_claims']}` claims (`{sf_dev['claim_doc_pairs']}` pairs) | `{b['scifact']['spdx_license']}` (`False`) | Hit@1 `{sf_dev['oracle_rationale_ranking']['bm25']['hit@1']:.3f}`, Hit@3 `{sf_dev['oracle_rationale_ranking']['bm25']['hit@3']:.3f}` | **Hit@1 `{sf_dev['oracle_rationale_ranking']['hybrid_reranker']['hit@1']:.3f}`, Hit@3 `{sf_dev['oracle_rationale_ranking']['hybrid_reranker']['hit@3']:.3f}`, MRR `{sf_dev['oracle_rationale_ranking']['hybrid_reranker']['mrr']:.3f}`** | 90% set coverage **`{sf_dev['split_conformal_90']['empirical_coverage']:.3f}`** (size `{sf_dev['split_conformal_90']['avg_set_size']:.2f}`) |",
        f"| `PubMedQA` (`{b['pubmedqa']['data_source']}`) | `{pqa_3w['n_evaluated']}` (`pqa_labeled`) | `{b['pubmedqa']['spdx_license']}` (`False`) | Yes/No Macro-F1 `{pqa_bin['bm25_macro_f1']:.3f}` | **Yes/No Macro-F1 `{pqa_bin['hybrid_reranker_macro_f1']:.3f}`** | 3-way 90% coverage **`{pqa_3w['conformal_coverage_90']:.3f}`** (`LONG_ANSWER` excluded) |",
        f"| `ConflictBank` (`{cb['data_source']}`) | `{cb['n_conflicting_qa_contexts']}` QA + `{cb['claim_evidence_verification']['n_evaluated']}` pairs | `{cb['spdx_license']}` (`False`) | Conflict detect `{cb['bm25_contested_detection_rate']:.3f}` (binary `0.000`) | **Conflict detect `{cb['hybrid_conformal_conflict_detection_rate']:.3f}`**, stance F1 **`{cb['claim_evidence_verification']['hybrid_reranker_macro_f1']:.3f}`** | Multi-label `{{SUPPORT, CONTRADICT}}` conflict sets |",
        f"| `FaithEval-Inconsistent` (`{fe['data_source']}`) | `{fe['n_evaluated']}` multi-doc | `{fe['spdx_license']}` (`False`) | Polarity conflict `{fe['bm25_polarity_conflict_detection_rate']:.3f}` (forced `0.000`) | **Multi-answer conflict detect `{fe['hybrid_multi_answer_conflict_detection_rate']:.3f}`** | Opposing-document grounding + Socratic crux |",
        f"| `CLAIMCHECK` (`{b['claimcheck']['data_source']}`) | `{cc_ir['n_pairs']}` critiques | `{b['claimcheck']['spdx_license']}` (`False`) | Critique Hit@1 `{cc_ir['bm25_hit@1']:.3f}`, MRR `{cc_ir['bm25_mrr']:.3f}` | **Critique Hit@1 `{cc_ir['hybrid_reranker_hit@1']:.3f}`, Hit@3 `{cc_ir['hybrid_reranker_hit@3']:.3f}`, MRR `{cc_ir['hybrid_reranker_mrr']:.3f}`** | 90% coverage **`{cc_st['conformal_coverage_90']:.3f}`** |",
        f"| `SciVer` (`{b['sciver']['data_source']}`) | `{sv['n_evaluated']}` claims | `{b['sciver']['spdx_license']}` (`False`) | Stance Macro-F1 `{sv['bm25_macro_f1']:.3f}` | **Stance Macro-F1 `{sv['hybrid_reranker_macro_f1']:.3f}`** (acc `{sv['hybrid_reranker_accuracy']:.3f}`) | 90% coverage **`{sv['conformal_coverage_90']:.3f}`** (leak-free evidence) |",
        f"| `ChemBench` (`{b['chembench']['data_source']}`) | `{chem['n_evaluated']}` MCQ (`{b['chembench']['total_raw_tasks']}` raw) | `{b['chembench']['spdx_license']}` (**`True`**) | Option Hit@1 `{chem['bm25_option_hit@1']:.3f}`, MRR `{chem['bm25_option_mrr']:.3f}` | **Option Hit@1 `{chem['hybrid_reranker_option_hit@1']:.3f}`, MRR `{chem['hybrid_reranker_option_mrr']:.3f}`** | Canary-guarded `eval_only=True` verified |",
        f"| `LitQA2` (`{b['litqa2']['data_source']}`) | `{lq['n_evaluated']}` MCQ (`{b['litqa2']['total_raw_tasks']}` raw) | `{b['litqa2']['spdx_license']}` (**`True`**) | Option Hit@1 `{lq['bm25_option_hit@1']:.3f}`, MRR `{lq['bm25_option_mrr']:.3f}` | **Option Hit@1 `{lq['hybrid_reranker_option_hit@1']:.3f}`, MRR `{lq['hybrid_reranker_option_mrr']:.3f}`** | Canary-guarded `eval_only=True` verified |",
    ]
    return "\n".join(lines)


def test_committed_open_benchmarks_json_integrity_and_sha256() -> None:
    assert RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}"
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert data["data_source"] == "real_downloaded"
    assert data["num_datasets_evaluated"] == 8
    assert data["leakage_audit"]["verifier_trained_strictly_on"] == "scifact_train_809_claims"
    assert data["leakage_audit"]["scifact_train_dev_claim_id_overlap"] == 0
    assert data["leakage_audit"]["pubmedqa_long_answer_excluded"] is True
    assert data["leakage_audit"]["eval_only_guards_verified"]["chembench"] is True
    assert data["leakage_audit"]["eval_only_guards_verified"]["litqa2"] is True

    expected_datasets = {
        "scifact",
        "pubmedqa",
        "conflictbank",
        "faitheval_inconsistent",
        "claimcheck",
        "sciver",
        "chembench",
        "litqa2",
    }
    assert set(data["benchmarks"].keys()) == expected_datasets

    for ds_id, entry in data["benchmarks"].items():
        assert entry["data_source"] == "real_downloaded", f"{ds_id} is not real_downloaded"
        sha_fields = [v for k, v in entry.items() if k.startswith("sha256")]
        assert sha_fields, f"{ds_id} missing sha256 metadata"
        for digest in sha_fields:
            assert SHA256_RE.match(str(digest)), f"Invalid sha256 {digest!r} on {ds_id}"


def test_readme_and_architecture_docs_match_benchmark_json_exactly() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    readme_text = README_PATH.read_text(encoding="utf-8")
    arch_text = ARCH_PATH.read_text(encoding="utf-8")
    demo_text = DEMO_OUTPUT_PATH.read_text(encoding="utf-8")

    expected_scifact = render_expected_scifact_table(data)
    expected_multi = render_expected_multi_dataset_table(data)

    for doc_label, doc_text in (("README.md", readme_text), ("ARCHITECTURE_AND_HISTORY.md", arch_text)):
        assert _extract_marked_block(doc_text, "SCIFACT_FRONTIER_TABLE") == expected_scifact, (
            f"SCIFACT_FRONTIER_TABLE drift in {doc_label}"
        )
        assert _extract_marked_block(doc_text, "OPEN_BENCHMARKS_2026_TABLE") == expected_multi, (
            f"OPEN_BENCHMARKS_2026_TABLE drift in {doc_label}"
        )

    # Verify prose headline numbers in README.md match JSON & demo/output.txt
    sf_hyb = data["benchmarks"]["scifact"]["dev_split"]["oracle_rationale_ranking"]["hybrid_reranker"]
    sf_cp = data["benchmarks"]["scifact"]["dev_split"]["split_conformal_90"]
    assert f"{sf_hyb['hit@1']:.3f}" in readme_text
    assert f"{sf_hyb['hit@3']:.3f}" in readme_text
    assert f"{sf_hyb['mrr']:.3f}" in readme_text
    assert f"{sf_cp['empirical_coverage']:.3f}" in readme_text
    assert "0.483" in demo_text and "0.483" in readme_text
    assert "0.813" in demo_text and "0.813" in readme_text

    # Zero CJK characters across all tracked documentation, code, Lean, and JSON files
    for candidate in (
        *REPO_ROOT.glob("src/**/*.py"),
        *REPO_ROOT.glob("scripts/**/*.py"),
        *REPO_ROOT.glob("tests/**/*.py"),
        *REPO_ROOT.glob("community_extensions/**/*.py"),
        *REPO_ROOT.glob("lean/**/*"),
        README_PATH,
        ARCH_PATH,
        RESULTS_PATH,
        REPO_ROOT / "autolab_wire.py",
    ):
        if candidate.is_file() and candidate.suffix in {".py", ".md", ".json", ".lean", ".toml"}:
            txt = candidate.read_text(encoding="utf-8")
            assert CJK_RE.search(txt) is None, f"CJK character found in {candidate}"
