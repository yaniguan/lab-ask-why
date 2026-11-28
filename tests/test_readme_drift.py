"""README and ARCHITECTURE_AND_HISTORY drift & honest-bolding guard against results/open_benchmarks_2026.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "open_benchmarks_2026.json"
README_PATH = REPO_ROOT / "README.md"
ARCH_PATH = REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md"
DEMO_OUTPUT_PATH = REPO_ROOT / "demo" / "output.txt"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_NUM_IN_CELL_RE = re.compile(r"(\*\*)?`?([0-9]+\.[0-9]+)`?(\*\*)?")


def _extract_marked_block(doc_text: str, marker_name: str) -> str:
    start_tag = f"<!-- {marker_name}_START -->"
    end_tag = f"<!-- {marker_name}_END -->"
    assert start_tag in doc_text, f"Missing {start_tag}"
    assert end_tag in doc_text, f"Missing {end_tag}"
    s_idx = doc_text.index(start_tag) + len(start_tag)
    e_idx = doc_text.index(end_tag)
    return doc_text[s_idx:e_idx].strip()


def _format_competing_cells(
    trivial_val: float | None,
    bm25_val: float | None,
    competitors: Sequence[tuple[str, float | None]],
    *,
    trivial_note: str = "",
) -> tuple[str, str, list[str]]:
    """Dynamically bold a model cell ONLY if it strictly beats Trivial + BM25 and is best in row."""
    t_round = round(trivial_val, 3) if trivial_val is not None else None
    b_round = round(bm25_val, 3) if bm25_val is not None else None
    comp_rounds = [(name, (round(v, 3) if v is not None else None)) for name, v in competitors]

    present_vals = [
        x
        for x in [t_round, b_round, *[rv for _, rv in comp_rounds]]
        if x is not None
    ]
    row_max = max(present_vals) if present_vals else -1.0

    t_str = (
        f"`{t_round:.3f}`" + (f" ({trivial_note})" if trivial_note else "")
        if t_round is not None
        else "--"
    )
    if b_round is None:
        b_str = "--"
    elif (
        b_round >= row_max - 1e-9
        and (t_round is None or b_round > t_round + 1e-9)
        and all(rv is None or b_round > rv + 1e-9 for _, rv in comp_rounds)
    ):
        b_str = f"**`{b_round:.3f}`**"
    else:
        b_str = f"`{b_round:.3f}`"

    out_comp: list[str] = []
    for _, rv in comp_rounds:
        if rv is None:
            out_comp.append("--")
            continue
        beats_trivial = t_round is None or rv > t_round + 1e-9
        beats_bm25 = b_round is None or rv > b_round + 1e-9
        is_row_best = rv >= row_max - 1e-9
        if beats_trivial and beats_bm25 and is_row_best:
            out_comp.append(f"**`{rv:.3f}`**")
        else:
            out_comp.append(f"`{rv:.3f}`")

    return t_str, b_str, out_comp


def _fmt_conformal_summary(
    cov: float,
    size: float,
    single_rate: float,
    detail_by_cls: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
) -> str:
    short = {"SUPPORT": "S", "CONTRADICT": "C", "NEI": "N"}
    cls_parts: list[str] = []
    for lab in labels:
        d = detail_by_cls.get(lab)
        if not d or int(d.get("n", 0)) <= 0:
            continue
        lo, hi = d["wilson_95_ci"]
        cls_parts.append(f"{short.get(lab, lab)} `{d['coverage']:.2f}` [`{lo:.2f}`,`{hi:.2f}`]")
    joined = ", ".join(cls_parts)
    return (
        f"cov `{cov:.3f}` (size `{size:.2f}`, 1-lbl `{single_rate:.2f}`; {joined})"
    )


def assert_no_undeserved_bolding_in_table(
    table_markdown: str,
    *,
    numeric_col_indices: Sequence[int],
    baseline_col_indices: Sequence[int],
) -> None:
    """Parse a Markdown table and fail if any bolded model cell does not beat baselines or row max."""
    rows = [
        line.strip()
        for line in table_markdown.strip().splitlines()
        if line.strip().startswith("|")
    ]
    assert len(rows) >= 3, "Expected Markdown table with header, separator, and data rows"
    for row_line in rows[2:]:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        parsed_vals: dict[int, tuple[float, bool]] = {}
        for col_idx in numeric_col_indices:
            cell = cells[col_idx]
            if cell == "--":
                continue
            is_bold = "**" in cell
            m = _NUM_IN_CELL_RE.search(cell)
            assert m is not None, f"Could not parse numeric value in cell {cell!r} of row {row_line}"
            val = float(m.group(2))
            parsed_vals[col_idx] = (val, is_bold)

        if not parsed_vals:
            continue
        row_max = max(v for v, _ in parsed_vals.values())
        baseline_max = max(
            (parsed_vals[bi][0] for bi in baseline_col_indices if bi in parsed_vals),
            default=-1.0,
        )

        for col_idx, (val, is_bold) in parsed_vals.items():
            if not is_bold:
                continue
            assert val >= row_max - 1e-9, (
                f"Bolded cell in col {col_idx} ({val:.3f}) is below row max {row_max:.3f}: {row_line}"
            )
            if col_idx not in baseline_col_indices:
                assert val > baseline_max + 1e-9, (
                    f"Bolded model cell in col {col_idx} ({val:.3f}) fails to beat baseline max "
                    f"{baseline_max:.3f}: {row_line}"
                )


def render_expected_scifact_table(data: Mapping[str, Any]) -> str:
    sf = data["benchmarks"]["scifact"]["dev_split"]
    rk = sf["oracle_rationale_ranking"]
    st = sf["stance_verification_3way"]
    oc = sf["open_corpus_retrieval"]
    off = sf["official_scifact_metrics"]
    cp = sf["split_conformal_90"]
    cand_cp = cp["candidate_abstract_conformal_90"]
    nd = sf.get("neural_biencoder_nli")

    rnd = rk["random_20_seeds"]
    bm = rk["bm25"]
    svd = rk["dense_svd"]
    hyb = rk["hybrid_reranker"]

    n_rk = nd["oracle_rationale_ranking"]["neural_nli_reranker"] if nd else None
    n_oc = nd["open_corpus_retrieval"] if nd else None
    n_off = nd["official_scifact_metrics"] if nd else None
    n_st = nd["stance_verification_3way"] if nd else None
    n_cp = nd["split_conformal_90"] if nd else None

    metric_specs: list[tuple[str, float | None, str, float | None, float | None, float | None, float | None]] = [
        (
            "Oracle Rationale `Hit@1` (`209` pairs)",
            rnd["hit@1"],
            "random",
            bm["hit@1"],
            svd["hit@1"],
            hyb["hit@1"],
            n_rk["hit@1"] if n_rk else None,
        ),
        (
            "Oracle Rationale `Hit@3` (`209` pairs)",
            rnd["hit@3"],
            "random",
            bm["hit@3"],
            svd["hit@3"],
            hyb["hit@3"],
            n_rk["hit@3"] if n_rk else None,
        ),
        (
            "Oracle Rationale `MRR` (`209` pairs)",
            rnd["mrr"],
            "random",
            bm["mrr"],
            svd["mrr"],
            hyb["mrr"],
            n_rk["mrr"] if n_rk else None,
        ),
        (
            "Oracle `Set-Recall@3` (`209` pairs)",
            None,
            "",
            bm["rationale_set_recall@3"],
            svd["rationale_set_recall@3"],
            hyb["rationale_set_recall@3"],
            n_rk["rationale_set_recall@3"] if n_rk else None,
        ),
        (
            "Open-Corpus `doc_recall@5` (`5,183` docs)",
            5.0 / 5183.0,
            "random",
            oc["bm25_doc_recall@5"],
            oc["dense_svd_doc_recall@5"],
            oc["hybrid_rrf_doc_recall@5"],
            n_oc["neural_hybrid_doc_recall@5"] if n_oc else None,
        ),
        (
            "Official Abstract `Label-Only F1` (`300` claims)",
            0.0,
            "abstain",
            off["bm25"]["abstract_label_only"]["f1"],
            None,
            off["hybrid_reranker"]["abstract_label_only"]["f1"],
            n_off["abstract_label_only"]["f1"] if n_off else None,
        ),
        (
            "Official Abstract `Label+Rationales F1` (`300` claims)",
            0.0,
            "abstain",
            off["bm25"]["abstract_rationales"]["f1"],
            None,
            off["hybrid_reranker"]["abstract_rationales"]["f1"],
            n_off["abstract_rationales"]["f1"] if n_off else None,
        ),
        (
            "Official Sentence `Selection-Only F1` (`300` claims)",
            0.0,
            "abstain",
            off["bm25"]["sentence_selection"]["f1"],
            None,
            off["hybrid_reranker"]["sentence_selection"]["f1"],
            n_off["sentence_selection"]["f1"] if n_off else None,
        ),
        (
            "Official Sentence `Selection+Label F1` (`300` claims)",
            0.0,
            "abstain",
            off["bm25"]["sentence_label"]["f1"],
            None,
            off["hybrid_reranker"]["sentence_label"]["f1"],
            n_off["sentence_label"]["f1"] if n_off else None,
        ),
        (
            "3-Way Stance `Accuracy` (`300` claims)",
            st["majority_class_accuracy"],
            f"always {st['majority_label']}",
            st["bm25_accuracy"],
            None,
            st["hybrid_reranker_accuracy"],
            n_st["neural_nli_accuracy"] if n_st else None,
        ),
        (
            "3-Way Stance `Macro-F1` (`300` claims)",
            st["majority_class_macro_f1"],
            f"always {st['majority_label']}",
            st["bm25_macro_f1"],
            st["dense_svd_macro_f1"],
            st["hybrid_reranker_macro_f1"],
            n_st["neural_nli_macro_f1"] if n_st else None,
        ),
    ]

    lines = [
        "| Metric (`SciFact dev`, `n=300` claims / `209` pairs) | Best Trivial Baseline | `Okapi BM25` | `Dense-SVD` | `Sparse Hybrid` (`numpy`) | `Neural BGE + DeBERTa-v3 NLI` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, triv, t_note, bm_v, svd_v, sp_v, neu_v in metric_specs:
        t_cell, b_cell, comp_cells = _format_competing_cells(
            triv,
            bm_v,
            [("svd", svd_v), ("sparse", sp_v), ("neural", neu_v)],
            trivial_note=t_note,
        )
        lines.append(
            f"| {label} | {t_cell} | {b_cell} | {comp_cells[0]} | {comp_cells[1]} | {comp_cells[2]} |"
        )

    sp_open_conf = _fmt_conformal_summary(
        cp["empirical_coverage"],
        cp["avg_set_size"],
        cp["single_label_set_rate"],
        cp["coverage_by_class_detail"],
        ("SUPPORT", "CONTRADICT", "NEI"),
    )
    sp_cand_conf = _fmt_conformal_summary(
        cand_cp["empirical_coverage"],
        cand_cp["avg_set_size"],
        cand_cp["single_label_set_rate"],
        cand_cp["coverage_by_class_detail"],
        ("SUPPORT", "CONTRADICT", "NEI"),
    )
    lines.append("")
    lines.append(
        f"- **Sparse Hybrid 90% Mondrian Conformal (`SciFact dev` open-corpus, `n=300`)**: {sp_open_conf}"
    )
    lines.append(
        f"- **Sparse Hybrid 90% Mondrian Conformal (`SciFact dev` per-abstract, `n=323`)**: {sp_cand_conf}"
    )
    if n_cp is not None:
        neu_open_conf = _fmt_conformal_summary(
            n_cp["empirical_coverage"],
            n_cp["avg_set_size"],
            n_cp["single_label_set_rate"],
            n_cp["coverage_by_class_detail"],
            ("SUPPORT", "CONTRADICT", "NEI"),
        )
        lines.append(
            f"- **Neural BGE + DeBERTa-v3 NLI 90% Mondrian Conformal (`SciFact dev` open-corpus, `n=300`)**: {neu_open_conf}"
        )
    return "\n".join(lines)


def render_expected_multi_dataset_table(data: Mapping[str, Any]) -> str:
    b = data["benchmarks"]
    sf = b["scifact"]["dev_split"]
    sf_nd = sf.get("neural_biencoder_nli")
    pqa_3w = b["pubmedqa"]["three_way_yes_no_maybe"]
    pqa_bin = b["pubmedqa"]["binary_yes_no"]
    cb = b["conflictbank"]
    cb_cd = cb["conflict_detection"]
    cb_cl = cb["claim_evidence_verification"]
    fe = b["faitheval_inconsistent"]
    fe_cd = fe["conflict_detection"]
    cc_ir = b["claimcheck"]["critique_rationale_retrieval"]
    cc_st = b["claimcheck"]["stance_verification"]
    sv = b["sciver"]["stance_verification"]
    chem = b["chembench"]["mcq_rationale_discrimination"]
    lq = b["litqa2"]["mcq_rationale_discrimination"]

    def _row(
        ds_cell: str,
        n_cell: str,
        lic_cell: str,
        metric_name: str,
        triv_v: float,
        triv_note: str,
        bm25_v: float,
        sparse_v: float,
        neural_v: float | None,
        audit_note: str,
    ) -> str:
        t_c, b_c, comps = _format_competing_cells(
            triv_v,
            bm25_v,
            [("sparse", sparse_v), ("neural", neural_v)],
            trivial_note=triv_note,
        )
        return (
            f"| {ds_cell} | {n_cell} | {lic_cell} | {metric_name} | "
            f"{t_c} | {b_c} | {comps[0]} | {comps[1]} | {audit_note} |"
        )

    cb_hyb_c = cb_cd["hybrid_bilateral"]
    cb_neu_c = cb_cd.get("neural_nli_bilateral")
    cb_audit = (
        f"Sparse TPR `{cb_hyb_c['true_positive_rate']:.3f}` / neg-ctrl FPR `{cb_hyb_c['false_positive_rate']:.3f}` "
        f"(AUROC `{cb_hyb_c['conflict_auroc']:.3f}`)"
        + (
            f"; Neural TPR `{cb_neu_c['true_positive_rate']:.3f}` / FPR `{cb_neu_c['false_positive_rate']:.3f}` "
            f"(AUROC `{cb_neu_c['conflict_auroc']:.3f}`)"
            if cb_neu_c
            else ""
        )
    )

    fe_hyb_c = fe_cd["hybrid_bilateral"]
    fe_neu_c = fe_cd.get("neural_nli_bilateral")
    fe_audit = (
        f"Sparse TPR `{fe_hyb_c['true_positive_rate']:.3f}` / neg-ctrl FPR `{fe_hyb_c['false_positive_rate']:.3f}` "
        f"(AUROC `{fe_hyb_c['conflict_auroc']:.3f}`)"
        + (
            f"; Neural TPR `{fe_neu_c['true_positive_rate']:.3f}` / FPR `{fe_neu_c['false_positive_rate']:.3f}` "
            f"(AUROC `{fe_neu_c['conflict_auroc']:.3f}`)"
            if fe_neu_c
            else ""
        )
    )

    lines = [
        "| Dataset (`data_source`) | Evaluated ($n$) | License (`eval_only`) | Metric | Best Trivial Baseline | `Okapi BM25` | `Sparse Hybrid` (`numpy`) | `Neural BGE + DeBERTa-v3` | Conformal Coverage (Size, 1-Lbl, Per-Class [95% CI]) / Negative-Control Audit |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
        _row(
            f"`SciFact` (`{b['scifact']['data_source']}`)",
            f"`{sf['total_claims']}` (`{sf['claim_doc_pairs']}` pairs)",
            f"`{b['scifact']['spdx_license']}` (`False`)",
            "Oracle `Hit@1`",
            sf["oracle_rationale_ranking"]["random_20_seeds"]["hit@1"],
            "random",
            sf["oracle_rationale_ranking"]["bm25"]["hit@1"],
            sf["oracle_rationale_ranking"]["hybrid_reranker"]["hit@1"],
            sf_nd["oracle_rationale_ranking"]["neural_nli_reranker"]["hit@1"] if sf_nd else None,
            _fmt_conformal_summary(
                sf["split_conformal_90"]["empirical_coverage"],
                sf["split_conformal_90"]["avg_set_size"],
                sf["split_conformal_90"]["single_label_set_rate"],
                sf["split_conformal_90"]["coverage_by_class_detail"],
                ("SUPPORT", "CONTRADICT", "NEI"),
            ),
        ),
        _row(
            f"`PubMedQA` 3-way (`{b['pubmedqa']['data_source']}`)",
            f"`{pqa_3w['n_evaluated']}` (`pqa_labeled`)",
            f"`{b['pubmedqa']['spdx_license']}` (`False`)",
            "3-Way `Accuracy`",
            pqa_3w["majority_class_accuracy"],
            "always yes",
            pqa_3w["bm25_accuracy"],
            pqa_3w["hybrid_reranker_accuracy"],
            pqa_3w.get("neural_nli_accuracy"),
            _fmt_conformal_summary(
                pqa_3w["conformal_coverage_90"],
                pqa_3w["conformal_avg_set_size_90"],
                pqa_3w["conformal_single_label_rate_90"],
                pqa_3w["conformal_coverage_by_class_detail_90"],
                ("SUPPORT", "CONTRADICT", "NEI"),
            ),
        ),
        _row(
            f"`PubMedQA` binary (`{b['pubmedqa']['data_source']}`)",
            f"`{pqa_bin['n_evaluated']}` (`yes`/`no`)",
            f"`{b['pubmedqa']['spdx_license']}` (`False`)",
            "Binary `Macro-F1`",
            pqa_bin["majority_class_macro_f1"],
            f"maj acc `{pqa_bin['majority_class_accuracy']:.3f}`",
            pqa_bin["bm25_macro_f1"],
            pqa_bin["hybrid_reranker_macro_f1"],
            pqa_bin.get("neural_nli_macro_f1"),
            "`LONG_ANSWER` withheld",
        ),
        _row(
            f"`ConflictBank` `CB_qa` (`{cb['data_source']}`)",
            f"`{cb['n_conflicting_qa_contexts']}` pos + `{cb['n_negative_control_qa_contexts']}` neg",
            f"`{cb['spdx_license']}` (`False`)",
            "Conflict `Balanced Acc`",
            0.500,
            "constant",
            cb_cd["bm25"]["balanced_accuracy"],
            cb_hyb_c["balanced_accuracy"],
            cb_neu_c["balanced_accuracy"] if cb_neu_c else None,
            cb_audit,
        ),
        _row(
            f"`ConflictBank` pairs (`{cb['data_source']}`)",
            f"`{cb_cl['n_evaluated']}` (`200` self + `146` cross)",
            f"`{cb['spdx_license']}` (`False`)",
            "Stance `Macro-F1`",
            cb_cl["majority_class_macro_f1"],
            f"self-200 acc `{cb['self_paired_support_verification']['hybrid_reranker_accuracy']:.3f}`",
            cb_cl["bm25_macro_f1"],
            cb_cl["hybrid_reranker_macro_f1"],
            cb_cl.get("neural_nli_macro_f1"),
            _fmt_conformal_summary(
                cb_cl["conformal_coverage_90"],
                cb_cl["conformal_avg_set_size_90"],
                cb_cl["conformal_single_label_rate_90"],
                cb_cl["conformal_coverage_by_class_detail_90"],
                ("SUPPORT", "CONTRADICT"),
            ),
        ),
        _row(
            f"`FaithEval-Inconsistent` (`{fe['data_source']}`)",
            f"`{fe['n_evaluated']}` pos + `{fe['n_negative_control']}` neg",
            f"`{fe['spdx_license']}` (`False`)",
            "Conflict `Balanced Acc`",
            0.500,
            "constant",
            fe_cd["bm25"]["balanced_accuracy"],
            fe_hyb_c["balanced_accuracy"],
            fe_neu_c["balanced_accuracy"] if fe_neu_c else None,
            fe_audit,
        ),
        _row(
            f"`CLAIMCHECK` critique IR (`{b['claimcheck']['data_source']}`)",
            f"`{cc_ir['n_pairs']}` critiques",
            f"`{b['claimcheck']['spdx_license']}` (`False`)",
            "Critique `Hit@1`",
            cc_ir["chance_hit@1"],
            "random",
            cc_ir["bm25_hit@1"],
            cc_ir["hybrid_reranker_hit@1"],
            cc_ir.get("neural_nli_hit@1"),
            f"Sparse MRR `{cc_ir['hybrid_reranker_mrr']:.3f}`"
            + (f", Neural MRR `{cc_ir['neural_nli_mrr']:.3f}`" if "neural_nli_mrr" in cc_ir else ""),
        ),
        _row(
            f"`CLAIMCHECK` stance (`{b['claimcheck']['data_source']}`)",
            f"`{cc_st['n_evaluated']}` claims",
            f"`{b['claimcheck']['spdx_license']}` (`False`)",
            "3-Way `Macro-F1`",
            cc_st["majority_class_macro_f1"],
            f"maj acc `{cc_st['majority_class_accuracy']:.3f}`",
            cc_st["bm25_macro_f1"],
            cc_st["hybrid_reranker_macro_f1"],
            cc_st.get("neural_nli_macro_f1"),
            _fmt_conformal_summary(
                cc_st["conformal_coverage_90"],
                cc_st["conformal_avg_set_size_90"],
                cc_st["conformal_single_label_rate_90"],
                cc_st["conformal_coverage_by_class_detail_90"],
                ("SUPPORT", "CONTRADICT", "NEI"),
            ),
        ),
        _row(
            f"`SciVer` (`{b['sciver']['data_source']}`)",
            f"`{sv['n_evaluated']}` claims",
            f"`{b['sciver']['spdx_license']}` (`False`)",
            "Stance `Macro-F1`",
            sv["majority_class_macro_f1"],
            f"maj acc `{sv['majority_class_accuracy']:.3f}`",
            sv["bm25_macro_f1"],
            sv["hybrid_reranker_macro_f1"],
            sv.get("neural_nli_macro_f1"),
            _fmt_conformal_summary(
                sv["conformal_coverage_90"],
                sv["conformal_avg_set_size_90"],
                sv["conformal_single_label_rate_90"],
                sv["conformal_coverage_by_class_detail_90"],
                ("SUPPORT", "CONTRADICT"),
            ),
        ),
        _row(
            f"`ChemBench` -- option string matching against task text (`{b['chembench']['data_source']}`)",
            f"`{chem['n_evaluated']}` MCQ (`{b['chembench']['total_raw_tasks']}` raw)",
            f"`{b['chembench']['spdx_license']}` (**`True`**)",
            "Option `Hit@1`",
            chem["chance_hit@1"],
            "chance",
            chem["bm25_option_hit@1"],
            chem["hybrid_reranker_option_hit@1"],
            chem.get("neural_nli_option_hit@1"),
            f"Chance MRR `{chem['chance_mrr']:.3f}`; `eval_only=True` verified",
        ),
        _row(
            f"`LitQA2` -- with gold passage (`{b['litqa2']['data_source']}`)",
            f"`{lq['n_evaluated']}` MCQ (`{b['litqa2']['total_raw_tasks']}` raw)",
            f"`{b['litqa2']['spdx_license']}` (**`True`**)",
            "Option `Hit@1`",
            lq["chance_hit@1"],
            "chance",
            lq["bm25_option_hit@1"],
            lq["hybrid_reranker_option_hit@1"],
            lq.get("neural_nli_option_hit@1"),
            f"Chance MRR `{lq['chance_mrr']:.3f}`; `eval_only=True` verified",
        ),
    ]
    return "\n".join(lines)


def test_committed_open_benchmarks_json_integrity_baselines_and_negative_controls() -> None:
    assert RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}"
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert data["data_source"] == "real_downloaded"
    assert data["num_datasets_evaluated"] == 8
    assert data["leakage_audit"]["verifier_trained_strictly_on"] == "scifact_train_809_claims"
    assert data["leakage_audit"]["scifact_train_fit_calib_split"] == [607, 202]
    assert data["leakage_audit"]["scifact_train_dev_claim_id_overlap"] == 0
    assert data["leakage_audit"]["pubmedqa_long_answer_excluded"] is True
    assert data["leakage_audit"]["conflict_negative_controls_evaluated"] is True
    assert data["leakage_audit"]["eval_only_guards_verified"]["chembench"] is True
    assert data["leakage_audit"]["eval_only_guards_verified"]["litqa2"] is True

    b = data["benchmarks"]
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
    assert set(b.keys()) == expected_datasets

    for ds_id, entry in b.items():
        assert entry["data_source"] == "real_downloaded", f"{ds_id} is not real_downloaded"
        sha_fields = [v for k, v in entry.items() if k.startswith("sha256")]
        assert sha_fields, f"{ds_id} missing sha256 metadata"
        for digest in sha_fields:
            assert SHA256_RE.match(str(digest)), f"Invalid sha256 {digest!r} on {ds_id}"

    # Issue #35: verify majority & chance baselines match empirical counts
    assert b["pubmedqa"]["three_way_yes_no_maybe"]["majority_class_accuracy"] == pytest.approx(0.552, abs=1e-3)
    assert b["pubmedqa"]["binary_yes_no"]["majority_class_accuracy"] == pytest.approx(552.0 / 890.0, abs=1e-3)
    assert b["sciver"]["stance_verification"]["majority_class_accuracy"] == pytest.approx(0.505, abs=1e-3)
    assert b["claimcheck"]["stance_verification"]["majority_class_accuracy"] == pytest.approx(56.0 / 108.0, abs=1e-3)
    assert b["chembench"]["mcq_rationale_discrimination"]["chance_hit@1"] == pytest.approx(0.248, abs=2e-3)
    assert b["litqa2"]["mcq_rationale_discrimination"]["chance_hit@1"] == pytest.approx(0.244, abs=2e-3)
    assert b["chembench"]["evaluation_mode"] == "option string matching against task text"
    assert b["litqa2"]["evaluation_mode"] == "with gold passage"

    # Issue #34: verify negative-control false_positive_rate is recorded on ConflictBank and FaithEval
    for ds_id in ("conflictbank", "faitheval_inconsistent"):
        cd = b[ds_id]["conflict_detection"]
        assert "false_positive_rate" in cd
        assert "true_positive_rate" in cd
        assert "balanced_accuracy" in cd
        assert cd["false_positive_rate"] < cd["true_positive_rate"]


def test_readme_and_architecture_docs_match_benchmark_json_and_bold_only_winners() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    readme_text = README_PATH.read_text(encoding="utf-8")
    arch_text = ARCH_PATH.read_text(encoding="utf-8")
    demo_text = DEMO_OUTPUT_PATH.read_text(encoding="utf-8")

    expected_scifact = render_expected_scifact_table(data)
    expected_multi = render_expected_multi_dataset_table(data)

    for doc_label, doc_text in (("README.md", readme_text), ("ARCHITECTURE_AND_HISTORY.md", arch_text)):
        actual_scifact = _extract_marked_block(doc_text, "SCIFACT_FRONTIER_TABLE")
        actual_multi = _extract_marked_block(doc_text, "OPEN_BENCHMARKS_2026_TABLE")
        assert actual_scifact == expected_scifact, f"SCIFACT_FRONTIER_TABLE drift in {doc_label}"
        assert actual_multi == expected_multi, f"OPEN_BENCHMARKS_2026_TABLE drift in {doc_label}"

        scifact_table_only = "\n".join(
            line for line in actual_scifact.splitlines() if line.strip().startswith("|")
        )
        assert_no_undeserved_bolding_in_table(
            scifact_table_only,
            numeric_col_indices=(1, 2, 3, 4, 5),
            baseline_col_indices=(1, 2),
        )
        assert_no_undeserved_bolding_in_table(
            actual_multi,
            numeric_col_indices=(4, 5, 6, 7),
            baseline_col_indices=(4, 5),
        )

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


def test_issue_35_bolding_validator_rejects_non_winning_or_below_baseline_bold_cells() -> None:
    # Case 1: Model cell bolded when below Best Trivial Baseline (e.g., 0.473 < 0.552)
    bad_below_trivial = "\n".join(
        [
            "| Dataset | n | Lic | Metric | Best Trivial Baseline | BM25 | Sparse | Neural | Note |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
            "| `PubMedQA` | `1000` | `MIT` | Acc | `0.552` (always yes) | `0.259` | **`0.473`** | `0.510` | note |",
        ]
    )
    with pytest.raises(AssertionError, match="below row max|fails to beat baseline"):
        assert_no_undeserved_bolding_in_table(
            bad_below_trivial,
            numeric_col_indices=(4, 5, 6, 7),
            baseline_col_indices=(4, 5),
        )

    # Case 2: Model cell bolded when beaten by another model column in the same row
    bad_not_row_best = "\n".join(
        [
            "| Dataset | n | Lic | Metric | Best Trivial Baseline | BM25 | Sparse | Neural | Note |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
            "| `SciFact` | `300` | `CC` | Hit@1 | `0.197` | `0.483` | **`0.660`** | **`0.785`** | note |",
        ]
    )
    with pytest.raises(AssertionError, match="below row max"):
        assert_no_undeserved_bolding_in_table(
            bad_not_row_best,
            numeric_col_indices=(4, 5, 6, 7),
            baseline_col_indices=(4, 5),
        )

    # Case 3: Model cell bolded when tying or losing to BM25 (e.g., 0.883 vs BM25 0.885)
    bad_below_bm25 = "\n".join(
        [
            "| Metric | Trivial | BM25 | SVD | Sparse | Neural |",
            "|---|---:|---:|---:|---:|---:|",
            "| `doc_recall@5` | `0.001` | `0.885` | `0.490` | **`0.883`** | `0.880` |",
        ]
    )
    with pytest.raises(AssertionError, match="below row max|fails to beat baseline"):
        assert_no_undeserved_bolding_in_table(
            bad_below_bm25,
            numeric_col_indices=(1, 2, 3, 4, 5),
            baseline_col_indices=(1, 2),
        )
