"""Unit tests for FrontierHybridVerifier, Split Conformal Stance Sets, and Open Benchmark CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lab_ask_why import (
    STANCE_LABELS,
    EntityNumericNegationIndex,
    FrontierHybridVerifier,
    SplitConformalStanceClassifier,
    TruncatedSVDSemanticIndex,
    attribute_assay_outcome_permille,
    chance_mcq_baseline,
    compute_official_scifact_metrics,
    conformal_stance_set_permille,
    macro_f1_score,
    majority_class_stance_baseline,
    mean_reciprocal_rank,
    split_scifact_train_fit_and_calib,
    wilson_score_interval,
)
import lab_ask_why.frontier_verifier as fv_mod
from lab_ask_why.frontier_verifier import _csr_matmul, _csr_rmatmul

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str) -> Any:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_truncated_svd_pure_numpy_vs_scipy_parity_and_background_isolation() -> None:
    texts = [
        "Homozygous alpha+-thalassaemia microcytosis protects against severe malarial anaemia.",
        "Microerythrocyte count elevation in alpha+-thalassemia confers malaria protection.",
        "GPX4 inhibition by RSL3 induces iron-dependent lipid peroxidation and ferroptosis.",
        "Ferrostatin-1 blocks lipid hydroperoxide accumulation during GPX4 inhibition.",
        "CRISPR SpCas9 endonuclease requires a canonical NGG PAM motif for DNA cleavage.",
        "PAM-proximal seed mismatches abolish Cas9 double-strand break activity.",
        "Zyxwvunikin99 secret evaluation biomarker appears exclusively in held-out doc.",
    ]
    mask = [True, True, True, True, True, True, False]
    svd_bg = TruncatedSVDSemanticIndex(texts, n_components=4, seed=42, idf_mask=mask)

    # Verify held-out doc 6 vocabulary never leaked into SVD vocabulary
    assert all("zyxwvunikin99" not in term for term in svd_bg.vocab.keys())

    # Verify semantic similarity clusters thalassaemia paraphrases above unrelated CRISPR sentence
    cos_thal = svd_bg.cosine_scores("alpha thalassemia erythrocyte malaria protection")
    assert cos_thal[0] > cos_thal[4]
    assert cos_thal[1] > cos_thal[4]

    # Determinism across repeated constructions
    svd_bg_2 = TruncatedSVDSemanticIndex(texts, n_components=4, seed=42, idf_mask=mask)
    np.testing.assert_allclose(svd_bg.embeddings, svd_bg_2.embeddings, rtol=1e-12, atol=1e-12)

    # Pure-NumPy CSR matmul / rmatmul parity check
    indptr = np.array([0, 2, 3, 5], dtype=np.int32)
    indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)
    data = np.array([0.6, 0.8, 1.0, 0.5, 0.5], dtype=np.float64)
    rhs = np.arange(12, dtype=np.float64).reshape(4, 3)
    lhs_t = np.arange(9, dtype=np.float64).reshape(3, 3)

    dense_a = np.array(
        [
            [0.6, 0.0, 0.8, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        _csr_matmul(3, 4, indptr, indices, data, rhs, sp_mat=None),
        dense_a @ rhs,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _csr_rmatmul(3, 4, indptr, indices, data, lhs_t, sp_mat=None),
        dense_a.T @ lhs_t,
        rtol=1e-12,
    )


def test_entity_numeric_negation_index_and_metrics() -> None:
    sents = [
        "BRCA1 p.Arg1699Gln reduces homology-directed repair by 65% at 37 C.",
        "Wild-type control did not alter baseline survival.",
    ]
    idx = EntityNumericNegationIndex(sents)
    sc = idx.overlap_scores("BRCA1 reduces repair by 65%.")
    assert sc[0] > sc[1] and sc[0] > 0.5

    pairs = [{"gold": {2}}, {"gold": {0, 3}}]
    ranks = [[1, 2, 0], [3, 0, 1, 2]]
    assert mean_reciprocal_rank(ranks, pairs) == pytest.approx(0.75)
    assert macro_f1_score(
        ["SUPPORT", "CONTRADICT", "NEI"],
        ["SUPPORT", "CONTRADICT", "NEI"],
        labels=STANCE_LABELS,
    ) == pytest.approx(1.0)


def test_frontier_hybrid_verifier_fit_conformal_sets_and_conflict_duality() -> None:
    corpus = {
        10: {
            "doc_id": 10,
            "title": "AMPK Activation by Metformin",
            "abstract": [
                "Background: Hepatic gluconeogenesis drives fasting hyperglycemia.",
                "Metformin activates hepatic AMPK signaling and increases glucose uptake.",
                "Conclusions: Metformin significantly increases AMPK phosphorylation.",
            ],
        },
        20: {
            "doc_id": 20,
            "title": "RSL3 Inhibits GPX4",
            "abstract": [
                "Objective: Study ferroptotic lipid peroxidation.",
                "RSL3 strongly inhibits GPX4 catalytic activity and reduces cell survival.",
                "Conclusions: RSL3 decreases GPX4 activity rather than activating GPX4.",
            ],
        },
        30: {
            "doc_id": 30,
            "title": "SpCas9 PAM Specificity",
            "abstract": [
                "SpCas9 recognizes canonical NGG PAM motifs.",
                "PAM-proximal seed mismatches abolish SpCas9 DNA cleavage.",
            ],
        },
        40: {
            "doc_id": 40,
            "title": "Aspirin Stability vs Hydrolysis",
            "abstract": [
                "Buffered pH 6.8 medium increases aspirin stability and prevents hydrolysis at 25 C.",
                "Unbuffered thermal incubation at 37 C fails to prevent aspirin hydrolysis and reduces stability.",
            ],
        },
    }
    train_claims = [
        {"id": 1, "claim": "Metformin activates hepatic AMPK signaling.", "evidence": {"10": [{"sentences": [1, 2], "label": "SUPPORT"}]}, "cited_doc_ids": [10]},
        {"id": 2, "claim": "PAM-proximal seed mismatches abolish SpCas9 cleavage.", "evidence": {"30": [{"sentences": [1], "label": "SUPPORT"}]}, "cited_doc_ids": [30]},
        {"id": 3, "claim": "Buffered pH 6.8 medium increases aspirin stability at 25 C.", "evidence": {"40": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [40]},
        {"id": 4, "claim": "Metformin increases AMPK phosphorylation.", "evidence": {"10": [{"sentences": [2], "label": "SUPPORT"}]}, "cited_doc_ids": [10]},
        {"id": 5, "claim": "SpCas9 recognizes canonical NGG PAM motifs.", "evidence": {"30": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [30]},
        {"id": 6, "claim": "RSL3 activates GPX4 catalytic activity and increases survival.", "evidence": {"20": [{"sentences": [1, 2], "label": "CONTRADICT"}]}, "cited_doc_ids": [20]},
        {"id": 7, "claim": "Metformin inhibits hepatic AMPK signaling.", "evidence": {"10": [{"sentences": [1], "label": "CONTRADICT"}]}, "cited_doc_ids": [10]},
        {"id": 8, "claim": "PAM-proximal seed mismatches enhance SpCas9 DNA cleavage.", "evidence": {"30": [{"sentences": [1], "label": "CONTRADICT"}]}, "cited_doc_ids": [30]},
        {"id": 9, "claim": "Thermal incubation at 37 C increases unbuffered aspirin stability.", "evidence": {"40": [{"sentences": [1], "label": "CONTRADICT"}]}, "cited_doc_ids": [40]},
        {"id": 10, "claim": "RSL3 increases GPX4 enzymatic activity.", "evidence": {"20": [{"sentences": [2], "label": "CONTRADICT"}]}, "cited_doc_ids": [20]},
        {"id": 11, "claim": "Metformin cures glioblastoma in zebrafish embryos.", "evidence": {}, "cited_doc_ids": [10]},
        {"id": 12, "claim": "SpCas9 regulates solar neutrino oscillation in stars.", "evidence": {}, "cited_doc_ids": [30]},
        {"id": 13, "claim": "RSL3 increases superconducting critical temperature.", "evidence": {}, "cited_doc_ids": [20]},
        {"id": 14, "claim": "Aspirin catalyzes nitrogen fixation in cyanobacteria.", "evidence": {}, "cited_doc_ids": [40]},
        {"id": 15, "claim": "Quantum entanglement accelerates ribosome translation.", "evidence": {}, "cited_doc_ids": [10]},
    ]

    fit_c, cal_c = split_scifact_train_fit_and_calib(train_claims, calib_fraction=0.20, seed=2026)
    assert len(fit_c) + len(cal_c) == len(train_claims)
    assert ({c["id"] for c in fit_c} & {c["id"] for c in cal_c}) == set()

    verifier = FrontierHybridVerifier.fit_on_scifact_train(
        corpus, train_claims, svd_components=6, alpha=0.10, seed=2026
    )
    assert verifier.training_provenance["trained"] is True
    assert verifier.training_provenance["train_source"] == "scifact_train_only"

    # Conflicting multi-sentence candidate context emits multi-label {"SUPPORT", "CONTRADICT"} set
    conflict_pred = verifier.verify_candidate_sentences(
        "Aqueous incubation increases aspirin stability.",
        corpus[40]["abstract"],
        title=corpus[40]["title"],
        top_k=2,
    )
    assert abs(sum(conflict_pred.probabilities.values()) - 1.0) < 1e-6
    assert "SUPPORT" in conflict_pred.conformal_set
    assert "CONTRADICT" in conflict_pred.conformal_set
    assert conflict_pred.is_contested_conflict is True

    # Empty candidate sentences resolve to NEI cleanly
    empty_pred = verifier.verify_candidate_sentences("Metformin activates AMPK.", [])
    assert empty_pred.predicted_label == "NEI"
    assert "NEI" in empty_pred.conformal_set

    # Finite-sample conformal quantile formula check
    q90 = SplitConformalStanceClassifier.conformal_quantile(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.95], alpha=0.10
    )
    assert q90 == pytest.approx(0.95)  # ceil(11 * 0.90) = 10 -> index 9 -> 0.95
    q95 = SplitConformalStanceClassifier.conformal_quantile(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.95], alpha=0.05
    )
    assert q95 == pytest.approx(1.0)  # ceil(11 * 0.95) = 11 > n -> 1.0
    q80 = SplitConformalStanceClassifier.conformal_quantile(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.95], alpha=0.20
    )
    assert q80 == pytest.approx(0.45)  # ceil(11 * 0.80) = 9 -> index 8 -> 0.45


def test_benchmark_cli_offline_fail_closed_and_allow_synthetic_labeling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench_mod = _load_script_module("benchmark_open_datasets", "scripts/benchmark_open_datasets.py")
    empty_cache = tmp_path / "empty_data"
    out_json = tmp_path / "out.json"

    # 1. --offline without cached data MUST fail non-zero unless --allow-synthetic is passed
    rc_fail = bench_mod.main(
        ["--offline", "--data-root", str(empty_cache), "--output", str(out_json)]
    )
    assert rc_fail != 0
    assert not out_json.exists()
    _ = capsys.readouterr()

    # 2. --offline --allow-synthetic succeeds and labels EVERY dict and EVERY CLI line
    rc_syn = bench_mod.main(
        [
            "--offline",
            "--allow-synthetic",
            "--data-root",
            str(empty_cache),
            "--output",
            str(out_json),
        ]
    )
    assert rc_syn == 0
    cli_stdout = capsys.readouterr().out.strip().splitlines()
    assert len(cli_stdout) >= 8
    for line in cli_stdout:
        assert "[data_source=synthetic_fixture]" in line

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["data_source"] == "synthetic_fixture"
    assert payload["num_datasets_evaluated"] == 8
    for ds_id, ds_entry in payload["benchmarks"].items():
        assert ds_entry["data_source"] == "synthetic_fixture", f"Missing data_source on {ds_id}"


def test_lean_parity_generator_and_permille_mirrors() -> None:
    parity_mod = _load_script_module("verify_lean_parity", "scripts/verify_lean_parity.py")
    lean_exprs, py_expected = parity_mod._build_160_parity_cases()
    assert len(lean_exprs) == 160
    assert len(py_expected) == 160

    lean_src = (REPO_ROOT / "lean" / "LabAskWhyLean.lean").read_text(encoding="utf-8")
    for thm in parity_mod.ALL_THEOREMS:
        short_name = thm.split(".")[-1]
        assert f"theorem {short_name}" in lean_src

    assert conformal_stance_set_permille(480, 470, 50, 700) == {
        "supportIn": True,
        "contradictIn": True,
        "neiIn": False,
    }
    assert (
        attribute_assay_outcome_permille(
            {
                "preGateCertified": True,
                "acousticNormal": True,
                "scheduleWindowsRespected": True,
                "tactileSeatedCleanly": True,
                "posControlPassed": True,
                "negControlBlankClean": True,
                "spectralResidualPermille": 40,
                "maxResidualPermille": 100,
                "precursorOverlapPermille": 30,
                "maxOverlapPermille": 150,
                "signalAboveActivityThresh": False,
            }
        )
        == "certifiedNegative"
    )

    try:
        lake_bin = parity_mod._resolve_lake_bin(None)
    except FileNotFoundError:
        pytest.skip("Lean 4 `lake` binary not installed in this environment")
    live_res = parity_mod.run_lean_verification(lake_bin=lake_bin, clean_after=True)
    assert live_res["theorems_checked"] >= 21
    assert live_res["zero_sorry_or_custom_axioms"] is True
    assert live_res["parity_cases_evaluated"] >= 100
    assert live_res["parity_mismatches"] == 0


def test_issue_35_36_pure_numpy_vs_scipy_qhat_parity_wilson_ci_and_official_scifact() -> None:
    corpus = {
        idx: {
            "doc_id": idx,
            "title": f"Biomedical Study {idx}",
            "abstract": [
                f"Background context for experimental pathway {idx}.",
                f"Compound C{idx} activates target kinase K{idx} and increases phosphorylation by {10 + idx}%.",
                f"Inhibitor I{idx} blocks kinase K{idx} and reduces survival rather than increasing phosphorylation.",
            ],
        }
        for idx in range(1, 25)
    }
    train_claims = []
    for i in range(1, 25):
        if i <= 8:
            train_claims.append(
                {
                    "id": 100 + i,
                    "claim": f"Compound C{i} activates kinase K{i} and increases phosphorylation.",
                    "evidence": {str(i): [{"sentences": [1], "label": "SUPPORT"}]},
                    "cited_doc_ids": [i],
                }
            )
        elif i <= 16:
            train_claims.append(
                {
                    "id": 100 + i,
                    "claim": f"Inhibitor I{i} activates kinase K{i} and increases survival.",
                    "evidence": {str(i): [{"sentences": [2], "label": "CONTRADICT"}]},
                    "cited_doc_ids": [i],
                }
            )
        else:
            train_claims.append(
                {
                    "id": 100 + i,
                    "claim": f"Compound C{i} regulates superconducting critical current in cuprates.",
                    "evidence": {},
                    "cited_doc_ids": [i],
                }
            )

    orig_sp = fv_mod._sp
    try:
        fv_mod._sp = None
        v_np = FrontierHybridVerifier.fit_on_scifact_train(
            corpus, train_claims, svd_components=8, alpha=0.10, calib_fraction=0.25, seed=2026
        )
    finally:
        fv_mod._sp = orig_sp

    v_sp = FrontierHybridVerifier.fit_on_scifact_train(
        corpus, train_claims, svd_components=8, alpha=0.10, calib_fraction=0.25, seed=2026
    )

    for key, val_sp in v_sp.training_provenance["conformal_calibration"].items():
        val_np = v_np.training_provenance["conformal_calibration"][key]
        assert abs(val_np - val_sp) < 1e-6, f"Mismatch on {key}: {val_np} vs {val_sp}"

    # Verify chunked _csr_matmul / _csr_rmatmul with small block_rows=2 matches dense math
    rng = np.random.default_rng(7)
    dense_m = np.where(rng.random((9, 7)) > 0.6, rng.normal(size=(9, 7)), 0.0)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for r in range(9):
        nz = np.flatnonzero(dense_m[r])
        indices.extend(int(c) for c in nz)
        data.extend(float(dense_m[r, c]) for c in nz)
        indptr.append(len(indices))
    ip_arr = np.asarray(indptr, dtype=np.int32)
    ix_arr = np.asarray(indices, dtype=np.int32)
    dt_arr = np.asarray(data, dtype=np.float64)
    rhs = rng.normal(size=(7, 4))
    lhs_t = rng.normal(size=(9, 4))
    np.testing.assert_allclose(
        _csr_matmul(9, 7, ip_arr, ix_arr, dt_arr, rhs, sp_mat=None, block_rows=2),
        dense_m @ rhs,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _csr_rmatmul(9, 7, ip_arr, ix_arr, dt_arr, lhs_t, sp_mat=None, block_rows=2),
        dense_m.T @ lhs_t,
        rtol=1e-12,
        atol=1e-12,
    )

    # Verify skewed high-frequency column rmatmul splits blocks via max_block_nnz
    skew_m = rng.normal(size=(600, 6))
    s_ip = np.arange(0, 3601, 6, dtype=np.int32)
    s_ix = np.tile(np.arange(6, dtype=np.int32), 600)
    s_dt = skew_m.reshape(-1)
    s_lhs = rng.normal(size=(600, 5))
    np.testing.assert_allclose(
        _csr_rmatmul(600, 6, s_ip, s_ix, s_dt, s_lhs, sp_mat=None, block_rows=2048),
        skew_m.T @ s_lhs,
        rtol=1e-12,
        atol=1e-12,
    )

    # Verify Wilson CI, majority baseline, chance MCQ baseline, and official SciFact F1
    lo, hi = wilson_score_interval(90, 100)
    assert 0.82 < lo < 0.90 < hi < 0.95
    maj = majority_class_stance_baseline(
        ["SUPPORT"] * 552 + ["CONTRADICT"] * 338 + ["NEI"] * 110,
        eval_labels=STANCE_LABELS,
    )
    assert maj["majority_class_accuracy"] == pytest.approx(0.552)
    ch = chance_mcq_baseline([4, 4, 4, 4])
    assert ch["chance_hit@1"] == pytest.approx(0.25)
    assert ch["chance_mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0)

    off = compute_official_scifact_metrics(
        [
            {"id": 1, "evidence": {"10": [{"sentences": [1, 2], "label": "SUPPORT"}]}},
            {"id": 2, "evidence": {}},
        ],
        {
            "1": {"10": {"label": "SUPPORT", "sentences": [1, 2]}},
            "2": {},
        },
    )
    assert off["abstract_label_only"]["f1"] == pytest.approx(1.0)
    assert off["abstract_rationales"]["f1"] == pytest.approx(1.0)
    assert off["sentence_selection"]["f1"] == pytest.approx(1.0)
    assert off["sentence_label"]["f1"] == pytest.approx(1.0)
