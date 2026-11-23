"""Unit and benchmark tests for lab-ask-why community_extensions (Ideas #01 and #02)."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_ce = importlib.import_module("community_extensions")
CandidatePerturbation = _ce.CandidatePerturbation
EvidenceSnippet = _ce.EvidenceSnippet
MechanisticHypothesis = _ce.MechanisticHypothesis
compute_perturbation_evsi = _ce.compute_perturbation_evsi
deconvolve_precursor_ghosts = _ce.deconvolve_precursor_ghosts
elo_to_prior = _ce.elo_to_prior
naive_peak_match_claim = _ce.naive_peak_match_claim
nnls_solve = _ce.nnls_solve
run_crux_tournament_benchmark = _ce.run_crux_tournament_benchmark
run_elo_hypothesis_tournament = _ce.run_elo_hypothesis_tournament
run_precursor_ghost_benchmark = _ce.run_precursor_ghost_benchmark
select_naive_bm25_experiment = _ce.select_naive_bm25_experiment
synthesize_crux_experiment = _ce.synthesize_crux_experiment


def test_elo_tournament_symmetry_and_priors() -> None:
    h_strong = MechanisticHypothesis(
        hypothesis_id="H_strong",
        mechanism="Direct catalytic activation",
        support_evidence=(
            EvidenceSnippet("d1", "Strong catalytic turnover.", "SUPPORT", 24.0),
        ),
        contradict_evidence=(),
    )
    h_weak = MechanisticHypothesis(
        hypothesis_id="H_weak",
        mechanism="Trace metal impurity artifact",
        support_evidence=(
            EvidenceSnippet("d2", "Minor baseline drift.", "SUPPORT", 9.0),
        ),
        contradict_evidence=(
            EvidenceSnippet("d3", "ICP-MS shows <1 ppb metal.", "CONTRADICT", 22.0),
        ),
    )
    ratings = run_elo_hypothesis_tournament([h_strong, h_weak])
    assert ratings["H_strong"] > ratings["H_weak"]
    assert math.isclose(ratings["H_strong"] + ratings["H_weak"], 3000.0, rel_tol=1e-9)

    priors = elo_to_prior(ratings)
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=1e-9)
    assert priors["H_strong"] > priors["H_weak"]


def test_evsi_zero_for_non_discriminating_assay_and_positive_for_crux() -> None:
    priors = {"H1": 0.50, "H2": 0.50}
    redundant = CandidatePerturbation(
        experiment_id="E_red",
        name="Identical readout assay",
        cost_usd=50.0,
        likelihood_positive_by_hypothesis={"H1": 0.85, "H2": 0.85},
    )
    evsi_red, net_red, mi_red = compute_perturbation_evsi(
        priors, redundant, decision_value_usd=1000.0
    )
    assert evsi_red == pytest.approx(0.0, abs=1e-12)
    assert net_red == pytest.approx(-50.0, abs=1e-12)
    assert mi_red == pytest.approx(0.0, abs=1e-12)

    crux = CandidatePerturbation(
        experiment_id="E_crux",
        name="Orthogonal point-mutant rescue",
        cost_usd=80.0,
        likelihood_positive_by_hypothesis={"H1": 0.95, "H2": 0.05},
    )
    evsi_crux, net_crux, mi_crux = compute_perturbation_evsi(
        priors, crux, decision_value_usd=1000.0
    )
    assert evsi_crux == pytest.approx(450.0, rel=1e-6)
    assert net_crux == pytest.approx(370.0, rel=1e-6)
    assert mi_crux > 0.70


def test_synthesize_crux_noinfo_abstention_and_selection() -> None:
    low_info_hyps = (
        MechanisticHypothesis(
            hypothesis_id="H1",
            mechanism="Unverified pathway A",
            support_evidence=(EvidenceSnippet("d1", "Weak mention.", "SUPPORT", 3.2),),
            contradict_evidence=(),
        ),
        MechanisticHypothesis(
            hypothesis_id="H2",
            mechanism="Unverified pathway B",
            support_evidence=(EvidenceSnippet("d2", "Weak mention.", "SUPPORT", 2.8),),
            contradict_evidence=(),
        ),
    )
    cand = CandidatePerturbation(
        experiment_id="E1",
        name="Test A vs B",
        cost_usd=40.0,
        likelihood_positive_by_hypothesis={"H1": 0.9, "H2": 0.1},
    )
    abstain = synthesize_crux_experiment(low_info_hyps, [cand], noinfo_threshold=8.0)
    assert abstain.status == "ABSTAIN_NOINFO"
    assert abstain.selected_experiment_id is None

    contested_hyps = (
        MechanisticHypothesis(
            hypothesis_id="H1",
            mechanism="Pathway A",
            support_evidence=(
                EvidenceSnippet("d1", "Strong support A.", "SUPPORT", 20.0),
            ),
            contradict_evidence=(
                EvidenceSnippet("d2", "Counterexample A.", "CONTRADICT", 16.0),
            ),
        ),
        MechanisticHypothesis(
            hypothesis_id="H2",
            mechanism="Pathway B",
            support_evidence=(
                EvidenceSnippet("d3", "Strong support B.", "SUPPORT", 19.5),
            ),
            contradict_evidence=(
                EvidenceSnippet("d4", "Counterexample B.", "CONTRADICT", 15.5),
            ),
        ),
    )
    confirm_trap = CandidatePerturbation(
        experiment_id="E_trap",
        name="High-BM25 repeat",
        cost_usd=30.0,
        likelihood_positive_by_hypothesis={"H1": 0.88, "H2": 0.87},
        bm25_lexical_overlap=31.0,
        is_redundant_confirmation=True,
    )
    crux_exp = CandidatePerturbation(
        experiment_id="E_crux",
        name="Mutant washout crux",
        cost_usd=75.0,
        likelihood_positive_by_hypothesis={"H1": 0.93, "H2": 0.07},
        bm25_lexical_overlap=12.0,
    )
    assert (
        select_naive_bm25_experiment(
            contested_hyps, [confirm_trap, crux_exp]
        ).experiment_id
        == "E_trap"
    )
    selected = synthesize_crux_experiment(contested_hyps, [confirm_trap, crux_exp])
    assert selected.status == "OK_CRUX_SELECTED"
    assert selected.deadlock_detected is True
    assert selected.selected_experiment_id == "E_crux"

    no_disc = synthesize_crux_experiment(
        contested_hyps, [confirm_trap], min_evsi_usd=25.0
    )
    assert no_disc.status == "NO_DISCRIMINATING_EXPERIMENT"
    assert no_disc.selected_experiment_id is None


def test_validation_guards_for_crux_and_ghost() -> None:
    with pytest.raises(ValueError, match="polarity"):
        EvidenceSnippet("d1", "Bad stance", "MAYBE", 10.0)
    with pytest.raises(ValueError, match="Invalid likelihood"):
        CandidatePerturbation("E_bad", "Bad", 10.0, {"H1": 1.2})
    with pytest.raises(ValueError, match="Incompatible shapes"):
        nnls_solve(np.ones((4, 2)), np.ones(3))


def test_crux_tournament_benchmark_metrics() -> None:
    summary = run_crux_tournament_benchmark(n_scenarios=40, seed=42)
    assert summary["n_scenarios"] == 40
    naive = summary["naive_top_bm25"]
    crux = summary["elo_evsi_crux"]
    assert isinstance(naive, dict) and isinstance(crux, dict)
    assert crux["deadlock_resolution_rate"] >= 0.85
    assert crux["deadlock_resolution_rate"] > naive["deadlock_resolution_rate"] + 0.75
    assert crux["wasted_redundant_experiments"] == 0
    assert naive["wasted_redundant_experiments"] > 20


def test_nnls_and_precursor_ghost_guard_outcomes() -> None:
    grid = np.linspace(10.0, 80.0, 300)
    p1 = np.exp(-0.5 * ((grid - 22.0) / 0.35) ** 2) + 0.8 * np.exp(
        -0.5 * ((grid - 38.0) / 0.35) ** 2
    )
    p2 = np.exp(-0.5 * ((grid - 29.0) / 0.35) ** 2) + 0.7 * np.exp(
        -0.5 * ((grid - 50.0) / 0.35) ** 2
    )

    weights = nnls_solve(np.column_stack([p1, p2]), 0.6 * p1 + 1.1 * p2)
    assert np.all(weights >= 0.0)
    assert weights[0] == pytest.approx(0.6, abs=1e-4)
    assert weights[1] == pytest.approx(1.1, abs=1e-4)

    # 1. Ghost-peak overlap trap: target peaks coincide with p1 + p2, zero true target present
    t_ghost = np.exp(-0.5 * ((grid - 22.05) / 0.35) ** 2) + 0.9 * np.exp(
        -0.5 * ((grid - 29.0) / 0.35) ** 2
    )
    obs_ghost = 0.95 * p1 + 0.90 * p2
    assert naive_peak_match_claim(obs_ghost, t_ghost) is True
    res_ghost = deconvolve_precursor_ghosts(obs_ghost, [p1, p2], t_ghost)
    assert res_ghost.precursor_overlap_permille > 150
    assert res_ghost.attribution_outcome == "ambiguous_characterization"
    assert res_ghost.admit_to_memory is False
    assert res_ghost.claim_blocked is True

    # 2. Execution hardware fault -> execution_artifact
    res_fault = deconvolve_precursor_ghosts(
        obs_ghost, [p1, p2], t_ghost, execution_clean=False
    )
    assert res_fault.attribution_outcome == "execution_artifact"
    assert res_fault.admit_to_memory is False

    # 3. Clean novel phase with unique non-overlapping Bragg lines -> certified_positive
    t_novel = np.exp(-0.5 * ((grid - 16.0) / 0.35) ** 2) + 0.9 * np.exp(
        -0.5 * ((grid - 44.0) / 0.35) ** 2
    )
    obs_novel = 0.05 * p1 + 0.95 * t_novel
    res_pos = deconvolve_precursor_ghosts(obs_novel, [p1, p2], t_novel)
    assert res_pos.attribution_outcome == "certified_positive"
    assert res_pos.admit_to_memory is True
    assert res_pos.claim_blocked is False

    # 4. Clean negative (only precursors present, disjoint from t_novel) -> certified_negative
    obs_neg = 0.80 * p1 + 0.70 * p2
    res_neg = deconvolve_precursor_ghosts(obs_neg, [p1, p2], t_novel)
    assert res_neg.attribution_outcome == "certified_negative"
    assert res_neg.admit_to_memory is True


def test_precursor_ghost_benchmark_metrics() -> None:
    summary = run_precursor_ghost_benchmark(n_spectra=60, seed=42)
    assert summary["n_spectra"] == 60
    naive = summary["naive_peak_match"]
    guard = summary["ghost_deconvolver_guard"]
    assert isinstance(naive, dict) and isinstance(guard, dict)
    assert naive["false_discovery_rate"] > 0.45
    assert guard["false_discovery_rate"] == 0.0
    assert guard["true_phase_recall"] >= 0.95
    assert guard["ambiguous_quarantined"] >= 20
