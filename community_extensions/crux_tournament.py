"""Elo Hypothesis Tournament & EVSI Crux-Experiment Synthesizer (Idea #01).

Inspired by Google AI Co-Scientist pairwise Elo tournament rankings and FutureHouse
Robin causal experiment loops. Maintains competing mechanistic hypotheses with
symmetric SUPPORT and CONTRADICT literature evidence, updates Elo ratings via
pairwise explanatory margins, enforces NOINFO abstention when retrieval scores fall
below threshold, and synthesizes the single wet-lab perturbation maximizing
Expected Value of Sample Information (EVSI) to resolve deadlocked scientific claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvidenceSnippet:
    """Retrieved primary-literature rationale sentence with explicit stance polarity."""

    doc_id: str
    text: str
    polarity: str  # "SUPPORT" or "CONTRADICT"
    bm25_score: float

    def __post_init__(self) -> None:
        if self.polarity not in ("SUPPORT", "CONTRADICT"):
            raise ValueError(f"Unsupported polarity: {self.polarity!r}")
        if self.bm25_score < 0.0 or not math.isfinite(self.bm25_score):
            raise ValueError("bm25_score must be finite and non-negative")


@dataclass(frozen=True)
class MechanisticHypothesis:
    """Competing mechanistic explanation backed by symmetric evidence pools."""

    hypothesis_id: str
    mechanism: str
    support_evidence: tuple[EvidenceSnippet, ...]
    contradict_evidence: tuple[EvidenceSnippet, ...]
    initial_elo: float = 1500.0


@dataclass(frozen=True)
class CandidatePerturbation:
    """Candidate wet-lab discriminating experiment with conditional readout likelihoods."""

    experiment_id: str
    name: str
    cost_usd: float
    likelihood_positive_by_hypothesis: Mapping[str, float]
    bm25_lexical_overlap: float = 0.0
    is_redundant_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.cost_usd < 0.0 or not math.isfinite(self.cost_usd):
            raise ValueError("cost_usd must be finite and non-negative")
        for hid, prob in self.likelihood_positive_by_hypothesis.items():
            if not (0.0 <= prob <= 1.0) or not math.isfinite(prob):
                raise ValueError(f"Invalid likelihood for {hid}: {prob}")


@dataclass(frozen=True)
class CruxSelectionResult:
    """Outcome of Elo debate and Bayesian EVSI crux-experiment synthesis."""

    status: (
        str  # "OK_CRUX_SELECTED", "ABSTAIN_NOINFO", or "NO_DISCRIMINATING_EXPERIMENT"
    )
    selected_experiment_id: str | None
    selected_experiment_name: str | None
    evsi_usd: float
    net_evsi_usd: float
    mutual_info_bits: float
    elo_ratings: dict[str, float]
    posterior_priors: dict[str, float]
    deadlock_detected: bool
    top_pair: tuple[str, str] | None


def _symmetric_evidence_strength(hyp: MechanisticHypothesis) -> float:
    """Compute stance-symmetric evidence score preserving both support and contradiction."""
    sup = sum(s.bm25_score for s in hyp.support_evidence)
    con = sum(s.bm25_score for s in hyp.contradict_evidence)
    return sup - 0.85 * con


def run_elo_hypothesis_tournament(
    hypotheses: Sequence[MechanisticHypothesis],
    k_factor: float = 32.0,
    n_rounds: int = 4,
    debate_scale: float = 12.0,
) -> dict[str, float]:
    """Update Elo ratings across deterministic round-robin symmetric evidence debates."""
    if not hypotheses:
        raise ValueError("hypotheses must be non-empty")
    if k_factor <= 0.0 or n_rounds < 1 or debate_scale <= 0.0:
        raise ValueError("k_factor, n_rounds, and debate_scale must be positive")

    ratings = {h.hypothesis_id: float(h.initial_elo) for h in hypotheses}
    strengths = {h.hypothesis_id: _symmetric_evidence_strength(h) for h in hypotheses}
    ordered_ids = [h.hypothesis_id for h in hypotheses]

    for _ in range(n_rounds):
        deltas = {hid: 0.0 for hid in ordered_ids}
        for idx_a in range(len(ordered_ids)):
            for idx_b in range(idx_a + 1, len(ordered_ids)):
                ha = ordered_ids[idx_a]
                hb = ordered_ids[idx_b]
                expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[hb] - ratings[ha]) / 400.0))
                margin = (strengths[ha] - strengths[hb]) / debate_scale
                margin_clipped = max(-30.0, min(30.0, margin))
                actual_a = 1.0 / (1.0 + math.exp(-margin_clipped))
                shift = k_factor * (actual_a - expected_a)
                deltas[ha] += shift
                deltas[hb] -= shift
        for hid in ordered_ids:
            ratings[hid] += deltas[hid]

    return ratings


def elo_to_prior(elo_ratings: Mapping[str, float]) -> dict[str, float]:
    """Convert Elo standings into a normalized Bradley-Terry prior over hypotheses."""
    if not elo_ratings:
        raise ValueError("elo_ratings must be non-empty")
    max_r = max(elo_ratings.values())
    raw = {hid: 10.0 ** ((r - max_r) / 400.0) for hid, r in elo_ratings.items()}
    total = sum(raw.values())
    return {hid: val / total for hid, val in raw.items()}


def compute_perturbation_evsi(
    priors: Mapping[str, float],
    perturbation: CandidatePerturbation,
    decision_value_usd: float = 1000.0,
) -> tuple[float, float, float]:
    """Compute (evsi_usd, net_evsi_usd, mutual_info_bits) for a candidate perturbation.

    Uses exact pre-posterior Bayes decision value:
        EVSI(e) = V * (sum_y max_h P(h, Y=y | e) - max_h P(h)) >= 0.
    """
    if decision_value_usd <= 0.0:
        raise ValueError("decision_value_usd must be positive")
    prior_sum = sum(priors.values())
    if prior_sum <= 0.0:
        raise ValueError("priors must sum to a positive value")

    norm_priors = {hid: p / prior_sum for hid, p in priors.items()}
    prior_best = decision_value_usd * max(norm_priors.values())

    joint_pos = {
        hid: norm_priors[hid]
        * float(perturbation.likelihood_positive_by_hypothesis[hid])
        for hid in norm_priors
    }
    joint_neg = {
        hid: norm_priors[hid]
        * (1.0 - float(perturbation.likelihood_positive_by_hypothesis[hid]))
        for hid in norm_priors
    }

    p_pos = sum(joint_pos.values())
    p_neg = sum(joint_neg.values())

    prepost_best = 0.0
    if p_pos > 1e-15:
        prepost_best += decision_value_usd * max(joint_pos.values())
    if p_neg > 1e-15:
        prepost_best += decision_value_usd * max(joint_neg.values())

    evsi_usd = max(0.0, prepost_best - prior_best)
    net_evsi_usd = evsi_usd - perturbation.cost_usd

    # Mutual information I(H; Y | e) in bits
    mi_bits = 0.0
    for hid, p_h in norm_priors.items():
        if p_h <= 1e-15:
            continue
        theta = float(perturbation.likelihood_positive_by_hypothesis[hid])
        if theta > 1e-15 and p_pos > 1e-15:
            mi_bits += p_h * theta * math.log2(theta / p_pos)
        if (1.0 - theta) > 1e-15 and p_neg > 1e-15:
            mi_bits += p_h * (1.0 - theta) * math.log2((1.0 - theta) / p_neg)
    mi_bits = max(0.0, mi_bits)

    return evsi_usd, net_evsi_usd, mi_bits


def synthesize_crux_experiment(
    hypotheses: Sequence[MechanisticHypothesis],
    candidates: Sequence[CandidatePerturbation],
    noinfo_threshold: float = 8.0,
    deadlock_elo_margin: float = 75.0,
    decision_value_usd: float = 1000.0,
    min_evsi_usd: float = 1.0,
) -> CruxSelectionResult:
    """Run Elo debate and synthesize the highest net-EVSI discriminating crux experiment."""
    if not hypotheses:
        raise ValueError("hypotheses must be non-empty")

    all_scores = [
        s.bm25_score
        for h in hypotheses
        for s in (*h.support_evidence, *h.contradict_evidence)
    ]
    top_score = max(all_scores) if all_scores else 0.0
    if top_score < noinfo_threshold or not all_scores:
        equal = 1.0 / len(hypotheses)
        return CruxSelectionResult(
            status="ABSTAIN_NOINFO",
            selected_experiment_id=None,
            selected_experiment_name=None,
            evsi_usd=0.0,
            net_evsi_usd=0.0,
            mutual_info_bits=0.0,
            elo_ratings={h.hypothesis_id: float(h.initial_elo) for h in hypotheses},
            posterior_priors={h.hypothesis_id: equal for h in hypotheses},
            deadlock_detected=False,
            top_pair=None,
        )

    elo_ratings = run_elo_hypothesis_tournament(hypotheses)
    priors = elo_to_prior(elo_ratings)
    ranked_hyps = sorted(elo_ratings.items(), key=lambda kv: kv[1], reverse=True)
    top_pair = (ranked_hyps[0][0], ranked_hyps[1][0]) if len(ranked_hyps) >= 2 else None

    total_sup = sum(len(h.support_evidence) for h in hypotheses)
    total_con = sum(len(h.contradict_evidence) for h in hypotheses)
    close_elo = (
        (ranked_hyps[0][1] - ranked_hyps[1][1]) <= deadlock_elo_margin
        if len(ranked_hyps) >= 2
        else False
    )
    deadlock_detected = bool((total_sup > 0 and total_con > 0) or close_elo)

    best_cand: CandidatePerturbation | None = None
    best_metrics = (0.0, -float("inf"), 0.0)

    for cand in candidates:
        evsi_usd, net_evsi_usd, mi_bits = compute_perturbation_evsi(
            priors, cand, decision_value_usd=decision_value_usd
        )
        if evsi_usd < min_evsi_usd:
            continue
        if (net_evsi_usd, mi_bits) > (best_metrics[1], best_metrics[2]):
            best_cand = cand
            best_metrics = (evsi_usd, net_evsi_usd, mi_bits)

    if best_cand is None:
        return CruxSelectionResult(
            status="NO_DISCRIMINATING_EXPERIMENT",
            selected_experiment_id=None,
            selected_experiment_name=None,
            evsi_usd=0.0,
            net_evsi_usd=0.0,
            mutual_info_bits=0.0,
            elo_ratings=elo_ratings,
            posterior_priors=priors,
            deadlock_detected=deadlock_detected,
            top_pair=top_pair,
        )

    return CruxSelectionResult(
        status="OK_CRUX_SELECTED",
        selected_experiment_id=best_cand.experiment_id,
        selected_experiment_name=best_cand.name,
        evsi_usd=best_metrics[0],
        net_evsi_usd=best_metrics[1],
        mutual_info_bits=best_metrics[2],
        elo_ratings=elo_ratings,
        posterior_priors=priors,
        deadlock_detected=deadlock_detected,
        top_pair=top_pair,
    )


def select_naive_bm25_experiment(
    hypotheses: Sequence[MechanisticHypothesis],
    candidates: Sequence[CandidatePerturbation],
) -> CandidatePerturbation:
    """Baseline selector ranking perturbations purely by lexical BM25 overlap."""
    if not hypotheses or not candidates:
        raise ValueError("hypotheses and candidates must be non-empty")
    return max(candidates, key=lambda c: (c.bm25_lexical_overlap, -c.cost_usd))


def _bayes_update_posterior(
    priors: Mapping[str, float],
    perturbation: CandidatePerturbation,
    observed_positive: bool,
) -> dict[str, float]:
    """Apply Bayes' rule after observing a binary wet-lab perturbation readout."""
    unnorm: dict[str, float] = {}
    for hid, p_h in priors.items():
        theta = float(perturbation.likelihood_positive_by_hypothesis[hid])
        likelihood = theta if observed_positive else (1.0 - theta)
        unnorm[hid] = p_h * likelihood
    denom = sum(unnorm.values())
    if denom <= 1e-15:
        equal = 1.0 / len(priors)
        return {hid: equal for hid in priors}
    return {hid: val / denom for hid, val in unnorm.items()}


def run_benchmark(n_scenarios: int = 40, seed: int = 42) -> dict[str, object]:
    """Benchmark `naive_top_bm25` vs `elo_evsi_crux` across 40 contested claims."""
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")
    rng = np.random.default_rng(seed)

    naive_resolved = 0
    naive_wasted = 0
    naive_evsi_list: list[float] = []

    crux_resolved = 0
    crux_wasted = 0
    crux_evsi_list: list[float] = []

    for idx in range(n_scenarios):
        # Two deadlocked lead mechanisms (H1, H2) and one weaker decoy mechanism (H3)
        true_hid = "H1" if (idx % 2 == 0) else "H2"
        s1_sup = float(rng.uniform(18.0, 24.0))
        s1_con = float(rng.uniform(14.0, 19.0))
        s2_sup = s1_sup + float(rng.uniform(-2.0, 2.0))
        s2_con = s1_con + float(rng.uniform(-2.0, 2.0))

        hypotheses = (
            MechanisticHypothesis(
                hypothesis_id="H1",
                mechanism="Allosteric kinase conformation lock",
                support_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_1",
                        "Kinase lock increases thermal shift.",
                        "SUPPORT",
                        s1_sup,
                    ),
                ),
                contradict_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_2",
                        "ATP-competitive washout restores rate.",
                        "CONTRADICT",
                        s1_con,
                    ),
                ),
            ),
            MechanisticHypothesis(
                hypothesis_id="H2",
                mechanism="Covalent cysteine adduct formation",
                support_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_3",
                        "Mass spec shift persists post-dialysis.",
                        "SUPPORT",
                        s2_sup,
                    ),
                ),
                contradict_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_4",
                        "Cysteine-to-serine mutant retains partial binding.",
                        "CONTRADICT",
                        s2_con,
                    ),
                ),
            ),
            MechanisticHypothesis(
                hypothesis_id="H3",
                mechanism="Non-specific colloidal aggregation",
                support_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_5",
                        "Minor detergent sensitivity at high dose.",
                        "SUPPORT",
                        6.5,
                    ),
                ),
                contradict_evidence=(
                    EvidenceSnippet(
                        f"doc_{idx}_6",
                        "Dynamic light scattering confirms monodisperse monomer.",
                        "CONTRADICT",
                        28.0,
                    ),
                ),
                initial_elo=1280.0,
            ),
        )

        # Candidate wet-lab perturbations:
        # E_confirm_1 & E_confirm_2 share high lexical BM25 overlap with H1/H2 abstracts
        # (e.g., repeating standard IC50 / thermal shift) where both H1 and H2 predict ~0.88!
        p_shared = float(rng.uniform(0.84, 0.91))
        candidates = (
            CandidatePerturbation(
                experiment_id="E_confirm_1",
                name="Repeat bulk IC50 dose-response",
                cost_usd=65.0,
                likelihood_positive_by_hypothesis={
                    "H1": p_shared,
                    "H2": p_shared - 0.02,
                    "H3": 0.35,
                },
                bm25_lexical_overlap=float(rng.uniform(26.0, 32.0)),
                is_redundant_confirmation=True,
            ),
            CandidatePerturbation(
                experiment_id="E_confirm_2",
                name="Standard thermal shift replication",
                cost_usd=70.0,
                likelihood_positive_by_hypothesis={
                    "H1": p_shared - 0.01,
                    "H2": p_shared,
                    "H3": 0.25,
                },
                bm25_lexical_overlap=float(rng.uniform(22.0, 27.0)),
                is_redundant_confirmation=True,
            ),
            CandidatePerturbation(
                experiment_id="E_noisy",
                name="Low-SNR cell viability screen",
                cost_usd=50.0,
                likelihood_positive_by_hypothesis={"H1": 0.58, "H2": 0.44, "H3": 0.50},
                bm25_lexical_overlap=float(rng.uniform(14.0, 18.0)),
                is_redundant_confirmation=False,
            ),
            CandidatePerturbation(
                experiment_id="E_crux_orthogonal",
                name="Rapid-dilution jump + C184S intact-MS washout",
                cost_usd=85.0,
                likelihood_positive_by_hypothesis={"H1": 0.94, "H2": 0.06, "H3": 0.50},
                bm25_lexical_overlap=float(rng.uniform(11.0, 16.0)),
                is_redundant_confirmation=False,
            ),
        )

        elo_ratings = run_elo_hypothesis_tournament(hypotheses)
        priors = elo_to_prior(elo_ratings)

        # 1. Naive top-BM25 lexical baseline
        naive_choice = select_naive_bm25_experiment(hypotheses, candidates)
        naive_evsi, _, _ = compute_perturbation_evsi(priors, naive_choice)
        naive_evsi_list.append(naive_evsi)
        if naive_choice.is_redundant_confirmation:
            naive_wasted += 1
        y_naive = bool(
            rng.random() < naive_choice.likelihood_positive_by_hypothesis[true_hid]
        )
        post_naive = _bayes_update_posterior(priors, naive_choice, y_naive)
        if (
            max(post_naive, key=post_naive.get) == true_hid
            and post_naive[true_hid] >= 0.75
        ):
            naive_resolved += 1

        # 2. Elo + EVSI crux synthesizer
        crux_res = synthesize_crux_experiment(hypotheses, candidates)
        crux_map = {c.experiment_id: c for c in candidates}
        assert crux_res.selected_experiment_id is not None
        crux_choice = crux_map[crux_res.selected_experiment_id]
        crux_evsi_list.append(crux_res.evsi_usd)
        if crux_choice.is_redundant_confirmation:
            crux_wasted += 1
        y_crux = bool(
            rng.random() < crux_choice.likelihood_positive_by_hypothesis[true_hid]
        )
        post_crux = _bayes_update_posterior(priors, crux_choice, y_crux)
        if (
            max(post_crux, key=post_crux.get) == true_hid
            and post_crux[true_hid] >= 0.75
        ):
            crux_resolved += 1

    return {
        "idea_id": "idea_01_crux_tournament",
        "n_scenarios": n_scenarios,
        "naive_top_bm25": {
            "deadlock_resolution_rate": round(naive_resolved / n_scenarios, 4),
            "wasted_redundant_experiments": naive_wasted,
            "mean_evsi_usd": round(float(np.mean(naive_evsi_list)), 2),
        },
        "elo_evsi_crux": {
            "deadlock_resolution_rate": round(crux_resolved / n_scenarios, 4),
            "wasted_redundant_experiments": crux_wasted,
            "mean_evsi_usd": round(float(np.mean(crux_evsi_list)), 2),
        },
    }
