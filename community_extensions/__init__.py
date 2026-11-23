"""Community research extensions for lab-ask-why (L6 epistemic co-scientist).

Modules:
- crux_tournament: Elo Hypothesis Tournament & EVSI Crux-Experiment Synthesizer (Idea #01).
- precursor_ghost_guard: Analytical Precursor Ghost-Peak Deconvolver (Idea #02).
"""

from __future__ import annotations

from .crux_tournament import (
    CandidatePerturbation,
    CruxSelectionResult,
    EvidenceSnippet,
    MechanisticHypothesis,
    compute_perturbation_evsi,
    elo_to_prior,
    run_elo_hypothesis_tournament,
    select_naive_bm25_experiment,
    synthesize_crux_experiment,
)
from .crux_tournament import (
    run_benchmark as run_crux_tournament_benchmark,
)
from .precursor_ghost_guard import (
    GhostDeconvolutionResult,
    deconvolve_precursor_ghosts,
    naive_peak_match_claim,
    nnls_solve,
)
from .precursor_ghost_guard import (
    run_benchmark as run_precursor_ghost_benchmark,
)

__all__ = [
    "CandidatePerturbation",
    "CruxSelectionResult",
    "EvidenceSnippet",
    "GhostDeconvolutionResult",
    "MechanisticHypothesis",
    "compute_perturbation_evsi",
    "deconvolve_precursor_ghosts",
    "elo_to_prior",
    "naive_peak_match_claim",
    "nnls_solve",
    "run_crux_tournament_benchmark",
    "run_elo_hypothesis_tournament",
    "run_precursor_ghost_benchmark",
    "select_naive_bm25_experiment",
    "synthesize_crux_experiment",
]
