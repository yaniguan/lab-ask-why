"""Analytical Precursor Ghost-Peak Deconvolver (Idea #02).

Inspired by the autonomous solid-state synthesis XRD characterization debate
(A-Lab precursor peak overlap analysis and CederGroupHub/alab_gpss_public).
Deconvolves observed 1D diffraction or optical spectra as non-negative linear
combinations of starting precursor reference patterns plus a putative novel
target phase, computing `precursor_overlap_permille` and `residual_rwp_permille`
in exact alignment with `LabAskWhyLean.AssayTelemetry` (`unidentifiable_spectrum_never_certified`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GhostDeconvolutionResult:
    """Result of non-negative precursor + putative target spectral deconvolution."""

    precursor_weights: tuple[float, ...]
    target_weight: float
    precursor_overlap_permille: int
    residual_rwp_permille: int
    signal_above_activity_thresh: bool
    attribution_outcome: str
    admit_to_memory: bool
    claim_blocked: bool


def nnls_solve(
    design: np.ndarray,
    observed: np.ndarray,
    max_iter: int = 400,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve min_{w >= 0} ||design @ w - observed||_2^2 via projected coordinate descent."""
    a_mat = np.asarray(design, dtype=np.float64)
    y_vec = np.asarray(observed, dtype=np.float64)
    if a_mat.ndim != 2 or y_vec.ndim != 1 or a_mat.shape[0] != y_vec.shape[0]:
        raise ValueError(
            "Incompatible shapes for NNLS design matrix and observed vector"
        )

    n_cols = a_mat.shape[1]
    if n_cols == 0:
        return np.zeros(0, dtype=np.float64)

    gram = a_mat.T @ a_mat
    rhs = a_mat.T @ y_vec
    diag = np.diag(gram)
    weights = np.zeros(n_cols, dtype=np.float64)

    for _ in range(max_iter):
        max_change = 0.0
        for col in range(n_cols):
            if diag[col] <= 1e-15:
                continue
            grad_j = float(gram[col] @ weights - rhs[col])
            updated = max(0.0, float(weights[col] - grad_j / diag[col]))
            diff = abs(updated - float(weights[col]))
            max_change = max(max_change, diff)
            weights[col] = updated
        if max_change <= tol:
            break

    return weights


def naive_peak_match_claim(
    observed: np.ndarray,
    target_profile: np.ndarray,
    peak_rel_height: float = 0.25,
    intensity_floor: float = 0.20,
    match_fraction_thresh: float = 0.70,
) -> bool:
    """Naive phase matcher checking only if observed intensity appears at target Bragg windows."""
    y_vec = np.asarray(observed, dtype=np.float64)
    t_vec = np.asarray(target_profile, dtype=np.float64)
    t_max = float(np.max(t_vec)) if t_vec.size > 0 else 0.0
    if t_max <= 1e-12:
        return False

    target_mask = t_vec >= (peak_rel_height * t_max)
    if not np.any(target_mask):
        return False

    matched = y_vec[target_mask] >= (intensity_floor * t_max)
    return float(np.mean(matched)) >= match_fraction_thresh


def deconvolve_precursor_ghosts(
    observed: np.ndarray,
    precursor_profiles: Sequence[np.ndarray],
    target_profile: np.ndarray,
    *,
    execution_clean: bool = True,
    max_residual_permille: int = 100,
    max_overlap_permille: int = 150,
    activity_weight_thresh: float = 0.25,
) -> GhostDeconvolutionResult:
    """Deconvolve starting precursors against a putative novel phase and attribute outcome."""
    y_vec = np.maximum(0.0, np.asarray(observed, dtype=np.float64))
    t_vec = np.maximum(0.0, np.asarray(target_profile, dtype=np.float64))
    if y_vec.ndim != 1 or t_vec.shape != y_vec.shape:
        raise ValueError(
            "observed and target_profile must be 1D arrays of equal length"
        )

    prec_cols = [
        np.maximum(0.0, np.asarray(p, dtype=np.float64)) for p in precursor_profiles
    ]
    for p_col in prec_cols:
        if p_col.shape != y_vec.shape:
            raise ValueError("All precursor profiles must match observed length")

    if prec_cols:
        p_mat = np.column_stack(prec_cols)
        w_prec_only = nnls_solve(p_mat, y_vec)
        y_prec_only = p_mat @ w_prec_only
        joint_mat = np.column_stack([p_mat, t_vec])
    else:
        w_prec_only = np.zeros(0, dtype=np.float64)
        y_prec_only = np.zeros_like(y_vec)
        joint_mat = t_vec[:, None]

    w_joint = nnls_solve(joint_mat, y_vec)
    w_joint_prec = w_joint[:-1]
    w_target = float(w_joint[-1])
    y_joint = joint_mat @ w_joint

    obs_norm = max(float(np.linalg.norm(y_vec)), 1e-12)
    res_norm = float(np.linalg.norm(y_vec - y_joint))
    residual_rwp_permille = int(min(1000, round(1000.0 * res_norm / obs_norm)))

    t_max = max(float(np.max(t_vec)), 1e-12)
    target_mask = t_vec >= (0.20 * t_max)
    obs_in_target = float(np.sum(y_vec[target_mask]))
    prec_explained_in_target = float(
        np.sum(np.minimum(y_vec[target_mask], y_prec_only[target_mask]))
    )
    ref_floor = float(activity_weight_thresh * np.sum(t_vec[target_mask]))
    denom = max(obs_in_target, ref_floor, 1e-12)
    precursor_overlap_permille = int(
        min(1000, max(0, round(1000.0 * prec_explained_in_target / denom)))
    )

    # Apparent signal in target peak region (mirrors raw hit detector before attribution guard)
    apparent_signal_ratio = obs_in_target / max(
        float(np.sum(t_vec[target_mask])), 1e-12
    )
    signal_above_thresh = bool(
        w_target >= activity_weight_thresh
        or apparent_signal_ratio >= activity_weight_thresh
    )

    if not execution_clean:
        outcome = "execution_artifact"
    elif (
        residual_rwp_permille > max_residual_permille
        or precursor_overlap_permille > max_overlap_permille
    ):
        outcome = "ambiguous_characterization"
    elif signal_above_thresh:
        outcome = "certified_positive"
    else:
        outcome = "certified_negative"

    admit = outcome in ("certified_positive", "certified_negative")
    naive_would_claim = naive_peak_match_claim(y_vec, t_vec)
    claim_blocked = bool(naive_would_claim and outcome != "certified_positive")

    return GhostDeconvolutionResult(
        precursor_weights=tuple(round(float(v), 6) for v in w_joint_prec),
        target_weight=round(w_target, 6),
        precursor_overlap_permille=precursor_overlap_permille,
        residual_rwp_permille=residual_rwp_permille,
        signal_above_activity_thresh=signal_above_thresh,
        attribution_outcome=outcome,
        admit_to_memory=admit,
        claim_blocked=claim_blocked,
    )


def _gaussian_peaks(
    grid: np.ndarray,
    centers: Sequence[float],
    amplitudes: Sequence[float],
    sigma: float = 0.32,
) -> np.ndarray:
    profile = np.zeros_like(grid, dtype=np.float64)
    for center, amp in zip(centers, amplitudes):
        profile += float(amp) * np.exp(-0.5 * ((grid - float(center)) / sigma) ** 2)
    return profile


def run_benchmark(n_spectra: int = 60, seed: int = 42) -> dict[str, object]:
    """Benchmark `naive_peak_match` vs `ghost_deconvolver_guard` across 60 mixed spectra."""
    if n_spectra < 4:
        raise ValueError("n_spectra must be >= 4")
    rng = np.random.default_rng(seed)
    grid = np.linspace(10.0, 80.0, 350)

    naive_tp = 0
    naive_fp = 0
    guard_tp = 0
    guard_fp = 0
    true_positives_total = 0
    ambiguous_quarantined = 0

    for idx in range(n_spectra):
        regime = idx % 4
        # Base precursor peaks
        p1_centers = [18.5, 31.2, 44.8, 59.0]
        p2_centers = [24.0, 36.6, 51.5, 66.2]
        p1 = _gaussian_peaks(grid, p1_centers, [1.0, 0.75, 0.60, 0.45])
        p2 = _gaussian_peaks(grid, p2_centers, [0.9, 1.0, 0.55, 0.40])

        if regime == 0:
            # Genuine novel phase with distinct non-overlapping Bragg reflections
            true_positives_total += 1
            t_prof = _gaussian_peaks(
                grid, [21.4, 28.7, 40.3, 55.1], [1.0, 0.90, 0.80, 0.65]
            )
            w1 = float(rng.uniform(0.02, 0.08))
            w2 = float(rng.uniform(0.02, 0.08))
            wt = float(rng.uniform(0.75, 1.15))
            noise = rng.normal(0.0, 0.012, size=grid.shape)
            obs = np.maximum(0.0, w1 * p1 + w2 * p2 + wt * t_prof + noise)
            is_true_novel = True
        elif regime == 1:
            # A-Lab ghost-peak overlap trap: putative target shares peak positions with p1 + p2,
            # and run contains ONLY unreacted precursors (zero true novel phase!)
            t_prof = _gaussian_peaks(
                grid,
                [18.55, 31.15, 36.65, 51.45],
                [0.95, 0.70, 0.90, 0.50],
            )
            w1 = float(rng.uniform(0.75, 1.10))
            w2 = float(rng.uniform(0.75, 1.10))
            noise = rng.normal(0.0, 0.012, size=grid.shape)
            obs = np.maximum(0.0, w1 * p1 + w2 * p2 + noise)
            is_true_novel = False
        elif regime == 2:
            # Unidentified side-phase / impurity with partial coincidental overlap + high R_wp
            t_prof = _gaussian_peaks(grid, [24.05, 36.60, 48.2], [0.85, 0.90, 0.65])
            impurity = _gaussian_peaks(
                grid, [15.2, 29.8, 48.2, 62.7, 73.1], [0.9, 1.1, 0.8, 0.9, 0.7]
            )
            obs = np.maximum(
                0.0,
                0.65 * p2 + 0.85 * impurity + rng.normal(0.0, 0.015, size=grid.shape),
            )
            is_true_novel = False
        else:
            # Clean negative run: precursors unreacted, target lines completely absent & disjoint
            t_prof = _gaussian_peaks(
                grid, [14.2, 27.9, 41.6, 63.4], [1.0, 0.85, 0.75, 0.50]
            )
            obs = np.maximum(
                0.0, 0.80 * p1 + 0.70 * p2 + rng.normal(0.0, 0.010, size=grid.shape)
            )
            is_true_novel = False

        if naive_peak_match_claim(obs, t_prof):
            if is_true_novel:
                naive_tp += 1
            else:
                naive_fp += 1

        res = deconvolve_precursor_ghosts(obs, [p1, p2], t_prof)
        if res.attribution_outcome == "ambiguous_characterization":
            ambiguous_quarantined += 1
        if res.attribution_outcome == "certified_positive":
            if is_true_novel:
                guard_tp += 1
            else:
                guard_fp += 1

    naive_claims = max(1, naive_tp + naive_fp)
    guard_claims = max(1, guard_tp + guard_fp)

    return {
        "idea_id": "idea_02_precursor_ghost_guard",
        "n_spectra": n_spectra,
        "naive_peak_match": {
            "true_positives": naive_tp,
            "false_positives": naive_fp,
            "false_discovery_rate": round(naive_fp / naive_claims, 4),
            "true_phase_recall": round(naive_tp / max(1, true_positives_total), 4),
        },
        "ghost_deconvolver_guard": {
            "true_positives": guard_tp,
            "false_positives": guard_fp,
            "false_discovery_rate": round(guard_fp / guard_claims, 4),
            "true_phase_recall": round(guard_tp / max(1, true_positives_total), 4),
            "ambiguous_quarantined": ambiguous_quarantined,
        },
    }
