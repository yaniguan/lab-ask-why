"""Analytical Precursor Ghost-Peak Deconvolver (Idea #02).

Inspired by the autonomous solid-state synthesis XRD characterization debate
(A-Lab precursor peak overlap analysis and CederGroupHub/alab_gpss_public).
Deconvolves observed 1D diffraction or optical spectra as non-negative linear
combinations of starting precursor reference patterns plus a putative novel
target phase, computing conservative upward (`ceil`) `precursor_overlap_permille`
and `residual_rwp_permille` under scale-invariant unit-peak reference normalization
in exact alignment with `LabAskWhyLean.AssayTelemetry` (`unidentifiable_spectrum_never_certified`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

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


def _ceil_permille(ratio: float) -> int:
    """Conservatively quantize a non-negative error/overlap ratio upward to integer permille."""
    val = float(ratio)
    if not math.isfinite(val):
        raise ValueError(f"Expected finite ratio for permille conversion, got {ratio!r}")
    if val <= 1e-12:
        return 0
    clamped = max(0.0, min(1.0, val))
    scaled = (Decimal(str(clamped)) * Decimal(1000)).to_integral_value(rounding=ROUND_CEILING)
    return max(0, min(1000, int(scaled)))


def _unit_peak_normalize(profile: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a non-negative 1D profile to unit L-infinity peak (`I_max = 1.0`)."""
    peak = float(np.max(profile)) if profile.size > 0 else 0.0
    if peak <= 1e-12:
        return np.zeros_like(profile, dtype=np.float64), 0.0
    return profile / peak, peak


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
    y_raw = np.maximum(0.0, np.asarray(observed, dtype=np.float64))
    t_raw = np.maximum(0.0, np.asarray(target_profile, dtype=np.float64))
    y_vec, y_peak = _unit_peak_normalize(y_raw)
    t_vec, t_peak = _unit_peak_normalize(t_raw)
    if y_peak <= 1e-12 or t_peak <= 1e-12:
        return False

    target_mask = t_vec >= peak_rel_height
    if not np.any(target_mask):
        return False

    matched = y_vec[target_mask] >= intensity_floor
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
    """Deconvolve starting precursors against a putative novel phase under unit-peak invariance."""
    obs_arr = np.asarray(observed, dtype=np.float64)
    tgt_arr = np.asarray(target_profile, dtype=np.float64)
    if not np.all(np.isfinite(obs_arr)) or not np.all(np.isfinite(tgt_arr)):
        raise ValueError("observed and target_profile must contain finite values")

    y_raw = np.maximum(0.0, obs_arr)
    t_raw = np.maximum(0.0, tgt_arr)
    if y_raw.ndim != 1 or t_raw.shape != y_raw.shape:
        raise ValueError(
            "observed and target_profile must be 1D arrays of equal length"
        )

    prec_raw_cols: list[np.ndarray] = []
    for p in precursor_profiles:
        p_arr = np.asarray(p, dtype=np.float64)
        if not np.all(np.isfinite(p_arr)):
            raise ValueError("precursor_profiles must contain finite values")
        if p_arr.shape != y_raw.shape:
            raise ValueError("All precursor profiles must match observed length")
        prec_raw_cols.append(np.maximum(0.0, p_arr))

    # Peak-normalize observed, precursor cards, and target reference card to unit L-inf peak
    # so standard CIF cards (I_max = 100 or 1000) behave identically to unit-scaled profiles.
    y_vec, y_peak = _unit_peak_normalize(y_raw)
    t_vec, t_peak = _unit_peak_normalize(t_raw)
    prec_cols = [_unit_peak_normalize(p_col)[0] for p_col in prec_raw_cols]

    if y_peak <= 1e-12:
        outcome = "execution_artifact" if not execution_clean else "certified_negative"
        return GhostDeconvolutionResult(
            precursor_weights=tuple(0.0 for _ in prec_cols),
            target_weight=0.0,
            precursor_overlap_permille=0,
            residual_rwp_permille=0,
            signal_above_activity_thresh=False,
            attribution_outcome=outcome,
            admit_to_memory=outcome in ("certified_positive", "certified_negative"),
            claim_blocked=False,
        )

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
    w_target = float(w_joint[-1]) if t_peak > 1e-12 else 0.0
    y_joint = joint_mat @ w_joint

    obs_norm = max(float(np.linalg.norm(y_vec)), 1e-12)
    res_norm = float(np.linalg.norm(y_vec - y_joint))
    residual_rwp_permille = _ceil_permille(res_norm / obs_norm)

    if t_peak > 1e-12:
        target_mask = t_vec >= 0.20
    else:
        target_mask = np.zeros_like(t_vec, dtype=bool)

    if np.any(target_mask):
        obs_in_target = float(np.sum(y_vec[target_mask]))
        prec_explained_in_target = float(
            np.sum(np.minimum(y_vec[target_mask], y_prec_only[target_mask]))
        )
        ref_floor = float(activity_weight_thresh * np.sum(t_vec[target_mask]))
        denom = max(obs_in_target, ref_floor, 1e-12)
        precursor_overlap_permille = _ceil_permille(prec_explained_in_target / denom)
        apparent_signal_ratio = obs_in_target / max(
            float(np.sum(t_vec[target_mask])), 1e-12
        )
    else:
        precursor_overlap_permille = 0
        apparent_signal_ratio = 0.0

    signal_above_thresh = bool(
        t_peak > 1e-12
        and (
            w_target >= activity_weight_thresh
            or apparent_signal_ratio >= activity_weight_thresh
        )
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
        p1_centers = [18.5, 31.2, 44.8, 59.0]
        p2_centers = [24.0, 36.6, 51.5, 66.2]
        p1 = _gaussian_peaks(grid, p1_centers, [1.0, 0.75, 0.60, 0.45])
        p2 = _gaussian_peaks(grid, p2_centers, [0.9, 1.0, 0.55, 0.40])

        if regime == 0:
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
