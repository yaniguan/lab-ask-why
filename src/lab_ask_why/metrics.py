"""Rationale retrieval metrics, positional baselines, and abstention calibration."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def hit_at(
    rankings: Sequence[Sequence[int]],
    pairs: Sequence[Mapping[str, Any]],
    k: int,
) -> float:
    """Binary Hit@k / Success@k: whether at least one gold sentence appears in `r[:k]`."""
    if not pairs or not rankings:
        return float("nan")
    hits = [bool(set(r[:k]) & set(p["gold"])) for r, p in zip(rankings, pairs)]
    return float(np.mean(hits)) if hits else float("nan")


# Backward-compatible alias while distinguishing Hit@k from fractional Recall@k
recall_at = hit_at


def sentence_recall_at(
    rankings: Sequence[Sequence[int]],
    pairs: Sequence[Mapping[str, Any]],
    k: int,
) -> float:
    """Standard fractional IR Recall@k: `|set(r[:k]) & gold| / |gold|` averaged over pairs."""
    if not pairs or not rankings:
        return float("nan")
    vals: list[float] = []
    for r, p in zip(rankings, pairs):
        gold = set(p["gold"])
        if not gold:
            continue
        vals.append(len(set(r[:k]) & gold) / len(gold))
    return float(np.mean(vals)) if vals else float("nan")


def rationale_set_recall_at(
    rankings: Sequence[Sequence[int]],
    pairs: Sequence[Mapping[str, Any]],
    k: int,
) -> float:
    """Official SciFact rationale-set completeness: whether any full minimal rationale set is in `r[:k]`."""
    if not pairs or not rankings:
        return float("nan")
    vals: list[bool] = []
    for r, p in zip(rankings, pairs):
        top_set = set(r[:k])
        rsets = p.get("rationale_sets")
        if rsets:
            vals.append(any( set(rs) <= top_set for rs in rsets ))
        elif p.get("gold"):
            vals.append(set(p["gold"]) <= top_set)
    return float(np.mean(vals)) if vals else float("nan")


def lead_rankings(pairs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Deterministic lead-k baseline `[0, 1, ..., n-1]` (abstract objective/intro bias)."""
    return [list(range(len(p["sents"]))) for p in pairs]


def conclusion_first_rankings(pairs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Deterministic conclusion-first (`tail-k`) baseline `[n-1, n-2, ..., 0]`."""
    return [list(range(len(p["sents"]) - 1, -1, -1)) for p in pairs]


def rank_scores(scores: np.ndarray, *, tie_seed: int | None = None) -> list[int]:
    """Rank sentence scores descending, optionally breaking exact ties with a seeded RNG."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return []
    if tie_seed is None:
        return [int(i) for i in np.argsort(-arr, kind="stable")]
    rng = np.random.default_rng(tie_seed)
    jitter = rng.random(arr.size)
    return [int(i) for i in np.lexsort((jitter, -arr))]


def compute_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Exact tie-aware Mann-Whitney U AUROC for `pos_scores` vs `neg_scores` (pure NumPy)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Pairwise comparison: 1.0 if pos > neg, 0.5 if pos == neg, 0.0 if pos < neg
    diff = pos[:, None] - neg[None, :]
    wins = float(np.sum(diff > 0.0)) + 0.5 * float(np.sum(diff == 0.0))
    return wins / float(pos.size * neg.size)


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[int]],
    pairs: Sequence[Mapping[str, Any]],
) -> float:
    """Mean Reciprocal Rank (MRR) of the highest-ranked gold rationale sentence per pair."""
    if not pairs or not rankings:
        return float("nan")
    rr_vals: list[float] = []
    for r, p in zip(rankings, pairs):
        gold = set(p["gold"])
        if not gold:
            continue
        rr = 0.0
        for rank_idx, sent_idx in enumerate(r, start=1):
            if sent_idx in gold:
                rr = 1.0 / float(rank_idx)
                break
        rr_vals.append(rr)
    return float(np.mean(rr_vals)) if rr_vals else float("nan")


def macro_f1_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> float:
    """Unweighted macro-averaged F1 across `labels` (or sorted union of observed classes)."""
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    active_labels = (
        list(labels)
        if labels is not None
        else sorted({str(y) for y in y_true} | {str(y) for y in y_pred})
    )
    if not active_labels:
        return float("nan")
    f1s: list[float] = []
    for lab in active_labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / float(denom)) if denom > 0 else 0.0)
    return float(np.mean(f1s))

