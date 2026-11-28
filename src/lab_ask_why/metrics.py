"""Rationale retrieval metrics, official SciFact evaluators, Wilson CIs, and baselines."""
from __future__ import annotations

from collections import Counter
import math
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


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Exact 95% Wilson score confidence interval `[low, high]` for a binomial proportion."""
    n = int(total)
    if n <= 0:
        return float("nan"), float("nan")
    k = max(0, min(n, int(successes)))
    p_hat = float(k) / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    half = (
        float(z)
        * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n))) / float(n))
        / denom
    )
    return max(0.0, min(1.0, center - half)), max(0.0, min(1.0, center + half))


def majority_class_stance_baseline(
    y_true: Sequence[str],
    *,
    eval_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Constant majority-class predictor accuracy and macro-F1 over `eval_labels`."""
    if not y_true:
        return {
            "majority_label": "",
            "majority_class_accuracy": float("nan"),
            "majority_class_macro_f1": float("nan"),
        }
    active = list(eval_labels) if eval_labels is not None else sorted({str(y) for y in y_true})
    counts = Counter(str(y) for y in y_true)
    # Deterministic tie break along active label order
    maj_label = max(active, key=lambda lab: (counts.get(lab, 0), -active.index(lab)))
    y_const = [maj_label] * len(y_true)
    acc = float(counts.get(maj_label, 0)) / float(len(y_true))
    mf1 = macro_f1_score(y_true, y_const, labels=active)
    return {
        "majority_label": maj_label,
        "majority_class_accuracy": acc,
        "majority_class_macro_f1": mf1,
    }


def chance_mcq_baseline(option_counts: Sequence[int]) -> dict[str, float]:
    """Exact uniform random chance `Hit@1` and `MRR` across MCQ items with `K_i` options."""
    valid = [int(k) for k in option_counts if int(k) >= 1]
    if not valid:
        return {
            "chance_hit@1": float("nan"),
            "chance_mrr": float("nan"),
            "mean_num_options": float("nan"),
        }
    hit1_vals = [1.0 / float(k) for k in valid]
    mrr_vals = [sum(1.0 / float(r) for r in range(1, k + 1)) / float(k) for k in valid]
    return {
        "chance_hit@1": float(np.mean(hit1_vals)),
        "chance_mrr": float(np.mean(mrr_vals)),
        "mean_num_options": float(np.mean(valid)),
    }


def _safe_prf1(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    prec = (float(tp) / float(n_pred)) if n_pred > 0 else 0.0
    rec = (float(tp) / float(n_gold)) if n_gold > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def compute_official_scifact_metrics(
    claims: Sequence[Mapping[str, Any]],
    predictions: Mapping[Any, Mapping[Any, Mapping[str, Any]]],
    *,
    max_sentences_per_doc: int = 3,
) -> dict[str, dict[str, float]]:
    """Compute official Wadden et al. (EMNLP 2020) VeriSci SciFact abstract & sentence F1 metrics.

    Parameters
    ----------
    claims:
        Gold SciFact claim records (`id`, `evidence`: `doc_id -> [{"sentences": [...], "label": ...}]`).
    predictions:
        Mapping `claim_id -> {doc_id -> {"label": "SUPPORT"|"CONTRADICT", "sentences": [int, ...]}}`.
        Abstracts predicted as `NEI` should be omitted (or have `label == "NEI"`, which is ignored).
    max_sentences_per_doc:
        Official SciFact cap on predicted rationale sentences per abstract (default `3`).

    Returns
    -------
    Dictionary with `abstract_label_only`, `abstract_rationales`, `sentence_selection`, and
    `sentence_label` `{precision, recall, f1}` scores across all claims (including `NOINFO`).
    """
    abs_label_tp = 0
    abs_rat_tp = 0
    abs_pred_total = 0
    abs_gold_total = 0

    sent_sel_pred_hits = 0
    sent_lab_pred_hits = 0
    sent_pred_total = 0
    sent_sel_gold_hits = 0
    sent_lab_gold_hits = 0
    sent_gold_total = 0

    # Normalize top-level prediction keys to str for int/str claim_id lookup invariance
    norm_preds: dict[str, dict[str, Mapping[str, Any]]] = {}
    for cid, doc_map in predictions.items():
        if isinstance(doc_map, Mapping):
            norm_preds[str(cid)] = {str(did): v for did, v in doc_map.items() if isinstance(v, Mapping)}

    for claim in claims:
        cid_str = str(claim.get("id"))
        raw_ev = claim.get("evidence") or {}
        gold_docs: dict[str, tuple[str, list[frozenset[int]], frozenset[int]]] = {}
        if isinstance(raw_ev, Mapping):
            for raw_did, raw_sets in raw_ev.items():
                entries = [raw_sets] if isinstance(raw_sets, Mapping) else list(raw_sets or ())
                rsets: list[frozenset[int]] = []
                g_lab = ""
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    lab_raw = str(entry.get("label") or "").strip().upper()
                    if lab_raw in {"SUPPORT", "SUPPORTS"}:
                        g_lab = "SUPPORT"
                    elif lab_raw in {"CONTRADICT", "CONTRADICTS"}:
                        g_lab = "CONTRADICT"
                    s_idxs = frozenset(int(x) for x in (entry.get("sentences") or ()))
                    if s_idxs:
                        rsets.append(s_idxs)
                if g_lab in {"SUPPORT", "CONTRADICT"} and rsets:
                    union_s = frozenset().union(*rsets)
                    gold_docs[str(raw_did)] = (g_lab, rsets, union_s)

        abs_gold_total += len(gold_docs)
        for _, (_, _, union_s) in gold_docs.items():
            sent_gold_total += len(union_s)

        pred_docs = norm_preds.get(cid_str, {})
        active_preds: dict[str, tuple[str, list[int], frozenset[int]]] = {}
        for did_str, p_entry in pred_docs.items():
            p_lab_raw = str(p_entry.get("label") or "").strip().upper()
            if p_lab_raw in {"SUPPORT", "SUPPORTS"}:
                p_lab = "SUPPORT"
            elif p_lab_raw in {"CONTRADICT", "CONTRADICTS"}:
                p_lab = "CONTRADICT"
            else:
                continue
            dedup_sents: list[int] = []
            for s in p_entry.get("sentences") or ():
                si = int(s)
                if si not in dedup_sents:
                    dedup_sents.append(si)
            capped = dedup_sents[: max(1, int(max_sentences_per_doc))]
            active_preds[did_str] = (p_lab, capped, frozenset(capped))

        abs_pred_total += len(active_preds)
        for did_str, (p_lab, capped, p_set) in active_preds.items():
            sent_pred_total += len(p_set)
            g_info = gold_docs.get(did_str)
            if g_info is None:
                continue
            g_lab, rsets, _ = g_info
            label_ok = p_lab == g_lab
            if label_ok:
                abs_label_tp += 1
            contains_full_rset = any(rset <= p_set for rset in rsets)
            if label_ok and p_set and contains_full_rset:
                abs_rat_tp += 1

            for s_idx in p_set:
                if any((s_idx in rset) and (rset <= p_set) for rset in rsets):
                    sent_sel_pred_hits += 1
                    if label_ok:
                        sent_lab_pred_hits += 1

        for did_str, (g_lab, rsets, union_s) in gold_docs.items():
            p_info = active_preds.get(did_str)
            if p_info is None:
                continue
            p_lab, _, p_set = p_info
            label_ok = p_lab == g_lab
            for s_idx in union_s:
                if any((s_idx in rset) and (rset <= p_set) for rset in rsets):
                    sent_sel_gold_hits += 1
                    if label_ok:
                        sent_lab_gold_hits += 1

    # Sentence-level precision uses predicted hits / total predicted sentences;
    # sentence-level recall uses recalled gold sentences / total gold sentences (VeriSci exact rule).
    s_sel_prec = (float(sent_sel_pred_hits) / float(sent_pred_total)) if sent_pred_total > 0 else 0.0
    s_sel_rec = (float(sent_sel_gold_hits) / float(sent_gold_total)) if sent_gold_total > 0 else 0.0
    s_sel_f1 = (
        (2.0 * s_sel_prec * s_sel_rec / (s_sel_prec + s_sel_rec))
        if (s_sel_prec + s_sel_rec) > 0.0
        else 0.0
    )

    s_lab_prec = (float(sent_lab_pred_hits) / float(sent_pred_total)) if sent_pred_total > 0 else 0.0
    s_lab_rec = (float(sent_lab_gold_hits) / float(sent_gold_total)) if sent_gold_total > 0 else 0.0
    s_lab_f1 = (
        (2.0 * s_lab_prec * s_lab_rec / (s_lab_prec + s_lab_rec))
        if (s_lab_prec + s_lab_rec) > 0.0
        else 0.0
    )

    return {
        "abstract_label_only": _safe_prf1(abs_label_tp, abs_pred_total, abs_gold_total),
        "abstract_rationales": _safe_prf1(abs_rat_tp, abs_pred_total, abs_gold_total),
        "sentence_selection": {
            "precision": s_sel_prec,
            "recall": s_sel_rec,
            "f1": s_sel_f1,
        },
        "sentence_label": {
            "precision": s_lab_prec,
            "recall": s_lab_rec,
            "f1": s_lab_f1,
        },
    }

