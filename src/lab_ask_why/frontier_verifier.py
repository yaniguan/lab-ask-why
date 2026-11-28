"""Frontier 2-stage hybrid retrieval, cross-feature reranker, and split-conformal stance verifier.

Architecture (Layer L6 Epistemic Brain):
1. Stage 1 Hybrid Retrieval (`Reciprocal Rank Fusion`, RRF):
   - Channel 1: Biomedical Okapi BM25 (`BM25`, zero-parameter lexical baseline).
   - Channel 2: Subword character (3/4-gram) + biomedical word/concept TF-IDF
     Truncated-SVD (`LSA` dense semantic subspace, fit strictly on background docs).
   - Channel 3: Entity / acronym / allele + numeric + negation/directional overlap index.
2. Stage 2 Pairwise Claim-Sentence Interaction Reranker & 3-Way Stance Classifier:
   - Cross-feature reranker scoring candidate sentences via negation polarity alignment,
     quantitative directionality (`increase`/`decrease`/`inhibit`/`activate`), lexical-dense
     cosine agreement, within-pool BM25 & LSA z-scores, rare-entity coverage, title-context
     anaphora rescue, and abstract discourse position.
   - 3-way stance classifier over `("SUPPORT", "CONTRADICT", "NEI")` trained strictly on
     `SciFact train` (`809` claims: deterministic `train_fit` + held-out `train_calib` split,
     zero leakage from `SciFact dev` or external evaluation benchmarks).
   - Calibrated via Split Conformal Prediction (`1 - alpha = 0.90` prediction sets over
     `{"SUPPORT", "CONTRADICT", "NEI"}`) so genuinely ambiguous or conflicting claims emit
     multi-label epistemic sets such as `{"SUPPORT", "CONTRADICT"}` rather than forced
     binary verdict collapse.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy import sparse as _sp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - pure-NumPy fallback path
    _sp = None

from lab_ask_why.bm25 import BM25, normalize_biomedical_text, tokenize
from lab_ask_why.data import build_noinfo_pairs, build_pairs
from lab_ask_why.metrics import (
    compute_auroc,
    compute_official_scifact_metrics,
    hit_at,
    macro_f1_score,
    majority_class_stance_baseline,
    mean_reciprocal_rank,
    rank_scores,
    rationale_set_recall_at,
    sentence_recall_at,
    wilson_score_interval,
)
from lab_ask_why.retrieval import (
    _NEGATION_TOKENS,
    _TOKEN_TO_CONCEPTS,
    _extract_main_assertion_clause,
    _normalize_orthography_and_stem,
    _reciprocal_rank_fusion,
    _strip_unreferenced_without_adjuncts,
    _SubwordConceptCosineIndex,
    _token_similarity,
    classify_rationale_polarity,
)

STANCE_LABELS: tuple[str, str, str] = ("SUPPORT", "CONTRADICT", "NEI")
_STANCE_TO_IDX: dict[str, int] = {lab: idx for idx, lab in enumerate(STANCE_LABELS)}

# Quantitative and mechanistic directionality groups (domain-general scientific opposites).
_DIRECTIONAL_OPPOSITES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset(
            {
                "increase",
                "increases",
                "increased",
                "increasing",
                "elevate",
                "elevates",
                "elevated",
                "raise",
                "raises",
                "raised",
                "higher",
                "high",
                "upregulate",
                "upregulates",
                "upregulated",
                "enhance",
                "enhances",
                "enhanced",
                "promote",
                "promotes",
                "promoted",
                "accelerate",
                "accelerates",
                "accelerated",
                "amplify",
                "amplifies",
                "amplified",
                "stimulate",
                "stimulates",
                "stimulated",
                "activate",
                "activates",
                "activated",
                "induce",
                "induces",
                "induced",
                "prolong",
                "prolongs",
                "prolonged",
                "expand",
                "expands",
                "expanded",
                "greater",
                "superior",
                "gain",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreases",
                "decreased",
                "decreasing",
                "reduce",
                "reduces",
                "reduced",
                "reducing",
                "lower",
                "low",
                "downregulate",
                "downregulates",
                "downregulated",
                "inhibit",
                "inhibits",
                "inhibited",
                "suppress",
                "suppresses",
                "suppressed",
                "attenuate",
                "attenuates",
                "attenuated",
                "repress",
                "represses",
                "repressed",
                "inactivate",
                "inactivates",
                "inactivated",
                "prevent",
                "prevents",
                "prevented",
                "abolish",
                "abolishes",
                "abolished",
                "block",
                "blocks",
                "blocked",
                "shorten",
                "shortens",
                "shortened",
                "less",
                "fewer",
                "inferior",
                "poorer",
                "impair",
                "impairs",
                "impaired",
                "diminish",
                "diminishes",
                "diminished",
                "deplete",
                "depletes",
                "depleted",
                "loss",
            }
        ),
    ),
    (
        frozenset(
            {
                "vulnerability",
                "vulnerable",
                "susceptibility",
                "susceptible",
                "risk",
                "harm",
                "harmful",
                "toxic",
                "toxicity",
                "severe",
                "mortality",
                "lethal",
                "fatal",
                "pathogenic",
                "worsen",
                "worsens",
                "exacerbate",
                "exacerbates",
                "damage",
                "destabilize",
                "hydrolysis",
                "hydrolyze",
                "hydrolyzed",
            }
        ),
        frozenset(
            {
                "protection",
                "protective",
                "protect",
                "protects",
                "protected",
                "resistance",
                "resistant",
                "survival",
                "benefit",
                "beneficial",
                "safe",
                "safety",
                "tolerated",
                "ameliorate",
                "ameliorates",
                "mitigate",
                "mitigates",
                "rescue",
                "rescues",
                "stabilize",
                "stabilizes",
                "preserve",
                "preserves",
            }
        ),
    ),
    (
        frozenset(
            {
                "sufficient",
                "necessary",
                "essential",
                "required",
                "causes",
                "caused",
                "effective",
                "efficacious",
                "present",
                "significant",
                "dependent",
                "correlates",
                "associated",
            }
        ),
        frozenset(
            {
                "dispensable",
                "unnecessary",
                "independent",
                "unrelated",
                "benign",
                "ineffective",
                "absent",
                "insignificant",
                "unchanged",
                "unaffected",
                "insufficient",
            }
        ),
    ),
)

_ALL_DIRECTIONAL_TOKENS: frozenset[str] = frozenset(
    tok for pos_g, neg_g in _DIRECTIONAL_OPPOSITES for tok in (pos_g | neg_g)
)

_CONCLUSION_CUES = frozenset(
    {
        "conclusion",
        "conclusions",
        "results",
        "findings",
        "summary",
        "demonstrate",
        "demonstrates",
        "demonstrated",
        "indicate",
        "indicates",
        "indicated",
        "reveal",
        "reveals",
        "revealed",
        "confirm",
        "confirms",
        "confirmed",
        "collectively",
        "overall",
        "therefore",
        "thus",
    }
)
_INTRO_CUES = frozenset(
    {
        "background",
        "objective",
        "objectives",
        "introduction",
        "aim",
        "aims",
        "purpose",
        "investigate",
        "investigated",
        "examined",
        "explored",
    }
)
_CONTRASTIVE_PHRASE_RE = re.compile(
    r"\b(?:however|in\s+contrast|contrary\s+to|unlike|rather\s+than|instead\s+of|"
    r"failed\s+to|fails\s+to|did\s+not|does\s+not|do\s+not|was\s+not|were\s+not|"
    r"no\s+significant|no\s+evidence|not\s+associated|no\s+association|no\s+difference|"
    r"no\s+effect|neither)\b",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_ENTITY_ACRONYM_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9+-]*|[A-Za-z]+\d+[A-Za-z0-9+-]*|\d+[A-Za-z]+[0-9+-]*)\b"
)


def canonicalize_stance_label(raw_label: Any) -> str:
    """Canonicalize upstream label strings onto `('SUPPORT', 'CONTRADICT', 'NEI')`."""
    if raw_label is None:
        return "NEI"
    norm = str(raw_label).strip().upper().replace("-", "_").replace(" ", "_")
    if norm in {"SUPPORT", "SUPPORTS", "SUPPORTED", "ENTAIL", "ENTAILED", "TRUE", "YES"}:
        return "SUPPORT"
    if norm in {"CONTRADICT", "CONTRADICTS", "CONTRADICTED", "REFUTE", "REFUTED", "FALSE", "NO"}:
        return "CONTRADICT"
    if norm in {"NEI", "NOINFO", "NO_INFO", "NOT_ENOUGH_INFO", "MAYBE", "INSUFFICIENT_EVIDENCE", ""}:
        return "NEI"
    raise ValueError(f"Unrecognized stance label: {raw_label!r}")


@functools.lru_cache(maxsize=65536)
def _token_svd_bundle(tok: str) -> tuple[tuple[str, int], ...]:
    """Cached per-token word, stem, biomedical synonym, and char 3/4-gram bundle."""
    items: Counter[str] = Counter()
    items[f"w:{tok}"] += 2
    stem = _normalize_orthography_and_stem(tok)
    items[f"st:{stem}"] += 2
    for cid in _TOKEN_TO_CONCEPTS.get(tok, ()):
        items[f"sy:{cid}"] += 3
    for cid in _TOKEN_TO_CONCEPTS.get(stem, ()):
        items[f"sy:{cid}"] += 3
    if len(stem) >= 4 and not stem[0].isdigit():
        padded = f"<{stem}>"
        for i in range(len(padded) - 3 + 1):
            items[f"c3:{padded[i : i + 3]}"] += 1
        if len(padded) >= 5:
            for i in range(len(padded) - 4 + 1):
                items[f"c4:{padded[i : i + 4]}"] += 1
    return tuple(items.items())


def _extract_svd_features(text: str) -> Counter[str]:
    """Extract word tokens, biomedical stems/synonyms, and subword char 3/4-grams for LSA."""
    feats: Counter[str] = Counter()
    if not text:
        return feats
    toks = tokenize(text)
    for tok in toks:
        for k, v in _token_svd_bundle(tok):
            feats[k] += v
    # Capture adjacent content-stem bigrams for phrase semantics in SVD
    stems = [_normalize_orthography_and_stem(t) for t in toks if t not in _NEGATION_TOKENS]
    for i in range(len(stems) - 1):
        feats[f"bg:{stems[i]}_{stems[i + 1]}"] += 1
    return feats


_CSR_BLOCK_ROWS = 2048
_CSR_MAX_BLOCK_NNZ = 262144


def _csr_matmul(
    n_rows: int,
    n_cols: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    dense_rhs: np.ndarray,
    *,
    sp_mat: Any | None = None,
    block_rows: int = _CSR_BLOCK_ROWS,
    max_block_nnz: int = _CSR_MAX_BLOCK_NNZ,
) -> np.ndarray:
    """Compute `CSR @ dense_rhs` identically via `scipy.sparse` or bounded-nnz blocked NumPy."""
    if sp_mat is not None:
        return np.asarray(sp_mat @ dense_rhs, dtype=np.float64)
    out = np.zeros((n_rows, dense_rhs.shape[1]), dtype=np.float64)
    if indices.size == 0 or n_rows == 0:
        return out
    step = max(1, int(block_rows))
    nnz_cap = max(1024, int(max_block_nnz))
    r0 = 0
    while r0 < n_rows:
        p0 = int(indptr[r0])
        r_by_rows = min(n_rows, r0 + step)
        if int(indptr[r_by_rows]) - p0 > nnz_cap:
            r_by_nnz = int(np.searchsorted(indptr, p0 + nnz_cap, side="right")) - 1
            r1 = max(r0 + 1, min(r_by_rows, r_by_nnz))
        else:
            r1 = r_by_rows
        p1 = int(indptr[r1])
        if p1 > p0:
            sub_ptr = indptr[r0 : r1 + 1] - p0
            non_empty = np.diff(sub_ptr) > 0
            if np.any(non_empty):
                starts = sub_ptr[:-1][non_empty]
                weighted = data[p0:p1, None] * dense_rhs[indices[p0:p1]]
                out[r0:r1][non_empty] = np.add.reduceat(weighted, starts, axis=0)
        r0 = r1
    return out


def _csr_transpose_arrays(
    n_rows: int,
    n_cols: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert `(n_rows, n_cols)` CSR arrays into transposed `(n_cols, n_rows)` CSR arrays in O(nnz)."""
    if indices.size == 0:
        return (
            np.zeros(n_cols + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(indptr))
    order = np.argsort(indices, kind="stable")
    t_indices = row_ids[order]
    t_data = data[order]
    col_counts = np.bincount(indices[order], minlength=n_cols)
    t_indptr = np.zeros(n_cols + 1, dtype=np.int32)
    np.cumsum(col_counts, out=t_indptr[1:])
    return t_indptr, t_indices, t_data


def _csr_rmatmul(
    n_rows: int,
    n_cols: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    dense_lhs_t: np.ndarray,
    *,
    sp_mat: Any | None = None,
    block_rows: int = _CSR_BLOCK_ROWS,
    t_csr: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute `CSR.T @ dense_lhs_t` identically via `scipy.sparse` or bounded-memory blocked NumPy."""
    if sp_mat is not None:
        return np.asarray(sp_mat.T @ dense_lhs_t, dtype=np.float64)
    if indices.size == 0 or n_rows == 0 or n_cols == 0:
        return np.zeros((n_cols, dense_lhs_t.shape[1]), dtype=np.float64)
    if t_csr is None:
        t_indptr, t_indices, t_data = _csr_transpose_arrays(n_rows, n_cols, indptr, indices, data)
    else:
        t_indptr, t_indices, t_data = t_csr
    return _csr_matmul(
        n_cols,
        n_rows,
        t_indptr,
        t_indices,
        t_data,
        dense_lhs_t,
        sp_mat=None,
        block_rows=block_rows,
    )


class TruncatedSVDSemanticIndex:
    """Subword char/word TF-IDF Truncated-SVD (`LSA`) dense semantic subspace index.

    Fits vocabulary IDF and right singular vectors strictly on background documents
    when `idf_mask` is supplied so evaluation documents never leak into the subspace.
    """

    def __init__(
        self,
        texts: Sequence[str],
        *,
        n_components: int = 96,
        max_features: int = 20000,
        n_iter: int = 3,
        seed: int = 42,
        idf_mask: Sequence[bool] | None = None,
    ) -> None:
        self.num_docs = len(texts)
        self.requested_components = int(n_components)
        self.seed = int(seed)

        all_feats = [_extract_svd_features(t) for t in texts]
        if idf_mask is not None and any(idf_mask):
            bg_indices = [i for i, keep in enumerate(idf_mask) if keep]
        else:
            bg_indices = list(range(self.num_docs))

        n_bg = max(1, len(bg_indices))
        df: Counter[str] = Counter()
        for idx in bg_indices:
            for term in all_feats[idx].keys():
                df[term] += 1

        min_df = 2 if n_bg >= 60 else 1
        eligible = [(term, count) for term, count in df.items() if count >= min_df]
        if not eligible and df:
            eligible = list(df.items())

        # Sort deterministically: prefer informative mid-frequency features, break ties alphabetically
        eligible.sort(key=lambda kv: (-kv[1], kv[0]))
        if len(eligible) > max_features:
            eligible = eligible[:max_features]

        self.vocab: dict[str, int] = {term: idx for idx, (term, _) in enumerate(eligible)}
        v_size = len(self.vocab)
        self.idf = np.zeros(max(1, v_size), dtype=np.float64)
        for term, idx in self.vocab.items():
            d_val = float(df[term])
            self.idf[idx] = math.log(1.0 + (float(n_bg) - d_val + 0.5) / (d_val + 0.5))

        if v_size == 0 or self.num_docs == 0:
            self.n_components = 1
            self.components = np.ones((1, 1), dtype=np.float64)
            self.embeddings = np.zeros((self.num_docs, 1), dtype=np.float64)
            return

        all_indptr, all_idx, all_val = self._build_csr_arrays(all_feats)
        all_sp = (
            _sp.csr_matrix((all_val, all_idx, all_indptr), shape=(self.num_docs, v_size))
            if _sp is not None
            else None
        )

        if len(bg_indices) == self.num_docs:
            bg_indptr, bg_indices_arr, bg_data, bg_sp = all_indptr, all_idx, all_val, all_sp
        elif all_sp is not None:
            bg_sp = all_sp[bg_indices].tocsr()
            bg_indptr, bg_indices_arr, bg_data = bg_sp.indptr, bg_sp.indices, bg_sp.data
        else:
            bg_feats = [all_feats[i] for i in bg_indices]
            bg_indptr, bg_indices_arr, bg_data = self._build_csr_arrays(bg_feats)
            del bg_feats
            bg_sp = None

        # Free Python Counter feature dicts prior to randomized SVD iterations to cap peak RSS
        del all_feats
        del df

        k = max(1, min(int(n_components), len(bg_indices) - 1 if len(bg_indices) > 1 else 1, v_size))
        self.n_components = k

        if min(len(bg_indices), v_size) <= 64 and k >= min(len(bg_indices), v_size):
            # Small-matrix exact SVD path for tiny unit-test corpora
            dense_bg = _csr_matmul(
                len(bg_indices),
                v_size,
                bg_indptr,
                bg_indices_arr,
                bg_data,
                np.eye(v_size, dtype=np.float64),
                sp_mat=bg_sp,
            )
            _, _, vt = np.linalg.svd(dense_bg, full_matrices=False)
            vt_k = vt[:k].copy()
        else:
            oversample = min(12, max(2, v_size - k))
            sketch_dim = min(v_size, k + oversample)
            rng = np.random.default_rng(self.seed)
            omega = rng.standard_normal((v_size, sketch_dim))
            bg_t_csr = (
                _csr_transpose_arrays(len(bg_indices), v_size, bg_indptr, bg_indices_arr, bg_data)
                if bg_sp is None
                else None
            )
            y_mat = _csr_matmul(
                len(bg_indices), v_size, bg_indptr, bg_indices_arr, bg_data, omega, sp_mat=bg_sp
            )
            q_mat, _ = np.linalg.qr(y_mat, mode="reduced")
            for _ in range(max(1, int(n_iter))):
                z_mat = _csr_rmatmul(
                    len(bg_indices),
                    v_size,
                    bg_indptr,
                    bg_indices_arr,
                    bg_data,
                    q_mat,
                    sp_mat=bg_sp,
                    t_csr=bg_t_csr,
                )
                z_orth, _ = np.linalg.qr(z_mat, mode="reduced")
                y_mat = _csr_matmul(
                    len(bg_indices), v_size, bg_indptr, bg_indices_arr, bg_data, z_orth, sp_mat=bg_sp
                )
                q_mat, _ = np.linalg.qr(y_mat, mode="reduced")

            b_t = _csr_rmatmul(
                len(bg_indices),
                v_size,
                bg_indptr,
                bg_indices_arr,
                bg_data,
                q_mat,
                sp_mat=bg_sp,
                t_csr=bg_t_csr,
            )
            del bg_t_csr
            # b_t = X_bg.T @ Q has shape (v_size, q_cols); B = Q.T @ X_bg = b_t.T
            _, _, vt = np.linalg.svd(b_t.T, full_matrices=False)
            vt_k = vt[:k].copy()

        # Canonicalize singular vector signs so coordinates are invariant across OS/BLAS builds
        max_abs_cols = np.argmax(np.abs(vt_k), axis=1)
        signs = np.where(vt_k[np.arange(vt_k.shape[0]), max_abs_cols] >= 0.0, 1.0, -1.0)
        vt_k *= signs[:, None]
        self.components = vt_k.T  # shape (v_size, k)
        self.n_components = self.components.shape[1]

        raw_proj = _csr_matmul(
            self.num_docs, v_size, all_indptr, all_idx, all_val, self.components, sp_mat=all_sp
        )
        self.embeddings = self._l2_normalize_rows(raw_proj)

    @staticmethod
    def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return np.where(norms > 1e-12, mat / np.where(norms > 1e-12, norms, 1.0), 0.0)

    def _build_csr_arrays(
        self, feat_counters: Sequence[Counter[str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_map = self.vocab
        idf_arr = self.idf
        indptr = np.zeros(len(feat_counters) + 1, dtype=np.int32)
        indices_list: list[int] = []
        data_list: list[float] = []
        cursor = 0
        for row_idx, fc in enumerate(feat_counters):
            row_cols: list[int] = []
            row_vals: list[float] = []
            sq_norm = 0.0
            for term, count in fc.items():
                col = v_map.get(term)
                if col is None:
                    continue
                val = (1.0 + math.log(float(count))) * float(idf_arr[col])
                row_cols.append(col)
                row_vals.append(val)
                sq_norm += val * val
            if row_cols and sq_norm > 1e-15:
                order = np.argsort(row_cols)
                inv_norm = 1.0 / math.sqrt(sq_norm)
                for pos in order:
                    indices_list.append(row_cols[int(pos)])
                    data_list.append(row_vals[int(pos)] * inv_norm)
                cursor += len(row_cols)
            indptr[row_idx + 1] = cursor
        return (
            indptr,
            np.asarray(indices_list, dtype=np.int32),
            np.asarray(data_list, dtype=np.float64),
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Project arbitrary texts into the L2-normalized dense Truncated-SVD subspace."""
        if not texts:
            return np.zeros((0, self.n_components), dtype=np.float64)
        if not self.vocab:
            return np.zeros((len(texts), self.n_components), dtype=np.float64)
        if len(texts) == 1:
            fc = _extract_svd_features(texts[0])
            if not fc:
                return np.zeros((1, self.n_components), dtype=np.float64)
            v_map = self.vocab
            idf_arr = self.idf
            cols: list[int] = []
            vals: list[float] = []
            sq_norm = 0.0
            for term, count in fc.items():
                col = v_map.get(term)
                if col is not None:
                    w = (1.0 + math.log(float(count))) * float(idf_arr[col])
                    cols.append(col)
                    vals.append(w)
                    sq_norm += w * w
            if not cols or sq_norm <= 1e-15:
                return np.zeros((1, self.n_components), dtype=np.float64)
            unit_vals = np.asarray(vals, dtype=np.float64) / math.sqrt(sq_norm)
            proj_1 = (unit_vals @ self.components[np.asarray(cols, dtype=np.int32)])[None, :]
            return self._l2_normalize_rows(proj_1)

        feats = [_extract_svd_features(t) for t in texts]
        indptr, idx, val = self._build_csr_arrays(feats)
        v_size = len(self.vocab)
        sp_mat = (
            _sp.csr_matrix((val, idx, indptr), shape=(len(texts), v_size))
            if (_sp is not None and len(texts) >= 32)
            else None
        )
        proj = _csr_matmul(len(texts), v_size, indptr, idx, val, self.components, sp_mat=sp_mat)
        return self._l2_normalize_rows(proj)

    def cosine_scores(
        self,
        query: str,
        cand_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute dense LSA cosine similarities between `query` and indexed items."""
        if self.num_docs == 0:
            return np.zeros(0, dtype=np.float64)
        q_vec = self.embed([query])[0]
        if float(np.linalg.norm(q_vec)) <= 1e-12:
            n_out = self.num_docs if cand_ids is None else len(cand_ids)
            return np.zeros(n_out, dtype=np.float64)
        target = (
            self.embeddings
            if cand_ids is None
            else self.embeddings[np.asarray(cand_ids, dtype=np.int32)]
        )
        if target.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return np.clip(target @ q_vec, -1.0, 1.0)


@functools.lru_cache(maxsize=65536)
def _extract_entity_num_neg_tuple(text: str) -> tuple[tuple[str, float], ...]:
    """Cached biomedical entity/acronym, numeric literal, allele, and negation extractor."""
    weights: dict[str, float] = {}
    if not text:
        return ()
    for m in _ENTITY_ACRONYM_RE.findall(text):
        weights[f"ent:{m.lower()}"] = 2.5
    for num in _NUMERIC_TOKEN_RE.findall(normalize_biomedical_text(text)):
        weights[f"num:{num}"] = 2.2
    for tok in tokenize(text):
        if tok.endswith(("+", "-", "+/-")):
            weights[f"all:{tok}"] = 2.5
        elif tok in _NEGATION_TOKENS:
            weights[f"neg:{tok}"] = 0.8
    return tuple(weights.items())


def _extract_entity_num_neg_features(text: str) -> dict[str, float]:
    """Extract biomedical entity anchors, numeric literals, alleles, and negation cues."""
    return dict(_extract_entity_num_neg_tuple(text))


class EntityNumericNegationIndex:
    """Inverted overlap scorer for biomedical symbols/entities, numbers, and negation/direction cues."""

    def __init__(
        self,
        texts: Sequence[str],
        *,
        idf_mask: Sequence[bool] | None = None,
    ) -> None:
        self.num_docs = len(texts)
        doc_feats = [_extract_entity_num_neg_features(t) for t in texts]
        df: Counter[str] = Counter()
        if idf_mask is not None and any(idf_mask):
            n_bg = 0
            for fc, keep in zip(doc_feats, idf_mask):
                if keep:
                    n_bg += 1
                    for k in fc.keys():
                        df[k] += 1
            n_idf = float(max(1, n_bg))
        else:
            for fc in doc_feats:
                for k in fc.keys():
                    df[k] += 1
            n_idf = float(max(1, self.num_docs))

        self.idf: dict[str, float] = {
            k: math.log(1.0 + (n_idf - d + 0.5) / (d + 0.5)) for k, d in df.items()
        }
        self.default_idf = math.log(1.0 + (n_idf + 0.5) / 0.5)
        raw_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_idx, fc in enumerate(doc_feats):
            for k, w in fc.items():
                raw_postings[k].append((doc_idx, w))

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            k: (
                np.asarray([i for i, _ in pairs], dtype=np.int32),
                np.asarray([w for _, w in pairs], dtype=np.float64),
            )
            for k, pairs in raw_postings.items()
        }

    def overlap_scores(
        self,
        query: str,
        cand_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute entity/numeric/negation overlap scores in `[0, 1]`."""
        n_target = self.num_docs if cand_ids is None else len(cand_ids)
        if n_target == 0:
            return np.zeros(0, dtype=np.float64)
        q_feats = _extract_entity_num_neg_features(query)
        if not q_feats:
            return np.zeros(n_target, dtype=np.float64)

        budget = sum(
            w * self.idf.get(k, self.default_idf) for k, w in q_feats.items()
        )
        if budget <= 0.0:
            return np.zeros(n_target, dtype=np.float64)

        if cand_ids is None:
            acc = np.zeros(self.num_docs, dtype=np.float64)
            for k, qw in q_feats.items():
                post = self.postings.get(k)
                if post is None:
                    continue
                d_idx, _ = post
                acc[d_idx] += qw * self.idf.get(k, self.default_idf)
            return np.clip(acc / budget, 0.0, 1.0)

        idx_arr = np.asarray(cand_ids, dtype=np.int32)
        acc = np.zeros(idx_arr.size, dtype=np.float64)
        for k, qw in q_feats.items():
            post = self.postings.get(k)
            if post is None:
                continue
            d_idx, _ = post
            pos = np.searchsorted(d_idx, idx_arr)
            valid = (pos < d_idx.size) & (d_idx[np.minimum(pos, d_idx.size - 1)] == idx_arr)
            if np.any(valid):
                acc[valid] += qw * self.idf.get(k, self.default_idf)
        return np.clip(acc / budget, 0.0, 1.0)


def _zscore_pool(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - mean) / std, -4.0, 6.0)


_UNREFERENCED_CONTRAST_NOT_RE = re.compile(
    r"(?:\(\s*not\s+([^)]+)\)|,\s*not\s+([^,.;:()]+))",
    re.IGNORECASE,
)


def _strip_unreferenced_contrastive_not(clause: str, ref_tokens: set[str]) -> str:
    """Strip parenthetical `(not X)` or `, not X` contrasts when `X` has zero overlap with `ref_tokens`."""
    ref_stems = {_normalize_orthography_and_stem(t) for t in ref_tokens if t not in _NEGATION_TOKENS}

    def _sub_fn(m: re.Match[str]) -> str:
        span_txt = m.group(1) if m.group(1) is not None else m.group(2)
        obj_toks = [t for t in tokenize(span_txt or "") if t not in _NEGATION_TOKENS]
        if not obj_toks:
            return m.group(0)
        obj_stems = {_normalize_orthography_and_stem(t) for t in obj_toks}
        if (set(obj_toks) & ref_tokens) or (obj_stems & ref_stems):
            return m.group(0)
        return " "

    return _UNREFERENCED_CONTRAST_NOT_RE.sub(_sub_fn, clause)


def _directional_interaction(claim: str, sentence: str) -> tuple[float, float, float, float]:
    """Compute `(scoped_neg_xor, scoped_neg_both, dir_same, dir_opposite)` for `(claim, sentence)`."""
    c_all = set(tokenize(claim))
    s_all = set(tokenize(sentence))
    if not c_all or not s_all:
        return 0.0, 0.0, 0.0, 0.0

    c_main = _strip_unreferenced_contrastive_not(
        _strip_unreferenced_without_adjuncts(_extract_main_assertion_clause(claim), s_all),
        s_all,
    )
    s_main = _strip_unreferenced_contrastive_not(
        _strip_unreferenced_without_adjuncts(_extract_main_assertion_clause(sentence), c_all),
        c_all,
    )

    c_toks = set(tokenize(c_main)) or c_all
    s_toks = set(tokenize(s_main)) or s_all

    c_neg = bool(c_toks & _NEGATION_TOKENS)
    s_neg = bool(s_toks & _NEGATION_TOKENS)
    neg_xor = 1.0 if (c_neg != s_neg) else 0.0
    neg_both = 1.0 if (c_neg and s_neg) else 0.0

    same_hits = 0
    opp_hits = 0
    for pos_g, neg_g in _DIRECTIONAL_OPPOSITES:
        c_pos = bool(c_toks & pos_g)
        c_neg_g = bool(c_toks & neg_g)
        s_pos = bool(s_toks & pos_g)
        s_neg_g = bool(s_toks & neg_g)
        if (c_pos and s_neg_g and not s_pos) or (c_neg_g and s_pos and not s_neg_g):
            opp_hits += 1
        elif (c_pos and s_pos and not s_neg_g) or (c_neg_g and s_neg_g and not s_pos):
            same_hits += 1

    # Flip directional agreement when exactly one side carries a scoped negation!
    if neg_xor > 0.5 and (same_hits > 0 or opp_hits > 0):
        same_hits, opp_hits = opp_hits, same_hits

    return (
        neg_xor,
        neg_both,
        min(2.0, float(same_hits)),
        min(2.0, float(opp_hits)),
    )


@functools.lru_cache(maxsize=65536)
def _sentence_profile(
    sent: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    frozenset[str],
    frozenset[tuple[str, str]],
    frozenset[str],
    frozenset[str],
    float,
    bool,
    bool,
]:
    """Cached per-sentence lexical, stem-bigram, numeric, entity, and discourse profile."""
    s_all_toks = tuple(tokenize(sent))
    s_content = tuple(t for t in s_all_toks if t not in _NEGATION_TOKENS)
    s_set = frozenset(s_content)
    s_stems = [_normalize_orthography_and_stem(t) for t in s_content]
    s_bg = frozenset((s_stems[j], s_stems[j + 1]) for j in range(len(s_stems) - 1))
    s_nums = frozenset(_NUMERIC_TOKEN_RE.findall(normalize_biomedical_text(sent)))
    s_ents = frozenset(m.lower() for m in _ENTITY_ACRONYM_RE.findall(sent)) | frozenset(s_all_toks)
    s_tok_set = frozenset(s_all_toks)
    has_emp_pred = 1.0 if (s_tok_set & _ALL_DIRECTIONAL_TOKENS) else 0.0
    has_conc = bool(s_tok_set & _CONCLUSION_CUES)
    has_intro = bool(s_tok_set & _INTRO_CUES)
    return (
        s_all_toks,
        s_content,
        s_set,
        s_bg,
        s_nums,
        s_ents,
        has_emp_pred,
        has_conc,
        has_intro,
    )


def extract_candidate_rerank_matrix(
    claim: str,
    sentences: Sequence[str],
    *,
    bm25: BM25,
    svd_index: TruncatedSVDSemanticIndex | None = None,
    subword_index: _SubwordConceptCosineIndex | None = None,
    entity_index: EntityNumericNegationIndex | None = None,
    sent_global_ids: Sequence[int] | None = None,
    local_positions: Sequence[tuple[int, int]] | None = None,
    titles: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract `(N_cand, 26)` pairwise claim-sentence interaction matrix and channel cache."""
    n_cand = len(sentences)
    if n_cand == 0:
        return np.zeros((0, 26), dtype=np.float64), {}

    if sent_global_ids is not None:
        bm25_raw = bm25.score_indexed(claim, sent_global_ids)
        bm25_norm = bm25.score_indexed_normalized(claim, sent_global_ids)
    else:
        local_bm25 = BM25(sentences)
        bm25_raw = np.maximum(bm25.scores(claim, sentences), local_bm25.score_indexed(claim))
        bm25_norm = np.maximum(
            bm25.normalized_scores(claim, sentences),
            local_bm25.score_indexed_normalized(claim),
        )

    if svd_index is not None:
        if sent_global_ids is not None:
            lsa_cos = svd_index.cosine_scores(claim, sent_global_ids)
        else:
            q_emb = svd_index.embed([claim])[0]
            s_emb = svd_index.embed(sentences)
            lsa_cos = np.clip(s_emb @ q_emb, -1.0, 1.0) if float(np.linalg.norm(q_emb)) > 1e-12 else np.zeros(n_cand)
    else:
        lsa_cos = np.zeros(n_cand, dtype=np.float64)

    if subword_index is not None and sent_global_ids is not None:
        sub_cos = subword_index.cosine_scores(claim, sent_global_ids)
    else:
        tmp_sub = _SubwordConceptCosineIndex(sentences)
        sub_cos = tmp_sub.cosine_scores(claim)

    if entity_index is not None and sent_global_ids is not None:
        ent_ov = entity_index.overlap_scores(claim, sent_global_ids)
    else:
        tmp_ent = EntityNumericNegationIndex(sentences)
        ent_ov = tmp_ent.overlap_scores(claim)

    lsa_pos = np.maximum(0.0, lsa_cos)
    stage1_rrf = _reciprocal_rank_fusion(
        [bm25_raw, lsa_pos, ent_ov],
        rrf_k=60,
        weights=[1.25, 0.95, 0.85],
    )
    max_rrf = (1.25 + 0.95 + 0.85) / 61.0
    rrf_norm = np.clip(stage1_rrf / max_rrf, 0.0, 1.0)

    bm25_z = _zscore_pool(bm25_raw)
    lsa_z = _zscore_pool(lsa_cos)
    bm25_max = float(np.max(bm25_raw)) if bm25_raw.size > 0 else 0.0
    bm25_max_ratio = bm25_raw / (bm25_max + 1e-6) if bm25_max > 0.0 else np.zeros(n_cand)
    lsa_max = float(np.max(lsa_pos)) if lsa_pos.size > 0 else 0.0
    lsa_max_ratio = lsa_pos / (lsa_max + 1e-6) if lsa_max > 0.0 else np.zeros(n_cand)

    bm25_order = np.argsort(-bm25_raw, kind="stable")
    bm25_ranks = np.empty(n_cand, dtype=np.float64)
    bm25_ranks[bm25_order] = np.arange(n_cand, dtype=np.float64)
    bm25_rank_inv = 1.0 / (1.0 + bm25_ranks)

    q_content = list(dict.fromkeys(t for t in tokenize(claim) if t not in _NEGATION_TOKENS))
    q_idfs = np.array(
        [bm25.idf.get(t, bm25.default_idf) for t in q_content],
        dtype=np.float64,
    )
    q_total_idf = float(np.sum(q_idfs)) if q_idfs.size > 0 else 1.0
    top_rare_indices = (
        np.argsort(-q_idfs)[: min(2, len(q_content))]
        if q_content
        else np.zeros(0, dtype=np.int32)
    )
    q_stems = [_normalize_orthography_and_stem(t) for t in q_content]
    q_bigrams = [(q_stems[i], q_stems[i + 1]) for i in range(len(q_stems) - 1)]
    q_nums = set(_NUMERIC_TOKEN_RE.findall(normalize_biomedical_text(claim)))
    q_ents = {m.lower() for m in _ENTITY_ACRONYM_RE.findall(claim)}

    title_tok_cache: dict[str, list[str]] = {}
    feats = np.zeros((n_cand, 26), dtype=np.float64)

    neg_xor_arr = np.zeros(n_cand, dtype=np.float64)
    dir_same_arr = np.zeros(n_cand, dtype=np.float64)
    dir_opp_arr = np.zeros(n_cand, dtype=np.float64)
    num_mismatch_arr = np.zeros(n_cand, dtype=np.float64)
    idf_cov_arr = np.zeros(n_cand, dtype=np.float64)
    rare_min_arr = np.zeros(n_cand, dtype=np.float64)

    q_set = frozenset(q_content)
    for i, sent in enumerate(sentences):
        (
            s_all_toks,
            s_content,
            s_set,
            s_bg,
            s_nums,
            s_ents,
            has_emp_pred,
            has_conc,
            has_intro,
        ) = _sentence_profile(sent)

        if q_content and s_content:
            sims = np.array(
                [max((_token_similarity(qt, st) for st in s_content), default=0.0) for qt in q_content],
                dtype=np.float64,
            )
            idf_cov = float(np.sum(q_idfs * np.where(sims >= 0.55, sims, 0.0))) / q_total_idf
            rare_min = float(np.min(sims[top_rare_indices])) if top_rare_indices.size > 0 else 0.0
            jaccard = len(q_set & s_set) / float(len(q_set | s_set))
        else:
            sims = np.zeros(len(q_content), dtype=np.float64)
            idf_cov = 0.0
            rare_min = 0.0
            jaccard = 0.0

        idf_cov_arr[i] = idf_cov
        rare_min_arr[i] = rare_min

        if q_bigrams and s_bg:
            bg_ov = sum(1 for bg in q_bigrams if bg in s_bg) / float(len(q_bigrams))
        else:
            bg_ov = 0.0

        num_match = 1.0 if (q_nums and (q_nums & s_nums)) else 0.0
        num_mismatch = 1.0 if (q_nums and s_nums and not (q_nums & s_nums)) else 0.0
        num_mismatch_arr[i] = num_mismatch

        if q_ents:
            ent_sym_ov = len(q_ents & s_ents) / float(len(q_ents))
        else:
            ent_sym_ov = 0.0

        neg_xor, neg_both, dir_same, dir_opp = _directional_interaction(claim, sent)
        neg_xor_arr[i] = neg_xor
        dir_same_arr[i] = dir_same
        dir_opp_arr[i] = dir_opp
        abs_dir_pair = 1.0 if (dir_same > 0.0 or dir_opp > 0.0) else 0.0

        title_bridge = 0.0
        if titles is not None and i < len(titles) and titles[i] and idf_cov > 0.08 and q_content:
            t_str = titles[i]
            if t_str not in title_tok_cache:
                title_tok_cache[t_str] = [
                    t for t in tokenize(t_str) if t not in _NEGATION_TOKENS
                ]
            t_toks = title_tok_cache[t_str]
            if t_toks:
                unmatched_mask = sims < 0.55
                if np.any(unmatched_mask):
                    bridged = 0.0
                    for qi in np.flatnonzero(unmatched_mask):
                        qt = q_content[int(qi)]
                        best_t = max((_token_similarity(qt, tt) for tt in t_toks), default=0.0)
                        if best_t >= 0.60:
                            bridged += float(q_idfs[int(qi)]) * best_t
                    title_bridge = bridged / q_total_idf

        if local_positions is not None and i < len(local_positions):
            loc_idx, doc_len = local_positions[i]
        else:
            loc_idx, doc_len = i, n_cand

        rel_pos = float(loc_idx) / float(max(1, doc_len - 1)) if doc_len > 1 else 0.5
        is_lead = 1.0 if (loc_idx == 0 and doc_len >= 4) else 0.0
        is_tail = 1.0 if (loc_idx >= doc_len - 2 and doc_len >= 3) else 0.0

        disc_cue = 1.0 if has_conc else (-1.0 if (has_intro and loc_idx <= 1) else 0.0)
        len_ratio = min(2.0, float(len(s_all_toks)) / 20.0)

        feats[i, 0] = float(bm25_norm[i])
        feats[i, 1] = float(bm25_z[i])
        feats[i, 2] = float(bm25_max_ratio[i])
        feats[i, 3] = float(bm25_rank_inv[i])
        feats[i, 4] = float(lsa_cos[i])
        feats[i, 5] = float(lsa_z[i])
        feats[i, 6] = float(lsa_max_ratio[i])
        feats[i, 7] = float(sub_cos[i])
        feats[i, 8] = float(bm25_norm[i] * lsa_pos[i])
        feats[i, 9] = float(rrf_norm[i])
        feats[i, 10] = float(idf_cov)
        feats[i, 11] = float(jaccard)
        feats[i, 12] = float(rare_min)
        feats[i, 13] = float(bg_ov)
        feats[i, 14] = float(num_match)
        feats[i, 15] = float(num_mismatch)
        feats[i, 16] = float(ent_sym_ov)
        feats[i, 17] = float(has_emp_pred)
        feats[i, 18] = float(abs_dir_pair)
        feats[i, 19] = float(title_bridge)
        feats[i, 20] = float(rel_pos)
        feats[i, 21] = float(is_lead)
        feats[i, 22] = float(is_tail)
        feats[i, 23] = float(disc_cue)
        feats[i, 24] = float(len_ratio)
        # Key nonlinear cross-interactions for rationale vs intro separation
        feats[i, 25] = float((idf_cov + 0.5 * title_bridge) * (0.4 + 0.6 * rel_pos + 0.3 * has_emp_pred))

        _ = neg_both

    channels = {
        "bm25_raw": bm25_raw,
        "bm25_norm": bm25_norm,
        "lsa_cosine": lsa_cos,
        "subword_cosine": sub_cos,
        "entity_overlap": ent_ov,
        "stage1_rrf": stage1_rrf,
        "neg_xor": neg_xor_arr,
        "dir_same": dir_same_arr,
        "dir_opp": dir_opp_arr,
        "num_mismatch": num_mismatch_arr,
        "idf_cov": idf_cov_arr,
        "rare_min": rare_min_arr,
    }
    return feats, channels


def _fit_binary_logistic_newton(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    *,
    l2_reg: float = 2.0,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 35,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Deterministic standardized damped Newton-IRLS binary logistic regression (pure NumPy)."""
    n_samples, n_feats = x_mat.shape
    mean = np.mean(x_mat, axis=0)
    std = np.std(x_mat, axis=0)
    std_safe = np.where(std > 1e-8, std, 1.0)
    z_mat = (x_mat - mean) / std_safe

    # Augment bias column at index n_feats
    z_aug = np.hstack([z_mat, np.ones((n_samples, 1), dtype=np.float64)])
    sw = (
        np.ones(n_samples, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    sw = sw / max(float(np.mean(sw)), 1e-12)

    theta = np.zeros(n_feats + 1, dtype=np.float64)
    pos_rate = float(np.clip(np.average(y_vec, weights=sw), 1e-4, 1.0 - 1e-4))
    theta[-1] = math.log(pos_rate / (1.0 - pos_rate))

    reg_diag = np.full(n_feats + 1, float(l2_reg), dtype=np.float64)
    reg_diag[-1] = 1e-6  # Do not shrink intercept

    for _ in range(max_iter):
        logits = np.clip(z_aug @ theta, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = z_aug.T @ (sw * (probs - y_vec)) + reg_diag * theta
        w_diag = sw * np.maximum(probs * (1.0 - probs), 1e-6)
        hess = (z_aug.T * w_diag) @ z_aug + np.diag(reg_diag)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=1e-8)[0]
        theta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    return theta[:-1].copy(), float(theta[-1]), mean, std_safe


def _fit_multinomial_logistic(
    x_mat: np.ndarray,
    y_idx: np.ndarray,
    *,
    n_classes: int = 3,
    l2_reg: float = 2.5,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 450,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Nesterov-accelerated L2 multinomial logistic regression (pure NumPy)."""
    n_samples, n_feats = x_mat.shape
    mean = np.mean(x_mat, axis=0)
    std = np.std(x_mat, axis=0)
    std_safe = np.where(std > 1e-8, std, 1.0)
    z_mat = (x_mat - mean) / std_safe

    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_onehot[np.arange(n_samples), y_idx] = 1.0

    sw = (
        np.ones(n_samples, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    sw_norm = sw / max(float(np.sum(sw)), 1e-12)

    w_mat = np.zeros((n_feats, n_classes), dtype=np.float64)
    class_priors = np.clip(y_onehot.T @ sw_norm, 1e-4, 1.0)
    b_vec = np.log(class_priors)
    b_vec -= float(np.mean(b_vec))

    # Exact Lipschitz upper bound of smooth objective for guaranteed Nesterov convergence
    spectral_norm_sq = float(np.linalg.norm(z_mat * np.sqrt(sw_norm[:, None]), ord=2) ** 2)
    reg_scaled = float(l2_reg) / float(max(1, n_samples))
    lip = max(1e-3, 0.5 * spectral_norm_sq + reg_scaled)
    lr = 1.0 / lip

    w_prev = w_mat.copy()
    b_prev = b_vec.copy()
    t_prev = 1.0

    for _ in range(max_iter):
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev))
        beta = (t_prev - 1.0) / t_next
        w_y = w_mat + beta * (w_mat - w_prev)
        b_y = b_vec + beta * (b_vec - b_prev)

        logits = z_mat @ w_y + b_y[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        diff = (probs - y_onehot) * sw_norm[:, None]
        grad_w = z_mat.T @ diff + reg_scaled * w_y
        grad_b = np.sum(diff, axis=0)

        w_prev, b_prev = w_mat, b_vec
        w_mat = w_y - lr * grad_w
        b_vec = b_y - lr * grad_b
        t_prev = t_next

        if float(np.max(np.abs(grad_w))) < 1e-6 and float(np.max(np.abs(grad_b))) < 1e-6:
            break

    return w_mat, b_vec, mean, std_safe


@dataclass(frozen=True)
class ConformalStancePrediction:
    """Calibrated 3-way stance prediction with Split Conformal epistemic set."""

    predicted_label: str
    probabilities: dict[str, float]
    conformal_set: list[str]
    alpha: float
    is_contested_conflict: bool
    top_rationale_indices: list[int]
    top_rationale_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "probabilities": {k: round(float(v), 6) for k, v in self.probabilities.items()},
            "conformal_set": list(self.conformal_set),
            "alpha": float(self.alpha),
            "is_contested_conflict": bool(self.is_contested_conflict),
            "top_rationale_indices": list(self.top_rationale_indices),
            "top_rationale_scores": [round(float(s), 6) for s in self.top_rationale_scores],
        }


class CrossFeatureReranker:
    """Stage 2 pairwise claim-sentence interaction reranker."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.feat_mean = np.asarray(feat_mean, dtype=np.float64)
        self.feat_std = np.asarray(feat_std, dtype=np.float64)

    @classmethod
    def default_prior(cls) -> "CrossFeatureReranker":
        """Analytic default weights used only before supervised fitting on `SciFact train`."""
        w = np.zeros(26, dtype=np.float64)
        w[0] = 2.4   # bm25_norm
        w[1] = 0.85  # bm25_zscore
        w[2] = 0.70  # bm25_max_ratio
        w[4] = 1.20  # lsa_cosine
        w[5] = 0.45  # lsa_zscore
        w[7] = 0.80  # subword_cosine
        w[8] = 0.90  # lexical_dense_interaction
        w[9] = 1.10  # stage1_rrf_norm
        w[10] = 1.30 # idf_content_coverage
        w[12] = 0.75 # rare_entity_min_sim
        w[17] = 0.35 # has_empirical_predicate
        w[19] = 0.55 # title_context_bridge
        w[20] = 0.30 # rel_sent_pos
        w[21] = -0.35 # is_lead_intro
        w[25] = 0.80 # nonlinear discourse*coverage interaction
        return cls(
            weights=w,
            bias=-2.2,
            feat_mean=np.zeros(26, dtype=np.float64),
            feat_std=np.ones(26, dtype=np.float64),
        )

    def score_features(self, feat_matrix: np.ndarray) -> np.ndarray:
        """Return calibrated relevance probabilities in `[0, 1]` for each candidate row."""
        if feat_matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        z = (feat_matrix - self.feat_mean) / self.feat_std
        logits = np.clip(z @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [round(float(x), 8) for x in self.weights.tolist()],
            "bias": round(float(self.bias), 8),
            "feat_mean": [round(float(x), 8) for x in self.feat_mean.tolist()],
            "feat_std": [round(float(x), 8) for x in self.feat_std.tolist()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CrossFeatureReranker":
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=float(data["bias"]),
            feat_mean=np.asarray(data["feat_mean"], dtype=np.float64),
            feat_std=np.asarray(data["feat_std"], dtype=np.float64),
        )


def _extract_stance_feature_vector(
    claim: str,
    sentences: Sequence[str],
    feat_matrix: np.ndarray,
    channels: Mapping[str, np.ndarray],
    rerank_probs: np.ndarray,
    *,
    top_k: int = 3,
) -> tuple[np.ndarray, float, float]:
    """Build 28-dim claim-level stance feature vector from top-k reranked sentences.

    Includes multiplicative relevance x polarity/directionality interactions so
    contradiction cues only fire on genuinely relevant sentences, and returns
    `(support_evidence_mass, contradict_evidence_mass)` for multi-evidence conflict sets.
    """
    out = np.zeros(28, dtype=np.float64)
    if len(sentences) == 0 or feat_matrix.shape[0] == 0:
        out[23] = 1.0  # explicit empty-evidence NEI indicator
        return out, 0.0, 0.0

    order = np.argsort(-rerank_probs, kind="stable")
    k = max(1, min(int(top_k), len(sentences)))
    top_idx = order[:k]
    top_probs = rerank_probs[top_idx]
    w_sum = float(np.sum(top_probs))
    weights = top_probs / w_sum if w_sum > 1e-9 else np.full(k, 1.0 / float(k))

    i0 = int(top_idx[0])
    i1 = int(top_idx[1]) if k >= 2 else i0

    bm25_norm = channels["bm25_norm"]
    bm25_raw = channels["bm25_raw"]
    lsa_cos = channels["lsa_cosine"]
    sub_cos = channels["subword_cosine"]
    neg_xor = channels["neg_xor"]
    dir_same = channels["dir_same"]
    dir_opp = channels["dir_opp"]
    num_mis = channels["num_mismatch"]
    idf_cov = channels["idf_cov"]
    rare_min = channels["rare_min"]

    pol_sup = np.zeros(k, dtype=np.float64)
    pol_con = np.zeros(k, dtype=np.float64)
    contrast_cue = np.zeros(k, dtype=np.float64)
    for pos, s_idx in enumerate(top_idx):
        s_txt = sentences[int(s_idx)]
        pol = classify_rationale_polarity(claim, s_txt)
        if pol == "SUPPORT":
            pol_sup[pos] = 1.0
        elif pol == "CONTRADICT":
            pol_con[pos] = 1.0
        if _CONTRASTIVE_PHRASE_RE.search(s_txt):
            contrast_cue[pos] = 1.0

    # Per-sentence relevance-gated directional evidence mass
    rel_gate = np.maximum(
        top_probs, 0.5 * bm25_norm[top_idx] + 0.5 * np.maximum(0.0, lsa_cos[top_idx])
    )
    con_signal = np.clip(
        0.55 * pol_con
        + 0.45 * neg_xor[top_idx]
        + 0.50 * np.minimum(1.0, dir_opp[top_idx])
        + 0.30 * num_mis[top_idx]
        + 0.25 * contrast_cue,
        0.0,
        1.5,
    )
    sup_signal = np.where(
        con_signal >= 0.55,
        0.0,
        np.clip(0.60 * pol_sup + 0.40 * np.minimum(1.0, dir_same[top_idx]), 0.0, 1.2),
    )

    sup_mass = float(np.max(rel_gate * sup_signal)) if k > 0 else 0.0
    con_mass = float(np.max(rel_gate * np.minimum(1.0, con_signal))) if k > 0 else 0.0

    q_toks = [t for t in tokenize(claim) if t not in _NEGATION_TOKENS]
    pooled_top_toks = {
        t
        for s_idx in top_idx
        for t in _sentence_profile(sentences[int(s_idx)])[1]
    }
    if q_toks and pooled_top_toks:
        matched_q = sum(
            1
            for qt in q_toks
            if any(_token_similarity(qt, st) >= 0.55 for st in pooled_top_toks)
        )
        unmatched_q_frac = 1.0 - (float(matched_q) / float(len(q_toks)))
    elif q_toks:
        unmatched_q_frac = 1.0
    else:
        unmatched_q_frac = 0.0

    p0 = float(top_probs[0])
    cov0 = float(idf_cov[i0])
    out[0] = p0
    out[1] = float(rerank_probs[i1]) if k >= 2 else 0.0
    out[2] = float(np.mean(top_probs))
    out[3] = float(np.max(bm25_norm[top_idx]))
    out[4] = math.log1p(max(0.0, float(np.max(bm25_raw[top_idx]))))
    out[5] = float(np.max(lsa_cos[top_idx]))
    out[6] = float(np.max(sub_cos[top_idx]))
    out[7] = float(np.max(idf_cov[top_idx]))
    out[8] = float(np.max(rare_min[top_idx]))
    out[9] = float(unmatched_q_frac)
    out[10] = float(neg_xor[i0])
    out[11] = float(np.sum(weights * neg_xor[top_idx]))
    out[12] = float(dir_opp[i0])
    out[13] = float(np.sum(weights * dir_opp[top_idx]))
    out[14] = float(dir_same[i0])
    out[15] = float(np.sum(weights * dir_same[top_idx]))
    out[16] = float(num_mis[i0])
    out[17] = float(pol_con[0])
    out[18] = float(np.sum(weights * pol_con))
    out[19] = float(pol_sup[0])
    out[20] = float(np.sum(weights * pol_sup))
    out[21] = float(contrast_cue[0])
    out[22] = float(sup_mass - con_mass)
    out[23] = 0.0
    # Multiplicative relevance-gated stance interaction features
    out[24] = float(sup_mass)
    out[25] = float(con_mass)
    out[26] = float((p0 * cov0) * (neg_xor[i0] + dir_opp[i0] + pol_con[0]))
    out[27] = float((1.0 - p0) * (0.5 + unmatched_q_frac))
    return out, sup_mass, con_mass


class SplitConformalStanceClassifier:
    """3-way stance classifier (`SUPPORT` / `CONTRADICT` / `NEI`) with Split Conformal sets.

    Calibrates finite-sample nonconformity quantiles `q_hat` strictly on a held-out
    calibration split (`SciFact train_calib`) at `1 - alpha` (default `0.90`).
    Supports both Mondrian (label-conditional, guaranteeing `>= 1 - alpha` coverage per
    class AND marginally) and standard marginal Split Conformal Prediction.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        *,
        alpha: float = 0.10,
        q_hat_marginal: float = 0.72,
        q_hat_per_class: Mapping[str, float] | None = None,
        conformal_mode: str = "mondrian",
        temperature: float = 1.0,
        is_calibrated: bool = False,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Conformal miscoverage alpha must lie in (0, 1), got {alpha!r}")
        if conformal_mode not in {"mondrian", "marginal"}:
            raise ValueError(f"Unsupported conformal_mode {conformal_mode!r}")
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.feat_mean = np.asarray(feat_mean, dtype=np.float64)
        self.feat_std = np.asarray(feat_std, dtype=np.float64)
        self.alpha = float(alpha)
        self.q_hat_marginal = float(q_hat_marginal)
        self.q_hat_per_class: dict[str, float] = (
            {lab: float(q_hat_per_class[lab]) for lab in STANCE_LABELS}
            if q_hat_per_class is not None
            else {lab: float(q_hat_marginal) for lab in STANCE_LABELS}
        )
        self.conformal_mode = conformal_mode
        self.temperature = max(0.1, float(temperature))
        self.is_calibrated = bool(is_calibrated)

    @classmethod
    def default_prior(cls, *, alpha: float = 0.10) -> "SplitConformalStanceClassifier":
        """Analytic default 3-way weights used only when no training split is provided."""
        w = np.zeros((28, 3), dtype=np.float64)
        # SUPPORT column (0)
        w[0, 0] = 1.3
        w[3, 0] = 1.1
        w[5, 0] = 0.9
        w[7, 0] = 1.2
        w[14, 0] = 0.8
        w[19, 0] = 1.1
        w[20, 0] = 0.9
        w[22, 0] = 1.1
        w[24, 0] = 1.4
        # CONTRADICT column (1)
        w[0, 1] = 1.1
        w[3, 1] = 0.9
        w[5, 1] = 0.8
        w[7, 1] = 1.0
        w[10, 1] = 1.3
        w[11, 1] = 1.0
        w[12, 1] = 1.3
        w[16, 1] = 0.9
        w[17, 1] = 1.3
        w[18, 1] = 1.0
        w[21, 1] = 0.6
        w[22, 1] = -1.1
        w[25, 1] = 1.5
        w[26, 1] = 1.2
        # NEI column (2)
        w[0, 2] = -1.6
        w[3, 2] = -1.5
        w[5, 2] = -1.1
        w[7, 2] = -1.4
        w[9, 2] = 1.8
        w[23, 2] = 4.0
        w[27, 2] = 1.6
        return cls(
            weights=w,
            bias=np.array([0.1, -0.1, 0.0], dtype=np.float64),
            feat_mean=np.zeros(28, dtype=np.float64),
            feat_std=np.ones(28, dtype=np.float64),
            alpha=alpha,
            q_hat_marginal=0.70,
            q_hat_per_class={"SUPPORT": 0.68, "CONTRADICT": 0.74, "NEI": 0.68},
        )

    def _raw_softmax(self, feat_matrix: np.ndarray) -> np.ndarray:
        z = (feat_matrix - self.feat_mean) / self.feat_std
        logits = (z @ self.weights + self.bias[None, :]) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict_proba_from_features(
        self,
        feat_vec: np.ndarray,
        *,
        sup_mass: float = 0.0,
        con_mass: float = 0.0,
    ) -> tuple[np.ndarray, bool]:
        """Compute calibrated 3-way `(SUPPORT, CONTRADICT, NEI)` probabilities without heuristic blending."""
        _ = (sup_mass, con_mass)
        probs = self._raw_softmax(feat_vec[None, :])[0].copy()
        return probs, False

    @staticmethod
    def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
        """Exact finite-sample Split Conformal quantile `ceil((n + 1)*(1 - alpha)) / n`."""
        arr = np.sort(np.asarray(scores, dtype=np.float64))
        n = arr.size
        if n == 0:
            return 1.0 - float(alpha)
        rank_k = math.ceil((float(n) + 1.0) * (1.0 - float(alpha)))
        if rank_k > n:
            return 1.0
        idx = max(0, min(n - 1, rank_k - 1))
        return float(arr[idx])

    def calibrate_split_conformal(
        self,
        calib_probs: np.ndarray,
        calib_labels: Sequence[str],
        *,
        alpha: float | None = None,
    ) -> dict[str, float]:
        """Calibrate `q_hat_marginal` and Mondrian `q_hat_per_class` on held-out calibration probabilities."""
        if alpha is not None:
            if not (0.0 < alpha < 1.0):
                raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
            self.alpha = float(alpha)

        y_canon = [canonicalize_stance_label(y) for y in calib_labels]
        if len(y_canon) == 0:
            return dict(self.q_hat_per_class)

        nonconf_all: list[float] = []
        by_cls: dict[str, list[float]] = {lab: [] for lab in STANCE_LABELS}
        for i, lab in enumerate(y_canon):
            c_idx = _STANCE_TO_IDX[lab]
            s_i = 1.0 - float(calib_probs[i, c_idx])
            nonconf_all.append(s_i)
            by_cls[lab].append(s_i)

        self.q_hat_marginal = self.conformal_quantile(nonconf_all, self.alpha)
        for lab in STANCE_LABELS:
            if len(by_cls[lab]) >= 5:
                self.q_hat_per_class[lab] = self.conformal_quantile(by_cls[lab], self.alpha)
            else:
                self.q_hat_per_class[lab] = self.q_hat_marginal
        self.is_calibrated = True
        return {
            "alpha": self.alpha,
            "q_hat_marginal": self.q_hat_marginal,
            **{f"q_hat_{lab.lower()}": self.q_hat_per_class[lab] for lab in STANCE_LABELS},
        }

    def to_calibrator_dict(self) -> dict[str, Any]:
        """Serialize classifier weights, normalization stats, and conformal quantiles to a compact dict."""
        return {
            "weights": [[round(float(v), 8) for v in row] for row in self.weights.tolist()],
            "bias": [round(float(v), 8) for v in self.bias.tolist()],
            "feat_mean": [round(float(v), 8) for v in self.feat_mean.tolist()],
            "feat_std": [round(float(v), 8) for v in self.feat_std.tolist()],
            "alpha": float(self.alpha),
            "q_hat_marginal": round(float(self.q_hat_marginal), 8),
            "q_hat_per_class": {
                lab: round(float(self.q_hat_per_class[lab]), 8) for lab in STANCE_LABELS
            },
            "conformal_mode": self.conformal_mode,
            "temperature": round(float(self.temperature), 8),
            "is_calibrated": bool(self.is_calibrated),
        }

    @classmethod
    def from_calibrator_dict(cls, data: Mapping[str, Any]) -> "SplitConformalStanceClassifier":
        """Reconstruct a `SplitConformalStanceClassifier` from `to_calibrator_dict` output."""
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            feat_mean=np.asarray(data["feat_mean"], dtype=np.float64),
            feat_std=np.asarray(data["feat_std"], dtype=np.float64),
            alpha=float(data["alpha"]),
            q_hat_marginal=float(data["q_hat_marginal"]),
            q_hat_per_class=dict(data["q_hat_per_class"]),
            conformal_mode=str(data.get("conformal_mode", "mondrian")),
            temperature=float(data.get("temperature", 1.0)),
            is_calibrated=bool(data.get("is_calibrated", True)),
        )

    def save_calibrator(self, path: str | Path) -> Path:
        """Persist calibrated stance weights and Mondrian quantiles to a compact JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(self.to_calibrator_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return dest

    @classmethod
    def load_calibrator(cls, path: str | Path) -> "SplitConformalStanceClassifier":
        """Load calibrated stance weights and Mondrian quantiles from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "candidate_stance_classifier" in payload:
            return cls.from_calibrator_dict(payload["candidate_stance_classifier"])
        return cls.from_calibrator_dict(payload)

    def prediction_set_from_probs(
        self,
        probs: np.ndarray,
        *,
        alpha_override: float | None = None,
    ) -> list[str]:
        """Construct the `1 - alpha` Split Conformal prediction set over `STANCE_LABELS`."""
        if alpha_override is not None and abs(float(alpha_override) - self.alpha) > 1e-9:
            # Fallback monotonic scaling if caller queries a non-calibrated alpha at runtime
            scale = (1.0 - float(alpha_override)) / max(1e-6, 1.0 - self.alpha)
            thresholds = {
                lab: min(1.0, self.q_hat_per_class[lab] * scale) for lab in STANCE_LABELS
            }
        elif self.conformal_mode == "mondrian":
            thresholds = self.q_hat_per_class
        else:
            thresholds = {lab: self.q_hat_marginal for lab in STANCE_LABELS}

        pred_set: list[str] = []
        for idx, lab in enumerate(STANCE_LABELS):
            nonconf = 1.0 - float(probs[idx])
            if nonconf <= thresholds[lab] + 1e-12:
                pred_set.append(lab)

        if not pred_set:
            # Guarantee non-empty epistemic set on extreme boundary ties
            best_idx = int(np.argmax(probs))
            pred_set.append(STANCE_LABELS[best_idx])
        return pred_set


def split_scifact_train_fit_and_calib(
    train_claims: Sequence[Mapping[str, Any]],
    *,
    calib_fraction: float = 0.25,
    seed: int = 2026,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministically partition `SciFact train` (`809` claims: `607` fit + `202` cal at `0.25`).

    Stratifies across `SUPPORT`, `CONTRADICT`, and `NEI` so the held-out conformal
    calibration fold mirrors the empirical label distribution with zero overlap.
    """
    if not (0.05 <= calib_fraction <= 0.50):
        raise ValueError("calib_fraction must lie in [0.05, 0.50]")

    buckets: dict[str, list[Mapping[str, Any]]] = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    for c in train_claims:
        ev = c.get("evidence") or {}
        if not ev:
            buckets["NEI"].append(c)
        else:
            seen_labels = {
                str(entry.get("label", "")).upper()
                for entries in ev.values()
                for entry in (entries if isinstance(entries, Sequence) else ())
                if isinstance(entry, Mapping)
            }
            if "CONTRADICT" in seen_labels:
                buckets["CONTRADICT"].append(c)
            else:
                buckets["SUPPORT"].append(c)

    rng = np.random.default_rng(seed)
    fit_split: list[Mapping[str, Any]] = []
    cal_split: list[Mapping[str, Any]] = []
    for lab in STANCE_LABELS:
        items = list(buckets[lab])
        if not items:
            continue
        perm = rng.permutation(len(items))
        n_cal = max(1, int(round(len(items) * calib_fraction))) if len(items) >= 4 else 0
        cal_idx = set(int(x) for x in perm[:n_cal])
        for i, item in enumerate(items):
            if i in cal_idx:
                cal_split.append(item)
            else:
                fit_split.append(item)
    return fit_split, cal_split


class FrontierHybridVerifier:
    """2-Stage Hybrid RRF Retriever + Cross-Feature Reranker + Split-Conformal Stance Verifier.

    Stage 1 fuses Okapi BM25, subword/word TF-IDF Truncated-SVD (`LSA`), and biomedical
    entity/numeric/negation overlap via Reciprocal Rank Fusion (`RRF`).
    Stage 2 applies a pairwise claim-sentence interaction reranker and 3-way stance
    classifier (`SUPPORT` / `CONTRADICT` / `NEI`) trained strictly on `SciFact train`
    and calibrated via Split Conformal Prediction (`1 - alpha = 0.90`).
    """

    def __init__(
        self,
        corpus: Mapping[Any, Mapping[str, Any]],
        *,
        reranker: CrossFeatureReranker | None = None,
        stance_classifier: SplitConformalStanceClassifier | None = None,
        exclude_doc_ids: Iterable[Any] | None = None,
        svd_components: int = 96,
        alpha: float = 0.10,
        seed: int = 42,
        training_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("FrontierHybridVerifier requires a non-empty corpus")
        self.corpus = corpus
        self.alpha = float(alpha)
        self.seed = int(seed)
        excluded = {str(did) for did in (exclude_doc_ids or ())}
        self.excluded_doc_ids = frozenset(excluded)

        self.doc_ids: list[Any] = []
        self.abstract_blobs: list[str] = []
        bg_doc_blobs: list[str] = []
        doc_bg_mask: list[bool] = []

        self.all_sentences: list[str] = []
        self.all_sent_titles: list[str] = []
        self.all_sent_local_pos: list[tuple[int, int]] = []
        bg_sentences: list[str] = []
        sent_bg_mask: list[bool] = []
        self.doc_sent_spans: dict[Any, tuple[int, int]] = {}
        self.sent_to_doc: list[tuple[Any, int]] = []

        for doc_id, doc in corpus.items():
            title = str(doc.get("title", ""))
            sents = [str(s) for s in doc.get("abstract", ())]
            self.doc_ids.append(doc_id)
            blob = (title + " " + " ".join(sents)).strip()
            self.abstract_blobs.append(blob)
            is_bg = str(doc_id) not in excluded
            doc_bg_mask.append(is_bg)
            if is_bg:
                bg_doc_blobs.append(blob)

            start = len(self.all_sentences)
            n_s = len(sents)
            for j, s in enumerate(sents):
                self.all_sentences.append(s)
                self.all_sent_titles.append(title)
                self.all_sent_local_pos.append((j, n_s))
                self.sent_to_doc.append((doc_id, j))
                sent_bg_mask.append(is_bg)
                if is_bg:
                    bg_sentences.append(s)
            self.doc_sent_spans[doc_id] = (start, len(self.all_sentences))

        use_bg_docs = bg_doc_blobs if (excluded and bg_doc_blobs) else None
        use_bg_sents = bg_sentences if (excluded and bg_sentences) else None
        use_doc_mask = doc_bg_mask if (excluded and bg_doc_blobs) else None
        use_sent_mask = sent_bg_mask if (excluded and bg_sentences) else None

        self.doc_bm25 = BM25(self.abstract_blobs, idf_sentences=use_bg_docs)
        self.sent_bm25 = BM25(
            self.all_sentences if self.all_sentences else [""],
            idf_sentences=use_bg_sents,
        )
        self.doc_svd = TruncatedSVDSemanticIndex(
            self.abstract_blobs,
            n_components=svd_components,
            seed=seed,
            idf_mask=use_doc_mask,
        )
        self.sent_svd = TruncatedSVDSemanticIndex(
            self.all_sentences if self.all_sentences else [""],
            n_components=svd_components,
            seed=seed + 1,
            idf_mask=use_sent_mask,
        )
        self.doc_entity_idx = EntityNumericNegationIndex(
            self.abstract_blobs, idf_mask=use_doc_mask
        )
        self.sent_entity_idx = EntityNumericNegationIndex(
            self.all_sentences if self.all_sentences else [""],
            idf_mask=use_sent_mask,
        )
        self.sent_subword_idx = _SubwordConceptCosineIndex(
            self.all_sentences if self.all_sentences else [""],
            idf_mask=use_sent_mask,
        )

        self.reranker = reranker if reranker is not None else CrossFeatureReranker.default_prior()
        self.stance_classifier = (
            stance_classifier
            if stance_classifier is not None
            else SplitConformalStanceClassifier.default_prior(alpha=alpha)
        )
        self.candidate_stance_classifier: SplitConformalStanceClassifier = self.stance_classifier
        self.conflict_threshold: float = 0.32
        self.official_doc_non_nei_threshold: float = 0.62
        self.official_sent_rerank_threshold: float = 0.50
        self.training_provenance: dict[str, Any] = dict(
            training_provenance
            or {
                "trained": False,
                "train_source": "analytic_default_prior",
                "alpha": self.alpha,
            }
        )

    @classmethod
    def fit_on_scifact_train(
        cls,
        corpus: Mapping[Any, Mapping[str, Any]],
        train_claims: Sequence[Mapping[str, Any]],
        *,
        exclude_doc_ids: Iterable[Any] | None = None,
        svd_components: int = 96,
        alpha: float = 0.10,
        calib_fraction: float = 0.25,
        seed: int = 2026,
    ) -> "FrontierHybridVerifier":
        """Train Stage-2 reranker & 3-way stance verifier strictly on `SciFact train` (`809` claims).

        1. Fits Stage-1 BM25, Truncated-SVD (`LSA`), and Entity/Numeric/Negation indices on
           `corpus` while excluding `exclude_doc_ids` (e.g., all `SciFact dev` doc IDs) from
           IDF and SVD basis estimation so zero evaluation-split statistics leak.
        2. Splits `train_claims` (`809` claims) into `train_fit` (`607` claims, `75%`) and
           `train_calib` (`202` claims, `25%`).
        3. Fits `CrossFeatureReranker` on `(claim, candidate_sentence)` pairs from `train_fit`.
        4. Fits separate `SplitConformalStanceClassifier` heads for open-corpus retrieval and
           candidate-abstract verification on `train_fit` using the exact test-time feature paths.
        5. Calibrates `1 - alpha` Mondrian Split Conformal prediction sets and the bilateral
           cross-passage conflict threshold strictly on held-out `train_calib`.
        """
        fit_claims, calib_claims = split_scifact_train_fit_and_calib(
            train_claims, calib_fraction=calib_fraction, seed=seed
        )
        # Exclude `calib_claims`-exclusive docs alongside `exclude_doc_ids` from Stage-1 IDF/SVD
        # so `calib_claims` and held-out `dev` claims are strictly exchangeable at calibration time.
        fit_doc_ids = {
            str(did)
            for c in fit_claims
            for did in [
                *list((c.get("evidence") or {}).keys()),
                *[str(x) for x in (c.get("cited_doc_ids") or ())],
            ]
        }
        calib_doc_ids = {
            str(did)
            for c in calib_claims
            for did in [
                *list((c.get("evidence") or {}).keys()),
                *[str(x) for x in (c.get("cited_doc_ids") or ())],
            ]
        }
        calib_only_doc_ids = calib_doc_ids - fit_doc_ids
        combined_excluded = {str(did) for did in (exclude_doc_ids or ())} | calib_only_doc_ids
        if len(combined_excluded) >= len(corpus):
            combined_excluded = {str(did) for did in (exclude_doc_ids or ())}

        verifier = cls(
            corpus,
            exclude_doc_ids=combined_excluded,
            svd_components=svd_components,
            alpha=alpha,
            seed=seed,
        )

        # --- Step 1: Fit Stage-2 Pairwise Claim-Sentence Reranker on `fit_claims` ---
        fit_ev_claims = [c for c in fit_claims if c.get("evidence")]
        fit_pairs = build_pairs(fit_ev_claims, corpus, strict=False)
        fit_noinfo_pairs = build_noinfo_pairs(fit_claims, corpus, strict=False)

        rerank_x_parts: list[np.ndarray] = []
        rerank_y_parts: list[np.ndarray] = []

        for p in fit_pairs:
            did = p["doc_id"]
            sents = p["sents"]
            if not sents:
                continue
            span = verifier.doc_sent_spans.get(did)
            gids = list(range(span[0], span[1])) if span is not None else None
            loc_pos = [(j, len(sents)) for j in range(len(sents))]
            titles = [p["title"]] * len(sents)
            f_mat, _ = extract_candidate_rerank_matrix(
                p["claim"],
                sents,
                bm25=verifier.sent_bm25,
                svd_index=verifier.sent_svd,
                subword_index=verifier.sent_subword_idx,
                entity_index=verifier.sent_entity_idx,
                sent_global_ids=gids,
                local_positions=loc_pos,
                titles=titles,
            )
            y_sent = np.array(
                [1.0 if j in p["gold"] else 0.0 for j in range(len(sents))],
                dtype=np.float64,
            )
            rerank_x_parts.append(f_mat)
            rerank_y_parts.append(y_sent)

        # Include a deterministic subsample of NOINFO cited-abstract sentences as hard negatives
        for p in fit_noinfo_pairs[:120]:
            did = p["doc_id"]
            sents = p["sents"]
            if not sents:
                continue
            span = verifier.doc_sent_spans.get(did)
            gids = list(range(span[0], span[1])) if span is not None else None
            loc_pos = [(j, len(sents)) for j in range(len(sents))]
            titles = [p["title"]] * len(sents)
            f_mat, _ = extract_candidate_rerank_matrix(
                p["claim"],
                sents,
                bm25=verifier.sent_bm25,
                svd_index=verifier.sent_svd,
                subword_index=verifier.sent_subword_idx,
                entity_index=verifier.sent_entity_idx,
                sent_global_ids=gids,
                local_positions=loc_pos,
                titles=titles,
            )
            rerank_x_parts.append(f_mat)
            rerank_y_parts.append(np.zeros(len(sents), dtype=np.float64))

        if rerank_x_parts:
            rx = np.vstack(rerank_x_parts)
            ry = np.concatenate(rerank_y_parts)
            n_pos = max(1.0, float(np.sum(ry > 0.5)))
            n_neg = max(1.0, float(np.sum(ry <= 0.5)))
            pos_w = min(4.0, math.sqrt(n_neg / n_pos))
            r_sw = np.where(ry > 0.5, pos_w, 1.0)
            rr_w, rr_b, rr_mean, rr_std = _fit_binary_logistic_newton(
                rx, ry, l2_reg=4.0, sample_weight=r_sw
            )
            verifier.reranker = CrossFeatureReranker(rr_w, rr_b, rr_mean, rr_std)

        # --- Step 2: Fit Pooled Polarity + Regime-Aware Stance Heads on `fit_claims` ---
        open_x_rows: list[np.ndarray] = []
        open_y_rows: list[int] = []
        cand_x_rows: list[np.ndarray] = []
        cand_y_rows: list[int] = []

        def _claim_gold_stance(c_obj: Mapping[str, Any]) -> str:
            ev = c_obj.get("evidence") or {}
            if not ev:
                return "NEI"
            labs = {
                str(e.get("label", "")).upper()
                for sets in ev.values()
                for e in (sets if isinstance(sets, Sequence) else ())
                if isinstance(e, Mapping)
            }
            return "CONTRADICT" if "CONTRADICT" in labs else "SUPPORT"

        for c in fit_claims:
            gold_lab = _claim_gold_stance(c)
            y_int = _STANCE_TO_IDX[gold_lab]
            claim_str = str(c["claim"])

            oc_vec, _, _ = verifier._stance_features_for_open_claim(claim_str, doc_top_m=5, sent_top_k=3)
            open_x_rows.append(oc_vec)
            open_y_rows.append(y_int)

            ev = c.get("evidence") or {}
            if ev:
                for raw_did, sets in ev.items():
                    doc_key = verifier._resolve_doc_key(raw_did)
                    if doc_key is None:
                        continue
                    pair_labs = {
                        str(e.get("label", "")).upper()
                        for e in (sets if isinstance(sets, Sequence) else ())
                        if isinstance(e, Mapping)
                    }
                    p_lab = "CONTRADICT" if "CONTRADICT" in pair_labs else "SUPPORT"
                    doc_obj = corpus[doc_key]
                    vec_doc, _, _ = verifier._stance_features_for_candidate_sentences(
                        claim_str,
                        list(doc_obj.get("abstract", ())),
                        title=str(doc_obj.get("title", "")),
                        top_k=3,
                    )
                    cand_x_rows.append(vec_doc)
                    cand_y_rows.append(_STANCE_TO_IDX[p_lab])
            else:
                for raw_did in (c.get("cited_doc_ids") or ())[:1]:
                    doc_key = verifier._resolve_doc_key(raw_did)
                    if doc_key is not None:
                        doc_obj = corpus[doc_key]
                        vec_doc, _, _ = verifier._stance_features_for_candidate_sentences(
                            claim_str,
                            list(doc_obj.get("abstract", ())),
                            title=str(doc_obj.get("title", "")),
                            top_k=3,
                        )
                        cand_x_rows.append(vec_doc)
                        cand_y_rows.append(_STANCE_TO_IDX["NEI"])

        def _fit_cv_stance_head(
            x_rows: Sequence[np.ndarray], y_rows: Sequence[int]
        ) -> SplitConformalStanceClassifier:
            sx = np.vstack(x_rows)
            sy = np.asarray(y_rows, dtype=np.int32)
            counts = np.bincount(sy, minlength=3).astype(np.float64)
            inv_freq = np.sum(counts) / (3.0 * np.maximum(counts, 1.0))

            best_l2 = 4.0
            best_pow = 0.5
            best_f1 = -1.0
            best_oof_logits = np.zeros((len(sy), 3), dtype=np.float64)

            for w_pow in (0.0, 0.5, 1.0):
                s_sw = np.power(inv_freq[sy], w_pow)
                for cand_l2 in (1.5, 4.0, 10.0, 20.0):
                    oof_preds: list[str] = []
                    oof_true: list[str] = []
                    oof_logits = np.zeros((len(sy), 3), dtype=np.float64)
                    for fold in range(3):
                        val_mask = (np.arange(len(sy)) % 3) == fold
                        tr_mask = ~val_mask
                        if not np.any(tr_mask) or not np.any(val_mask):
                            continue
                        w_cv, b_cv, m_cv, s_cv = _fit_multinomial_logistic(
                            sx[tr_mask], sy[tr_mask], l2_reg=cand_l2, sample_weight=s_sw[tr_mask]
                        )
                        z_val = (sx[val_mask] - m_cv) / s_cv
                        val_logits = z_val @ w_cv + b_cv[None, :]
                        oof_logits[val_mask] = val_logits
                        for pi, yi in zip(np.argmax(val_logits, axis=1), sy[val_mask]):
                            oof_preds.append(STANCE_LABELS[int(pi)])
                            oof_true.append(STANCE_LABELS[int(yi)])
                    f1_cv = macro_f1_score(oof_true, oof_preds, labels=STANCE_LABELS)
                    if not math.isnan(f1_cv) and f1_cv > best_f1:
                        best_f1 = f1_cv
                        best_l2 = cand_l2
                        best_pow = w_pow
                        best_oof_logits = oof_logits

            s_sw_best = np.power(inv_freq[sy], best_pow)
            st_w, st_b, st_mean, st_std = _fit_multinomial_logistic(
                sx, sy, l2_reg=best_l2, sample_weight=s_sw_best
            )

            # Post-hoc temperature scaling on OOF `fit_claims` logits (0% `calib_claims` leakage)
            best_temp = 1.0
            if len(sy) >= 12:
                best_nll = float("inf")
                for cand_t in (0.45, 0.60, 0.75, 0.90, 1.0, 1.15, 1.35):
                    scaled = best_oof_logits / cand_t
                    scaled -= np.max(scaled, axis=1, keepdims=True)
                    p_oof = np.exp(scaled)
                    p_oof /= np.sum(p_oof, axis=1, keepdims=True)
                    nll = -float(np.mean(np.log(np.clip(p_oof[np.arange(len(sy)), sy], 1e-9, 1.0))))
                    if nll < best_nll:
                        best_nll = nll
                        best_temp = cand_t

            return SplitConformalStanceClassifier(
                st_w,
                st_b,
                st_mean,
                st_std,
                alpha=alpha,
                conformal_mode="mondrian",
                temperature=best_temp,
                is_calibrated=False,
            )

        if open_x_rows:
            verifier.stance_classifier = _fit_cv_stance_head(open_x_rows, open_y_rows)
        if cand_x_rows:
            verifier.candidate_stance_classifier = _fit_cv_stance_head(cand_x_rows, cand_y_rows)
        else:
            verifier.candidate_stance_classifier = verifier.stance_classifier

        # --- Step 3: Split Conformal & Bilateral Conflict Calibration on Held-Out `calib_claims` ---
        cal_probs_list: list[np.ndarray] = []
        cal_labels_list: list[str] = []
        cand_cal_probs: list[np.ndarray] = []
        cand_cal_labels: list[str] = []
        neg_conflict_scores: list[float] = []

        for c in calib_claims:
            gold_lab = _claim_gold_stance(c)
            claim_str = str(c["claim"])
            c_vec, c_sup, c_con = verifier._stance_features_for_open_claim(
                claim_str, doc_top_m=5, sent_top_k=3
            )
            p_vec, _ = verifier.stance_classifier.predict_proba_from_features(
                c_vec, sup_mass=c_sup, con_mass=c_con
            )
            cal_probs_list.append(p_vec)
            cal_labels_list.append(gold_lab)

            ev = c.get("evidence") or {}
            if ev:
                for raw_did, sets in ev.items():
                    doc_key = verifier._resolve_doc_key(raw_did)
                    if doc_key is None:
                        continue
                    pair_labs = {
                        str(e.get("label", "")).upper()
                        for e in (sets if isinstance(sets, Sequence) else ())
                        if isinstance(e, Mapping)
                    }
                    p_lab = "CONTRADICT" if "CONTRADICT" in pair_labs else "SUPPORT"
                    doc_obj = corpus[doc_key]
                    abs_sents = [str(s) for s in doc_obj.get("abstract", ())]
                    d_vec, d_sup, d_con = verifier._stance_features_for_candidate_sentences(
                        claim_str,
                        abs_sents,
                        title=str(doc_obj.get("title", "")),
                        top_k=3,
                    )
                    dp_vec, _ = verifier.candidate_stance_classifier.predict_proba_from_features(
                        d_vec, sup_mass=d_sup, con_mass=d_con
                    )
                    cand_cal_probs.append(dp_vec)
                    cand_cal_labels.append(p_lab)
                    if len(abs_sents) >= 2:
                        neg_conflict_scores.append(
                            verifier.score_bilateral_conflict(claim_str, abs_sents)["conflict_score"]
                        )
                        if len(abs_sents) >= 4:
                            mid = len(abs_sents) // 2
                            neg_conflict_scores.append(
                                verifier.score_bilateral_conflict(
                                    claim_str,
                                    [" ".join(abs_sents[:mid]), " ".join(abs_sents[mid:])],
                                )["conflict_score"]
                            )
            else:
                for raw_did in (c.get("cited_doc_ids") or ()):
                    doc_key = verifier._resolve_doc_key(raw_did)
                    if doc_key is not None:
                        doc_obj = corpus[doc_key]
                        d_vec, d_sup, d_con = verifier._stance_features_for_candidate_sentences(
                            claim_str,
                            list(doc_obj.get("abstract", ())),
                            title=str(doc_obj.get("title", "")),
                            top_k=3,
                        )
                        dp_vec, _ = verifier.candidate_stance_classifier.predict_proba_from_features(
                            d_vec, sup_mass=d_sup, con_mass=d_con
                        )
                        cand_cal_probs.append(dp_vec)
                        cand_cal_labels.append("NEI")

        conformal_summary: dict[str, float] = {}
        cand_conformal_summary: dict[str, float] = {}
        if cal_probs_list:
            cal_mat = np.vstack(cal_probs_list)
            conformal_summary = verifier.stance_classifier.calibrate_split_conformal(
                cal_mat, cal_labels_list, alpha=alpha
            )
        if cand_cal_probs:
            cand_conformal_summary = verifier.candidate_stance_classifier.calibrate_split_conformal(
                np.vstack(cand_cal_probs), cand_cal_labels, alpha=alpha
            )

        if len(neg_conflict_scores) >= 25:
            q95 = SplitConformalStanceClassifier.conformal_quantile(neg_conflict_scores, alpha=0.05)
            verifier.conflict_threshold = float(np.clip(max(0.42, q95), 0.42, 0.85))

        # Calibrate official SciFact abstract selection threshold strictly on `fit_claims`
        if len(fit_claims) >= 20:
            fit_doc_cands: list[tuple[str, str, str, float, list[tuple[int, float]]]] = []
            for fc in fit_claims:
                fc_id = str(fc.get("id"))
                fc_str = str(fc["claim"])
                for r_rank, did in enumerate(verifier.retrieve_stage1_docs(fc_str, top_m=3)):
                    lo, hi = verifier.doc_sent_spans[did]
                    if lo >= hi:
                        continue
                    cp = verifier.verify_candidate_sentences(
                        fc_str,
                        verifier.all_sentences[lo:hi],
                        title=verifier.all_sent_titles[lo],
                        top_k=3,
                    )
                    p_sup = float(cp.probabilities["SUPPORT"])
                    p_con = float(cp.probabilities["CONTRADICT"])
                    non_nei = p_sup + p_con
                    top_rr = cp.top_rationale_scores[0] if cp.top_rationale_scores else 0.0
                    doc_conf = 0.55 * non_nei + 0.45 * top_rr - 0.04 * float(r_rank)
                    d_lab = "CONTRADICT" if p_con > p_sup else "SUPPORT"
                    pairs_idx_sc = list(zip(cp.top_rationale_indices, cp.top_rationale_scores))
                    fit_doc_cands.append((fc_id, str(did), d_lab, doc_conf, pairs_idx_sc))

            best_off_f1 = -1.0
            best_doc_th = verifier.official_doc_non_nei_threshold
            for cand_th in (0.52, 0.58, 0.64, 0.68, 0.72, 0.76):
                trial_preds: dict[str, dict[str, dict[str, Any]]] = {}
                for fc_id, did_s, d_lab, d_conf, idx_sc in fit_doc_cands:
                    if d_conf < cand_th:
                        continue
                    s_sel = [ix for ix, sc in idx_sc if sc >= 0.48][:2]
                    if not s_sel and idx_sc:
                        s_sel = [idx_sc[0][0]]
                    trial_preds.setdefault(fc_id, {})[did_s] = {
                        "label": d_lab,
                        "sentences": s_sel,
                    }
                off_m = compute_official_scifact_metrics(fit_claims, trial_preds, max_sentences_per_doc=3)
                score_f1 = 0.5 * off_m["abstract_label_only"]["f1"] + 0.5 * off_m["sentence_label"]["f1"]
                if score_f1 > best_off_f1:
                    best_off_f1 = score_f1
                    best_doc_th = cand_th
            verifier.official_doc_non_nei_threshold = best_doc_th

        train_id_digest = hashlib.sha256(
            ",".join(sorted(str(c.get("id", "")) for c in train_claims)).encode("utf-8")
        ).hexdigest()[:16]
        verifier.training_provenance = {
            "trained": True,
            "train_source": "scifact_train_only",
            "total_train_claims": len(train_claims),
            "train_fit_claims": len(fit_claims),
            "train_calib_claims": len(calib_claims),
            "train_claim_ids_sha256_prefix": train_id_digest,
            "excluded_eval_doc_count": len(verifier.excluded_doc_ids),
            "svd_components": verifier.sent_svd.n_components,
            "alpha": verifier.alpha,
            "conformal_calibration": conformal_summary,
            "candidate_conformal_calibration": cand_conformal_summary,
            "conflict_threshold": round(float(verifier.conflict_threshold), 6),
        }
        return verifier

    def save_calibrator(self, path: str | Path) -> Path:
        """Save compact Stage-2 reranker, calibrated conformal stance heads, and conflict threshold."""
        payload = {
            "schema_version": "lab-ask-why-calibrator/v1",
            "reranker": self.reranker.to_dict(),
            "stance_classifier": self.stance_classifier.to_calibrator_dict(),
            "candidate_stance_classifier": self.candidate_stance_classifier.to_calibrator_dict(),
            "conflict_threshold": round(float(self.conflict_threshold), 8),
            "training_provenance": dict(self.training_provenance),
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return dest

    def apply_calibrator_file(self, path: str | Path) -> "FrontierHybridVerifier":
        """Load saved Stage-2 reranker and calibrated Mondrian stance heads onto `self`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "reranker" in payload:
            self.reranker = CrossFeatureReranker.from_dict(payload["reranker"])
        if "stance_classifier" in payload:
            self.stance_classifier = SplitConformalStanceClassifier.from_calibrator_dict(
                payload["stance_classifier"]
            )
        if "candidate_stance_classifier" in payload:
            self.candidate_stance_classifier = SplitConformalStanceClassifier.from_calibrator_dict(
                payload["candidate_stance_classifier"]
            )
        else:
            self.candidate_stance_classifier = self.stance_classifier
        if "conflict_threshold" in payload:
            self.conflict_threshold = float(payload["conflict_threshold"])
        if "training_provenance" in payload and isinstance(payload["training_provenance"], Mapping):
            self.training_provenance = dict(payload["training_provenance"])
        return self

    def _resolve_doc_key(self, raw_doc_id: Any) -> Any | None:
        if raw_doc_id in self.doc_sent_spans:
            return raw_doc_id
        s_id = str(raw_doc_id)
        if s_id in self.doc_sent_spans:
            return s_id
        if s_id.isdigit() and int(s_id) in self.doc_sent_spans:
            return int(s_id)
        return None

    def _stance_features_for_candidate_sentences(
        self,
        claim: str,
        sentences: Sequence[str],
        *,
        title: str = "",
        top_k: int = 3,
    ) -> tuple[np.ndarray, float, float]:
        """Extract candidate-sentence stance features identically at fit, calibration, and test time."""
        clean_sents = [str(s) for s in sentences if str(s).strip()]
        if not clean_sents:
            return _extract_stance_feature_vector(
                claim, [], np.zeros((0, 26)), {}, np.zeros(0), top_k=top_k
            )
        loc_pos = [(j, len(clean_sents)) for j in range(len(clean_sents))]
        titles = [title] * len(clean_sents)
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            clean_sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=None,
            entity_index=None,
            sent_global_ids=None,
            local_positions=loc_pos,
            titles=titles,
        )
        r_probs = self.reranker.score_features(f_mat)
        return _extract_stance_feature_vector(claim, clean_sents, f_mat, chans, r_probs, top_k=top_k)

    def score_bilateral_conflict(
        self,
        claim: str,
        passages_or_sentences: Sequence[str],
    ) -> dict[str, Any]:
        """Compute Bilateral Cross-Passage Conflict Score:

        `score_conflict = max_{i != j} min(P(SUPPORT | claim, u_i), P(CONTRADICT | claim, u_j))`
        combining (a) relevance-gated explicit lexical/directional polarity opposition across
        distinct sentences and (b) cross-passage counterfactual entity/numeric substitution
        when distinct passages share a strong subject-predicate anchor yet omit a specific
        target entity/value affirmed by another passage.
        """
        raw_items = [str(x).strip() for x in passages_or_sentences if str(x).strip()]
        doc_split_re = re.compile(r"(?=\bDocument(?:\s+\d+)?\s*:)")
        if len(raw_items) == 1:
            doc_blocks = [d.strip() for d in doc_split_re.split(raw_items[0]) if d.strip()]
            has_multi_passage = len(doc_blocks) >= 2
            passages = doc_blocks if has_multi_passage else raw_items
        else:
            # Distinguish multi-passage blocks from a flat sentence list of one abstract
            avg_words = float(np.mean([len(x.split()) for x in raw_items])) if raw_items else 0.0
            has_multi_passage = len(raw_items) <= 6 and avg_words >= 32.0
            passages = raw_items

        # Flatten sentences while recording parent passage index
        all_sents: list[str] = []
        sent_passage_id: list[int] = []
        for p_idx, p_txt in enumerate(passages):
            if has_multi_passage:
                parts = [
                    s.strip()
                    for line in p_txt.splitlines()
                    for s in re.split(r"(?<=[.!?])\s+", line)
                    if s.strip()
                ]
            else:
                parts = [p_txt]
            for s in parts:
                all_sents.append(s)
                sent_passage_id.append(p_idx)

        if len(all_sents) < 2:
            return {
                "conflict_score": 0.0,
                "is_conflict": False,
                "max_support_prob": 0.0,
                "max_contradict_prob": 0.0,
            }

        loc_pos = [(j, len(all_sents)) for j in range(len(all_sents))]
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            all_sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=None,
            entity_index=None,
            sent_global_ids=None,
            local_positions=loc_pos,
            titles=None,
        )
        r_probs = self.reranker.score_features(f_mat)
        idf_cov = chans["idf_cov"]
        bm25_norm = chans["bm25_norm"]
        neg_xor = chans["neg_xor"]
        dir_opp = chans["dir_opp"]
        dir_same = chans["dir_same"]
        num_mis = chans["num_mismatch"]

        n_s = len(all_sents)
        p_sup_s = np.zeros(n_s, dtype=np.float64)
        p_con_s = np.zeros(n_s, dtype=np.float64)

        for i, s_txt in enumerate(all_sents):
            prof = _sentence_profile(s_txt)
            has_intro = prof[8]
            if has_intro and i == 0:
                continue
            pol = classify_rationale_polarity(claim, s_txt)
            rel = max(float(r_probs[i]), 0.65 * float(idf_cov[i]) + 0.35 * float(bm25_norm[i]))
            if float(idf_cov[i]) < 0.30:
                continue
            con_cue = (
                (0.55 if pol == "CONTRADICT" else 0.0)
                + 0.45 * float(neg_xor[i])
                + 0.50 * min(1.0, float(dir_opp[i]))
                + 0.35 * float(num_mis[i])
            )
            if con_cue >= 0.50:
                p_con_s[i] = min(1.0, rel * min(1.0, con_cue))
            elif pol == "SUPPORT":
                sup_cue = min(1.0, 0.92 + 0.08 * min(1.0, float(dir_same[i])))
                p_sup_s[i] = min(1.0, rel * sup_cue)

        best_explicit = 0.0
        for i in range(n_s):
            if p_sup_s[i] <= 0.0:
                continue
            for j in range(n_s):
                if i == j or p_con_s[j] <= 0.0:
                    continue
                cand = min(float(p_sup_s[i]), float(p_con_s[j]))
                if cand > best_explicit:
                    best_explicit = cand

        # Cross-passage counterfactual entity/value substitution score when >= 2 distinct passages
        best_subst = 0.0
        if has_multi_passage and len(passages) >= 2:
            q_toks = list(
                dict.fromkeys(
                    t
                    for t in tokenize(claim)
                    if t not in _NEGATION_TOKENS and t not in {"proposed", "answer"}
                )
            )
            q_idfs = np.array(
                [self.sent_bm25.idf.get(t, self.sent_bm25.default_idf) for t in q_toks],
                dtype=np.float64,
            )
            q_tot_idf = float(np.sum(q_idfs)) if q_idfs.size > 0 else 1.0
            pass_toks = [
                [t for t in tokenize(p) if t not in _NEGATION_TOKENS]
                for p in passages
            ]
            pass_matched: list[set[int]] = []
            for p_t in pass_toks:
                m_set = {
                    qi
                    for qi, qt in enumerate(q_toks)
                    if p_t and max((_token_similarity(qt, pt) for pt in p_t), default=0.0) >= 0.75
                }
                pass_matched.append(m_set)

            for a in range(len(passages)):
                ma = pass_matched[a]
                if not ma:
                    continue
                sup_cov = float(np.sum(q_idfs[list(ma)])) / q_tot_idf
                if sup_cov < 0.60:
                    continue
                for b in range(len(passages)):
                    if a == b:
                        continue
                    mb = pass_matched[b]
                    shared = ma & mb
                    missing = ma - mb
                    if len(shared) < 2 or not missing:
                        continue
                    # Verify missing claim entity is genuinely absent from passage b
                    truly_absent = all(
                        max(
                            (_token_similarity(q_toks[mi], pt) for pt in pass_toks[b]),
                            default=0.0,
                        )
                        < 0.45
                        for mi in missing
                    )
                    if not truly_absent:
                        continue
                    # Require passage b to introduce distinct non-claim content tokens in the anchor sentence
                    b_sents_idx = [si for si, pid in enumerate(sent_passage_id) if pid == b]
                    best_anchor_sent_ov = 0.0
                    has_alt_entity = False
                    q_set_all = set(q_toks)
                    for si in b_sents_idx:
                        stoks = [t for t in tokenize(all_sents[si]) if t not in _NEGATION_TOKENS]
                        sh_in_s = sum(
                            1
                            for qi in shared
                            if any(_token_similarity(q_toks[qi], st) >= 0.75 for st in stoks)
                        )
                        if sh_in_s >= 2:
                            ov = float(sh_in_s) / float(len(shared))
                            if ov > best_anchor_sent_ov:
                                best_anchor_sent_ov = ov
                            if any(
                                max((_token_similarity(qt, st) for qt in q_set_all), default=0.0) < 0.40
                                for st in stoks
                            ):
                                has_alt_entity = True
                    if not has_alt_entity or best_anchor_sent_ov < 0.50:
                        continue
                    anchor_idf_frac = float(np.sum(q_idfs[list(shared)])) / q_tot_idf
                    if anchor_idf_frac < 0.30:
                        continue
                    con_subst = min(1.0, 0.92 * min(1.0, anchor_idf_frac / 0.45) * best_anchor_sent_ov)
                    cand_sub = min(sup_cov, con_subst)
                    if cand_sub > best_subst:
                        best_subst = cand_sub

        score = max(best_explicit, best_subst)
        return {
            "conflict_score": float(score),
            "is_conflict": bool(score >= self.conflict_threshold),
            "max_support_prob": float(max(float(np.max(p_sup_s)), best_subst)),
            "max_contradict_prob": float(max(float(np.max(p_con_s)), best_subst)),
        }

    def _stance_features_for_open_claim(
        self,
        claim: str,
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
    ) -> tuple[np.ndarray, float, float]:
        top_docs = self.retrieve_stage1_docs(claim, top_m=doc_top_m)
        gids: list[int] = []
        for did in top_docs:
            lo, hi = self.doc_sent_spans[did]
            gids.extend(range(lo, hi))
        sents = [self.all_sentences[g] for g in gids]
        if not sents:
            return _extract_stance_feature_vector(
                claim, [], np.zeros((0, 26)), {}, np.zeros(0), top_k=sent_top_k
            )
        loc_pos = [self.all_sent_local_pos[g] for g in gids]
        titles = [self.all_sent_titles[g] for g in gids]
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=self.sent_subword_idx,
            entity_index=self.sent_entity_idx,
            sent_global_ids=gids,
            local_positions=loc_pos,
            titles=titles,
        )
        r_probs = self.reranker.score_features(f_mat)
        return _extract_stance_feature_vector(
            claim, sents, f_mat, chans, r_probs, top_k=sent_top_k
        )

    def retrieve_stage1_docs(self, claim: str, *, top_m: int = 5) -> list[Any]:
        """Stage 1 document retrieval fusing BM25 + Truncated-SVD (`LSA`) + Entity/Numeric RRF."""
        bm25_sc = self.doc_bm25.score_indexed(claim)
        bm25_norm = self.doc_bm25.score_indexed_normalized(claim)
        svd_raw = self.doc_svd.cosine_scores(claim)
        svd_sc = np.where(svd_raw >= 0.25, svd_raw, 0.0)
        ent_raw = self.doc_entity_idx.overlap_scores(claim)
        ent_sc = np.where((bm25_norm >= 0.16) | (svd_raw >= 0.30), ent_raw, 0.0)
        rrf = _reciprocal_rank_fusion(
            [bm25_sc, svd_sc, ent_sc],
            rrf_k=60,
            weights=[1.75, 0.30, 0.25],
        )
        # Preserve quantitative BM25 lexical strength while breaking close/paraphrase ties via RRF + LSA
        bm25_max = float(np.max(bm25_sc)) if bm25_sc.size > 0 else 0.0
        bm25_rel = (bm25_sc / bm25_max) if bm25_max > 0.0 else np.zeros_like(bm25_sc)
        combined = 0.72 * bm25_rel + 0.28 * (rrf / ((1.75 + 0.30 + 0.25) / 61.0)) + 0.08 * np.maximum(0.0, svd_raw)
        m = max(1, min(int(top_m), len(self.doc_ids)))
        order = np.lexsort((-svd_raw, -bm25_sc, -combined))[:m]
        return [self.doc_ids[int(i)] for i in order]

    def rank_sentences_in_doc(self, claim: str, doc_id: Any) -> tuple[list[int], np.ndarray]:
        """Rank sentences inside `doc_id` using Stage-2 `CrossFeatureReranker`."""
        resolved = self._resolve_doc_key(doc_id)
        if resolved is None:
            raise KeyError(f"doc_id {doc_id!r} not in indexed corpus")
        lo, hi = self.doc_sent_spans[resolved]
        gids = list(range(lo, hi))
        sents = [self.all_sentences[g] for g in gids]
        if not sents:
            return [], np.zeros(0, dtype=np.float64)
        loc_pos = [self.all_sent_local_pos[g] for g in gids]
        titles = [self.all_sent_titles[g] for g in gids]
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=self.sent_subword_idx,
            entity_index=self.sent_entity_idx,
            sent_global_ids=gids,
            local_positions=loc_pos,
            titles=titles,
        )
        r_probs = self.reranker.score_features(f_mat)
        order = np.lexsort((-chans["stage1_rrf"], -r_probs))
        return [int(i) for i in order], r_probs

    def verify_candidate_sentences(
        self,
        claim: str,
        sentences: Sequence[str],
        *,
        title: str = "",
        top_k: int = 3,
        alpha: float | None = None,
    ) -> ConformalStancePrediction:
        """Rerank arbitrary candidate sentences and predict 3-way Split Conformal stance set."""
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        clean_sents = [str(s) for s in sentences if str(s).strip()]
        if not clean_sents:
            empty_vec, sup_m, con_m = _extract_stance_feature_vector(
                claim, [], np.zeros((0, 26)), {}, np.zeros(0), top_k=top_k
            )
            probs, is_conf = self.candidate_stance_classifier.predict_proba_from_features(
                empty_vec, sup_mass=sup_m, con_mass=con_m
            )
            pset = self.candidate_stance_classifier.prediction_set_from_probs(probs, alpha_override=alpha)
            return ConformalStancePrediction(
                predicted_label="NEI",
                probabilities={STANCE_LABELS[i]: float(probs[i]) for i in range(3)},
                conformal_set=pset if "NEI" in pset else ["NEI", *pset],
                alpha=float(alpha if alpha is not None else self.candidate_stance_classifier.alpha),
                is_contested_conflict=is_conf,
                top_rationale_indices=[],
                top_rationale_scores=[],
            )

        loc_pos = [(j, len(clean_sents)) for j in range(len(clean_sents))]
        titles = [title] * len(clean_sents)
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            clean_sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=None,
            entity_index=None,
            sent_global_ids=None,
            local_positions=loc_pos,
            titles=titles,
        )
        r_probs = self.reranker.score_features(f_mat)
        order = np.lexsort((-chans["stage1_rrf"], -r_probs))
        k = max(1, min(int(top_k), len(clean_sents)))
        top_idx = [int(i) for i in order[:k]]

        s_vec, sup_m, con_m = _extract_stance_feature_vector(
            claim, clean_sents, f_mat, chans, r_probs, top_k=top_k
        )
        probs, _ = self.candidate_stance_classifier.predict_proba_from_features(
            s_vec, sup_mass=sup_m, con_mass=con_m
        )
        pset = list(
            self.candidate_stance_classifier.prediction_set_from_probs(probs, alpha_override=alpha)
        )
        b_conf = self.score_bilateral_conflict(claim, clean_sents)
        is_conf = bool(b_conf["is_conflict"])
        if is_conf and (alpha is None or float(alpha) <= 0.20):
            for req_lab in ("SUPPORT", "CONTRADICT"):
                if req_lab not in pset:
                    pset.append(req_lab)
            pset = [lab for lab in STANCE_LABELS if lab in pset]

        pred_lab = STANCE_LABELS[int(np.argmax(probs))]
        return ConformalStancePrediction(
            predicted_label=pred_lab,
            probabilities={STANCE_LABELS[i]: float(probs[i]) for i in range(3)},
            conformal_set=pset,
            alpha=float(alpha if alpha is not None else self.candidate_stance_classifier.alpha),
            is_contested_conflict=is_conf,
            top_rationale_indices=top_idx,
            top_rationale_scores=[float(r_probs[i]) for i in top_idx],
        )

    def verify_claim_open_corpus(
        self,
        claim: str,
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Run full 2-stage Hybrid RRF + Stage-2 Reranker + Split-Conformal Stance Verification."""
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        top_docs = self.retrieve_stage1_docs(claim, top_m=doc_top_m)
        gids: list[int] = []
        for did in top_docs:
            lo, hi = self.doc_sent_spans[did]
            gids.extend(range(lo, hi))

        if not gids:
            pred = self.verify_candidate_sentences(claim, [], top_k=sent_top_k, alpha=alpha)
            return {
                "claim": claim,
                "top_doc_ids": top_docs,
                "top_sentences": [],
                "stance": pred.to_dict(),
            }

        sents = [self.all_sentences[g] for g in gids]
        loc_pos = [self.all_sent_local_pos[g] for g in gids]
        titles = [self.all_sent_titles[g] for g in gids]
        f_mat, chans = extract_candidate_rerank_matrix(
            claim,
            sents,
            bm25=self.sent_bm25,
            svd_index=self.sent_svd,
            subword_index=self.sent_subword_idx,
            entity_index=self.sent_entity_idx,
            sent_global_ids=gids,
            local_positions=loc_pos,
            titles=titles,
        )
        r_probs = self.reranker.score_features(f_mat)
        order = np.lexsort((-chans["stage1_rrf"], -r_probs))
        k = max(1, min(int(sent_top_k), len(gids)))

        ranked_sents: list[dict[str, Any]] = []
        for pos in order[:k]:
            p_int = int(pos)
            gid = gids[p_int]
            did, loc_idx = self.sent_to_doc[gid]
            sent_txt = self.all_sentences[gid]
            ranked_sents.append(
                {
                    "doc_id": did,
                    "sent_idx": loc_idx,
                    "rerank_score": float(r_probs[p_int]),
                    "bm25_score": float(chans["bm25_raw"][p_int]),
                    "lsa_cosine": float(chans["lsa_cosine"][p_int]),
                    "polarity": classify_rationale_polarity(claim, sent_txt),
                    "sentence": sent_txt,
                    "title": self.all_sent_titles[gid],
                }
            )

        s_vec, sup_m, con_m = _extract_stance_feature_vector(
            claim, sents, f_mat, chans, r_probs, top_k=sent_top_k
        )
        probs, _ = self.stance_classifier.predict_proba_from_features(
            s_vec, sup_mass=sup_m, con_mass=con_m
        )
        pset = self.stance_classifier.prediction_set_from_probs(probs, alpha_override=alpha)
        pred_lab = STANCE_LABELS[int(np.argmax(probs))]
        top_sent_texts = [str(x["sentence"]) for x in ranked_sents]
        b_conf = self.score_bilateral_conflict(claim, top_sent_texts)
        stance_pred = ConformalStancePrediction(
            predicted_label=pred_lab,
            probabilities={STANCE_LABELS[i]: float(probs[i]) for i in range(3)},
            conformal_set=pset,
            alpha=float(alpha if alpha is not None else self.stance_classifier.alpha),
            is_contested_conflict=bool(b_conf["is_conflict"]),
            top_rationale_indices=[int(x["sent_idx"]) for x in ranked_sents],
            top_rationale_scores=[float(x["rerank_score"]) for x in ranked_sents],
        )
        return {
            "claim": claim,
            "top_doc_ids": top_docs,
            "top_sentences": ranked_sents,
            "stance": stance_pred.to_dict(),
        }

    def evaluate_scifact_split(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 3,
        doc_top_m: int = 5,
    ) -> dict[str, Any]:
        """Compare Random, BM25, Dense-SVD, Stage-1 RRF, and Hybrid+Reranker on a SciFact split."""
        ev_claims = [c for c in claims if c.get("evidence")]
        pairs = build_pairs(ev_claims, self.corpus, strict=True)
        noinfo_pairs = build_noinfo_pairs(claims, self.corpus, strict=False)

        rand_h1: list[float] = []
        rand_h3: list[float] = []
        rand_mrr: list[float] = []
        for seed_i in range(20):
            rng = np.random.default_rng(seed_i)
            r_perm = [list(rng.permutation(len(p["sents"]))) for p in pairs]
            rand_h1.append(hit_at(r_perm, pairs, 1))
            rand_h3.append(hit_at(r_perm, pairs, top_k))
            rand_mrr.append(mean_reciprocal_rank(r_perm, pairs))

        bm25_ranks: list[list[int]] = []
        svd_ranks: list[list[int]] = []
        rrf_ranks: list[list[int]] = []
        hybrid_ranks: list[list[int]] = []

        bm25_stance_true: list[str] = []
        bm25_stance_pred: list[str] = []
        svd_stance_pred: list[str] = []
        cand_conformal_covered: list[bool] = []
        cand_conformal_sizes: list[int] = []
        cand_conformal_by_class: dict[str, list[bool]] = {lab: [] for lab in STANCE_LABELS}

        for p in pairs:
            did = self._resolve_doc_key(p["doc_id"])
            assert did is not None
            lo, hi = self.doc_sent_spans[did]
            gids = list(range(lo, hi))
            sents = p["sents"]
            loc_pos = [self.all_sent_local_pos[g] for g in gids]
            titles = [self.all_sent_titles[g] for g in gids]
            f_mat, chans = extract_candidate_rerank_matrix(
                p["claim"],
                sents,
                bm25=self.sent_bm25,
                svd_index=self.sent_svd,
                subword_index=self.sent_subword_idx,
                entity_index=self.sent_entity_idx,
                sent_global_ids=gids,
                local_positions=loc_pos,
                titles=titles,
            )
            r_probs = self.reranker.score_features(f_mat)

            b_ord = rank_scores(chans["bm25_raw"])
            s_ord = rank_scores(chans["lsa_cosine"])
            rrf_ord = rank_scores(chans["stage1_rrf"])
            hyb_ord = [int(i) for i in np.lexsort((-chans["stage1_rrf"], -r_probs))]

            bm25_ranks.append(b_ord)
            svd_ranks.append(s_ord)
            rrf_ranks.append(rrf_ord)
            hybrid_ranks.append(hyb_ord)

            g_pair_lab = canonicalize_stance_label(p["label"])
            bm25_stance_true.append(g_pair_lab)
            b_pol = classify_rationale_polarity(p["claim"], sents[b_ord[0]]) if b_ord else "NEUTRAL"
            bm25_stance_pred.append("CONTRADICT" if b_pol == "CONTRADICT" else ("SUPPORT" if b_pol == "SUPPORT" else "NEI"))
            s_pol = classify_rationale_polarity(p["claim"], sents[s_ord[0]]) if s_ord else "NEUTRAL"
            svd_stance_pred.append("CONTRADICT" if s_pol == "CONTRADICT" else ("SUPPORT" if s_pol == "SUPPORT" else "NEI"))

            c_pred = self.verify_candidate_sentences(p["claim"], sents, title=p["title"], top_k=top_k)
            c_cov = g_pair_lab in c_pred.conformal_set
            cand_conformal_covered.append(c_cov)
            cand_conformal_sizes.append(len(c_pred.conformal_set))
            cand_conformal_by_class[g_pair_lab].append(c_cov)

        for np_item in noinfo_pairs:
            c_pred_ni = self.verify_candidate_sentences(
                np_item["claim"], np_item["sents"], title=np_item["title"], top_k=top_k
            )
            c_cov_ni = "NEI" in c_pred_ni.conformal_set
            cand_conformal_covered.append(c_cov_ni)
            cand_conformal_sizes.append(len(c_pred_ni.conformal_set))
            cand_conformal_by_class["NEI"].append(c_cov_ni)

        # Evaluate 3-way stance & Split Conformal coverage across ALL claims (`SUPPORT`/`CONTRADICT`/`NEI`)
        y_true_all: list[str] = []
        y_pred_hybrid: list[str] = []
        y_pred_bm25_3way: list[str] = []
        y_pred_svd_3way: list[str] = []
        conformal_covered: list[bool] = []
        conformal_sizes: list[int] = []
        conformal_by_class: dict[str, list[bool]] = {lab: [] for lab in STANCE_LABELS}
        multi_label_sets = 0

        open_doc_hits: list[float] = []
        open_sent_h1: list[float] = []
        open_sent_h3: list[float] = []
        open_sent_mrr: list[float] = []
        ev_max_rerank: list[float] = []
        ni_max_rerank: list[float] = []

        bm25_doc_hits: list[float] = []
        svd_doc_hits: list[float] = []
        bm25_official_preds: dict[str, dict[str, dict[str, Any]]] = {}
        hybrid_official_preds: dict[str, dict[str, dict[str, Any]]] = {}

        for c in claims:
            cid_str = str(c.get("id"))
            claim_str = str(c["claim"])
            ev = c.get("evidence") or {}
            if not ev:
                gold_lab = "NEI"
            else:
                labs = {
                    str(e.get("label", "")).upper()
                    for sets in ev.values()
                    for e in (sets if isinstance(sets, Sequence) else ())
                    if isinstance(e, Mapping)
                }
                gold_lab = "CONTRADICT" if "CONTRADICT" in labs else "SUPPORT"
            y_true_all.append(gold_lab)

            # Baseline open-corpus doc recall comparisons (BM25 vs Dense-SVD vs Hybrid RRF)
            b_ranked_doc_pos = self.doc_bm25.rank_candidates(claim_str, top_k=doc_top_m)
            if ev:
                gold_docs = {str(did) for did in ev.keys()}
                b_top_docs = {str(self.doc_ids[i]) for i in b_ranked_doc_pos}
                s_top_docs = {
                    str(self.doc_ids[int(i)])
                    for i in np.argsort(-self.doc_svd.cosine_scores(claim_str), kind="stable")[:doc_top_m]
                }
                bm25_doc_hits.append(len(gold_docs & b_top_docs) / len(gold_docs))
                svd_doc_hits.append(len(gold_docs & s_top_docs) / len(gold_docs))

            # Build official SciFact predictions for BM25 baseline
            b_doc_map: dict[str, dict[str, Any]] = {}
            for d_pos in b_ranked_doc_pos:
                did = self.doc_ids[d_pos]
                lo, hi = self.doc_sent_spans[did]
                if lo >= hi:
                    continue
                gids_d = list(range(lo, hi))
                s_norm = self.sent_bm25.score_indexed_normalized(claim_str, gids_d)
                if float(np.max(s_norm)) < 0.35:
                    continue
                s_ord_d = [int(i) for i in np.argsort(-s_norm, kind="stable") if float(s_norm[int(i)]) >= 0.30][:3]
                if not s_ord_d:
                    s_ord_d = [int(np.argmax(s_norm))]
                pol_d = classify_rationale_polarity(claim_str, self.all_sentences[lo + s_ord_d[0]])
                if pol_d == "NEUTRAL":
                    continue
                b_doc_map[str(did)] = {
                    "label": "CONTRADICT" if pol_d == "CONTRADICT" else "SUPPORT",
                    "sentences": s_ord_d[:3],
                }
            bm25_official_preds[cid_str] = b_doc_map

            # Full open-corpus 2-stage Hybrid + Reranker + Split Conformal verification
            res = self.verify_claim_open_corpus(
                claim_str, doc_top_m=doc_top_m, sent_top_k=max(top_k, 5)
            )
            st = res["stance"]
            y_pred_hybrid.append(str(st["predicted_label"]))
            pset = list(st["conformal_set"])
            is_cov = gold_lab in pset
            conformal_covered.append(is_cov)
            conformal_by_class[gold_lab].append(is_cov)
            conformal_sizes.append(len(pset))
            if len(pset) >= 2:
                multi_label_sets += 1

            # Build official SciFact predictions for Sparse Hybrid verifier across top-3 retrieved docs
            h_doc_map: dict[str, dict[str, Any]] = {}
            for r_rank, did in enumerate(res["top_doc_ids"][:3]):
                lo, hi = self.doc_sent_spans[did]
                if lo >= hi:
                    continue
                d_sents = self.all_sentences[lo:hi]
                d_title = self.all_sent_titles[lo]
                c_pred_d = self.verify_candidate_sentences(claim_str, d_sents, title=d_title, top_k=3)
                if not c_pred_d.top_rationale_scores:
                    continue
                p_sup_d = float(c_pred_d.probabilities["SUPPORT"])
                p_con_d = float(c_pred_d.probabilities["CONTRADICT"])
                non_nei_d = p_sup_d + p_con_d
                doc_conf_d = (
                    0.55 * non_nei_d
                    + 0.45 * float(c_pred_d.top_rationale_scores[0])
                    - 0.04 * float(r_rank)
                )
                if doc_conf_d < self.official_doc_non_nei_threshold:
                    continue
                sel_s = [
                    idx
                    for idx, sc in zip(c_pred_d.top_rationale_indices, c_pred_d.top_rationale_scores)
                    if sc >= 0.48
                ][:2]
                if not sel_s:
                    sel_s = [c_pred_d.top_rationale_indices[0]]
                h_doc_map[str(did)] = {
                    "label": "CONTRADICT" if p_con_d > p_sup_d else "SUPPORT",
                    "sentences": sel_s,
                }
            hybrid_official_preds[cid_str] = h_doc_map

            # Lexical BM25 and Dense-SVD 3-way baselines on open-corpus retrieval
            b_norm_max = float(np.max(self.sent_bm25.score_indexed_normalized(claim_str)))
            if b_norm_max < 0.35:
                y_pred_bm25_3way.append("NEI")
            else:
                b_best_gid = self.sent_bm25.rank_candidates(claim_str, top_k=1)[0]
                b_pol = classify_rationale_polarity(claim_str, self.all_sentences[b_best_gid])
                y_pred_bm25_3way.append("CONTRADICT" if b_pol == "CONTRADICT" else "SUPPORT")

            s_cos_all = self.sent_svd.cosine_scores(claim_str)
            s_best_gid = int(np.argmax(s_cos_all))
            if float(s_cos_all[s_best_gid]) < 0.42:
                y_pred_svd_3way.append("NEI")
            else:
                s_pol = classify_rationale_polarity(claim_str, self.all_sentences[s_best_gid])
                y_pred_svd_3way.append("CONTRADICT" if s_pol == "CONTRADICT" else "SUPPORT")

            top_sents = res["top_sentences"]
            max_rr = float(top_sents[0]["rerank_score"]) if top_sents else 0.0
            if ev:
                ev_max_rerank.append(max_rr)
                gold_docs = {str(did) for did in ev.keys()}
                pred_docs = {str(did) for did in res["top_doc_ids"]}
                open_doc_hits.append(len(gold_docs & pred_docs) / len(gold_docs))

                gold_sent_pairs = {
                    (str(did), int(si))
                    for did, sets in ev.items()
                    for s in sets
                    for si in s.get("sentences", ())
                }
                pred_pairs = [(str(it["doc_id"]), int(it["sent_idx"])) for it in top_sents]
                open_sent_h1.append(float(bool(set(pred_pairs[:1]) & gold_sent_pairs)))
                open_sent_h3.append(float(bool(set(pred_pairs[:top_k]) & gold_sent_pairs)))
                rr_val = 0.0
                for rk, sp in enumerate(pred_pairs, start=1):
                    if sp in gold_sent_pairs:
                        rr_val = 1.0 / float(rk)
                        break
                open_sent_mrr.append(rr_val)
            else:
                ni_max_rerank.append(max_rr)

        def _pack_rank_metrics(ranks: Sequence[Sequence[int]]) -> dict[str, float]:
            return {
                "hit@1": hit_at(ranks, pairs, 1),
                f"hit@{top_k}": hit_at(ranks, pairs, top_k),
                "mrr": mean_reciprocal_rank(ranks, pairs),
                f"sentence_recall@{top_k}": sentence_recall_at(ranks, pairs, top_k),
                f"rationale_set_recall@{top_k}": rationale_set_recall_at(ranks, pairs, top_k),
            }

        maj_3way = majority_class_stance_baseline(y_true_all, eval_labels=STANCE_LABELS)
        single_label_sets = sum(1 for sz in conformal_sizes if sz == 1)
        open_cov_detail = {
            lab: {
                "n": len(vals),
                "coverage": float(np.mean(vals)) if vals else float("nan"),
                "wilson_95_ci": list(wilson_score_interval(int(sum(vals)), len(vals))),
            }
            for lab, vals in conformal_by_class.items()
        }
        cand_cov_detail = {
            lab: {
                "n": len(vals),
                "coverage": float(np.mean(vals)) if vals else float("nan"),
                "wilson_95_ci": list(wilson_score_interval(int(sum(vals)), len(vals))),
            }
            for lab, vals in cand_conformal_by_class.items()
        }

        return {
            "total_claims": len(claims),
            "evidence_claims": len(ev_claims),
            "nei_claims": len(claims) - len(ev_claims),
            "claim_doc_pairs": len(pairs),
            "noinfo_cited_pairs": len(noinfo_pairs),
            "oracle_rationale_ranking": {
                "random_20_seeds": {
                    "hit@1": float(np.mean(rand_h1)) if rand_h1 else float("nan"),
                    f"hit@{top_k}": float(np.mean(rand_h3)) if rand_h3 else float("nan"),
                    "mrr": float(np.mean(rand_mrr)) if rand_mrr else float("nan"),
                },
                "bm25": _pack_rank_metrics(bm25_ranks),
                "dense_svd": _pack_rank_metrics(svd_ranks),
                "stage1_hybrid_rrf": _pack_rank_metrics(rrf_ranks),
                "hybrid_reranker": _pack_rank_metrics(hybrid_ranks),
            },
            "open_corpus_retrieval": {
                f"bm25_doc_recall@{doc_top_m}": float(np.mean(bm25_doc_hits)) if bm25_doc_hits else float("nan"),
                f"dense_svd_doc_recall@{doc_top_m}": float(np.mean(svd_doc_hits)) if svd_doc_hits else float("nan"),
                f"hybrid_rrf_doc_recall@{doc_top_m}": float(np.mean(open_doc_hits)) if open_doc_hits else float("nan"),
                "hybrid_reranker_sentence_hit@1": float(np.mean(open_sent_h1)) if open_sent_h1 else float("nan"),
                f"hybrid_reranker_sentence_hit@{top_k}": float(np.mean(open_sent_h3)) if open_sent_h3 else float("nan"),
                "hybrid_reranker_sentence_mrr": float(np.mean(open_sent_mrr)) if open_sent_mrr else float("nan"),
                "reranker_auroc_vs_nei": compute_auroc(ev_max_rerank, ni_max_rerank),
            },
            "official_scifact_metrics": {
                "bm25": compute_official_scifact_metrics(claims, bm25_official_preds, max_sentences_per_doc=3),
                "hybrid_reranker": compute_official_scifact_metrics(
                    claims, hybrid_official_preds, max_sentences_per_doc=3
                ),
            },
            "stance_verification_3way": {
                **maj_3way,
                "bm25_macro_f1": macro_f1_score(y_true_all, y_pred_bm25_3way, labels=STANCE_LABELS),
                "bm25_accuracy": (
                    float(np.mean([yt == yp for yt, yp in zip(y_true_all, y_pred_bm25_3way)]))
                    if y_true_all
                    else float("nan")
                ),
                "dense_svd_macro_f1": macro_f1_score(y_true_all, y_pred_svd_3way, labels=STANCE_LABELS),
                "hybrid_reranker_macro_f1": macro_f1_score(y_true_all, y_pred_hybrid, labels=STANCE_LABELS),
                "hybrid_reranker_accuracy": float(
                    np.mean([yt == yp for yt, yp in zip(y_true_all, y_pred_hybrid)])
                )
                if y_true_all
                else float("nan"),
                "oracle_bm25_binary_macro_f1": macro_f1_score(
                    bm25_stance_true, bm25_stance_pred, labels=("SUPPORT", "CONTRADICT")
                ),
                "oracle_svd_binary_macro_f1": macro_f1_score(
                    bm25_stance_true, svd_stance_pred, labels=("SUPPORT", "CONTRADICT")
                ),
            },
            "split_conformal_90": {
                "target_coverage": 1.0 - self.stance_classifier.alpha,
                "alpha": self.stance_classifier.alpha,
                "empirical_coverage": float(np.mean(conformal_covered)) if conformal_covered else float("nan"),
                "empirical_wilson_95_ci": list(
                    wilson_score_interval(int(sum(conformal_covered)), len(conformal_covered))
                ),
                "avg_set_size": float(np.mean(conformal_sizes)) if conformal_sizes else float("nan"),
                "single_label_set_rate": (
                    float(single_label_sets) / float(len(claims)) if claims else 0.0
                ),
                "multi_label_set_rate": (
                    float(multi_label_sets) / float(len(claims)) if claims else 0.0
                ),
                "coverage_by_class": {
                    lab: (float(np.mean(vals)) if vals else float("nan"))
                    for lab, vals in conformal_by_class.items()
                },
                "coverage_by_class_detail": open_cov_detail,
                "candidate_abstract_conformal_90": {
                    "empirical_coverage": (
                        float(np.mean(cand_conformal_covered))
                        if cand_conformal_covered
                        else float("nan")
                    ),
                    "avg_set_size": (
                        float(np.mean(cand_conformal_sizes))
                        if cand_conformal_sizes
                        else float("nan")
                    ),
                    "single_label_set_rate": (
                        float(sum(1 for sz in cand_conformal_sizes if sz == 1))
                        / float(len(cand_conformal_sizes))
                        if cand_conformal_sizes
                        else 0.0
                    ),
                    "coverage_by_class": {
                        lab: (float(np.mean(vals)) if vals else float("nan"))
                        for lab, vals in cand_conformal_by_class.items()
                    },
                    "coverage_by_class_detail": cand_cov_detail,
                },
            },
            "training_provenance": dict(self.training_provenance),
        }
