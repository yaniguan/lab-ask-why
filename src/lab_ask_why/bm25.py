"""Biomedical lexical tokenizer and indexed Okapi BM25 sentence/document ranker."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Negation tokens ('not', 'no', 'neither', 'without', 'never', 'none') are
# deliberately excluded from STOP so opposing scientific stances retain negation cues.
STOP = frozenset(
    "a an the of in on to for and or is are was were be by with as at from "
    "that this it its which than into".split()
)

GREEK_MAP: dict[str, str] = {
    "\u03b1": "alpha",   # α
    "\u03b2": "beta",    # β
    "\u03b3": "gamma",   # γ
    "\u03b4": "delta",   # δ
    "\u03b5": "epsilon", # ε
    "\u03b6": "zeta",    # ζ
    "\u03b7": "eta",     # η
    "\u03b8": "theta",   # θ
    "\u03ba": "kappa",   # κ
    "\u03bb": "lambda",  # λ
    "\u03bc": "micro",   # μ (Greek small mu)
    "\u00b5": "micro",   # µ (Unicode micro sign)
    "\u03bd": "nu",      # ν
    "\u03be": "xi",      # ξ
    "\u03c0": "pi",      # π
    "\u03c1": "rho",     # ρ
    "\u03c3": "sigma",   # σ
    "\u03c4": "tau",     # τ
    "\u03c6": "phi",     # φ
    "\u03c7": "chi",     # χ
    "\u03c8": "psi",     # ψ
    "\u03c9": "omega",   # ω
}

_GREEK_RE = re.compile("|".join(re.escape(k) for k in GREEK_MAP))
_COMMA_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PAREN_PLUS_RE = re.compile(r"\s*\(\+\)")
_PAREN_MINUS_RE = re.compile(r"\s*\(-\)")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z0-9]+(?:\+|-(?=\s|$))?")


def normalize_biomedical_text(text: str) -> str:
    """Normalize Greek biomedical symbols, allele (+/-) markers, and numeric commas."""
    lowered = text.lower()

    def _replace_greek(match: re.Match[str]) -> str:
        word = GREEK_MAP[match.group(0)]
        start, end = match.span()
        prefix = " " if start > 0 and lowered[start - 1].isalnum() else ""
        suffix = " " if end < len(lowered) and lowered[end].isalnum() else ""
        return f"{prefix}{word}{suffix}"

    norm = _GREEK_RE.sub(_replace_greek, lowered)
    norm = _PAREN_PLUS_RE.sub("+", norm)
    norm = _PAREN_MINUS_RE.sub("-", norm)
    while True:
        updated = _COMMA_THOUSANDS_RE.sub("", norm)
        if updated == norm:
            break
        norm = updated
    return norm


def tokenize(text: str) -> list[str]:
    """Tokenize biomedical prose preserving Greek names, decimals, +/- alleles, and negations."""
    if not text:
        return []
    norm = normalize_biomedical_text(text)
    return [tok for tok in _TOKEN_RE.findall(norm) if tok not in STOP]


class BM25:
    """Indexed Okapi BM25 with query-TF saturation, zero-length guards, and Laplace OOV IDF."""

    def __init__(
        self,
        sentences: Sequence[str],
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1.2,
    ) -> None:
        if not sentences:
            raise ValueError("BM25 requires at least one sentence")
        if k1 < 0 or not (0.0 <= b <= 1.0) or k3 < 0:
            raise ValueError(f"Invalid BM25 hyperparameters: k1={k1}, b={b}, k3={k3}")

        self.k1 = float(k1)
        self.b = float(b)
        self.k3 = float(k3)
        self.num_docs = len(sentences)

        df: Counter[str] = Counter()
        lengths = np.empty(self.num_docs, dtype=np.float64)
        raw_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for idx, sent in enumerate(sentences):
            toks = tokenize(sent)
            lengths[idx] = float(len(toks))
            tf = Counter(toks)
            for term, count in tf.items():
                df[term] += 1
                raw_postings[term].append((idx, float(count)))

        # Clamp avgdl >= 1.0 so stopword-only / blank sentence lists never divide by zero.
        mean_len = float(np.mean(lengths)) if self.num_docs > 0 else 1.0
        self.avgdl = max(mean_len, 1.0)
        self.doc_len = lengths
        self.norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / self.avgdl))

        n = float(self.num_docs)
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (n - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }
        # Smooth Laplace IDF for OOV query terms (d = 0) when BM25 is fit on background docs.
        self.default_idf: float = math.log(1.0 + (n + 0.5) / 0.5)

        # Compact inverted index for fast full-corpus and subset retrieval without re-tokenizing.
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            term: (
                np.array([i for i, _ in pairs], dtype=np.int32),
                np.array([c for _, c in pairs], dtype=np.float64),
            )
            for term, pairs in raw_postings.items()
        }

    @classmethod
    def from_corpus(
        cls,
        corpus: Mapping[Any, Mapping[str, Any]],
        *,
        exclude_doc_ids: Iterable[Any] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1.2,
    ) -> tuple["BM25", dict[Any, tuple[int, int]]]:
        """Fit BM25 over corpus sentences, optionally excluding evaluation/test documents.

        Returns `(bm25, doc_spans)` mapping each included `doc_id` to `(start_idx, end_idx)`
        in the indexed sentence array.
        """
        excluded = {str(did) for did in (exclude_doc_ids or ())}
        sentences: list[str] = []
        doc_spans: dict[Any, tuple[int, int]] = {}
        for doc_id, doc in corpus.items():
            if str(doc_id) in excluded:
                continue
            abs_sents = list(doc.get("abstract", ()))
            start = len(sentences)
            sentences.extend(abs_sents)
            doc_spans[doc_id] = (start, len(sentences))
        if not sentences:
            raise ValueError("No background sentences remained after applying exclude_doc_ids")
        return cls(sentences, k1=k1, b=b, k3=k3), doc_spans

    def _query_weights(self, query: str) -> dict[str, float]:
        q_tf = Counter(tokenize(query))
        if not q_tf:
            return {}
        if self.k3 == 0.0:
            return {term: 1.0 for term in q_tf}
        return {
            term: ((self.k3 + 1.0) * count) / (self.k3 + count)
            for term, count in q_tf.items()
        }

    def score_indexed(
        self,
        query: str,
        sent_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Score pre-indexed sentences via the inverted posting index."""
        q_weights = self._query_weights(query)
        if sent_ids is None:
            scores = np.zeros(self.num_docs, dtype=np.float64)
            for term, qw in q_weights.items():
                posting = self.postings.get(term)
                if posting is None:
                    continue
                doc_idx, tf = posting
                idf = self.idf.get(term, self.default_idf)
                scores[doc_idx] += (
                    qw * idf * tf * (self.k1 + 1.0) / (tf + self.norm[doc_idx])
                )
            return scores

        idx_arr = np.asarray(sent_ids, dtype=np.int32)
        if idx_arr.size == 0 or not q_weights:
            return np.zeros(idx_arr.size, dtype=np.float64)

        # Fast path for contiguous slices (e.g., sentences belonging to a single abstract)
        is_contiguous = (
            idx_arr.size == 1
            or bool(np.all(np.diff(idx_arr) == 1))
        )
        start = int(idx_arr[0])
        end = start + int(idx_arr.size)

        out = np.zeros(idx_arr.size, dtype=np.float64)
        for term, qw in q_weights.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_idx, tf = posting
            idf = self.idf.get(term, self.default_idf)
            if is_contiguous:
                lo = int(np.searchsorted(doc_idx, start, side="left"))
                hi = int(np.searchsorted(doc_idx, end, side="left"))
                if lo < hi:
                    sub_docs = doc_idx[lo:hi]
                    sub_tf = tf[lo:hi]
                    out[sub_docs - start] += (
                        qw * idf * sub_tf * (self.k1 + 1.0) / (sub_tf + self.norm[sub_docs])
                    )
            else:
                pos = np.searchsorted(doc_idx, idx_arr)
                valid = (pos < doc_idx.size) & (doc_idx[np.minimum(pos, doc_idx.size - 1)] == idx_arr)
                if np.any(valid):
                    matched_pos = pos[valid]
                    sub_tf = tf[matched_pos]
                    sub_docs = idx_arr[valid]
                    out[valid] += (
                        qw * idf * sub_tf * (self.k1 + 1.0) / (sub_tf + self.norm[sub_docs])
                    )
        return out

    def scores(self, query: str, sentences: Sequence[str]) -> np.ndarray:
        """Score arbitrary candidate sentences (supports unindexed or background-excluded docs)."""
        n_cand = len(sentences)
        if n_cand == 0:
            return np.zeros(0, dtype=np.float64)
        q_weights = self._query_weights(query)
        if not q_weights:
            return np.zeros(n_cand, dtype=np.float64)

        out = np.zeros(n_cand, dtype=np.float64)
        for i, sent in enumerate(sentences):
            toks = tokenize(sent)
            if not toks:
                continue
            tf = Counter(toks)
            norm = self.k1 * (1.0 - self.b + self.b * (len(toks) / self.avgdl))
            score = 0.0
            for term, qw in q_weights.items():
                t_count = tf.get(term, 0)
                if t_count:
                    idf = self.idf.get(term, self.default_idf)
                    score += qw * idf * t_count * (self.k1 + 1.0) / (t_count + norm)
            out[i] = score
        return out

    def rank_candidates(
        self,
        query: str,
        sent_ids: Sequence[int] | None = None,
        *,
        top_k: int | None = None,
        tie_seed: int | None = None,
    ) -> list[int]:
        """Rank pre-indexed sentence IDs in descending BM25 order."""
        scores = self.score_indexed(query, sent_ids)
        if scores.size == 0:
            return []
        if tie_seed is None:
            order = np.argsort(-scores, kind="stable")
        else:
            rng = np.random.default_rng(tie_seed)
            jitter = rng.random(scores.size)
            order = np.lexsort((jitter, -scores))
        if top_k is not None:
            order = order[: max(0, int(top_k))]
        if sent_ids is None:
            return [int(i) for i in order]
        return [int(sent_ids[int(i)]) for i in order]
