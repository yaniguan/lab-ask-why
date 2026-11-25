"""Two-stage hybrid open-corpus retrieval and symmetric pro/con rationale probing."""
from __future__ import annotations

import functools
import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from lab_ask_why.bm25 import BM25, STOP, normalize_biomedical_text, tokenize
from lab_ask_why.metrics import compute_auroc
from lab_ask_why.schema import (
    POLARITY_VALUES,
    DiscriminatingObservable,
    EpistemicProbeResult,
    RationaleEvidence,
    to_permille_floor,
)
from lab_ask_why.socratic import (
    extract_differential_terms,
    extract_protocol_descriptors,
    synthesize_control_wells,
    why_questions,
)

_NEGATION_TOKENS = frozenset(
    {
        "not",
        "no",
        "neither",
        "nor",
        "without",
        "never",
        "none",
        "cannot",
        "lack",
        "lacks",
        "lacking",
        "lacked",
        "failed",
        "fails",
        "fail",
        "unable",
    }
)

_ANTONYM_GROUPS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset(
            {
                "increase",
                "increases",
                "increased",
                "raises",
                "raise",
                "elevates",
                "higher",
                "enhances",
                "promotes",
                "activates",
                "activate",
                "induces",
                "accelerates",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreases",
                "decreased",
                "reduces",
                "reduce",
                "reduced",
                "lower",
                "inhibits",
                "inhibit",
                "suppresses",
                "abolishes",
                "prevents",
                "prevent",
                "blocks",
            }
        ),
    ),
    (
        frozenset(
            {
                "vulnerability",
                "susceptibility",
                "risk",
                "harm",
                "toxicity",
                "severe",
                "mortality",
                "lethal",
            }
        ),
        frozenset(
            {
                "protection",
                "protects",
                "protective",
                "resistance",
                "survival",
                "benefit",
                "safe",
                "tolerated",
            }
        ),
    ),
    (
        frozenset({"sufficient", "necessary", "essential", "required", "causes"}),
        frozenset({"dispensable", "unnecessary", "independent", "unrelated", "benign"}),
    ),
)

# Curated biomedical & wet-lab concept synonym equivalence classes for zero-weight paraphrase recall.
_BIOMED_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("aspirin", "acetylsalicylic", "salicylate", "salicylic", "asa"),
    (
        "hydrolysis",
        "hydrolyze",
        "hydrolyzed",
        "hydrolyzes",
        "hydrolyse",
        "hydrolysed",
        "deacetylation",
        "deacetylated",
        "ester",
        "esterolysis",
    ),
    ("aqueous", "water", "hydrated"),
    ("thermal", "heat", "heated", "heating", "incubation", "incubate", "incubated", "temperature"),
    ("solubility", "soluble", "dissolution", "dissolve", "dissolves", "dissolved"),
    ("buffer", "buffered", "unbuffered", "stabilizer", "stabilizers"),
    ("erythrocyte", "erythrocytes", "rbc", "microerythrocyte", "microcytosis"),
    ("hemoglobin", "haemoglobin", "hb"),
    ("anemia", "anaemia"),
    ("thalassemia", "thalassaemia"),
    ("leukemia", "leukaemia"),
    ("ischemia", "ischaemia", "ischemic", "ischaemic"),
    ("tumor", "tumors", "tumour", "tumours", "neoplasm", "carcinoma", "malignancy"),
    ("apoptosis", "apoptotic"),
    ("renal", "kidney", "nephropathy"),
    ("hepatic", "liver", "hepatocyte", "hepatocytes"),
    ("cardiac", "myocardial", "heart"),
    ("pulmonary", "lung"),
    ("cerebral", "brain", "neuronal"),
    ("pediatric", "paediatric", "childhood", "children"),
    ("fetal", "foetal"),
    ("estrogen", "oestrogen"),
    ("esophageal", "oesophageal"),
)

_TOKEN_TO_CONCEPTS: dict[str, tuple[str, ...]] = defaultdict(tuple)  # type: ignore[arg-type]
_concept_builder: dict[str, list[str]] = defaultdict(list)
for _gid, _group in enumerate(_BIOMED_SYNONYM_GROUPS):
    _cid = f"syn_{_gid}"
    for _word in _group:
        _concept_builder[_word].append(_cid)
_TOKEN_TO_CONCEPTS = {k: tuple(v) for k, v in _concept_builder.items()}

_UK_US_SUBS: tuple[tuple[str, str], ...] = (
    ("haem", "hem"),
    ("anaem", "anem"),
    ("thalassaem", "thalassem"),
    ("leukaem", "leukem"),
    ("ischaem", "ischem"),
    ("oesophag", "esophag"),
    ("oestrog", "estrog"),
    ("oedema", "edema"),
    ("foet", "fet"),
    ("paediatr", "pediatr"),
    ("tumour", "tumor"),
    ("signalling", "signaling"),
    ("modelling", "modeling"),
    ("labelled", "labeled"),
    ("labelling", "labeling"),
    ("randomised", "randomized"),
    ("characterised", "characterized"),
    ("catalysed", "catalyzed"),
    ("analysed", "analyzed"),
    ("immunisation", "immunization"),
    ("organisation", "organization"),
    ("ageing", "aging"),
    ("behaviour", "behavior"),
    ("colour", "color"),
    ("favour", "favor"),
)

_LEADING_CONCESSIVE_RE = re.compile(
    r"^\s*(?:although|even\s+though|though|whereas|while|despite|in\s+spite\s+of)\b[^,;]*,\s*(.+)$",
    re.IGNORECASE,
)
_TRAILING_CONCESSIVE_RE = re.compile(
    r"^(.+?),\s*(?:although|even\s+though|though|despite|in\s+spite\s+of)\b.+$",
    re.IGNORECASE,
)
_ADVERSATIVE_SPLIT_RE = re.compile(
    r"^.+?(?:[,;]\s*(?:but|yet|however|nevertheless|nonetheless)\b\s*,?|\s+\b(?:but|yet)\b)\s*(.+)$",
    re.IGNORECASE,
)
_SCOPED_WITHOUT_RE = re.compile(
    r"\b(?:without|with\s+no)\b([^,.;:]*)",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=65536)
def _normalize_orthography_and_stem(tok: str) -> str:
    """Deterministic US/UK spelling canonicalization and light biomedical suffix stemming."""
    w = tok
    for uk, us in _UK_US_SUBS:
        if uk in w:
            w = w.replace(uk, us)
    if len(w) <= 4 or w[-1] in "+-" or any(ch.isdigit() for ch in w):
        return w
    for suffix in ("ations", "ation", "itions", "ition", "ingly", "edly", "ments", "ment", "ities", "ity", "ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


@functools.lru_cache(maxsize=65536)
def _char_ngrams(tok: str, n: int = 3) -> frozenset[str]:
    if len(tok) < 4:
        return frozenset({tok}) if tok else frozenset()
    padded = f"<{tok}>"
    return frozenset(padded[i : i + n] for i in range(len(padded) - n + 1))


@functools.lru_cache(maxsize=65536)
def _token_subword_bundle(tok: str) -> tuple[tuple[str, int], ...]:
    """Cached per-token stem, biomedical concept, and affix subword feature bundle."""
    if tok in _NEGATION_TOKENS:
        return ()
    items: Counter[str] = Counter()
    stem = _normalize_orthography_and_stem(tok)
    items[f"st:{stem}"] += 2
    for cid in _TOKEN_TO_CONCEPTS.get(tok, ()):
        items[f"sy:{cid}"] += 3
    for cid in _TOKEN_TO_CONCEPTS.get(stem, ()):
        items[f"sy:{cid}"] += 3
    if len(stem) >= 5 and not stem[0].isdigit():
        items[f"pf:{stem[:5]}"] += 1
        items[f"sf:{stem[-4:]}"] += 1
    return tuple(items.items())


@functools.lru_cache(maxsize=131072)
def _token_similarity(tok_a: str, tok_b: str) -> float:
    """Deterministic lexical + orthographic/stem + synonym + subword similarity in [0, 1]."""
    if tok_a == tok_b:
        return 1.0
    # Never conflate opposite +/- allele markers on the same gene/surface marker
    if (tok_a.endswith("+") and tok_b.endswith("-")) or (tok_a.endswith("-") and tok_b.endswith("+")):
        if tok_a[:-1] == tok_b[:-1]:
            return 0.0
    stem_a = _normalize_orthography_and_stem(tok_a)
    stem_b = _normalize_orthography_and_stem(tok_b)
    if stem_a == stem_b:
        return 0.94
    syn_a = set(_TOKEN_TO_CONCEPTS.get(tok_a, ())) | set(_TOKEN_TO_CONCEPTS.get(stem_a, ()))
    syn_b = set(_TOKEN_TO_CONCEPTS.get(tok_b, ())) | set(_TOKEN_TO_CONCEPTS.get(stem_b, ()))
    if syn_a and (syn_a & syn_b):
        return 0.90
    if len(stem_a) < 4 or len(stem_b) < 4 or stem_a[0].isdigit() or stem_b[0].isdigit():
        return 0.0
    grams_a = _char_ngrams(stem_a, 3)
    grams_b = _char_ngrams(stem_b, 3)
    if not grams_a or not grams_b:
        return 0.0
    dice = (2.0 * len(grams_a & grams_b)) / float(len(grams_a) + len(grams_b))
    return round(0.80 * dice, 6) if dice >= 0.55 else 0.0


def _extract_subword_concept_features(text: str) -> Counter[str]:
    """Extract deterministic stem, biomedical concept, and char-4gram TF counts."""
    feats: Counter[str] = Counter()
    for tok in tokenize(text):
        for k, v in _token_subword_bundle(tok):
            feats[k] += v
    return feats


class _SubwordConceptCosineIndex:
    """Zero-dependency L2-normalized TF-IDF subword + biomedical concept cosine index."""

    def __init__(
        self,
        texts: Sequence[str],
        *,
        idf_mask: Sequence[bool] | None = None,
    ) -> None:
        self.num_docs = len(texts)
        doc_feats: list[Counter[str]] = [_extract_subword_concept_features(t) for t in texts]

        df: Counter[str] = Counter()
        if idf_mask is not None and any(idf_mask):
            n_bg = 0
            for fc, keep in zip(doc_feats, idf_mask):
                if keep:
                    n_bg += 1
                    for term in fc.keys():
                        df[term] += 1
            n_idf = float(max(1, n_bg))
        else:
            for fc in doc_feats:
                for term in fc.keys():
                    df[term] += 1
            n_idf = float(max(1, self.num_docs))

        self.idf: dict[str, float] = {
            term: math.log(1.0 + (n_idf - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }
        if n_idf < 10.0 and self.idf:
            self.default_idf = max(self.idf.values())
        else:
            self.default_idf = math.log(1.0 + (n_idf + 0.5) / 0.5)

        raw_idx: dict[str, list[int]] = defaultdict(list)
        raw_val: dict[str, list[float]] = defaultdict(list)
        idf_get = self.idf.get
        def_idf = self.default_idf

        for doc_idx, fc in enumerate(doc_feats):
            if not fc:
                continue
            weights: list[tuple[str, float]] = []
            sq_norm = 0.0
            for term, count in fc.items():
                w = (1.0 + math.log(float(count))) * idf_get(term, def_idf)
                weights.append((term, w))
                sq_norm += w * w
            if sq_norm <= 1e-15:
                continue
            inv_norm = 1.0 / math.sqrt(sq_norm)
            for term, w in weights:
                raw_idx[term].append(doc_idx)
                raw_val[term].append(w * inv_norm)

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            term: (
                np.asarray(idxs, dtype=np.int32),
                np.asarray(raw_val[term], dtype=np.float64),
            )
            for term, idxs in raw_idx.items()
        }

    def cosine_scores(
        self,
        query: str,
        cand_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        n_target = self.num_docs if cand_ids is None else len(cand_ids)
        if n_target == 0:
            return np.zeros(0, dtype=np.float64)
        q_fc = _extract_subword_concept_features(query)
        if not q_fc:
            return np.zeros(n_target, dtype=np.float64)

        q_weights: dict[str, float] = {}
        sq_norm = 0.0
        for term, count in q_fc.items():
            w = (1.0 + math.log(float(count))) * self.idf.get(term, self.default_idf)
            q_weights[term] = w
            sq_norm += w * w
        if sq_norm <= 1e-15:
            return np.zeros(n_target, dtype=np.float64)
        inv_norm = 1.0 / math.sqrt(sq_norm)

        if cand_ids is None:
            scores = np.zeros(self.num_docs, dtype=np.float64)
            for term, qw in q_weights.items():
                posting = self.postings.get(term)
                if posting is None:
                    continue
                doc_idx, d_val = posting
                scores[doc_idx] += (qw * inv_norm) * d_val
            return np.clip(scores, 0.0, 1.0)

        idx_arr = np.asarray(cand_ids, dtype=np.int32)
        is_contiguous = idx_arr.size == 1 or bool(np.all(np.diff(idx_arr) == 1))
        start = int(idx_arr[0])
        end = start + int(idx_arr.size)
        out = np.zeros(idx_arr.size, dtype=np.float64)

        for term, qw in q_weights.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_idx, d_val = posting
            q_unit = qw * inv_norm
            if is_contiguous:
                lo = int(np.searchsorted(doc_idx, start, side="left"))
                hi = int(np.searchsorted(doc_idx, end, side="left"))
                if lo < hi:
                    out[doc_idx[lo:hi] - start] += q_unit * d_val[lo:hi]
            else:
                pos = np.searchsorted(doc_idx, idx_arr)
                valid = (pos < doc_idx.size) & (doc_idx[np.minimum(pos, doc_idx.size - 1)] == idx_arr)
                if np.any(valid):
                    out[valid] += q_unit * d_val[pos[valid]]
        return np.clip(out, 0.0, 1.0)


def _reciprocal_rank_fusion(
    score_arrays: Sequence[np.ndarray],
    *,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Fuse multiple score arrays via Reciprocal Rank Fusion, ignoring non-positive entries."""
    if not score_arrays:
        return np.zeros(0, dtype=np.float64)
    n = score_arrays[0].size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w_list = [1.0] * len(score_arrays) if weights is None else [float(w) for w in weights]
    fused = np.zeros(n, dtype=np.float64)
    for arr, weight in zip(score_arrays, w_list):
        if weight <= 0.0:
            continue
        order = np.argsort(-arr, kind="stable")
        for rank_idx, item_idx in enumerate(order):
            idx_int = int(item_idx)
            if float(arr[idx_int]) <= 0.0:
                break
            fused[idx_int] += weight / float(rrf_k + rank_idx + 1)
    return fused


def _extract_main_assertion_clause(text: str) -> str:
    """Isolate the matrix assertion clause from concessive/contrastive subordinate clauses."""
    cleaned = text.strip()
    m_lead = _LEADING_CONCESSIVE_RE.match(cleaned)
    if m_lead:
        tail = m_lead.group(1).strip()
        if len(tokenize(tail)) >= 2:
            cleaned = tail
    else:
        m_trail = _TRAILING_CONCESSIVE_RE.match(cleaned)
        if m_trail:
            head = m_trail.group(1).strip()
            if len(tokenize(head)) >= 2:
                cleaned = head

    m_adv = _ADVERSATIVE_SPLIT_RE.match(cleaned)
    if m_adv:
        rhs = m_adv.group(1).strip()
        rhs_lower = rhs.lower()
        if not rhs_lower.startswith(("without ", "with no ")) and len(tokenize(rhs)) >= 2:
            cleaned = rhs

    return cleaned


def _strip_unreferenced_without_adjuncts(clause: str, reference_tokens: set[str]) -> str:
    """Strip prepositional 'without ...' / 'with no ...' adjuncts whose objects are absent from reference."""
    ref_stems = {_normalize_orthography_and_stem(t) for t in reference_tokens if t not in _NEGATION_TOKENS}
    ref_syns = {
        cid
        for t in reference_tokens
        for cid in (*_TOKEN_TO_CONCEPTS.get(t, ()), *_TOKEN_TO_CONCEPTS.get(_normalize_orthography_and_stem(t), ()))
    }

    def _replace_if_unreferenced(match: re.Match[str]) -> str:
        obj_toks = [t for t in tokenize(match.group(1)) if t not in _NEGATION_TOKENS]
        if not obj_toks:
            return match.group(0)
        obj_stems = {_normalize_orthography_and_stem(t) for t in obj_toks}
        obj_syns = {
            cid
            for t in obj_toks
            for cid in (*_TOKEN_TO_CONCEPTS.get(t, ()), *_TOKEN_TO_CONCEPTS.get(_normalize_orthography_and_stem(t), ()))
        }
        if (obj_stems & ref_stems) or (obj_syns & ref_syns):
            return match.group(0)
        return " "

    return _SCOPED_WITHOUT_RE.sub(_replace_if_unreferenced, clause)


def classify_rationale_polarity(claim: str, sentence: str) -> str:
    """Symmetric clause-scoped lexical polarity classifier ('SUPPORT', 'CONTRADICT', 'NEUTRAL')."""
    c_all = set(tokenize(claim))
    s_all = set(tokenize(sentence))
    if not c_all or not s_all:
        return "NEUTRAL"

    c_content = c_all - _NEGATION_TOKENS
    s_content = s_all - _NEGATION_TOKENS
    if not c_content or not s_content:
        return "NEUTRAL"

    # Check if there is any lexical, subword, antonym, or biomedical concept alignment at all
    has_alignment = bool(c_content & s_content)
    if not has_alignment:
        for pos_g, neg_g in _ANTONYM_GROUPS:
            if (c_content & (pos_g | neg_g)) and (s_content & (pos_g | neg_g)):
                has_alignment = True
                break
    if not has_alignment:
        for ct in c_content:
            if any(_token_similarity(ct, st) >= 0.55 for st in s_content):
                has_alignment = True
                break
    if not has_alignment:
        return "NEUTRAL"

    c_main = _extract_main_assertion_clause(claim)
    s_main = _extract_main_assertion_clause(sentence)

    c_scoped = _strip_unreferenced_without_adjuncts(c_main, s_all)
    s_scoped = _strip_unreferenced_without_adjuncts(s_main, c_all)

    c_toks = set(tokenize(c_scoped))
    s_toks = set(tokenize(s_scoped))
    if not s_toks:
        s_toks = s_all
    if not c_toks:
        c_toks = c_all

    c_neg = bool(c_toks & _NEGATION_TOKENS)
    s_neg = bool(s_toks & _NEGATION_TOKENS)
    if c_neg != s_neg:
        return "CONTRADICT"

    for pos_group, neg_group in _ANTONYM_GROUPS:
        if (c_toks & pos_group and s_toks & neg_group and not (s_toks & pos_group)) or (
            c_toks & neg_group and s_toks & pos_group and not (s_toks & neg_group)
        ):
            return "CONTRADICT"

    return "SUPPORT"


class TwoStageRetriever:
    """Open-corpus two-stage hybrid retriever (BM25 + dense/subword RRF + cross-interaction)."""

    def __init__(
        self,
        corpus: Mapping[Any, Mapping[str, Any]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1.2,
        abstention_threshold: float = 0.35,
        exclude_doc_ids: Iterable[Any] | None = None,
        embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
        rerank_fn: Callable[[str, Sequence[str]], Sequence[tuple[float, str]]] | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("TwoStageRetriever requires a non-empty corpus")
        self.corpus = corpus
        self.abstention_threshold = float(abstention_threshold)
        self._is_explicitly_calibrated = False
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn

        excluded = {str(did) for did in (exclude_doc_ids or ())}
        self.excluded_doc_ids = frozenset(excluded)

        self.doc_ids: list[Any] = []
        abstract_blobs: list[str] = []
        bg_abstract_blobs: list[str] = []
        doc_bg_mask: list[bool] = []
        all_sentences: list[str] = []
        bg_sentences: list[str] = []
        sent_bg_mask: list[bool] = []
        self.doc_sent_spans: dict[Any, tuple[int, int]] = {}
        self.sent_to_doc: list[tuple[Any, int]] = []
        self._doc_id_to_pos: dict[Any, int] = {}

        for doc_id, doc in corpus.items():
            pos = len(self.doc_ids)
            self._doc_id_to_pos[doc_id] = pos
            title = str(doc.get("title", ""))
            sents = [str(s) for s in doc.get("abstract", ())]
            self.doc_ids.append(doc_id)
            blob = (title + " " + " ".join(sents)).strip()
            abstract_blobs.append(blob)
            is_bg = str(doc_id) not in excluded
            doc_bg_mask.append(is_bg)
            if is_bg:
                bg_abstract_blobs.append(blob)
            start = len(all_sentences)
            for j, s in enumerate(sents):
                all_sentences.append(s)
                self.sent_to_doc.append((doc_id, j))
                sent_bg_mask.append(is_bg)
                if is_bg:
                    bg_sentences.append(s)
            self.doc_sent_spans[doc_id] = (start, len(all_sentences))

        use_bg_docs = bg_abstract_blobs if (excluded and bg_abstract_blobs) else None
        use_bg_sents = bg_sentences if (excluded and bg_sentences) else None
        use_doc_mask = doc_bg_mask if (excluded and bg_abstract_blobs) else None
        use_sent_mask = sent_bg_mask if (excluded and bg_sentences) else None

        self.abstract_blobs = abstract_blobs
        self.all_sentences = all_sentences
        self.doc_bm25 = BM25(abstract_blobs, k1=k1, b=b, k3=k3, idf_sentences=use_bg_docs)
        self.sent_bm25 = BM25(
            all_sentences if all_sentences else [""],
            k1=k1,
            b=b,
            k3=k3,
            idf_sentences=use_bg_sents,
        )
        self._doc_subword = _SubwordConceptCosineIndex(abstract_blobs, idf_mask=use_doc_mask)
        self._sent_subword = _SubwordConceptCosineIndex(
            all_sentences if all_sentences else [""],
            idf_mask=use_sent_mask,
        )

        self._doc_embeddings: np.ndarray | None = None
        self._sent_embeddings: np.ndarray | None = None
        if self.embed_fn is not None:
            self._doc_embeddings = self._l2_normalize_matrix(self.embed_fn(abstract_blobs))
            if all_sentences:
                self._sent_embeddings = self._l2_normalize_matrix(self.embed_fn(all_sentences))

    @staticmethod
    def _l2_normalize_matrix(mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        return arr / safe_norms

    def _external_cosine(
        self,
        query: str,
        target_matrix: np.ndarray | None,
        cand_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        if self.embed_fn is None or target_matrix is None or target_matrix.shape[0] == 0:
            n = 0 if target_matrix is None else (target_matrix.shape[0] if cand_ids is None else len(cand_ids))
            return np.zeros(n, dtype=np.float64)
        q_vec = self._l2_normalize_matrix(self.embed_fn([query]))[0]
        sub = target_matrix if cand_ids is None else target_matrix[np.asarray(cand_ids, dtype=np.int32)]
        return np.clip(sub @ q_vec, 0.0, 1.0)

    def _cross_interaction_scores(
        self,
        claim: str,
        cand_global_ids: Sequence[int],
        doc_context_map: Mapping[Any, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (context_aware_cross_score, strict_sentence_paraphrase_score) in [0, 1]."""
        n_cand = len(cand_global_ids)
        if n_cand == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        q_toks = [t for t in tokenize(claim) if t not in _NEGATION_TOKENS]
        if not q_toks:
            return np.zeros(n_cand, dtype=np.float64), np.zeros(0, dtype=np.float64)

        q_unique = list(dict.fromkeys(q_toks))
        if self.sent_bm25.idf_num_docs < 10:
            q_idfs = np.ones(len(q_unique), dtype=np.float64)
        else:
            q_idfs = np.array(
                [self.sent_bm25.idf.get(t, self.sent_bm25.default_idf) for t in q_unique],
                dtype=np.float64,
            )
        total_idf = float(np.sum(q_idfs))
        if total_idf <= 0.0:
            return np.zeros(n_cand, dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Precompute which query tokens are completely OOV in both lexical and concept vocabularies
        known_in_corpus: list[bool] = []
        for t in q_unique:
            stem = _normalize_orthography_and_stem(t)
            in_vocab = (
                (t in self.sent_bm25.idf)
                or (f"st:{stem}" in self._sent_subword.idf)
                or any(f"sy:{cid}" in self._sent_subword.idf for cid in _TOKEN_TO_CONCEPTS.get(t, ()))
                or any(f"sy:{cid}" in self._sent_subword.idf for cid in _TOKEN_TO_CONCEPTS.get(stem, ()))
            )
            known_in_corpus.append(in_vocab)

        title_tok_cache: dict[Any, list[str]] = {}
        cross_ctx = np.zeros(n_cand, dtype=np.float64)
        sent_para = np.zeros(n_cand, dtype=np.float64)

        for idx, gid in enumerate(cand_global_ids):
            did, _ = self.sent_to_doc[gid]
            sent_text = self.all_sentences[gid]
            s_toks = [t for t in tokenize(sent_text) if t not in _NEGATION_TOKENS]
            if not s_toks:
                continue

            if did not in title_tok_cache:
                title_tok_cache[did] = [
                    t for t in tokenize(str(self.corpus[did].get("title", ""))) if t not in _NEGATION_TOKENS
                ]
            t_toks = title_tok_cache[did]

            sent_sims = np.zeros(len(q_unique), dtype=np.float64)
            ctx_sims = np.zeros(len(q_unique), dtype=np.float64)
            unmatched_oov = 0

            for qi, qt in enumerate(q_unique):
                best_s = max((_token_similarity(qt, st) for st in s_toks), default=0.0)
                sent_sims[qi] = best_s
                if best_s < 1.0 and t_toks:
                    best_t = max((_token_similarity(qt, tt) for tt in t_toks), default=0.0)
                    ctx_sims[qi] = max(best_s, 0.35 * best_t)
                else:
                    ctx_sims[qi] = best_s
                if best_s < 0.45 and not known_in_corpus[qi]:
                    unmatched_oov += 1

            # Require at least one meaningful sentence-level alignment before applying title context
            if float(np.max(sent_sims)) < 0.50:
                continue

            weighted_sent = float(np.sum(q_idfs * sent_sims)) / total_idf
            weighted_ctx = float(np.sum(q_idfs * ctx_sims)) / total_idf
            aligned_q_frac = float(np.mean(sent_sims >= 0.55))
            s_aligned = [
                max((_token_similarity(qt, st) for qt in q_unique), default=0.0)
                for st in s_toks
            ]
            aligned_s_frac = float(np.mean([v >= 0.55 for v in s_aligned])) if s_aligned else 0.0
            f1_align = (
                (2.0 * aligned_q_frac * aligned_s_frac / (aligned_q_frac + aligned_s_frac))
                if (aligned_q_frac + aligned_s_frac) > 0.0
                else 0.0
            )

            # Adjacent query-token bigram order bonus inside the candidate sentence
            bigram_bonus = 0.0
            if len(q_unique) >= 2 and len(s_toks) >= 2:
                s_stems = [_normalize_orthography_and_stem(t) for t in s_toks]
                q_stems = [_normalize_orthography_and_stem(t) for t in q_unique]
                s_bigrams = {(s_stems[i], s_stems[i + 1]) for i in range(len(s_stems) - 1)}
                q_bigrams = [(q_stems[i], q_stems[i + 1]) for i in range(len(q_stems) - 1)]
                if q_bigrams:
                    bigram_bonus = 0.08 * (
                        sum(1 for bg in q_bigrams if bg in s_bigrams) / float(len(q_bigrams))
                    )

            doc_prior = float(doc_context_map.get(did, 0.0))
            cross_ctx[idx] = min(1.0, weighted_ctx + bigram_bonus + 0.12 * doc_prior)

            oov_pen = (
                ((1.0 - unmatched_oov / float(len(q_unique))) ** 2)
                if (self.sent_bm25.idf_num_docs >= 10 and unmatched_oov >= 2)
                else 1.0
            )
            has_paraphrase_bridge = bool(np.any((sent_sims >= 0.55) & (sent_sims < 0.999)))
            if has_paraphrase_bridge:
                sent_para[idx] = min(
                    1.0,
                    max(0.0, (0.55 * weighted_sent + 0.45 * f1_align) * oov_pen),
                )

        return cross_ctx, sent_para

    def retrieve(
        self,
        claim: str,
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve top-`doc_top_m` abstracts via hybrid RRF and rerank candidate sentences."""
        doc_bm25_raw = self.doc_bm25.score_indexed(claim)
        doc_bm25_norm = self.doc_bm25.score_indexed_normalized(claim)
        doc_sub_cos = self._doc_subword.cosine_scores(claim)

        stage1_channels: list[np.ndarray] = [doc_bm25_raw, doc_sub_cos]
        stage1_weights: list[float] = [1.35, 0.35]
        if self.embed_fn is not None:
            doc_ext_cos = self._external_cosine(claim, self._doc_embeddings)
            stage1_channels.append(doc_ext_cos)
            stage1_weights.append(1.0)
        else:
            doc_ext_cos = np.zeros_like(doc_bm25_raw)

        doc_rrf = _reciprocal_rank_fusion(stage1_channels, rrf_k=60, weights=stage1_weights)
        m = max(1, min(int(doc_top_m), len(self.doc_ids)))
        # Break zero-RRF ties deterministically by original order
        top_doc_pos = np.lexsort((-doc_sub_cos, -doc_bm25_raw, -doc_rrf))[:m]
        top_docs = [self.doc_ids[int(i)] for i in top_doc_pos]

        doc_context_map = {
            self.doc_ids[int(i)]: float(
                max(doc_bm25_norm[int(i)], doc_sub_cos[int(i)], doc_ext_cos[int(i)])
            )
            for i in top_doc_pos
        }

        cand_global_ids: list[int] = []
        for did in top_docs:
            lo, hi = self.doc_sent_spans[did]
            cand_global_ids.extend(range(lo, hi))

        if not cand_global_ids:
            return {
                "top_doc_ids": top_docs,
                "top_sentences": [],
                "max_sentence_score": 0.0,
                "max_raw_bm25_score": 0.0,
                "abstain": True,
            }

        sent_bm25_raw = self.sent_bm25.score_indexed(claim, cand_global_ids)
        sent_bm25_norm = self.sent_bm25.score_indexed_normalized(claim, cand_global_ids)
        sent_sub_cos = self._sent_subword.cosine_scores(claim, cand_global_ids)
        cross_ctx, sent_para = self._cross_interaction_scores(
            claim, cand_global_ids, doc_context_map
        )

        if self.embed_fn is not None:
            sent_ext_cos = self._external_cosine(claim, self._sent_embeddings, cand_global_ids)
            dense_channel = np.maximum(sent_sub_cos, sent_ext_cos)
            para_calib = np.maximum(sent_para, sent_ext_cos * np.maximum(sent_para, 0.75))
        else:
            dense_channel = sent_sub_cos
            # Gate pure paraphrase fallback on positive subword/concept cosine support
            para_calib = np.where(sent_sub_cos >= 0.15, sent_para, sent_para * (sent_sub_cos / 0.15))

        hybrid_scores = np.clip(
            np.maximum(sent_bm25_norm, para_calib)
            + 0.04 * np.minimum(sent_bm25_norm, cross_ctx),
            0.0,
            1.0,
        )

        rerank_polarities: list[str | None] = [None] * len(cand_global_ids)
        if self.rerank_fn is not None:
            cand_texts = [self.all_sentences[gid] for gid in cand_global_ids]
            rerank_out = list(self.rerank_fn(claim, cand_texts))
            if len(rerank_out) != len(cand_texts):
                raise ValueError(
                    f"rerank_fn returned {len(rerank_out)} items for {len(cand_texts)} candidates"
                )
            rr_scores = np.zeros(len(cand_texts), dtype=np.float64)
            for idx, (sc_val, pol_val) in enumerate(rerank_out):
                pol_clean = str(pol_val).upper().strip()
                if pol_clean not in POLARITY_VALUES:
                    raise ValueError(
                        f"rerank_fn returned invalid polarity {pol_val!r}; expected {POLARITY_VALUES}"
                    )
                rr_scores[idx] = float(sc_val)
                rerank_polarities[idx] = pol_clean
            # Normalize arbitrary cross-encoder logits/scores into [0, 1] if outside [0, 1]
            if np.any(rr_scores < 0.0) or np.any(rr_scores > 1.0):
                rr_probs = 1.0 / (1.0 + np.exp(-np.clip(rr_scores, -30.0, 30.0)))
            else:
                rr_probs = rr_scores
            hybrid_scores = np.clip(rr_probs, 0.0, 1.0)
            order_full = np.argsort(-rr_scores, kind="stable")
        else:
            sent_rrf = _reciprocal_rank_fusion(
                [sent_bm25_raw, dense_channel, cross_ctx],
                rrf_k=60,
                weights=[1.15, 0.65, 1.0],
            )
            order_full = np.lexsort((-sent_bm25_raw, -hybrid_scores, -sent_rrf))

        k = max(1, min(int(sent_top_k), len(cand_global_ids)))
        order = order_full[:k]

        ranked_sents: list[dict[str, Any]] = []
        for pos in order:
            p_int = int(pos)
            gid = cand_global_ids[p_int]
            did, local_idx = self.sent_to_doc[gid]
            entry: dict[str, Any] = {
                "doc_id": did,
                "sent_idx": local_idx,
                "score": float(hybrid_scores[p_int]),
                "raw_bm25_score": float(sent_bm25_raw[p_int]),
                "sentence": self.all_sentences[gid],
                "title": str(self.corpus[did].get("title", "")),
            }
            if rerank_polarities[p_int] is not None:
                entry["polarity"] = rerank_polarities[p_int]
            ranked_sents.append(entry)

        max_score = float(np.max(hybrid_scores)) if hybrid_scores.size > 0 else 0.0
        max_raw_bm25 = float(np.max(sent_bm25_raw)) if sent_bm25_raw.size > 0 else 0.0
        return {
            "top_doc_ids": top_docs,
            "top_sentences": ranked_sents,
            "max_sentence_score": max_score,
            "max_raw_bm25_score": max_raw_bm25,
            "abstain": max_score < self.abstention_threshold,
        }

    @staticmethod
    def _compute_midpoint_threshold(
        evidence_scores: Sequence[float],
        noinfo_scores: Sequence[float],
        fallback: float,
    ) -> float:
        if evidence_scores and noinfo_scores:
            return 0.5 * (
                float(np.median(evidence_scores)) + float(np.median(noinfo_scores))
            )
        return float(fallback)

    def fit_abstention_threshold(
        self,
        calibration_claims: Sequence[Mapping[str, Any]],
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
    ) -> float:
        """Fit and persist `self.abstention_threshold` on a dedicated calibration claim split."""
        ev_scores: list[float] = []
        ni_scores: list[float] = []
        for c in calibration_claims:
            res = self.retrieve(str(c["claim"]), doc_top_m=doc_top_m, sent_top_k=sent_top_k)
            if c.get("evidence"):
                ev_scores.append(float(res["max_sentence_score"]))
            else:
                ni_scores.append(float(res["max_sentence_score"]))
        self.abstention_threshold = self._compute_midpoint_threshold(
            ev_scores, ni_scores, self.abstention_threshold
        )
        self._is_explicitly_calibrated = True
        return self.abstention_threshold

    def evaluate_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
        calibration_claims: Sequence[Mapping[str, Any]] | None = None,
        n_cv_folds: int = 5,
    ) -> dict[str, float]:
        """Evaluate open-corpus retrieval on all claims without in-sample threshold leakage."""
        doc_hits: list[float] = []
        sent_hits: list[float] = []
        evidence_max_scores: list[float] = []
        noinfo_max_scores: list[float] = []
        item_records: list[tuple[bool, float]] = []

        for c in claims:
            res = self.retrieve(str(c["claim"]), doc_top_m=doc_top_m, sent_top_k=sent_top_k)
            score = float(res["max_sentence_score"])
            evidence = c.get("evidence") or {}
            if evidence:
                gold_docs = {str(did) for did in evidence.keys()}
                pred_docs = {str(did) for did in res["top_doc_ids"]}
                doc_hits.append(len(gold_docs & pred_docs) / len(gold_docs))

                gold_sent_pairs = {
                    (str(did), int(si))
                    for did, sets in evidence.items()
                    for s in sets
                    for si in s.get("sentences", ())
                }
                pred_sent_pairs = {
                    (str(item["doc_id"]), int(item["sent_idx"]))
                    for item in res["top_sentences"]
                }
                sent_hits.append(float(bool(gold_sent_pairs & pred_sent_pairs)))
                evidence_max_scores.append(score)
                item_records.append((True, score))
            else:
                noinfo_max_scores.append(score)
                item_records.append((False, score))

        if calibration_claims is not None:
            cal_ev: list[float] = []
            cal_ni: list[float] = []
            for c in calibration_claims:
                r_cal = self.retrieve(str(c["claim"]), doc_top_m=doc_top_m, sent_top_k=sent_top_k)
                (cal_ev if c.get("evidence") else cal_ni).append(float(r_cal["max_sentence_score"]))
            threshold = self._compute_midpoint_threshold(cal_ev, cal_ni, self.abstention_threshold)
            noinfo_false_discoveries = [s >= threshold for s in noinfo_max_scores]
        elif self._is_explicitly_calibrated:
            threshold = self.abstention_threshold
            noinfo_false_discoveries = [s >= threshold for s in noinfo_max_scores]
        elif len(evidence_max_scores) >= 2 and len(noinfo_max_scores) >= 2 and n_cv_folds >= 2:
            # Out-of-fold K-fold evaluation so NOINFO FDR never calibrates on held-in test claims
            k_folds = min(int(n_cv_folds), len(evidence_max_scores), len(noinfo_max_scores))
            fold_thresholds: list[float] = []
            noinfo_false_discoveries = []
            for fold_idx in range(k_folds):
                tr_ev = [
                    sc for i, (is_ev, sc) in enumerate(item_records) if is_ev and (i % k_folds) != fold_idx
                ]
                tr_ni = [
                    sc for i, (is_ev, sc) in enumerate(item_records) if (not is_ev) and (i % k_folds) != fold_idx
                ]
                fold_th = self._compute_midpoint_threshold(tr_ev, tr_ni, self.abstention_threshold)
                fold_thresholds.append(fold_th)
                for i, (is_ev, sc) in enumerate(item_records):
                    if (not is_ev) and (i % k_folds) == fold_idx:
                        noinfo_false_discoveries.append(sc >= fold_th)
            threshold = float(np.mean(fold_thresholds))
        else:
            threshold = self.abstention_threshold
            noinfo_false_discoveries = [s >= threshold for s in noinfo_max_scores]

        return {
            f"doc_recall@{doc_top_m}": float(np.mean(doc_hits)) if doc_hits else float("nan"),
            f"sentence_hit@{sent_top_k}": float(np.mean(sent_hits)) if sent_hits else float("nan"),
            "open_auroc_vs_noinfo": compute_auroc(evidence_max_scores, noinfo_max_scores),
            "noinfo_fdr": float(np.mean(noinfo_false_discoveries)) if noinfo_false_discoveries else float("nan"),
            "calibrated_threshold": float(threshold),
        }

    def probe_claim(
        self,
        claim: str,
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 6,
        candidate_descriptors: Mapping[str, float] | None = None,
        candidate_smiles: str | None = None,
        parsed_spans: Mapping[str, Any] | None = None,
    ) -> EpistemicProbeResult:
        """Probe an arbitrary claim across the corpus and emit a typed EpistemicProbeResult."""
        res = self.retrieve(claim, doc_top_m=doc_top_m, sent_top_k=sent_top_k)
        max_score = float(res["max_sentence_score"])
        # Conservative floor permille projection aligned with LabAskWhyLean.EpistemicProbe
        confidence = to_permille_floor(max_score) / 1000.0

        supporting: list[RationaleEvidence] = []
        contradicting: list[RationaleEvidence] = []
        raw_top_texts: list[str] = []

        # Keep rationales that meaningfully clear positive relevance
        min_item_floor = min(0.05, 0.5 * max(0.02, self.abstention_threshold))
        for item in res["top_sentences"]:
            if float(item["score"]) < min_item_floor:
                continue
            pol = (
                str(item["polarity"])
                if "polarity" in item
                else classify_rationale_polarity(claim, item["sentence"])
            )
            ev = RationaleEvidence(
                doc_id=item["doc_id"],
                sent_idx=item["sent_idx"],
                sentence=item["sentence"],
                bm25_score=float(item["score"]),
                polarity=pol,
                title=item["title"],
            )
            if pol == "CONTRADICT":
                contradicting.append(ev)
                raw_top_texts.append(item["sentence"])
            elif pol == "SUPPORT":
                supporting.append(ev)
                raw_top_texts.append(item["sentence"])

        if max_score < self.abstention_threshold or (not supporting and not contradicting):
            status = "INSUFFICIENT_EVIDENCE"
        elif supporting and contradicting:
            status = "CONTESTED"
        elif contradicting:
            status = "CONTRADICTED"
        else:
            status = "SUPPORTED"

        stance_hint = "CONTRADICT" if status == "CONTRADICTED" else None
        questions = why_questions(claim, raw_top_texts[:3] or None, stance=stance_hint)
        diff_terms = (
            extract_differential_terms(claim, raw_top_texts[:3], max_terms=4)
            if raw_top_texts
            else []
        )

        assumptions = [
            f"Observed lexical overlap reflects a direct causal link rather than confounding via {', '.join(diff_terms[:2]) or 'background covariates'}.",
            f"Experimental regime in retrieved literature ({res['top_doc_ids'][:3]}) matches the target population/conditions of '{claim.strip().rstrip('.')}'.",
        ]

        resolved_descriptors = extract_protocol_descriptors(
            [claim, *raw_top_texts[:4]],
            parsed_spans=parsed_spans,
            base_descriptors=candidate_descriptors,
            ensure_wetlab_defaults=(status == "CONTESTED"),
        )
        control_wells = (
            synthesize_control_wells(
                claim,
                diff_terms,
                top_sentences=raw_top_texts[:4],
                parsed_spans=parsed_spans,
                candidate_descriptors=resolved_descriptors,
            )
            if status != "INSUFFICIENT_EVIDENCE"
            else []
        )

        observable = DiscriminatingObservable(
            observable_name=(
                f"perturbation_readout_{diff_terms[0]}"
                if diff_terms
                else "direct_replication_assay"
            ),
            differential_terms=diff_terms,
            proposed_perturbation=questions[-1],
            candidate_descriptors=resolved_descriptors,
            candidate_smiles=candidate_smiles,
            control_wells=control_wells,
        )
        return EpistemicProbeResult(
            claim=claim.strip(),
            supporting_rationales=supporting,
            contradicting_rationales=contradicting,
            assumptions=assumptions,
            why_questions=questions,
            discriminating_observable=observable,
            retrieval_confidence=confidence,
            epistemic_status=status,
        )
