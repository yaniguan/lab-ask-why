"""Pretrained Neural Bi-Encoder + DeBERTa-v3 Cross-Encoder NLI Verifier (CPU-runnable).

Architecture (Layer L6 Neural Epistemic Verifier):
1. Stage 1 Hybrid Dense + Lexical Retrieval (`Reciprocal Rank Fusion`, RRF):
   - Channel 1: Pretrained dense biomedical/scientific bi-encoder (`BAAI/bge-small-en-v1.5`,
     384-dim CLS-pooled L2-normalized embeddings with query retrieval prefix).
   - Channel 2: Biomedical Okapi `BM25` fit strictly on non-excluded background abstracts.
   - Channel 3: `EntityNumericNegationIndex` (acronyms, allele symbols, numbers, negations).
2. Stage 2 Pretrained Cross-Encoder 3-Way NLI & Mondrian Split Conformal Calibration:
   - Cross-encoder NLI (`cross-encoder/nli-deberta-v3-small`) scoring `(premise, claim)`
     directly over `("SUPPORT", "CONTRADICT", "NEI")` via length-sorted batched
     `torch.no_grad()` CPU inference.
   - Lightweight post-hoc calibration + Mondrian Split Conformal quantiles (`1 - alpha = 0.90`)
     fit strictly on `SciFact train` (`607` `train_fit` + `202` `train_calib`; `0` leakage
     from `SciFact dev` or external benchmarks).
   - Dedicated Bilateral Cross-Passage Conflict Score:
     `score_conflict = max_{i != j} min(P_NLI(SUPPORT | doc_i, claim), P_NLI(CONTRADICT | doc_j, claim))`
     calibrated strictly on non-conflicting `SciFact train_calib` single-passage controls.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lab_ask_why.bm25 import BM25
from lab_ask_why.data import build_noinfo_pairs, build_pairs
from lab_ask_why.frontier_verifier import (
    STANCE_LABELS,
    ConformalStancePrediction,
    EntityNumericNegationIndex,
    SplitConformalStanceClassifier,
    _STANCE_TO_IDX,
    _fit_multinomial_logistic,
    canonicalize_stance_label,
    split_scifact_train_fit_and_calib,
)
from lab_ask_why.labbench_adapter import segment_protocol_sentences
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
from lab_ask_why.retrieval import _reciprocal_rank_fusion

DEFAULT_BIENCODER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DOC_SPLIT_RE = re.compile(r"(?=\bDocument(?:\s+\d+)?\s*:)")

_MODEL_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any, dict[str, int]]] = {}


def is_neural_backend_available() -> bool:
    """Return True iff `torch` and `transformers` are importable in the active Python environment."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _load_pretrained_backbone(
    biencoder_model: str = DEFAULT_BIENCODER_MODEL,
    nli_model: str = DEFAULT_NLI_MODEL,
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any, Any, Any, dict[str, int]]:
    """Load and memoize CPU `biencoder` and `nli` tokenizers/models."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_key = (biencoder_model, nli_model)
    cached = _MODEL_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    bi_tok = AutoTokenizer.from_pretrained(biencoder_model, local_files_only=local_files_only)
    bi_mod = AutoModel.from_pretrained(biencoder_model, local_files_only=local_files_only)
    bi_mod.eval()
    bi_mod.to("cpu")

    nli_tok = AutoTokenizer.from_pretrained(nli_model, local_files_only=local_files_only)
    nli_mod = AutoModelForSequenceClassification.from_pretrained(
        nli_model, local_files_only=local_files_only
    )
    nli_mod.eval()
    nli_mod.to("cpu")

    id2label = {int(k): str(v).lower() for k, v in (nli_mod.config.id2label or {}).items()}
    label_map: dict[str, int] = {}
    for idx, name in id2label.items():
        if "entail" in name or "support" in name:
            label_map["SUPPORT"] = idx
        elif "contradict" in name or "refut" in name:
            label_map["CONTRADICT"] = idx
        elif "neutral" in name or "nei" in name:
            label_map["NEI"] = idx
    if set(label_map.keys()) != {"SUPPORT", "CONTRADICT", "NEI"}:
        label_map = {"CONTRADICT": 0, "SUPPORT": 1, "NEI": 2}

    bundle = (bi_tok, bi_mod, nli_tok, nli_mod, label_map)
    _MODEL_RUNTIME_CACHE[cache_key] = bundle
    return bundle


class NeuralBiEncoderNliVerifier:
    """2-Stage Dense Bi-Encoder + BM25 RRF Retriever & DeBERTa-v3 NLI Split-Conformal Verifier."""

    def __init__(
        self,
        corpus: Mapping[Any, Mapping[str, Any]],
        *,
        exclude_doc_ids: Iterable[Any] | None = None,
        biencoder_model: str = DEFAULT_BIENCODER_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
        alpha: float = 0.10,
        batch_size: int = 96,
        embedding_cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not corpus:
            raise ValueError("NeuralBiEncoderNliVerifier requires a non-empty corpus")
        self.corpus = corpus
        self.biencoder_model = biencoder_model
        self.nli_model = nli_model
        self.alpha = float(alpha)
        self.batch_size = max(16, int(batch_size))
        self.local_files_only = bool(local_files_only)
        self._embedding_cache_dir = embedding_cache_dir

        (
            self._bi_tok,
            self._bi_mod,
            self._nli_tok,
            self._nli_mod,
            self._nli_idx_map,
        ) = _load_pretrained_backbone(
            biencoder_model=biencoder_model,
            nli_model=nli_model,
            local_files_only=local_files_only,
        )

        excluded = {str(did) for did in (exclude_doc_ids or ())}
        self.excluded_doc_ids = frozenset(excluded)

        self.doc_ids: list[Any] = []
        self.abstract_blobs: list[str] = []
        bg_doc_blobs: list[str] = []
        doc_bg_mask: list[bool] = []

        self.all_sentences: list[str] = []
        self.all_sent_titles: list[str] = []
        self.all_sent_contextual: list[str] = []
        bg_sentences: list[str] = []
        sent_bg_mask: list[bool] = []
        self.doc_sent_spans: dict[Any, tuple[int, int]] = {}
        self.sent_to_doc: list[tuple[Any, int]] = []

        for doc_id, doc in corpus.items():
            title = str(doc.get("title", ""))
            sents = [str(s) for s in doc.get("abstract", ())]
            self.doc_ids.append(doc_id)
            blob = (title + ". " + " ".join(sents)).strip() if title else " ".join(sents).strip()
            self.abstract_blobs.append(blob)
            is_bg = str(doc_id) not in excluded
            doc_bg_mask.append(is_bg)
            if is_bg:
                bg_doc_blobs.append(blob)

            start = len(self.all_sentences)
            for j, s in enumerate(sents):
                self.all_sentences.append(s)
                self.all_sent_titles.append(title)
                self.all_sent_contextual.append(f"{title}. {s}" if title else s)
                self.sent_to_doc.append((doc_id, j))
                sent_bg_mask.append(is_bg)
                if is_bg:
                    bg_sentences.append(s)
            self.doc_sent_spans[doc_id] = (start, len(self.all_sentences))

        use_bg_docs = bg_doc_blobs if (excluded and bg_doc_blobs) else None
        use_bg_sents = bg_sentences if (excluded and bg_sentences) else None
        use_doc_mask = doc_bg_mask if (excluded and bg_doc_blobs) else None

        self.doc_bm25 = BM25(self.abstract_blobs, idf_sentences=use_bg_docs)
        self.sent_bm25 = BM25(
            self.all_sentences if self.all_sentences else [""],
            idf_sentences=use_bg_sents,
        )
        self.doc_entity_idx = EntityNumericNegationIndex(self.abstract_blobs, idf_mask=use_doc_mask)

        self._query_emb_cache: dict[str, np.ndarray] = {}
        self._passage_emb_cache: dict[str, np.ndarray] = {}
        self._nli_pair_cache: dict[tuple[str, str], np.ndarray] = {}

        # Encode ONLY the 5,183 abstract documents upfront (~20s on CPU);
        # candidate sentences are encoded lazily in batches only for retrieved abstracts!
        self.doc_embeddings = self._load_or_encode_doc_embeddings(embedding_cache_dir)

        self.open_stance_head = SplitConformalStanceClassifier(
            weights=np.zeros((10, 3), dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
            feat_mean=np.zeros(10, dtype=np.float64),
            feat_std=np.ones(10, dtype=np.float64),
            alpha=self.alpha,
            conformal_mode="mondrian",
        )
        self.cand_stance_head = SplitConformalStanceClassifier(
            weights=np.zeros((10, 3), dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
            feat_mean=np.zeros(10, dtype=np.float64),
            feat_std=np.ones(10, dtype=np.float64),
            alpha=self.alpha,
            conformal_mode="mondrian",
        )
        self.conflict_threshold: float = 0.45
        self.training_provenance: dict[str, Any] = {
            "trained": False,
            "biencoder_model": self.biencoder_model,
            "nli_model": self.nli_model,
            "alpha": self.alpha,
        }

    def _load_or_encode_doc_embeddings(self, cache_dir: Path | None) -> np.ndarray:
        digest = hashlib.sha256(
            (
                self.biencoder_model
                + "\n"
                + "\n".join(self.abstract_blobs[:64])
                + f"\n{len(self.abstract_blobs)}"
            ).encode("utf-8")
        ).hexdigest()[:20]

        cache_file = cache_dir / f"bge_docs_{digest}.npz" if cache_dir is not None else None
        if cache_file is not None and cache_file.is_file():
            try:
                loaded = np.load(cache_file)
                d_emb = np.asarray(loaded["doc_emb"], dtype=np.float32)
                if d_emb.shape[0] == len(self.abstract_blobs):
                    return d_emb
            except Exception:
                pass

        doc_emb = self.encode_texts(self.abstract_blobs, is_query=False, max_length=192)
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_file, doc_emb=doc_emb)
            except OSError:
                pass
        return doc_emb

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        max_length: int = 160,
    ) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings via `BAAI/bge-small-en-v1.5`."""
        import torch

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        prefixed = (
            [f"{_BGE_QUERY_PREFIX}{t}" for t in texts]
            if (is_query and "bge" in self.biencoder_model.lower())
            else list(texts)
        )
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        out_chunks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for start in range(0, len(order), bs):
                batch_idx = order[start : start + bs]
                batch_txt = [prefixed[i] for i in batch_idx]
                enc = self._bi_tok(
                    batch_txt,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                cls_vec = self._bi_mod(**enc).last_hidden_state[:, 0]
                normed = torch.nn.functional.normalize(cls_vec, p=2, dim=-1)
                out_chunks.append(normed.cpu().numpy().astype(np.float32))

        stacked = np.vstack(out_chunks)
        restored = np.empty_like(stacked)
        for sorted_pos, orig_idx in enumerate(order):
            restored[orig_idx] = stacked[sorted_pos]
        return restored

    def encode_queries_cached(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, 384), dtype=np.float32)
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_emb_cache]
        if missing:
            embs = self.encode_texts(missing, is_query=True, max_length=128)
            for q, vec in zip(missing, embs):
                self._query_emb_cache[q] = vec
        return np.vstack([self._query_emb_cache[q] for q in queries])

    def encode_passages_cached(self, passages: Sequence[str], *, max_length: int = 160) -> np.ndarray:
        if not passages:
            return np.zeros((0, 384), dtype=np.float32)
        missing = [p for p in dict.fromkeys(passages) if p not in self._passage_emb_cache]
        if missing:
            embs = self.encode_texts(missing, is_query=False, max_length=max_length)
            for p, vec in zip(missing, embs):
                self._passage_emb_cache[p] = vec
        return np.vstack([self._passage_emb_cache[p] for p in passages])

    def score_nli_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        max_length: int = 192,
    ) -> np.ndarray:
        """Score `(premise, claim)` pairs into `(N, 3)` probabilities `(SUPPORT, CONTRADICT, NEI)`."""
        import torch

        if not pairs:
            return np.zeros((0, 3), dtype=np.float64)

        missing = list(dict.fromkeys(p for p in pairs if p not in self._nli_pair_cache))
        if missing:
            order = sorted(
                range(len(missing)),
                key=lambda i: len(missing[i][0]) + len(missing[i][1]),
            )
            bs = max(32, self.batch_size)
            idx_sup = self._nli_idx_map["SUPPORT"]
            idx_con = self._nli_idx_map["CONTRADICT"]
            idx_nei = self._nli_idx_map["NEI"]

            with torch.no_grad():
                for start in range(0, len(order), bs):
                    b_idx = order[start : start + bs]
                    premises = [missing[i][0] for i in b_idx]
                    hypotheses = [missing[i][1] for i in b_idx]
                    enc = self._nli_tok(
                        premises,
                        hypotheses,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    logits = self._nli_mod(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                    for pos_in_b, m_idx in enumerate(b_idx):
                        row = probs[pos_in_b]
                        canon = np.array(
                            [row[idx_sup], row[idx_con], row[idx_nei]], dtype=np.float64
                        )
                        canon /= max(float(np.sum(canon)), 1e-12)
                        self._nli_pair_cache[missing[m_idx]] = canon

        return np.vstack([self._nli_pair_cache[p] for p in pairs])

    def prewarm_scifact_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        doc_top_m: int = 5,
        nli_top_s: int = 5,
    ) -> None:
        """Batch-encode retrieved sentences & NLI pairs across an entire SciFact claim split at once."""
        if not claims:
            return
        claim_strs = [str(c["claim"]) for c in claims]
        self.encode_queries_cached(claim_strs)

        # Collect all sentence global IDs touched by stage-1 retrieved docs and cited/evidence docs
        touched_gids: list[int] = []
        per_claim_gids: list[tuple[str, list[int]]] = []
        extra_nli_pairs: list[tuple[str, str]] = []

        for c in claims:
            c_str = str(c["claim"])
            top_docs = self.retrieve_stage1_docs(c_str, top_m=doc_top_m)
            gids: list[int] = []
            for did in top_docs:
                lo, hi = self.doc_sent_spans[did]
                gids.extend(range(lo, hi))
            per_claim_gids.append((c_str, gids))
            touched_gids.extend(gids)

            ev = c.get("evidence") or {}
            for raw_did in [*list(ev.keys()), *(c.get("cited_doc_ids") or ())]:
                d_key = self._resolve_doc_key(raw_did)
                if d_key is not None:
                    lo, hi = self.doc_sent_spans[d_key]
                    touched_gids.extend(range(lo, hi))
                    for g in range(lo, hi):
                        extra_nli_pairs.append((self.all_sent_contextual[g], c_str))

        if touched_gids:
            uniq_ctx = [self.all_sent_contextual[g] for g in dict.fromkeys(touched_gids)]
            self.encode_passages_cached(uniq_ctx, max_length=160)

        # Collect top-S NLI pairs per claim in one single batch
        open_nli_pairs: list[tuple[str, str]] = list(extra_nli_pairs)
        for c_str, gids in per_claim_gids:
            if not gids:
                continue
            q_vec = self._query_emb_cache[c_str]
            s_embs = np.vstack([self._passage_emb_cache[self.all_sent_contextual[g]] for g in gids])
            dense_cos = np.clip(s_embs @ q_vec, -1.0, 1.0).astype(np.float64)
            bm25_norm = self.sent_bm25.score_indexed_normalized(c_str, gids)
            bm25_raw = self.sent_bm25.score_indexed(c_str, gids)
            bm25_max = float(np.max(bm25_raw)) if bm25_raw.size > 0 else 0.0
            bm25_rel = (bm25_raw / bm25_max) if bm25_max > 0.0 else np.zeros_like(bm25_raw)
            stage1_rel = 0.55 * np.maximum(0.0, dense_cos) + 0.45 * np.maximum(
                bm25_norm, 0.5 * bm25_rel
            )
            pre_order = np.argsort(-stage1_rel, kind="stable")[: min(int(nli_top_s), len(gids))]
            for i in pre_order:
                open_nli_pairs.append((self.all_sent_contextual[gids[int(i)]], c_str))

        if open_nli_pairs:
            self.score_nli_pairs(open_nli_pairs, max_length=192)

    def prewarm_candidate_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        nli_top_s: int = 4,
    ) -> None:
        """Batch-encode and batch-NLI-score an entire candidate-verification dataset at once."""
        if not items:
            return
        claims = [str(it["claim"]) for it in items]
        self.encode_queries_cached(claims)

        all_ctx_sents: list[str] = []
        prepared: list[tuple[str, list[str], list[str]]] = []
        for it in items:
            c_str = str(it["claim"])
            title = str(it.get("title", ""))
            clean = [str(s).strip() for s in it["evidence_sents"] if str(s).strip()]
            ctx = [f"{title}. {s}" if title else s for s in clean]
            prepared.append((c_str, clean, ctx))
            all_ctx_sents.extend(ctx)

        if all_ctx_sents:
            self.encode_passages_cached(all_ctx_sents, max_length=160)

        nli_pairs: list[tuple[str, str]] = []
        for c_str, clean, ctx in prepared:
            if not clean:
                continue
            if len(clean) <= nli_top_s:
                for ct in ctx:
                    nli_pairs.append((ct, c_str))
            else:
                q_vec = self._query_emb_cache[c_str]
                s_embs = np.vstack([self._passage_emb_cache[ct] for ct in ctx])
                dense_all = np.clip(s_embs @ q_vec, -1.0, 1.0).astype(np.float64)
                bm25_norm_all = np.maximum(
                    self.sent_bm25.normalized_scores(c_str, clean),
                    BM25(clean).score_indexed_normalized(c_str),
                )
                stage1_all = 0.60 * np.maximum(0.0, dense_all) + 0.40 * bm25_norm_all
                top_i = np.argsort(-stage1_all, kind="stable")[:nli_top_s]
                for idx in top_i:
                    nli_pairs.append((ctx[int(idx)], c_str))

        if nli_pairs:
            self.score_nli_pairs(nli_pairs, max_length=192)

    def prewarm_conflict_cases(
        self,
        cases: Sequence[tuple[str, list[str], Any]],
    ) -> None:
        """Batch-encode and batch-NLI-score all multi-passage and negative-control conflict items."""
        if not cases:
            return
        self.encode_queries_cached([c for c, _, _ in cases])
        all_sents: list[str] = []
        split_cache: list[tuple[str, list[list[str]], list[str]]] = []
        for claim, raw_passages, _ in cases:
            raw_units = [str(u).strip() for u in raw_passages if str(u).strip()]
            if len(raw_units) == 1:
                doc_parts = [d.strip() for d in _DOC_SPLIT_RE.split(raw_units[0]) if d.strip()]
                units = doc_parts if len(doc_parts) >= 2 else [
                    s.strip() for s in segment_protocol_sentences(raw_units[0]) if s.strip()
                ]
            else:
                units = raw_units
            unit_sents = [
                [s.strip() for s in segment_protocol_sentences(u) if s.strip()]
                for u in units
            ]
            for us in unit_sents:
                if len(us) > 2:
                    all_sents.extend(us)
            split_cache.append((claim, unit_sents, units))

        if all_sents:
            self.encode_passages_cached(all_sents, max_length=160)

        nli_pairs: list[tuple[str, str]] = []
        for claim, unit_sents, units in split_cache:
            if len(units) < 2:
                continue
            q_vec = self._query_emb_cache[claim]
            for u, us in zip(units, unit_sents):
                if len(us) <= 2:
                    nli_pairs.append((u, claim))
                else:
                    embs = np.vstack([self._passage_emb_cache[s] for s in us])
                    top2 = sorted(int(i) for i in np.argsort(-(embs @ q_vec), kind="stable")[:2])
                    nli_pairs.append((" ".join(us[i] for i in top2), claim))

        if nli_pairs:
            self.score_nli_pairs(nli_pairs, max_length=224)

    def prewarm_mcq_items(self, mcq_items: Sequence[Mapping[str, Any]]) -> None:
        """Batch-encode and batch-NLI-score all MCQ option discrimination items at once."""
        if not mcq_items:
            return
        questions = [str(it["question"]).strip() for it in mcq_items if str(it.get("question", "")).strip()]
        self.encode_queries_cached(questions)

        long_sents: list[str] = []
        all_opts: list[str] = []
        for it in mcq_items:
            sents = [str(s).strip() for s in it["sentences"] if str(s).strip()]
            if len(sents) > 3:
                long_sents.extend(sents)
            all_opts.append(str(it["ideal"]).strip())
            all_opts.extend(str(d).strip() for d in it["distractors"] if str(d).strip())

        if long_sents or all_opts:
            self.encode_passages_cached([*long_sents, *all_opts], max_length=160)

        premises: list[str] = []
        nli_pairs: list[tuple[str, str]] = []
        for it in mcq_items:
            q = str(it["question"]).strip()
            ideal = str(it["ideal"]).strip()
            dists = [str(d).strip() for d in it["distractors"] if str(d).strip()]
            sents = [str(s).strip() for s in it["sentences"] if str(s).strip()]
            if not q or not ideal or not dists or not sents:
                continue
            if len(sents) > 3:
                q_vec = self._query_emb_cache[q]
                embs = np.vstack([self._passage_emb_cache[s] for s in sents])
                top_i = sorted(int(i) for i in np.argsort(-(embs @ q_vec), kind="stable")[:3])
                prem = " ".join(sents[i] for i in top_i)
            else:
                prem = " ".join(sents)
            premises.append(prem)
            for opt in [ideal, *dists]:
                hyp = f"{q} Proposed answer: {opt}" if q.endswith("?") else f"{q.rstrip('.')} -- {opt}"
                nli_pairs.append((prem, hyp))

        if premises:
            self.encode_passages_cached(premises, max_length=224)
        if nli_pairs:
            self.score_nli_pairs(nli_pairs, max_length=224)

    def _resolve_doc_key(self, raw_doc_id: Any) -> Any | None:
        if raw_doc_id in self.doc_sent_spans:
            return raw_doc_id
        s_id = str(raw_doc_id)
        if s_id in self.doc_sent_spans:
            return s_id
        if s_id.isdigit() and int(s_id) in self.doc_sent_spans:
            return int(s_id)
        return None

    def retrieve_stage1_docs(self, claim: str, *, top_m: int = 5) -> list[Any]:
        """Stage 1 hybrid document retrieval fusing `BGE-small` dense cosine + `BM25` + `Entity` RRF."""
        q_vec = self.encode_queries_cached([claim])[0]
        dense_cos = np.clip(self.doc_embeddings @ q_vec, -1.0, 1.0).astype(np.float64)
        bm25_sc = self.doc_bm25.score_indexed(claim)
        ent_sc = self.doc_entity_idx.overlap_scores(claim)
        rrf = _reciprocal_rank_fusion(
            [bm25_sc, np.maximum(0.0, dense_cos), ent_sc],
            rrf_k=60,
            weights=[1.35, 1.25, 0.35],
        )
        bm25_max = float(np.max(bm25_sc)) if bm25_sc.size > 0 else 0.0
        bm25_rel = (bm25_sc / bm25_max) if bm25_max > 0.0 else np.zeros_like(bm25_sc)
        combined = (
            0.42 * bm25_rel
            + 0.43 * np.maximum(0.0, dense_cos)
            + 0.15 * (rrf / ((1.35 + 1.25 + 0.35) / 61.0))
        )
        m = max(1, min(int(top_m), len(self.doc_ids)))
        order = np.lexsort((-dense_cos, -bm25_sc, -combined))[:m]
        return [self.doc_ids[int(i)] for i in order]

    def _aggregate_nli_features(
        self,
        nli_probs: np.ndarray,
        rel_scores: np.ndarray,
        bm25_norm: np.ndarray,
        dense_cos: np.ndarray,
    ) -> np.ndarray:
        """Build 10-dim stance feature vector from sentence-level DeBERTa-v3 NLI & retrieval scores."""
        feat = np.zeros(10, dtype=np.float64)
        if nli_probs.shape[0] == 0:
            feat[9] = 1.0
            return feat

        order = np.argsort(-rel_scores, kind="stable")
        k = min(3, len(order))
        top = order[:k]
        w_raw = np.maximum(rel_scores[top], 1e-4)
        w = w_raw / float(np.sum(w_raw))

        p_sup = nli_probs[top, 0]
        p_con = nli_probs[top, 1]
        p_nei = nli_probs[top, 2]

        feat[0] = float(np.sum(w * p_sup))
        feat[1] = float(np.sum(w * p_con))
        feat[2] = float(np.sum(w * p_nei))
        feat[3] = float(np.max(p_sup * np.clip(rel_scores[top], 0.0, 1.0)))
        feat[4] = float(np.max(p_con * np.clip(rel_scores[top], 0.0, 1.0)))
        feat[5] = float(p_sup[0] - p_con[0])
        feat[6] = float(np.max(rel_scores[top]))
        feat[7] = float(np.max(bm25_norm[top]))
        feat[8] = float(np.max(dense_cos[top]))
        feat[9] = 0.0
        return feat

    def _open_claim_bundle(
        self, claim: str, *, doc_top_m: int = 5, nli_top_s: int = 5
    ) -> dict[str, Any]:
        top_docs = self.retrieve_stage1_docs(claim, top_m=doc_top_m)
        gids: list[int] = []
        for did in top_docs:
            lo, hi = self.doc_sent_spans[did]
            gids.extend(range(lo, hi))
        if not gids:
            return {
                "top_docs": top_docs,
                "ranked_gids": [],
                "rel_scores": np.zeros(0),
                "nli_probs": np.zeros((0, 3)),
                "feat": self._aggregate_nli_features(
                    np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
                ),
            }

        q_vec = self.encode_queries_cached([claim])[0]
        ctx_texts = [self.all_sent_contextual[g] for g in gids]
        s_embs = self.encode_passages_cached(ctx_texts, max_length=160)
        dense_cos = np.clip(s_embs @ q_vec, -1.0, 1.0).astype(np.float64)
        bm25_norm = self.sent_bm25.score_indexed_normalized(claim, gids)
        bm25_raw = self.sent_bm25.score_indexed(claim, gids)
        bm25_max = float(np.max(bm25_raw)) if bm25_raw.size > 0 else 0.0
        bm25_rel = (bm25_raw / bm25_max) if bm25_max > 0.0 else np.zeros_like(bm25_raw)
        stage1_rel = 0.55 * np.maximum(0.0, dense_cos) + 0.45 * np.maximum(bm25_norm, 0.5 * bm25_rel)

        pre_order = np.argsort(-stage1_rel, kind="stable")[: min(int(nli_top_s), len(gids))]
        sel_gids = [gids[int(i)] for i in pre_order]
        pairs = [(self.all_sent_contextual[g], claim) for g in sel_gids]
        nli_probs = self.score_nli_pairs(pairs, max_length=192)
        non_nei = 1.0 - nli_probs[:, 2]
        rel_scores = 0.50 * stage1_rel[pre_order] + 0.50 * non_nei
        rerank_order = np.argsort(-rel_scores, kind="stable")

        ordered_gids = [sel_gids[int(i)] for i in rerank_order]
        ordered_rel = rel_scores[rerank_order]
        ordered_nli = nli_probs[rerank_order]
        ordered_bm25 = bm25_norm[pre_order][rerank_order]
        ordered_dense = dense_cos[pre_order][rerank_order]

        feat = self._aggregate_nli_features(ordered_nli, ordered_rel, ordered_bm25, ordered_dense)
        return {
            "top_docs": top_docs,
            "ranked_gids": ordered_gids,
            "rel_scores": ordered_rel,
            "nli_probs": ordered_nli,
            "feat": feat,
        }

    def _candidate_sentences_bundle(
        self,
        claim: str,
        sentences: Sequence[str],
        *,
        title: str = "",
        nli_top_s: int = 4,
    ) -> dict[str, Any]:
        clean = [str(s).strip() for s in sentences if str(s).strip()]
        if not clean:
            return {
                "ranked_indices": [],
                "rel_scores": np.zeros(0),
                "nli_probs": np.zeros((0, 3)),
                "feat": self._aggregate_nli_features(
                    np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
                ),
            }

        q_vec = self.encode_queries_cached([claim])[0]
        contextual = [f"{title}. {s}" if title else s for s in clean]
        s_embs_all = self.encode_passages_cached(contextual, max_length=160)
        dense_all = np.clip(s_embs_all @ q_vec, -1.0, 1.0).astype(np.float64)
        bm25_norm_all = np.maximum(
            self.sent_bm25.normalized_scores(claim, clean),
            BM25(clean).score_indexed_normalized(claim),
        )
        stage1_all = 0.60 * np.maximum(0.0, dense_all) + 0.40 * bm25_norm_all

        if len(clean) <= nli_top_s:
            pre_idx = list(range(len(clean)))
        else:
            pre_idx = [int(i) for i in np.argsort(-stage1_all, kind="stable")[:nli_top_s]]

        dense_cos = dense_all[pre_idx]
        bm25_norm = bm25_norm_all[pre_idx]
        stage1_sub = stage1_all[pre_idx]

        pairs = [(contextual[i], claim) for i in pre_idx]
        nli_probs = self.score_nli_pairs(pairs, max_length=192)
        non_nei = 1.0 - nli_probs[:, 2]
        rel_scores = 0.45 * stage1_sub + 0.55 * non_nei
        order = np.argsort(-rel_scores, kind="stable")

        ranked_indices = [pre_idx[int(i)] for i in order]
        if len(ranked_indices) < len(clean):
            seen = set(ranked_indices)
            for idx in np.argsort(-stage1_all, kind="stable"):
                ix = int(idx)
                if ix not in seen:
                    ranked_indices.append(ix)

        ordered_rel = rel_scores[order]
        ordered_nli = nli_probs[order]
        ordered_bm25 = bm25_norm[order]
        ordered_dense = dense_cos[order]
        feat = self._aggregate_nli_features(ordered_nli, ordered_rel, ordered_bm25, ordered_dense)
        return {
            "ranked_indices": ranked_indices,
            "rel_scores": ordered_rel,
            "nli_probs": ordered_nli,
            "feat": feat,
        }

    @staticmethod
    def _fit_calibrated_head(
        x_rows: Sequence[np.ndarray],
        y_rows: Sequence[int],
        *,
        alpha: float = 0.10,
    ) -> SplitConformalStanceClassifier:
        sx = np.vstack(x_rows)
        sy = np.asarray(y_rows, dtype=np.int32)
        w_mat, b_vec, mean, std = _fit_multinomial_logistic(sx, sy, l2_reg=2.0, sample_weight=None)
        return SplitConformalStanceClassifier(
            w_mat, b_vec, mean, std, alpha=alpha, conformal_mode="mondrian"
        )

    @classmethod
    def fit_on_scifact_train(
        cls,
        corpus: Mapping[Any, Mapping[str, Any]],
        train_claims: Sequence[Mapping[str, Any]],
        *,
        exclude_doc_ids: Iterable[Any] | None = None,
        biencoder_model: str = DEFAULT_BIENCODER_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
        alpha: float = 0.10,
        calib_fraction: float = 0.25,
        seed: int = 2026,
        batch_size: int = 96,
        embedding_cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> "NeuralBiEncoderNliVerifier":
        """Fit lightweight NLI aggregation head & Mondrian Split Conformal quantiles on `SciFact train` (`607`/`202`)."""
        fit_claims, calib_claims = split_scifact_train_fit_and_calib(
            train_claims, calib_fraction=calib_fraction, seed=seed
        )
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
        combined_excl = {str(did) for did in (exclude_doc_ids or ())} | (
            calib_doc_ids - fit_doc_ids
        )
        if len(combined_excl) >= len(corpus):
            combined_excl = {str(did) for did in (exclude_doc_ids or ())}

        verifier = cls(
            corpus,
            exclude_doc_ids=combined_excl,
            biencoder_model=biencoder_model,
            nli_model=nli_model,
            alpha=alpha,
            batch_size=batch_size,
            embedding_cache_dir=embedding_cache_dir,
            local_files_only=local_files_only,
        )

        # Pre-warm all train sentence embeddings and NLI pairs in 2 batched passes
        verifier.prewarm_scifact_claims(train_claims, doc_top_m=5, nli_top_s=5)

        def _gold_stance(c_obj: Mapping[str, Any]) -> str:
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

        open_fit_x: list[np.ndarray] = []
        open_fit_y: list[int] = []
        cand_fit_x: list[np.ndarray] = []
        cand_fit_y: list[int] = []

        for c in fit_claims:
            g_lab = _gold_stance(c)
            c_str = str(c["claim"])
            ob = verifier._open_claim_bundle(c_str, doc_top_m=5, nli_top_s=5)
            open_fit_x.append(ob["feat"])
            open_fit_y.append(_STANCE_TO_IDX[g_lab])

            ev = c.get("evidence") or {}
            if ev:
                for raw_did, sets in ev.items():
                    d_key = verifier._resolve_doc_key(raw_did)
                    if d_key is None:
                        continue
                    p_labs = {
                        str(e.get("label", "")).upper()
                        for e in (sets if isinstance(sets, Sequence) else ())
                        if isinstance(e, Mapping)
                    }
                    p_lab = "CONTRADICT" if "CONTRADICT" in p_labs else "SUPPORT"
                    doc = corpus[d_key]
                    cb = verifier._candidate_sentences_bundle(
                        c_str,
                        list(doc.get("abstract", ())),
                        title=str(doc.get("title", "")),
                    )
                    cand_fit_x.append(cb["feat"])
                    cand_fit_y.append(_STANCE_TO_IDX[p_lab])
            else:
                for raw_did in (c.get("cited_doc_ids") or ())[:1]:
                    d_key = verifier._resolve_doc_key(raw_did)
                    if d_key is not None:
                        doc = corpus[d_key]
                        cb = verifier._candidate_sentences_bundle(
                            c_str,
                            list(doc.get("abstract", ())),
                            title=str(doc.get("title", "")),
                        )
                        cand_fit_x.append(cb["feat"])
                        cand_fit_y.append(_STANCE_TO_IDX["NEI"])

        if open_fit_x:
            verifier.open_stance_head = cls._fit_calibrated_head(open_fit_x, open_fit_y, alpha=alpha)
        if cand_fit_x:
            verifier.cand_stance_head = cls._fit_calibrated_head(cand_fit_x, cand_fit_y, alpha=alpha)

        # Pre-warm single-abstract calibration conflict pairs in one batch
        cal_neg_cases: list[tuple[str, list[str], Any]] = []
        for c in calib_claims:
            ev = c.get("evidence") or {}
            for raw_did in ev.keys():
                d_key = verifier._resolve_doc_key(raw_did)
                if d_key is not None:
                    sents = [str(s) for s in corpus[d_key].get("abstract", ())]
                    if len(sents) >= 2:
                        cal_neg_cases.append((str(c["claim"]), sents, None))
        verifier.prewarm_conflict_cases(cal_neg_cases)

        open_cal_probs: list[np.ndarray] = []
        open_cal_labels: list[str] = []
        cand_cal_probs: list[np.ndarray] = []
        cand_cal_labels: list[str] = []
        neg_conflict_scores: list[float] = []

        for c in calib_claims:
            g_lab = _gold_stance(c)
            c_str = str(c["claim"])
            ob = verifier._open_claim_bundle(c_str, doc_top_m=5, nli_top_s=5)
            op, _ = verifier.open_stance_head.predict_proba_from_features(ob["feat"])
            open_cal_probs.append(op)
            open_cal_labels.append(g_lab)

            ev = c.get("evidence") or {}
            if ev:
                for raw_did, sets in ev.items():
                    d_key = verifier._resolve_doc_key(raw_did)
                    if d_key is None:
                        continue
                    p_labs = {
                        str(e.get("label", "")).upper()
                        for e in (sets if isinstance(sets, Sequence) else ())
                        if isinstance(e, Mapping)
                    }
                    p_lab = "CONTRADICT" if "CONTRADICT" in p_labs else "SUPPORT"
                    doc = corpus[d_key]
                    sents = [str(s) for s in doc.get("abstract", ())]
                    cb = verifier._candidate_sentences_bundle(
                        c_str, sents, title=str(doc.get("title", ""))
                    )
                    cp, _ = verifier.cand_stance_head.predict_proba_from_features(cb["feat"])
                    cand_cal_probs.append(cp)
                    cand_cal_labels.append(p_lab)
                    if len(sents) >= 2:
                        neg_conflict_scores.append(
                            verifier.score_bilateral_conflict(c_str, sents)["conflict_score"]
                        )
            else:
                for raw_did in (c.get("cited_doc_ids") or ()):
                    d_key = verifier._resolve_doc_key(raw_did)
                    if d_key is not None:
                        doc = corpus[d_key]
                        cb = verifier._candidate_sentences_bundle(
                            c_str,
                            list(doc.get("abstract", ())),
                            title=str(doc.get("title", "")),
                        )
                        cp, _ = verifier.cand_stance_head.predict_proba_from_features(cb["feat"])
                        cand_cal_probs.append(cp)
                        cand_cal_labels.append("NEI")

        open_cal_summary = (
            verifier.open_stance_head.calibrate_split_conformal(
                np.vstack(open_cal_probs), open_cal_labels, alpha=alpha
            )
            if open_cal_probs
            else {}
        )
        cand_cal_summary = (
            verifier.cand_stance_head.calibrate_split_conformal(
                np.vstack(cand_cal_probs), cand_cal_labels, alpha=alpha
            )
            if cand_cal_probs
            else {}
        )

        if len(neg_conflict_scores) >= 25:
            q95 = SplitConformalStanceClassifier.conformal_quantile(neg_conflict_scores, alpha=0.05)
            verifier.conflict_threshold = float(np.clip(max(0.35, q95), 0.35, 0.85))

        verifier.training_provenance = {
            "trained": True,
            "train_source": "scifact_train_only",
            "biencoder_model": verifier.biencoder_model,
            "nli_model": verifier.nli_model,
            "total_train_claims": len(train_claims),
            "train_fit_claims": len(fit_claims),
            "train_calib_claims": len(calib_claims),
            "excluded_eval_doc_count": len(verifier.excluded_doc_ids),
            "alpha": verifier.alpha,
            "open_conformal_calibration": open_cal_summary,
            "candidate_conformal_calibration": cand_cal_summary,
            "conflict_threshold": round(float(verifier.conflict_threshold), 6),
        }
        return verifier

    def score_bilateral_conflict(
        self,
        claim: str,
        passages_or_sentences: Sequence[str],
    ) -> dict[str, Any]:
        """Compute Bilateral Cross-Passage NLI Conflict Score:

        `conflict_score = max_{i != j} min(P_NLI(SUPPORT | u_i, claim), P_NLI(CONTRADICT | u_j, claim))`
        """
        raw_units = [str(u).strip() for u in passages_or_sentences if str(u).strip()]
        if len(raw_units) == 1:
            doc_parts = [d.strip() for d in _DOC_SPLIT_RE.split(raw_units[0]) if d.strip()]
            if len(doc_parts) >= 2:
                units = doc_parts
            else:
                units = [s.strip() for s in segment_protocol_sentences(raw_units[0]) if s.strip()]
        else:
            units = raw_units

        if len(units) < 2:
            return {
                "conflict_score": 0.0,
                "is_conflict": False,
                "max_support_prob": 0.0,
                "max_contradict_prob": 0.0,
                "n_units": len(units),
            }

        prepared_units: list[str] = []
        q_vec = self.encode_queries_cached([claim])[0]
        for u in units:
            sents = [s.strip() for s in segment_protocol_sentences(u) if s.strip()]
            if len(sents) <= 2:
                prepared_units.append(u)
            else:
                embs = self.encode_passages_cached(sents, max_length=160)
                sims = embs @ q_vec
                top2 = sorted(int(i) for i in np.argsort(-sims, kind="stable")[:2])
                prepared_units.append(" ".join(sents[i] for i in top2))

        nli = self.score_nli_pairs([(u, claim) for u in prepared_units], max_length=224)
        p_sup = nli[:, 0]
        p_con = nli[:, 1]

        best_score = 0.0
        n_u = len(prepared_units)
        for i in range(n_u):
            for j in range(n_u):
                if i == j:
                    continue
                cand = min(float(p_sup[i]), float(p_con[j]))
                if cand > best_score:
                    best_score = cand

        return {
            "conflict_score": float(best_score),
            "is_conflict": bool(best_score >= self.conflict_threshold),
            "max_support_prob": float(np.max(p_sup)),
            "max_contradict_prob": float(np.max(p_con)),
            "n_units": n_u,
        }

    def verify_candidate_sentences(
        self,
        claim: str,
        sentences: Sequence[str],
        *,
        title: str = "",
        top_k: int = 3,
        alpha: float | None = None,
    ) -> ConformalStancePrediction:
        """Score candidate sentences with BGE + DeBERTa-v3 NLI and return calibrated conformal set."""
        cb = self._candidate_sentences_bundle(claim, sentences, title=title)
        probs, _ = self.cand_stance_head.predict_proba_from_features(cb["feat"])
        pset = self.cand_stance_head.prediction_set_from_probs(probs, alpha_override=alpha)
        pred_lab = STANCE_LABELS[int(np.argmax(probs))]
        ranked = cb["ranked_indices"][: max(1, min(int(top_k), len(cb["ranked_indices"])))]
        r_scores = [
            float(cb["rel_scores"][i]) if i < len(cb["rel_scores"]) else 0.0
            for i in range(len(ranked))
        ]
        # Reuse already-scored sentence NLI probabilities to evaluate cross-sentence conflict with 0 extra passes
        nli_mat = cb["nli_probs"]
        best_conf = 0.0
        if nli_mat.shape[0] >= 2:
            for i in range(nli_mat.shape[0]):
                for j in range(nli_mat.shape[0]):
                    if i != j:
                        best_conf = max(best_conf, min(float(nli_mat[i, 0]), float(nli_mat[j, 1])))
        return ConformalStancePrediction(
            predicted_label=pred_lab,
            probabilities={STANCE_LABELS[i]: float(probs[i]) for i in range(3)},
            conformal_set=pset,
            alpha=float(alpha if alpha is not None else self.cand_stance_head.alpha),
            is_contested_conflict=bool(best_conf >= self.conflict_threshold),
            top_rationale_indices=ranked,
            top_rationale_scores=r_scores,
        )

    def score_mcq_options(
        self,
        question: str,
        options: Sequence[str],
        sentences: Sequence[str],
    ) -> list[float]:
        """Score MCQ options via DeBERTa-v3 NLI entailment margin over passage sentences."""
        clean_sents = [str(s).strip() for s in sentences if str(s).strip()]
        if not clean_sents or not options:
            return [0.0] * len(options)

        q_vec = self.encode_queries_cached([question])[0]
        if len(clean_sents) > 3:
            s_embs = self.encode_passages_cached(clean_sents, max_length=160)
            top_idx = sorted(int(i) for i in np.argsort(-(s_embs @ q_vec), kind="stable")[:3])
            premise = " ".join(clean_sents[i] for i in top_idx)
        else:
            premise = " ".join(clean_sents)

        hyps = [
            f"{question} Proposed answer: {opt}" if question.endswith("?") else f"{question.rstrip('.')} -- {opt}"
            for opt in options
        ]
        nli_probs = self.score_nli_pairs([(premise, h) for h in hyps], max_length=224)
        opt_embs = self.encode_passages_cached(list(options), max_length=64)
        prem_emb = self.encode_passages_cached([premise], max_length=224)[0]
        dense_sim = np.clip(opt_embs @ prem_emb, 0.0, 1.0).astype(np.float64)

        scores = (nli_probs[:, 0] - 0.35 * nli_probs[:, 1]) + 0.25 * dense_sim
        return [float(x) for x in scores]

    def evaluate_scifact_split(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 3,
        doc_top_m: int = 5,
    ) -> dict[str, Any]:
        """Evaluate oracle ranking, open-corpus RRF, official SciFact F1, and Mondrian conformal sets."""
        ev_claims = [c for c in claims if c.get("evidence")]
        pairs = build_pairs(ev_claims, self.corpus, strict=True)
        noinfo_pairs = build_noinfo_pairs(claims, self.corpus, strict=False)

        self.prewarm_scifact_claims(claims, doc_top_m=doc_top_m, nli_top_s=8)

        # Also batch-prewarm candidate-doc NLI pairs needed for official SciFact evaluation
        off_pairs_to_warm: list[tuple[str, str]] = []
        for c in claims:
            c_str = str(c["claim"])
            q_vec = self._query_emb_cache[c_str]
            for did in self.retrieve_stage1_docs(c_str, top_m=3):
                lo, hi = self.doc_sent_spans[did]
                if lo >= hi:
                    continue
                ctx_d = [self.all_sent_contextual[g] for g in range(lo, hi)]
                d_embs = self.encode_passages_cached(ctx_d, max_length=160)
                d_cos = np.clip(d_embs @ q_vec, -1.0, 1.0)
                d_bm25 = self.sent_bm25.score_indexed_normalized(c_str, list(range(lo, hi)))
                if float(np.max(d_cos)) < 0.56 and float(np.max(d_bm25)) < 0.22:
                    continue
                for ct in ctx_d:
                    off_pairs_to_warm.append((ct, c_str))
        if off_pairs_to_warm:
            self.score_nli_pairs(off_pairs_to_warm, max_length=192)

        bge_ranks: list[list[int]] = []
        neural_ranks: list[list[int]] = []
        for p in pairs:
            did = self._resolve_doc_key(p["doc_id"])
            assert did is not None
            lo, hi = self.doc_sent_spans[did]
            gids = list(range(lo, hi))
            q_vec = self._query_emb_cache[p["claim"]]
            ctx_p = [self.all_sent_contextual[g] for g in gids]
            d_cos = np.clip(self.encode_passages_cached(ctx_p) @ q_vec, -1.0, 1.0).astype(np.float64)
            bge_ranks.append(rank_scores(d_cos))

            nli_p = self.score_nli_pairs([(ct, p["claim"]) for ct in ctx_p], max_length=192)
            bm25_n = self.sent_bm25.score_indexed_normalized(p["claim"], gids)
            comb = 0.55 * (1.0 - nli_p[:, 2]) + 0.30 * np.maximum(0.0, d_cos) + 0.15 * bm25_n
            neural_ranks.append(rank_scores(comb))

        y_true_all: list[str] = []
        y_pred_neural: list[str] = []
        conformal_covered: list[bool] = []
        conformal_sizes: list[int] = []
        conformal_by_class: dict[str, list[bool]] = {lab: [] for lab in STANCE_LABELS}

        open_doc_hits: list[float] = []
        bge_doc_hits: list[float] = []
        open_sent_h1: list[float] = []
        open_sent_h3: list[float] = []
        open_sent_mrr: list[float] = []
        ev_max_rel: list[float] = []
        ni_max_rel: list[float] = []

        official_preds: dict[str, dict[str, dict[str, Any]]] = {}

        for c in claims:
            cid = str(c.get("id"))
            c_str = str(c["claim"])
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

            q_vec = self._query_emb_cache[c_str]
            if ev:
                gold_docs = {str(did) for did in ev.keys()}
                bge_top = {
                    str(self.doc_ids[int(i)])
                    for i in np.argsort(-(self.doc_embeddings @ q_vec), kind="stable")[:doc_top_m]
                }
                bge_doc_hits.append(len(gold_docs & bge_top) / len(gold_docs))

            ob = self._open_claim_bundle(c_str, doc_top_m=doc_top_m, nli_top_s=8)
            probs, _ = self.open_stance_head.predict_proba_from_features(ob["feat"])
            pset = self.open_stance_head.prediction_set_from_probs(probs)
            pred_lab = STANCE_LABELS[int(np.argmax(probs))]
            y_pred_neural.append(pred_lab)

            is_cov = gold_lab in pset
            conformal_covered.append(is_cov)
            conformal_by_class[gold_lab].append(is_cov)
            conformal_sizes.append(len(pset))

            doc_pred_map: dict[str, dict[str, Any]] = {}
            for did in ob["top_docs"][:3]:
                lo, hi = self.doc_sent_spans[did]
                if lo >= hi:
                    continue
                d_gids = list(range(lo, hi))
                ctx_d = [self.all_sent_contextual[g] for g in d_gids]
                d_cos = np.clip(self.encode_passages_cached(ctx_d) @ q_vec, -1.0, 1.0).astype(
                    np.float64
                )
                d_bm25 = self.sent_bm25.score_indexed_normalized(c_str, d_gids)
                if float(np.max(d_cos)) < 0.56 and float(np.max(d_bm25)) < 0.22:
                    continue
                nli_doc = self.score_nli_pairs([(ct, c_str) for ct in ctx_d], max_length=192)
                s_sup = nli_doc[:, 0]
                s_con = nli_doc[:, 1]
                s_non_nei = 1.0 - nli_doc[:, 2]
                s_rel = 0.65 * s_non_nei + 0.25 * np.maximum(0.0, d_cos) + 0.10 * d_bm25
                best_s = int(np.argmax(s_rel))
                max_sup = float(np.max(s_sup))
                max_con = float(np.max(s_con))
                if max(max_sup, max_con) < 0.54 or float(s_rel[best_s]) < 0.50:
                    continue
                doc_lab = "CONTRADICT" if max_con > max_sup else "SUPPORT"
                target_probs = s_con if doc_lab == "CONTRADICT" else s_sup
                sel_sents = [
                    int(idx)
                    for idx in np.argsort(-target_probs, kind="stable")[:top_k]
                    if float(target_probs[int(idx)]) >= 0.45
                ]
                if not sel_sents:
                    sel_sents = [int(np.argmax(target_probs))]
                doc_pred_map[str(did)] = {"label": doc_lab, "sentences": sel_sents[:3]}
            official_preds[cid] = doc_pred_map

            max_rel = float(ob["rel_scores"][0]) if ob["rel_scores"].size > 0 else 0.0
            if ev:
                ev_max_rel.append(max_rel)
                gold_docs = {str(did) for did in ev.keys()}
                pred_docs = {str(did) for did in ob["top_doc_ids" if "top_doc_ids" in ob else "top_docs"]}
                open_doc_hits.append(len(gold_docs & pred_docs) / len(gold_docs))

                gold_sent_pairs = {
                    (str(did), int(si))
                    for did, sets in ev.items()
                    for s in sets
                    for si in s.get("sentences", ())
                }
                pred_pairs = [
                    (str(self.sent_to_doc[g][0]), int(self.sent_to_doc[g][1]))
                    for g in ob["ranked_gids"]
                ]
                open_sent_h1.append(float(bool(set(pred_pairs[:1]) & gold_sent_pairs)))
                open_sent_h3.append(float(bool(set(pred_pairs[:top_k]) & gold_sent_pairs)))
                rr_val = 0.0
                for rk, sp in enumerate(pred_pairs, start=1):
                    if sp in gold_sent_pairs:
                        rr_val = 1.0 / float(rk)
                        break
                open_sent_mrr.append(rr_val)
            else:
                ni_max_rel.append(max_rel)

        def _pack_rank(ranks: Sequence[Sequence[int]]) -> dict[str, float]:
            return {
                "hit@1": hit_at(ranks, pairs, 1),
                f"hit@{top_k}": hit_at(ranks, pairs, top_k),
                "mrr": mean_reciprocal_rank(ranks, pairs),
                f"sentence_recall@{top_k}": sentence_recall_at(ranks, pairs, top_k),
                f"rationale_set_recall@{top_k}": rationale_set_recall_at(ranks, pairs, top_k),
            }

        maj = majority_class_stance_baseline(y_true_all, eval_labels=STANCE_LABELS)
        cov_ci_by_class = {
            lab: {
                "n": len(vals),
                "coverage": float(np.mean(vals)) if vals else float("nan"),
                "wilson_95_ci": list(wilson_score_interval(int(sum(vals)), len(vals))),
            }
            for lab, vals in conformal_by_class.items()
        }
        official_metrics = compute_official_scifact_metrics(claims, official_preds, max_sentences_per_doc=3)

        return {
            "total_claims": len(claims),
            "noinfo_cited_pairs": len(noinfo_pairs),
            "biencoder_model": self.biencoder_model,
            "nli_model": self.nli_model,
            "oracle_rationale_ranking": {
                "bge_biencoder": _pack_rank(bge_ranks),
                "neural_nli_reranker": _pack_rank(neural_ranks),
            },
            "open_corpus_retrieval": {
                f"bge_doc_recall@{doc_top_m}": float(np.mean(bge_doc_hits)) if bge_doc_hits else float("nan"),
                f"neural_hybrid_doc_recall@{doc_top_m}": float(np.mean(open_doc_hits)) if open_doc_hits else float("nan"),
                "neural_nli_sentence_hit@1": float(np.mean(open_sent_h1)) if open_sent_h1 else float("nan"),
                f"neural_nli_sentence_hit@{top_k}": float(np.mean(open_sent_h3)) if open_sent_h3 else float("nan"),
                "neural_nli_sentence_mrr": float(np.mean(open_sent_mrr)) if open_sent_mrr else float("nan"),
                "neural_auroc_vs_nei": compute_auroc(ev_max_rel, ni_max_rel),
            },
            "official_scifact_metrics": official_metrics,
            "stance_verification_3way": {
                **maj,
                "neural_nli_macro_f1": macro_f1_score(y_true_all, y_pred_neural, labels=STANCE_LABELS),
                "neural_nli_accuracy": (
                    float(np.mean([yt == yp for yt, yp in zip(y_true_all, y_pred_neural)]))
                    if y_true_all
                    else float("nan")
                ),
            },
            "split_conformal_90": {
                "target_coverage": 1.0 - self.open_stance_head.alpha,
                "alpha": self.open_stance_head.alpha,
                "empirical_coverage": float(np.mean(conformal_covered)) if conformal_covered else float("nan"),
                "empirical_wilson_95_ci": list(
                    wilson_score_interval(int(sum(conformal_covered)), len(conformal_covered))
                ),
                "avg_set_size": float(np.mean(conformal_sizes)) if conformal_sizes else float("nan"),
                "single_label_set_rate": (
                    float(sum(1 for s in conformal_sizes if s == 1)) / float(len(conformal_sizes))
                    if conformal_sizes
                    else 0.0
                ),
                "multi_label_set_rate": (
                    float(sum(1 for s in conformal_sizes if s >= 2)) / float(len(conformal_sizes))
                    if conformal_sizes
                    else 0.0
                ),
                "coverage_by_class": {
                    lab: (float(np.mean(vals)) if vals else float("nan"))
                    for lab, vals in conformal_by_class.items()
                },
                "coverage_by_class_detail": cov_ci_by_class,
            },
            "training_provenance": dict(self.training_provenance),
        }
