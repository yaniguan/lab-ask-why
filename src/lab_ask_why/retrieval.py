"""Two-stage open-corpus BM25 retrieval and symmetric pro/con rationale probing."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from lab_ask_why.bm25 import BM25, tokenize
from lab_ask_why.metrics import compute_auroc
from lab_ask_why.schema import (
    DiscriminatingObservable,
    EpistemicProbeResult,
    RationaleEvidence,
)
from lab_ask_why.socratic import extract_differential_terms, why_questions

_NEGATION_TOKENS = frozenset(
    {"not", "no", "neither", "without", "never", "none", "lack", "lacks", "failed", "fails", "fail", "unable"}
)
_ANTONYM_GROUPS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({"increase", "increases", "increased", "raises", "raise", "elevates", "higher", "enhances", "promotes"}),
        frozenset({"decrease", "decreases", "decreased", "reduces", "reduce", "reduced", "lower", "inhibits", "inhibit", "suppresses", "abolishes"}),
    ),
    (
        frozenset({"vulnerability", "susceptibility", "risk", "harm", "toxicity", "severe", "mortality"}),
        frozenset({"protection", "protects", "protective", "resistance", "survival", "benefit", "safe"}),
    ),
    (
        frozenset({"sufficient", "necessary", "essential", "required", "causes"}),
        frozenset({"dispensable", "unnecessary", "independent", "unrelated", "benign"}),
    ),
)


def classify_rationale_polarity(claim: str, sentence: str) -> str:
    """Symmetric lexical polarity classifier separating supporting vs countervailing cues."""
    c_toks = set(tokenize(claim))
    s_toks = set(tokenize(sentence))
    if not s_toks:
        return "NEUTRAL"

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
    """Open-corpus two-stage BM25 retriever (abstract top-M -> sentence top-k)."""

    def __init__(
        self,
        corpus: Mapping[Any, Mapping[str, Any]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1.2,
        abstention_threshold: float = 8.0,
    ) -> None:
        if not corpus:
            raise ValueError("TwoStageRetriever requires a non-empty corpus")
        self.corpus = corpus
        self.abstention_threshold = float(abstention_threshold)

        self.doc_ids: list[Any] = []
        abstract_blobs: list[str] = []
        all_sentences: list[str] = []
        self.doc_sent_spans: dict[Any, tuple[int, int]] = {}
        self.sent_to_doc: list[tuple[Any, int]] = []

        for doc_id, doc in corpus.items():
            title = str(doc.get("title", ""))
            sents = [str(s) for s in doc.get("abstract", ())]
            self.doc_ids.append(doc_id)
            abstract_blobs.append((title + " " + " ".join(sents)).strip())
            start = len(all_sentences)
            for j, s in enumerate(sents):
                all_sentences.append(s)
                self.sent_to_doc.append((doc_id, j))
            self.doc_sent_spans[doc_id] = (start, len(all_sentences))

        self.all_sentences = all_sentences
        self.doc_bm25 = BM25(abstract_blobs, k1=k1, b=b, k3=k3)
        self.sent_bm25 = BM25(all_sentences, k1=k1, b=b, k3=k3)

    def retrieve(
        self,
        claim: str,
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve top-`doc_top_m` abstracts and rank their sentences to return `sent_top_k`."""
        doc_scores = self.doc_bm25.score_indexed(claim)
        m = max(1, min(int(doc_top_m), len(self.doc_ids)))
        top_doc_pos = np.argsort(-doc_scores, kind="stable")[:m]
        top_docs = [self.doc_ids[int(i)] for i in top_doc_pos]

        cand_global_ids: list[int] = []
        for did in top_docs:
            lo, hi = self.doc_sent_spans[did]
            cand_global_ids.extend(range(lo, hi))

        if not cand_global_ids:
            return {
                "top_doc_ids": top_docs,
                "top_sentences": [],
                "max_sentence_score": 0.0,
                "abstain": True,
            }

        sent_scores = self.sent_bm25.score_indexed(claim, cand_global_ids)
        k = max(1, min(int(sent_top_k), sent_scores.size))
        order = np.argsort(-sent_scores, kind="stable")[:k]

        ranked_sents: list[dict[str, Any]] = []
        for pos in order:
            gid = cand_global_ids[int(pos)]
            did, local_idx = self.sent_to_doc[gid]
            ranked_sents.append(
                {
                    "doc_id": did,
                    "sent_idx": local_idx,
                    "score": float(sent_scores[int(pos)]),
                    "sentence": self.all_sentences[gid],
                    "title": str(self.corpus[did].get("title", "")),
                }
            )

        max_score = float(np.max(sent_scores)) if sent_scores.size > 0 else 0.0
        return {
            "top_doc_ids": top_docs,
            "top_sentences": ranked_sents,
            "max_sentence_score": max_score,
            "abstain": max_score < self.abstention_threshold,
        }

    def evaluate_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        doc_top_m: int = 5,
        sent_top_k: int = 3,
    ) -> dict[str, float]:
        """Evaluate open-corpus retrieval on all claims (evidence-bearing + NOINFO)."""
        doc_hits: list[float] = []
        sent_hits: list[float] = []
        evidence_max_scores: list[float] = []
        noinfo_max_scores: list[float] = []

        for c in claims:
            res = self.retrieve(str(c["claim"]), doc_top_m=doc_top_m, sent_top_k=sent_top_k)
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
                evidence_max_scores.append(res["max_sentence_score"])
            else:
                noinfo_max_scores.append(res["max_sentence_score"])

        # Calibrate abstention threshold at median between NOINFO and evidence medians when both exist
        if evidence_max_scores and noinfo_max_scores:
            threshold = 0.5 * (
                float(np.median(evidence_max_scores)) + float(np.median(noinfo_max_scores))
            )
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
    ) -> EpistemicProbeResult:
        """Probe an arbitrary claim across the corpus and emit a typed EpistemicProbeResult."""
        res = self.retrieve(claim, doc_top_m=doc_top_m, sent_top_k=sent_top_k)
        max_score = float(res["max_sentence_score"])
        confidence = round(1.0 - math.exp(-max(0.0, max_score) / 10.0), 6)

        supporting: list[RationaleEvidence] = []
        contradicting: list[RationaleEvidence] = []
        raw_top_texts: list[str] = []

        for item in res["top_sentences"]:
            if item["score"] <= 0.0:
                continue
            pol = classify_rationale_polarity(claim, item["sentence"])
            ev = RationaleEvidence(
                doc_id=item["doc_id"],
                sent_idx=item["sent_idx"],
                sentence=item["sentence"],
                bm25_score=item["score"],
                polarity=pol,
                title=item["title"],
            )
            raw_top_texts.append(item["sentence"])
            if pol == "CONTRADICT":
                contradicting.append(ev)
            else:
                supporting.append(ev)

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
        diff_terms = extract_differential_terms(claim, raw_top_texts[:3], max_terms=4) if raw_top_texts else []

        assumptions = [
            f"Observed lexical overlap reflects a direct causal link rather than confounding via {', '.join(diff_terms[:2]) or 'background covariates'}.",
            f"Experimental regime in retrieved literature ({res['top_doc_ids'][:3]}) matches the target population/conditions of '{claim.strip().rstrip('.')}'.",
        ]
        observable = DiscriminatingObservable(
            observable_name=f"perturbation_readout_{diff_terms[0]}" if diff_terms else "direct_replication_assay",
            differential_terms=diff_terms,
            proposed_perturbation=questions[-1],
            candidate_descriptors={str(k): float(v) for k, v in (candidate_descriptors or {}).items()},
            candidate_smiles=candidate_smiles,
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
