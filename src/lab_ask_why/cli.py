"""Command-line interface and SciFact benchmark runner for lab-ask-why."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from lab_ask_why.bm25 import BM25
from lab_ask_why.data import (
    DEFAULT_ROOT,
    OFFLINE_MINI_CLAIMS,
    OFFLINE_MINI_CORPUS,
    build_noinfo_pairs,
    build_pairs,
    configure_utf8_stdout,
    ensure_data,
    read_jsonl,
)
from lab_ask_why.labbench_adapter import load_labbench_items, probe_labbench_item
from lab_ask_why.metrics import (
    compute_auroc,
    conclusion_first_rankings,
    hit_at,
    lead_rankings,
    rank_scores,
    rationale_set_recall_at,
    sentence_recall_at,
)
from lab_ask_why.retrieval import TwoStageRetriever
from lab_ask_why.schema import EpistemicProbeResult, export_probes_jsonl
from lab_ask_why.socratic import why_questions


def evaluate_scifact_benchmark(
    corpus: Mapping[Any, Mapping[str, Any]],
    all_claims: Sequence[Mapping[str, Any]],
    *,
    idf_scope: str = "full",
    top_k: int = 3,
    doc_top_m: int = 5,
    tie_seed: int | None = None,
) -> dict[str, Any]:
    """Run full oracle-abstract, positional-baseline, NOINFO-AUROC, and open-corpus evaluation."""
    evidence_claims = [c for c in all_claims if c.get("evidence")]
    pairs = build_pairs(evidence_claims, corpus, strict=True)
    noinfo_pairs = build_noinfo_pairs(all_claims, corpus, strict=False)

    if idf_scope == "background":
        eval_doc_ids = {str(p["doc_id"]) for p in pairs} | {str(p["doc_id"]) for p in noinfo_pairs}
        has_bg = any(str(did) not in eval_doc_ids for did in corpus.keys())
        bg_exclude_ids: set[str] | None = eval_doc_ids if has_bg else None
        bm25, _ = BM25.from_corpus(
            corpus,
            exclude_doc_ids=bg_exclude_ids,
        )
    elif idf_scope == "full":
        bg_exclude_ids = None
        bm25, _ = BM25.from_corpus(corpus, exclude_doc_ids=None)
    else:
        raise ValueError(f"Unsupported idf_scope: {idf_scope!r}")

    bm25_scores_list = [bm25.scores(p["claim"], p["sents"]) for p in pairs]
    bm25_rank = [
        rank_scores(sc, tie_seed=(None if tie_seed is None else tie_seed + idx))
        for idx, sc in enumerate(bm25_scores_list)
    ]

    lead_rank = lead_rankings(pairs)
    tail_rank = conclusion_first_rankings(pairs)

    ks = sorted({1, int(top_k)})
    rand: dict[int, list[float]] = {k: [] for k in ks}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ranks = [list(rng.permutation(len(p["sents"]))) for p in pairs]
        for k in ks:
            rand[k].append(hit_at(ranks, pairs, k))

    # NOINFO abstention separation (max-sentence BM25 score on SUPPORT/CONTRADICT vs NOINFO cited docs)
    pos_max = [float(np.max(sc)) if sc.size else 0.0 for sc in bm25_scores_list]
    neg_max = [
        float(np.max(bm25.scores(p["claim"], p["sents"]))) if p["sents"] else 0.0
        for p in noinfo_pairs
    ]
    noinfo_auroc = compute_auroc(pos_max, neg_max)

    # Open-corpus two-stage retrieval across all corpus abstracts and all claims
    open_retriever = TwoStageRetriever(corpus, exclude_doc_ids=bg_exclude_ids)
    open_metrics = open_retriever.evaluate_claims(
        all_claims,
        doc_top_m=doc_top_m,
        sent_top_k=top_k,
    )

    by_label: dict[str, dict[str, Any]] = {}
    for lab in ("SUPPORT", "CONTRADICT"):
        idx = [i for i, p in enumerate(pairs) if p["label"] == lab]
        sub_p = [pairs[i] for i in idx]
        sub_r = [bm25_rank[i] for i in idx]
        by_label[lab] = {
            "n": len(idx),
            "hit@1": hit_at(sub_r, sub_p, 1),
            f"hit@{top_k}": hit_at(sub_r, sub_p, top_k),
            "sentence_recall@1": sentence_recall_at(sub_r, sub_p, 1),
            f"sentence_recall@{top_k}": sentence_recall_at(sub_r, sub_p, top_k),
            "rationale_set_recall@1": rationale_set_recall_at(sub_r, sub_p, 1),
            f"rationale_set_recall@{top_k}": rationale_set_recall_at(sub_r, sub_p, top_k),
        }

    examples: list[dict[str, Any]] = []
    for lab in ("SUPPORT", "CONTRADICT"):
        ex_idx = next((i for i, p in enumerate(pairs) if p["label"] == lab), None)
        if ex_idx is None:
            continue
        p = pairs[ex_idx]
        r = bm25_rank[ex_idx]
        top_j = r[:top_k]
        top_sents = [p["sents"][j] for j in top_j]
        examples.append(
            {
                "claim": p["claim"],
                "title": p["title"],
                "label": lab,
                "gold_sentences": sorted(p["gold"]),
                "top_sentences": [
                    {
                        "rank": rank_num,
                        "sent_idx": j,
                        "is_gold": j in p["gold"],
                        "sentence": p["sents"][j],
                    }
                    for rank_num, j in enumerate(top_j, 1)
                ],
                "why_questions": why_questions(p["claim"], top_sents, stance=lab),
            }
        )

    n_sents = [len(p["sents"]) for p in pairs]
    labels = Counter(p["label"] for p in pairs)
    return {
        "num_corpus_docs": len(corpus),
        "total_claims": len(all_claims),
        "evidence_claims": len(evidence_claims),
        "noinfo_claims": len(all_claims) - len(evidence_claims),
        "noinfo_pairs": len(noinfo_pairs),
        "claim_doc_pairs": len(pairs),
        "labels": dict(labels),
        "sentences_mean": float(np.mean(n_sents)) if n_sents else 0.0,
        "sentences_median": float(np.median(n_sents)) if n_sents else 0.0,
        "idf_scope": idf_scope,
        "top_k": top_k,
        "doc_top_m": doc_top_m,
        "oracle_metrics": {
            "bm25": {
                "hit@1": hit_at(bm25_rank, pairs, 1),
                f"hit@{top_k}": hit_at(bm25_rank, pairs, top_k),
                "sentence_recall@1": sentence_recall_at(bm25_rank, pairs, 1),
                f"sentence_recall@{top_k}": sentence_recall_at(bm25_rank, pairs, top_k),
                "rationale_set_recall@1": rationale_set_recall_at(bm25_rank, pairs, 1),
                f"rationale_set_recall@{top_k}": rationale_set_recall_at(bm25_rank, pairs, top_k),
            },
            "lead_k": {
                "hit@1": hit_at(lead_rank, pairs, 1),
                f"hit@{top_k}": hit_at(lead_rank, pairs, top_k),
            },
            "conclusion_first": {
                "hit@1": hit_at(tail_rank, pairs, 1),
                f"hit@{top_k}": hit_at(tail_rank, pairs, top_k),
            },
            "random_20_seeds": {
                "hit@1_mean": float(np.mean(rand[1])) if rand[1] else float("nan"),
                f"hit@{top_k}_mean": float(np.mean(rand[top_k])) if rand[top_k] else float("nan"),
                "hit@1_std": float(np.std(rand[1])) if rand[1] else float("nan"),
                f"hit@{top_k}_std": float(np.std(rand[top_k])) if rand[top_k] else float("nan"),
            },
            "by_label": by_label,
        },
        "noinfo_abstention": {
            "evidence_max_bm25_mean": float(np.mean(pos_max)) if pos_max else float("nan"),
            "noinfo_max_bm25_mean": float(np.mean(neg_max)) if neg_max else float("nan"),
            "auroc": noinfo_auroc,
        },
        "open_corpus": open_metrics,
        "examples": examples,
    }


def format_benchmark_text(summary: Mapping[str, Any]) -> str:
    """Format benchmark dictionary into deterministic human-readable console report."""
    k = int(summary["top_k"])
    m = int(summary["doc_top_m"])
    om = summary["oracle_metrics"]
    bm = om["bm25"]
    lead = om["lead_k"]
    tail = om["conclusion_first"]
    rnd = om["random_20_seeds"]
    ni = summary["noinfo_abstention"]
    oc = summary["open_corpus"]

    def _fmt(v: float) -> str:
        return "   n/a" if np.isnan(v) else f"{v:6.3f}"

    lines: list[str] = []
    lines.append("SciFact dev: rationale sentence retrieval & open-corpus epistemic probing")
    lines.append(
        f"total claims: {summary['total_claims']}   with evidence: {summary['evidence_claims']}   "
        f"NOINFO: {summary['noinfo_claims']}   claim-doc pairs: {summary['claim_doc_pairs']}   "
        f"labels: {summary['labels']}"
    )
    lines.append(
        f"sentences per abstract: mean {summary['sentences_mean']:.1f}, "
        f"median {summary['sentences_median']:.0f}   (IDF scope: {summary['idf_scope']})"
    )
    lines.append("")
    lines.append(
        f"{'method':<22}{'hit@1':>10}{f'hit@{k}':>10}"
        f"{'sent_rec@1':>12}{f'sent_rec@{k}':>12}"
        f"{'set_rec@1':>11}{f'set_rec@{k}':>11}"
    )
    lines.append(
        f"{'BM25':<22}{_fmt(bm['hit@1']):>10}{_fmt(bm[f'hit@{k}']):>10}"
        f"{_fmt(bm['sentence_recall@1']):>12}{_fmt(bm[f'sentence_recall@{k}']):>12}"
        f"{_fmt(bm['rationale_set_recall@1']):>11}{_fmt(bm[f'rationale_set_recall@{k}']):>11}"
    )
    lines.append(f"{'lead-k (position)':<22}{_fmt(lead['hit@1']):>10}{_fmt(lead[f'hit@{k}']):>10}")
    lines.append(f"{'conclusion-first':<22}{_fmt(tail['hit@1']):>10}{_fmt(tail[f'hit@{k}']):>10}")
    lines.append(
        f"{'random (20 seeds)':<22}{_fmt(rnd['hit@1_mean']):>10}{_fmt(rnd[f'hit@{k}_mean']):>10}"
    )
    lines.append(
        f"{'  random std':<22}{_fmt(rnd['hit@1_std']):>10}{_fmt(rnd[f'hit@{k}_std']):>10}"
    )
    for lab in ("SUPPORT", "CONTRADICT"):
        sub = om["by_label"][lab]
        if sub["n"] == 0:
            continue
        lines.append(
            f"{'BM25 ' + lab:<22}{_fmt(sub['hit@1']):>10}{_fmt(sub[f'hit@{k}']):>10}"
            f"{_fmt(sub['sentence_recall@1']):>12}{_fmt(sub[f'sentence_recall@{k}']):>12}"
            f"{_fmt(sub['rationale_set_recall@1']):>11}{_fmt(sub[f'rationale_set_recall@{k}']):>11}"
            f"   (n={sub['n']})"
        )

    lines.append("")
    lines.append(
        f"Epistemic abstention (SUPPORT/CONTRADICT vs NOINFO cited abstracts, n_noinfo_pairs={summary['noinfo_pairs']}): "
        f"max-BM25 mean {ni['evidence_max_bm25_mean']:.2f} vs {ni['noinfo_max_bm25_mean']:.2f} | "
        f"AUROC = {_fmt(ni['auroc']).strip()}"
    )
    lines.append(
        f"Open-corpus 2-stage BM25 (all {summary['num_corpus_docs']:,} abstracts, top-{m} docs -> top-{k} sents): "
        f"doc_recall@{m} = {_fmt(oc[f'doc_recall@{m}']).strip()} | "
        f"sentence_hit@{k} = {_fmt(oc[f'sentence_hit@{k}']).strip()} | "
        f"NOINFO FDR = {_fmt(oc['noinfo_fdr']).strip()}"
    )

    for ex in summary["examples"]:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Claim: {ex['claim']}")
        lines.append(f"Abstract: {ex['title']}")
        lines.append(f"Gold label: {ex['label']}   gold rationale sentences: {ex['gold_sentences']}")
        lines.append(f"Top-{k} BM25 sentences:")
        for item in ex["top_sentences"]:
            mark = "*" if item["is_gold"] else " "
            lines.append(f"  {item['rank']}.{mark}[s{item['sent_idx']}] {item['sentence']}")
        lines.append("Grounded why questions:")
        for q in ex["why_questions"]:
            lines.append(f"  - {q}")

    lines.append("")
    lines.append("(* = gold rationale sentence)")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-ask-why",
        description="Epistemic Socratic co-scientist: rationale retrieval & discriminating test probes.",
    )
    parser.add_argument("--claim", type=str, default=None, help="Probe a single scientific claim.")
    parser.add_argument("--corpus", type=str, default=None, help="Path to custom corpus JSONL file.")
    parser.add_argument("--claims-file", type=str, default=None, help="Path to custom claims JSONL file.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT, help="Directory for SciFact cache.")
    parser.add_argument("--top-k", "--k", dest="top_k", type=int, default=3, help="Top-k sentences to retrieve.")
    parser.add_argument("--doc-top-m", type=int, default=5, help="Top-M abstracts in open-corpus stage 1.")
    parser.add_argument(
        "--idf-scope",
        choices=("background", "full"),
        default="background",
        help="Fit BM25 IDF strictly on non-eval background documents ('background') or full corpus ('full').",
    )
    parser.add_argument("--tie-seed", type=int, default=None, help="Optional RNG seed for 0-score tie breaking.")
    parser.add_argument("--offline", action="store_true", help="Disable network downloads; use local/mini fixtures.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format ('text' or machine-readable 'json').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Shorthand alias for --format json.",
    )
    parser.add_argument(
        "--export-probes",
        type=str,
        default=None,
        help="Write structured EpistemicProbeResult JSONL records to this file.",
    )
    parser.add_argument(
        "--export-schema",
        type=str,
        default=None,
        help="Write EpistemicProbeResult JSON Schema to this file and exit.",
    )
    parser.add_argument(
        "--labbench-jsonl",
        type=str,
        default=None,
        help="Evaluate rationale probing on a LAB-Bench LitQA2 or ProtocolQA JSONL file.",
    )
    return parser


def _load_corpus_and_claims(args: argparse.Namespace) -> tuple[dict[Any, dict[str, Any]], list[dict[str, Any]]]:
    if args.corpus:
        if not os.path.exists(args.corpus):
            raise FileNotFoundError(f"Corpus file not found: {args.corpus}")
        corpus_rows = read_jsonl(args.corpus)
        corpus = {d["doc_id"]: d for d in corpus_rows}
        if args.claims_file:
            if not os.path.exists(args.claims_file):
                raise FileNotFoundError(f"Claims file not found: {args.claims_file}")
            claims = read_jsonl(args.claims_file)
        else:
            claims = []
        return corpus, claims

    if args.offline:
        data_dir = os.path.join(args.data_root, "data")
        archive_path = os.path.join(args.data_root, "data.tar.gz")
        has_local = (
            os.path.exists(os.path.join(data_dir, "corpus.jsonl"))
            and os.path.exists(os.path.join(data_dir, "claims_dev.jsonl"))
        ) or os.path.exists(archive_path)
        if not has_local:
            corpus = {d["doc_id"]: d for d in OFFLINE_MINI_CORPUS}
            claims = (
                read_jsonl(args.claims_file)
                if args.claims_file
                else list(OFFLINE_MINI_CLAIMS)
            )
            return corpus, claims

    data_dir = ensure_data(root=args.data_root, offline=args.offline)
    corpus = {d["doc_id"]: d for d in read_jsonl("corpus.jsonl", data_dir=data_dir)}
    claims_path = args.claims_file if args.claims_file else "claims_dev.jsonl"
    claims = read_jsonl(claims_path, data_dir=data_dir)
    return corpus, claims


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdout()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_fmt = "json" if args.json else args.format

    if args.top_k < 1 or args.doc_top_m < 1:
        print("Error: --top-k and --doc-top-m must be >= 1", file=sys.stderr)
        return 2

    try:
        if args.export_schema:
            schema = EpistemicProbeResult.json_schema()
            with open(args.export_schema, mode="w", encoding="utf-8") as fh:
                json.dump(schema, fh, indent=2)
            if out_fmt == "json":
                print(json.dumps(schema, indent=2))
            else:
                print(f"Wrote EpistemicProbeResult JSON Schema to {args.export_schema}")
            return 0

        if args.labbench_jsonl:
            items = load_labbench_items(args.labbench_jsonl, purpose="eval")
            probed = [probe_labbench_item(it, top_k=args.top_k) for it in items]
            win_rate = (
                float(np.mean([p["ideal_vs_distractor_margin"] > 0.0 for p in probed]))
                if probed
                else 0.0
            )
            payload = {
                "items_evaluated": len(probed),
                "ideal_margin_win_rate": round(win_rate, 4),
                "samples": probed[:3],
            }
            if out_fmt == "json":
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(
                    f"LAB-Bench rationale probe: n={len(probed)} | "
                    f"ideal-over-distractor rationale margin rate = {win_rate:.3f}"
                )
            return 0

        corpus, claims = _load_corpus_and_claims(args)

        if args.claim is not None:
            retriever = TwoStageRetriever(corpus)
            probe = retriever.probe_claim(
                args.claim,
                doc_top_m=args.doc_top_m,
                sent_top_k=max(args.top_k, 4),
            )
            if args.export_probes:
                export_probes_jsonl([probe], args.export_probes)
            if out_fmt == "json":
                print(probe.to_json(indent=2))
            else:
                print(f"Claim: {probe.claim}")
                print(
                    f"Status: {probe.epistemic_status} "
                    f"(confidence={probe.retrieval_confidence:.3f})"
                )
                for r in (probe.supporting_rationales + probe.contradicting_rationales)[: args.top_k]:
                    print(f"  [{r.polarity} doc={r.doc_id} s{r.sent_idx} ({r.bm25_score:.2f})] {r.sentence}")
                print("Grounded why questions:")
                for q in probe.why_questions:
                    print(f"  - {q}")
            return 0

        if not claims:
            raise ValueError("No claims provided; pass --claim or --claims-file.")

        summary = evaluate_scifact_benchmark(
            corpus,
            claims,
            idf_scope=args.idf_scope,
            top_k=args.top_k,
            doc_top_m=args.doc_top_m,
            tie_seed=args.tie_seed,
        )

        if args.export_probes:
            retriever = TwoStageRetriever(corpus)
            probes = [
                retriever.probe_claim(
                    str(c["claim"]),
                    doc_top_m=args.doc_top_m,
                    sent_top_k=max(args.top_k, 4),
                )
                for c in claims[:25]
            ]
            export_probes_jsonl(probes, args.export_probes)

        if out_fmt == "json":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(format_benchmark_text(summary))
        return 0

    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
