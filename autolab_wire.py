"""Auto-Lab 15-repository wire node for lab-ask-why (Layer L6 Epistemic Brain).

Connects upstream protocol/sensory spans from ``lab-ears`` (L2) to symmetric
Okapi BM25 + subword rationale retrieval, Socratic ``why_questions`` framing, and
structured control-well synthesis before calibrating trust regions in ``lab-decide`` (L6),
with post-run causal attribution certificates closed through ``lab-attrib`` (L4).
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ID = "lab-ask-why"
LAYER = "L6"
SCHEMA_VERSION = "autolab-wire/v1"

REPO_DIR = Path(__file__).resolve().parent
_SRC_DIR = REPO_DIR / "src"
for _candidate_path in (str(_SRC_DIR), str(REPO_DIR)):
    if _candidate_path not in sys.path:
        sys.path.insert(0, _candidate_path)

from lab_ask_why.bm25 import BM25  # noqa: E402
from lab_ask_why.retrieval import classify_rationale_polarity  # noqa: E402
from lab_ask_why.schema import (  # noqa: E402
    DiscriminatingObservable,
    to_permille_floor,
)
from lab_ask_why.socratic import (  # noqa: E402
    extract_differential_terms,
    extract_protocol_descriptors,
    synthesize_control_wells,
    why_questions,
)

# Built-in 6-sentence offline wet-lab solubility & protocol mini-corpus
# (3 supporting + 3 countervailing/contradicting sentences; zero network needed).
DEFAULT_WETLAB_EVIDENCE_SENTENCES: tuple[str, ...] = (
    "Buffered aqueous incubation of 100 uL candidate solution at 25 C to 37 C for 15 min with gentle rpm agitation increases equilibrium small-molecule aspirin solubility and reproducible 96-well plate absorbance readout.",
    "Controlled microliter liquid transfer into pH-buffered cosolvent wells enhances dissolution kinetics and preserves parent ester assay stability across standard incubation windows.",
    "Calibrated UV spectrophotometric and chromatographic characterization confirms linear concentration response for dissolved salicylic and acetylsalicylic standards after protocol mixing.",
    "Prolonged thermal incubation at 37 C without matched vehicle blank controls fails to prevent ester hydrolysis of aspirin into salicylic acid and distorts apparent solubility measurements.",
    "Unfiltered 96-well optical turbidity and precursor ghost peaks do not distinguish true molecular dissolution from colloidal precipitation or residual starting material.",
    "Rapid high-volume dispense speeds without cosolvent equilibration do not achieve thermodynamic solubility equilibrium within a short 15 min plate reader assay.",
)

_DEFAULT_FALLBACK_CLAIM = (
    "Buffered aqueous incubation protocol achieves reproducible small-molecule aspirin solubility"
)
_DEFAULT_WETLAB_QUERY_ANCHOR = (
    "aqueous solubility aspirin incubation protocol transfer temperature stability"
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _canonicalize_attribution_outcome(raw_outcome: Any) -> str:
    """Normalize lab-attrib outcome strings (snake_case, camelCase, UPPER_SNAKE) to canonical form."""
    if not raw_outcome:
        return ""
    text = str(raw_outcome).strip()
    # Insert underscore before camelCase transitions
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).upper()
    snake = re.sub(r"[^A-Z0-9]+", "_", snake).strip("_")
    aliases = {
        "CERTIFIED_POSITIVE": "CERTIFIED_POSITIVE",
        "CERTIFIED_NEGATIVE": "CERTIFIED_NEGATIVE",
        "EXECUTION_ARTIFACT": "EXECUTION_ARTIFACT",
        "AMBIGUOUS_CHARACTERIZATION": "AMBIGUOUS_CHARACTERIZATION",
    }
    return aliases.get(snake, snake)


def _is_valid_optional_sha256(entry: Mapping[str, Any]) -> bool:
    """Validate optional 64-hex SHA-256 digest fields when supplied on a ledger record."""
    for key in ("sha256", "ledger_entry_sha256", "entry_sha256", "entry_hash"):
        if key in entry and entry[key] is not None:
            digest = str(entry[key]).strip()
            if not _SHA256_HEX_RE.match(digest):
                return False
    return True


def _extract_span_context(parsed_spans: Mapping[str, Any]) -> str:
    """Summarize protocol spans handed off from lab-ears for BM25 query grounding."""
    parts: list[str] = []
    for list_key in ("amounts", "temperatures", "times", "speeds"):
        raw_items = parsed_spans.get(list_key)
        if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
            for item in raw_items:
                if isinstance(item, Mapping):
                    text = str(item.get("text") or item.get("value") or "").strip()
                else:
                    text = str(item).strip()
                if text:
                    parts.append(text)

    vol = parsed_spans.get("volume_ul")
    if isinstance(vol, (int, float)) and not isinstance(vol, bool):
        parts.append(f"{float(vol):g} uL")
    temp = parsed_spans.get("temp_c")
    if isinstance(temp, (int, float)) and not isinstance(temp, bool):
        parts.append(f"{float(temp):g} C")
    dur = parsed_spans.get("duration_min")
    if isinstance(dur, (int, float)) and not isinstance(dur, bool):
        parts.append(f"{float(dur):g} min")

    return " ".join(parts)


def _resolve_claim_text(goal_text: str, parsed_spans: Mapping[str, Any]) -> str:
    """Ensure a non-empty alphanumeric scientific claim string for Socratic probing."""
    cleaned = goal_text.strip()
    if cleaned and any(ch.isalnum() for ch in cleaned):
        return cleaned

    span_summary = _extract_span_context(parsed_spans)
    if span_summary and any(ch.isalnum() for ch in span_summary):
        return f"Wet-lab protocol ({span_summary}) achieves reproducible aqueous solubility"

    return _DEFAULT_FALLBACK_CLAIM


def _normalize_corpus_entries(
    raw_sentences: Sequence[Any] | None,
) -> tuple[list[str], list[str | None], bool]:
    """Normalize evidence sentences into (texts, explicit_polarities, used_builtin)."""
    if raw_sentences is None:
        return (
            list(DEFAULT_WETLAB_EVIDENCE_SENTENCES),
            [None] * len(DEFAULT_WETLAB_EVIDENCE_SENTENCES),
            True,
        )

    texts: list[str] = []
    explicit_polarities: list[str | None] = []
    for entry in raw_sentences:
        if isinstance(entry, Mapping):
            text = str(entry.get("sentence") or entry.get("text") or "").strip()
            pol_raw = str(entry.get("polarity") or "").upper().strip()
            pol = pol_raw if pol_raw in {"SUPPORT", "CONTRADICT", "NEUTRAL"} else None
        else:
            text = str(entry).strip()
            pol = None
        if text:
            texts.append(text)
            explicit_polarities.append(pol)
    return texts, explicit_polarities, False


def _ingest_attribution_feedback(
    packet: Mapping[str, Any],
    claim_text: str,
) -> tuple[list[tuple[str, str, str]], dict[str, int], bool]:
    """Ingest verified lab-attrib certificates and isolate execution/characterization faults.

    Returns ``(admitted_entries, feedback_counts, ambiguous_characterization)``, where each
    admitted entry is ``(display_sentence, scoring_text, explicit_polarity)``.
    """
    raw_entries: list[Mapping[str, Any]] = []
    ledger_seq = packet.get("certified_ledger_entries")
    if isinstance(ledger_seq, Sequence) and not isinstance(ledger_seq, (str, bytes)):
        for item in ledger_seq:
            if isinstance(item, Mapping):
                raw_entries.append(item)

    attr_res = packet.get("attribution_result")
    if isinstance(attr_res, Mapping):
        raw_entries.append(attr_res)
    elif packet.get("attribution_outcome"):
        raw_entries.append(
            {
                "claim": claim_text,
                "attribution_outcome": packet.get("attribution_outcome"),
                "summary": packet.get("attribution_summary") or "",
            }
        )

    admitted: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    counts = {
        "certified_positive_ingested": 0,
        "certified_negative_ingested": 0,
        "quarantined_execution_artifacts": 0,
        "ambiguous_characterization_count": 0,
    }
    ambiguous = False

    for entry in raw_entries:
        outcome = _canonicalize_attribution_outcome(
            entry.get("attribution_outcome") or entry.get("outcome")
        )
        if not outcome:
            continue

        admit_flag = entry.get("admit_to_memory", entry.get("admitToScientificMemory", True))
        if outcome == "EXECUTION_ARTIFACT" or admit_flag is False or not _is_valid_optional_sha256(entry):
            if outcome == "AMBIGUOUS_CHARACTERIZATION":
                ambiguous = True
                counts["ambiguous_characterization_count"] += 1
            else:
                counts["quarantined_execution_artifacts"] += 1
            continue

        if outcome == "AMBIGUOUS_CHARACTERIZATION":
            ambiguous = True
            counts["ambiguous_characterization_count"] += 1
            continue

        if outcome not in ("CERTIFIED_POSITIVE", "CERTIFIED_NEGATIVE"):
            continue

        entry_claim = str(entry.get("claim") or claim_text).strip()
        summary_raw = str(
            entry.get("summary")
            or entry.get("sentence")
            or entry.get("rationale")
            or entry.get("detail")
            or ""
        ).strip()

        if not summary_raw:
            if outcome == "CERTIFIED_NEGATIVE":
                summary_raw = f"Certified negative assay readout disproved claim: {entry_claim}."
            else:
                summary_raw = f"Certified positive assay readout confirmed claim: {entry_claim}."

        dedup_id = str(
            entry.get("sha256")
            or entry.get("ledger_entry_sha256")
            or entry.get("entry_sha256")
            or summary_raw
        ).strip()
        dedup_pair = (dedup_id, outcome)
        if dedup_pair in seen_keys:
            continue
        seen_keys.add(dedup_pair)

        polarity = "CONTRADICT" if outcome == "CERTIFIED_NEGATIVE" else "SUPPORT"
        scoring_blob = f"{entry_claim} {summary_raw}".strip()
        admitted.append((summary_raw, scoring_blob, polarity))
        if polarity == "CONTRADICT":
            counts["certified_negative_ingested"] += 1
        else:
            counts["certified_positive_ingested"] += 1

    return admitted, counts, ambiguous


def _rank_support_and_contradict(
    claim_text: str,
    bm25_query: str,
    sentences: Sequence[str],
    scoring_texts: Sequence[str],
    explicit_polarities: Sequence[str | None],
    *,
    used_builtin: bool,
    top_k: int = 3,
) -> tuple[list[str], list[str], float]:
    """Rank candidate rationale sentences via Okapi BM25 and partition strictly by stance polarity."""
    if not sentences:
        return [], [], 0.0

    bm25 = BM25(scoring_texts)
    scores = bm25.score_indexed(bm25_query)
    norm_scores = bm25.score_indexed_normalized(bm25_query)
    ranked_indices = bm25.rank_candidates(bm25_query)

    polarities = [
        (
            explicit_polarities[idx]
            if explicit_polarities[idx] is not None
            else classify_rationale_polarity(claim_text, sentences[idx])
        )
        for idx in range(len(sentences))
    ]

    # Strictly require 'SUPPORT' vs 'CONTRADICT' so 'NEUTRAL' sentences never leak into support_top
    support_list = [
        sentences[idx]
        for idx in ranked_indices
        if scores[idx] > 0.0 and polarities[idx] == "SUPPORT"
    ][:top_k]
    contradict_list = [
        sentences[idx]
        for idx in ranked_indices
        if scores[idx] > 0.0 and polarities[idx] == "CONTRADICT"
    ][:top_k]

    max_norm = float(max(norm_scores)) if norm_scores.size > 0 else 0.0

    # When using the built-in wet-lab corpus, ensure generic claims with sparse lexical
    # overlap still ground against the default wet-lab protocol anchor without network I/O.
    if used_builtin and (not support_list or not contradict_list):
        augmented_query = f"{bm25_query} {_DEFAULT_WETLAB_QUERY_ANCHOR}".strip()
        aug_scores = bm25.score_indexed(augmented_query)
        aug_norm = bm25.score_indexed_normalized(augmented_query)
        aug_ranked = bm25.rank_candidates(augmented_query)
        if aug_norm.size > 0:
            max_norm = max(max_norm, float(max(aug_norm)))
        if not support_list:
            support_list = [
                sentences[idx]
                for idx in aug_ranked
                if aug_scores[idx] > 0.0 and polarities[idx] == "SUPPORT"
            ][:top_k]
        if not contradict_list:
            contradict_list = [
                sentences[idx]
                for idx in aug_ranked
                if aug_scores[idx] > 0.0 and polarities[idx] == "CONTRADICT"
            ][:top_k]

    if not support_list and not contradict_list:
        max_norm = 0.0

    return support_list, contradict_list, max_norm


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Process an ``autolab-wire/v1`` packet through Layer L6 ``lab-ask-why``."""
    out: dict[str, Any] = copy.deepcopy(dict(packet) if packet is not None else {})
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if out["halted"] is True or bool(out["halted"]):
        return out

    goal_text = str(out.get("goal_text", "") or "")
    raw_spans = out.get("parsed_spans", {})
    parsed_spans: Mapping[str, Any] = raw_spans if isinstance(raw_spans, Mapping) else {}
    raw_evidence = out.get("evidence_sentences")

    claim_text = _resolve_claim_text(goal_text, parsed_spans)
    span_context = _extract_span_context(parsed_spans)
    bm25_query = f"{claim_text} {span_context}".strip()

    sentences, explicit_polarities, used_builtin = _normalize_corpus_entries(
        raw_evidence
        if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes))
        else None,
    )
    scoring_texts = list(sentences)

    admitted_ledger, feedback_counts, ambiguous_characterization = _ingest_attribution_feedback(
        out, claim_text
    )
    for disp_text, score_blob, pol in admitted_ledger:
        sentences.append(disp_text)
        scoring_texts.append(score_blob)
        explicit_polarities.append(pol)

    support_list, contradict_list, max_norm_score = _rank_support_and_contradict(
        claim_text,
        bm25_query,
        sentences,
        scoring_texts,
        explicit_polarities,
        used_builtin=used_builtin,
    )

    # Interleave top supporting and contradicting rationales so differential mechanisms
    # from both sides ground Socratic question synthesis.
    grounded_context: list[str] = []
    for idx in range(max(len(support_list), len(contradict_list))):
        if idx < len(support_list):
            grounded_context.append(support_list[idx])
        if idx < len(contradict_list):
            grounded_context.append(contradict_list[idx])

    if support_list and contradict_list:
        epistemic_status = "CONTESTED"
    elif support_list:
        epistemic_status = "SUPPORTED"
    elif contradict_list:
        epistemic_status = "CONTRADICTED"
    else:
        epistemic_status = "INSUFFICIENT_EVIDENCE"

    retrieval_confidence = (
        to_permille_floor(max_norm_score) / 1000.0
        if epistemic_status != "INSUFFICIENT_EVIDENCE"
        else 0.0
    )
    abstain_recommended = epistemic_status == "INSUFFICIENT_EVIDENCE"

    stance_hint = "CONTRADICT" if epistemic_status == "CONTRADICTED" else None
    questions_list = why_questions(
        claim_text,
        grounded_context[:4] or None,
        stance=stance_hint,
    )

    clean_claim = claim_text.strip().rstrip(".?!").strip()
    if ambiguous_characterization:
        orth_question = (
            f"What orthogonal Raman or LC-MS characterization measurement with matched "
            f"precursor and vehicle blank controls would resolve the ambiguous spectral "
            f"readout for '{clean_claim}'?"
        )
        questions_list = [*questions_list[:-1], orth_question]

    diff_terms = (
        extract_differential_terms(claim_text, grounded_context[:4], max_terms=4)
        if grounded_context
        else []
    )
    if ambiguous_characterization:
        for modality_term in ("raman", "lcms"):
            if modality_term not in diff_terms:
                diff_terms.append(modality_term)

    raw_cand_desc = out.get("candidate_descriptors")
    caller_desc = (
        {str(k): float(v) for k, v in raw_cand_desc.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if isinstance(raw_cand_desc, Mapping)
        else None
    )
    candidate_descriptors = extract_protocol_descriptors(
        [claim_text, *grounded_context[:4]],
        parsed_spans=parsed_spans,
        base_descriptors=caller_desc,
        ensure_wetlab_defaults=(
            epistemic_status != "INSUFFICIENT_EVIDENCE"
            or ambiguous_characterization
            or bool(parsed_spans)
        ),
    )
    control_wells = (
        synthesize_control_wells(
            claim_text,
            diff_terms,
            top_sentences=grounded_context[:4],
            parsed_spans=parsed_spans,
            candidate_descriptors=candidate_descriptors,
        )
        if (epistemic_status != "INSUFFICIENT_EVIDENCE" or ambiguous_characterization)
        else []
    )

    observable_name = (
        "orthogonal_raman_lcms_characterization"
        if ambiguous_characterization
        else (
            f"perturbation_readout_{diff_terms[0]}"
            if diff_terms
            else "direct_replication_assay"
        )
    )
    observable = DiscriminatingObservable(
        observable_name=observable_name,
        differential_terms=diff_terms,
        proposed_perturbation=questions_list[-1],
        candidate_descriptors=candidate_descriptors,
        candidate_smiles=(
            str(out["candidate_smiles"]) if out.get("candidate_smiles") is not None else None
        ),
        control_wells=control_wells,
    )

    is_contested = epistemic_status == "CONTESTED"
    suppress_experiment = bool(out.get("suppress_discriminating_experiment", False))
    strict_ask_why = bool(out.get("strict_ask_why", False))

    if suppress_experiment and is_contested:
        out["halted"] = True
        out["halt_reason"] = "L6_CONTESTED_WITHOUT_EXPERIMENT"
    elif suppress_experiment and ambiguous_characterization:
        out["halted"] = True
        out["halt_reason"] = "L6_AMBIGUOUS_WITHOUT_ORTHOGONAL_ASSAY"
    elif strict_ask_why and abstain_recommended and not ambiguous_characterization:
        out["halted"] = True
        out["halt_reason"] = "L6_INSUFFICIENT_EVIDENCE_ABSTAIN"

    has_discriminating_experiment = bool(
        not out["halted"]
        and (epistemic_status != "INSUFFICIENT_EVIDENCE" or ambiguous_characterization)
    )

    out["epistemic_probe"] = {
        "claim": claim_text,
        "epistemic_status": epistemic_status,
        "retrieval_confidence": retrieval_confidence,
        "abstain_recommended": abstain_recommended,
        "support_top": support_list,
        "contradict_top": contradict_list,
        "why_questions": questions_list,
        "discriminating_question": questions_list[-1],
        "has_discriminating_experiment": has_discriminating_experiment,
        "discriminating_observable": observable.to_dict(),
        "control_wells": [cw.to_dict() for cw in control_wells],
        "orthogonal_characterization_required": ambiguous_characterization,
        "recommended_orthogonal_modalities": (
            ["Raman", "LC-MS"] if ambiguous_characterization else []
        ),
        "attribution_feedback": feedback_counts,
    }

    if out["halted"]:
        status = "BLOCK"
        if out["halt_reason"] == "L6_CONTESTED_WITHOUT_EXPERIMENT":
            summary = (
                f"Blocked contested claim ({len(support_list)} support, {len(contradict_list)} contradict) "
                "because discriminating experiment was suppressed."
            )
        elif out["halt_reason"] == "L6_AMBIGUOUS_WITHOUT_ORTHOGONAL_ASSAY":
            summary = (
                "Blocked ambiguous characterization readout because orthogonal Raman/LC-MS "
                "characterization probe was suppressed."
            )
        else:
            summary = (
                "Blocked packet under strict checkAskWhy mode due to zero-overlap / "
                "INSUFFICIENT_EVIDENCE rationale retrieval."
            )
    elif abstain_recommended and not ambiguous_characterization:
        status = "ABSTAIN"
        summary = (
            "Zero supporting or contradicting rationales matched claim; recommending "
            "INSUFFICIENT_EVIDENCE abstention to lab-decide."
        )
    else:
        status = "PASS"
        summary = (
            f"Ranked {len(support_list)} support and {len(contradict_list)} contradict BM25 "
            f"rationales ({epistemic_status}); generated {len(questions_list)} Socratic why-questions "
            f"and {len(control_wells)} structured control-well specs."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": status,
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "contested_claim_requires_experiment",
            "first_principles": (
                "Falsifiable experimental control design requires that whenever literature "
                "or post-run certified attribution contains both supporting and contradicting "
                "rationales, execution cannot proceed without an explicit discriminating "
                "perturbation and matched blank control to separate true molecular dissolution "
                "from hydrolysis or precursor ghost-peak artifacts."
            ),
        }
    )
    return out


if __name__ == "__main__":
    demo_packet = process_packet(
        {
            "goal_text": (
                "Transfer 100 uL aspirin solution and incubate at 37 C for 15 min "
                "at 300 rpm to evaluate aqueous solubility."
            ),
            "parsed_spans": {
                "amounts": ["100 uL"],
                "temperatures": ["37 C"],
                "times": ["15 min"],
                "speeds": ["300 rpm"],
                "volume_ul": 100.0,
                "temp_c": 37.0,
                "duration_min": 15.0,
            },
        }
    )
    print(json.dumps(demo_packet, indent=2))
