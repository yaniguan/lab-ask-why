"""Auto-Lab 15-repository wire node for lab-ask-why (Layer L6 Epistemic Brain).

Connects upstream protocol/sensory spans from ``lab-ears`` (L2) to symmetric
Okapi BM25 + hybrid subword/SVD rationale retrieval, Split-Conformal 3-way stance
sets, Socratic ``why_questions`` framing, and structured control-well synthesis
before calibrating trust regions in ``lab-decide`` (L6), with post-run causal
attribution certificates closed through ``lab-attrib`` (L4).
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
from lab_ask_why.frontier_verifier import FrontierHybridVerifier  # noqa: E402
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
_SHA256_KEYS = ("sha256", "ledger_entry_sha256", "entry_sha256", "entry_hash")
_VALID_OUTCOMES = frozenset(
    {
        "CERTIFIED_POSITIVE",
        "CERTIFIED_NEGATIVE",
        "EXECUTION_ARTIFACT",
        "AMBIGUOUS_CHARACTERIZATION",
    }
)
_ALLOWED_DATA_SOURCES = frozenset({"real_downloaded", "synthetic_fixture", "caller_provided"})


def _canonicalize_attribution_outcome(raw_outcome: Any) -> str:
    """Normalize lab-attrib outcome strings (snake_case, camelCase, UPPER_SNAKE) to canonical form."""
    if raw_outcome is None or not isinstance(raw_outcome, str):
        return ""
    text = raw_outcome.strip()
    if not text:
        return ""
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
    for key in _SHA256_KEYS:
        if key in entry and entry[key] is not None:
            if not isinstance(entry[key], str):
                return False
            digest = entry[key].strip()
            if not _SHA256_HEX_RE.match(digest):
                return False
    return True


def _has_required_valid_sha256(entry: Mapping[str, Any]) -> bool:
    """Require at least one valid 64-hex SHA-256 digest and zero malformed digests."""
    if not _is_valid_optional_sha256(entry):
        return False
    for key in _SHA256_KEYS:
        val = entry.get(key)
        if isinstance(val, str) and _SHA256_HEX_RE.match(val.strip()):
            return True
    return False


def _halt_fail_closed(
    out: dict[str, Any],
    halt_reason: str,
    summary: str,
    *,
    data_source: str = "synthetic_fixture",
) -> dict[str, Any]:
    """Fail-closed helper emitting a clean `BLOCK` packet without raising uncaught exceptions."""
    out["halted"] = True
    out["halt_reason"] = halt_reason
    out["data_source"] = data_source
    out["epistemic_probe"] = {
        "claim": "",
        "data_source": data_source,
        "epistemic_status": "INSUFFICIENT_EVIDENCE",
        "conformal_stance_set": ["NEI"],
        "conformal_alpha": 0.10,
        "stance_probabilities": {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 1.0},
        "retrieval_confidence": 0.0,
        "abstain_recommended": True,
        "support_top": [],
        "contradict_top": [],
        "why_questions": [],
        "discriminating_question": "",
        "has_discriminating_experiment": False,
        "discriminating_observable": None,
        "control_wells": [],
        "orthogonal_characterization_required": False,
        "recommended_orthogonal_modalities": [],
        "attribution_feedback": {
            "certified_positive_ingested": 0,
            "certified_negative_ingested": 0,
            "quarantined_execution_artifacts": 0,
            "ambiguous_characterization_count": 0,
        },
    }
    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "BLOCK",
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "fail_closed_input_guard",
            "first_principles": summary,
        }
    )
    return out


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
) -> tuple[list[str], list[str | None], bool, str | None]:
    """Normalize evidence sentences into `(texts, explicit_polarities, used_builtin, error_msg)`."""
    if raw_sentences is None:
        return (
            list(DEFAULT_WETLAB_EVIDENCE_SENTENCES),
            [None] * len(DEFAULT_WETLAB_EVIDENCE_SENTENCES),
            True,
            None,
        )

    texts: list[str] = []
    explicit_polarities: list[str | None] = []
    for entry in raw_sentences:
        if isinstance(entry, Mapping):
            raw_txt = entry.get("sentence") if "sentence" in entry else entry.get("text")
            if not isinstance(raw_txt, str) or not raw_txt.strip():
                return [], [], False, "Evidence mapping entry must contain a non-empty string 'sentence' or 'text'."
            text = raw_txt.strip()
            if "polarity" in entry and entry["polarity"] is not None:
                if not isinstance(entry["polarity"], str):
                    return [], [], False, "Evidence entry 'polarity' must be a string."
                pol_raw = entry["polarity"].upper().strip()
                if pol_raw not in {"SUPPORT", "CONTRADICT", "NEUTRAL"}:
                    return [], [], False, f"Invalid evidence polarity {entry['polarity']!r}."
                pol: str | None = pol_raw
            else:
                pol = None
        elif isinstance(entry, str):
            text = entry.strip()
            if not text:
                return [], [], False, "Evidence sentence string must be non-empty."
            pol = None
        else:
            return [], [], False, f"Unsupported evidence_sentences element of type {type(entry).__name__}."
        texts.append(text)
        explicit_polarities.append(pol)
    return texts, explicit_polarities, False, None


def _validate_negative_certificate_entry(entry: Mapping[str, Any]) -> str | None:
    """Enforce strict soundness preconditions on `CERTIFIED_NEGATIVE` certificates."""
    if "claim" in entry and entry["claim"] is not None:
        if not isinstance(entry["claim"], str) or not entry["claim"].strip():
            return "Negative certificate 'claim' must be a non-empty string."

    summary_val = None
    for k in ("summary", "sentence", "rationale", "detail"):
        if k in entry and entry[k] is not None:
            if not isinstance(entry[k], str):
                return f"Negative certificate {k!r} must be a string."
            if entry[k].strip():
                summary_val = entry[k].strip()
                break
    if not summary_val:
        return "CERTIFIED_NEGATIVE certificate requires a non-empty string rationale/summary."

    if not _has_required_valid_sha256(entry):
        return "CERTIFIED_NEGATIVE certificate requires a valid 64-hex SHA-256 digest."

    for bool_k in (
        "admit_to_memory",
        "admitToScientificMemory",
        "execution_clean",
        "executionClean",
        "signal_above_activity_thresh",
        "signalAboveActivityThresh",
    ):
        if bool_k in entry and entry[bool_k] is not None and not isinstance(entry[bool_k], bool):
            return f"Certificate flag {bool_k!r} must be a boolean."

    if entry.get("admit_to_memory") is False or entry.get("admitToScientificMemory") is False:
        return "CERTIFIED_NEGATIVE certificate contradicts admit_to_memory=False."
    if entry.get("execution_clean") is False or entry.get("executionClean") is False:
        return "CERTIFIED_NEGATIVE certificate violates executionClean=True."
    if entry.get("signal_above_activity_thresh") is True or entry.get("signalAboveActivityThresh") is True:
        return "CERTIFIED_NEGATIVE certificate violates signalAboveActivityThresh=False."

    res_pm = entry.get("spectral_residual_permille", entry.get("spectralResidualPermille"))
    max_res_pm = entry.get("max_residual_permille", entry.get("maxResidualPermille"))
    if res_pm is not None and max_res_pm is not None:
        if (
            isinstance(res_pm, bool)
            or isinstance(max_res_pm, bool)
            or not isinstance(res_pm, (int, float))
            or not isinstance(max_res_pm, (int, float))
            or float(res_pm) > float(max_res_pm)
        ):
            return "CERTIFIED_NEGATIVE certificate violates spectralResidualPermille <= maxResidualPermille."

    ov_pm = entry.get("precursor_overlap_permille", entry.get("precursorOverlapPermille"))
    max_ov_pm = entry.get("max_overlap_permille", entry.get("maxOverlapPermille"))
    if ov_pm is not None and max_ov_pm is not None:
        if (
            isinstance(ov_pm, bool)
            or isinstance(max_ov_pm, bool)
            or not isinstance(ov_pm, (int, float))
            or not isinstance(max_ov_pm, (int, float))
            or float(ov_pm) > float(max_ov_pm)
        ):
            return "CERTIFIED_NEGATIVE certificate violates precursorOverlapPermille <= maxOverlapPermille."

    return None


def _ingest_attribution_feedback(
    packet: Mapping[str, Any],
    claim_text: str,
) -> tuple[list[tuple[str, str, str]], dict[str, int], bool, str | None]:
    """Ingest verified lab-attrib certificates and fail closed on malformed negative certificates.

    Returns ``(admitted_entries, feedback_counts, ambiguous_characterization, cert_error)``.
    """
    counts = {
        "certified_positive_ingested": 0,
        "certified_negative_ingested": 0,
        "quarantined_execution_artifacts": 0,
        "ambiguous_characterization_count": 0,
    }
    raw_entries: list[Mapping[str, Any]] = []

    if "certified_ledger_entries" in packet and packet["certified_ledger_entries"] is not None:
        ledger_seq = packet["certified_ledger_entries"]
        if not isinstance(ledger_seq, Sequence) or isinstance(ledger_seq, (str, bytes)):
            return [], counts, False, "Packet 'certified_ledger_entries' must be a list/sequence of mappings."
        for item in ledger_seq:
            if not isinstance(item, Mapping):
                return [], counts, False, "Each element of 'certified_ledger_entries' must be a mapping."
            raw_entries.append(item)

    if "attribution_result" in packet and packet["attribution_result"] is not None:
        attr_res = packet["attribution_result"]
        if not isinstance(attr_res, Mapping):
            return [], counts, False, "Packet 'attribution_result' must be a mapping."
        raw_entries.append(attr_res)
    elif "attribution_outcome" in packet and packet["attribution_outcome"] is not None:
        raw_out = packet["attribution_outcome"]
        if not isinstance(raw_out, str) or not raw_out.strip():
            return [], counts, False, "Packet 'attribution_outcome' must be a non-empty string."
        raw_sum = packet.get("attribution_summary")
        if raw_sum is not None and not isinstance(raw_sum, str):
            return [], counts, False, "Packet 'attribution_summary' must be a string."
        synth_entry: dict[str, Any] = {
            "claim": claim_text,
            "attribution_outcome": raw_out,
            "summary": raw_sum or "",
        }
        for sha_k in _SHA256_KEYS:
            if sha_k in packet:
                synth_entry[sha_k] = packet[sha_k]
        raw_entries.append(synth_entry)

    admitted: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    ambiguous = False

    for entry in raw_entries:
        for bool_k in ("admit_to_memory", "admitToScientificMemory"):
            if bool_k in entry and entry[bool_k] is not None and not isinstance(entry[bool_k], bool):
                return [], counts, False, f"Ledger certificate {bool_k!r} must be a boolean."

        raw_outcome_val = (
            entry.get("attribution_outcome")
            if entry.get("attribution_outcome") is not None
            else entry.get("outcome")
        )
        if raw_outcome_val is None:
            return [], counts, False, "Ledger certificate entry is missing 'attribution_outcome'."
        outcome = _canonicalize_attribution_outcome(raw_outcome_val)
        if outcome not in _VALID_OUTCOMES:
            return [], counts, False, f"Unrecognized certificate attribution_outcome {raw_outcome_val!r}."

        if outcome == "CERTIFIED_NEGATIVE":
            err = _validate_negative_certificate_entry(entry)
            if err is not None:
                return [], counts, False, err

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

        entry_claim = str(entry.get("claim") or claim_text).strip()
        summary_raw = str(
            entry.get("summary")
            or entry.get("sentence")
            or entry.get("rationale")
            or entry.get("detail")
            or ""
        ).strip()

        if not summary_raw:
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

    return admitted, counts, ambiguous, None


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


def _compute_conformal_stance_metadata(
    claim_text: str,
    sentences: Sequence[str],
    epistemic_status: str,
) -> tuple[list[str], dict[str, float]]:
    """Compute Split Conformal 90% stance set and 3-way probabilities on packet corpus."""
    if not sentences or epistemic_status == "INSUFFICIENT_EVIDENCE":
        return ["NEI"], {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 1.0}

    mini_corpus = {0: {"doc_id": 0, "title": "wire_packet_corpus", "abstract": list(sentences)}}
    verifier = FrontierHybridVerifier(mini_corpus, svd_components=min(8, max(1, len(sentences))))
    pred = verifier.verify_candidate_sentences(claim_text, sentences, top_k=min(4, len(sentences)))
    cset = list(pred.conformal_set)
    probs = dict(pred.probabilities)

    if epistemic_status == "CONTESTED":
        for req in ("SUPPORT", "CONTRADICT"):
            if req not in cset:
                cset.append(req)
    elif epistemic_status == "SUPPORTED" and "SUPPORT" not in cset:
        cset.insert(0, "SUPPORT")
    elif epistemic_status == "CONTRADICTED" and "CONTRADICT" not in cset:
        cset.insert(0, "CONTRADICT")

    ordered = [lab for lab in ("SUPPORT", "CONTRADICT", "NEI") if lab in cset]
    return ordered, probs


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Process an ``autolab-wire/v1`` packet through Layer L6 ``lab-ask-why`` (fail-closed)."""
    if packet is None or not isinstance(packet, Mapping):
        base: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "halted": False,
            "halt_reason": None,
            "trace": [],
            "bottom_reasoning_traces": [],
        }
        return _halt_fail_closed(
            base,
            "L6_INVALID_PACKET_TYPE",
            f"Expected Mapping packet for {REPO_ID}, got {type(packet).__name__}.",
        )

    out: dict[str, Any] = copy.deepcopy(dict(packet))
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if not isinstance(out["trace"], list) or not isinstance(out["bottom_reasoning_traces"], list):
        out["trace"] = list(out["trace"]) if isinstance(out["trace"], Sequence) and not isinstance(out["trace"], (str, bytes)) else []
        out["bottom_reasoning_traces"] = (
            list(out["bottom_reasoning_traces"])
            if isinstance(out["bottom_reasoning_traces"], Sequence) and not isinstance(out["bottom_reasoning_traces"], (str, bytes))
            else []
        )
        return _halt_fail_closed(
            out,
            "L6_INVALID_PACKET_FIELD",
            "Packet 'trace' and 'bottom_reasoning_traces' must be lists.",
        )

    # Strict boolean validation on `halted` before short-circuiting
    if not isinstance(out["halted"], bool):
        return _halt_fail_closed(
            out,
            "L6_INVALID_BOOLEAN_FLAG",
            f"Packet 'halted' must be a boolean, got {type(out['halted']).__name__}.",
        )

    if out["halted"] is True:
        return out

    # Strict boolean validation on control flags (never coerce "false" -> True!)
    for bool_flag in ("suppress_discriminating_experiment", "strict_ask_why"):
        if bool_flag in out and not isinstance(out[bool_flag], bool):
            return _halt_fail_closed(
                out,
                "L6_INVALID_BOOLEAN_FLAG",
                f"Packet flag {bool_flag!r} must be a boolean, got {type(out[bool_flag]).__name__}.",
            )
    if (
        "speech_band_dropped" in out
        and out["speech_band_dropped"] is not None
        and not isinstance(out["speech_band_dropped"], bool)
    ):
        return _halt_fail_closed(
            out,
            "L6_INVALID_BOOLEAN_FLAG",
            f"Packet flag 'speech_band_dropped' must be a boolean, got {type(out['speech_band_dropped']).__name__}.",
        )

    # Strict claim / goal_text validation when explicitly supplied
    explicit_claim: str | None = None
    if "claim" in out:
        raw_claim = out["claim"]
        if not isinstance(raw_claim, str) or not any(ch.isalnum() for ch in raw_claim.strip()):
            return _halt_fail_closed(
                out,
                "L6_INVALID_CLAIM",
                "Packet 'claim' must be a non-empty alphanumeric string.",
            )
        explicit_claim = raw_claim.strip()

    if "goal_text" in out:
        raw_goal = out["goal_text"]
        if not isinstance(raw_goal, str) or not any(ch.isalnum() for ch in raw_goal.strip()):
            return _halt_fail_closed(
                out,
                "L6_INVALID_CLAIM",
                "Packet 'goal_text' must be a non-empty alphanumeric string.",
            )
        goal_text = raw_goal
    else:
        goal_text = explicit_claim or ""

    if "parsed_spans" in out and out["parsed_spans"] is not None and not isinstance(out["parsed_spans"], Mapping):
        return _halt_fail_closed(
            out,
            "L6_INVALID_PACKET_FIELD",
            "Packet 'parsed_spans' must be a mapping when provided.",
        )
    raw_spans = out.get("parsed_spans") or {}
    parsed_spans: Mapping[str, Any] = raw_spans if isinstance(raw_spans, Mapping) else {}

    if "evidence_sentences" in out and out["evidence_sentences"] is not None:
        raw_evidence = out["evidence_sentences"]
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            return _halt_fail_closed(
                out,
                "L6_INVALID_EVIDENCE_SENTENCES",
                "Packet 'evidence_sentences' must be a list/sequence of strings or mappings.",
            )
    else:
        raw_evidence = None

    sentences, explicit_polarities, used_builtin, ev_err = _normalize_corpus_entries(raw_evidence)
    if ev_err is not None:
        return _halt_fail_closed(out, "L6_INVALID_EVIDENCE_SENTENCES", ev_err)

    caller_ds = out.get("data_source")
    if caller_ds is not None:
        if not isinstance(caller_ds, str) or caller_ds not in _ALLOWED_DATA_SOURCES:
            return _halt_fail_closed(
                out,
                "L6_INVALID_DATA_SOURCE",
                f"Packet 'data_source' must be one of {sorted(_ALLOWED_DATA_SOURCES)}, got {caller_ds!r}.",
            )
        data_source = caller_ds
    else:
        data_source = "synthetic_fixture" if used_builtin else "caller_provided"

    claim_text = explicit_claim if explicit_claim is not None else _resolve_claim_text(goal_text, parsed_spans)
    span_context = _extract_span_context(parsed_spans)
    bm25_query = f"{claim_text} {span_context}".strip()
    scoring_texts = list(sentences)

    admitted_ledger, feedback_counts, ambiguous_characterization, cert_err = _ingest_attribution_feedback(
        out, claim_text
    )
    if cert_err is not None:
        return _halt_fail_closed(
            out,
            "L6_MALFORMED_NEGATIVE_CERTIFICATE",
            cert_err,
            data_source=data_source,
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

    conformal_stance_set, stance_probabilities = _compute_conformal_stance_metadata(
        claim_text, sentences, epistemic_status
    )

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
        {
            str(k): float(v)
            for k, v in raw_cand_desc.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
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
    suppress_experiment = out.get("suppress_discriminating_experiment", False) is True
    strict_ask_why = out.get("strict_ask_why", False) is True

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

    out["data_source"] = data_source
    out["epistemic_probe"] = {
        "claim": claim_text,
        "data_source": data_source,
        "epistemic_status": epistemic_status,
        "conformal_stance_set": conformal_stance_set,
        "conformal_alpha": 0.10,
        "stance_probabilities": stance_probabilities,
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
            f"rationales ({epistemic_status}, conformal_set={conformal_stance_set}, "
            f"data_source={data_source}); generated {len(questions_list)} Socratic why-questions "
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
