"""Socratic 'why' question generation and falsifiable control-well synthesis."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from lab_ask_why.bm25 import tokenize
from lab_ask_why.schema import ControlWellSpec

_GENERIC_RATIONALE_BOILERPLATE = frozenset(
    {
        "conclusion",
        "conclusions",
        "background",
        "objective",
        "objectives",
        "methods",
        "results",
        "discussion",
        "study",
        "studies",
        "found",
        "show",
        "showed",
        "shown",
        "shows",
        "suggest",
        "suggests",
        "suggested",
        "propose",
        "proposed",
        "however",
        "therefore",
        "thus",
        "may",
        "can",
        "could",
        "would",
        "also",
        "used",
        "using",
        "between",
        "among",
        "whether",
        "appear",
        "appears",
        "associated",
        "significant",
        "significantly",
        "substantially",
        "contribute",
        "contributes",
        "careful",
        "consideration",
        "interpretation",
        "alternative",
        "explanation",
        "possibility",
        "requires",
        "identified",
        "least",
        "more",
        "less",
        "often",
        "much",
        "one",
        "two",
        "other",
        "normal",
        "model",
        "predicted",
        "result",
        "common",
        "uncommon",
        "create",
        "creates",
        "created",
        "creating",
        "increase",
        "increased",
        "increases",
        "decrease",
        "decreased",
        "decreases",
        "individuals",
        "children",
        "subjects",
        "patients",
        "have",
        "has",
        "had",
        "easily",
        "convincingly",
        "replicated",
        "hundreds",
        "follow",
    }
)

_VOL_UL_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:uL|\u00b5L|\u03bcL|ul)\b")
_VOL_ML_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mL|ml)\b")
_TEMP_C_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*(?:\u00b0\s*)?C\b")
_DUR_MIN_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:min|mins|minutes|minute)\b", re.IGNORECASE)
_DUR_HR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours|hour)\b", re.IGNORECASE)
_SPEED_RPM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*rpm\b", re.IGNORECASE)
_PH_RE = re.compile(r"\bpH\s*(\d+(?:\.\d+)?)\b")
_CONC_UM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:uM|\u00b5M|\u03bcM|um)\b")
_CONC_MM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*mM\b")

_HYDROLYSIS_CUES = frozenset(
    {
        "hydrolysis",
        "hydrolyzed",
        "hydrolyzes",
        "hydrolyse",
        "hydrolysed",
        "deacetylation",
        "ester",
        "salicylate",
        "salicylic",
        "degradation",
        "instability",
    }
)
_TURBIDITY_CUES = frozenset(
    {
        "turbidity",
        "precipitation",
        "colloidal",
        "aggregation",
        "ghost",
        "precursor",
        "unfiltered",
        "scattering",
    }
)
_DENATURATION_CUES = frozenset(
    {
        "denaturation",
        "denature",
        "denatures",
        "denatured",
        "polymerase",
        "taq",
        "enzyme",
        "enzymatic",
        "kinase",
        "inactivation",
    }
)


def _format_embedded_claim(clean_claim: str) -> str:
    """Lowercase leading character only when the first word is a standard capitalized word or 'A'.

    Preserves biomedical acronyms, gene symbols, chemical formulas, and digits
    (e.g., 'GWAS', 'RNA', 'ATP', 'IL-6', 'COVID-19', 'E. coli', '1,000').
    """
    first_word = clean_claim.split()[0]
    if first_word in {"A", "An", "The"}:
        return clean_claim[0].lower() + clean_claim[1:]
    if (
        len(first_word) > 1
        and first_word[0].isascii()
        and first_word[0].isupper()
        and first_word[1:].isalpha()
        and first_word[1:].islower()
    ):
        return clean_claim[0].lower() + clean_claim[1:]
    return clean_claim


def extract_differential_terms(
    claim: str,
    top_sentences: Sequence[str],
    *,
    max_terms: int = 3,
) -> list[str]:
    """Extract salient mechanism/confounder tokens present in retrieved sentences but absent from claim."""
    claim_tokens = set(tokenize(claim))
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    order_idx = 0
    for sent in top_sentences:
        for tok in tokenize(sent):
            if (
                tok in claim_tokens
                or tok in _GENERIC_RATIONALE_BOILERPLATE
                or len(tok) < 4
                or tok[0].isdigit()
                or tok in {"http", "https", "www", "gov", "org"}
            ):
                continue
            counts[tok] += 1
            if tok not in first_seen:
                first_seen[tok] = order_idx
                order_idx += 1

    # Prefer recurring or specific biomedical mechanism terms from top-ranked sentences
    ranked = sorted(
        counts.keys(),
        key=lambda t: (-counts[t], -min(len(t), 12), first_seen[t]),
    )
    deduped: list[str] = []
    seen_stems: set[str] = set()
    for term in ranked:
        stem = term.rstrip("s") if len(term) > 4 else term
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        deduped.append(term)
        if len(deduped) >= max(1, int(max_terms)):
            break
    return deduped


def extract_protocol_descriptors(
    texts: Sequence[str],
    *,
    parsed_spans: Mapping[str, Any] | None = None,
    base_descriptors: Mapping[str, float] | None = None,
    ensure_wetlab_defaults: bool = False,
) -> dict[str, float]:
    """Extract structured numeric operating descriptors from protocol spans and biomedical text."""
    desc: dict[str, float] = {}
    scan_blobs: list[str] = [str(t) for t in texts if t]

    if isinstance(parsed_spans, Mapping):
        for num_key in ("volume_ul", "temp_c", "duration_min", "speed_rpm", "ph", "concentration_um"):
            raw_val = parsed_spans.get(num_key)
            if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
                desc[num_key] = float(raw_val)
        for list_key in ("amounts", "temperatures", "times", "speeds"):
            items = parsed_spans.get(list_key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                for item in items:
                    if isinstance(item, Mapping):
                        txt = str(item.get("text") or item.get("value") or "").strip()
                    else:
                        txt = str(item).strip()
                    if txt:
                        scan_blobs.insert(0, txt)

    combined = " ; ".join(scan_blobs)
    if "volume_ul" not in desc:
        m_ul = _VOL_UL_RE.search(combined)
        m_ml = _VOL_ML_RE.search(combined)
        if m_ul:
            desc["volume_ul"] = float(m_ul.group(1))
        elif m_ml:
            desc["volume_ul"] = float(m_ml.group(1)) * 1000.0

    if "temp_c" not in desc:
        m_temp = _TEMP_C_RE.search(combined)
        if m_temp:
            desc["temp_c"] = float(m_temp.group(1))

    if "duration_min" not in desc:
        m_min = _DUR_MIN_RE.search(combined)
        m_hr = _DUR_HR_RE.search(combined)
        if m_min:
            desc["duration_min"] = float(m_min.group(1))
        elif m_hr:
            desc["duration_min"] = float(m_hr.group(1)) * 60.0

    if "speed_rpm" not in desc:
        m_rpm = _SPEED_RPM_RE.search(combined)
        if m_rpm:
            desc["speed_rpm"] = float(m_rpm.group(1))

    if "ph" not in desc:
        m_ph = _PH_RE.search(combined)
        if m_ph:
            desc["ph"] = float(m_ph.group(1))

    if "concentration_um" not in desc:
        m_um = _CONC_UM_RE.search(combined)
        m_mm = _CONC_MM_RE.search(combined)
        if m_um:
            desc["concentration_um"] = float(m_um.group(1))
        elif m_mm:
            desc["concentration_um"] = float(m_mm.group(1)) * 1000.0

    if ensure_wetlab_defaults:
        desc.setdefault("volume_ul", 100.0)
        desc.setdefault("temp_c", 25.0)
        desc.setdefault("duration_min", 15.0)

    if base_descriptors:
        for k, v in base_descriptors.items():
            desc[str(k)] = float(v)

    return desc


def synthesize_control_wells(
    claim: str,
    differential_terms: Sequence[str],
    *,
    top_sentences: Sequence[str] = (),
    parsed_spans: Mapping[str, Any] | None = None,
    candidate_descriptors: Mapping[str, float] | None = None,
) -> list[ControlWellSpec]:
    """Synthesize deterministic matched blank, reference, and perturbation control-well specs."""
    desc = extract_protocol_descriptors(
        [claim, *top_sentences],
        parsed_spans=parsed_spans,
        base_descriptors=candidate_descriptors,
        ensure_wetlab_defaults=True,
    )
    vol = float(desc.get("volume_ul", 100.0))
    temp = float(desc.get("temp_c", 25.0))
    dur = float(desc.get("duration_min", 15.0))

    shared_base: dict[str, float | str] = {
        "volume_ul": vol,
        "temp_c": temp,
        "duration_min": dur,
    }
    if "speed_rpm" in desc:
        shared_base["speed_rpm"] = float(desc["speed_rpm"])
    if "ph" in desc:
        shared_base["ph"] = float(desc["ph"])

    cue_tokens = set(tokenize(" ".join([claim, *differential_terms, *top_sentences])))
    has_hydrolysis = bool(cue_tokens & _HYDROLYSIS_CUES)
    has_turbidity = bool(cue_tokens & _TURBIDITY_CUES)
    has_denaturation = bool(cue_tokens & _DENATURATION_CUES)

    wells: list[ControlWellSpec] = [
        ControlWellSpec(
            role="VEHICLE_BLANK",
            parameters={
                **shared_base,
                "analyte_dose_factor": 0.0,
                "matrix": "matched_vehicle_buffer",
            },
            expected_readout_under_support=(
                "Near-zero baseline blank signal confirming zero vehicle absorbance or background drift."
            ),
            expected_readout_under_contradict=(
                "Non-zero blank drift or reagent background artifact confounding apparent signal."
            ),
        ),
        ControlWellSpec(
            role="POSITIVE_STANDARD",
            parameters={
                **shared_base,
                "analyte_dose_factor": 1.0,
                "standard_calibration": "certified_reference_standard",
            },
            expected_readout_under_support=(
                "Linear calibrated reference response within expected assay dynamic range."
            ),
            expected_readout_under_contradict=(
                "Reference standard deviation outside calibration tolerance indicating assay fault."
            ),
        ),
    ]

    if has_hydrolysis:
        wells.append(
            ControlWellSpec(
                role="INACTIVATED_CONTROL",
                parameters={
                    **shared_base,
                    "temp_c": 4.0,
                    "analyte_dose_factor": 1.0,
                    "quench_mode": "low_temp_hydrolysis_quench",
                },
                expected_readout_under_support=(
                    "Intact parent ester peak retained with negligible hydrolysis product formation."
                ),
                expected_readout_under_contradict=(
                    "Immediate degradant peak present prior to incubation, indicating degraded stock."
                ),
            )
        )
        perturb_temp = 25.0 if temp >= 30.0 else 37.0
        wells.append(
            ControlWellSpec(
                role="PERTURBATION_ARM",
                parameters={
                    **shared_base,
                    "temp_c": perturb_temp,
                    "analyte_dose_factor": 1.0,
                    "confounder_target": "hydrolysis",
                    "orthogonal_readout": "LC-MS_parent_vs_degradant",
                },
                expected_readout_under_support=(
                    "Intact parent molecule concentration scales cleanly without ester hydrolysis conversion."
                ),
                expected_readout_under_contradict=(
                    "Apparent optical shift tracks salicylic/hydrolysis degradant rather than intact parent solubility."
                ),
            )
        )
    elif has_turbidity:
        wells.append(
            ControlWellSpec(
                role="PERTURBATION_ARM",
                parameters={
                    **shared_base,
                    "analyte_dose_factor": 1.0,
                    "confounder_target": "turbidity",
                    "filtration_pore_um": 0.22,
                    "detergent_triton_pct": 0.01,
                },
                expected_readout_under_support=(
                    "Soluble readout persists after 0.22 um filtration and detergent addition."
                ),
                expected_readout_under_contradict=(
                    "Apparent readout collapses post-filtration, exposing colloidal precipitation or precursor turbidity."
                ),
            )
        )
    elif has_denaturation:
        wells.append(
            ControlWellSpec(
                role="INACTIVATED_CONTROL",
                parameters={
                    **shared_base,
                    "temp_c": 95.0,
                    "duration_min": 5.0,
                    "analyte_dose_factor": 1.0,
                    "quench_mode": "thermal_inactivation_control",
                },
                expected_readout_under_support=(
                    "Complete loss of catalytic activity after heat denaturation confirms active enzyme requirement."
                ),
                expected_readout_under_contradict=(
                    "Persistent signal after heat denaturation reveals non-enzymatic artifact."
                ),
            )
        )
        safe_temp = 25.0 if temp >= 42.0 else round(temp + 10.0, 2)
        wells.append(
            ControlWellSpec(
                role="PERTURBATION_ARM",
                parameters={
                    **shared_base,
                    "temp_c": safe_temp,
                    "analyte_dose_factor": 1.0,
                    "confounder_target": "denaturation",
                },
                expected_readout_under_support=(
                    "Controlled thermal window preserves catalytic activity and target readout."
                ),
                expected_readout_under_contradict=(
                    "Thermal perturbation abolishes activity, confirming denaturation confounder."
                ),
            )
        )
    else:
        lead_factor = str(differential_terms[0]) if differential_terms else "primary_perturbation"
        wells.append(
            ControlWellSpec(
                role="PERTURBATION_ARM",
                parameters={
                    **shared_base,
                    "analyte_dose_factor": 0.5,
                    "confounder_target": lead_factor,
                },
                expected_readout_under_support=(
                    f"Controlled single-factor perturbation of '{lead_factor}' shifts readout in the direction predicted by '{claim.strip().rstrip('.')}'."
                ),
                expected_readout_under_contradict=(
                    f"Single-factor perturbation of '{lead_factor}' produces null or inverted readout, falsifying the claim."
                ),
            )
        )

    return wells


def why_questions(
    claim: str,
    top_sentences: Sequence[str] | None = None,
    *,
    stance: str | None = None,
    llm_fn: Callable[[str, Sequence[str], str | None], list[str]] | None = None,
) -> list[str]:
    """Generate three Socratic epistemic probes for `claim`, optionally grounded in `top_sentences`.

    Raises `ValueError` if `claim` is empty or contains only whitespace/punctuation.
    """
    if not isinstance(claim, str):
        raise ValueError("Claim must be a string")
    c = claim.strip().rstrip(".?!").strip()
    if not c or not any(ch.isalnum() for ch in c):
        raise ValueError("Claim must contain non-empty alphanumeric content")

    if llm_fn is not None:
        return list(llm_fn(c, list(top_sentences or ()), stance))

    embedded = _format_embedded_claim(c)
    if not top_sentences:
        return [
            f"What evidence makes you think that {embedded}?",
            f"What are you assuming must be true for '{c}' to hold?",
            f"What result would change your mind about '{c}'?",
        ]

    diff_terms = extract_differential_terms(c, top_sentences, max_terms=3)
    if not diff_terms:
        return [
            f"What evidence makes you think that {embedded}?",
            f"What are you assuming must be true for '{c}' to hold?",
            f"What result would change your mind about '{c}'?",
        ]

    focus = ", ".join(diff_terms)
    lead_term = diff_terms[0]
    polarity_hint = (
        f"contradicting observation around '{lead_term}'"
        if stance == "CONTRADICT"
        else f"mechanistic role of '{lead_term}'"
    )
    return [
        f"What direct evidence links {focus} to the claim that {embedded}?",
        f"What are you assuming about the {polarity_hint} ({focus}) for '{c}' to hold?",
        f"What controlled perturbation or measurement of {focus} would change your mind about '{c}'?",
    ]
