"""Typed EpistemicProbeResult schema and Lean-aligned permille guards for L6 handoff."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Mapping, Sequence

EPISTEMIC_STATUSES = ("SUPPORTED", "CONTRADICTED", "CONTESTED", "INSUFFICIENT_EVIDENCE")
POLARITY_VALUES = ("SUPPORT", "CONTRADICT", "NEUTRAL")
CONTROL_WELL_ROLES = (
    "VEHICLE_BLANK",
    "POSITIVE_STANDARD",
    "INACTIVATED_CONTROL",
    "PERTURBATION_ARM",
)


def to_permille_floor(fraction: float) -> int:
    """Conservatively quantize a [0, 1] lower-bounded confidence/score downward via floor."""
    val = float(fraction)
    if not math.isfinite(val):
        raise ValueError(f"Expected finite fraction for floor permille conversion, got {fraction!r}")
    clamped = max(0.0, min(1.0, val))
    scaled = (Decimal(str(clamped)) * Decimal(1000)).to_integral_value(rounding=ROUND_FLOOR)
    return max(0, min(1000, int(scaled)))


def to_permille_ceil(fraction: float) -> int:
    """Conservatively quantize a [0, 1] upper-bounded error/overlap upward via ceil."""
    val = float(fraction)
    if not math.isfinite(val):
        raise ValueError(f"Expected finite fraction for ceil permille conversion, got {fraction!r}")
    clamped = max(0.0, min(1.0, val))
    scaled = (Decimal(str(clamped)) * Decimal(1000)).to_integral_value(rounding=ROUND_CEILING)
    return max(0, min(1000, int(scaled)))


def _floor_6dp(value: float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"Expected finite float, got {value!r}")
    q = Decimal(str(val)).quantize(Decimal("0.000001"), rounding=ROUND_FLOOR)
    return float(q)


@dataclass(frozen=True)
class RationaleEvidence:
    """Single retrieved rationale sentence with provenance and BM25/hybrid score."""

    doc_id: int | str
    sent_idx: int
    sentence: str
    bm25_score: float
    polarity: str = "SUPPORT"
    title: str = ""

    def __post_init__(self) -> None:
        if self.polarity not in POLARITY_VALUES:
            raise ValueError(f"Invalid polarity {self.polarity!r}; expected one of {POLARITY_VALUES}")
        score = float(self.bm25_score)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"bm25_score must be finite and >= 0, got {self.bm25_score!r}")
        object.__setattr__(self, "sent_idx", int(self.sent_idx))
        object.__setattr__(self, "bm25_score", _floor_6dp(score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sent_idx": int(self.sent_idx),
            "sentence": self.sentence,
            "bm25_score": _floor_6dp(self.bm25_score),
            "polarity": self.polarity,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RationaleEvidence":
        return cls(
            doc_id=data["doc_id"],
            sent_idx=int(data["sent_idx"]),
            sentence=str(data["sentence"]),
            bm25_score=float(data["bm25_score"]),
            polarity=str(data.get("polarity", "SUPPORT")),
            title=str(data.get("title", "")),
        )


@dataclass(frozen=True)
class ControlWellSpec:
    """Machine-executable control or single-factor perturbation well specification."""

    role: str
    parameters: dict[str, float | str]
    expected_readout_under_support: str
    expected_readout_under_contradict: str

    def __post_init__(self) -> None:
        if self.role not in CONTROL_WELL_ROLES:
            raise ValueError(
                f"Invalid ControlWellSpec.role {self.role!r}; expected one of {CONTROL_WELL_ROLES}"
            )
        if not self.expected_readout_under_support or not self.expected_readout_under_support.strip():
            raise ValueError("expected_readout_under_support must be non-empty")
        if not self.expected_readout_under_contradict or not self.expected_readout_under_contradict.strip():
            raise ValueError("expected_readout_under_contradict must be non-empty")

        clean_params: dict[str, float | str] = {}
        for k, v in (self.parameters or {}).items():
            key = str(k)
            if isinstance(v, bool):
                clean_params[key] = "true" if v else "false"
            elif isinstance(v, (int, float)):
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError(f"Non-finite parameter {key}={v!r} in ControlWellSpec")
                clean_params[key] = fv
            else:
                clean_params[key] = str(v)
        object.__setattr__(self, "parameters", clean_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "parameters": dict(self.parameters),
            "expected_readout_under_support": self.expected_readout_under_support,
            "expected_readout_under_contradict": self.expected_readout_under_contradict,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlWellSpec":
        raw_params = data.get("parameters") or {}
        if not isinstance(raw_params, Mapping):
            raise ValueError("ControlWellSpec.parameters must be a mapping")
        return cls(
            role=str(data["role"]),
            parameters=dict(raw_params),
            expected_readout_under_support=str(data["expected_readout_under_support"]),
            expected_readout_under_contradict=str(data["expected_readout_under_contradict"]),
        )


@dataclass(frozen=True)
class DiscriminatingObservable:
    """Proposed experimental observable separating competing mechanistic explanations."""

    observable_name: str
    differential_terms: list[str]
    proposed_perturbation: str
    candidate_descriptors: dict[str, float] = field(default_factory=dict)
    candidate_smiles: str | None = None
    control_wells: list[ControlWellSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        clean_desc: dict[str, float] = {}
        for k, v in (self.candidate_descriptors or {}).items():
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"Non-finite candidate_descriptor {k}={v!r}")
            clean_desc[str(k)] = fv
        clean_wells: list[ControlWellSpec] = [
            cw if isinstance(cw, ControlWellSpec) else ControlWellSpec.from_dict(cw)
            for cw in (self.control_wells or ())
        ]
        object.__setattr__(self, "differential_terms", [str(t) for t in self.differential_terms])
        object.__setattr__(self, "candidate_descriptors", clean_desc)
        object.__setattr__(self, "control_wells", clean_wells)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "observable_name": self.observable_name,
            "differential_terms": list(self.differential_terms),
            "proposed_perturbation": self.proposed_perturbation,
            "candidate_descriptors": {k: float(v) for k, v in self.candidate_descriptors.items()},
            "candidate_smiles": self.candidate_smiles,
        }
        if self.control_wells:
            out["control_wells"] = [cw.to_dict() for cw in self.control_wells]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscriminatingObservable":
        return cls(
            observable_name=str(data["observable_name"]),
            differential_terms=[str(x) for x in data.get("differential_terms", [])],
            proposed_perturbation=str(data["proposed_perturbation"]),
            candidate_descriptors={
                str(k): float(v) for k, v in (data.get("candidate_descriptors") or {}).items()
            },
            candidate_smiles=data.get("candidate_smiles"),
            control_wells=[
                ControlWellSpec.from_dict(cw) for cw in (data.get("control_wells") or [])
            ],
        )


@dataclass(frozen=True)
class EpistemicProbeResult:
    """Structured L6 disagreement record linking lab-ask-why to lab-decide and labshift."""

    claim: str
    supporting_rationales: list[RationaleEvidence]
    contradicting_rationales: list[RationaleEvidence]
    assumptions: list[str]
    why_questions: list[str]
    discriminating_observable: DiscriminatingObservable
    retrieval_confidence: float
    epistemic_status: str = "INSUFFICIENT_EVIDENCE"
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.claim or not self.claim.strip():
            raise ValueError("EpistemicProbeResult.claim must be non-empty")
        conf = float(self.retrieval_confidence)
        if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"retrieval_confidence must be in [0, 1], got {self.retrieval_confidence}"
            )
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise ValueError(
                f"epistemic_status {self.epistemic_status!r} must be one of {EPISTEMIC_STATUSES}"
            )
        object.__setattr__(self, "retrieval_confidence", _floor_6dp(conf))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim": self.claim,
            "epistemic_status": self.epistemic_status,
            "retrieval_confidence": _floor_6dp(self.retrieval_confidence),
            "supporting_rationales": [r.to_dict() for r in self.supporting_rationales],
            "contradicting_rationales": [r.to_dict() for r in self.contradicting_rationales],
            "assumptions": list(self.assumptions),
            "why_questions": list(self.why_questions),
            "discriminating_observable": self.discriminating_observable.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpistemicProbeResult":
        return cls(
            schema_version=str(data.get("schema_version", "1.0.0")),
            claim=str(data["claim"]),
            epistemic_status=str(data["epistemic_status"]),
            retrieval_confidence=float(data["retrieval_confidence"]),
            supporting_rationales=[
                RationaleEvidence.from_dict(r) for r in data.get("supporting_rationales", [])
            ],
            contradicting_rationales=[
                RationaleEvidence.from_dict(r) for r in data.get("contradicting_rationales", [])
            ],
            assumptions=[str(a) for a in data.get("assumptions", [])],
            why_questions=[str(q) for q in data.get("why_questions", [])],
            discriminating_observable=DiscriminatingObservable.from_dict(
                data["discriminating_observable"]
            ),
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """Return JSON Schema Draft 2020-12 contract for sibling-repo validation."""
        rationale_schema: dict[str, Any] = {
            "type": "object",
            "required": ["doc_id", "sent_idx", "sentence", "bm25_score", "polarity"],
            "properties": {
                "doc_id": {"type": ["integer", "string"]},
                "sent_idx": {"type": "integer", "minimum": 0},
                "sentence": {"type": "string", "minLength": 1},
                "bm25_score": {"type": "number", "minimum": 0.0},
                "polarity": {"type": "string", "enum": list(POLARITY_VALUES)},
                "title": {"type": "string"},
            },
            "additionalProperties": False,
        }
        control_well_schema: dict[str, Any] = {
            "type": "object",
            "required": [
                "role",
                "parameters",
                "expected_readout_under_support",
                "expected_readout_under_contradict",
            ],
            "properties": {
                "role": {"type": "string", "enum": list(CONTROL_WELL_ROLES)},
                "parameters": {
                    "type": "object",
                    "additionalProperties": {"type": ["number", "string"]},
                },
                "expected_readout_under_support": {"type": "string", "minLength": 1},
                "expected_readout_under_contradict": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/yaniguan/lab-ask-why/schemas/epistemic-probe-result-v1.json",
            "title": "EpistemicProbeResult",
            "type": "object",
            "required": [
                "schema_version",
                "claim",
                "epistemic_status",
                "retrieval_confidence",
                "supporting_rationales",
                "contradicting_rationales",
                "assumptions",
                "why_questions",
                "discriminating_observable",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "claim": {"type": "string", "minLength": 1},
                "epistemic_status": {"type": "string", "enum": list(EPISTEMIC_STATUSES)},
                "retrieval_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "supporting_rationales": {"type": "array", "items": rationale_schema},
                "contradicting_rationales": {"type": "array", "items": rationale_schema},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "why_questions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "discriminating_observable": {
                    "type": "object",
                    "required": [
                        "observable_name",
                        "differential_terms",
                        "proposed_perturbation",
                        "candidate_descriptors",
                    ],
                    "properties": {
                        "observable_name": {"type": "string"},
                        "differential_terms": {"type": "array", "items": {"type": "string"}},
                        "proposed_perturbation": {"type": "string"},
                        "candidate_descriptors": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "candidate_smiles": {"type": ["string", "null"]},
                        "control_wells": {"type": "array", "items": control_well_schema},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }


EPISTEMIC_PROBE_JSON_SCHEMA = EpistemicProbeResult.json_schema()


def check_ask_why_permille(
    probe_or_score: EpistemicProbeResult | float,
    *,
    noinfo_thresh: float = 0.35,
    support_hits: int | None = None,
    contradict_hits: int | None = None,
    has_discriminating_experiment: bool | None = None,
) -> dict[str, str] | None:
    """Runtime Python mirror of `LabAskWhyLean.checkAskWhy` using conservative floor/ceil permille bounds."""
    thresh_permille = to_permille_ceil(noinfo_thresh)
    if isinstance(probe_or_score, EpistemicProbeResult):
        top_permille = to_permille_floor(probe_or_score.retrieval_confidence)
        sup = (
            len(probe_or_score.supporting_rationales)
            if support_hits is None
            else int(support_hits)
        )
        con = (
            len(probe_or_score.contradicting_rationales)
            if contradict_hits is None
            else int(contradict_hits)
        )
        has_exp = (
            bool(probe_or_score.discriminating_observable.proposed_perturbation.strip())
            if has_discriminating_experiment is None
            else bool(has_discriminating_experiment)
        )
    else:
        top_permille = to_permille_floor(float(probe_or_score))
        sup = int(support_hits or 0)
        con = int(contradict_hits or 0)
        has_exp = True if has_discriminating_experiment is None else bool(has_discriminating_experiment)

    if top_permille < thresh_permille:
        return {
            "repo": "lab-ask-why",
            "rule": "noinfo_abstain",
            "detail": "retrieval evidence below NOINFO threshold; must abstain",
        }
    if sup + con <= 0:
        return {
            "repo": "lab-ask-why",
            "rule": "nonempty_rationale",
            "detail": "zero supporting or contradicting rationale sentences retrieved",
        }
    if sup > 0 and con > 0 and not has_exp:
        return {
            "repo": "lab-ask-why",
            "rule": "contested_requires_experiment",
            "detail": "both support and contradict evidence present without a discriminating experiment",
        }
    return None


def check_epistemic_graph_permille(
    probes: Sequence[EpistemicProbeResult | Mapping[str, Any]],
    *,
    implications: Sequence[tuple[int, int]] = (),
    exclusions: Sequence[tuple[int, int]] = (),
    noinfo_thresh: float = 0.35,
) -> dict[str, str] | None:
    """Runtime Python mirror of `LabAskWhyLean.checkEpistemicGraph` for multi-claim consistency."""
    normalized: list[tuple[int, int, int, int, bool]] = []
    default_thresh = to_permille_ceil(noinfo_thresh)

    for p in probes:
        if isinstance(p, EpistemicProbeResult):
            top_pm = to_permille_floor(p.retrieval_confidence)
            th_pm = default_thresh
            sup = len(p.supporting_rationales)
            con = len(p.contradicting_rationales)
            has_exp = bool(p.discriminating_observable.proposed_perturbation.strip())
        else:
            if "topScorePermille" in p:
                top_pm = int(p["topScorePermille"])
            else:
                top_pm = to_permille_floor(float(p.get("retrieval_confidence", 0.0)))
            if "noinfoThreshPermille" in p:
                th_pm = int(p["noinfoThreshPermille"])
            else:
                th_pm = to_permille_ceil(float(p.get("noinfo_thresh", noinfo_thresh)))
            sup = int(p.get("supportHits", len(p.get("supporting_rationales", ()))))
            con = int(p.get("contradictHits", len(p.get("contradicting_rationales", ()))))
            has_exp = bool(
                p.get(
                    "hasDiscriminatingExperiment",
                    p.get("has_discriminating_experiment", False),
                )
            )
        if top_pm < th_pm:
            return {
                "repo": "lab-ask-why",
                "rule": "noinfo_abstain",
                "detail": "retrieval evidence below NOINFO threshold; must abstain",
            }
        if sup + con <= 0:
            return {
                "repo": "lab-ask-why",
                "rule": "nonempty_rationale",
                "detail": "zero supporting or contradicting rationale sentences retrieved",
            }
        if sup > 0 and con > 0 and not has_exp:
            return {
                "repo": "lab-ask-why",
                "rule": "contested_requires_experiment",
                "detail": "both support and contradict evidence present without a discriminating experiment",
            }
        normalized.append((top_pm, th_pm, sup, con, has_exp))

    n_probes = len(normalized)
    for src_idx, dst_idx in implications:
        if src_idx < 0 or dst_idx < 0 or src_idx >= n_probes or dst_idx >= n_probes:
            return {
                "repo": "lab-ask-why",
                "rule": "implication_index_bounds",
                "detail": "implication edge index out of bounds for probe graph",
            }
        _, _, sup_a, _, exp_a = normalized[src_idx]
        _, _, _, con_b, _ = normalized[dst_idx]
        if sup_a > 0 and con_b > 0 and not exp_a:
            return {
                "repo": "lab-ask-why",
                "rule": "entailment_conflict_requires_experiment",
                "detail": "supported claim entails a contradicted prerequisite without a discriminating experiment",
            }

    for idx_a, idx_b in exclusions:
        if idx_a < 0 or idx_b < 0 or idx_a >= n_probes or idx_b >= n_probes:
            return {
                "repo": "lab-ask-why",
                "rule": "exclusion_index_bounds",
                "detail": "mutual-exclusion edge index out of bounds for probe graph",
            }
        _, _, sup_a, _, exp_a = normalized[idx_a]
        _, _, sup_b, _, exp_b = normalized[idx_b]
        if sup_a > 0 and sup_b > 0 and not (exp_a and exp_b):
            return {
                "repo": "lab-ask-why",
                "rule": "mutual_exclusion_conflict_requires_experiment",
                "detail": "mutually exclusive claims both carry supporting evidence without discriminating experiments",
            }

    return None


def attribute_assay_outcome_permille(telemetry: Mapping[str, Any]) -> str:
    """Runtime Python mirror of `LabAskWhyLean.attributeAssayOutcome`."""
    exec_clean = (
        bool(telemetry["preGateCertified"])
        and bool(telemetry["acousticNormal"])
        and bool(telemetry["scheduleWindowsRespected"])
        and bool(telemetry["tactileSeatedCleanly"])
        and bool(telemetry["posControlPassed"])
        and bool(telemetry["negControlBlankClean"])
    )
    if not exec_clean:
        return "executionArtifact"
    identifiable = (
        int(telemetry["spectralResidualPermille"]) <= int(telemetry["maxResidualPermille"])
        and int(telemetry["precursorOverlapPermille"]) <= int(telemetry["maxOverlapPermille"])
    )
    if not identifiable:
        return "ambiguousCharacterization"
    if bool(telemetry["signalAboveActivityThresh"]):
        return "certifiedPositive"
    return "certifiedNegative"


def conformal_stance_set_permille(
    p_support: int,
    p_contradict: int,
    p_nei: int,
    q_permille: int,
) -> dict[str, bool]:
    """Runtime Python mirror of `LabAskWhyLean.conformalStanceSetPermille`."""
    ps = int(p_support)
    pc = int(p_contradict)
    pn = int(p_nei)
    q = int(q_permille)
    s_in = 1000 <= ps + q
    c_in = 1000 <= pc + q
    n_in = 1000 <= pn + q
    if s_in or c_in or n_in:
        return {"supportIn": s_in, "contradictIn": c_in, "neiIn": n_in}
    if pc <= ps and pn <= ps:
        return {"supportIn": True, "contradictIn": False, "neiIn": False}
    if pn <= pc:
        return {"supportIn": False, "contradictIn": True, "neiIn": False}
    return {"supportIn": False, "contradictIn": False, "neiIn": True}


def export_probes_jsonl(results: Sequence[EpistemicProbeResult], path: str) -> int:
    """Serialize `EpistemicProbeResult` records as UTF-8 JSONL for lab-decide / labshift."""
    with open(path, mode="w", encoding="utf-8") as fh:
        for item in results:
            fh.write(item.to_json() + "\n")
    return len(results)


def load_probes_jsonl(path: str) -> list[EpistemicProbeResult]:
    with open(path, mode="r", encoding="utf-8") as fh:
        return [EpistemicProbeResult.from_dict(json.loads(line)) for line in fh if line.strip()]
