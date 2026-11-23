"""Typed EpistemicProbeResult schema for L6 handoff to lab-decide and labshift."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

EPISTEMIC_STATUSES = ("SUPPORTED", "CONTRADICTED", "CONTESTED", "INSUFFICIENT_EVIDENCE")
POLARITY_VALUES = ("SUPPORT", "CONTRADICT", "NEUTRAL")


@dataclass(frozen=True)
class RationaleEvidence:
    """Single retrieved rationale sentence with provenance and BM25 score."""

    doc_id: int | str
    sent_idx: int
    sentence: str
    bm25_score: float
    polarity: str = "SUPPORT"
    title: str = ""

    def __post_init__(self) -> None:
        if self.polarity not in POLARITY_VALUES:
            raise ValueError(f"Invalid polarity {self.polarity!r}; expected one of {POLARITY_VALUES}")
        object.__setattr__(self, "sent_idx", int(self.sent_idx))
        object.__setattr__(self, "bm25_score", round(float(self.bm25_score), 6))

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sent_idx": int(self.sent_idx),
            "sentence": self.sentence,
            "bm25_score": round(float(self.bm25_score), 6),
            "polarity": self.polarity,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RationaleEvidence":
        return cls(
            doc_id=data["doc_id"],
            sent_idx=int(data["sent_idx"]),
            sentence=str(data["sentence"]),
            bm25_score=float(data["bm25_score"]),
            polarity=str(data.get("polarity", "SUPPORT")),
            title=str(data.get("title", "")),
        )


@dataclass(frozen=True)
class DiscriminatingObservable:
    """Proposed experimental observable separating competing mechanistic explanations."""

    observable_name: str
    differential_terms: list[str]
    proposed_perturbation: str
    candidate_descriptors: dict[str, float] = field(default_factory=dict)
    candidate_smiles: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_name": self.observable_name,
            "differential_terms": list(self.differential_terms),
            "proposed_perturbation": self.proposed_perturbation,
            "candidate_descriptors": {k: float(v) for k, v in self.candidate_descriptors.items()},
            "candidate_smiles": self.candidate_smiles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscriminatingObservable":
        return cls(
            observable_name=str(data["observable_name"]),
            differential_terms=[str(x) for x in data.get("differential_terms", [])],
            proposed_perturbation=str(data["proposed_perturbation"]),
            candidate_descriptors={
                str(k): float(v) for k, v in (data.get("candidate_descriptors") or {}).items()
            },
            candidate_smiles=data.get("candidate_smiles"),
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
        if not (0.0 <= float(self.retrieval_confidence) <= 1.0):
            raise ValueError(
                f"retrieval_confidence must be in [0, 1], got {self.retrieval_confidence}"
            )
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise ValueError(
                f"epistemic_status {self.epistemic_status!r} must be one of {EPISTEMIC_STATUSES}"
            )
        object.__setattr__(self, "retrieval_confidence", round(float(self.retrieval_confidence), 6))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim": self.claim,
            "epistemic_status": self.epistemic_status,
            "retrieval_confidence": round(float(self.retrieval_confidence), 6),
            "supporting_rationales": [r.to_dict() for r in self.supporting_rationales],
            "contradicting_rationales": [r.to_dict() for r in self.contradicting_rationales],
            "assumptions": list(self.assumptions),
            "why_questions": list(self.why_questions),
            "discriminating_observable": self.discriminating_observable.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpistemicProbeResult":
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
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }


EPISTEMIC_PROBE_JSON_SCHEMA = EpistemicProbeResult.json_schema()


def export_probes_jsonl(results: Sequence[EpistemicProbeResult], path: str) -> int:
    """Serialize `EpistemicProbeResult` records as UTF-8 JSONL for lab-decide / labshift."""
    with open(path, mode="w", encoding="utf-8") as fh:
        for item in results:
            fh.write(item.to_json() + "\n")
    return len(results)


def load_probes_jsonl(path: str) -> list[EpistemicProbeResult]:
    with open(path, mode="r", encoding="utf-8") as fh:
        return [EpistemicProbeResult.from_dict(json.loads(line)) for line in fh if line.strip()]
