"""Unit tests for the Auto-Lab 15-repository wire node (autolab-wire/v1) in lab-ask-why."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import autolab_wire  # noqa: E402
from autolab_wire import (  # noqa: E402
    DEFAULT_WETLAB_EVIDENCE_SENTENCES,
    LAYER,
    REPO_DIR,
    REPO_ID,
    process_packet,
)


def test_wire_metadata_and_cwd_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert REPO_ID == "lab-ask-why"
    assert LAYER == "L6"
    assert REPO_DIR == REPO_ROOT
    assert len(DEFAULT_WETLAB_EVIDENCE_SENTENCES) == 6

    # Execute from an empty temporary directory with network blocked to verify
    # zero os.getcwd() reliance and zero network downloads.
    monkeypatch.chdir(tmp_path)
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=AssertionError("Network access must never occur in autolab_wire"),
    ):
        res = process_packet({})

    assert res["schema_version"] == "autolab-wire/v1"
    assert res["halted"] is False
    assert res["halt_reason"] is None
    assert len(res["epistemic_probe"]["support_top"]) >= 1
    assert len(res["epistemic_probe"]["contradict_top"]) >= 1
    assert json.loads(json.dumps(res))["epistemic_probe"]["has_discriminating_experiment"] is True


def test_normal_parsed_spans_handoff_from_lab_ears() -> None:
    upstream_packet = {
        "schema_version": "autolab-wire/v1",
        "halted": False,
        "halt_reason": None,
        "goal_text": (
            "Transfer 100 uL buffered aspirin solution, incubate at 37 C for 15 min "
            "at 300 rpm to assay aqueous solubility."
        ),
        "parsed_spans": {
            "spans": [
                {"label": "Amount", "text": "100 uL"},
                {"label": "Temperature", "text": "37 C"},
                {"label": "Time", "text": "15 min"},
                {"label": "Speed", "text": "300 rpm"},
            ],
            "amounts": ["100 uL"],
            "temperatures": ["37 C"],
            "times": ["15 min"],
            "speeds": ["300 rpm"],
            "volume_ul": 100.0,
            "temp_c": 37.0,
            "duration_min": 15.0,
        },
        "speech_band_dropped": True,
        "trace": [
            {
                "repo": "lab-ears",
                "layer": "L2",
                "status": "PASS",
                "summary": "Parsed 4 protocol spans and stripped [300, 3400] Hz speech band.",
            }
        ],
        "bottom_reasoning_traces": [
            {
                "repo": "lab-ears",
                "invariant": "full_skirt_speech_band_drop",
                "first_principles": "Speech formants stripped prior to feature storage.",
            }
        ],
    }

    out = process_packet(upstream_packet)

    # Verify deep-copy isolation: upstream packet is untouched
    assert "epistemic_probe" not in upstream_packet
    assert len(upstream_packet["trace"]) == 1
    assert len(upstream_packet["bottom_reasoning_traces"]) == 1

    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None

    probe = out["epistemic_probe"]
    assert len(probe["support_top"]) >= 1
    assert len(probe["contradict_top"]) >= 1
    assert len(probe["why_questions"]) == 3
    assert probe["discriminating_question"] == probe["why_questions"][-1]
    assert "change your mind" in probe["discriminating_question"]
    assert probe["has_discriminating_experiment"] is True
    assert probe["epistemic_status"] == "CONTESTED"
    assert 0.0 < probe["retrieval_confidence"] <= 1.0
    assert probe["abstain_recommended"] is False
    assert probe["discriminating_observable"]["candidate_descriptors"]["volume_ul"] == 100.0
    assert probe["discriminating_observable"]["candidate_descriptors"]["temp_c"] == 37.0
    assert probe["discriminating_observable"]["candidate_descriptors"]["duration_min"] == 15.0
    assert len(probe["control_wells"]) >= 3
    assert any(cw["role"] == "VEHICLE_BLANK" for cw in probe["control_wells"])
    assert any(cw["role"] == "PERTURBATION_ARM" for cw in probe["control_wells"])
    assert any("solubility" in s.lower() or "ester" in s.lower() for s in probe["support_top"])
    assert any("hydrolysis" in s.lower() or "not" in s.lower() or "without" in s.lower() for s in probe["contradict_top"])

    assert len(out["trace"]) == 2
    assert out["trace"][-1]["repo"] == "lab-ask-why"
    assert out["trace"][-1]["layer"] == "L6"
    assert out["trace"][-1]["status"] == "PASS"

    assert len(out["bottom_reasoning_traces"]) == 2
    brt = out["bottom_reasoning_traces"][-1]
    assert brt["repo"] == "lab-ask-why"
    assert brt["invariant"] == "contested_claim_requires_experiment"
    assert "control" in brt["first_principles"].lower()


def test_contested_claim_blocks_when_discriminating_experiment_suppressed() -> None:
    packet = {
        "goal_text": "Incubate 100 uL aspirin at 37 C for 15 min to confirm aqueous solubility.",
        "parsed_spans": {
            "volume_ul": 100.0,
            "temp_c": 37.0,
            "duration_min": 15.0,
        },
        "suppress_discriminating_experiment": True,
    }

    out = process_packet(packet)

    assert out["halted"] is True
    assert out["halt_reason"] == "L6_CONTESTED_WITHOUT_EXPERIMENT"

    probe = out["epistemic_probe"]
    assert len(probe["support_top"]) >= 1
    assert len(probe["contradict_top"]) >= 1
    assert probe["discriminating_question"] == probe["why_questions"][-1]
    assert probe["has_discriminating_experiment"] is False

    assert out["trace"][-1]["repo"] == "lab-ask-why"
    assert out["trace"][-1]["layer"] == "L6"
    assert out["trace"][-1]["status"] == "BLOCK"
    assert (
        out["bottom_reasoning_traces"][-1]["invariant"]
        == "contested_claim_requires_experiment"
    )


def test_custom_evidence_sentences_uncontested_vs_contested() -> None:
    uncontested_packet = {
        "goal_text": "Phosphate buffer increases aspirin dissolution rate.",
        "evidence_sentences": [
            "Phosphate buffer at pH 7.4 increases aspirin dissolution rate in 96-well plates.",
            "Gentle orbital mixing further enhances buffered aspirin dissolution kinetics.",
        ],
        "suppress_discriminating_experiment": True,
    }
    res_uncontested = process_packet(uncontested_packet)
    assert res_uncontested["epistemic_probe"]["support_top"]
    assert res_uncontested["epistemic_probe"]["contradict_top"] == []
    assert res_uncontested["epistemic_probe"]["epistemic_status"] == "SUPPORTED"
    assert res_uncontested["halted"] is False
    assert res_uncontested["halt_reason"] is None

    contested_custom_packet = {
        "goal_text": "Phosphate buffer increases aspirin dissolution rate.",
        "evidence_sentences": [
            "Phosphate buffer at pH 7.4 increases aspirin dissolution rate in 96-well plates.",
            "Phosphate buffer without stabilizers does not prevent rapid aspirin ester hydrolysis.",
        ],
        "suppress_discriminating_experiment": True,
    }
    res_contested = process_packet(contested_custom_packet)
    assert res_contested["epistemic_probe"]["support_top"]
    assert res_contested["epistemic_probe"]["contradict_top"]
    assert res_contested["halted"] is True
    assert res_contested["halt_reason"] == "L6_CONTESTED_WITHOUT_EXPERIMENT"


def test_issue_24_and_28_neutral_exclusion_and_zero_overlap_insufficient_evidence() -> None:
    neutral_pkt = process_packet(
        {
            "goal_text": "Samples incubated at 37 C dissolve.",
            "evidence_sentences": [
                {"sentence": "Samples were incubated at 37 C.", "polarity": "NEUTRAL"}
            ],
        }
    )
    assert neutral_pkt["epistemic_probe"]["support_top"] == []
    assert neutral_pkt["epistemic_probe"]["contradict_top"] == []
    assert neutral_pkt["epistemic_probe"]["epistemic_status"] == "INSUFFICIENT_EVIDENCE"
    assert neutral_pkt["epistemic_probe"]["abstain_recommended"] is True

    zero_overlap_pkt = process_packet(
        {
            "goal_text": "Quantum Hall conductance plateau in graphene.",
            "evidence_sentences": ["PCR Taq polymerase denatures at 98 C."],
        }
    )
    z_probe = zero_overlap_pkt["epistemic_probe"]
    assert z_probe["support_top"] == []
    assert z_probe["contradict_top"] == []
    assert z_probe["epistemic_status"] == "INSUFFICIENT_EVIDENCE"
    assert z_probe["abstain_recommended"] is True
    assert z_probe["has_discriminating_experiment"] is False
    assert z_probe["discriminating_question"] == z_probe["why_questions"][-1]
    assert zero_overlap_pkt["trace"][-1]["status"] == "ABSTAIN"

    strict_pkt = process_packet(
        {
            "goal_text": "Quantum Hall conductance plateau in graphene.",
            "evidence_sentences": ["PCR Taq polymerase denatures at 98 C."],
            "strict_ask_why": True,
        }
    )
    assert strict_pkt["halted"] is True
    assert strict_pkt["halt_reason"] == "L6_INSUFFICIENT_EVIDENCE_ABSTAIN"
    assert strict_pkt["trace"][-1]["status"] == "BLOCK"


def test_issue_28_closed_loop_lab_attrib_certificates_and_orthogonal_probe() -> None:
    # 1. CERTIFIED_NEGATIVE disproof turns uncontested prior claim into CONTESTED
    # while EXECUTION_ARTIFACT and malformed-SHA entries are quarantined from memory.
    retry_pkt = process_packet(
        {
            "goal_text": "Phosphate buffer increases aspirin dissolution rate.",
            "parsed_spans": {"volume_ul": 120.0, "temp_c": 37.0, "duration_min": 20.0},
            "evidence_sentences": [
                "Phosphate buffer at pH 7.4 increases aspirin dissolution rate in 96-well plates."
            ],
            "certified_ledger_entries": [
                {
                    "claim": "Phosphate buffer increases aspirin dissolution rate.",
                    "attribution_outcome": "CERTIFIED_NEGATIVE",
                    "summary": (
                        "Verified negative readout: pH 7.4 incubation hydrolyzed aspirin "
                        "without increasing parent concentration."
                    ),
                    "sha256": "a" * 64,
                },
                {
                    "claim": "Phosphate buffer increases aspirin dissolution rate.",
                    "attribution_outcome": "EXECUTION_ARTIFACT",
                    "summary": "Clogged pipette tip produced false zero absorbance.",
                    "sha256": "b" * 64,
                },
                {
                    "claim": "Phosphate buffer increases aspirin dissolution rate.",
                    "attribution_outcome": "CERTIFIED_POSITIVE",
                    "summary": "Tampered entry with invalid digest.",
                    "sha256": "not-a-valid-sha256",
                },
            ],
            "suppress_discriminating_experiment": True,
        }
    )
    r_probe = retry_pkt["epistemic_probe"]
    assert len(r_probe["support_top"]) == 1
    assert len(r_probe["contradict_top"]) == 1
    assert "Verified negative readout" in r_probe["contradict_top"][0]
    assert all("Clogged pipette" not in s for s in r_probe["support_top"] + r_probe["contradict_top"])
    assert r_probe["attribution_feedback"]["certified_negative_ingested"] == 1
    assert r_probe["attribution_feedback"]["quarantined_execution_artifacts"] == 2
    assert r_probe["epistemic_status"] == "CONTESTED"
    assert retry_pkt["halted"] is True
    assert retry_pkt["halt_reason"] == "L6_CONTESTED_WITHOUT_EXPERIMENT"

    # 2. AMBIGUOUS_CHARACTERIZATION routes to orthogonal Raman / LC-MS characterization probe
    ambig_pkt = process_packet(
        {
            "goal_text": "Solid-state calcination yields pure target polymorph.",
            "evidence_sentences": [
                "Calcination at 700 C yields putative target Bragg peaks."
            ],
            "attribution_outcome": "AMBIGUOUS_CHARACTERIZATION",
        }
    )
    a_probe = ambig_pkt["epistemic_probe"]
    assert a_probe["orthogonal_characterization_required"] is True
    assert a_probe["recommended_orthogonal_modalities"] == ["Raman", "LC-MS"]
    assert "Raman" in a_probe["discriminating_question"]
    assert "LC-MS" in a_probe["discriminating_question"]
    assert (
        a_probe["discriminating_observable"]["observable_name"]
        == "orthogonal_raman_lcms_characterization"
    )


def test_halted_packet_short_circuits_unchanged() -> None:
    halted_packet = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L2_SPEECH_PRIVACY_BLOCK",
        "goal_text": "Transfer 100 uL aspirin solution at 37 C.",
        "trace": [
            {
                "repo": "lab-ears",
                "layer": "L2",
                "status": "BLOCK",
                "summary": "Retained band overlaps [300, 3400] Hz speech formants.",
            }
        ],
        "bottom_reasoning_traces": [],
    }

    out = process_packet(halted_packet)

    assert out == halted_packet
    assert out is not halted_packet
    assert out["trace"] is not halted_packet["trace"]
    assert "epistemic_probe" not in out
    assert out["halt_reason"] == "L2_SPEECH_PRIVACY_BLOCK"
    assert len(out["trace"]) == 1
    assert autolab_wire.REPO_ID == "lab-ask-why"
