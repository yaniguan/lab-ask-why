"""Offline unit tests for 2019-2026 open scientific dataset registry, guards, and schema adapters."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any
from unittest import mock

import jsonschema
import pytest

from lab_ask_why import (
    EPISTEMIC_PROBE_JSON_SCHEMA,
    EVAL_ONLY_CANARY_NEEDLES,
    OPEN_DATASET_REGISTRY,
    OPEN_DATASET_SPECS,
    VALID_NORMALIZED_LABELS,
    NormalizedClaimRecord,
    OpenDatasetSpec,
    adapt_record,
    adapt_records,
    build_corpus_and_claims_from_records,
    contains_eval_canary,
    enforce_dataset_guard,
    format_dataset_registry_table,
    get_dataset_spec,
    list_open_datasets,
    normalize_veracity_label,
    probe_dataset_records,
    probe_normalized_record,
)
from lab_ask_why.cli import evaluate_scifact_benchmark, main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def _load_fetch_script_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "fetch_open_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_open_datasets", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_completeness_spdx_and_2024_2026_coverage() -> None:
    assert len(OPEN_DATASET_SPECS) >= 14
    assert len(OPEN_DATASET_REGISTRY) == len(OPEN_DATASET_SPECS)

    modern_specs = list_open_datasets(min_year=2024)
    assert len(modern_specs) >= 10

    required_ids = {
        "labbench2",
        "peerqa_xt",
        "sciver",
        "claimcheck",
        "chembench",
        "litqa2",
        "lab_bench",
        "conflictbank",
        "faitheval_inconsistent",
        "sciriff",
        "bioasq",
        "scifact_open",
        "scifact",
        "pubmedqa",
    }
    assert required_ids.issubset(OPEN_DATASET_REGISTRY.keys())

    allowed_spdx = {
        "MIT",
        "Apache-2.0",
        "CC-BY-2.5",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-NC-2.0",
        "CC-BY-NC-SA-4.0",
        "ODC-BY-1.0",
        "NOASSERTION",
    }
    for spec in OPEN_DATASET_SPECS:
        assert spec.url.startswith("https://")
        assert "/" in spec.hf_or_git_id
        assert spec.spdx_license in allowed_spdx
        if "NC" in spec.spdx_license or spec.spdx_license == "NOASSERTION":
            assert spec.commercial_use_ok is False
        else:
            assert spec.commercial_use_ok is True

    # Alias lookup coverage
    assert get_dataset_spec("LAB-Bench").dataset_id == "lab_bench"
    assert get_dataset_spec("LitQA2").dataset_id == "litqa2"
    assert get_dataset_spec("LitQA3").dataset_id == "labbench2"
    assert get_dataset_spec("jablonkagroup/ChemBench").dataset_id == "chembench"
    assert get_dataset_spec("SciFact-Open").dataset_id == "scifact_open"
    assert get_dataset_spec("Warrieryes/CB_qa").dataset_id == "conflictbank"

    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset_spec("nonexistent-dataset-xyz")


def test_open_dataset_spec_validation_invariants() -> None:
    with pytest.raises(ValueError, match="commercial_use_ok=True"):
        OpenDatasetSpec(
            dataset_id="bad_nc",
            title="Invalid NC Spec",
            year=2025,
            url="https://huggingface.co/datasets/example/bad",
            hf_or_git_id="example/bad",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=True,
            eval_only=False,
        )

    with pytest.raises(ValueError, match="https://"):
        OpenDatasetSpec(
            dataset_id="bad_http",
            title="Insecure URL",
            year=2025,
            url="http://example.com/dataset",
            hf_or_git_id="example/bad",
            spdx_license="MIT",
            commercial_use_ok=True,
            eval_only=False,
        )


@pytest.mark.parametrize("eval_ds", ["lab_bench", "litqa2", "labbench2", "chembench"])
@pytest.mark.parametrize("bad_purpose", ["train", "sft", "rlvr", "pretrain", "distill"])
def test_eval_only_guards_block_non_eval_purposes(eval_ds: str, bad_purpose: str) -> None:
    spec = get_dataset_spec(eval_ds)
    assert spec.eval_only is True

    # Eval mode succeeds cleanly
    assert enforce_dataset_guard(eval_ds, purpose="eval").dataset_id == spec.dataset_id

    # Any training/SFT/RL purpose raises PermissionError immediately
    with pytest.raises(PermissionError, match="eval_only=True"):
        enforce_dataset_guard(eval_ds, purpose=bad_purpose)

    with pytest.raises(PermissionError, match="eval_only=True"):
        adapt_record(
            eval_ds,
            {"id": "1", "question": "Test question?", "ideal": "Answer", "key-passage": "Evidence."},
            purpose=bad_purpose,
        )


def test_canary_and_commercial_license_guards() -> None:
    # Canary needle embedded inside a permissive dataset still blocks training ingestion
    poisoned_row = {
        "pubid": "999",
        "question": "Does compound X inhibit kinase Y?",
        "context": {"contexts": [f" Marker {EVAL_ONLY_CANARY_NEEDLES[-1]} embedded in text."]},
        "final_decision": "yes",
    }
    assert contains_eval_canary(poisoned_row) is True
    with pytest.raises(PermissionError, match="canary"):
        adapt_record("pubmedqa", poisoned_row, purpose="train")

    # Non-commercial datasets are blocked when require_commercial_ok=True
    for nc_ds in ("scifact", "scifact_open", "peerqa_xt", "faitheval_inconsistent", "claimcheck"):
        with pytest.raises(PermissionError, match="commercial_use_ok=False"):
            enforce_dataset_guard(nc_ds, purpose="eval", require_commercial_ok=True)

    # Permissive datasets succeed under require_commercial_ok=True
    for ok_ds in ("pubmedqa", "sciver", "conflictbank", "sciriff", "bioasq"):
        assert enforce_dataset_guard(ok_ds, purpose="train", require_commercial_ok=True).commercial_use_ok


def test_all_14_dataset_schema_adapters_offline_mini_fixtures() -> None:
    mini_corpus = {
        501: {
            "doc_id": 501,
            "title": "Pooled S2ORC Abstract on Ferroptosis",
            "abstract": [
                "GPX4 reduces lipid hydroperoxides using glutathione.",
                "Pharmacological inhibition of GPX4 by RSL3 triggers iron-dependent ferroptosis.",
                "Ferrostatin-1 rescues cell viability without restoring GPX4 expression.",
            ],
        }
    }

    fixtures: list[tuple[str, dict[str, Any], dict[Any, dict[str, Any]] | None, str]] = [
        # 1. SciFact-Orig (list-valued evidence[doc_id])
        (
            "scifact",
            {
                "id": 11,
                "claim": "RSL3 inhibits GPX4 to induce iron-dependent ferroptosis.",
                "evidence": {"501": [{"sentences": [1], "label": "SUPPORT"}]},
                "cited_doc_ids": [501],
            },
            mini_corpus,
            "SUPPORT",
        ),
        # 2. SciFact-Open (dict-valued evidence[doc_id] with provenance='pooling')
        (
            "scifact_open",
            {
                "id": 170,
                "claim": "Ferrostatin-1 restores GPX4 protein expression in RSL3-treated cells.",
                "evidence": {
                    "501": {
                        "provenance": "pooling",
                        "label": "CONTRADICT",
                        "sentences": [2],
                        "model_ranks": {"multivers": 1},
                    }
                },
            },
            mini_corpus,
            "CONTRADICT",
        ),
        # 3. LitQA2
        (
            "litqa2",
            {
                "id": "litqa2-001",
                "question": "Which antibiotic did Acinetobacter lwoffii evolve resistance against?",
                "ideal": "ciprofloxacin",
                "distractors": ["meropenem", "gentamicin"],
                "key-passage": (
                    "A. baumannii readily evolved resistance to meropenem and gentamicin. "
                    "In contrast, A. lwoffii evolved high-level resistance exclusively to ciprofloxacin."
                ),
                "subtask": "litqa-v2-public",
            },
            None,
            "SUPPORT",
        ),
        # 4. LAB-Bench ProtocolQA (probing distractor -> CONTRADICT)
        (
            "lab_bench",
            {
                "id": "protoqa-001",
                "question": "How should you prevent PBMC osmotic lysis in step 2?",
                "ideal": "Add 0.85 g NaCl to maintain isotonicity.",
                "distractors": ["Omit NaCl completely to keep buffer hypotonic."],
                "protocol": "Step 1: Prepare 100 mM Tricine buffer. Step 2: Add 0.85 g NaCl per 100 mL water to prevent osmotic lysis of PBMCs.",
                "probe_distractor": True,
            },
            None,
            "CONTRADICT",
        ),
        # 5. LABBench2 (LitQA3 / insufficient info -> NOINFO)
        (
            "labbench2",
            {
                "id": "lb2-001",
                "question": "What is the K_d of mutant Z99A reported in Table S4?",
                "ideal": "Insufficient information to answer this question",
                "context": "Table S4 reports wild-type K_d = 12 nM at 25 C. Mutant Z99A was not purified.",
                "subtask": "litqa3",
            },
            None,
            "NOINFO",
        ),
        # 6. ChemBench (JSON string target_scores)
        (
            "chembench",
            {
                "uuid": "chem-001",
                "name": "nmr_aldehyde_signals",
                "subfield": "analytical_chemistry",
                "description": "1H NMR splitting in tert-butylacetaldehyde shows two distinct proton environments (tert-butyl singlet and CH2/CHO coupled signals).",
                "examples": [
                    {
                        "input": "How many distinct 1H NMR signal groups appear for (CH3)3CCH2CHO?",
                        "target_scores": '{"2": 0, "3": 1, "4": 0}',
                    }
                ],
            },
            None,
            "SUPPORT",
        ),
        # 7. SciVer (multimodal scientific claim verification)
        (
            "sciver",
            {
                "paperid": "2409.07233v1",
                "request_id": "2",
                "claim_type": "analytical",
                "type": "table",
                "origin_statement": "Catalyst Ni-MOF increases CO2 Faradaic efficiency to 94.2% at -0.8 V.",
                "perturbed_statement": "Catalyst Ni-MOF decreases CO2 Faradaic efficiency below 50% at -0.8 V.",
                "perturbed_explanation": "Table 2 shows Ni-MOF reaches 94.2% Faradaic efficiency rather than decreasing below 50%.",
                "claim": "Catalyst Ni-MOF decreases CO2 Faradaic efficiency below 50% at -0.8 V.",
                "label": "False",
            },
            None,
            "CONTRADICT",
        ),
        # 8. CLAIMCHECK (nested OpenReview critique JSON)
        (
            "claimcheck",
            {
                "id": "neurips24_rev1",
                "response": {
                    "Weakness associated with claims": [
                        {
                            "Weakness span": "The authors claim O(N) memory scaling, yet Equation 7 materializes an N x N attention matrix requiring O(N^2) memory.",
                            "Target claims": ["Our layer achieves strict O(N) memory complexity in sequence length N."],
                            "Weakness Annotation": {
                                "weakness_type": {"contradictory": True, "insufficient": False}
                            },
                        }
                    ]
                },
                "meta": {
                    "title": "Subquadratic Sequence Modeling",
                    "abstract": "We propose a linear-memory sequence architecture achieving O(N) scaling.",
                },
            },
            None,
            "CONTRADICT",
        ),
        # 9. ConflictBank (CB_qa paired parametric + conflict evidence)
        (
            "conflictbank",
            {
                "id": "cb-qa-1",
                "subject": "Aspirin Ester Stability",
                "default_claim": "Buffered pH 6.8 medium prevents rapid aspirin ester hydrolysis at 25 C.",
                "default_evidence": "Buffered pH 6.8 medium stabilizes acetylsalicylic acid and prevents ester hydrolysis at 25 C.",
                "misinformation_conflict_evidence_evidence": "Even in pH 6.8 buffer, heating to 80 C accelerates aspirin hydrolysis into salicylic acid.",
            },
            None,
            "CONTRADICT",
        ),
        # 10. FaithEval-Inconsistent (conflicting multi-document answers)
        (
            "faitheval_inconsistent",
            {
                "qid": "fe-001",
                "question": "Which kinase phosphorylates substrate S14 in primary hepatocytes?",
                "context": (
                    "Document 1: Mass spectrometry confirms AMPK directly phosphorylates substrate S14 in hepatocytes. "
                    "Document 2: Genetic knockout proves mTORC1, and not AMPK, phosphorylates substrate S14 in hepatocytes."
                ),
                "answers": ["AMPK", "mTORC1"],
                "subset": "BioMed",
                "justification": "Documents 1 and 2 report contradictory kinases.",
            },
            None,
            "CONTRADICT",
        ),
        # 11. SciRIFF
        (
            "sciriff",
            {
                "_instance_id": "scifact_entailment:val:4",
                "input": (
                    "Abstract: CRISPR-Cas9 endonuclease requires an NGG PAM sequence. "
                    "PAM-proximal seed mismatches abolish double-strand DNA cleavage.\n"
                    "Claim: Seed-region mismatches abolish CRISPR-Cas9 DNA cleavage."
                ),
                "output": '{"verdict": "SUPPORT"}',
                "metadata": {"task_family": "claim_verification", "domains": ["biomedicine"]},
            },
            None,
            "SUPPORT",
        ),
        # 12. PeerQA-XT
        (
            "peerqa_xt",
            {
                "qid": "pqa-xt-1",
                "pid": "bmc/s12859-025-001",
                "question": "Did the authors control for batch effects across scRNA-seq donors?",
                "answer": "Yes, Harmony integration was applied across all 12 donor libraries prior to clustering.",
                "domain": "biomedical",
            },
            None,
            "SUPPORT",
        ),
        # 13. PubMedQA
        (
            "pubmedqa",
            {
                "pubid": 25429730,
                "question": "Are group 2 innate lymphoid cells elevated in chronic rhinosinusitis with nasal polyps?",
                "context": {
                    "contexts": [
                        "Chronic rhinosinusitis with nasal polyps (CRSwNP) exhibits type-2 mucosal inflammation.",
                        "Flow cytometry showed ILC2 frequencies were significantly elevated in CRSwNP mucosa versus controls.",
                    ]
                },
                "long_answer": "ILC2s are markedly elevated in CRSwNP patients.",
                "final_decision": "yes",
            },
            None,
            "SUPPORT",
        ),
        # 14. BioASQ
        (
            "bioasq",
            {
                "id": "bioasq-42",
                "question": "Does metformin activate AMP-activated protein kinase in hepatocytes?",
                "exact_answer": "yes",
                "ideal_answer": "Metformin inhibits complex I and activates hepatic AMP-activated protein kinase (AMPK).",
                "snippets": [
                    {
                        "text": "Metformin activates AMP-activated protein kinase (AMPK) in hepatocytes to suppress gluconeogenesis."
                    }
                ],
            },
            None,
            "SUPPORT",
        ),
    ]

    normalized_all: list[NormalizedClaimRecord] = []
    for ds_id, raw_row, corp, expected_label in fixtures:
        rec = adapt_record(ds_id, raw_row, purpose="eval", corpus=corp)
        assert rec.dataset_id == ds_id
        assert rec.label == expected_label
        assert rec.label in VALID_NORMALIZED_LABELS
        assert len(rec.claim) > 5
        assert len(rec.evidence_sentences) >= 1

        probe = probe_normalized_record(rec, purpose="eval", top_k=3)
        jsonschema.validate(instance=probe.to_dict(), schema=EPISTEMIC_PROBE_JSON_SCHEMA)
        assert len(probe.why_questions) == 3
        normalized_all.append(rec)

    # Verify batch corpus/claims conversion plugs directly into evaluate_scifact_benchmark + 2-stage retriever
    corpus_map, scifact_claims = build_corpus_and_claims_from_records(normalized_all, purpose="eval")
    assert len(corpus_map) == 14
    assert len(scifact_claims) == 14
    bench = evaluate_scifact_benchmark(corpus_map, scifact_claims, idf_scope="full", top_k=3, doc_top_m=5)
    assert bench["total_claims"] == 14
    assert bench["oracle_metrics"]["bm25"]["hit@3"] > 0.7


def test_contested_conflict_record_triggers_symmetric_pro_con_and_control_wells() -> None:
    rec = adapt_record(
        "conflictbank",
        {
            "id": "cb-contested",
            "default_claim": "Aspirin remains chemically stable in aqueous solution at 25 C.",
            "default_evidence": "Buffered aqueous solution at 25 C keeps aspirin stable and prevents ester hydrolysis.",
            "misinformation_conflict_evidence_evidence": "Unbuffered water at 80 C does not keep aspirin stable and rapidly hydrolyzes the ester into salicylic acid.",
        },
        purpose="eval",
    )
    probe = probe_normalized_record(rec, purpose="eval", top_k=3)
    assert probe.epistemic_status == "CONTESTED"
    assert len(probe.supporting_rationales) >= 1
    assert len(probe.contradicting_rationales) >= 1
    assert len(probe.discriminating_observable.control_wells) >= 3

    # Empty-evidence NOINFO record abstains cleanly with 0.0 confidence
    empty_rec = NormalizedClaimRecord(
        dataset_id="pubmedqa",
        record_id="empty-1",
        claim="Does hypothetical peptide Z99 extend murine lifespan?",
        evidence_sentences=[],
        label="NOINFO",
    )
    empty_probe = probe_normalized_record(empty_rec, purpose="eval")
    assert empty_probe.epistemic_status == "INSUFFICIENT_EVIDENCE"
    assert empty_probe.retrieval_confidence == 0.0


def test_cli_list_datasets_and_dataset_jsonl_probing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(["--list-datasets"])
    assert rc == 0
    out_table = capsys.readouterr().out
    assert "labbench2" in out_table
    assert "chembench" in out_table
    assert format_dataset_registry_table()[:30] in out_table

    rc_json = cli_main(["--list-datasets", "--json"])
    assert rc_json == 0
    parsed = json.loads(capsys.readouterr().out)
    assert isinstance(parsed, list) and len(parsed) >= 14

    sample_jsonl = tmp_path / "pubmedqa_mini.jsonl"
    sample_jsonl.write_text(
        json.dumps(
            {
                "pubid": "101",
                "question": "Does GPX4 inhibition induce ferroptosis in renal tubular cells?",
                "context": {
                    "contexts": [
                        "RSL3-mediated GPX4 inhibition induces lipid peroxidation and ferroptosis in renal tubular cells."
                    ]
                },
                "final_decision": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    probes_out = tmp_path / "probes.jsonl"
    rc_probe = cli_main(
        [
            "--dataset",
            "pubmedqa",
            "--dataset-jsonl",
            str(sample_jsonl),
            "--export-probes",
            str(probes_out),
            "--json",
        ]
    )
    assert rc_probe == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records_probed"] == 1
    assert probes_out.exists()

    # Passing --purpose train on an eval_only dataset exits with error code 2
    chem_jsonl = tmp_path / "chem_mini.jsonl"
    chem_jsonl.write_text(
        json.dumps(
            {
                "uuid": "c1",
                "description": "Molarity calculation.",
                "examples": [{"input": "Molarity of 1 mol NaCl in 1 L?", "target": "1 M"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc_blocked = cli_main(
        ["--dataset", "chembench", "--dataset-jsonl", str(chem_jsonl), "--purpose", "train"]
    )
    assert rc_blocked == 2
    _ = capsys.readouterr()


def test_fetch_open_datasets_script_offline_mock_and_guards(tmp_path: Path) -> None:
    mod = _load_fetch_script_module()
    fake_hf_payload = {
        "rows": [
            {
                "row": {
                    "pubid": 777,
                    "question": "Does CRISPR-Cas9 require an NGG PAM motif for cleavage?",
                    "context": {
                        "contexts": [
                            "Canonical SpCas9 endonuclease cleavage requires recognition of an NGG PAM motif."
                        ]
                    },
                    "long_answer": "Yes, SpCas9 requires an NGG PAM.",
                    "final_decision": "yes",
                }
            }
        ]
    }
    with mock.patch.object(mod, "_http_get_json", return_value=fake_hf_payload):
        rc = mod.main(
            [
                "--dataset",
                "pubmedqa",
                "--max-rows",
                "1",
                "--output-dir",
                str(tmp_path),
                "--probe",
                "--json",
            ]
        )
    assert rc == 0
    assert (tmp_path / "pubmedqa" / "raw_sample.jsonl").exists()
    assert (tmp_path / "pubmedqa" / "normalized_claims.jsonl").exists()

    # eval_only guard enforced in fetch_open_datasets.py before any network call
    rc_eval_guard = mod.main(
        ["--dataset", "litqa2", "--purpose", "train", "--output-dir", str(tmp_path)]
    )
    assert rc_eval_guard == 2


def test_normalize_veracity_label_variants_and_zero_cjk_in_repo() -> None:
    assert normalize_veracity_label("ENTAILED") == "SUPPORT"
    assert normalize_veracity_label("Refuted") == "CONTRADICT"
    assert normalize_veracity_label("maybe") == "NOINFO"
    assert normalize_veracity_label(True) == "SUPPORT"
    assert normalize_veracity_label(False) == "CONTRADICT"
    with pytest.raises(ValueError, match="Unrecognized veracity label"):
        normalize_veracity_label("totally_invalid_verdict")

    # Enforce 100% pure English (zero CJK characters) across code, scripts, tests, and docs
    checked_files = [
        *REPO_ROOT.glob("src/**/*.py"),
        *REPO_ROOT.glob("scripts/**/*.py"),
        *REPO_ROOT.glob("tests/**/*.py"),
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md",
    ]
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert CJK_RE.search(text) is None, f"Found CJK character in {path}"
