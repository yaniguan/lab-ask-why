"""Comprehensive offline unit & regression test suite for lab-ask-why (Issues #1..#22)."""
from __future__ import annotations

import io
import json
import math
import tarfile
import warnings
from pathlib import Path
from unittest import mock

import jsonschema
import numpy as np
import pytest

from lab_ask_why import (
    BM25,
    CANARY_NEEDLES,
    EPISTEMIC_PROBE_JSON_SCHEMA,
    STOP,
    CounterfactualProbeScorer,
    EpistemicProbeResult,
    TwoStageRetriever,
    build_noinfo_pairs,
    build_pairs,
    classify_rationale_polarity,
    compute_auroc,
    conclusion_first_rankings,
    ensure_data,
    export_probes_jsonl,
    hit_at,
    lead_rankings,
    load_labbench_items,
    load_probes_jsonl,
    probe_labbench_item,
    rank_scores,
    rationale_set_recall_at,
    read_jsonl,
    recall_at,
    safe_extract_tar,
    sentence_recall_at,
    tokenize,
    why_questions,
)
from lab_ask_why.cli import evaluate_scifact_benchmark, format_benchmark_text, main as cli_main
from lab_ask_why.data import download_file_atomic

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tar_bytes(entries: list[tuple[str, bytes, int, str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload, mtype, linkname in entries:
            info = tarfile.TarInfo(name=name)
            info.type = mtype
            if mtype in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = linkname
                info.size = 0
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_issue_01_safe_tar_extract_blocks_traversal_and_appledouble(tmp_path: Path) -> None:
    malicious_traversal = _make_tar_bytes([("../escape.txt", b"pwn", tarfile.REGTYPE, "")])
    with tarfile.open(fileobj=io.BytesIO(malicious_traversal), mode="r:gz") as tar:
        with pytest.raises(ValueError, match="traversal|escapes"):
            safe_extract_tar(tar, str(tmp_path / "dest"))

    malicious_symlink = _make_tar_bytes([("data/evil_link", b"", tarfile.SYMTYPE, "/etc/passwd")])
    with tarfile.open(fileobj=io.BytesIO(malicious_symlink), mode="r:gz") as tar:
        with pytest.raises(ValueError, match="symlink"):
            safe_extract_tar(tar, str(tmp_path / "dest2"))

    clean_with_appledouble = _make_tar_bytes(
        [
            ("._data", b"appledouble", tarfile.REGTYPE, ""),
            ("data/._claims_dev.jsonl", b"appledouble2", tarfile.REGTYPE, ""),
            ("data/claims_dev.jsonl", b'{"id": 1, "claim": "ok", "evidence": {}}\n', tarfile.REGTYPE, ""),
        ]
    )
    dest = tmp_path / "clean_dest"
    with tarfile.open(fileobj=io.BytesIO(clean_with_appledouble), mode="r:gz") as tar:
        extracted = safe_extract_tar(tar, str(dest))
    assert extracted == ["data/claims_dev.jsonl"]
    assert not (dest / "._data").exists()
    assert (dest / "data" / "claims_dev.jsonl").exists()


def test_issue_02_atomic_download_sha256_and_corrupt_archive_unlink(tmp_path: Path) -> None:
    corrupt_root = tmp_path / "corrupt_cache"
    corrupt_root.mkdir()
    archive = corrupt_root / "data.tar.gz"
    archive.write_bytes(b"truncated-not-a-valid-gzip")

    with pytest.raises(RuntimeError, match="Corrupt archive|SHA-256"):
        ensure_data(root=str(corrupt_root), expected_sha256=None, offline=True)
    assert not archive.exists()

    with pytest.raises(FileNotFoundError, match="offline"):
        ensure_data(root=str(tmp_path / "empty_root"), offline=True)

    target = tmp_path / "dl" / "data.tar.gz"
    fake_resp = io.BytesIO(b"hello world")
    fake_resp.__enter__ = lambda self: self  # type: ignore[attr-defined]
    fake_resp.__exit__ = lambda self, *a: False  # type: ignore[attr-defined]
    with mock.patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download_file_atomic("https://example.org/data.tar.gz", str(target), expected_sha256="0" * 64)
    assert not target.exists()
    assert list(target.parent.glob("*.part*")) == []


def test_issue_03_hit_at_vs_sentence_recall_vs_rationale_set_recall() -> None:
    corpus = {10: {"title": "Doc 10", "abstract": [f"s{i}" for i in range(8)]}}
    claims = [
        {
            "id": 1,
            "claim": "Test multi-sentence conjunction",
            "evidence": {"10": [{"sentences": [2, 5, 7], "label": "SUPPORT"}]},
        }
    ]
    pairs = build_pairs(claims, corpus)
    assert pairs[0]["rationale_sets"] == [frozenset({2, 5, 7})]
    rankings = [[2, 5, 0, 7]]
    assert hit_at(rankings, pairs, 1) == pytest.approx(1.0)
    assert recall_at(rankings, pairs, 1) == pytest.approx(1.0)
    assert sentence_recall_at(rankings, pairs, 1) == pytest.approx(1.0 / 3.0)
    assert rationale_set_recall_at(rankings, pairs, 1) == pytest.approx(0.0)
    assert rationale_set_recall_at(rankings, pairs, 4) == pytest.approx(1.0)


def test_issue_04_read_jsonl_utf8_greek_symbols(tmp_path: Path) -> None:
    p = tmp_path / "greek.jsonl"
    text = "Homozygous \u03b1+-Thalassaemia, \u03b2-catenin, 10 \u03bcg/mL \u2265 threshold"
    p.write_bytes((json.dumps({"title": text}, ensure_ascii=False) + "\n").encode("utf-8"))
    with mock.patch("locale.getpreferredencoding", return_value="cp1252"):
        rows = read_jsonl(str(p))
    assert rows[0]["title"] == text


def test_issue_05_bm25_empty_and_stopword_only_corpora() -> None:
    with pytest.raises(ValueError, match="at least one sentence"):
        BM25([])

    bm25_stop = BM25(["the and of in to", "a an is was"])
    assert bm25_stop.avgdl == 1.0
    sc = bm25_stop.scores("alpha thalassemia", ["the and of", ""])
    assert np.all(np.isfinite(sc)) and np.all(sc == 0.0)
    assert bm25_stop.scores("", ["alpha beta"]).tolist() == [0.0]
    assert bm25_stop.scores("alpha", []).size == 0


def test_issue_06_bm25_query_term_frequency_saturation() -> None:
    sents = [
        "thalassemia patients carry globin deletions",
        "rare penetrance modifier alters hemoglobin switching",
    ]
    bm25 = BM25(sents, k1=1.2, b=0.75, k3=1.2)
    s_once = bm25.scores("thalassemia", sents)[0]
    s_ten = bm25.scores(" ".join(["thalassemia"] * 10), sents)[0]
    assert s_once < s_ten < 2.2 * s_once

    bm25_dedup = BM25(sents, k3=0.0)
    assert bm25_dedup.scores("thalassemia", sents)[0] == pytest.approx(
        bm25_dedup.scores("thalassemia thalassemia thalassemia", sents)[0]
    )


def test_issue_07_bm25_background_idf_exclusion_and_oov_laplace() -> None:
    corpus = {
        1: {"abstract": ["common pathway regulates cell cycle"]},
        2: {"abstract": ["common pathway regulates apoptosis"]},
        99: {"abstract": ["zyxwvutsrq rare biomarker predicts response"]},
    }
    bm25_bg, spans = BM25.from_corpus(corpus, exclude_doc_ids=[99])
    assert 99 not in spans
    assert "zyxwvutsrq" not in bm25_bg.idf
    assert bm25_bg.default_idf == pytest.approx(math.log(1.0 + (2.0 + 0.5) / 0.5))
    assert bm25_bg.scores("zyxwvutsrq biomarker", corpus[99]["abstract"])[0] > 0.0


def test_issue_08_biomedical_tokenizer_greek_alleles_numbers_negations() -> None:
    for neg in ("not", "no", "neither", "without"):
        assert neg not in STOP

    t_greek = tokenize("Homozygous \u03b1+-Thalassaemia and \u03b2-catenin at 1.1 \u03bcg/mL")
    t_ascii = tokenize("homozygous alpha (+)- thalassemia and alpha(+)-thalassaemia")
    assert "alpha+" in t_greek and t_ascii.count("alpha+") == 2
    assert "beta" in t_greek and "micro" in t_greek and "1.1" in t_greek
    assert tokenize("1,000 genomes identified 2,500,000 variants") == [
        "1000",
        "genomes",
        "identified",
        "2500000",
        "variants",
    ]
    assert "not" in tokenize("Drug X does not inhibit kinase Y without cofactor")


def test_issue_09_why_questions_empty_validation_and_acronym_casing() -> None:
    for bad in ("", "   ", "...", "?!."):
        with pytest.raises(ValueError):
            why_questions(bad)

    assert "that GWAS identifies" in why_questions("GWAS identifies synthetic associations.")[0]
    assert "that RNA polymerase" in why_questions("RNA polymerase II stalls at damaged bases.")[0]
    assert "that IL-6 blockade" in why_questions("IL-6 blockade reduces cytokine release.")[0]
    assert "that COVID-19 patients" in why_questions("COVID-19 patients exhibit lymphopenia.")[0]
    assert "that a high microerythrocyte" in why_questions("A high microerythrocyte count raises risk.")[0]


def test_issue_10_noinfo_claims_loading_and_auroc_calibration() -> None:
    corpus = {
        1: {"title": "T1", "abstract": ["CRISPR Cas9 cleaves target DNA near NGG PAM."]},
        2: {"title": "T2", "abstract": ["Solar neutrino flux oscillates across flavors."]},
    }
    claims = [
        {"id": 1, "claim": "CRISPR Cas9 cleaves target DNA", "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
        {"id": 2, "claim": "CRISPR Cas9 increases telomere length", "evidence": {}, "cited_doc_ids": [2]},
    ]
    noinfo = build_noinfo_pairs(claims, corpus)
    assert len(noinfo) == 1 and noinfo[0]["label"] == "NOINFO"
    assert compute_auroc([12.5, 9.0], [1.2, 2.3]) == pytest.approx(1.0)
    assert compute_auroc([4.0], [4.0]) == pytest.approx(0.5)


def test_issue_11_build_pairs_defensive_validation() -> None:
    corpus = {1: {"title": "T1", "abstract": ["alpha pathway activates target", "second sentence"]}}
    with pytest.raises(ValueError, match="Empty rationale"):
        build_pairs([{"id": 1, "claim": "c", "evidence": {"1": []}}], corpus, strict=True)
    with pytest.raises(ValueError, match="missing|not found"):
        build_pairs(
            [{"id": 1, "claim": "c", "evidence": {"999": [{"sentences": [0], "label": "SUPPORT"}]}}],
            corpus,
            strict=True,
        )
    with pytest.raises(ValueError, match="Heterogeneous"):
        build_pairs(
            [
                {
                    "id": 1,
                    "claim": "c",
                    "evidence": {
                        "1": [
                            {"sentences": [0], "label": "SUPPORT"},
                            {"sentences": [1], "label": "CONTRADICT"},
                        ]
                    },
                }
            ],
            corpus,
            strict=True,
        )
    with pytest.raises(ValueError, match="out of bounds"):
        build_pairs(
            [{"id": 1, "claim": "c", "evidence": {"1": [{"sentences": [5], "label": "SUPPORT"}]}}],
            corpus,
            strict=True,
        )


def test_issue_12_empty_slice_and_single_class_benchmark_no_crash() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(hit_at([], [], 1))
        assert math.isnan(sentence_recall_at([], [], 3))
        assert math.isnan(rationale_set_recall_at([], [], 3))

    corpus = {1: {"title": "T1", "abstract": ["alpha pathway activates target", "second sentence"]}}
    single_class_claims = [
        {"id": 1, "claim": "alpha pathway activates target", "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
        {"id": 2, "claim": "unrelated quantum gravity hypothesis", "evidence": {}, "cited_doc_ids": [1]},
    ]
    bench = evaluate_scifact_benchmark(corpus, single_class_claims)
    rendered = format_benchmark_text(bench)
    assert "BM25 SUPPORT" in rendered
    assert len(bench["examples"]) == 1


def test_issue_13_inverted_index_parity_and_rank_candidates() -> None:
    rng = np.random.default_rng(7)
    vocab = ["alpha+", "beta", "kinase", "inhibitor", "mutation", "pathway", "not", "protective"]
    sents = [" ".join(rng.choice(vocab, size=6)) for _ in range(500)]
    bm25 = BM25(sents)
    q = "alpha+ kinase inhibitor does not block pathway"
    s_full_idx = bm25.score_indexed(q)
    s_full_raw = bm25.scores(q, sents)
    np.testing.assert_allclose(s_full_idx, s_full_raw, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bm25.score_indexed(q, list(range(25, 40))), s_full_raw[25:40], rtol=1e-12)
    np.testing.assert_allclose(bm25.score_indexed(q, [3, 19, 105, 499]), s_full_raw[[3, 19, 105, 499]], rtol=1e-12)
    top3 = bm25.rank_candidates(q, top_k=3)
    assert len(top3) == 3 and s_full_idx[top3[0]] >= s_full_idx[top3[1]] >= s_full_idx[top3[2]]


def test_issue_14_grounded_why_questions_and_llm_hook() -> None:
    grounded = why_questions(
        "Kinase inhibition blocks proliferation.",
        top_sentences=["Autophagy induction and phosphorylation loss drive cell arrest."],
        stance="CONTRADICT",
    )
    assert "autophagy" in grounded[0] or "phosphorylation" in grounded[0]
    assert "contradicting observation" in grounded[1]
    custom = why_questions(
        "Test claim.",
        top_sentences=["Evidence sentence."],
        llm_fn=lambda c, sents, st: [f"LLM:{c}:{len(sents)}:{st}"],
    )
    assert custom == ["LLM:Test claim:1:None"]


def test_issue_15_ci_workflow_caches_archive_runs_pytest_and_diffs_output() -> None:
    wf = (REPO_ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
    assert "actions/cache@v4" in wf
    assert "pytest" in wf
    assert "diff -u demo/output.txt" in wf


def test_issue_16_pinned_numpy_dependency() -> None:
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "numpy>=1.26,<3.0" in req
    assert "numpy>=1.26,<3.0" in pyproj


def test_issue_17_positional_lead_and_conclusion_baselines_and_tie_seed() -> None:
    dummy_pairs = [{"sents": ["a", "b", "c", "d"], "gold": {3}}]
    assert lead_rankings(dummy_pairs) == [[0, 1, 2, 3]]
    assert conclusion_first_rankings(dummy_pairs) == [[3, 2, 1, 0]]
    ties = np.zeros(12)
    assert rank_scores(ties, tie_seed=1) != rank_scores(ties, tie_seed=2)


def test_issue_18_two_stage_open_corpus_retrieval_and_contested_stance() -> None:
    corpus = {
        1: {"doc_id": 1, "title": "Coffee Activates Autophagy", "abstract": ["Polyphenols in coffee increase hepatic autophagy flux in mice."]},
        2: {"doc_id": 2, "title": "Coffee Does Not Increase Autophagy", "abstract": ["Decaffeinated coffee does not increase hepatic autophagy flux in rats."]},
        3: {"doc_id": 3, "title": "Stellar Nucleosynthesis", "abstract": ["Supernovae synthesize heavy elements in galaxies."]},
    }
    retriever = TwoStageRetriever(corpus, abstention_threshold=0.5)
    probe = retriever.probe_claim("Coffee polyphenols increase hepatic autophagy flux.")
    assert probe.epistemic_status == "CONTESTED"
    assert len(probe.supporting_rationales) >= 1
    assert len(probe.contradicting_rationales) >= 1
    assert classify_rationale_polarity("Drug increases survival", "Drug reduces survival") == "CONTRADICT"


def test_issue_19_cli_flags_and_exit_codes(tmp_path: Path) -> None:
    probes_out = tmp_path / "cli_probes.jsonl"
    rc = cli_main(
        [
            "--claim",
            "1,000 genomes rare variants create synthetic GWAS associations.",
            "--offline",
            "--json",
            "--export-probes",
            str(probes_out),
        ]
    )
    assert rc == 0
    assert len(load_probes_jsonl(str(probes_out))) == 1
    assert cli_main(["--top-k", "0", "--offline"]) == 2
    assert cli_main(["--claim", "   ", "--offline"]) == 2


def test_issue_20_offline_mini_corpus_benchmark_runs_without_network(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["--offline", "--data-root", str(REPO_ROOT / "nonexistent_cache_dir")])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "SciFact dev: rationale sentence retrieval" in captured


def test_issue_21_epistemic_probe_result_json_schema_export_and_roundtrip(tmp_path: Path) -> None:
    schema_out = tmp_path / "schema.json"
    assert cli_main(["--export-schema", str(schema_out)]) == 0
    loaded_schema = json.loads(schema_out.read_text(encoding="utf-8"))
    assert loaded_schema["title"] == "EpistemicProbeResult"

    corpus = {1: {"doc_id": 1, "title": "T", "abstract": ["Kinase inhibitor blocks tumor growth in vivo."]}}
    probe = TwoStageRetriever(corpus, abstention_threshold=0.1).probe_claim(
        "Kinase inhibitor blocks tumor growth.",
        candidate_descriptors={"Molecular Weight": 312.4},
        candidate_smiles="c1ccccc1",
    )
    jsonschema.validate(instance=probe.to_dict(), schema=EPISTEMIC_PROBE_JSON_SCHEMA)
    jsonl_path = tmp_path / "roundtrip.jsonl"
    export_probes_jsonl([probe], str(jsonl_path))
    reloaded = load_probes_jsonl(str(jsonl_path))
    assert reloaded[0] == probe


def test_issue_22_labbench_protocolqa_adapter_canary_guard_and_counterfactual_scorer(tmp_path: Path) -> None:
    pqa_file = tmp_path / "protocolqa_sample.jsonl"
    sample_row = {
        "id": "pqa-001",
        "protocol": (
            "1. Resuspend cell pellet in lysis buffer with 1% SDS. "
            "2. Heat sample at 95 C for 5 minutes before PCR mastermix addition. "
            "3. Add Taq polymerase immediately without cooling."
        ),
        "question": "Why did PCR amplification fail completely in this protocol?",
        "ideal": "Adding Taq polymerase at 95 C without cooling denatured the enzyme.",
        "distractors": ["Buffer lacked ethidium bromide.", "Agarose gel voltage was too low."],
        "canary": CANARY_NEEDLES[0],
    }
    pqa_file.write_text(json.dumps(sample_row) + "\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="canary"):
        load_labbench_items(str(pqa_file), purpose="train")

    items = load_labbench_items(str(pqa_file), purpose="eval")
    probed = probe_labbench_item(items[0], top_k=2)
    assert probed["ideal_vs_distractor_margin"] > 0.0

    cf_item = {
        "id": "pqa-001-cf",
        "question": "Why did Taq enzymatic activity drop before thermal cycling?",
        "ideal": "1% SDS detergent in lysis buffer inhibited Taq polymerase.",
        "distractors": ["Too much glycerol in loading dye."],
        "sentences": items[0]["sentences"],
    }
    pkt = CounterfactualProbeScorer(threshold=0.70).score_pair(items[0], cf_item)
    assert pkt.shifted_rationale is True and pkt.consistency == pytest.approx(1.0)
