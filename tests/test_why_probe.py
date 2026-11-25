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
    CONTROL_WELL_ROLES,
    EPISTEMIC_PROBE_JSON_SCHEMA,
    STOP,
    ControlWellSpec,
    CounterfactualProbeScorer,
    EpistemicProbeResult,
    TwoStageRetriever,
    build_noinfo_pairs,
    build_pairs,
    check_ask_why_permille,
    check_epistemic_graph_permille,
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
    synthesize_control_wells,
    to_permille_ceil,
    to_permille_floor,
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


def test_issue_24_tokenizer_alleles_delta_alphanumeric_and_scoped_polarity() -> None:
    assert tokenize("alpha(+)-thalassaemia") == ["alpha+", "thalassaemia"]
    assert tokenize("alpha(-)-thalassaemia") == ["alpha-", "thalassaemia"]
    assert tokenize("CD4-/CD8+") == ["cd4-", "cd8+"]
    assert tokenize("Trp53-/- Trp53+/+ Trp53+/-") == ["trp53-", "trp53+", "trp53+/-"]
    assert tokenize("\u0394F508 rad52\u2206") == ["delta", "f508", "rad52", "delta"]
    assert tokenize("16S rRNA 5-HT2A 293T 3D 1.1") == [
        "16s",
        "rrna",
        "5ht2a",
        "293t",
        "3d",
        "1.1",
    ]
    assert tokenize("genome-wide double-strand") == ["genome", "wide", "double", "strand"]

    assert (
        classify_rationale_polarity(
            "Aspirin prevents ester hydrolysis.",
            "Aspirin cannot prevent ester hydrolysis.",
        )
        == "CONTRADICT"
    )
    assert (
        classify_rationale_polarity(
            "Aspirin prevents ester hydrolysis.",
            "Aspirin doesn't prevent ester hydrolysis.",
        )
        == "CONTRADICT"
    )
    assert (
        classify_rationale_polarity(
            "Distal 5-prime mismatches are tolerated.",
            "Distal 5-prime mismatches are tolerated without eliminating double-strand breaks.",
        )
        == "SUPPORT"
    )
    assert (
        classify_rationale_polarity(
            "Drug X increases survival.",
            "Drug X increases survival with no dose-limiting toxicity.",
        )
        == "SUPPORT"
    )
    assert (
        classify_rationale_polarity(
            "Drug X increases survival.",
            "Although early trials claimed Drug X increases survival, randomized cohorts show Drug X decreases survival.",
        )
        == "CONTRADICT"
    )
    assert (
        classify_rationale_polarity(
            "Quantum Hall conductance plateau in graphene.",
            "PCR Taq polymerase denatures at 98 C.",
        )
        == "NEUTRAL"
    )

    # NEUTRAL NLI/reranker rationales must never populate supporting_rationales
    neutral_corpus = {
        1: {"doc_id": 1, "title": "Setup", "abstract": ["Samples were incubated at 37 C."]}
    }
    ret_neutral = TwoStageRetriever(
        neutral_corpus,
        abstention_threshold=0.05,
        rerank_fn=lambda _c, sents: [(0.9, "NEUTRAL") for _ in sents],
    )
    p_neutral = ret_neutral.probe_claim("Samples incubated at 37 C dissolve.")
    assert p_neutral.supporting_rationales == []
    assert p_neutral.contradicting_rationales == []
    assert p_neutral.epistemic_status == "INSUFFICIENT_EVIDENCE"


def test_issue_25_query_budget_scaling_cv_calibration_and_background_idf() -> None:
    # 1. Small-corpus exact match must resolve as SUPPORTED rather than falsely abstaining
    small_corpus = {
        1: {"doc_id": 1, "title": "Autophagy", "abstract": ["Metformin activates hepatic AMPK signaling."]},
        2: {"doc_id": 2, "title": "Control", "abstract": ["Vehicle saline did not alter weight."]},
    }
    ret_small = TwoStageRetriever(small_corpus)
    probe_exact = ret_small.probe_claim("Metformin activates AMPK.")
    assert probe_exact.epistemic_status == "SUPPORTED"
    assert probe_exact.retrieval_confidence >= 0.80

    # 2. High-verbosity NOINFO claim with 0% disease/predicate overlap must abstain
    large_corpus = {
        i: {"doc_id": i, "title": f"Doc {i}", "abstract": [f"Background gene_{i} cellular metabolism study."]}
        for i in range(200)
    }
    large_corpus[999] = {
        "doc_id": 999,
        "title": "Sleep Perfusion",
        "abstract": [
            "Chronic intermittent hypoxia alters renal cortical perfusion in aged Sprague Dawley rats during sleep."
        ],
    }
    ret_large = TwoStageRetriever(large_corpus)
    probe_noinfo = ret_large.probe_claim(
        "Chronic intermittent hypoxia cures pancreatic ductal adenocarcinoma in aged Sprague Dawley rats during sleep."
    )
    assert probe_noinfo.epistemic_status == "INSUFFICIENT_EVIDENCE"
    assert probe_noinfo.retrieval_confidence < 0.25

    # 3. Separate fit_abstention_threshold vs non-mutating evaluate_claims
    cal_claims = [
        {"id": 1, "claim": "Metformin activates AMPK.", "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
        {"id": 2, "claim": "Unrelated astrophysics supernova emission.", "evidence": {}},
    ]
    orig_thresh = ret_small.abstention_threshold
    _ = ret_small.evaluate_claims(cal_claims)
    assert ret_small.abstention_threshold == orig_thresh
    fitted = ret_small.fit_abstention_threshold(cal_claims)
    assert ret_small.abstention_threshold == pytest.approx(fitted)

    # 4. Background IDF scope excludes evaluation doc vocabulary while keeping docs searchable
    leak_corpus = {
        1: {"doc_id": 1, "title": "BG1", "abstract": ["Common kinase signaling regulates growth."]},
        2: {"doc_id": 2, "title": "BG2", "abstract": ["Common kinase pathway alters metabolism."]},
        77: {"doc_id": 77, "title": "EvalDoc", "abstract": ["Zyxwvunikin activates autophagic flux."]},
    }
    eval_claims = [
        {
            "id": 10,
            "claim": "Zyxwvunikin activates autophagic flux.",
            "evidence": {"77": [{"sentences": [0], "label": "SUPPORT"}]},
            "cited_doc_ids": [77],
        }
    ]
    ret_bg = TwoStageRetriever(leak_corpus, exclude_doc_ids=[77])
    assert "zyxwvunikin" not in ret_bg.sent_bm25.idf
    assert "zyxwvunikin" not in ret_bg.doc_bm25.idf
    found = ret_bg.retrieve("Zyxwvunikin activates autophagic flux.", doc_top_m=2, sent_top_k=1)
    assert found["top_doc_ids"][0] == 77
    bench_bg = evaluate_scifact_benchmark(leak_corpus, eval_claims, idf_scope="background")
    assert bench_bg["open_corpus"]["doc_recall@5"] == pytest.approx(1.0)


def test_issue_27_permille_bounds_and_multi_claim_consistency() -> None:
    assert to_permille_floor(0.349999) == 349
    assert to_permille_floor(0.3495) == 349
    assert to_permille_floor(0.35) == 350
    assert to_permille_floor(0.29) == 290

    assert to_permille_ceil(0.10) == 100
    assert to_permille_ceil(0.1000001) == 101
    assert to_permille_ceil(0.1005) == 101
    assert to_permille_ceil(0.15) == 150
    assert to_permille_ceil(0.1500001) == 151
    assert to_permille_ceil(0.1505) == 151

    assert (
        check_ask_why_permille(0.3495, noinfo_thresh=0.35, support_hits=1, contradict_hits=0)[
            "rule"
        ]
        == "noinfo_abstain"
    )
    assert check_ask_why_permille(0.35, noinfo_thresh=0.35, support_hits=1, contradict_hits=0) is None

    # Multi-claim entailment conflict: p_parent supported, p_prereq contradicted, neither locally contested
    p_parent = {
        "topScorePermille": 820,
        "noinfoThreshPermille": 350,
        "supportHits": 2,
        "contradictHits": 0,
        "hasDiscriminatingExperiment": False,
    }
    p_prereq = {
        "topScorePermille": 790,
        "noinfoThreshPermille": 350,
        "supportHits": 0,
        "contradictHits": 2,
        "hasDiscriminatingExperiment": False,
    }
    viol = check_epistemic_graph_permille([p_parent, p_prereq], implications=[(0, 1)])
    assert viol is not None and viol["rule"] == "entailment_conflict_requires_experiment"

    p_parent_exp = {**p_parent, "hasDiscriminatingExperiment": True}
    assert check_epistemic_graph_permille([p_parent_exp, p_prereq], implications=[(0, 1)]) is None

    # Mutual exclusion conflict
    mx_viol = check_epistemic_graph_permille([p_parent, p_parent], exclusions=[(0, 1)])
    assert mx_viol is not None and mx_viol["rule"] == "mutual_exclusion_conflict_requires_experiment"
    assert check_epistemic_graph_permille([p_parent_exp, p_parent_exp], exclusions=[(0, 1)]) is None

    lean_src = (REPO_ROOT / "lean" / "LabAskWhyLean.lean").read_text(encoding="utf-8")
    assert "theorem entailment_conflict_requires_experiment" in lean_src
    assert "theorem ceil_permille_preserves_violation" in lean_src
    assert "sorry" not in lean_src
    wf = (REPO_ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
    assert "lake build" in wf


def test_issue_29_hybrid_paraphrase_hooks_and_control_well_synthesis() -> None:
    # 1. Zero-BM25 mechanistic paraphrase recalled via deterministic subword/concept hybrid index
    synonym_corpus = {
        1: {
            "doc_id": 1,
            "title": "Salicylate Stability",
            "abstract": [
                "Aspirin suffers rapid aqueous hydrolysis into salicylate during 37 C incubation."
            ],
        }
    }
    retriever = TwoStageRetriever(synonym_corpus, abstention_threshold=0.5)
    probe = retriever.probe_claim(
        "Acetylsalicylic acid undergoes thermal ester deacetylation in unbuffered water."
    )
    assert probe.epistemic_status == "SUPPORTED"
    assert probe.retrieval_confidence >= 0.5

    # 2. Pluggable embed_fn (RRF) and rerank_fn (NLI stance override)
    custom_corpus = {
        1: {"doc_id": 1, "title": "Distractor", "abstract": ["Alpha token superficial mention."]},
        2: {"doc_id": 2, "title": "DenseHit", "abstract": ["Orthogonal latent mechanism blocks target."]},
    }

    def _mock_embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 2), dtype=np.float64)
        for idx, t in enumerate(texts):
            if "latent" in t.lower() or "query_latent" in t.lower():
                out[idx] = [1.0, 0.0]
            else:
                out[idx] = [0.0, 1.0]
        return out

    ret_hooks = TwoStageRetriever(
        custom_corpus,
        abstention_threshold=0.2,
        embed_fn=_mock_embed,
        rerank_fn=lambda _q, sents: [
            (0.92, "CONTRADICT") if "latent" in s.lower() else (0.05, "NEUTRAL")
            for s in sents
        ],
    )
    p_hooks = ret_hooks.probe_claim("query_latent signal increases activity.")
    assert p_hooks.epistemic_status == "CONTRADICTED"
    assert len(p_hooks.contradicting_rationales) == 1
    assert p_hooks.supporting_rationales == []

    # 3. Contested claim synthesizes candidate_descriptors and structured ControlWellSpec panel
    contested_corpus = {
        1: {"doc_id": 1, "title": "Support", "abstract": ["Incubation at 37 C increases aspirin solubility."]},
        2: {"doc_id": 2, "title": "Counter", "abstract": ["Incubation at 37 C without blank controls does not prevent aspirin hydrolysis."]},
    }
    contested = TwoStageRetriever(contested_corpus, abstention_threshold=0.1).probe_claim(
        "Incubation at 37 C increases aspirin solubility."
    )
    assert contested.epistemic_status == "CONTESTED"
    assert contested.discriminating_observable.candidate_descriptors["temp_c"] == 37.0
    wells = contested.discriminating_observable.control_wells
    assert len(wells) >= 3
    roles = {w.role for w in wells}
    assert "VEHICLE_BLANK" in roles
    assert "PERTURBATION_ARM" in roles
    assert roles <= set(CONTROL_WELL_ROLES)
    jsonschema.validate(instance=contested.to_dict(), schema=EPISTEMIC_PROBE_JSON_SCHEMA)

    # Backward compatibility: payload omitting control_wells still validates and round-trips
    legacy_dict = contested.to_dict()
    legacy_dict["discriminating_observable"].pop("control_wells", None)
    jsonschema.validate(instance=legacy_dict, schema=EPISTEMIC_PROBE_JSON_SCHEMA)
    assert EpistemicProbeResult.from_dict(legacy_dict).discriminating_observable.control_wells == []

    # Turbidity & denaturation confounder branches in synthesize_control_wells
    turb_wells = synthesize_control_wells(
        "Candidate compound dissolves cleanly.",
        ["turbidity", "colloidal"],
    )
    assert any(w.parameters.get("confounder_target") == "turbidity" for w in turb_wells)
    denat_wells = synthesize_control_wells(
        "Taq polymerase amplifies target at 95 C.",
        ["denaturation"],
    )
    assert any(w.role == "INACTIVATED_CONTROL" for w in denat_wells)
    with pytest.raises(ValueError, match="role"):
        ControlWellSpec("BAD_ROLE", {}, "ok", "fail")

