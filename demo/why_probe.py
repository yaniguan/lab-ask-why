"""Ask 'why do you think so?' instead of judging who is right.

Evaluates biomedical Okapi BM25 rationale retrieval, positional and random
baselines, NOINFO abstention calibration, open-corpus two-stage retrieval, and
rationale-grounded Socratic 'why' questions on SciFact (Wadden et al., 2020).

Data: SciFact is released under CC BY-NC 2.0. It is downloaded safely on demand
to gitignored data/ and should not be committed.

Run:
  python demo/why_probe.py
  python demo/why_probe.py --claim "GWAS identifies rare synthetic variants." --offline
  python demo/why_probe.py --format json --offline
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_ask_why.bm25 import BM25, STOP, normalize_biomedical_text, tokenize
from lab_ask_why.cli import main as cli_main
from lab_ask_why.data import (
    DEFAULT_DATA_DIR as DATA,
    DEFAULT_ROOT as ROOT,
    SCIFACT_SHA256,
    SCIFACT_URL as URL,
    build_noinfo_pairs,
    build_pairs,
    ensure_data,
    read_jsonl,
    safe_extract_tar,
)
from lab_ask_why.metrics import (
    compute_auroc,
    conclusion_first_rankings,
    hit_at,
    lead_rankings,
    rationale_set_recall_at,
    recall_at,
    sentence_recall_at,
)
from lab_ask_why.retrieval import TwoStageRetriever
from lab_ask_why.schema import EpistemicProbeResult
from lab_ask_why.socratic import why_questions


def main(argv: Sequence[str] | None = None) -> int:
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
