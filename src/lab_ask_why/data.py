"""Safe archive download/extraction, UTF-8 JSONL I/O, and SciFact pair construction."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"

DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
)
DEFAULT_DATA_DIR = os.path.join(DEFAULT_ROOT, "data")

# Deterministic offline mini-corpus for --offline CLI smoke runs and tests without S3 access.
OFFLINE_MINI_CORPUS: list[dict[str, Any]] = [
    {
        "doc_id": 101,
        "title": "Rare Variants Create Synthetic Genome-Wide Associations",
        "abstract": [
            "Genome-wide association studies (GWAS) have identified at least 2,000 common variants associated with complex traits.",
            "However, common variants often explain only a small fraction of trait heritability.",
            "Variants much less common than the tag SNP may create synthetic associations by occurring more often on one common allele background.",
            "In conclusion, uncommon or rare genetic variants with larger penetrance effects can create synthetic GWAS signals.",
        ],
    },
    {
        "doc_id": 102,
        "title": "Increased Microerythrocyte Count in Homozygous \u03b1+-Thalassaemia Protects Against Severe Malarial Anaemia",
        "abstract": [
            "Severe malarial anaemia (SMA) is a major cause of childhood mortality in endemic regions.",
            "Individuals homozygous for alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count.",
            "Children homozygous for \u03b1+-thalassaemia lose less haemoglobin for a >1.1 x 10(12)/l erythrocyte drop due to lower mean cell Hb.",
            "CONCLUSIONS: Increased microerythrocyte count in homozygous alpha(+)-thalassaemia contributes to protection, not vulnerability, against severe anaemia.",
        ],
    },
    {
        "doc_id": 103,
        "title": "CRISPR-Cas9 Off-Target Cleavage Depends on PAM-Proximal Seed Complementarity",
        "abstract": [
            "Cas9 endonuclease activity requires recognition of a canonical NGG protospacer adjacent motif.",
            "Mismatches within the 8-12 bp PAM-proximal seed region strongly abolish DNA cleavage.",
            "Distal 5-prime mismatches are tolerated without eliminating double-strand breaks.",
        ],
    },
    {
        "doc_id": 104,
        "title": "Background Transcriptomic Atlas of Mammalian Hematopoiesis",
        "abstract": [
            "Single-cell RNA sequencing profiles lineage commitment across bone marrow progenitors.",
            "Erythroid differentiation upregulates globin transcription and iron transport machinery.",
        ],
    },
]

OFFLINE_MINI_CLAIMS: list[dict[str, Any]] = [
    {
        "id": 1,
        "claim": "1,000 genomes project enables mapping of rare variants with larger penetrance effects creating synthetic GWAS associations.",
        "evidence": {"101": [{"sentences": [2, 3], "label": "SUPPORT"}]},
        "cited_doc_ids": [101],
    },
    {
        "id": 2,
        "claim": "A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.",
        "evidence": {"102": [{"sentences": [1, 3], "label": "CONTRADICT"}]},
        "cited_doc_ids": [102],
    },
    {
        "id": 3,
        "claim": "CRISPR-Cas9 increases telomere length in senescent cardiomyocytes.",
        "evidence": {},
        "cited_doc_ids": [103],
    },
]


def configure_utf8_stdout() -> None:
    """Ensure Windows cp1252 consoles render Greek/biomedical Unicode cleanly."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def compute_file_sha256(path: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_appledouble(member_name: str) -> bool:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    return any(p.startswith("._") or p == "__MACOSX" for p in parts)


def validate_tar_member(member: tarfile.TarInfo, dest_dir: str) -> bool:
    """Validate a tar member against path traversal, link escapes, and special files.

    Returns `False` if the member is benign AppleDouble metadata that should be skipped;
    raises `ValueError` on unsafe paths, links, or special device entries.
    """
    raw_name = member.name
    if not raw_name:
        raise ValueError("Tar archive contains an empty member name")

    posix_parts = PurePosixPath(raw_name.replace("\\", "/")).parts
    if os.path.isabs(raw_name) or raw_name.startswith(("/", "\\")):
        raise ValueError(f"Unsafe absolute tar member path: {raw_name!r}")
    if len(raw_name) >= 2 and raw_name[1] == ":":
        raise ValueError(f"Unsafe drive-qualified tar member path: {raw_name!r}")
    if ".." in posix_parts:
        raise ValueError(f"Path traversal ('..') detected in tar member: {raw_name!r}")

    if member.issym() or member.islnk():
        raise ValueError(f"Refusing symlink/hardlink tar member: {raw_name!r}")
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        raise ValueError(f"Refusing special device/FIFO tar member: {raw_name!r}")

    abs_dest = os.path.abspath(dest_dir)
    abs_target = os.path.abspath(os.path.join(abs_dest, *posix_parts))
    if os.path.commonpath([abs_dest, abs_target]) != abs_dest:
        raise ValueError(f"Tar member escapes destination root: {raw_name!r}")

    if _is_appledouble(raw_name):
        return False
    return True


def safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> list[str]:
    """Safely extract tar archive into `dest_dir`, skipping AppleDouble files and blocking traversal."""
    safe_members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        if validate_tar_member(member, dest_dir):
            safe_members.append(member)

    os.makedirs(dest_dir, exist_ok=True)
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest_dir, members=safe_members, filter="data")
    else:  # pragma: no cover - Python < 3.12 fallback after validate_tar_member
        tar.extractall(dest_dir, members=safe_members)
    return [m.name for m in safe_members]


def download_file_atomic(
    url: str,
    dest_path: str,
    *,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Download `url` to `dest_path + '.part'`, verify optional SHA-256, and atomically rename."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    part_path = f"{dest_path}.part.{os.getpid()}"
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        digest = hasher.hexdigest()
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}"
            )
        os.replace(part_path, dest_path)
        return digest
    except Exception:
        if os.path.exists(part_path):
            try:
                os.unlink(part_path)
            except OSError:
                pass
        raise


def ensure_data(
    root: str = DEFAULT_ROOT,
    *,
    url: str = SCIFACT_URL,
    expected_sha256: str | None = SCIFACT_SHA256,
    timeout: float = 60.0,
    offline: bool = False,
) -> str:
    """Ensure SciFact `claims_dev.jsonl` and `corpus.jsonl` are safely downloaded and extracted."""
    data_dir = os.path.join(root, "data")
    claims_dev = os.path.join(data_dir, "claims_dev.jsonl")
    corpus_file = os.path.join(data_dir, "corpus.jsonl")
    if os.path.exists(claims_dev) and os.path.exists(corpus_file):
        return data_dir

    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "data.tar.gz")

    if os.path.exists(archive) and expected_sha256 is not None:
        actual = compute_file_sha256(archive)
        if actual.lower() != expected_sha256.lower():
            try:
                os.unlink(archive)
            except OSError:
                pass
            if offline:
                raise RuntimeError(
                    f"Cached archive {archive} failed SHA-256 verification and was removed; "
                    "re-run without --offline to download a clean copy."
                )

    if not os.path.exists(archive):
        if offline:
            raise FileNotFoundError(
                f"SciFact data not found in {data_dir} and --offline mode is enabled."
            )
        print("Downloading SciFact ...")
        download_file_atomic(
            url,
            archive,
            expected_sha256=expected_sha256,
            timeout=timeout,
        )

    try:
        with tarfile.open(archive, mode="r:*") as tar:
            safe_extract_tar(tar, root)
    except (tarfile.TarError, EOFError, OSError) as exc:
        if os.path.exists(archive):
            try:
                os.unlink(archive)
            except OSError:
                pass
        raise RuntimeError(
            f"Corrupt archive {archive!r} could not be extracted and was removed ({exc}). "
            "Re-run to fetch a fresh copy."
        ) from exc

    return data_dir


def read_jsonl(name_or_path: str, data_dir: str = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL records from `name_or_path` (or inside `data_dir` if relative)."""
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(data_dir, name_or_path)
    with open(path, mode="r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _lookup_doc(corpus: Mapping[Any, Mapping[str, Any]], raw_doc_id: Any) -> tuple[Any, Mapping[str, Any]]:
    if raw_doc_id in corpus:
        return raw_doc_id, corpus[raw_doc_id]
    try:
        int_id = int(raw_doc_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Document ID {raw_doc_id!r} not found in corpus") from exc
    if int_id in corpus:
        return int_id, corpus[int_id]
    str_id = str(raw_doc_id)
    if str_id in corpus:
        return str_id, corpus[str_id]
    raise ValueError(f"Document ID {raw_doc_id!r} is missing from corpus")


def build_pairs(
    claims: Sequence[Mapping[str, Any]],
    corpus: Mapping[Any, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Construct validated `(claim, gold evidence doc)` evaluation items.

    Preserves both `gold` (union of rationale sentence indices) and `rationale_sets`
    (`list[frozenset[int]]` of minimal complete rationale sets).
    """
    pairs: list[dict[str, Any]] = []
    for c in claims:
        evidence = c.get("evidence") or {}
        if not isinstance(evidence, Mapping):
            if strict:
                raise ValueError(f"Expected mapping for claim 'evidence', got {type(evidence).__name__}")
            continue

        for raw_doc_id, sets in evidence.items():
            if not sets:
                if strict:
                    raise ValueError(
                        f"Empty rationale set list for claim {c.get('id', c.get('claim'))!r}, doc_id={raw_doc_id!r}"
                    )
                continue

            try:
                doc_id, doc = _lookup_doc(corpus, raw_doc_id)
            except ValueError:
                if strict:
                    raise
                continue

            sents = list(doc.get("abstract", ()))
            labels = {s.get("label") for s in sets if isinstance(s, Mapping)}
            if len(labels) != 1 or None in labels:
                if strict:
                    raise ValueError(
                        f"Heterogeneous or missing rationale labels {labels!r} for doc_id={raw_doc_id!r}"
                    )
                continue
            label = next(iter(labels))

            rationale_sets: list[frozenset[int]] = []
            valid_sets = True
            for s in sets:
                raw_idxs = s.get("sentences")
                if not raw_idxs:
                    if strict:
                        raise ValueError(f"Rationale entry has empty 'sentences' for doc_id={raw_doc_id!r}")
                    valid_sets = False
                    break
                idxs: list[int] = []
                for idx in raw_idxs:
                    i = int(idx)
                    if i < 0 or i >= len(sents):
                        if strict:
                            raise ValueError(
                                f"Sentence index {i} out of bounds for doc_id={raw_doc_id!r} (len={len(sents)})"
                            )
                        valid_sets = False
                        break
                    idxs.append(i)
                if not valid_sets:
                    break
                rationale_sets.append(frozenset(idxs))

            if not valid_sets or not rationale_sets:
                continue

            gold_union = set().union(*rationale_sets)
            pairs.append(
                {
                    "claim_id": c.get("id"),
                    "claim": str(c["claim"]),
                    "doc_id": doc_id,
                    "sents": sents,
                    "gold": gold_union,
                    "rationale_sets": rationale_sets,
                    "label": str(label),
                    "title": str(doc.get("title", "")),
                }
            )
    return pairs


def build_noinfo_pairs(
    claims: Sequence[Mapping[str, Any]],
    corpus: Mapping[Any, Mapping[str, Any]],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Build `(claim, cited_doc_id)` pairs for `NOINFO` claims (claims with empty `evidence`)."""
    noinfo_pairs: list[dict[str, Any]] = []
    for c in claims:
        if c.get("evidence"):
            continue
        cited_ids = c.get("cited_doc_ids") or []
        for raw_doc_id in cited_ids:
            try:
                doc_id, doc = _lookup_doc(corpus, raw_doc_id)
            except ValueError:
                if strict:
                    raise
                continue
            sents = list(doc.get("abstract", ()))
            if not sents:
                continue
            noinfo_pairs.append(
                {
                    "claim_id": c.get("id"),
                    "claim": str(c["claim"]),
                    "doc_id": doc_id,
                    "sents": sents,
                    "gold": set(),
                    "rationale_sets": [],
                    "label": "NOINFO",
                    "title": str(doc.get("title", "")),
                }
            )
    return noinfo_pairs
