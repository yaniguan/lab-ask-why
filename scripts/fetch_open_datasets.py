"""On-demand downloader and schema-normalizer for 2019-2026 open scientific datasets.

Fetches sample or split records from Hugging Face Datasets Server / GitHub into the
gitignored `data/open_datasets/<dataset_id>/` directory (zero repo redistribution),
enforces `eval_only` canary guards (`LAB-Bench`, `LitQA2`, `LABBench2`, `ChemBench`),
and writes normalized JSONL ready for `lab-ask-why` BM25 + Socratic probing.

Usage:
  python scripts/fetch_open_datasets.py --list
  python scripts/fetch_open_datasets.py --verify-urls
  python scripts/fetch_open_datasets.py --dataset pubmedqa --max-rows 20 --probe
  python scripts/fetch_open_datasets.py --dataset litqa2 --max-rows 10 --purpose eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_ask_why.data import DEFAULT_ROOT, configure_utf8_stdout
from lab_ask_why.datasets import (
    OPEN_DATASET_SPECS,
    OpenDatasetSpec,
    adapt_records,
    enforce_dataset_guard,
    format_dataset_registry_table,
    get_dataset_spec,
    list_open_datasets,
    probe_normalized_record,
)

DEFAULT_OPEN_DATASETS_DIR = os.path.join(DEFAULT_ROOT, "open_datasets")
_USER_AGENT = "lab-ask-why/1.0 (+https://github.com/yaniguan/lab-ask-why)"


def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify_url_live(url: str, *, timeout: float = 15.0) -> int:
    """Return HTTP status code for `url` via HEAD with Range-GET fallback."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 405}:
            raise
    req_get = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-64"},
    )
    with urllib.request.urlopen(req_get, timeout=timeout) as resp_get:
        return int(resp_get.status)


def _write_jsonl_atomic(rows: Sequence[Mapping[str, Any]], dest_path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    part_path = f"{dest_path}.part.{os.getpid()}"
    try:
        with open(part_path, mode="w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(part_path, dest_path)
    except Exception:
        if os.path.exists(part_path):
            try:
                os.unlink(part_path)
            except OSError:
                pass
        raise
    return len(rows)


def fetch_raw_sample_rows(
    spec: OpenDatasetSpec,
    *,
    max_rows: int = 25,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch up to `max_rows` raw records for `spec` via HF datasets-server or GitHub raw JSON."""
    limit = max(1, min(int(max_rows), 100))

    if spec.dataset_id == "claimcheck":
        raw_url = "https://raw.githubusercontent.com/JHU-CLSP/CLAIMCHECK/main/data/texts/source/pilot.json"
        payload = _http_get_json(raw_url, timeout=timeout)
        rows: list[dict[str, Any]] = []
        if isinstance(payload, Mapping):
            for k, v in payload.items():
                if isinstance(v, Mapping):
                    item = dict(v)
                    item.setdefault("id", str(k))
                    rows.append(item)
                if len(rows) >= limit:
                    break
        return rows

    if spec.dataset_id == "sciver":
        raw_url = "https://huggingface.co/datasets/chengyewang/SciVer/resolve/main/valset.json"
        payload = _http_get_json(raw_url, timeout=timeout)
        if isinstance(payload, list):
            return [dict(x) for x in payload[:limit] if isinstance(x, Mapping)]
        return []

    if spec.dataset_id in {"scifact", "scifact_open"}:
        raise RuntimeError(
            f"{spec.dataset_id!r} distributes multi-file archive bundles; use `ensure_data()` "
            f"for SciFact or download from {spec.url}."
        )

    ds_q = urllib.parse.quote(spec.hf_or_git_id, safe="")
    cfg_q = urllib.parse.quote(spec.default_config, safe="")
    split_q = urllib.parse.quote(spec.default_split, safe="")
    rows_url = (
        f"https://datasets-server.huggingface.co/rows?"
        f"dataset={ds_q}&config={cfg_q}&split={split_q}&offset=0&length={limit}"
    )
    payload = _http_get_json(rows_url, timeout=timeout)
    out: list[dict[str, Any]] = []
    for entry in payload.get("rows") or ():
        if isinstance(entry, Mapping) and isinstance(entry.get("row"), Mapping):
            out.append(dict(entry["row"]))
    return out


def fetch_and_normalize_dataset(
    dataset_id: str,
    *,
    output_root: str = DEFAULT_OPEN_DATASETS_DIR,
    max_rows: int = 25,
    purpose: str = "eval",
    require_commercial_ok: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Download sample records on demand into gitignored `output_root/<dataset_id>/` and normalize."""
    spec = enforce_dataset_guard(
        dataset_id,
        purpose=purpose,
        require_commercial_ok=require_commercial_ok,
    )
    raw_rows = fetch_raw_sample_rows(spec, max_rows=max_rows, timeout=timeout)
    normalized = adapt_records(
        spec.dataset_id,
        raw_rows,
        purpose=purpose,
        require_commercial_ok=require_commercial_ok,
    )

    ds_dir = os.path.join(output_root, spec.dataset_id)
    raw_path = os.path.join(ds_dir, "raw_sample.jsonl")
    norm_path = os.path.join(ds_dir, "normalized_claims.jsonl")
    _write_jsonl_atomic(raw_rows, raw_path)
    _write_jsonl_atomic([r.to_dict() for r in normalized], norm_path)

    return {
        "dataset_id": spec.dataset_id,
        "spdx_license": spec.spdx_license,
        "eval_only": spec.eval_only,
        "commercial_use_ok": spec.commercial_use_ok,
        "rows_fetched": len(raw_rows),
        "raw_path": raw_path,
        "normalized_path": norm_path,
        "normalized_records": normalized,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="List, live-verify, or fetch on-demand open scientific datasets into gitignored data/."
    )
    p.add_argument("--list", action="store_true", help="Print open dataset registry table and exit.")
    p.add_argument("--verify-urls", action="store_true", help="Verify all dataset and paper URLs return HTTP 2xx.")
    p.add_argument("--dataset", type=str, default=None, help="Dataset ID or alias to fetch on demand.")
    p.add_argument("--max-rows", type=int, default=20, help="Maximum sample rows to download (1..100).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OPEN_DATASETS_DIR, help="Gitignored destination root.")
    p.add_argument("--purpose", type=str, default="eval", help="Intended use ('eval' or 'train'); enforces eval_only.")
    p.add_argument("--commercial-only", action="store_true", help="Reject non-commercial dataset licenses.")
    p.add_argument("--probe", action="store_true", help="Run BM25 + Socratic probe on first fetched record.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)

    if args.list or (not args.verify_urls and not args.dataset):
        specs = list_open_datasets(commercial_only=args.commercial_only)
        if args.json:
            print(json.dumps([s.to_dict() for s in specs], indent=2))
        else:
            print(format_dataset_registry_table(specs))
        return 0

    if args.verify_urls:
        results: list[dict[str, Any]] = []
        failures = 0
        for spec in OPEN_DATASET_SPECS:
            for kind, u in (("url", spec.url), ("paper_url", spec.paper_url)):
                if not u:
                    continue
                try:
                    code = verify_url_live(u)
                    ok = 200 <= code < 400
                except Exception as exc:
                    code = 0
                    ok = False
                    results.append({"dataset_id": spec.dataset_id, "kind": kind, "url": u, "ok": False, "error": str(exc)})
                    failures += 1
                    continue
                if not ok:
                    failures += 1
                results.append({"dataset_id": spec.dataset_id, "kind": kind, "url": u, "status": code, "ok": ok})
        if args.json:
            print(json.dumps({"checked": len(results), "failures": failures, "results": results}, indent=2))
        else:
            for item in results:
                tag = "OK" if item["ok"] else "FAIL"
                print(f"[{tag}] {item['dataset_id']:<22} ({item['kind']}): {item['url']}")
        return 0 if failures == 0 else 1

    try:
        summary = fetch_and_normalize_dataset(
            args.dataset,
            output_root=args.output_dir,
            max_rows=args.max_rows,
            purpose=args.purpose,
            require_commercial_ok=args.commercial_only,
        )
        records = summary.pop("normalized_records")
        if args.probe and records:
            summary["sample_probe"] = probe_normalized_record(records[0], purpose=args.purpose).to_dict()
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(
                f"Fetched {summary['rows_fetched']} records for {summary['dataset_id']} "
                f"({summary['spdx_license']}, eval_only={summary['eval_only']}) -> {summary['normalized_path']}"
            )
            if "sample_probe" in summary:
                sp = summary["sample_probe"]
                print(f"Sample claim: {sp['claim']}")
                print(f"Epistemic status: {sp['epistemic_status']} (confidence={sp['retrieval_confidence']:.3f})")
        return 0
    except (KeyError, PermissionError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
