"""Label-blind one-day refinement of capped two-day KR Google News slices."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import collect_v70_kr_gnews_extension as parent


ROOT = Path(__file__).resolve().parent
POOL = ROOT / "data" / "untouched_pool"
RESEARCH = ROOT / "research"
CONTRACT = "KR_GNEWS_DAILY_SATURATION_REFINEMENT_V1"
SOURCE = "GOOGLE_NEWS_RSS_KR_DAILY_REFINEMENT"
PARENT_STATUS = RESEARCH / "KR_GNEWS_SATURATION_REFINEMENT_STATUS.json"
STATUS_PATH = RESEARCH / "KR_GNEWS_DAILY_REFINEMENT_STATUS.json"


def sha256(path: Path) -> str:
    return parent.sha256(path)


def load_roots() -> tuple[list[dict[str, str]], dict[str, Any], str]:
    if not PARENT_STATUS.is_file():
        raise RuntimeError("SATURATION_REFINEMENT_STATUS_MISSING")
    status_sha = sha256(PARENT_STATUS)
    status = json.loads(PARENT_STATUS.read_text(encoding="utf-8"))
    audit_path = ROOT / status["audit_path"]
    if sha256(audit_path) != status["audit_sha256"]:
        raise RuntimeError("SATURATION_REFINEMENT_AUDIT_SHA_MISMATCH")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("contract") != "KR_GNEWS_RECURSIVE_SATURATION_REFINEMENT_V1":
        raise RuntimeError("SATURATION_REFINEMENT_PARENT_CONTRACT_MISMATCH")
    roots: dict[str, dict[str, str]] = {}
    for root in audit.get("root_statuses", []):
        company = str(root.get("company", ""))
        for request in root.get("requests", []):
            duration = float(request.get("duration_days", 0))
            if not bool(request.get("saturated")) or not (1.0 < duration <= 2.0):
                continue
            ticker = str(request.get("ticker", root.get("ticker", "")))
            start = str(request["slice_start"])
            end = str(request["slice_end_exclusive"])
            key = f"{ticker}|{start}|{end}"
            roots[key] = {
                "root_key": key, "ticker": ticker, "company": company,
                "slice_start": start, "slice_end_exclusive": end,
            }
    return [roots[key] for key in sorted(roots)], audit, status_sha


def fetch_root(
    root: dict[str, str], timeout: float, retries: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = pd.Timestamp(root["slice_start"])
    end = pd.Timestamp(root["slice_end_exclusive"])
    midpoint = start + pd.Timedelta(days=1)
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for child_start, child_end in ((start, min(midpoint, end)), (min(midpoint, end), end)):
        if not child_start < child_end:
            continue
        rows, quarantined, request = parent.query_once(
            root["ticker"], root["company"], child_start, child_end, timeout, retries
        )
        for row in rows:
            row["source"] = SOURCE
        request = dict(request)
        request["duration_days"] = float((child_end - child_start) / pd.Timedelta(days=1))
        request["saturated_at_one_day"] = bool(
            request.get("status") == "OK"
            and int(request.get("items", 0)) >= parent.SATURATION_COUNT
        )
        records.extend(rows)
        quarantine.extend(quarantined)
        requests.append(request)
    return records, quarantine, {
        **root, "requests": requests, "request_count": len(requests),
        "request_failures": sum(item.get("status") != "OK" for item in requests),
        "one_day_saturated_count": sum(bool(item.get("saturated_at_one_day")) for item in requests),
        "raw_rows": len(records),
    }


def collect(
    roots: list[dict[str, str]], workers: int, timeout: float, retries: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(roots)))) as pool:
        futures = {pool.submit(fetch_root, root, timeout, retries): root for root in roots}
        for done, future in enumerate(as_completed(futures), 1):
            root = futures[future]
            try:
                rows, quarantined, status = future.result()
            except Exception as exc:
                rows, quarantined = [], []
                status = {
                    **root, "requests": [], "request_count": 0, "request_failures": 1,
                    "one_day_saturated_count": 0, "raw_rows": 0,
                    "unexpected_error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            records.extend(rows)
            quarantine.extend(quarantined)
            statuses.append(status)
            if done % 10 == 0 or done == len(futures):
                print(f"[KR GNEWS DAILY] roots={done}/{len(futures)} raw_rows={len(records)}", flush=True)
    frame, quarantine, dedup = parent.resolve_articles(records, quarantine)
    audit = {
        "configured_capped_two_day_roots": len(roots),
        "completed_roots": len(statuses),
        "http_requests": sum(int(item.get("request_count", 0)) for item in statuses),
        "request_failures": sum(int(item.get("request_failures", 0)) for item in statuses),
        "one_day_saturated_intervals": sum(int(item.get("one_day_saturated_count", 0)) for item in statuses),
        "deduplication": dedup,
    }
    return frame, quarantine, statuses, audit


def write_outputs(
    frame: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    audit: dict[str, Any],
    parent_status_sha: str,
) -> dict[str, Any]:
    POOL.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    snapshot = POOL / f"gnews_kr_2025-08-20_2026-08-27_daily_{content_sha[:12]}.csv.gz"
    parent.atomic_gzip(snapshot, csv_bytes)
    quarantine_payload = {
        "contract": CONTRACT, "selection_uses_labels": False,
        "rows": sorted(quarantine, key=lambda row: (
            str(row.get("reason", "")), str(row.get("article_id", "")),
            str(row.get("url", "")), str(row.get("ticker", "")),
        )),
    }
    quarantine_bytes = parent.canonical_json_bytes(quarantine_payload)
    quarantine_sha = hashlib.sha256(quarantine_bytes).hexdigest()
    quarantine_path = POOL / f"gnews_kr_daily_quarantine_{quarantine_sha[:12]}.json.gz"
    parent.atomic_gzip(quarantine_path, quarantine_bytes)
    audit_payload = {
        "contract": CONTRACT, "source": SOURCE, "selection_uses_labels": False,
        "forbidden_inputs_read": [],
        "authorized_inputs": [
            {"path": str(parent.UNIVERSE_PATH.relative_to(ROOT)), "sha256": parent.EXPECTED_UNIVERSE_SHA256},
            {"path": str(PARENT_STATUS.relative_to(ROOT)), "sha256": parent_status_sha},
        ],
        "refinement_rule": "split capped two-day intervals into one-day requests; no further tuning",
        "collection": audit, "root_statuses": statuses,
        "snapshot": {
            "path": str(snapshot.relative_to(ROOT)), "rows": len(frame),
            "content_sha256_uncompressed_csv": content_sha,
            "file_sha256": sha256(snapshot), "bytes": snapshot.stat().st_size,
        },
        "quarantine": {
            "path": str(quarantine_path.relative_to(ROOT)), "rows": len(quarantine),
            "content_sha256": quarantine_sha, "file_sha256": sha256(quarantine_path),
        },
        "complete_interval_claimed": False,
        "partial_progress_preserved": bool(
            audit["request_failures"] or audit["one_day_saturated_intervals"]
        ),
    }
    audit_bytes = parent.canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_kr_daily_audit_{audit_sha[:12]}.json"
    parent.atomic_bytes(audit_path, audit_bytes)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_PRESERVED" if audit_payload["partial_progress_preserved"] else "BEST_EFFORT_COMPLETE",
        "contract": CONTRACT, "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"], "audit_path": str(audit_path.relative_to(ROOT)),
        "audit_sha256": sha256(audit_path), "quarantine": audit_payload["quarantine"],
        "collection": audit, "downstream_queue_not_modified": True,
    }
    parent.atomic_bytes(STATUS_PATH, json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("workers/timeout must be positive and retries non-negative")
    universe, input_audit = parent.file_input_audit()
    roots, parent_audit, parent_status_sha = load_roots()
    tickers = {root["ticker"] for root in roots}
    if not tickers.issubset(set(universe.ticker.astype(str))):
        raise RuntimeError("DAILY_ROOT_OUTSIDE_PINNED_UNIVERSE")
    summary = {
        "status": "AUDIT_OK", "contract": CONTRACT, "selection_uses_labels": False,
        "authorized_universe": input_audit, "parent_status_sha256": parent_status_sha,
        "capped_two_day_roots": len(roots), "root_tickers": len(tickers),
        "request_floor": len(roots) * 2, "forbidden_inputs_read": [],
        "full_collection_executed": False,
    }
    if args.audit_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    frame, quarantine, statuses, audit = collect(roots, args.workers, args.timeout, args.retries)
    print(json.dumps(
        write_outputs(frame, quarantine, statuses, audit, parent_status_sha),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
