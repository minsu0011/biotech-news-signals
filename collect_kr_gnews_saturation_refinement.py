"""Label-blind recursive refinement of V70's saturated KR Google News slices.

The V70 parent snapshot remains immutable.  This collector reads only its own
acquisition checkpoint plus the pinned V29 universe, recursively bisects the
half-month requests that still returned the 100-item RSS cap, and writes a new
content-addressed snapshot.  It never opens roles, prices, returns, or labels.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import collect_v70_kr_gnews_extension as parent


ROOT = Path(__file__).resolve().parent
POOL = ROOT / "data" / "untouched_pool"
RESEARCH = ROOT / "research"
STATE = ROOT / "state"
CONTRACT = "KR_GNEWS_RECURSIVE_SATURATION_REFINEMENT_V1"
SOURCE = "GOOGLE_NEWS_RSS_KR_SATURATION_REFINEMENT"
STATUS_PATH = RESEARCH / "KR_GNEWS_SATURATION_REFINEMENT_STATUS.json"
CHECKPOINT_PATH = STATE / "KR_GNEWS_SATURATION_REFINEMENT_CHECKPOINT.json.gz"
MIN_SLICE_DAYS = 2


def parent_checkpoint() -> tuple[dict[str, Any], str]:
    path = parent.CHECKPOINT_PATH
    if not path.is_file():
        raise RuntimeError("V70_PARENT_CHECKPOINT_MISSING")
    file_sha = parent.sha256(path)
    value = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    valid = (
        value.get("contract") == parent.CONTRACT
        and value.get("universe_sha256") == parent.EXPECTED_UNIVERSE_SHA256
        and len(value.get("completed_slices", [])) == 13
        and value.get("selection_uses_labels") is False
    )
    if not valid:
        raise RuntimeError("V70_PARENT_CHECKPOINT_CONTRACT_MISMATCH")
    return value, file_sha


def saturated_roots(value: dict[str, Any]) -> list[dict[str, str]]:
    roots: dict[str, dict[str, str]] = {}
    for task in value.get("statuses", []):
        ticker = str(task.get("ticker", ""))
        company = str(task.get("company", ""))
        for request in task.get("subrequests", []):
            if request.get("status") != "OK" or int(request.get("items", 0)) < parent.SATURATION_COUNT:
                continue
            start = str(request["slice_start"])
            end = str(request["slice_end_exclusive"])
            key = f"{ticker}|{start}|{end}"
            roots[key] = {
                "root_key": key, "ticker": ticker, "company": company,
                "slice_start": start, "slice_end_exclusive": end,
            }
    return [roots[key] for key in sorted(roots)]


def refine_interval(
    ticker: str,
    company: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    retries: int,
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records, quarantine, request = parent.query_once(
        ticker, company, start, end, timeout, retries
    )
    request = dict(request)
    request["depth"] = depth
    request["duration_days"] = float((end - start) / pd.Timedelta(days=1))
    saturated = bool(
        request.get("status") == "OK"
        and int(request.get("items", 0)) >= parent.SATURATION_COUNT
    )
    request["saturated"] = saturated
    duration_days = int((end - start) / pd.Timedelta(days=1))
    if not saturated or duration_days <= MIN_SLICE_DAYS:
        for row in records:
            row["source"] = SOURCE
        return records, quarantine, [request]

    midpoint = start + pd.Timedelta(days=max(1, duration_days // 2))
    if midpoint >= end:
        for row in records:
            row["source"] = SOURCE
        return records, quarantine, [request]

    # The parent V70 snapshot already preserves the capped parent result.  The
    # refinement snapshot contains child results only, avoiding redundant rows.
    child_records: list[dict[str, Any]] = []
    child_quarantine = list(quarantine)
    child_statuses = [request]
    for child_start, child_end in ((start, midpoint), (midpoint, end)):
        rows, quarantined, statuses = refine_interval(
            ticker, company, child_start, child_end, timeout, retries, depth + 1
        )
        child_records.extend(rows)
        child_quarantine.extend(quarantined)
        child_statuses.extend(statuses)
    return child_records, child_quarantine, child_statuses


def refine_root(
    root: dict[str, str], timeout: float, retries: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = pd.Timestamp(root["slice_start"])
    end = pd.Timestamp(root["slice_end_exclusive"])
    duration_days = int((end - start) / pd.Timedelta(days=1))
    midpoint = start + pd.Timedelta(days=max(1, duration_days // 2))
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for child_start, child_end in ((start, midpoint), (midpoint, end)):
        rows, quarantined, requests = refine_interval(
            root["ticker"], root["company"], child_start, child_end,
            timeout, retries, 1,
        )
        records.extend(rows)
        quarantine.extend(quarantined)
        statuses.extend(requests)
    return records, quarantine, {
        **root,
        "requests": statuses,
        "request_count": len(statuses),
        "request_failures": sum(item.get("status") != "OK" for item in statuses),
        "terminal_saturated_count": sum(
            bool(item.get("saturated")) and float(item.get("duration_days", 0)) <= MIN_SLICE_DAYS
            for item in statuses
        ),
        "raw_rows": len(records),
    }


def checkpoint_payload(
    parent_sha: str,
    roots_sha: str,
    completed: list[str],
    records: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "parent_checkpoint_sha256": parent_sha,
        "roots_sha256": roots_sha,
        "completed_roots": completed,
        "records": records,
        "quarantine": quarantine,
        "statuses": statuses,
        "selection_uses_labels": False,
    }


def load_checkpoint(parent_sha: str, roots_sha: str) -> dict[str, Any] | None:
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(CHECKPOINT_PATH.read_bytes()).decode("utf-8"))
    except Exception:
        return None
    valid = (
        value.get("contract") == CONTRACT
        and value.get("parent_checkpoint_sha256") == parent_sha
        and value.get("roots_sha256") == roots_sha
        and value.get("selection_uses_labels") is False
    )
    return value if valid else None


def collect(
    roots: list[dict[str, str]],
    parent_sha: str,
    workers: int,
    timeout: float,
    retries: int,
    resume: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    roots_sha = hashlib.sha256(parent.canonical_json_bytes(roots)).hexdigest()
    checkpoint = load_checkpoint(parent_sha, roots_sha) if resume else None
    records = list(checkpoint.get("records", [])) if checkpoint else []
    quarantine = list(checkpoint.get("quarantine", [])) if checkpoint else []
    statuses = list(checkpoint.get("statuses", [])) if checkpoint else []
    completed = list(checkpoint.get("completed_roots", [])) if checkpoint else []
    completed_set = set(completed)
    pending = [root for root in roots if root["root_key"] not in completed_set]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as pool:
        futures = {
            pool.submit(refine_root, root, timeout, retries): root for root in pending
        }
        for done, future in enumerate(as_completed(futures), 1):
            root = futures[future]
            try:
                rows, quarantined, status = future.result()
            except Exception as exc:
                rows, quarantined = [], []
                status = {
                    **root, "request_count": 0, "request_failures": 1,
                    "terminal_saturated_count": 0, "raw_rows": 0,
                    "unexpected_error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            records.extend(rows)
            quarantine.extend(quarantined)
            statuses.append(status)
            completed.append(root["root_key"])
            if done % 10 == 0 or done == len(futures):
                parent.atomic_gzip(
                    CHECKPOINT_PATH,
                    parent.canonical_json_bytes(checkpoint_payload(
                        parent_sha, roots_sha, completed, records, quarantine, statuses
                    )),
                )
                print(
                    f"[KR GNEWS REFINE] roots={done}/{len(futures)} raw_rows={len(records)}",
                    flush=True,
                )
    frame, quarantine, dedup = parent.resolve_articles(records, quarantine)
    audit = {
        "checkpoint_resumed": bool(checkpoint),
        "configured_saturated_roots": len(roots),
        "completed_roots": len(completed),
        "http_requests": sum(int(item.get("request_count", 0)) for item in statuses),
        "request_failures": sum(int(item.get("request_failures", 0)) for item in statuses),
        "terminal_saturated_intervals": sum(
            int(item.get("terminal_saturated_count", 0)) for item in statuses
        ),
        "deduplication": dedup,
    }
    return frame, quarantine, statuses, audit


def write_outputs(
    frame: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    audit: dict[str, Any],
    parent_sha: str,
) -> dict[str, Any]:
    POOL.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    snapshot = POOL / f"gnews_kr_2025-08-20_2026-08-27_refine_{content_sha[:12]}.csv.gz"
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
    quarantine_path = POOL / f"gnews_kr_refine_quarantine_{quarantine_sha[:12]}.json.gz"
    parent.atomic_gzip(quarantine_path, quarantine_bytes)
    audit_payload = {
        "contract": CONTRACT,
        "source": SOURCE,
        "selection_uses_labels": False,
        "forbidden_inputs_read": [],
        "authorized_inputs": [
            {"path": str(parent.UNIVERSE_PATH.relative_to(ROOT)), "sha256": parent.EXPECTED_UNIVERSE_SHA256},
            {"path": str(parent.CHECKPOINT_PATH.relative_to(ROOT)), "sha256": parent_sha},
        ],
        "refinement_rule": f"recursively bisect V70 half-month responses capped at 100 until interval <= {MIN_SLICE_DAYS} days",
        "collection": audit,
        "root_statuses": statuses,
        "snapshot": {
            "path": str(snapshot.relative_to(ROOT)), "rows": len(frame),
            "content_sha256_uncompressed_csv": content_sha,
            "file_sha256": parent.sha256(snapshot), "bytes": snapshot.stat().st_size,
        },
        "quarantine": {
            "path": str(quarantine_path.relative_to(ROOT)), "rows": len(quarantine),
            "content_sha256": quarantine_sha, "file_sha256": parent.sha256(quarantine_path),
        },
        "complete_interval_claimed": False,
        "partial_progress_preserved": bool(
            audit["request_failures"] or audit["terminal_saturated_intervals"]
        ),
    }
    audit_bytes = parent.canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_kr_refine_audit_{audit_sha[:12]}.json"
    parent.atomic_bytes(audit_path, audit_bytes)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_PRESERVED" if audit_payload["partial_progress_preserved"] else "BEST_EFFORT_COMPLETE",
        "contract": CONTRACT, "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"],
        "audit_path": str(audit_path.relative_to(ROOT)), "audit_sha256": parent.sha256(audit_path),
        "quarantine": audit_payload["quarantine"], "collection": audit,
        "downstream_queue_not_modified": True,
    }
    parent.atomic_bytes(
        STATUS_PATH,
        json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("workers/timeout must be positive and retries non-negative")
    universe, input_audit = parent.file_input_audit()
    value, parent_sha = parent_checkpoint()
    roots = saturated_roots(value)
    root_tickers = {root["ticker"] for root in roots}
    if not root_tickers.issubset(set(universe.ticker.astype(str))):
        raise RuntimeError("SATURATED_ROOT_OUTSIDE_PINNED_UNIVERSE")
    summary = {
        "status": "AUDIT_OK", "contract": CONTRACT, "selection_uses_labels": False,
        "authorized_universe": input_audit, "parent_checkpoint_sha256": parent_sha,
        "saturated_half_month_roots": len(roots), "root_tickers": len(root_tickers),
        "minimum_terminal_slice_days": MIN_SLICE_DAYS,
        "base_refinement_request_floor": len(roots) * 2,
        "forbidden_inputs_read": [], "full_collection_executed": False,
    }
    if args.audit_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    frame, quarantine, statuses, audit = collect(
        roots, parent_sha, args.workers, args.timeout, args.retries, not args.no_resume
    )
    print(json.dumps(
        write_outputs(frame, quarantine, statuses, audit, parent_sha),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
