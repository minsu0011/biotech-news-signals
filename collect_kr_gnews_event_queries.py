"""Label-blind KR Google News acquisition with two predeclared event query families."""
from __future__ import annotations

import argparse
import gzip
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
STATE = ROOT / "state"
CONTRACT = "KR_GNEWS_PREDECLARED_EVENT_QUERIES_V1"
SOURCE = "GOOGLE_NEWS_RSS_KR_EVENT_QUERY"
STATUS_PATH = RESEARCH / "KR_GNEWS_EVENT_QUERY_STATUS.json"
CHECKPOINT_PATH = STATE / "KR_GNEWS_EVENT_QUERY_CHECKPOINT.json.gz"
QUERY_FAMILIES = {
    "CLINICAL_REGULATORY_IP": "(임상 OR 허가 OR 승인 OR FDA OR 식약처 OR 특허)",
    "CORPORATE_FINANCING_RESULTS": "(계약 OR 수출 OR 투자 OR 인수 OR 합병 OR 증자 OR 실적)",
}


def query_request(
    ticker: str,
    company: str,
    family: str,
    terms: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_company = str(company).replace('"', "").strip()
    query = (
        f'"{clean_company}" {terms} '
        f'after:{start.date().isoformat()} before:{end.date().isoformat()}'
    )
    response, error, attempts = parent.safe_request(query, timeout, retries)
    status = {
        "ticker": ticker, "company": company, "query_family": family,
        "slice_start": start.isoformat(), "slice_end_exclusive": end.isoformat(),
        "attempts": attempts, "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "status": "ERROR", "http_status": response.status_code if response is not None else None,
        "error": error, "items": 0, "kept": 0, "outside_slice": 0, "invalid": 0,
    }
    if response is None or response.status_code != 200:
        return [], [], status
    records, quarantine, counts = parent.parse_query_response(
        response, ticker, company, start, end
    )
    for row in records:
        row["source"] = SOURCE
        row["query_family"] = family
    status.update(counts)
    status["status"] = "OK"
    return records, quarantine, status


def fetch_task(
    ticker: str,
    company: str,
    family: str,
    terms: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, quarantine, base = query_request(
        ticker, company, family, terms, start, end, timeout, retries
    )
    subrequests: list[dict[str, Any]] = []
    adaptive = bool(base["status"] == "OK" and int(base["items"]) >= parent.SATURATION_COUNT)
    if adaptive:
        days = int((end - start) / pd.Timedelta(days=1))
        midpoint = start + pd.Timedelta(days=max(1, days // 2))
        if midpoint < end:
            for child_start, child_end in ((start, midpoint), (midpoint, end)):
                rows, quarantined, request = query_request(
                    ticker, company, family, terms, child_start, child_end, timeout, retries
                )
                records.extend(rows)
                quarantine.extend(quarantined)
                subrequests.append(request)
    failures = int(base["status"] != "OK") + sum(item["status"] != "OK" for item in subrequests)
    saturated_children = sum(
        item["status"] == "OK" and int(item["items"]) >= parent.SATURATION_COUNT
        for item in subrequests
    )
    return records, quarantine, {
        "task_key": f"{family}|{ticker}|{start.isoformat()}|{end.isoformat()}",
        "ticker": ticker, "company": company, "query_family": family,
        "slice_start": start.isoformat(), "slice_end_exclusive": end.isoformat(),
        "adaptive_half_month_split": adaptive,
        "request_count": 1 + len(subrequests), "request_failures": failures,
        "saturated_after_half_split": saturated_children,
        "base_request": base, "subrequests": subrequests,
        "status": (
            "ERROR" if base["status"] != "OK" else
            "PARTIAL_ADAPTIVE_FAILURE" if failures else
            "SATURATED_AFTER_HALF_SPLIT" if saturated_children else "OK"
        ),
    }


def checkpoint_payload(
    universe_sha: str,
    completed_blocks: list[str],
    records: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract": CONTRACT, "universe_sha256": universe_sha,
        "query_families": QUERY_FAMILIES, "completed_blocks": completed_blocks,
        "records": records, "quarantine": quarantine, "statuses": statuses,
        "selection_uses_labels": False,
    }


def load_checkpoint(universe_sha: str) -> dict[str, Any] | None:
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(CHECKPOINT_PATH.read_bytes()).decode("utf-8"))
    except Exception:
        return None
    valid = (
        value.get("contract") == CONTRACT
        and value.get("universe_sha256") == universe_sha
        and value.get("query_families") == QUERY_FAMILIES
        and value.get("selection_uses_labels") is False
    )
    return value if valid else None


def collect(
    universe: pd.DataFrame,
    slices: list[tuple[pd.Timestamp, pd.Timestamp]],
    universe_sha: str,
    workers: int,
    timeout: float,
    retries: int,
    resume: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = load_checkpoint(universe_sha) if resume else None
    records = list(checkpoint.get("records", [])) if checkpoint else []
    quarantine = list(checkpoint.get("quarantine", [])) if checkpoint else []
    statuses = list(checkpoint.get("statuses", [])) if checkpoint else []
    completed_blocks = list(checkpoint.get("completed_blocks", [])) if checkpoint else []
    completed = set(completed_blocks)
    total_blocks = len(QUERY_FAMILIES) * len(slices)
    block_number = 0
    for family, terms in QUERY_FAMILIES.items():
        for start, end in slices:
            block_number += 1
            block_key = f"{family}|{start.isoformat()}|{end.isoformat()}"
            if block_key in completed:
                continue
            with ThreadPoolExecutor(max_workers=min(workers, len(universe))) as pool:
                futures = {
                    pool.submit(
                        fetch_task, str(row.ticker), str(row.company), family, terms,
                        start, end, timeout, retries,
                    ): str(row.ticker)
                    for row in universe.itertuples(index=False)
                }
                for done, future in enumerate(as_completed(futures), 1):
                    ticker = futures[future]
                    try:
                        rows, quarantined, status = future.result()
                    except Exception as exc:
                        rows, quarantined = [], []
                        status = {
                            "task_key": f"{block_key}|{ticker}", "ticker": ticker,
                            "query_family": family, "slice_start": start.isoformat(),
                            "slice_end_exclusive": end.isoformat(), "status": "UNEXPECTED_ERROR",
                            "request_count": 0, "request_failures": 1,
                            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                        }
                    records.extend(rows)
                    quarantine.extend(quarantined)
                    statuses.append(status)
                    if done % 100 == 0 or done == len(futures):
                        print(
                            f"[KR GNEWS EVENT] block={block_number}/{total_blocks} tasks={done}/{len(futures)} raw_rows={len(records)}",
                            flush=True,
                        )
            completed_blocks.append(block_key)
            completed.add(block_key)
            parent.atomic_gzip(
                CHECKPOINT_PATH,
                parent.canonical_json_bytes(checkpoint_payload(
                    universe_sha, completed_blocks, records, quarantine, statuses
                )),
            )
    frame, quarantine, dedup = parent.resolve_articles(records, quarantine)
    audit = {
        "checkpoint_resumed": bool(checkpoint), "configured_blocks": total_blocks,
        "completed_blocks": len(completed_blocks), "base_tasks_completed": len(statuses),
        "http_requests": sum(int(item.get("request_count", 0)) for item in statuses),
        "request_failures": sum(int(item.get("request_failures", 0)) for item in statuses),
        "adaptive_split_tasks": sum(bool(item.get("adaptive_half_month_split")) for item in statuses),
        "saturated_after_half_split_tasks": sum(
            int(item.get("saturated_after_half_split", 0)) > 0 for item in statuses
        ),
        "deduplication": dedup,
    }
    return frame, quarantine, statuses, audit


def write_outputs(
    frame: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    audit: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    # The immutable audit records both fixed query families; the downstream
    # event schema intentionally remains identical to the V70 snapshot schema.
    frame = frame.reindex(columns=parent.OUTPUT_COLUMNS)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    snapshot = POOL / f"gnews_kr_2025-08-20_2026-08-27_eventq_{content_sha[:12]}.csv.gz"
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
    quarantine_path = POOL / f"gnews_kr_eventq_quarantine_{quarantine_sha[:12]}.json.gz"
    parent.atomic_gzip(quarantine_path, quarantine_bytes)
    audit_payload = {
        "contract": CONTRACT, "source": SOURCE, "selection_uses_labels": False,
        "authorized_input": input_audit, "forbidden_inputs_read": [],
        "query_families": QUERY_FAMILIES,
        "query_family_selection": "predeclared semantic event taxonomy; no outcomes, prices, or roles",
        "collection": audit, "request_failures": [
            item for item in statuses if item.get("status") not in {"OK", "SATURATED_AFTER_HALF_SPLIT"}
        ],
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
            audit["request_failures"] or audit["saturated_after_half_split_tasks"]
        ),
    }
    audit_bytes = parent.canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_kr_eventq_audit_{audit_sha[:12]}.json"
    parent.atomic_bytes(audit_path, audit_bytes)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_PRESERVED" if audit_payload["partial_progress_preserved"] else "BEST_EFFORT_COMPLETE",
        "contract": CONTRACT, "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"], "audit_path": str(audit_path.relative_to(ROOT)),
        "audit_sha256": parent.sha256(audit_path), "quarantine": audit_payload["quarantine"],
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
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("workers/timeout must be positive and retries non-negative")
    universe, input_audit = parent.file_input_audit()
    slices = parent.monthly_slices(parent.START_UTC, parent.END_EXCLUSIVE_UTC)
    summary = {
        "status": "AUDIT_OK", "contract": CONTRACT, "selection_uses_labels": False,
        "authorized_input": input_audit, "query_families": QUERY_FAMILIES,
        "monthly_slices": len(slices), "base_request_estimate": len(universe) * len(slices) * len(QUERY_FAMILIES),
        "adaptive_increment": "2 for each capped ticker-month-family response",
        "forbidden_inputs_read": [], "full_collection_executed": False,
    }
    if args.audit_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    frame, quarantine, statuses, audit = collect(
        universe, slices, input_audit["sha256"], args.workers, args.timeout,
        args.retries, not args.no_resume,
    )
    print(json.dumps(
        write_outputs(frame, quarantine, statuses, audit, input_audit),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
