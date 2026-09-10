"""Label-blind monthly Google News RSS acquisition for the pinned US universe."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import collect_v70_kr_gnews_extension as common


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POOL = DATA / "untouched_pool"
RESEARCH = ROOT / "research"
STATE = ROOT / "state"
UNIVERSE_PATH = DATA / "universe_v33_us_rss.csv"
EXPECTED_UNIVERSE_SHA256 = "06d0b82f3706b2868aa63c76d154155fe11ff121f15662b50dda67996c9d48eb"
EXPECTED_UNIVERSE_ROWS = 796
START_UTC = pd.Timestamp("2025-08-20T00:00:00Z")
END_EXCLUSIVE_UTC = pd.Timestamp("2026-08-28T00:00:00Z")
CONTRACT = "US_GNEWS_MONTHLY_SLICED_V1"
SOURCE = "GOOGLE_NEWS_RSS_US_EXTENSION"
STATUS_PATH = RESEARCH / "US_GNEWS_EXTENSION_STATUS.json"
CHECKPOINT_PATH = STATE / "US_GNEWS_EXTENSION_CHECKPOINT.json.gz"
QUERY_PROFILE = "BROAD_COMPANY"
QUERY_TERMS = ""
QUERY_PROFILES = {
    "BROAD_COMPANY": "",
    "CLINICAL_REGULATORY_IP": '("clinical trial" OR FDA OR approval OR patent OR regulatory)',
    "CORPORATE_FINANCING_RESULTS": '(acquisition OR merger OR financing OR offering OR earnings OR partnership)',
}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 MARKET_BIO label-blind US Google News RSS collector"})


def input_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    actual = common.sha256(UNIVERSE_PATH)
    if actual != EXPECTED_UNIVERSE_SHA256:
        raise RuntimeError("PINNED_V33_US_UNIVERSE_CHANGED")
    frame = pd.read_csv(UNIVERSE_PATH, dtype=str, keep_default_na=False)
    if (
        len(frame) != EXPECTED_UNIVERSE_ROWS
        or not frame.ticker.is_unique
        or frame.company.str.strip().eq("").any()
        or not frame.ticker.str.fullmatch(r"[A-Z0-9.-]+").all()
    ):
        raise RuntimeError("PINNED_V33_US_UNIVERSE_CONTRACT_CHANGED")
    return frame.sort_values("ticker").reset_index(drop=True), {
        "path": str(UNIVERSE_PATH.relative_to(ROOT)), "sha256": actual,
        "rows": len(frame), "unique_tickers": int(frame.ticker.nunique()),
        "selection_uses_labels": False,
    }


def safe_request(query: str, timeout: float, retries: int) -> tuple[requests.Response | None, str, int]:
    response: requests.Response | None = None
    error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                common.RSS_URL,
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": SESSION.headers["User-Agent"], "Accept": "application/rss+xml,application/xml;q=0.9"},
                timeout=timeout,
            )
            if response.status_code == 200:
                return response, "", attempt + 1
            error = f"HTTP_{response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.5 * (2**attempt)
            time.sleep(min(delay + random.random() * 0.2, 10.0))
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    return response, error, retries + 1


def query_once(
    ticker: str,
    company: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_company = str(company).replace('"', "").strip()
    terms = f" {QUERY_TERMS}" if QUERY_TERMS else ""
    query = f'"{clean_company}"{terms} after:{start.date().isoformat()} before:{end.date().isoformat()}'
    response, error, attempts = safe_request(query, timeout, retries)
    status = {
        "ticker": ticker, "slice_start": start.isoformat(), "slice_end_exclusive": end.isoformat(),
        "attempts": attempts, "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "status": "ERROR", "http_status": response.status_code if response is not None else None,
        "error": error, "items": 0, "kept": 0, "outside_slice": 0, "invalid": 0,
    }
    if response is None or response.status_code != 200:
        return [], [], status
    records, quarantine, counts = common.parse_query_response(
        response, ticker, company, start, end
    )
    for row in records:
        row["market"] = "US"
        row["source"] = SOURCE
    status.update(counts); status["status"] = "OK"
    return records, quarantine, status


def fetch_task(
    ticker: str,
    company: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, quarantine, base = query_once(ticker, company, start, end, timeout, retries)
    subrequests: list[dict[str, Any]] = []
    adaptive = bool(base["status"] == "OK" and int(base["items"]) >= common.SATURATION_COUNT)
    if adaptive:
        days = int((end - start) / pd.Timedelta(days=1))
        midpoint = start + pd.Timedelta(days=max(1, days // 2))
        if midpoint < end:
            for child_start, child_end in ((start, midpoint), (midpoint, end)):
                rows, quarantined, request = query_once(
                    ticker, company, child_start, child_end, timeout, retries
                )
                records.extend(rows); quarantine.extend(quarantined); subrequests.append(request)
    failures = int(base["status"] != "OK") + sum(item["status"] != "OK" for item in subrequests)
    saturated_halves = sum(
        item["status"] == "OK" and int(item["items"]) >= common.SATURATION_COUNT
        for item in subrequests
    )
    return records, quarantine, {
        "task_key": f"{ticker}|{start.isoformat()}|{end.isoformat()}",
        "ticker": ticker, "company": company, "slice_start": start.isoformat(),
        "slice_end_exclusive": end.isoformat(), "adaptive_half_month_split": adaptive,
        "request_count": 1 + len(subrequests), "request_failures": failures,
        "saturated_after_half_split": saturated_halves,
        "base_request": base, "subrequests": subrequests,
        "status": (
            "ERROR" if base["status"] != "OK" else
            "PARTIAL_ADAPTIVE_FAILURE" if failures else
            "SATURATED_AFTER_HALF_SPLIT" if saturated_halves else "OK"
        ),
    }


def checkpoint_value(
    universe_sha: str,
    completed: list[str],
    records: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    circuit_breaker_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "contract": CONTRACT, "universe_sha256": universe_sha,
        "completed_slices": completed, "records": records,
        "quarantine": quarantine, "statuses": statuses,
        "circuit_breaker_events": circuit_breaker_events or [],
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
        and value.get("selection_uses_labels") is False
    )
    return value if valid else None


def reopen_systemic_failure_slices(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Reopen legacy slices incorrectly marked complete during provider outages."""
    statuses = list(checkpoint.get("statuses", []))
    completed = list(checkpoint.get("completed_slices", []))
    kept_completed: list[str] = []
    events: list[dict[str, Any]] = []
    for slice_key in completed:
        try:
            start, end = slice_key.split("|", 1)
        except ValueError:
            kept_completed.append(slice_key)
            continue
        members = [
            item for item in statuses
            if item.get("slice_start") == start and item.get("slice_end_exclusive") == end
        ]
        retryable_errors = sum(
            item.get("status") == "ERROR"
            and item.get("base_request", {}).get("http_status") in {429, 500, 502, 503, 504}
            for item in members
        )
        kept_rows = sum(
            int(item.get("base_request", {}).get("kept", 0))
            + sum(int(child.get("kept", 0)) for child in item.get("subrequests", []))
            for item in members
        )
        reopen = (
            len(members) >= 100
            and retryable_errors / max(len(members), 1) >= 0.95
            and kept_rows == 0
        )
        if not reopen:
            kept_completed.append(slice_key)
            continue
        events.append({
            "slice": slice_key,
            "tasks": len(members),
            "retryable_error_tasks": retryable_errors,
            "retryable_error_fraction": retryable_errors / len(members),
            "kept_rows": kept_rows,
            "action": "REOPEN_SYSTEMIC_FAILURE_ON_RESUME",
            "selection_uses_labels": False,
        })
    checkpoint["completed_slices"] = kept_completed
    return events


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
    reopened_events = reopen_systemic_failure_slices(checkpoint) if checkpoint else []
    records = list(checkpoint.get("records", [])) if checkpoint else []
    quarantine = list(checkpoint.get("quarantine", [])) if checkpoint else []
    statuses = list(checkpoint.get("statuses", [])) if checkpoint else []
    completed = list(checkpoint.get("completed_slices", [])) if checkpoint else []
    circuit_breaker_events = list(checkpoint.get("circuit_breaker_events", [])) if checkpoint else []
    circuit_breaker_events.extend(reopened_events)
    completed_set = set(completed)
    for number, (start, end) in enumerate(slices, 1):
        slice_key = f"{start.isoformat()}|{end.isoformat()}"
        if slice_key in completed_set:
            continue
        records_before = len(records)
        statuses_before = len(statuses)
        with ThreadPoolExecutor(max_workers=min(workers, len(universe))) as pool:
            futures = {
                pool.submit(fetch_task, str(row.ticker), str(row.company), start, end, timeout, retries): str(row.ticker)
                for row in universe.itertuples(index=False)
            }
            for done, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    rows, quarantined, status = future.result()
                except Exception as exc:
                    rows, quarantined = [], []
                    status = {
                        "task_key": f"{slice_key}|{ticker}", "ticker": ticker,
                        "slice_start": start.isoformat(), "slice_end_exclusive": end.isoformat(),
                        "status": "UNEXPECTED_ERROR", "request_count": 0, "request_failures": 1,
                        "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    }
                records.extend(rows); quarantine.extend(quarantined); statuses.append(status)
                if done % 100 == 0 or done == len(futures):
                    print(f"[US GNEWS] slice={number}/{len(slices)} tasks={done}/{len(futures)} raw_rows={len(records)}", flush=True)
        slice_statuses = statuses[statuses_before:]
        retryable_errors = sum(
            item.get("status") == "ERROR"
            and item.get("base_request", {}).get("http_status") in {429, 500, 502, 503, 504}
            for item in slice_statuses
        )
        breaker_triggered = (
            len(slice_statuses) >= 100
            and retryable_errors / max(len(slice_statuses), 1) >= 0.95
            and len(records) == records_before
        )
        if breaker_triggered:
            event = {
                "slice": slice_key,
                "tasks": len(slice_statuses),
                "retryable_error_tasks": retryable_errors,
                "retryable_error_fraction": retryable_errors / len(slice_statuses),
                "new_raw_rows": len(records) - records_before,
                "action": "PARTIAL_CHECKPOINT_AND_STOP",
                "selection_uses_labels": False,
            }
            circuit_breaker_events.append(event)
            common.atomic_gzip(
                CHECKPOINT_PATH,
                common.canonical_json_bytes(checkpoint_value(
                    universe_sha, completed, records, quarantine, statuses,
                    circuit_breaker_events,
                )),
            )
            print(f"[US GNEWS] upstream circuit breaker: {json.dumps(event, sort_keys=True)}", flush=True)
            break
        completed.append(slice_key); completed_set.add(slice_key)
        common.atomic_gzip(
            CHECKPOINT_PATH,
            common.canonical_json_bytes(checkpoint_value(
                universe_sha, completed, records, quarantine, statuses,
                circuit_breaker_events,
            )),
        )
    frame, quarantine, dedup = common.resolve_articles(records, quarantine)
    audit = {
        "checkpoint_resumed": bool(checkpoint), "configured_slices": len(slices),
        "completed_slices": len(completed), "base_tasks_completed": len(statuses),
        "http_requests": sum(int(item.get("request_count", 0)) for item in statuses),
        "request_failures": sum(int(item.get("request_failures", 0)) for item in statuses),
        "adaptive_split_tasks": sum(bool(item.get("adaptive_half_month_split")) for item in statuses),
        "saturated_after_half_split_tasks": sum(int(item.get("saturated_after_half_split", 0)) > 0 for item in statuses),
        "circuit_breaker_events": circuit_breaker_events,
        "deduplication": dedup,
    }
    return frame, quarantine, statuses, audit


def write_outputs(
    frame: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    audit: dict[str, Any],
    authorized_input: dict[str, Any],
) -> dict[str, Any]:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    profile_slug = QUERY_PROFILE.casefold()
    snapshot = POOL / f"gnews_us_2025-08-20_2026-08-27_{profile_slug}_{content_sha[:12]}.csv.gz"
    common.atomic_gzip(snapshot, csv_bytes)
    quarantine_payload = {
        "contract": CONTRACT, "selection_uses_labels": False,
        "rows": sorted(quarantine, key=lambda row: (
            str(row.get("reason", "")), str(row.get("article_id", "")),
            str(row.get("url", "")), str(row.get("ticker", "")),
        )),
    }
    quarantine_bytes = common.canonical_json_bytes(quarantine_payload)
    quarantine_sha = hashlib.sha256(quarantine_bytes).hexdigest()
    quarantine_path = POOL / f"gnews_us_quarantine_{quarantine_sha[:12]}.json.gz"
    common.atomic_gzip(quarantine_path, quarantine_bytes)
    audit_payload = {
        "contract": CONTRACT, "source": SOURCE, "selection_uses_labels": False,
        "authorized_input": authorized_input, "forbidden_inputs_read": [],
        "interval": {"start_inclusive_utc": START_UTC.isoformat(), "end_exclusive_utc": END_EXCLUSIVE_UTC.isoformat()},
        "locale": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
        "query_profile": QUERY_PROFILE, "query_terms": QUERY_TERMS,
        "timestamp_contract": "RSS pubDate exact UTC seconds",
        "collection": audit,
        "request_failures": [item for item in statuses if item.get("status") not in {"OK", "SATURATED_AFTER_HALF_SPLIT"}],
        "snapshot": {
            "path": str(snapshot.relative_to(ROOT)), "rows": len(frame),
            "content_sha256_uncompressed_csv": content_sha,
            "file_sha256": common.sha256(snapshot), "bytes": snapshot.stat().st_size,
        },
        "quarantine": {
            "path": str(quarantine_path.relative_to(ROOT)), "rows": len(quarantine),
            "content_sha256": quarantine_sha, "file_sha256": common.sha256(quarantine_path),
        },
        "complete_interval_claimed": False,
        "partial_progress_preserved": bool(audit["request_failures"] or audit["saturated_after_half_split_tasks"]),
    }
    audit_bytes = common.canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_us_audit_{audit_sha[:12]}.json"
    common.atomic_bytes(audit_path, audit_bytes)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_PRESERVED" if audit_payload["partial_progress_preserved"] else "BEST_EFFORT_COMPLETE",
        "contract": CONTRACT, "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"], "audit_path": str(audit_path.relative_to(ROOT)),
        "audit_sha256": common.sha256(audit_path), "quarantine": audit_payload["quarantine"],
        "collection": audit, "downstream_queue_not_modified": True,
    }
    common.atomic_bytes(STATUS_PATH, json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--query-profile",choices=sorted(QUERY_PROFILES),default="BROAD_COMPANY")
    return parser.parse_args()


def main() -> None:
    global CONTRACT, SOURCE, STATUS_PATH, CHECKPOINT_PATH, QUERY_PROFILE, QUERY_TERMS
    args = parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("workers/timeout must be positive and retries non-negative")
    QUERY_PROFILE=args.query_profile;QUERY_TERMS=QUERY_PROFILES[QUERY_PROFILE]
    if QUERY_PROFILE!="BROAD_COMPANY":
        slug=QUERY_PROFILE.casefold()
        CONTRACT=f"US_GNEWS_MONTHLY_{QUERY_PROFILE}_V1"
        SOURCE=f"GOOGLE_NEWS_RSS_US_{QUERY_PROFILE}"
        STATUS_PATH=RESEARCH/f"US_GNEWS_{slug}_STATUS.json"
        CHECKPOINT_PATH=STATE/f"US_GNEWS_{slug}_CHECKPOINT.json.gz"
    universe, authorized = input_audit()
    slices = common.monthly_slices(START_UTC, END_EXCLUSIVE_UTC)
    summary = {
        "status": "AUDIT_OK", "contract": CONTRACT, "selection_uses_labels": False,
        "authorized_input": authorized, "query_profile":QUERY_PROFILE,
        "query_terms":QUERY_TERMS,"monthly_slices": len(slices),
        "base_request_estimate": len(universe) * len(slices),
        "adaptive_increment": "2 per capped ticker-month response",
        "forbidden_inputs_read": [], "full_collection_executed": False,
    }
    if args.audit_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    frame, quarantine, statuses, audit = collect(
        universe, slices, authorized["sha256"], args.workers, args.timeout,
        args.retries, not args.no_resume,
    )
    print(json.dumps(write_outputs(frame, quarantine, statuses, audit, authorized), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
