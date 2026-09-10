"""Label-blind, monthly sliced Google News RSS acquisition for KR healthcare.

Only the pinned 413-row V29 KR healthcare universe is read.  No event prior,
role, price, return, selection, or outcome file is opened by this collector.
Google RSS ``pubDate`` is parsed as an exact UTC-second timestamp and filtered
to the requested slice.  Articles returned by two or more ticker queries are
quarantined as ``AMBIGUOUS_MULTI_TICKER`` rather than assigned arbitrarily.

The normal run writes an immutable content-addressed discovery snapshot.  The
downstream V59 queue builder independently re-applies V36 event/article/URL and
same-ticker 35-minute prior exclusion plus current-pool 35-minute spacing.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import os
import re
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POOL = DATA / "untouched_pool"
RESEARCH = ROOT / "research"
STATE = ROOT / "state"
UNIVERSE_PATH = DATA / "universe_v29_kr_health.csv"
EXPECTED_UNIVERSE_SHA256 = "f29ecdae6525aaaa07d02ec82a7c6cddf6734598a6759c06815c05255e3fb0a7"
EXPECTED_UNIVERSE_ROWS = 413
START_UTC = pd.Timestamp("2025-08-20T00:00:00Z")
END_EXCLUSIVE_UTC = pd.Timestamp("2026-08-28T00:00:00Z")  # Includes all of 2026-08-27 UTC.
RSS_URL = "https://news.google.com/rss/search"
SOURCE = "GOOGLE_NEWS_RSS_KR_V70_EXTENSION"
CONTRACT = "KR_GNEWS_MONTHLY_SLICED_V70"
SATURATION_COUNT = 100
MAX_RESPONSE_BYTES = 15 * 1024 * 1024
DEFAULT_WORKERS = 8
USER_AGENT = "Mozilla/5.0 MARKET_BIO label-blind Google News RSS collector/70"
STATUS_PATH = RESEARCH / "V70_KR_GNEWS_EXTENSION_STATUS.json"
CHECKPOINT_PATH = STATE / "V70_KR_GNEWS_EXTENSION_CHECKPOINT.json.gz"
_THREAD_LOCAL = threading.local()

OUTPUT_COLUMNS = [
    "event_id",
    "market",
    "ticker",
    "company",
    "event_time_utc",
    "source",
    "form",
    "headline",
    "body",
    "event_type",
    "timestamp_quality",
    "url",
    "article_id",
    "selection_uses_labels",
    "headline_company_exact_evidence",
    "headline_company_boundary_evidence",
    "query_slice_start_utc",
    "query_slice_end_exclusive_utc",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    ).encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def deterministic_gzip_bytes(payload: bytes) -> bytes:
    from io import BytesIO

    target = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
        compressed.write(payload)
    return target.getvalue()


def atomic_gzip(path: Path, payload: bytes) -> None:
    atomic_bytes(path, deterministic_gzip_bytes(payload))


def month_start(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=value.year, month=value.month, day=1, tz="UTC")


def next_month(value: pd.Timestamp) -> pd.Timestamp:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def monthly_slices(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not start < end_exclusive:
        raise ValueError("start must precede end_exclusive")
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end_exclusive:
        boundary = next_month(month_start(cursor))
        stop = min(boundary, end_exclusive)
        result.append((cursor, stop))
        cursor = stop
    return result


def file_input_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    actual = sha256(UNIVERSE_PATH)
    if actual != EXPECTED_UNIVERSE_SHA256:
        raise RuntimeError("pinned V29 KR healthcare universe changed")
    universe = pd.read_csv(
        UNIVERSE_PATH,
        usecols=["market", "ticker", "company", "industry_code"],
        dtype=str,
        keep_default_na=False,
    )
    universe["ticker"] = universe.ticker.astype(str).str.strip().str.zfill(6)
    if len(universe) != EXPECTED_UNIVERSE_ROWS or not universe.ticker.is_unique:
        raise RuntimeError("V29 KR healthcare universe row/uniqueness contract changed")
    if not universe.market.eq("KR").all():
        raise RuntimeError("V29 KR healthcare universe contains a non-KR row")
    if not universe.ticker.str.fullmatch(r"[0-9]{6}").all() or universe.company.str.strip().eq("").any():
        raise RuntimeError("V29 KR healthcare ticker/company contract changed")
    return universe.sort_values("ticker").reset_index(drop=True), {
        "path": str(UNIVERSE_PATH.relative_to(ROOT)),
        "sha256": actual,
        "rows": len(universe),
        "unique_tickers": int(universe.ticker.nunique()),
        "selection_uses_labels": False,
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def company_evidence(company: str, headline: str) -> tuple[bool, bool]:
    company_norm = normalize_text(company)
    headline_norm = normalize_text(headline)
    exact = bool(company_norm and company_norm in headline_norm)
    if not company_norm:
        return False, False
    boundary_pattern = rf"(?<![0-9a-z가-힣]){re.escape(company_norm)}(?![0-9a-z가-힣])"
    boundary = bool(re.search(boundary_pattern, headline_norm, flags=re.IGNORECASE))
    return exact, boundary


def canonical_google_link(value: str) -> str | None:
    try:
        parsed = urlparse(str(value).strip())
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.hostname != "news.google.com":
        return None
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if not re.fullmatch(r"/(?:rss/)?articles/[A-Za-z0-9_-]+", path):
        return None
    # Google emits ``?oc=5`` for the same article in the V34 source.  Pinning
    # that harmless canonical query keeps URL/article overlap compatible with
    # V34 while eliminating locale/tracking parameter variants.
    return f"https://news.google.com{path}?oc=5"


def exact_pubdate(value: str) -> pd.Timestamp | None:
    raw = str(value or "").strip()
    if not re.search(r"\b[0-9]{2}:[0-9]{2}:[0-9]{2}\b", raw):
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None or parsed.tzinfo is None:
        return None
    return pd.Timestamp(parsed.astimezone(timezone.utc).replace(microsecond=0))


def session() -> requests.Session:
    current = getattr(_THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        current.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml;q=0.9"}
        )
        _THREAD_LOCAL.session = current
    return current


def safe_request(
    query: str,
    timeout: float,
    retries: int,
) -> tuple[requests.Response | None, str, int]:
    response: requests.Response | None = None
    error = ""
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response = session().get(
                RSS_URL,
                params={"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"},
                timeout=timeout,
            )
            if response.status_code == 200:
                return response, "", attempts
            error = f"HTTP_{response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.5 * (2**attempt)
            time.sleep(min(delay, 10.0))
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    return response, error, attempts


def parse_query_response(
    response: requests.Response,
    ticker: str,
    company: str,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counters = {"items": 0, "kept": 0, "outside_slice": 0, "invalid": 0}
    if len(response.content) > MAX_RESPONSE_BYTES:
        quarantine.append(
            {"reason": "RESPONSE_TOO_LARGE", "ticker": ticker, "bytes": len(response.content)}
        )
        counters["invalid"] += 1
        return records, quarantine, counters
    try:
        items = ET.fromstring(response.content).findall(".//item")
    except ET.ParseError as exc:
        quarantine.append({"reason": "MALFORMED_XML", "ticker": ticker, "detail": str(exc)[:240]})
        counters["invalid"] += 1
        return records, quarantine, counters
    counters["items"] = len(items)
    for item in items:
        headline = html.unescape(str(item.findtext("title") or "").strip())
        raw_link = str(item.findtext("link") or item.findtext("guid") or "").strip()
        link = canonical_google_link(raw_link)
        raw_pubdate = str(item.findtext("pubDate") or "").strip()
        published = exact_pubdate(raw_pubdate)
        if not headline:
            quarantine.append({"reason": "HEADLINE_MISSING", "ticker": ticker, "url": raw_link})
            counters["invalid"] += 1
            continue
        if link is None:
            quarantine.append({"reason": "LINK_INVALID", "ticker": ticker, "url": raw_link[:500]})
            counters["invalid"] += 1
            continue
        if published is None:
            quarantine.append(
                {"reason": "PUBDATE_NOT_EXACT_SECOND", "ticker": ticker, "url": link, "pubDate": raw_pubdate[:120]}
            )
            counters["invalid"] += 1
            continue
        if not (slice_start <= published < slice_end):
            counters["outside_slice"] += 1
            continue
        publisher_node = item.find("source")
        publisher = html.unescape(str(publisher_node.text if publisher_node is not None else "").strip())
        article_hash = hashlib.sha256(link.encode("utf-8")).hexdigest()
        exact_evidence, boundary_evidence = company_evidence(company, headline)
        records.append(
            {
                "event_id": f"GNEWS:{ticker}:{article_hash}",
                "market": "KR",
                "ticker": ticker,
                "company": company,
                "event_time_utc": published.isoformat(),
                "source": SOURCE,
                "form": "",
                "headline": headline,
                "body": publisher,
                "event_type": app.infer_event_type(headline),
                "timestamp_quality": "EXACT_SECOND_GOOGLE_RSS_PUBDATE",
                "url": link,
                "article_id": f"GNEWS:{article_hash}",
                "selection_uses_labels": False,
                "headline_company_exact_evidence": exact_evidence,
                "headline_company_boundary_evidence": boundary_evidence,
                "query_slice_start_utc": slice_start.isoformat(),
                "query_slice_end_exclusive_utc": slice_end.isoformat(),
            }
        )
        counters["kept"] += 1
    return records, quarantine, counters


def query_once(
    ticker: str,
    company: str,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query = f'"{company}" after:{slice_start.date().isoformat()} before:{slice_end.date().isoformat()}'
    response, error, attempts = safe_request(query, timeout, retries)
    status = {
        "ticker": ticker,
        "slice_start": slice_start.isoformat(),
        "slice_end_exclusive": slice_end.isoformat(),
        "attempts": attempts,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "status": "ERROR",
        "http_status": response.status_code if response is not None else None,
        "error": error,
        "items": 0,
        "kept": 0,
        "outside_slice": 0,
        "invalid": 0,
    }
    if response is None or response.status_code != 200:
        return [], [], status
    records, quarantine, counts = parse_query_response(
        response, ticker, company, slice_start, slice_end
    )
    status.update(counts)
    status["status"] = "OK"
    return records, quarantine, status


def fetch_base_task(
    ticker: str,
    company: str,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, quarantine, base = query_once(
        ticker, company, slice_start, slice_end, timeout, retries
    )
    subrequests: list[dict[str, Any]] = []
    adaptive = bool(base["status"] == "OK" and int(base["items"]) >= SATURATION_COUNT)
    if adaptive:
        days = int((slice_end - slice_start) / pd.Timedelta(days=1))
        midpoint = slice_start + pd.Timedelta(days=max(1, days // 2))
        if midpoint < slice_end:
            for sub_start, sub_end in ((slice_start, midpoint), (midpoint, slice_end)):
                sub_records, sub_quarantine, sub_status = query_once(
                    ticker, company, sub_start, sub_end, timeout, retries
                )
                records.extend(sub_records)
                quarantine.extend(sub_quarantine)
                subrequests.append(sub_status)
    request_failures = int(base["status"] != "OK") + sum(
        int(item["status"] != "OK") for item in subrequests
    )
    saturated_halves = sum(
        int(item["status"] == "OK" and int(item["items"]) >= SATURATION_COUNT)
        for item in subrequests
    )
    status = {
        "base_key": f"{ticker}|{slice_start.isoformat()}|{slice_end.isoformat()}",
        "ticker": ticker,
        "company": company,
        "slice_start": slice_start.isoformat(),
        "slice_end_exclusive": slice_end.isoformat(),
        "adaptive_half_month_split": adaptive,
        "request_count": 1 + len(subrequests),
        "request_failures": request_failures,
        "saturated_after_half_split": saturated_halves,
        "base_request": base,
        "subrequests": subrequests,
        "status": (
            "ERROR"
            if base["status"] != "OK"
            else "PARTIAL_ADAPTIVE_FAILURE"
            if request_failures
            else "SATURATED_AFTER_HALF_SPLIT"
            if saturated_halves
            else "OK"
        ),
    }
    return records, quarantine, status


def checkpoint_payload(
    input_audit: dict[str, Any],
    completed_slices: list[str],
    records: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "universe_sha256": input_audit["sha256"],
        "interval_start_utc": START_UTC.isoformat(),
        "interval_end_exclusive_utc": END_EXCLUSIVE_UTC.isoformat(),
        "completed_slices": completed_slices,
        "records": records,
        "quarantine": quarantine,
        "statuses": statuses,
        "selection_uses_labels": False,
    }


def save_checkpoint(payload: dict[str, Any]) -> None:
    atomic_gzip(CHECKPOINT_PATH, canonical_json_bytes(payload))


def load_checkpoint(input_audit: dict[str, Any]) -> dict[str, Any] | None:
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(CHECKPOINT_PATH.read_bytes()).decode("utf-8"))
    except Exception:
        return None
    valid = (
        value.get("contract") == CONTRACT
        and value.get("universe_sha256") == input_audit["sha256"]
        and value.get("interval_start_utc") == START_UTC.isoformat()
        and value.get("interval_end_exclusive_utc") == END_EXCLUSIVE_UTC.isoformat()
        and value.get("selection_uses_labels") is False
    )
    return value if valid else None


def resolve_articles(
    records: list[dict[str, Any]], quarantine: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), quarantine, {
            "raw_kept_rows": 0,
            "unique_links_seen": 0,
            "ambiguous_multi_ticker_links": 0,
            "inconsistent_metadata_links": 0,
            "deduplicated_same_ticker_rows": 0,
            "snapshot_rows": 0,
        }
    raw = pd.DataFrame(records)
    accepted: list[pd.Series] = []
    ambiguous = 0
    inconsistent = 0
    duplicate_rows = 0
    for link, group in raw.groupby("url", sort=True):
        tickers = sorted(set(group.ticker.astype(str)))
        if len(tickers) >= 2:
            ambiguous += 1
            quarantine.append(
                {
                    "reason": "AMBIGUOUS_MULTI_TICKER",
                    "url": link,
                    "article_id": str(group.article_id.iloc[0]),
                    "tickers": tickers,
                    "companies": sorted(set(group.company.astype(str))),
                    "headlines": sorted(set(group.headline.astype(str)))[:20],
                    "occurrences": len(group),
                }
            )
            continue
        if group.event_time_utc.nunique() != 1 or group.headline.map(normalize_text).nunique() != 1:
            inconsistent += 1
            quarantine.append(
                {
                    "reason": "INCONSISTENT_ARTICLE_METADATA",
                    "url": link,
                    "article_id": str(group.article_id.iloc[0]),
                    "ticker": tickers[0],
                    "event_times": sorted(set(group.event_time_utc.astype(str))),
                    "headlines": sorted(set(group.headline.astype(str)))[:20],
                    "occurrences": len(group),
                }
            )
            continue
        duplicate_rows += len(group) - 1
        stable = group.copy()
        stable["_row_hash"] = stable.apply(
            lambda row: hashlib.sha256(
                "|".join(str(row[column]) for column in OUTPUT_COLUMNS).encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
        accepted.append(stable.sort_values("_row_hash", kind="stable").iloc[0])
    frame = pd.DataFrame(accepted)
    if frame.empty:
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        frame = (
            frame.loc[:, OUTPUT_COLUMNS]
            .sort_values(["event_time_utc", "article_id", "ticker", "event_id"], kind="stable")
            .reset_index(drop=True)
        )
        if frame.article_id.duplicated().any() or frame.url.duplicated().any():
            raise RuntimeError("cross-ticker article/link deduplication failed")
    audit = {
        "raw_kept_rows": len(raw),
        "unique_links_seen": int(raw.url.nunique()),
        "ambiguous_multi_ticker_links": ambiguous,
        "inconsistent_metadata_links": inconsistent,
        "deduplicated_same_ticker_rows": duplicate_rows,
        "snapshot_rows": len(frame),
        "headline_company_exact_evidence_rows": int(
            frame.headline_company_exact_evidence.astype(bool).sum()
        ),
        "headline_company_boundary_evidence_rows": int(
            frame.headline_company_boundary_evidence.astype(bool).sum()
        ),
        "headline_company_no_exact_evidence_rows": int(
            (~frame.headline_company_exact_evidence.astype(bool)).sum()
        ),
    }
    return frame, quarantine, audit


def collect(
    universe: pd.DataFrame,
    slices: list[tuple[pd.Timestamp, pd.Timestamp]],
    input_audit: dict[str, Any],
    workers: int,
    timeout: float,
    retries: int,
    resume: bool,
    write_checkpoint: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = load_checkpoint(input_audit) if resume else None
    records = list(checkpoint.get("records", [])) if checkpoint else []
    quarantine = list(checkpoint.get("quarantine", [])) if checkpoint else []
    statuses = list(checkpoint.get("statuses", [])) if checkpoint else []
    completed_slices = list(checkpoint.get("completed_slices", [])) if checkpoint else []
    completed = set(completed_slices)
    resumed = bool(checkpoint)
    for slice_number, (slice_start, slice_end) in enumerate(slices, 1):
        slice_key = f"{slice_start.isoformat()}|{slice_end.isoformat()}"
        if slice_key in completed:
            continue
        with ThreadPoolExecutor(max_workers=min(workers, len(universe))) as pool:
            futures = {
                pool.submit(
                    fetch_base_task,
                    str(row.ticker),
                    str(row.company),
                    slice_start,
                    slice_end,
                    timeout,
                    retries,
                ): str(row.ticker)
                for row in universe.itertuples(index=False)
            }
            for done, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    task_records, task_quarantine, status = future.result()
                except Exception as exc:
                    task_records, task_quarantine = [], []
                    status = {
                        "ticker": ticker,
                        "slice_start": slice_start.isoformat(),
                        "slice_end_exclusive": slice_end.isoformat(),
                        "status": "UNEXPECTED_ERROR",
                        "request_count": 0,
                        "request_failures": 1,
                        "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    }
                records.extend(task_records)
                quarantine.extend(task_quarantine)
                statuses.append(status)
                if done % 50 == 0 or done == len(futures):
                    print(
                        f"[V70 KR GNEWS] slice={slice_number}/{len(slices)} tasks={done}/{len(futures)} raw_rows={len(records)}",
                        flush=True,
                    )
        completed_slices.append(slice_key)
        completed.add(slice_key)
        if write_checkpoint:
            save_checkpoint(
                checkpoint_payload(input_audit, completed_slices, records, quarantine, statuses)
            )
    frame, quarantine, dedup_audit = resolve_articles(records, quarantine)
    collection_audit = {
        "checkpoint_resumed": resumed,
        "completed_slices": len(completed_slices),
        "configured_slices": len(slices),
        "base_tasks_completed": len(statuses),
        "http_requests": sum(int(status.get("request_count", 0)) for status in statuses),
        "request_failures": sum(int(status.get("request_failures", 0)) for status in statuses),
        "adaptive_split_tasks": sum(bool(status.get("adaptive_half_month_split")) for status in statuses),
        "saturated_after_half_split_tasks": sum(
            int(status.get("saturated_after_half_split", 0)) > 0 for status in statuses
        ),
        "task_status_counts": dict(
            sorted(pd.Series([status.get("status", "UNKNOWN") for status in statuses]).value_counts().to_dict().items())
        ),
        "deduplication": dedup_audit,
    }
    return frame, quarantine, statuses, collection_audit


def write_outputs(
    frame: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    collection_audit: dict[str, Any],
    input_audit: dict[str, Any],
    slices: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    POOL.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    start_label = START_UTC.date().isoformat()
    end_label = (END_EXCLUSIVE_UTC - pd.Timedelta(days=1)).date().isoformat()
    snapshot_path = POOL / f"gnews_kr_{start_label}_{end_label}_{content_sha[:12]}.csv.gz"
    atomic_gzip(snapshot_path, csv_bytes)

    quarantine_payload = {
        "contract": CONTRACT,
        "selection_uses_labels": False,
        "rows": sorted(
            quarantine,
            key=lambda row: (
                str(row.get("reason", "")),
                str(row.get("article_id", "")),
                str(row.get("url", "")),
                str(row.get("ticker", "")),
            ),
        ),
    }
    quarantine_bytes = canonical_json_bytes(quarantine_payload)
    quarantine_sha = hashlib.sha256(quarantine_bytes).hexdigest()
    quarantine_path = POOL / f"gnews_kr_quarantine_{quarantine_sha[:12]}.json.gz"
    atomic_gzip(quarantine_path, quarantine_bytes)

    failures = [status for status in statuses if status.get("status") not in {"OK", "SATURATED_AFTER_HALF_SPLIT"}]
    audit_payload = {
        "contract": CONTRACT,
        "source": SOURCE,
        "selection_uses_labels": False,
        "credential_required": False,
        "authorized_input": input_audit,
        "interval": {
            "start_inclusive_utc": START_UTC.isoformat(),
            "end_exclusive_utc": END_EXCLUSIVE_UTC.isoformat(),
            "end_inclusive_date_label": end_label,
        },
        "slices": [
            {"start": start.isoformat(), "end_exclusive": end.isoformat()} for start, end in slices
        ],
        "query_contract": 'quoted company; after:YYYY-MM-DD before:YYYY-MM-DD; hl=ko gl=KR ceid=KR:ko',
        "timestamp_contract": "RSS pubDate with explicit HH:MM:SS and timezone, normalized to UTC seconds",
        "cross_ticker_contract": "two or more query tickers for one canonical link => AMBIGUOUS_MULTI_TICKER quarantine/exclude",
        "headline_company_evidence_contract": "NFKC/casefold exact substring and Unicode alphanumeric boundary flags; audited, not outcome-selected",
        "downstream_revalidation": (
            "v59_provider_acquisition._kr_candidates applies exact.outside_prior(event/article/URL/accession and same-ticker +/-35m), "
            "regular_session(KR), then exact.space_events current-pool ticker spacing"
        ),
        "collection": collection_audit,
        "request_failures": failures,
        "snapshot": {
            "path": str(snapshot_path.relative_to(ROOT)),
            "rows": len(frame),
            "content_sha256_uncompressed_csv": content_sha,
            "file_sha256": sha256(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
        },
        "quarantine": {
            "path": str(quarantine_path.relative_to(ROOT)),
            "rows": len(quarantine),
            "content_sha256": quarantine_sha,
            "file_sha256": sha256(quarantine_path),
        },
        "coverage_class": "BEST_EFFORT_MONTHLY_RSS_WITH_ADAPTIVE_HALF_MONTH_SPLIT",
        "partial_progress_preserved": bool(
            collection_audit["request_failures"]
            or collection_audit["saturated_after_half_split_tasks"]
        ),
        # Google RSS search is an undocumented, capped discovery surface.  Even
        # an error-free unsaturated run must never claim exhaustive coverage.
        "complete_interval_claimed": False,
    }
    audit_bytes = canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_kr_audit_{audit_sha[:12]}.json"
    atomic_bytes(audit_path, audit_bytes)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "BEST_EFFORT_COMPLETE"
            if collection_audit["request_failures"] == 0
            and collection_audit["saturated_after_half_split_tasks"] == 0
            else "PARTIAL_REQUEST_FAILURES_OR_SATURATION_PRESERVED"
        ),
        "contract": CONTRACT,
        "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"],
        "audit_path": str(audit_path.relative_to(ROOT)),
        "audit_sha256": sha256(audit_path),
        "quarantine": audit_payload["quarantine"],
        "collection": collection_audit,
        "downstream_queue_not_modified": True,
    }
    atomic_bytes(STATUS_PATH, json.dumps(status, ensure_ascii=False, indent=2, default=str).encode("utf-8") + b"\n")
    return status


def audit_summary(universe: pd.DataFrame, input_audit: dict[str, Any]) -> dict[str, Any]:
    slices = monthly_slices(START_UTC, END_EXCLUSIVE_UTC)
    base_requests = len(universe) * len(slices)
    return {
        "status": "AUDIT_OK",
        "contract": CONTRACT,
        "selection_uses_labels": False,
        "authorized_input": input_audit,
        "forbidden_inputs_read": [],
        "interval_start_inclusive_utc": START_UTC.isoformat(),
        "interval_end_exclusive_utc": END_EXCLUSIVE_UTC.isoformat(),
        "monthly_slices": len(slices),
        "slice_boundaries": [
            {"start": start.isoformat(), "end_exclusive": end.isoformat()} for start, end in slices
        ],
        "base_request_estimate": base_requests,
        "adaptive_request_increment": "2 per ticker-month whose base RSS response has >=100 items",
        "hard_request_ceiling_if_every_base_slice_saturates": base_requests * 3,
        "credential_required": False,
        "full_collection_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-tickers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.smoke_tickers < 1:
        raise SystemExit("workers/timeout/smoke-tickers must be positive and retries non-negative")
    universe, input_audit = file_input_audit()
    slices = monthly_slices(START_UTC, END_EXCLUSIVE_UTC)
    if args.audit_only:
        print(json.dumps(audit_summary(universe, input_audit), ensure_ascii=False, indent=2))
        return
    if args.smoke_test:
        universe = (
            universe.assign(
                _smoke_hash=universe.ticker.map(
                    lambda ticker: hashlib.sha256(f"V70_GNEWS_SMOKE|{ticker}".encode()).hexdigest()
                )
            )
            .sort_values("_smoke_hash")
            .head(args.smoke_tickers)
            .drop(columns="_smoke_hash")
        )
        slices = [slices[-1]]
        frame, quarantine, statuses, collection_audit = collect(
            universe,
            slices,
            input_audit,
            workers=min(args.workers, len(universe)),
            timeout=args.timeout,
            retries=args.retries,
            resume=False,
            write_checkpoint=False,
        )
        print(
            json.dumps(
                {
                    "status": "SMOKE_OK",
                    "contract": CONTRACT,
                    "tickers": universe[["ticker", "company"]].to_dict("records"),
                    "slice": {"start": slices[0][0].isoformat(), "end_exclusive": slices[0][1].isoformat()},
                    "collection": collection_audit,
                    "snapshot_candidate_rows": len(frame),
                    "quarantine_rows": len(quarantine),
                    "request_statuses": statuses,
                    "output_written": False,
                    "checkpoint_written": False,
                    "selection_uses_labels": False,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return
    frame, quarantine, statuses, collection_audit = collect(
        universe,
        slices,
        input_audit,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        resume=not args.no_resume,
        write_checkpoint=True,
    )
    status = write_outputs(
        frame, quarantine, statuses, collection_audit, input_audit, slices
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
