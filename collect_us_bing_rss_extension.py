"""Label-blind Bing News RSS acquisition for the pinned US healthcare universe."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests

import bio_news_30m_v3 as app
import collect_v70_kr_gnews_extension as common


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POOL = DATA / "untouched_pool"
RESEARCH = ROOT / "research"
STATE = ROOT / "state"
UNIVERSE = DATA / "universe_v33_us_rss.csv"
EXPECTED_SHA256 = "06d0b82f3706b2868aa63c76d154155fe11ff121f15662b50dda67996c9d48eb"
EXPECTED_ROWS = 796
START = pd.Timestamp("2025-08-20T00:00:00Z")
END = pd.Timestamp("2026-08-28T00:00:00Z")
RSS_URL = "https://www.bing.com/news/search"
SOURCE = "BING_NEWS_RSS_US_EXTENSION"
CONTRACT = "US_BING_NEWS_RSS_PINNED_UNIVERSE_V1"
CHECKPOINT = STATE / "US_BING_RSS_CHECKPOINT.json.gz"
STATUS = RESEARCH / "US_BING_RSS_EXTENSION_STATUS.json"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TRACKING_KEYS = {"ref", "refsrc", "source", "fbclid", "gclid", "guce_referrer", "guce_referrer_sig"}
GENERIC_FIRST_TOKENS = {
    "american", "china", "global", "health", "healthcare", "international",
    "national", "new", "the", "united", "universal", "bio",
}


def input_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    actual = common.sha256(UNIVERSE)
    if actual != EXPECTED_SHA256:
        raise RuntimeError("PINNED_US_UNIVERSE_CHANGED")
    frame = pd.read_csv(UNIVERSE, dtype=str, keep_default_na=False)
    valid = (
        len(frame) == EXPECTED_ROWS and frame.ticker.is_unique
        and frame.ticker.str.fullmatch(r"[A-Z0-9.-]+").all()
        and not frame.company.str.strip().eq("").any()
    )
    if not valid:
        raise RuntimeError("PINNED_US_UNIVERSE_CONTRACT_CHANGED")
    return frame.sort_values("ticker").reset_index(drop=True), {
        "path": str(UNIVERSE.relative_to(ROOT)), "sha256": actual,
        "rows": len(frame), "unique_tickers": int(frame.ticker.nunique()),
        "selection_uses_labels": False,
    }


def canonical_article_url(raw: str) -> str | None:
    try:
        parsed = urlparse(str(raw).strip())
    except ValueError:
        return None
    if parsed.hostname and parsed.hostname.casefold().endswith("bing.com"):
        target = (parse_qs(parsed.query).get("url") or [""])[0]
        try:
            parsed = urlparse(target)
        except ValueError:
            return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.casefold()
    port = f":{parsed.port}" if parsed.port and parsed.port not in {80, 443} else ""
    query = urlencode([
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_KEYS and not key.casefold().startswith("utm_")
    ])
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunparse(("https" if parsed.scheme == "https" else "http", host + port, path, "", query, ""))


def name_evidence(company: str, ticker: str, text: str) -> dict[str, bool]:
    normalized = common.normalize_text(text)
    company_norm = common.normalize_text(company)
    words = re.findall(r"[0-9a-z]+", company_norm)
    suffixes = {"inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited", "plc", "sa", "nv"}
    while words and words[-1] in suffixes:
        words.pop()
    alias = " ".join(words)
    full = bool(company_norm and company_norm in normalized)
    alias_match = bool(len(alias) >= 5 and alias in normalized)
    first = words[0] if words else ""
    first_match = bool(
        len(first) >= 5 and first not in GENERIC_FIRST_TOKENS
        and re.search(rf"(?<![0-9a-z]){re.escape(first)}(?![0-9a-z])", normalized)
    )
    ticker_norm = common.normalize_text(ticker)
    ticker_match = bool(
        len(ticker_norm) >= 3
        and re.search(rf"(?<![0-9a-z]){re.escape(ticker_norm)}(?![0-9a-z])", normalized)
    )
    return {"full": full, "alias": alias_match, "first_token": first_match, "ticker": ticker_match}


def safe_request(query: str, timeout: float, retries: int) -> tuple[requests.Response | None, str, int]:
    response: requests.Response | None = None
    error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                RSS_URL,
                params={"q": query, "format": "rss", "setlang": "en-US"},
                headers={"User-Agent": "Mozilla/5.0 MARKET_BIO label-blind Bing RSS collector"},
                timeout=timeout,
            )
            if response.status_code == 200:
                return response, "", attempt + 1
            error = f"HTTP_{response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(min(0.5 * (2**attempt), 5.0))
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt < retries:
                time.sleep(min(0.5 * (2**attempt), 5.0))
    return response, error, retries + 1


def fetch(ticker: str, company: str, timeout: float, retries: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query = f'"{company.replace(chr(34), "").strip()}" biotech'
    response, error, attempts = safe_request(query, timeout, retries)
    status: dict[str, Any] = {
        "ticker": ticker, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "attempts": attempts, "status": "ERROR", "error": error,
        "http_status": None if response is None else response.status_code,
        "items": 0, "kept": 0, "outside_interval": 0, "quarantined": 0,
    }
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    if response is None or response.status_code != 200:
        return records, quarantine, status
    if len(response.content) > MAX_RESPONSE_BYTES:
        status.update(status="INVALID", error="RESPONSE_TOO_LARGE")
        return records, quarantine, status
    try:
        items = ET.fromstring(response.content).findall(".//item")
    except ET.ParseError as exc:
        status.update(status="INVALID", error=f"MALFORMED_XML:{str(exc)[:160]}")
        return records, quarantine, status
    status["items"] = len(items)
    for item in items:
        headline = html.unescape(str(item.findtext("title") or "").strip())
        description = html.unescape(str(item.findtext("description") or "").strip())
        raw_link = str(item.findtext("link") or "").strip()
        url = canonical_article_url(raw_link)
        raw_date = str(item.findtext("pubDate") or "").strip()
        published = common.exact_pubdate(raw_date)
        publisher = ""
        for child in item:
            if child.tag.casefold().endswith("source"):
                publisher = html.unescape(str(child.text or "").strip()); break
        reason = ""
        if not headline:
            reason = "HEADLINE_MISSING"
        elif url is None:
            reason = "ARTICLE_URL_INVALID"
        elif published is None:
            reason = "PUBDATE_NOT_EXACT_SECOND"
        elif not (START <= published < END):
            status["outside_interval"] += 1; continue
        evidence = name_evidence(company, ticker, headline + " " + description)
        if not reason and not any(evidence.values()):
            reason = "NO_COMPANY_OR_TICKER_EVIDENCE"
        if reason:
            quarantine.append({
                "reason": reason, "ticker": ticker, "company": company,
                "url": (url or raw_link)[:500], "headline": headline[:500],
            })
            status["quarantined"] += 1
            continue
        article_hash = hashlib.sha256(url.encode()).hexdigest()
        records.append({
            "event_id": f"BINGNEWS:{ticker}:{article_hash}", "market": "US",
            "ticker": ticker, "company": company, "event_time_utc": published.isoformat(),
            "source": SOURCE, "form": "", "headline": headline,
            "body": " | ".join(value for value in (publisher, description) if value),
            "event_type": app.infer_event_type(headline),
            "timestamp_quality": "EXACT_SECOND_BING_RSS_PUBDATE", "url": url,
            "article_id": f"BINGNEWS:{article_hash}", "selection_uses_labels": False,
            "headline_company_exact_evidence": evidence["full"] or evidence["alias"],
            "headline_company_boundary_evidence": any(evidence.values()),
            "query_slice_start_utc": START.isoformat(),
            "query_slice_end_exclusive_utc": END.isoformat(),
        })
        status["kept"] += 1
    status.update(status="OK", error="")
    return records, quarantine, status


def checkpoint_payload(input_info: dict[str, Any], completed: list[str], records: list[dict[str, Any]], quarantine: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contract": CONTRACT, "universe_sha256": input_info["sha256"],
        "completed_tickers": completed, "records": records, "quarantine": quarantine,
        "statuses": statuses, "selection_uses_labels": False,
    }


def load_checkpoint(input_info: dict[str, Any]) -> dict[str, Any] | None:
    if not CHECKPOINT.is_file():
        return None
    try:
        value = json.loads(gzip.decompress(CHECKPOINT.read_bytes()).decode("utf-8"))
    except Exception:
        return None
    return value if (
        value.get("contract") == CONTRACT
        and value.get("universe_sha256") == input_info["sha256"]
        and value.get("selection_uses_labels") is False
    ) else None


def write_outputs(frame: pd.DataFrame, quarantine: list[dict[str, Any]], statuses: list[dict[str, Any]], audit: dict[str, Any], input_info: dict[str, Any]) -> dict[str, Any]:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    content_sha = hashlib.sha256(csv_bytes).hexdigest()
    snapshot = POOL / f"gnews_us_2025-08-20_2026-08-27_bing_{content_sha[:12]}.csv.gz"
    common.atomic_gzip(snapshot, csv_bytes)
    quarantine_bytes = common.canonical_json_bytes({"contract": CONTRACT, "rows": quarantine, "selection_uses_labels": False})
    quarantine_sha = hashlib.sha256(quarantine_bytes).hexdigest()
    quarantine_path = POOL / f"gnews_us_bing_quarantine_{quarantine_sha[:12]}.json.gz"
    common.atomic_gzip(quarantine_path, quarantine_bytes)
    audit_payload = {
        "contract": CONTRACT, "source": SOURCE, "selection_uses_labels": False,
        "authorized_input": input_info, "forbidden_inputs_read": [],
        "interval": {"start_inclusive_utc": START.isoformat(), "end_exclusive_utc": END.isoformat()},
        "collection": audit,
        "snapshot": {"path": str(snapshot.relative_to(ROOT)), "rows": len(frame), "file_sha256": common.sha256(snapshot), "content_sha256_uncompressed_csv": content_sha},
        "quarantine": {"path": str(quarantine_path.relative_to(ROOT)), "rows": len(quarantine), "file_sha256": common.sha256(quarantine_path)},
        "request_failures": [item for item in statuses if item.get("status") != "OK"],
        "complete_interval_claimed": False,
    }
    audit_bytes = common.canonical_json_bytes(audit_payload)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    audit_path = POOL / f"gnews_us_bing_audit_{audit_sha[:12]}.json"
    common.atomic_bytes(audit_path, audit_bytes)
    result = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "BEST_EFFORT_COMPLETE",
        "contract": CONTRACT, "selection_uses_labels": False,
        "snapshot": audit_payload["snapshot"], "audit_path": str(audit_path.relative_to(ROOT)),
        "audit_sha256": common.sha256(audit_path), "quarantine": audit_payload["quarantine"],
        "collection": audit, "downstream_queue_not_modified": True,
    }
    common.atomic_bytes(STATUS, json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("workers/timeout must be positive and retries non-negative")
    universe, input_info = input_audit()
    if args.audit_only:
        print(json.dumps({
            "status": "AUDIT_OK", "contract": CONTRACT, "authorized_input": input_info,
            "base_request_estimate": len(universe), "selection_uses_labels": False,
            "forbidden_inputs_read": [], "full_collection_executed": False,
        }, indent=2)); return
    checkpoint = load_checkpoint(input_info) if not args.no_resume else None
    records = list(checkpoint.get("records", [])) if checkpoint else []
    quarantine = list(checkpoint.get("quarantine", [])) if checkpoint else []
    statuses = list(checkpoint.get("statuses", [])) if checkpoint else []
    completed = list(checkpoint.get("completed_tickers", [])) if checkpoint else []
    completed_set = set(completed)
    pending = universe[~universe.ticker.astype(str).isin(completed_set)]
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as pool:
        futures = {
            pool.submit(fetch, str(row.ticker), str(row.company), args.timeout, args.retries): str(row.ticker)
            for row in pending.itertuples(index=False)
        }
        for done, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                rows, rejected, status = future.result()
            except Exception as exc:
                rows, rejected, status = [], [], {"ticker": ticker, "status": "UNEXPECTED_ERROR", "error": f"{type(exc).__name__}:{str(exc)[:180]}", "attempts": 0, "kept": 0}
            records.extend(rows); quarantine.extend(rejected); statuses.append(status)
            completed.append(ticker); completed_set.add(ticker)
            if done % 100 == 0 or done == len(futures):
                common.atomic_gzip(CHECKPOINT, common.canonical_json_bytes(checkpoint_payload(input_info, completed, records, quarantine, statuses)))
                print(f"[US BING RSS] tasks={done}/{len(futures)} raw_rows={len(records)}", flush=True)
    frame, quarantine, dedup = common.resolve_articles(records, quarantine)
    audit = {
        "checkpoint_resumed": bool(checkpoint), "completed_tickers": len(completed_set),
        "configured_tickers": len(universe), "http_requests": sum(int(item.get("attempts", 0)) for item in statuses),
        "request_failures": sum(item.get("status") != "OK" for item in statuses),
        "items": sum(int(item.get("items", 0)) for item in statuses),
        "pre_dedup_kept": sum(int(item.get("kept", 0)) for item in statuses),
        "deduplication": dedup,
    }
    print(json.dumps(write_outputs(frame, quarantine, statuses, audit, input_info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
