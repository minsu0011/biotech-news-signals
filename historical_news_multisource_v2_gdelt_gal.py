from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests

from historical_news_multisource_v2_local_recovery import lexical_similarity, publisher_class
from historical_news_multisource_v2_body_recovery import english_distinctive_anchor_consistent


ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
POLICY = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
RESEARCH = ROOT / "research" / "multisource_v2"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
CURRENT = WORKING / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
OUTPUT = WORKING / "V2_DISCOVERED_GDELT_GAL_ITEMS.parquet"
RAW = WORKING / "raw" / "gdelt_gal"
LEDGER = RESEARCH / "GDELT_GAL_COLLECTION_LEDGER.jsonl"
REPORT = RESEARCH / "GDELT_GAL_DISCOVERY_REPORT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

USER_AGENT = "MARKET_BIO_HIST_NEWS_RESEARCH/2.0 (public GDELT GAL; contact: local-research)"
BASE_URL = "https://data.gdeltproject.org/gdeltv3/gal"
MIN_HOST_INTERVAL_SECONDS = 1.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *values: Any) -> str:
    material = json.dumps([str(value or "") for value in values], ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def gal_bucket_times(
    event_time: Any,
    before_minutes: int = 15,
    after_minutes: int = 90,
) -> list[pd.Timestamp]:
    """Return GDELT GAL quarter-hour boundaries (:01, :16, :31, :46 UTC)."""
    event = pd.to_datetime(event_time, utc=True, errors="raise")
    start = event - pd.Timedelta(minutes=before_minutes)
    end = event + pd.Timedelta(minutes=after_minutes)
    anchor = start.floor("h") + pd.Timedelta(minutes=1)
    if anchor < start:
        steps = math.ceil((start - anchor).total_seconds() / 900.0)
        anchor += pd.Timedelta(minutes=15 * steps)
    buckets = []
    current = anchor
    while current <= end:
        buckets.append(current)
        current += pd.Timedelta(minutes=15)
    return buckets


def bucket_name(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y%m%d%H%M%S") + ".gal.json.gz"


def parse_gal_payload(payload: bytes) -> list[dict[str, Any]]:
    text = gzip.decompress(payload).decode("utf-8")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if str(value.get("url") or "").startswith(("http://", "https://")) and value.get("title"):
            rows.append(value)
    return rows


def gal_item_is_semantic_candidate(
    current_headline: Any,
    candidate_headline: Any,
    candidate_context: Any = "",
) -> tuple[bool, float, float]:
    sequence, jaccard = lexical_similarity(current_headline, candidate_headline)
    anchor_consistent = english_distinctive_anchor_consistent(
        current_headline, f"{candidate_headline} {candidate_context}"
    )[0]
    return bool(sequence >= 0.40 and jaccard >= 0.15 and anchor_consistent), sequence, jaccard


class GalClient:
    def __init__(self, max_requests: int):
        self.max_requests = max_requests
        self.network_requests = 0
        self.cache_hits = 0
        self.last_request = 0.0
        self.robots_status = "NOT_CHECKED"
        self.robots_allowed = False
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/octet-stream"})

    def _wait(self) -> None:
        remaining = MIN_HOST_INTERVAL_SECONDS - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)

    def check_robots(self) -> bool:
        if self.robots_status != "NOT_CHECKED":
            return self.robots_allowed
        try:
            response = self.session.get(f"{BASE_URL.rsplit('/', 2)[0]}/robots.txt", timeout=(10, 30))
            self.network_requests += 1
            self.last_request = time.monotonic()
        except requests.RequestException:
            self.robots_status = "ROBOTS_FETCH_FAILED"
            return False
        if response.status_code in {404, 410}:
            self.robots_allowed = True
            self.robots_status = "NO_DECLARED_RESTRICTION"
            return True
        if response.status_code != 200:
            self.robots_status = f"ROBOTS_HTTP_{response.status_code}"
            return False
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        sample = f"{BASE_URL}/20200101000100.gal.json.gz"
        self.robots_allowed = parser.can_fetch(USER_AGENT, sample)
        self.robots_status = "ALLOWED" if self.robots_allowed else "ROBOTS_DISALLOW"
        return self.robots_allowed

    def fetch_bucket(self, name: str) -> tuple[list[dict[str, Any]], str, str]:
        path = RAW / name
        if path.is_file():
            payload = path.read_bytes()
            self.cache_hits += 1
            try:
                return parse_gal_payload(payload), sha256_bytes(payload), "CACHE_HIT"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return [], sha256_bytes(payload), "CACHE_PARSE_FAILED"
        if self.network_requests >= self.max_requests:
            return [], "", "REQUEST_BUDGET_EXHAUSTED"
        if not self.check_robots():
            return [], "", self.robots_status
        if self.network_requests >= self.max_requests:
            return [], "", "REQUEST_BUDGET_EXHAUSTED"
        self._wait()
        try:
            response = self.session.get(f"{BASE_URL}/{name}", timeout=(10, 30))
            self.network_requests += 1
            self.last_request = time.monotonic()
        except requests.RequestException:
            return [], "", "REQUEST_FAILED"
        if response.status_code == 404:
            return [], "", "NOT_FOUND"
        if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
            return [], "", f"HTTP_{response.status_code}"
        payload = response.content
        digest = sha256_bytes(payload)
        try:
            rows = parse_gal_payload(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return [], digest, "PARSE_FAILED"
        RAW.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(payload)
        temp.replace(path)
        return rows, digest, "SUCCESS"


def choose_events(limit: int, collection_window_id: str) -> pd.DataFrame:
    queue = pd.read_parquet(QUEUE)
    queue = queue.loc[queue["market"].astype(str).eq("US")].copy()
    current = pd.read_parquet(CURRENT)
    frame = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    completed = {
        str(row.get("canonical_event_id")) for row in read_jsonl(LEDGER)
        if str(row.get("status")) in {"SUCCESS", "NO_MATCH"}
        and str(row.get("collection_window_id") or "B15_A90") == collection_window_id
    }
    frame = frame.loc[
        ~frame["canonical_event_id"].astype(str).isin(completed)
        & frame["origin_publisher_resolved"].astype(str).ne("UNKNOWN")
    ].copy()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True, errors="raise")
    return frame.sort_values(
        ["event_time_utc", "canonical_event_id"], ascending=[False, True], kind="mergesort"
    ).head(limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--before-minutes", type=int, default=15)
    parser.add_argument("--after-minutes", type=int, default=90)
    args = parser.parse_args()
    if args.events < 1 or args.max_requests < 2 or args.before_minutes < 0 or args.after_minutes < 0:
        raise SystemExit("invalid collection bounds")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("outcome-blind collection policy guard failed")

    collection_window_id = f"B{args.before_minutes}_A{args.after_minutes}"
    events = choose_events(args.events, collection_window_id)
    client = GalClient(args.max_requests)
    bucket_cache: dict[str, tuple[list[dict[str, Any]], str, str]] = {}
    discoveries: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_time = pd.Timestamp(event.event_time_utc)
        statuses = []
        event_candidates = 0
        for stamp in gal_bucket_times(event_time, args.before_minutes, args.after_minutes):
            name = bucket_name(stamp)
            if name not in bucket_cache:
                bucket_cache[name] = client.fetch_bucket(name)
            items, raw_sha, status = bucket_cache[name]
            statuses.append(status)
            for item in items:
                accepted, sequence, jaccard = gal_item_is_semantic_candidate(
                    event.current_headline, item.get("title", ""), item.get("desc", "")
                )
                if not accepted:
                    continue
                url = str(item.get("url") or "")
                if url == str(getattr(event, "canonical_url", "") or ""):
                    continue
                origin = str(item.get("outletName") or item.get("domain") or "UNKNOWN").strip()
                distinct = bool(origin and origin != "UNKNOWN" and origin.casefold() != str(event.origin_publisher_resolved).casefold())
                if not distinct:
                    continue
                seen = pd.to_datetime(item.get("date"), utc=True, errors="coerce")
                discovery_uid = stable_id("GAL_", event.canonical_event_id, url, item.get("title", ""))
                discoveries.append({
                    "schema_version": "HIST_NEWS_MULTISOURCE_V2_GDELT_GAL_DISCOVERY_V1",
                    "canonical_event_id": str(event.canonical_event_id),
                    "market": str(event.market),
                    "ticker": str(event.ticker),
                    "event_time_utc": event_time,
                    "decision_cutoff_utc": event_time + pd.Timedelta(minutes=2),
                    "current_article_uid": str(event.current_article_uid),
                    "current_origin_publisher": str(event.origin_publisher_resolved),
                    "current_headline": str(event.current_headline),
                    "discovery_provider": "GDELT_GAL_PUBLIC_BACKFILE",
                    "discovery_query": "QUARTER_HOUR_EVENT_WINDOW_HEADLINE_MATCH",
                    "headline": str(item.get("title") or ""),
                    "description": str(item.get("desc") or ""),
                    "direct_url_candidate": url,
                    "publisher_candidate": origin,
                    "hosting_domain": str(item.get("domain") or (urlsplit(url).hostname or "")),
                    "gal_seen_at_utc": seen,
                    "gal_bucket_utc": stamp,
                    "timestamp_provenance": "GDELT_GAL_SEEN_TIME_DISCOVERY_ONLY",
                    "timestamp_confidence": "LOW_NOT_PUBLICATION_TIME",
                    "headline_sequence_ratio": sequence,
                    "headline_char3_jaccard": jaccard,
                    "origin_distinct_from_current": distinct,
                    "publisher_class": publisher_class(origin),
                    "body_recovery_required": True,
                    "strict_independent_root_accepted": False,
                    "outcome_fields_read": "",
                    "raw_gal_sha256": raw_sha,
                    "discovery_uid": discovery_uid,
                })
                event_candidates += 1
        partial = any(value in {"REQUEST_BUDGET_EXHAUSTED", "REQUEST_FAILED", "ROBOTS_FETCH_FAILED"} for value in statuses)
        ledger_rows.append({
            "canonical_event_id": str(event.canonical_event_id),
            "market": "US",
            "provider": "GDELT_GAL_PUBLIC_BACKFILE",
            "collection_window_id": collection_window_id,
            "status": "PARTIAL" if partial else ("SUCCESS" if event_candidates else "NO_MATCH"),
            "bucket_status_counts": {
                str(key): int(value)
                for key, value in pd.Series(statuses, dtype="object").value_counts().items()
            },
            "accepted_discovery_items": int(event_candidates),
            "attempted_at_utc": utc_now(),
            "outcome_fields_read": [],
        })
        if client.network_requests >= args.max_requests:
            break

    append_jsonl(LEDGER, ledger_rows)
    new = pd.DataFrame(discoveries)
    prior_ids: set[str] = set()
    if OUTPUT.is_file():
        old = pd.read_parquet(OUTPUT)
        prior_ids = set(old["discovery_uid"].astype(str))
        combined = pd.concat([old, new], ignore_index=True, sort=False) if not new.empty else old
    else:
        combined = new
    if not combined.empty:
        combined = combined.sort_values(
            ["canonical_event_id", "gal_seen_at_utc", "discovery_uid"], kind="mergesort"
        ).drop_duplicates("discovery_uid", keep="first").reset_index(drop=True)
        atomic_parquet(OUTPUT, combined)
    new_ids = set(new.get("discovery_uid", pd.Series(dtype="object")).astype(str))
    new_unique = int(len(new_ids - prior_ids))
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_GDELT_GAL_DISCOVERY_REPORT_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "official_documentation": "https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/",
        "events_selected": int(len(events)),
        "collection_window_id": collection_window_id,
        "events_completed_this_run": int(sum(row["status"] in {"SUCCESS", "NO_MATCH"} for row in ledger_rows)),
        "unique_buckets_considered": int(len(bucket_cache)),
        "network_requests_including_robots": int(client.network_requests),
        "cache_hits": int(client.cache_hits),
        "robots_status": client.robots_status,
        "new_discovery_items": int(len(new)),
        "new_discovery_items_unique": new_unique,
        "total_discovery_items": int(len(combined)) if not combined.empty else 0,
        "gal_seen_time_used_as_publication_time": False,
        "origin_page_timestamp_required_for_causal_root": True,
        "source_freeze_ready": False,
        "model_training_allowed": False,
        "base_v1_mutated": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {OUTPUT.name: sha256_file(OUTPUT)} if OUTPUT.is_file() else {},
    }
    atomic_json(REPORT, report)
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state.update({
            "updated_at_utc": utc_now(),
            "gdelt_gal_discovery_items": report["total_discovery_items"],
            "source_frozen": False,
            "model_training_allowed": False,
        })
        atomic_json(STATE, state)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
