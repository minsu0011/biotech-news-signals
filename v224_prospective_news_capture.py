"""One-shot, resume-safe causal news capture for V224 Track B.

This collector writes only its dedicated append-only namespace plus
state/NEWS_PROSPECTIVE_CAPTURE_STATE.json.  It does not read or modify labels,
prices, central data roles, Research Seal, Final Meta, or model state.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import html
import io
import json
import os
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
NAMESPACE = ROOT / "prospective_news_capture" / "V224_NEWS"
V1_POLICY_PATH = NAMESPACE / "FROZEN_QUERY_POLICY.json"
V1_POLICY_SHA256 = "0da9e42e8476ce930256d53ccc60def84bbace8a1583ea175fd6b89b80a85c06"
POLICY_PATH = NAMESPACE / "FROZEN_QUERY_POLICY_V2.json"
SCHEMA_PATH = NAMESPACE / "NEWS_PROSPECTIVE_CAPTURE_SCHEMA_V2.json"
AMENDMENT_PATH = NAMESPACE / "V2_CAUSALITY_AMENDMENT.json"
STATE_PATH = ROOT / "state" / "NEWS_PROSPECTIVE_CAPTURE_STATE.json"
LOCK_PATH = NAMESPACE / ".capture.lock"
RAW_LEDGER = NAMESPACE / "ledger" / "raw_capture_ledger.jsonl"
NEWS_LEDGER = NAMESPACE / "ledger" / "news_records.jsonl"
OBSERVATION_LEDGER = NAMESPACE / "ledger" / "capture_observations.jsonl"
QUARANTINE_LEDGER = NAMESPACE / "ledger" / "quarantine.jsonl"
MIGRATION_LEDGER = NAMESPACE / "ledger" / "state_migration_history.jsonl"
PROVIDER_ENDPOINT = "https://news.google.com/rss/search"
USER_AGENT = "MARKET_BIO-V224-causal-news-capture/1.0 (public RSS; one-shot)"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_QUERIES = 6
DEFAULT_MIN_INTERVAL_SECONDS = 1.25
BROAD_SUCCESS_CADENCE_SECONDS = 90
ISSUER_SUCCESS_CADENCE_SECONDS = 8 * 60 * 60
CIRCUIT_BREAKER_MIN_ATTEMPTS = 4
CIRCUIT_BREAKER_FAILURE_RATIO = 0.75
CIRCUIT_BREAKER_CONSECUTIVE_FAILURES = 5
CIRCUIT_BREAKER_COOLDOWN_MINUTES = 30
DETERMINISTIC_FAILURE_QUARANTINE_COUNT = 3
SCHEMA_VERSION = 2
POLICY_VERSION = 2
TRACK = "V224_HIGH_CONFIDENCE_NEWS_FUTURE_DATA"

FORBIDDEN_FIELDS = {
    "y", "up_down", "future_return", "fwd_ret_30m", "correct", "correctness",
    "strategy_net", "model_correctness", "price", "entry_price", "exit_price",
}
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
DETERMINISTIC_HTTP = {400, 401, 403, 404, 405, 406, 409, 410, 415, 422}

TOPICS = {
    "CLINICAL_TRIAL": r"\bclinical\b|\btrial\b|\bphase\s*[123iIvV]+\b|임상|시험",
    "FDA_REGULATORY": r"\bfda\b|approval|regulator|regulatory|식약처|허가|승인",
    "FINANCING": r"financing|offering|convertible|warrant|증자|사채|자금조달",
    "M_AND_A": r"merger|acquisition|takeover|인수|합병",
    "LICENSING": r"licens|partnership|collaboration|기술이전|라이선스|협업",
    "EARNINGS": r"earnings|revenue|guidance|results|실적|매출|영업이익",
    "SAFETY": r"safety|adverse|recall|clinical hold|안전성|부작용|리콜",
    "MANUFACTURING": r"manufactur|facility|cGMP|생산|공장|제조",
    "PATENT": r"patent|intellectual property|특허",
    "MANAGEMENT": r"appoint|resign|chief executive|management|대표이사|사임|선임",
}

PUBLISHER_CLASSES = {
    "WIRE_SERVICE": ("reuters", "associated press", "ap news", "yonhap", "연합뉴스"),
    "FINANCIAL_PRESS": ("financial times", "wall street journal", "wsj", "cnbc", "marketwatch", "barron's"),
    "GENERAL_MAJOR_PRESS": ("new york times", "washington post", "bbc", "cnn", "nbc", "abc news", "kbs", "mbc", "sbs"),
    "BIOTECH_SPECIALIST": ("biospace", "fierce biotech", "endpoints news", "stat news", "바이오스펙테이터", "메디게이트"),
    "PR_DISTRIBUTION": ("globenewswire", "business wire", "pr newswire", "newsfile", "accesswire", "뉴스와이어"),
    "AGGREGATOR": ("google news", "yahoo news", "msn", "newsbreak", "네이버 뉴스", "다음 뉴스"),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("NAIVE_TIMESTAMP_FORBIDDEN")
    return parsed.astimezone(timezone.utc)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, default=str) + "\n").encode("utf-8")


def deterministic_gzip(value: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=target, compresslevel=9, mtime=0) as handle:
        handle.write(value)
    return target.getvalue()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                  allow_nan=False, default=str).encode("utf-8") + b"\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"CORRUPT_APPEND_ONLY_LEDGER:{path}:{number}") from exc
    return rows


class CaptureLock(AbstractContextManager["CaptureLock"]):
    def __init__(self, path: Path, resume: bool, now: datetime):
        self.path = path
        self.resume = resume
        self.now = now
        self.acquired = False

    def __enter__(self) -> "CaptureLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes({"pid": os.getpid(), "acquired_at_utc": iso_utc(self.now)})
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not self.resume:
                raise RuntimeError("NEWS_CAPTURE_LOCK_EXISTS_USE_RESUME")
            try:
                lock = json.loads(self.path.read_text(encoding="utf-8"))
                age = self.now - parse_utc(lock["acquired_at_utc"])
            except Exception as exc:
                raise RuntimeError("NEWS_CAPTURE_LOCK_CORRUPT") from exc
            if age < timedelta(hours=2):
                raise RuntimeError("NEWS_CAPTURE_ALREADY_RUNNING")
            stale = self.path.with_name(f"stale_lock_{sha256_bytes(self.path.read_bytes())}.json")
            if not stale.exists():
                os.replace(self.path, stale)
            else:
                self.path.unlink(missing_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(value or ""))).casefold()
    return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", value).split())


def clean_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", str(value or ""))
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def publisher_class(publisher: str, publisher_url: str, issuer_company: str | None = None) -> str:
    value = normalize_text(f"{publisher} {urlparse(publisher_url).netloc}")
    issuer_alias = remove_company_suffix(issuer_company or "")
    if issuer_alias and issuer_alias in value:
        return "ISSUER_PRESS_RELEASE"
    for name, needles in PUBLISHER_CLASSES.items():
        if any(normalize_text(needle) in value for needle in needles):
            return name
    return "OTHER"


def topic_id(text: str) -> str:
    for name, pattern in TOPICS.items():
        if re.search(pattern, text, re.I):
            return name
    return "OTHER"


def remove_company_suffix(company: str) -> str:
    value = normalize_text(company)
    suffixes = {
        "inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited",
        "plc", "holdings", "holding", "therapeutics", "pharmaceuticals", "pharmaceutical",
        "biotechnology", "biotech", "group", "주식회사",
    }
    pieces = value.split()
    while pieces and pieces[-1] in suffixes:
        pieces.pop()
    return " ".join(pieces)


def load_policy(policy_path: Path = POLICY_PATH) -> tuple[dict[str, Any], str]:
    payload = policy_path.read_bytes()
    policy = json.loads(payload)
    if (policy.get("track") != TRACK or policy.get("provider_endpoint") != PROVIDER_ENDPOINT
            or int(policy.get("policy_version", 0)) != POLICY_VERSION):
        raise RuntimeError("NEWS_CAPTURE_POLICY_IDENTITY_MISMATCH")
    v1_path = policy_path.with_name("FROZEN_QUERY_POLICY.json")
    if sha256_file(v1_path) != V1_POLICY_SHA256:
        raise RuntimeError("NEWS_CAPTURE_V1_POLICY_BYTES_CHANGED")
    if policy.get("supersedes", {}).get("sha256") != V1_POLICY_SHA256:
        raise RuntimeError("NEWS_CAPTURE_V2_SUPERSESSION_CHAIN_MISMATCH")
    if int(policy.get("broad_query_success_cadence_seconds", 0)) != BROAD_SUCCESS_CADENCE_SECONDS:
        raise RuntimeError("NEWS_CAPTURE_V2_BROAD_CADENCE_MISMATCH")
    if int(policy.get("issuer_query_success_cadence_seconds", 0)) != ISSUER_SUCCESS_CADENCE_SECONDS:
        raise RuntimeError("NEWS_CAPTURE_V2_ISSUER_CADENCE_MISMATCH")
    return policy, sha256_bytes(payload)


def verify_v1_immutable_evidence(namespace: Path = NAMESPACE) -> None:
    if sha256_file(V1_POLICY_PATH) != V1_POLICY_SHA256:
        raise RuntimeError("NEWS_CAPTURE_V1_POLICY_BYTES_CHANGED")
    # The amendment freezes the two V1 smoke raw objects.  Enforce them in the
    # live namespace; isolated test namespaces intentionally have no V1 raw.
    if namespace.resolve() != NAMESPACE.resolve():
        return
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    evidence = amendment["v1_smoke_evidence"]
    if int(evidence["records"]) != 62 or int(evidence["causal_t_plus_2_eligible"]) != 0:
        raise RuntimeError("NEWS_CAPTURE_V1_AMENDMENT_EVIDENCE_MISMATCH")
    for record in evidence["immutable_raw_files"]:
        path = namespace / record["path"]
        if sha256_file(path) != record["gzip_sha256"]:
            raise RuntimeError(f"NEWS_CAPTURE_V1_RAW_BYTES_CHANGED:{record['path']}")
        if sha256_bytes(gzip.decompress(path.read_bytes())) != record["raw_response_sha256"]:
            raise RuntimeError(f"NEWS_CAPTURE_V1_RAW_RESPONSE_CHANGED:{record['path']}")


def load_universes(policy: dict[str, Any], root: Path = ROOT) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    aliases: list[dict[str, str]] = []
    for market in ("US", "KR"):
        contract = policy["universes"][market]
        path = root / contract["path"]
        if sha256_file(path) != contract["sha256"]:
            raise RuntimeError(f"NEWS_CAPTURE_UNIVERSE_HASH_MISMATCH:{market}")
        import csv
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if len(rows) != int(contract["rows"]):
            raise RuntimeError(f"NEWS_CAPTURE_UNIVERSE_ROW_MISMATCH:{market}")
        cleaned = []
        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            company = str(row.get("company", "")).strip()
            if not ticker or not company:
                raise RuntimeError(f"NEWS_CAPTURE_UNIVERSE_EMPTY_IDENTITY:{market}")
            cleaned.append({"market": market, "ticker": ticker, "company": company})
            aliases.append({
                "market": market, "ticker": ticker, "company": company,
                "alias": remove_company_suffix(company), "full_alias": normalize_text(company),
            })
        output[market] = sorted(cleaned, key=lambda row: row["ticker"])
    return output, aliases


def query_hash(query: str) -> str:
    return sha256_bytes(query.encode("utf-8"))


def make_queue(policy: dict[str, Any], universes: dict[str, list[dict[str, str]]], now: datetime) -> list[dict[str, Any]]:
    due = iso_utc(now)
    queue = []
    for item in policy["broad_queries"]:
        queue.append({
            "query_id": item["query_id"], "kind": "BROAD", "market": item["market"],
            "ticker": None, "company": None, "query": item["query"], "query_sha256": query_hash(item["query"]),
            "cadence_seconds": int(policy["broad_query_success_cadence_seconds"]),
            "status": "READY", "next_due_at_utc": due, "attempts": 0,
            "consecutive_failures": 0, "deterministic_failure_count": 0,
            "deterministic_failure_fingerprint": None, "last_attempt_at_utc": None,
            "last_success_at_utc": None, "last_error_code": None,
        })
    for market in ("US", "KR"):
        template = policy["issuer_query_templates"][market]
        for issuer in universes[market]:
            query = template.format(company=issuer["company"].replace('"', ""))
            queue.append({
                "query_id": f"ISSUER_{market}_{issuer['ticker']}", "kind": "ISSUER", "market": market,
                "ticker": issuer["ticker"], "company": issuer["company"], "query": query,
                "query_sha256": query_hash(query), "status": "READY", "next_due_at_utc": due,
                "cadence_seconds": int(policy["issuer_query_success_cadence_seconds"]),
                "attempts": 0, "consecutive_failures": 0, "deterministic_failure_count": 0,
                "deterministic_failure_fingerprint": None, "last_attempt_at_utc": None,
                "last_success_at_utc": None, "last_error_code": None,
            })
    return queue


def initial_state(policy_sha256: str, queue: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "track": TRACK, "status": "READY",
        "query_policy_version": POLICY_VERSION,
        "query_policy_sha256": policy_sha256, "created_at_utc": iso_utc(now),
        "updated_at_utc": iso_utc(now), "next_recommended_run_utc": iso_utc(now),
        "queue": queue,
        "circuit_breaker": {"status": "CLOSED", "opened_at_utc": None, "open_until_utc": None,
                            "reason": None, "consecutive_failures": 0},
        "totals": {"runs": 0, "requests": 0, "successful_requests": 0, "failed_requests": 0,
                   "raw_captures": 0, "items_observed": 0, "new_articles": 0,
                   "t_plus_2_usable_articles": 0, "historical_only_articles": 0,
                   "quarantined_queries": 0},
        "last_run": None,
        "migration_history_path": "prospective_news_capture/V224_NEWS/ledger/state_migration_history.jsonl",
        "selection_uses_labels": False, "selection_uses_prices": False,
        "research_seal_or_final_rows_read": False,
    }


def _queue_attempt_digest(queue: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes({item["query_id"]: int(item["attempts"]) for item in queue}))


def migrate_v1_state(state: dict[str, Any], policy_sha256: str, queue: list[dict[str, Any]],
                     now: datetime, migration_ledger: Path) -> dict[str, Any]:
    if state.get("query_policy_sha256") != V1_POLICY_SHA256:
        raise RuntimeError("NEWS_CAPTURE_UNAUTHORIZED_STATE_POLICY_MIGRATION")
    before = copy.deepcopy(state)
    old_by_id = {item["query_id"]: item for item in state["queue"]}
    mutable = (
        "status", "next_due_at_utc", "attempts", "consecutive_failures",
        "deterministic_failure_count", "deterministic_failure_fingerprint",
        "last_attempt_at_utc", "last_success_at_utc", "last_error_code",
    )
    migration_deadline = now + timedelta(seconds=BROAD_SUCCESS_CADENCE_SECONDS)
    for item in queue:
        old = old_by_id.get(item["query_id"])
        if old is None:
            continue
        for key in mutable:
            item[key] = copy.deepcopy(old.get(key, item.get(key)))
        if item["kind"] == "BROAD" and parse_utc(item["next_due_at_utc"]) > migration_deadline:
            item["next_due_at_utc"] = iso_utc(migration_deadline)
    if set(old_by_id) != {item["query_id"] for item in queue}:
        raise RuntimeError("NEWS_CAPTURE_V1_V2_QUEUE_ID_MISMATCH")
    state["schema_version"] = SCHEMA_VERSION
    state["query_policy_version"] = POLICY_VERSION
    state["query_policy_sha256"] = policy_sha256
    state["queue"] = queue
    state["updated_at_utc"] = iso_utc(now)
    state["next_recommended_run_utc"] = iso_utc(min(
        parse_utc(item["next_due_at_utc"]) for item in queue if item["kind"] == "BROAD"
    ))
    state["migration_history_path"] = "prospective_news_capture/V224_NEWS/ledger/state_migration_history.jsonl"
    if state["totals"] != before["totals"]:
        raise RuntimeError("NEWS_CAPTURE_MIGRATION_TOTALS_CHANGED")
    if _queue_attempt_digest(state["queue"]) != _queue_attempt_digest(before["queue"]):
        raise RuntimeError("NEWS_CAPTURE_MIGRATION_ATTEMPTS_CHANGED")
    migration_id = "MIG:" + sha256_bytes(f"{V1_POLICY_SHA256}|{policy_sha256}".encode("utf-8"))
    if not any(row.get("migration_id") == migration_id for row in load_jsonl(migration_ledger)):
        append_jsonl(migration_ledger, [{
            "schema_version": SCHEMA_VERSION, "track": TRACK, "migration_id": migration_id,
            "migrated_at_utc": iso_utc(now), "from_policy_version": 1,
            "from_policy_sha256": V1_POLICY_SHA256, "to_policy_version": POLICY_VERSION,
            "to_policy_sha256": policy_sha256, "queue_rows": len(queue),
            "totals_before_sha256": sha256_bytes(canonical_json_bytes(before["totals"])),
            "totals_after_sha256": sha256_bytes(canonical_json_bytes(state["totals"])),
            "attempts_before_sha256": _queue_attempt_digest(before["queue"]),
            "attempts_after_sha256": _queue_attempt_digest(state["queue"]),
            "broad_due_no_later_than_utc": iso_utc(migration_deadline),
            "append_only_ledgers_or_raw_modified_by_migration": False,
            "outcome_price_seal_final_inputs_read": False,
        }])
    return state


def load_or_initialize_state(state_path: Path, policy_sha256: str, queue: list[dict[str, Any]],
                             now: datetime, resume: bool, migration_ledger: Path) -> dict[str, Any]:
    if state_path.exists():
        if not resume:
            raise RuntimeError("NEWS_CAPTURE_STATE_EXISTS_USE_RESUME")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("track") != TRACK:
            raise RuntimeError("NEWS_CAPTURE_STATE_IDENTITY_MISMATCH")
        if state.get("query_policy_sha256") == V1_POLICY_SHA256:
            state = migrate_v1_state(state, policy_sha256, queue, now, migration_ledger)
        elif (state.get("schema_version") != SCHEMA_VERSION
              or int(state.get("query_policy_version", 0)) != POLICY_VERSION
              or state.get("query_policy_sha256") != policy_sha256):
            raise RuntimeError("NEWS_CAPTURE_FROZEN_QUERY_POLICY_CHANGED")
        return state
    return initial_state(policy_sha256, queue, now)


def build_request_url(task: dict[str, Any], policy: dict[str, Any]) -> str:
    locale = policy["universes"][task["market"]]["locale"]
    params = {"q": task["query"], **locale}
    return PROVIDER_ENDPOINT + "?" + urllib.parse.urlencode(params)


def http_fetch(task: dict[str, Any], policy: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    url = build_request_url(task, policy)
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml;q=0.9",
        "Accept-Encoding": "identity",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                return {"ok": False, "status": int(response.status), "body": body[:MAX_RESPONSE_BYTES],
                        "headers": dict(response.headers.items()), "error_code": "RESPONSE_TOO_LARGE",
                        "deterministic": True, "request_url": url}
            return {"ok": int(response.status) == 200, "status": int(response.status), "body": body,
                    "headers": dict(response.headers.items()), "error_code": None, "deterministic": False,
                    "request_url": url}
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_RESPONSE_BYTES + 1)
        status = int(exc.code)
        return {"ok": False, "status": status, "body": body[:MAX_RESPONSE_BYTES],
                "headers": dict(exc.headers.items()), "error_code": f"HTTP_{status}",
                "deterministic": status in DETERMINISTIC_HTTP, "request_url": url}
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return {"ok": False, "status": None, "body": b"", "headers": {},
                "error_code": type(exc).__name__.upper(), "deterministic": False, "request_url": url}


def persist_raw(namespace: Path, task: dict[str, Any], fetched: dict[str, Any], captured_at: datetime) -> dict[str, Any]:
    body = bytes(fetched.get("body") or b"")
    raw_sha = sha256_bytes(body)
    gzip_bytes = deterministic_gzip(body)
    relative = Path("raw") / f"{captured_at:%Y}" / f"{captured_at:%m}" / f"{captured_at:%d}" / f"{raw_sha}.xml.gz"
    path = namespace / relative
    if path.exists():
        if sha256_file(path) != sha256_bytes(gzip_bytes):
            raise RuntimeError("NEWS_CAPTURE_CONTENT_ADDRESS_COLLISION")
    else:
        atomic_bytes(path, gzip_bytes)
    capture_id = "CAP:" + sha256_bytes(f"{task['query_id']}|{iso_utc(captured_at)}|{raw_sha}".encode("utf-8"))
    return {
        "capture_id": capture_id, "raw_capture_path": str(relative).replace("\\", "/"),
        "raw_capture_sha256": raw_sha, "raw_gzip_sha256": sha256_bytes(gzip_bytes),
        "raw_bytes": len(body), "gzip_bytes": len(gzip_bytes),
    }


def exact_pubdate(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def canonical_publisher(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", html.unescape(str(value or ""))).split())


def strip_publisher_suffix(headline: str, publisher: str) -> str:
    title = " ".join(html.unescape(str(headline or "")).split())
    suffix = " - " + publisher
    return title[:-len(suffix)].rstrip() if publisher and title.casefold().endswith(suffix.casefold()) else title


def map_tickers(text: str, task: dict[str, Any], aliases: list[dict[str, str]]) -> tuple[list[str], str]:
    normalized = normalize_text(text)
    matches = []
    candidates = aliases if task["kind"] == "BROAD" else [
        row for row in aliases if row["market"] == task["market"] and row["ticker"] == task["ticker"]
    ]
    for row in candidates:
        alias = row["alias"]
        full = row["full_alias"]
        ticker_marker = f"${row['ticker'].casefold()}"
        raw_lower = text.casefold()
        if (len(full) >= 5 and full in normalized) or (len(alias) >= 5 and alias in normalized) or ticker_marker in raw_lower:
            matches.append(row["ticker"])
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches, "CONTENT_ALIAS_EXACT"
    if len(matches) > 1:
        return matches, "AMBIGUOUS_MULTI_TICKER_CONTENT"
    return [], "QUERY_ONLY_UNVERIFIED_NO_TICKER_ASSIGNMENT" if task["kind"] == "ISSUER" else "UNMAPPED_BROAD"


def _content_encoded(item: ET.Element) -> str:
    for child in item:
        if str(child.tag).lower().endswith("encoded"):
            return clean_html(child.text or "")
    return ""


def parse_feed(body: bytes, task: dict[str, Any], captured_at: datetime, raw: dict[str, Any],
               aliases: list[dict[str, str]], policy_version: int,
               policy_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"MALFORMED_XML:{str(exc)[:160]}") from exc
    records = []
    invalid = []
    for position, item in enumerate(root.findall(".//item")):
        publisher_node = item.find("source")
        publisher = canonical_publisher(publisher_node.text if publisher_node is not None else "")
        publisher_url = str(publisher_node.attrib.get("url", "") if publisher_node is not None else "").strip()
        headline_raw = str(item.findtext("title") or "").strip()
        headline = strip_publisher_suffix(headline_raw, publisher)
        provider_url = str(item.findtext("link") or "").strip()
        guid = str(item.findtext("guid") or provider_url).strip()
        published = exact_pubdate(str(item.findtext("pubDate") or ""))
        description = clean_html(str(item.findtext("description") or ""))
        body_text = _content_encoded(item)
        if not headline or not provider_url or published is None or not publisher:
            invalid.append({
                "schema_version": SCHEMA_VERSION, "track": TRACK,
                "query_policy_version": policy_version, "query_policy_sha256": policy_sha256,
                "reason": "INVALID_RSS_ITEM_REQUIRED_METADATA",
                "query_id": task["query_id"], "capture_id": raw["capture_id"], "item_position": position,
                "headline_present": bool(headline), "provider_url_present": bool(provider_url),
                "published_at_exact": published is not None, "publisher_metadata_present": bool(publisher),
            })
            continue
        article_seed = guid or provider_url
        article_id = "GNEWS:" + sha256_bytes(article_seed.encode("utf-8"))
        mapped, mapping_evidence = map_tickers(f"{headline} {description} {body_text}", task, aliases)
        ticker = mapped[0] if len(mapped) == 1 else None
        text = f"{headline} {description} {body_text}"
        topic = topic_id(text)
        headline_key = normalize_text(headline)
        syndication_root = "SYN:" + sha256_bytes(headline_key.encode("utf-8"))
        event_seed = f"{task['market']}|{ticker or 'UNMAPPED'}|{published:%Y-%m-%dT%H:%MZ}|{topic}|{syndication_root}"
        event_cluster = "EVT:" + sha256_bytes(event_seed.encode("utf-8"))
        decision_cutoff = published + timedelta(minutes=2)
        usable = captured_at <= decision_cutoff and bool(raw["raw_capture_sha256"])
        reason = "PROVEN_LOCAL_FIRST_SEEN_BY_T_PLUS_2" if usable else "FIRST_SEEN_AFTER_DECISION_CUTOFF_HISTORICAL_ONLY"
        host = urlparse(provider_url).netloc.casefold()
        canonical_url = provider_url if host and "news.google.com" not in host else None
        issuer_company = task.get("company") if task.get("ticker") == ticker else None
        content_hash = sha256_bytes(canonical_json_bytes({
            "headline": headline, "body": body_text, "description": description,
            "publisher": publisher, "published_at_utc": iso_utc(published),
        }))
        records.append({
            "schema_version": SCHEMA_VERSION, "track": TRACK,
            "query_policy_version": policy_version, "query_policy_sha256": policy_sha256,
            "market": task["market"],
            "ticker": ticker, "candidate_tickers": mapped, "ticker_mapping_evidence": mapping_evidence,
            "canonical_event_id": event_cluster, "event_cluster_id": event_cluster,
            "article_id": article_id, "canonical_url": canonical_url, "provider_url": provider_url,
            "publisher": publisher, "publisher_url": publisher_url,
            "publisher_cluster_id": "PUB:" + sha256_bytes(normalize_text(f"{publisher}|{publisher_url}").encode("utf-8")),
            "publisher_class": publisher_class(publisher, publisher_url, issuer_company),
            "published_at_utc": iso_utc(published), "first_seen_at_utc": iso_utc(captured_at),
            "captured_at_utc": iso_utc(captured_at), "decision_cutoff_utc": iso_utc(decision_cutoff),
            "t_plus_2_usable": usable, "observed_before_cutoff": usable,
            "causal_eligibility_reason": reason, "historical_backfill": not usable,
            "timestamp_quality": "EXACT_SECOND_GOOGLE_RSS_PUBDATE",
            "first_seen_clock_source": "LOCAL_SYSTEM_UTC_AT_RESPONSE_COMPLETION",
            "headline": headline, "headline_raw": headline_raw, "body": body_text,
            "description": description, "source_provider": "GOOGLE_NEWS_RSS_PUBLIC",
            "source_type": "PUBLIC_AGGREGATOR_RSS_WITH_ORIGINAL_SOURCE_METADATA",
            "source_class": publisher_class(publisher, publisher_url, issuer_company),
            "topic_id": topic, "syndication_root_id": syndication_root,
            "semantic_similarity_to_root": 1.0, "content_sha256": content_hash,
            "raw_capture_path": raw["raw_capture_path"], "raw_capture_sha256": raw["raw_capture_sha256"],
            "raw_gzip_sha256": raw["raw_gzip_sha256"], "capture_id": raw["capture_id"],
            "query_id": task["query_id"], "query_kind": task["kind"],
            "selection_uses_labels": False, "selection_uses_prices": False,
            "research_seal_or_final_rows_read": False,
        })
    return records, invalid


def existing_first_seen(news_ledger: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in load_jsonl(news_ledger):
        article_id = str(row.get("article_id", ""))
        if article_id and article_id not in output:
            output[article_id] = row
    return output


def select_due_queries(state: dict[str, Any], now: datetime, maximum: int) -> list[dict[str, Any]]:
    due = [item for item in state["queue"] if item["status"] != "QUARANTINED"
           and parse_utc(item["next_due_at_utc"]) <= now]
    selected = []
    broad = [item for item in due if item["kind"] == "BROAD"]
    for market in ("US", "KR"):
        candidates = sorted((item for item in broad if item["market"] == market),
                            key=lambda item: (item["attempts"], item["query_id"]))
        if candidates and len(selected) < maximum:
            selected.append(candidates[0])
    remaining_broad = sorted((item for item in broad if item not in selected),
                             key=lambda item: (item["attempts"], item["query_id"]))
    for item in remaining_broad:
        if len(selected) >= min(maximum, 4):
            break
        selected.append(item)
    issuers = sorted((item for item in due if item["kind"] == "ISSUER"),
                     key=lambda item: (item["attempts"], item["last_attempt_at_utc"] or "", item["query_id"]))
    for item in issuers:
        if len(selected) >= maximum:
            break
        selected.append(item)
    return selected


def error_fingerprint(task: dict[str, Any], error_code: str, raw_sha: str | None) -> str:
    # Response bodies often contain changing request IDs even for the same
    # deterministic HTTP/schema failure.  Quarantine identity is therefore
    # the frozen query plus normalized deterministic error class.
    del raw_sha
    return sha256_bytes(f"{task['query_id']}|{error_code}".encode("utf-8"))


def schedule_success(task: dict[str, Any], now: datetime) -> None:
    task["status"] = "READY"
    task["last_success_at_utc"] = iso_utc(now)
    task["last_error_code"] = None
    task["consecutive_failures"] = 0
    task["deterministic_failure_count"] = 0
    task["deterministic_failure_fingerprint"] = None
    task["next_due_at_utc"] = iso_utc(now + timedelta(seconds=int(task["cadence_seconds"])))


def schedule_failure(task: dict[str, Any], now: datetime, error_code: str, deterministic: bool,
                     raw_sha: str | None = None) -> bool:
    task["consecutive_failures"] = int(task["consecutive_failures"]) + 1
    task["last_error_code"] = error_code
    quarantined = False
    if deterministic:
        fingerprint = error_fingerprint(task, error_code, raw_sha)
        if task.get("deterministic_failure_fingerprint") == fingerprint:
            task["deterministic_failure_count"] = int(task["deterministic_failure_count"]) + 1
        else:
            task["deterministic_failure_fingerprint"] = fingerprint
            task["deterministic_failure_count"] = 1
        if int(task["deterministic_failure_count"]) >= DETERMINISTIC_FAILURE_QUARANTINE_COUNT:
            task["status"] = "QUARANTINED"
            task["next_due_at_utc"] = iso_utc(now + timedelta(days=36500))
            quarantined = True
    else:
        task["deterministic_failure_count"] = 0
        task["deterministic_failure_fingerprint"] = None
    if not quarantined:
        delay_minutes = min(480, 5 * (2 ** min(int(task["consecutive_failures"]) - 1, 7)))
        task["status"] = "BACKOFF"
        task["next_due_at_utc"] = iso_utc(now + timedelta(minutes=delay_minutes))
    return quarantined


def resume_quarantined(state: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    rows = []
    for task in state["queue"]:
        if task["status"] == "QUARANTINED":
            rows.append({"schema_version": SCHEMA_VERSION, "track": TRACK,
                         "reason": "MANUAL_QUARANTINE_RESUME", "query_id": task["query_id"],
                         "resumed_at_utc": iso_utc(now), "prior_error_code": task.get("last_error_code")})
            task["status"] = "READY"
            task["next_due_at_utc"] = iso_utc(now)
            task["consecutive_failures"] = 0
            task["deterministic_failure_count"] = 0
            task["deterministic_failure_fingerprint"] = None
    return rows


def circuit_is_open(state: dict[str, Any], now: datetime) -> bool:
    breaker = state["circuit_breaker"]
    if breaker["status"] != "OPEN":
        return False
    if parse_utc(breaker["open_until_utc"]) > now:
        return True
    breaker.update({"status": "CLOSED", "opened_at_utc": None, "open_until_utc": None,
                    "reason": None, "consecutive_failures": 0})
    return False


def should_open_circuit(attempted: int, failed: int, consecutive: int) -> bool:
    return consecutive >= CIRCUIT_BREAKER_CONSECUTIVE_FAILURES or (
        attempted >= CIRCUIT_BREAKER_MIN_ATTEMPTS and failed / attempted >= CIRCUIT_BREAKER_FAILURE_RATIO
    )


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = sorted(set(schema["record_required_fields"]) - set(record))
    forbidden = sorted(FORBIDDEN_FIELDS & {str(name).casefold() for name in record})
    if missing or forbidden:
        raise RuntimeError(f"NEWS_CAPTURE_RECORD_SCHEMA_FAIL:missing={missing}:forbidden={forbidden}")
    if record["t_plus_2_usable"] != (parse_utc(record["first_seen_at_utc"]) <= parse_utc(record["decision_cutoff_utc"])
                                      and bool(record["raw_capture_sha256"])):
        raise RuntimeError("NEWS_CAPTURE_TPLUS2_CAUSAL_INVARIANT_FAIL")


def run_once(*, resume: bool, resume_quarantine: bool = False, max_queries: int = DEFAULT_MAX_QUERIES,
             timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
             min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
             root: Path = ROOT, namespace: Path = NAMESPACE, state_path: Path = STATE_PATH,
             policy_path: Path = POLICY_PATH, schema_path: Path = SCHEMA_PATH,
             fetcher: Callable[[dict[str, Any], dict[str, Any], float], dict[str, Any]] = http_fetch,
             now_fn: Callable[[], datetime] = utc_now, sleep_fn: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    started = now_fn()
    policy, policy_sha = load_policy(policy_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    verify_v1_immutable_evidence(namespace)
    universes, aliases = load_universes(policy, root)
    queue = make_queue(policy, universes, started)
    lock_path = namespace / ".capture.lock"
    with CaptureLock(lock_path, resume=resume, now=started):
        migration_ledger = namespace / "ledger" / "state_migration_history.jsonl"
        state = load_or_initialize_state(state_path, policy_sha, queue, started, resume, migration_ledger)
        # Persist a V1->V2 migration before any network request.  Each query is
        # checkpointed below as well, so a process interruption cannot reset
        # attempts, backoff, quarantine, or the 90-second broad schedule.
        state["updated_at_utc"] = iso_utc(started)
        atomic_json(state_path, state)
        quarantine_path = namespace / "ledger" / "quarantine.jsonl"
        raw_ledger = namespace / "ledger" / "raw_capture_ledger.jsonl"
        news_ledger = namespace / "ledger" / "news_records.jsonl"
        observation_ledger = namespace / "ledger" / "capture_observations.jsonl"
        if resume_quarantine:
            append_jsonl(quarantine_path, resume_quarantined(state, started))
        if circuit_is_open(state, started):
            state["status"] = "CIRCUIT_OPEN_BACKOFF"
            state["updated_at_utc"] = iso_utc(started)
            atomic_json(state_path, state)
            return {"status": state["status"], "requests": 0, "new_articles": 0,
                    "state_path": str(state_path), "output_written": True}

        due = select_due_queries(state, started, max(0, int(max_queries)))
        seen = existing_first_seen(news_ledger)
        run = {"requests": 0, "successes": 0, "failures": 0, "raw_captures": 0,
               "items_observed": 0, "new_articles": 0, "duplicates": 0,
               "t_plus_2_usable": 0, "historical_only": 0, "quarantined_queries": 0}
        consecutive_failures = 0
        last_request_monotonic: float | None = None
        for task in due:
            if last_request_monotonic is not None:
                remaining = min_interval_seconds - (time.monotonic() - last_request_monotonic)
                if remaining > 0:
                    sleep_fn(remaining)
            attempt_at = now_fn()
            task["attempts"] = int(task["attempts"]) + 1
            task["last_attempt_at_utc"] = iso_utc(attempt_at)
            fetched = fetcher(task, policy, timeout_seconds)
            captured_at = now_fn()
            last_request_monotonic = time.monotonic()
            run["requests"] += 1
            raw = persist_raw(namespace, task, fetched, captured_at)
            run["raw_captures"] += 1
            raw_row = {
                "schema_version": SCHEMA_VERSION, "track": TRACK,
                "query_policy_version": POLICY_VERSION, "query_policy_sha256": policy_sha,
                "capture_id": raw["capture_id"],
                "query_id": task["query_id"], "query_sha256": task["query_sha256"], "market": task["market"],
                "captured_at_utc": iso_utc(captured_at), "http_status": fetched.get("status"),
                "success": bool(fetched.get("ok")), "error_code": fetched.get("error_code"),
                "response_content_type": str(fetched.get("headers", {}).get("Content-Type", "")),
                **raw, "credentials_logged": False,
            }
            append_jsonl(raw_ledger, [raw_row])
            if fetched.get("ok"):
                try:
                    parsed, invalid = parse_feed(
                        bytes(fetched["body"]), task, captured_at, raw, aliases,
                        POLICY_VERSION, policy_sha,
                    )
                    for row in parsed:
                        validate_record(row, schema)
                    new_rows = []
                    observations = []
                    for row in parsed:
                        prior = seen.get(row["article_id"])
                        if prior is None:
                            new_rows.append(row)
                            seen[row["article_id"]] = row
                            run["new_articles"] += 1
                            if row["t_plus_2_usable"]:
                                run["t_plus_2_usable"] += 1
                            else:
                                run["historical_only"] += 1
                        else:
                            run["duplicates"] += 1
                        observations.append({
                            "schema_version": SCHEMA_VERSION, "track": TRACK,
                            "query_policy_version": POLICY_VERSION, "query_policy_sha256": policy_sha,
                            "article_id": row["article_id"],
                            "capture_id": raw["capture_id"], "query_id": task["query_id"],
                            "observed_at_utc": iso_utc(captured_at),
                            "first_seen_at_utc": prior["first_seen_at_utc"] if prior else row["first_seen_at_utc"],
                            "was_new_article": prior is None, "raw_capture_sha256": raw["raw_capture_sha256"],
                        })
                    append_jsonl(news_ledger, new_rows)
                    append_jsonl(observation_ledger, observations)
                    append_jsonl(quarantine_path, invalid)
                    run["items_observed"] += len(parsed)
                    schedule_success(task, captured_at)
                    run["successes"] += 1
                    consecutive_failures = 0
                except ValueError as exc:
                    error_code = str(exc).split(":", 1)[0]
                    quarantined = schedule_failure(task, captured_at, error_code, True, raw["raw_capture_sha256"])
                    append_jsonl(quarantine_path, [{
                        "schema_version": SCHEMA_VERSION, "track": TRACK, "reason": error_code,
                        "query_policy_version": POLICY_VERSION, "query_policy_sha256": policy_sha,
                        "query_id": task["query_id"], "capture_id": raw["capture_id"],
                        "captured_at_utc": iso_utc(captured_at), "deterministic": True,
                        "failure_count": task["deterministic_failure_count"], "quarantined": quarantined,
                    }])
                    run["failures"] += 1
                    run["quarantined_queries"] += int(quarantined)
                    consecutive_failures += 1
            else:
                error_code = str(fetched.get("error_code") or "UNKNOWN_FETCH_FAILURE")
                quarantined = schedule_failure(task, captured_at, error_code, bool(fetched.get("deterministic")),
                                               raw["raw_capture_sha256"])
                append_jsonl(quarantine_path, [{
                    "schema_version": SCHEMA_VERSION, "track": TRACK, "reason": error_code,
                    "query_policy_version": POLICY_VERSION, "query_policy_sha256": policy_sha,
                    "query_id": task["query_id"], "capture_id": raw["capture_id"],
                    "captured_at_utc": iso_utc(captured_at),
                    "deterministic": bool(fetched.get("deterministic")),
                    "failure_count": task["deterministic_failure_count"], "quarantined": quarantined,
                }])
                run["failures"] += 1
                run["quarantined_queries"] += int(quarantined)
                consecutive_failures += 1
            state["updated_at_utc"] = iso_utc(captured_at)
            atomic_json(state_path, state)
            if should_open_circuit(run["requests"], run["failures"], consecutive_failures):
                state["circuit_breaker"] = {
                    "status": "OPEN", "opened_at_utc": iso_utc(captured_at),
                    "open_until_utc": iso_utc(captured_at + timedelta(minutes=CIRCUIT_BREAKER_COOLDOWN_MINUTES)),
                    "reason": "RUN_FAILURE_THRESHOLD", "consecutive_failures": consecutive_failures,
                }
                break

        finished = now_fn()
        totals = state["totals"]
        totals["runs"] += 1
        for state_key, run_key in (("requests", "requests"), ("successful_requests", "successes"),
                                   ("failed_requests", "failures"), ("raw_captures", "raw_captures"),
                                   ("items_observed", "items_observed"), ("new_articles", "new_articles"),
                                   ("t_plus_2_usable_articles", "t_plus_2_usable"),
                                   ("historical_only_articles", "historical_only"),
                                   ("quarantined_queries", "quarantined_queries")):
            totals[state_key] += int(run[run_key])
        state["status"] = "CIRCUIT_OPEN_BACKOFF" if state["circuit_breaker"]["status"] == "OPEN" else "READY"
        state["updated_at_utc"] = iso_utc(finished)
        broad_due = [parse_utc(item["next_due_at_utc"]) for item in state["queue"]
                     if item["kind"] == "BROAD" and item["status"] != "QUARANTINED"]
        other_due = [parse_utc(item["next_due_at_utc"]) for item in state["queue"]
                     if item["status"] != "QUARANTINED"]
        recommended = min(broad_due or other_due or [finished + timedelta(seconds=BROAD_SUCCESS_CADENCE_SECONDS)])
        state["next_recommended_run_utc"] = iso_utc(max(finished, recommended))
        state["last_run"] = {
            "started_at_utc": iso_utc(started), "finished_at_utc": iso_utc(finished), **run,
            "due_queries_selected": len(due), "max_queries": int(max_queries),
            "min_request_interval_seconds": float(min_interval_seconds),
            "timeout_seconds": float(timeout_seconds), "credentials_logged": False,
        }
        atomic_json(state_path, state)
        return {
            "status": state["status"], **run, "due_queries_selected": len(due),
            "state_path": str(state_path), "namespace": str(namespace),
            "next_recommended_run_utc": state["next_recommended_run_utc"],
            "query_policy_sha256": policy_sha, "research_seal_or_final_rows_read": False,
            "query_policy_version": POLICY_VERSION,
            "output_written": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="V224 one-shot prospective public news capture")
    parser.add_argument("--once", action="store_true", help="run exactly one bounded collection cycle")
    parser.add_argument("--resume", action="store_true", help="resume the persistent query queue/state")
    parser.add_argument("--resume-quarantined", action="store_true", help="explicitly reactivate deterministic quarantines")
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--min-interval-seconds", type=float, default=DEFAULT_MIN_INTERVAL_SECONDS)
    args = parser.parse_args()
    if not args.once:
        parser.error("--once is required; this entrypoint never runs an internal daemon")
    if args.max_queries < 0 or args.max_queries > 20:
        parser.error("--max-queries must be between 0 and 20")
    result = run_once(
        resume=args.resume, resume_quarantine=args.resume_quarantined,
        max_queries=args.max_queries, timeout_seconds=args.timeout_seconds,
        min_interval_seconds=args.min_interval_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
