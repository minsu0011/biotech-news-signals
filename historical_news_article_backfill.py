"""Outcome-blind, local-first historical news article backfill.

This module deliberately has no dependency on price, label, prediction, role,
prospective, certificate, Research Seal, or Final Meta data.  It produces a
bounded, resumable article corpus under ``data/historical_news_backfill``.

Raw captures and registries are append-only.  Parquet files are reproducible
latest-revision snapshots of the append-only JSONL ledgers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

try:
    import runtime_limits as _runtime_limits
    RUNTIME_CPU_THREADS = int(_runtime_limits.THREAD_COUNT)
    RUNTIME_CPU_IDS = list(_runtime_limits.CPU_IDS)
except (ImportError, OSError, RuntimeError):
    RUNTIME_CPU_THREADS = int(os.cpu_count() or 1)
    RUNTIME_CPU_IDS = list(range(RUNTIME_CPU_THREADS))

import pandas as pd
import requests
from bs4 import BeautifulSoup

from historical_news_quality_contract import (
    CONTRACT_VERSION as QUALITY_CONTRACT_VERSION,
    TIER_A,
    TIER_B_FULL,
    TIER_B_HEADLINE,
    TIER_C,
    classify_historical_article_quality,
)


SCRIPT_VERSION = "1.4.0"
SCHEMA_VERSION = "HIST_NEWS_ARTICLE_V1"
USER_AGENT = "MARKET_BIO-HistoricalNewsBackfill/1.0 outcome-blind-research"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TERMINAL_HTTP = {400, 401, 403, 404, 410, 451}
TRACKING_QUERY_KEYS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "oc", "ref", "ref_src",
    "source", "utm_campaign", "utm_content", "utm_medium", "utm_source",
    "utm_term",
}
PRECISIONS = {"EXACT_SECOND", "EXACT_MINUTE", "HOUR_ONLY", "DATE_ONLY", "UNKNOWN"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}

ARTICLE_COLUMNS = [
    "schema_version", "article_uid", "article_revision", "canonical_event_id",
    "market", "ticker", "issuer", "event_time_utc", "decision_cutoff_utc",
    "canonical_url", "raw_url", "publisher_raw", "publisher_canonical",
    "origin_publisher", "hosting_publisher", "publisher_class", "source_provider",
    "source_type", "published_at_utc", "published_at_source",
    "published_at_precision", "published_at_confidence", "timezone_name",
    "timezone_inferred", "first_seen_at_utc", "first_seen_source",
    "first_seen_observed", "captured_at_utc", "headline", "body", "language",
    "article_id", "content_sha256", "raw_capture_path", "raw_capture_sha256",
    "raw_mime_type", "raw_capture_kind", "raw_capture_present",
    "discovery_window", "event_match_quality", "timestamp_leq_event_t2",
    "published_before_cutoff", "quality_contract_version", "tier",
    "causal_eligible_precondition", "headline_only_aux", "first_seen_unobserved",
    "tier_precondition_status", "tier_precondition_reasons", "ingested_at_utc",
]

DEFAULT_SOURCE_SPECS = (
    ("US_NEWS_REAL", "data/events_us_news_real.csv", "csv"),
    ("KR_NAVER_REAL", "data/events_naver_news_real.csv", "csv"),
    ("KR_NAVER_RAW_V35", "data/events_naver_kr_v35_raw.csv", "csv"),
    ("US_GOOGLE_RAW_V33", "data/events_google_us_v33_raw.csv", "csv"),
    ("KR_GOOGLE_RAW_V33", "data/events_google_news_v33_raw.csv", "csv"),
    ("KR_GOOGLE_RAW_V34", "data/events_google_kr_v34_raw.csv", "csv"),
    ("KR_NAVER_CACHE", "cache/naver_news_by_ticker", "json_dir"),
    ("QIU_REUTERS_CACHE", "cache/sources/QiuStockNews/json_files/news.json", "qiu_json"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave at most one partial tail record.  The next
                # append remains parseable and the malformed tail is audited.
                continue
            if isinstance(value, dict):
                yield value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value)
    has_markup = "<" in raw or bool(re.search(r"&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]+);", raw, re.I))
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True) if has_markup else raw
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        split = urlsplit(raw)
    except ValueError:
        return raw
    if split.scheme.lower() not in {"http", "https"} or not split.hostname:
        return raw
    host = split.hostname.lower().rstrip(".")
    port = split.port
    netloc = host
    if port and not ((split.scheme.lower() == "http" and port == 80) or (split.scheme.lower() == "https" and port == 443)):
        netloc = f"{host}:{port}"
    clean_query = [(k, v) for k, v in parse_qsl(split.query, keep_blank_values=True)
                   if k.lower() not in TRACKING_QUERY_KEYS and not k.lower().startswith("utm_")]
    path = re.sub(r"/{2,}", "/", split.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((split.scheme.lower(), netloc, path, urlencode(clean_query, doseq=True), ""))


def safe_public_url(value: str) -> bool:
    try:
        split = urlsplit(value)
    except ValueError:
        return False
    if split.scheme.lower() not in {"http", "https"} or not split.hostname or split.username or split.password:
        return False
    host = split.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast)


def _precision_from_literal(raw: str) -> str:
    value = raw.strip()
    if not value:
        return "UNKNOWN"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "DATE_ONLY"
    if re.search(r"(?:^|[T ])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?:\s*[AP]M)?", value, re.I):
        return "EXACT_SECOND"
    if re.search(r"(?:^|[T ])\d{1,2}:\d{2}(?:\s*[AP]M)?", value, re.I):
        return "EXACT_MINUTE"
    if re.search(r"[T ]\d{1,2}(?:Z|[+-]\d{2}:?\d{2})?$", value):
        return "HOUR_ONLY"
    return "UNKNOWN"


def parse_published(value: Any, source_provider: str = "") -> tuple[datetime | None, str, str, bool]:
    raw = str(value or "").strip()
    if not raw:
        return None, "UNKNOWN", "UNKNOWN", False
    precision = _precision_from_literal(raw)
    if "NAVER" in source_provider.upper() and precision == "EXACT_SECOND" and re.search(r":00(?:Z|[+-]|$)", raw):
        precision = "EXACT_MINUTE"
    timezone_offsets = {
        "UTC": "+00:00", "GMT": "+00:00",
        "EST": "-05:00", "EDT": "-04:00",
        "CST": "-06:00", "CDT": "-05:00",
        "MST": "-07:00", "MDT": "-06:00",
        "PST": "-08:00", "PDT": "-07:00",
    }
    parse_value = raw
    zone_match = re.search(r"\b(UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT)\b", raw, re.I)
    if zone_match:
        abbreviation = zone_match.group(1).upper()
        parse_value = re.sub(
            rf"\b{abbreviation}\b", timezone_offsets[abbreviation], raw, count=1, flags=re.I
        )
    try:
        stamp = pd.to_datetime(parse_value, utc=False, errors="raise").to_pydatetime()
    except Exception:
        return None, "UNKNOWN", "UNKNOWN", False
    timezone_inferred = stamp.tzinfo is None
    if timezone_inferred:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    confidence = "HIGH" if precision in {"EXACT_SECOND", "EXACT_MINUTE"} and not timezone_inferred else "LOW"
    if precision == "HOUR_ONLY" and not timezone_inferred:
        confidence = "MEDIUM"
    if precision == "DATE_ONLY":
        confidence = "LOW"
    return stamp, precision, confidence, timezone_inferred


def infer_language(headline: str, body: str) -> str:
    sample = (headline + " " + body)[:2000]
    if re.search(r"[\uac00-\ud7a3]", sample):
        return "ko"
    if re.search(r"[A-Za-z]", sample):
        return "en"
    return "und"


PUBLISHER_ALIASES = {
    "reuters": ("Reuters", "WIRE"),
    "thomson reuters": ("Reuters", "WIRE"),
    "naver news": ("Naver News", "AGGREGATOR"),
    "google news": ("Google News", "AGGREGATOR"),
    "associated press": ("Associated Press", "WIRE"),
    "ap": ("Associated Press", "WIRE"),
    "bloomberg": ("Bloomberg", "MAJOR_FINANCIAL"),
    "cnbc": ("CNBC", "MAJOR_FINANCIAL"),
    "yahoo finance": ("Yahoo Finance", "AGGREGATOR"),
    "business wire": ("Business Wire", "PR_DISTRIBUTOR"),
    "globenewswire": ("GlobeNewswire", "PR_DISTRIBUTOR"),
    "pr newswire": ("PR Newswire", "PR_DISTRIBUTOR"),
    "sec": ("SEC", "REGULATORY"),
    "kind": ("KIND", "REGULATORY"),
}

DOMAIN_PUBLISHERS = {
    "reuters.com": ("Reuters", "WIRE"),
    "news.google.com": ("Google News", "AGGREGATOR"),
    "news.naver.com": ("Naver News", "AGGREGATOR"),
    "n.news.naver.com": ("Naver News", "AGGREGATOR"),
    "finance.yahoo.com": ("Yahoo Finance", "AGGREGATOR"),
    "bloomberg.com": ("Bloomberg", "MAJOR_FINANCIAL"),
    "cnbc.com": ("CNBC", "MAJOR_FINANCIAL"),
    "businesswire.com": ("Business Wire", "PR_DISTRIBUTOR"),
    "globenewswire.com": ("GlobeNewswire", "PR_DISTRIBUTOR"),
    "prnewswire.com": ("PR Newswire", "PR_DISTRIBUTOR"),
    "sec.gov": ("SEC", "REGULATORY"),
    "kind.krx.co.kr": ("KIND", "REGULATORY"),
}


def _canonical_publisher(raw: str) -> tuple[str, str]:
    clean = normalize_text(raw).strip(" -|")
    key = re.sub(r"\s+", " ", clean).casefold()
    if key in PUBLISHER_ALIASES:
        return PUBLISHER_ALIASES[key]
    for alias, result in PUBLISHER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", key):
            return result
    if not clean:
        return "Unknown", "OTHER"
    return clean[:160], "OTHER"


def publisher_metadata(raw: str, url: str, headline: str, body: str, provider: str) -> dict[str, str]:
    host = (urlsplit(url).hostname or "").lower() if url else ""
    hosting, hosting_class = "Unknown", "OTHER"
    for domain, result in DOMAIN_PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            hosting, hosting_class = result
            break
    candidate = normalize_text(raw)
    if not candidate and " - " in headline:
        suffix = headline.rsplit(" - ", 1)[1].strip()
        if 1 < len(suffix) <= 80:
            candidate = suffix
    origin, origin_class = _canonical_publisher(candidate)
    attribution = (headline + " " + body[:400]).casefold()
    if "reuters" in attribution or "REUTERS" in str(provider).upper():
        origin, origin_class = "Reuters", "WIRE"
    if origin == "Unknown" and hosting not in {"Naver News", "Google News", "Yahoo Finance"}:
        origin, origin_class = hosting, hosting_class
    canonical = origin if origin != "Unknown" else hosting
    klass = origin_class if origin != "Unknown" else hosting_class
    return {
        "publisher_canonical": canonical,
        "origin_publisher": origin,
        "hosting_publisher": hosting,
        "publisher_class": klass,
    }


def iter_jsonld(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        graph = value.get("@graph")
        if graph is not None:
            yield from iter_jsonld(graph)
    elif isinstance(value, list):
        for item in value:
            yield from iter_jsonld(item)


def parse_html_article(raw: bytes, url: str, provider: str) -> dict[str, Any]:
    soup = BeautifulSoup(raw, "html.parser")
    jsonld: list[Mapping[str, Any]] = []
    for node in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            parsed = json.loads(node.string or node.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        jsonld.extend(iter_jsonld(parsed))

    def meta(*names: str) -> str:
        lowered = {x.casefold() for x in names}
        for node in soup.find_all("meta"):
            key = str(node.get("property") or node.get("name") or "").casefold()
            if key in lowered and node.get("content"):
                return str(node.get("content")).strip()
        return ""

    headline = meta("og:title", "twitter:title")
    publisher = meta("og:site_name", "application-name")
    published_raw = ""
    published_source = ""
    body = ""
    for item in jsonld:
        if not headline and item.get("headline"):
            headline = str(item.get("headline"))
        if not published_raw and item.get("datePublished"):
            published_raw = str(item.get("datePublished"))
            published_source = "JSON_LD_DATE_PUBLISHED"
        if not body and item.get("articleBody"):
            body = str(item.get("articleBody"))
        pub = item.get("publisher")
        if not publisher and isinstance(pub, Mapping) and pub.get("name"):
            publisher = str(pub.get("name"))
    if not published_raw:
        for key in ("article:published_time", "datePublished", "date", "pubdate", "publish-date"):
            value = meta(key)
            if value:
                published_raw, published_source = value, f"HTML_META:{key}"
                break
    if not published_raw:
        node = soup.find("time", attrs={"datetime": True})
        if node:
            published_raw, published_source = str(node.get("datetime")), "HTML_TIME_DATETIME"
    if not headline:
        headline = normalize_text(soup.title.get_text() if soup.title else "")
    if not body:
        article = soup.find("article") or soup.find("main") or soup.body
        if article:
            paragraphs = [normalize_text(p.get_text(" ", strip=True)) for p in article.find_all("p")]
            body = "\n".join(p for p in paragraphs if p)
    published, precision, confidence, tz_inferred = parse_published(published_raw, provider)
    canonical = meta("og:url") or url
    return {
        "headline": normalize_text(headline),
        "body": normalize_text(body),
        "publisher_raw": normalize_text(publisher),
        "canonical_url": canonicalize_url(urljoin(url, canonical)),
        "published_at": published,
        "published_at_source": published_source or "UNKNOWN",
        "published_at_precision": precision,
        "published_at_confidence": confidence,
        "timezone_inferred": tz_inferred,
    }


def source_candidate(raw: Mapping[str, Any], provider: str) -> dict[str, Any]:
    row = {str(k): v for k, v in raw.items()}
    return {
        "canonical_event_id": str(row.get("event_id") or row.get("canonical_event_id") or "").strip(),
        "market": str(row.get("market") or ("KR" if "NAVER" in provider else "US")).upper(),
        "ticker": str(row.get("ticker") or row.get("symbol") or "").strip(),
        "issuer": normalize_text(row.get("company") or row.get("issuer") or ""),
        "event_time": row.get("event_time_utc") or row.get("event_time"),
        "raw_url": str(row.get("url") or row.get("link") or "").strip(),
        "headline": normalize_text(row.get("headline") or row.get("title") or ""),
        "body": normalize_text(row.get("body") or row.get("text") or row.get("description") or ""),
        "article_id": str(row.get("article_id") or "").strip(),
        "publisher_raw": normalize_text(row.get("publisher") or row.get("publisher_raw") or ""),
        "source_provider": str(row.get("source") or provider),
        "source_type": str(row.get("source_type") or ("LOCAL_PROVIDER_CACHE" if "CACHE" in provider else "LOCAL_EVENT_TABLE")),
        "published_raw": row.get("published_at_utc") or row.get("published_at") or row.get("pub_time") or row.get("event_time_utc"),
        "timestamp_quality_raw": str(row.get("timestamp_quality") or ""),
        "first_seen_at_utc": row.get("first_seen_at_utc"),
        "first_seen_source": str(row.get("first_seen_source") or ""),
        "first_seen_genuine_historical": bool(row.get("first_seen_genuine_historical") is True),
        "event_match_quality": str(row.get("event_match_quality") or ""),
        "discovery_window": str(row.get("discovery_window") or ""),
        "local_record": row,
    }


def article_uid(candidate: Mapping[str, Any], canonical_url: str) -> str:
    identity = canonical_url or str(candidate.get("article_id") or "")
    if not identity:
        identity = "|".join([
            str(candidate.get("source_provider") or ""),
            str(candidate.get("published_raw") or ""),
            str(candidate.get("headline") or ""),
        ])
    return "ART_" + hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()[:32]


def raw_store(output_root: Path, raw: bytes, suffix: str = "bin") -> tuple[str, str]:
    digest = sha256_bytes(raw)
    rel = Path("raw") / digest[:2] / f"{digest}.{suffix}.gz"
    path = output_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with gzip.open(path, "rb") as handle:
            if sha256_bytes(handle.read()) != digest:
                raise RuntimeError(f"raw capture hash mismatch: {path}")
    else:
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        try:
            with path.open("xb") as handle:
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            pass
    return rel.as_posix(), digest


def build_normalized(candidate: Mapping[str, Any], output_root: Path, *, raw: bytes,
                     raw_mime: str, raw_kind: str, html_fields: Mapping[str, Any] | None = None,
                     revision: int = 1) -> dict[str, Any]:
    html_fields = dict(html_fields or {})
    captured = utc_now()
    headline = normalize_text(html_fields.get("headline") or candidate.get("headline"))
    body = normalize_text(html_fields.get("body") or candidate.get("body"))
    raw_url = str(candidate.get("raw_url") or "")
    canonical_url = canonicalize_url(html_fields.get("canonical_url") or raw_url)
    provider = str(candidate.get("source_provider") or "UNKNOWN")
    published = html_fields.get("published_at")
    precision = str(html_fields.get("published_at_precision") or "")
    confidence = str(html_fields.get("published_at_confidence") or "")
    tz_inferred = bool(html_fields.get("timezone_inferred", False))
    published_source = str(html_fields.get("published_at_source") or "")
    if published is None:
        published, precision, confidence, tz_inferred = parse_published(candidate.get("published_raw"), provider)
        published_source = "LOCAL_PROVIDER_TIMESTAMP" if published else "UNKNOWN"
        quality = str(candidate.get("timestamp_quality_raw") or "").upper()
        if quality in {"EXACT", "HIGH"} and published and not tz_inferred and precision in {"EXACT_SECOND", "EXACT_MINUTE"}:
            confidence = "HIGH"
    if precision not in PRECISIONS:
        precision = "UNKNOWN"
    if confidence not in CONFIDENCES:
        confidence = "UNKNOWN"

    first_seen = None
    first_seen_source = ""
    if candidate.get("first_seen_genuine_historical") is True:
        first_seen, _, _, _ = parse_published(candidate.get("first_seen_at_utc"), provider)
        first_seen_source = str(candidate.get("first_seen_source") or "EXPLICIT_HISTORICAL_PROVIDER")

    event_time, _, _, _ = parse_published(candidate.get("event_time"), provider)
    cutoff = event_time + timedelta(minutes=2) if event_time else None
    before_cutoff = bool(published and cutoff and published <= cutoff) if published and cutoff else None
    event_match = str(candidate.get("event_match_quality") or "").upper()
    if event_match not in {"HIGH", "MEDIUM", "LOW"}:
        event_match = "HIGH" if candidate.get("canonical_event_id") and candidate.get("ticker") else "LOW"
    pub = publisher_metadata(
        str(html_fields.get("publisher_raw") or candidate.get("publisher_raw") or ""),
        canonical_url, headline, body, provider,
    )
    raw_rel, raw_sha = raw_store(output_root, raw, "html" if "html" in raw_mime else "json")
    content_sha = sha256_bytes((headline + "\n" + body).encode("utf-8"))
    uid = article_uid(candidate, canonical_url)

    decision = classify_historical_article_quality(
        canonical_publisher=pub["publisher_canonical"],
        published_at_precision=precision,
        published_at_confidence=confidence,
        published_at=published,
        decision_cutoff=cutoff,
        event_match_quality=event_match,
        headline=headline,
        body=body,
        first_seen_at=first_seen,
        first_seen_observed=first_seen is not None,
        captured_at=captured,
        provenance_available=bool(raw_sha or content_sha),
    )
    tier = decision.tier
    eligible = decision.causal_eligible
    reasons = list(decision.reasons)

    return {
        "schema_version": SCHEMA_VERSION,
        "article_uid": uid,
        "article_revision": int(revision),
        "canonical_event_id": str(candidate.get("canonical_event_id") or ""),
        "market": str(candidate.get("market") or ""),
        "ticker": str(candidate.get("ticker") or ""),
        "issuer": str(candidate.get("issuer") or ""),
        "event_time_utc": iso_utc(event_time),
        "decision_cutoff_utc": iso_utc(cutoff),
        "canonical_url": canonical_url,
        "raw_url": raw_url,
        "publisher_raw": str(html_fields.get("publisher_raw") or candidate.get("publisher_raw") or ""),
        **pub,
        "source_provider": provider,
        "source_type": str(candidate.get("source_type") or ""),
        "published_at_utc": iso_utc(published),
        "published_at_source": published_source,
        "published_at_precision": precision,
        "published_at_confidence": confidence,
        "timezone_name": "UTC" if published else "",
        "timezone_inferred": tz_inferred,
        "first_seen_at_utc": iso_utc(first_seen),
        "first_seen_source": first_seen_source,
        "first_seen_observed": first_seen is not None,
        "captured_at_utc": iso_utc(captured),
        "headline": headline,
        "body": body,
        "language": infer_language(headline, body),
        "article_id": str(candidate.get("article_id") or ""),
        "content_sha256": content_sha,
        "raw_capture_path": raw_rel,
        "raw_capture_sha256": raw_sha,
        "raw_mime_type": raw_mime,
        "raw_capture_kind": raw_kind,
        "raw_capture_present": True,
        "discovery_window": str(candidate.get("discovery_window") or ("LOCAL_EXISTING" if raw_kind == "LOCAL_PROVIDER_RECORD" else "BOUNDED_PUBLIC_RECOVERY")),
        "event_match_quality": event_match,
        "timestamp_leq_event_t2": before_cutoff,
        "published_before_cutoff": before_cutoff,
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "tier": tier,
        "causal_eligible_precondition": eligible,
        "headline_only_aux": decision.headline_only_aux,
        "first_seen_unobserved": decision.first_seen_unobserved,
        "tier_precondition_status": "PASS" if eligible else ("HEADLINE_ONLY_AUX" if tier == TIER_B_HEADLINE else "AUX_ONLY"),
        "tier_precondition_reasons": reasons,
        "ingested_at_utc": iso_utc(captured),
    }


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    kind: str


def iter_source(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    if spec.kind == "csv":
        with spec.path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                yield source_candidate(row, spec.name)
        return
    if spec.kind == "json_dir":
        for path in sorted(spec.path.glob("*.json"), key=lambda p: p.name):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, Mapping):
                        yield source_candidate(row, spec.name)
        return
    if spec.kind == "qiu_json":
        try:
            value = json.loads(spec.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(value, Mapping):
            return
        for ticker in sorted(value):
            rows = value[ticker]
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                material = dict(row)
                material.update({
                    "ticker": ticker,
                    "market": "US",
                    "source": "REUTERS_QIU_LOCAL_CACHE",
                    "event_time_utc": row.get("pub_time"),
                    "event_id": "QIU:" + str(ticker) + ":" + sha256_bytes(canonical_json_bytes(row))[:20],
                    "publisher": "Reuters",
                    "timestamp_quality": "EXACT",
                })
                yield source_candidate(material, spec.name)


class RequestRegistry:
    def __init__(self, output_root: Path):
        self.path = output_root / "ledgers" / "REQUEST_REGISTRY.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.path):
            digest = str(row.get("request_digest") or "")
            if digest:
                self.latest[digest] = row

    @staticmethod
    def digest(kind: str, url: str) -> str:
        return sha256_bytes(canonical_json_bytes({"kind": kind, "method": "GET", "url": canonicalize_url(url)}))

    def may_attempt(self, kind: str, url: str, now: datetime) -> tuple[bool, str]:
        digest = self.digest(kind, url)
        prior = self.latest.get(digest)
        if not prior:
            return True, digest
        # A successful robots response is parsed only in process memory. A new
        # process must fetch it again before deciding whether a new article is
        # allowed; treating the ledger SUCCESS as reusable content would fail
        # closed forever for every previously visited host.
        if kind == "ROBOTS" and prior.get("status") == "SUCCESS":
            return True, digest
        if prior.get("status") in {"SUCCESS", "TERMINAL", "QUARANTINED", "ROBOTS_DISALLOW"}:
            return False, digest
        next_at = prior.get("next_attempt_at_utc")
        if next_at:
            stamp, _, _, _ = parse_published(next_at)
            if stamp and now < stamp:
                return False, digest
        return True, digest

    def record(self, *, digest: str, kind: str, url: str, status: str,
               http_status: int | None = None, error: str = "", retry_after_seconds: int = 0) -> None:
        prior = self.latest.get(digest, {})
        failures = int(prior.get("consecutive_failures") or 0)
        failures = 0 if status == "SUCCESS" else failures + (1 if status in {"RETRY", "FAILED"} else 0)
        if failures >= 3 and status in {"RETRY", "FAILED"}:
            status = "QUARANTINED"
        now = utc_now()
        next_at = iso_utc(now + timedelta(seconds=retry_after_seconds)) if retry_after_seconds and status != "QUARANTINED" else None
        row = {
            "request_digest": digest, "kind": kind, "method": "GET",
            "canonical_url": canonicalize_url(url), "status": status,
            "http_status": http_status, "error_class": error[:160],
            "consecutive_failures": failures, "attempted_at_utc": iso_utc(now),
            "next_attempt_at_utc": next_at,
        }
        append_jsonl(self.path, row)
        self.latest[digest] = row


class PublicFetcher:
    def __init__(self, output_root: Path, registry: RequestRegistry, max_requests: int):
        self.output_root = output_root
        self.registry = registry
        self.max_requests = max(0, int(max_requests))
        self.requests_used = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.2"})
        self.robots: dict[str, tuple[bool, RobotFileParser | None]] = {}
        self.last_host_request: dict[str, float] = {}

    def _bounded_get(self, url: str, kind: str) -> requests.Response | None:
        if self.requests_used >= self.max_requests or not safe_public_url(url):
            return None
        allowed, digest = self.registry.may_attempt(kind, url, utc_now())
        if not allowed:
            return None
        host = (urlsplit(url).hostname or "").lower()
        elapsed = time.monotonic() - self.last_host_request.get(host, 0.0)
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self.requests_used += 1
        self.last_host_request[host] = time.monotonic()
        try:
            response = self.session.get(url, timeout=(5, 15), allow_redirects=False, stream=True)
        except requests.RequestException as exc:
            prior = self.registry.latest.get(digest, {})
            delay = min(3600, 30 * (2 ** int(prior.get("consecutive_failures") or 0)))
            self.registry.record(digest=digest, kind=kind, url=url, status="RETRY", error=type(exc).__name__, retry_after_seconds=delay)
            return None
        if 300 <= response.status_code < 400 and response.headers.get("Location"):
            location = urljoin(url, response.headers["Location"])
            self.registry.record(digest=digest, kind=kind, url=url, status="SUCCESS", http_status=response.status_code)
            if safe_public_url(location):
                return self._bounded_get(location, kind)
            return None
        if response.status_code == 429:
            retry = response.headers.get("Retry-After", "60")
            try:
                delay = max(60, min(86400, int(retry)))
            except ValueError:
                delay = 300
            self.registry.record(digest=digest, kind=kind, url=url, status="RETRY", http_status=429,
                                 error="HTTP_429", retry_after_seconds=delay)
            return None
        if response.status_code in TERMINAL_HTTP:
            self.registry.record(digest=digest, kind=kind, url=url, status="TERMINAL",
                                 http_status=response.status_code, error=f"HTTP_{response.status_code}")
            return None
        if response.status_code >= 500:
            prior = self.registry.latest.get(digest, {})
            delay = min(3600, 30 * (2 ** int(prior.get("consecutive_failures") or 0)))
            self.registry.record(digest=digest, kind=kind, url=url, status="RETRY",
                                 http_status=response.status_code, error=f"HTTP_{response.status_code}", retry_after_seconds=delay)
            return None
        if response.status_code != 200:
            self.registry.record(digest=digest, kind=kind, url=url, status="FAILED",
                                 http_status=response.status_code, error=f"HTTP_{response.status_code}", retry_after_seconds=300)
            return None
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    self.registry.record(digest=digest, kind=kind, url=url, status="TERMINAL",
                                         http_status=200, error="RESPONSE_TOO_LARGE")
                    return None
                chunks.append(chunk)
        except requests.RequestException as exc:
            self.registry.record(digest=digest, kind=kind, url=url, status="RETRY",
                                 http_status=200, error=type(exc).__name__, retry_after_seconds=60)
            return None
        response._content = b"".join(chunks)
        self.registry.record(digest=digest, kind=kind, url=url, status="SUCCESS", http_status=200)
        return response

    def robots_allowed(self, url: str) -> bool:
        split = urlsplit(url)
        origin = f"{split.scheme.lower()}://{split.netloc.lower()}"
        if origin in self.robots:
            allowed, parser = self.robots[origin]
            return allowed and bool(parser and parser.can_fetch(USER_AGENT, url))
        robots_url = origin + "/robots.txt"
        response = self._bounded_get(robots_url, "ROBOTS")
        if response is None:
            # 404/410 robots means no declared restriction; all other missing
            # states fail closed for this run.
            digest = self.registry.digest("ROBOTS", robots_url)
            prior = self.registry.latest.get(digest, {})
            if prior.get("http_status") in {404, 410}:
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
                self.robots[origin] = (True, parser)
                return True
            self.robots[origin] = (False, None)
            return False
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.content.decode(response.encoding or "utf-8", errors="replace").splitlines())
        self.robots[origin] = (True, parser)
        return parser.can_fetch(USER_AGENT, url)

    def article(self, url: str) -> tuple[bytes, str] | None:
        if not safe_public_url(url) or self.requests_used >= self.max_requests:
            return None
        if not self.robots_allowed(url) or self.requests_used >= self.max_requests:
            return None
        response = self._bounded_get(url, "ARTICLE")
        if response is None:
            return None
        mime = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].strip().lower()
        if mime not in {"text/html", "application/xhtml+xml"}:
            return None
        return response.content, mime

    def document(self, url: str, kind: str, allowed_mimes: set[str]) -> tuple[bytes, str] | None:
        if not safe_public_url(url) or self.requests_used >= self.max_requests:
            return None
        if not self.robots_allowed(url) or self.requests_used >= self.max_requests:
            return None
        response = self._bounded_get(url, kind)
        if response is None:
            return None
        mime = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].strip().lower()
        if mime not in allowed_mimes:
            return None
        return response.content, mime


class BackfillEngine:
    def __init__(self, root: Path, output_root: Path | None = None,
                 source_specs: Iterable[SourceSpec] | None = None):
        self.root = root.resolve()
        self.output_root = (output_root or self.root / "data" / "historical_news_backfill").resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_root / "state" / "ARTICLE_BACKFILL_WORKER_STATE.json"
        self.legacy_state_path = self.output_root / "state" / "HIST_NEWS_BACKFILL_STATE.json"
        self.canonical_state_path = self.root / "state" / "HIST_NEWS_BACKFILL_STATE.json"
        self.article_ledger = self.output_root / "ledgers" / "ARTICLES_NORMALIZED.jsonl"
        self.event_source_ledger = self.output_root / "ledgers" / "EVENT_SOURCE_REGISTRY.jsonl"
        self.fetch_backlog_ledger = self.output_root / "ledgers" / "FETCH_BACKLOG.jsonl"
        self.discovery_query_ledger = self.output_root / "ledgers" / "DISCOVERY_QUERY_REGISTRY.jsonl"
        self.run_ledger = self.output_root / "ledgers" / "RUN_LEDGER.jsonl"
        for ledger in (self.article_ledger, self.event_source_ledger, self.fetch_backlog_ledger,
                       self.discovery_query_ledger, self.run_ledger):
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.touch(exist_ok=True)
        self.specs = list(source_specs or [SourceSpec(n, self.root / p, k) for n, p, k in DEFAULT_SOURCE_SPECS])
        self.state = self._load_state()
        self.article_latest = {str(x.get("article_uid")): x for x in read_jsonl(self.article_ledger) if x.get("article_uid")}
        self.event_source_seen = {str(x.get("event_source_digest")) for x in read_jsonl(self.event_source_ledger) if x.get("event_source_digest")}
        self.fetch_backlog_latest = {
            str(x.get("backlog_digest")): x for x in read_jsonl(self.fetch_backlog_ledger)
            if x.get("backlog_digest")
        }
        self.discovery_query_latest = {
            str(x.get("query_digest")): x for x in read_jsonl(self.discovery_query_ledger)
            if x.get("query_digest")
        }
        self.request_registry = RequestRegistry(self.output_root)

    def _load_state(self) -> dict[str, Any]:
        load_path = self.state_path if self.state_path.exists() else self.legacy_state_path
        if load_path.exists():
            with load_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("historical news state schema mismatch")
            return value
        now = iso_utc(utc_now())
        return {
            "schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
            "session_start": now, "productive_seconds": 0.0,
            "current_phase": "LOCAL_ARTICLE_DISCOVERY", "events_total": 0,
            "events_processed": 0, "articles_discovered": 0,
            "articles_downloaded": 0, "articles_parsed": 0,
            "causal_tier_a": 0, "causal_tier_b": 0, "tier_b_full": 0,
            "tier_b_headline": 0, "aux_tier_c": 0,
            "article_candidates_scanned": 0, "canonical_articles_count": 0,
            "pending_sources": [s.name for s in self.specs if s.path.exists()],
            "failed_sources": [], "source_cursors": {}, "last_checkpoint": None,
            "last_run_status": "INITIALIZED", "requests_total": 0,
            "request_failures": 0, "quarantined_requests": 0,
            "pending_article_fetches": 0,
            "source_exhausted": {}, "discovery_queries_attempted": 0,
            "discovery_expansion_pending": True,
            "discovery_expansion_reason": "LOCAL_SOURCES_NOT_EXHAUSTED",
            "cpu_worker_count": RUNTIME_CPU_THREADS, "cpu_ids": RUNTIME_CPU_IDS,
            "gpu_used": False,
        }

    def _merge_canonical_state(self) -> None:
        """Merge only this worker's subsection; never account useful runtime.

        The root scheduler is the sole authority for productive_seconds and
        session-wide phase/counters.  This worker therefore preserves every
        existing top-level field and replaces only ``article_backfill``.
        """
        if self.canonical_state_path.exists():
            try:
                canonical = json.loads(self.canonical_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                canonical = {}
        else:
            canonical = {}
        canonical.setdefault("session_start", self.state.get("session_start") or iso_utc(utc_now()))
        canonical.setdefault("productive_seconds", 0.0)
        canonical.setdefault("current_phase", "ARTICLE_BACKFILL_AVAILABLE")
        canonical.setdefault("events_total", 0)
        canonical.setdefault("events_processed", 0)
        canonical.setdefault("articles_discovered", 0)
        canonical.setdefault("articles_downloaded", 0)
        canonical.setdefault("articles_parsed", 0)
        canonical.setdefault("causal_tier_a", 0)
        canonical.setdefault("causal_tier_b", 0)
        canonical.setdefault("tier_b_full", 0)
        canonical.setdefault("tier_b_headline", 0)
        canonical.setdefault("aux_tier_c", 0)
        canonical.setdefault("pending_sources", [])
        canonical.setdefault("failed_sources", [])
        canonical.setdefault("last_checkpoint", None)
        canonical["article_backfill"] = {
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "status": self.state.get("last_run_status"),
            "worker_productive_seconds_diagnostic_only": self.state.get("productive_seconds", 0.0),
            "canonical_productive_seconds_modified": False,
            "events_processed": self.state.get("events_processed", 0),
            "articles_discovered": self.state.get("articles_discovered", 0),
            "articles_downloaded": self.state.get("articles_downloaded", 0),
            "articles_parsed": self.state.get("articles_parsed", 0),
            "causal_tier_a": self.state.get("causal_tier_a", 0),
            "causal_tier_b": self.state.get("causal_tier_b", 0),
            "tier_b_full": self.state.get("tier_b_full", self.state.get("causal_tier_b", 0)),
            "tier_b_headline": self.state.get("tier_b_headline", 0),
            "aux_tier_c": self.state.get("aux_tier_c", 0),
            "article_candidates_scanned": self.state.get("article_candidates_scanned", self.state.get("events_processed", 0)),
            "canonical_articles_count": self.state.get("canonical_articles_count", self.state.get("articles_parsed", 0)),
            "pending_article_fetches": self.state.get("pending_article_fetches", 0),
            "discovery_queries_attempted": self.state.get("discovery_queries_attempted", 0),
            "discovery_expansion_pending": self.state.get("discovery_expansion_pending", True),
            "discovery_expansion_reason": self.state.get("discovery_expansion_reason"),
            "cpu_worker_count": self.state.get("cpu_worker_count", RUNTIME_CPU_THREADS),
            "gpu_used": False,
            "requests_total": self.state.get("requests_total", 0),
            "request_failures": self.state.get("request_failures", 0),
            "quarantined_requests": self.state.get("quarantined_requests", 0),
            "last_checkpoint": self.state.get("last_checkpoint"),
            "worker_state_path": str(self.state_path),
            "articles_path": str(self.output_root / "ARTICLES_NORMALIZED.parquet"),
        }
        atomic_json(self.canonical_state_path, canonical)

    def local_audit(self) -> dict[str, Any]:
        sources = []
        for spec in self.specs:
            exists = spec.path.exists()
            stat = spec.path.stat() if exists else None
            sources.append({
                "name": spec.name,
                "path": str(spec.path.relative_to(self.root)) if spec.path.is_relative_to(self.root) else str(spec.path),
                "kind": spec.kind, "exists": exists,
                "bytes": stat.st_size if stat and spec.path.is_file() else None,
                "mtime_utc": iso_utc(datetime.fromtimestamp(stat.st_mtime, timezone.utc)) if stat else None,
                "outcome_fields_read": False,
            })
        audit = {
            "schema_version": SCHEMA_VERSION, "audited_at_utc": iso_utc(utc_now()),
            "policy": "EXPLICIT_ALLOWLIST_LOCAL_NEWS_ONLY",
            "forbidden_namespaces_read": [], "sources": sources,
        }
        atomic_json(self.output_root / "LOCAL_ARTICLE_CACHE_AUDIT.json", audit)
        return audit

    def _event_source(self, candidate: Mapping[str, Any], uid: str, status: str) -> None:
        key = canonical_json_bytes({
            "event": candidate.get("canonical_event_id"), "article_uid": uid,
            "provider": candidate.get("source_provider"), "url": canonicalize_url(candidate.get("raw_url")),
        })
        digest = sha256_bytes(key)
        if digest in self.event_source_seen:
            return
        append_jsonl(self.event_source_ledger, {
            "event_source_digest": digest,
            "canonical_event_id": str(candidate.get("canonical_event_id") or ""),
            "article_uid": uid, "source_provider": str(candidate.get("source_provider") or ""),
            "canonical_url": canonicalize_url(candidate.get("raw_url")),
            "status": status, "registered_at_utc": iso_utc(utc_now()),
        })
        self.event_source_seen.add(digest)

    def _append_article(self, row: dict[str, Any]) -> bool:
        uid = row["article_uid"]
        prior = self.article_latest.get(uid)
        if prior:
            # Same article may legitimately map to several event-source pairs;
            # membership is held in EVENT_SOURCE_REGISTRY, not duplicated here.
            return False
        append_jsonl(self.article_ledger, row)
        self.article_latest[uid] = row
        return True

    def _backlog_digest(self, candidate: Mapping[str, Any]) -> str:
        return sha256_bytes(canonical_json_bytes({
            "event": candidate.get("canonical_event_id"),
            "provider": candidate.get("source_provider"),
            "url": canonicalize_url(candidate.get("raw_url")),
        }))

    def _record_backlog(self, candidate: Mapping[str, Any], status: str,
                        detail: str = "") -> str:
        digest = self._backlog_digest(candidate)
        prior = self.fetch_backlog_latest.get(digest)
        if prior and prior.get("status") == status and status == "PENDING":
            return digest
        row = {
            "backlog_digest": digest, "status": status,
            "candidate": dict(candidate), "detail": detail[:240],
            "updated_at_utc": iso_utc(utc_now()),
        }
        append_jsonl(self.fetch_backlog_ledger, row)
        self.fetch_backlog_latest[digest] = row
        return digest

    def _recover_candidate(self, candidate: Mapping[str, Any], fetcher: PublicFetcher) -> tuple[int, int]:
        """Try one queued article and return (downloads, parse_operations)."""
        url = str(candidate.get("raw_url") or "")
        if not url or not safe_public_url(url):
            self._record_backlog(candidate, "TERMINAL", "UNSAFE_OR_EMPTY_URL")
            return 0, 0
        prior_uid = article_uid(candidate, canonicalize_url(url))
        prior = self.article_latest.get(prior_uid)
        if prior and len(str(prior.get("body") or "")) >= 80:
            self._record_backlog(candidate, "SUCCESS", "LOCAL_OR_PRIOR_BODY_SUFFICIENT")
            return 0, 0
        if fetcher.requests_used >= fetcher.max_requests:
            self._record_backlog(candidate, "PENDING", "REQUEST_BUDGET_EXHAUSTED")
            return 0, 0
        fetched = fetcher.article(url)
        if not fetched:
            request = self.request_registry.latest.get(self.request_registry.digest("ARTICLE", url), {})
            status = str(request.get("status") or "PENDING")
            if status not in {"TERMINAL", "QUARANTINED", "ROBOTS_DISALLOW"}:
                status = "PENDING"
            self._record_backlog(candidate, status, str(request.get("error_class") or "DEFERRED_OR_ROBOTS"))
            return 0, 0
        raw_html, mime = fetched
        html = parse_html_article(raw_html, url, str(candidate.get("source_provider") or ""))
        revision = int((prior or {}).get("article_revision") or 0) + 1
        recovered = build_normalized(candidate, self.output_root, raw=raw_html,
                                     raw_mime=mime, raw_kind="PUBLIC_HTTP_CAPTURE",
                                     html_fields=html, revision=revision)
        if prior and len(str(recovered.get("body") or "")) <= len(str(prior.get("body") or "")):
            self._record_backlog(candidate, "SUCCESS", "FETCHED_NO_RICHER_CONTENT")
            return 1, 1
        append_jsonl(self.article_ledger, recovered)
        self.article_latest[prior_uid] = recovered
        self._record_backlog(candidate, "SUCCESS", "PUBLIC_BODY_RECOVERED")
        return 1, 1

    def _write_snapshots(self) -> None:
        rows = list(self.article_latest.values())
        frame = pd.DataFrame(rows)
        for col in ARTICLE_COLUMNS:
            if col not in frame:
                frame[col] = None
        frame = frame[ARTICLE_COLUMNS].sort_values(["published_at_utc", "article_uid"], na_position="last") if len(frame) else frame[ARTICLE_COLUMNS]
        parquet = self.output_root / "ARTICLES_NORMALIZED.parquet"
        tmp = parquet.with_name(parquet.name + f".tmp.{os.getpid()}")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, parquet)

        quality_cols = ["article_uid", "quality_contract_version", "tier", "causal_eligible_precondition",
                        "headline_only_aux", "first_seen_unobserved", "publisher_class",
                        "published_at_precision", "published_at_confidence", "event_match_quality",
                        "raw_capture_present", "tier_precondition_status", "tier_precondition_reasons"]
        qpath = self.output_root / "ARTICLE_TIER_PRECONDITIONS.parquet"
        qtmp = qpath.with_name(qpath.name + f".tmp.{os.getpid()}")
        frame[quality_cols].to_parquet(qtmp, index=False)
        os.replace(qtmp, qpath)

        publisher_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("publisher_canonical")), str(row.get("publisher_class")))
            item = publisher_rows.setdefault(key, {
                "publisher_canonical": key[0], "publisher_class": key[1],
                "domains": set(), "aliases": set(), "source_types": set(),
            })
            host = urlsplit(str(row.get("canonical_url") or "")).hostname
            if host:
                item["domains"].add(host.lower())
            if row.get("publisher_raw"):
                item["aliases"].add(str(row["publisher_raw"]))
            if row.get("source_type"):
                item["source_types"].add(str(row["source_type"]))
        preg = pd.DataFrame([
            {**x, "domains": sorted(x["domains"]), "aliases": sorted(x["aliases"]),
             "source_types": sorted(x["source_types"])} for x in publisher_rows.values()
        ], columns=["publisher_canonical", "publisher_class", "domains", "aliases", "source_types"])
        ppath = self.output_root / "PUBLISHER_REGISTRY.parquet"
        ptmp = ppath.with_name(ppath.name + f".tmp.{os.getpid()}")
        preg.to_parquet(ptmp, index=False)
        os.replace(ptmp, ppath)

        schema_contract = {
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "path": str(parquet),
            "columns": ARTICLE_COLUMNS,
            "timestamp_precision_values": sorted(PRECISIONS),
            "timestamp_confidence_values": sorted(CONFIDENCES),
            "first_seen_rule": "NULL_UNLESS_EXPLICIT_GENUINE_HISTORICAL_PROVENANCE",
            "quality_contract_version": QUALITY_CONTRACT_VERSION,
            "tier_rule": {
                "A": "strict common preconditions plus genuine historical first_seen <= event t+2",
                "B_FULL": "known canonical publisher, precise high-confidence published_at <= t+2, HIGH event match, substantive headline/body, first_seen unobserved",
                "B_HEADLINE": "precise causal headline with insufficient body; auxiliary only",
                "C": "retrospective auxiliary; not strict causal eligible",
            },
            "outcome_dependent_fields": [],
        }
        atomic_json(self.output_root / "ARTICLES_NORMALIZED_SCHEMA.json", schema_contract)

    def _normalize_local_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Pure-per-item normalization plus content-addressed raw write.

        This is safe to run in parallel: raw_store uses exclusive creation and
        verifies existing content.  All JSONL/parquet/state commits remain on
        the caller's single thread in input order.
        """
        local_raw = canonical_json_bytes(candidate.get("local_record") or candidate)
        return build_normalized(candidate, self.output_root, raw=local_raw,
                                raw_mime="application/json", raw_kind="LOCAL_PROVIDER_RECORD")

    @staticmethod
    def _rss_query_url(event: Mapping[str, Any]) -> str:
        market = str(event.get("market") or "US").upper()
        ticker = str(event.get("ticker") or "").strip()
        issuer = normalize_text(event.get("issuer_name") or event.get("issuer") or "")
        stamp, _, _, _ = parse_published(event.get("event_time_utc"))
        if stamp:
            after = (stamp - timedelta(days=1)).date().isoformat()
            before = (stamp + timedelta(days=2)).date().isoformat()
            date_terms = f" after:{after} before:{before}"
        else:
            date_terms = ""
        identity = f'"{issuer}"' if issuer else ticker
        query = f"{identity} {ticker}{date_terms}".strip()
        locale = {"hl": "ko", "gl": "KR", "ceid": "KR:ko"} if market == "KR" else {"hl": "en-US", "gl": "US", "ceid": "US:en"}
        return "https://news.google.com/rss/search?" + urlencode({"q": query, **locale})

    def _record_discovery_query(self, event: Mapping[str, Any], url: str, status: str,
                                *, count: int = 0, raw_path: str = "", raw_sha: str = "",
                                detail: str = "") -> str:
        digest = sha256_bytes(canonical_json_bytes({
            "event": event.get("canonical_event_id"), "provider": "GOOGLE_NEWS_PUBLIC_RSS",
            "url": canonicalize_url(url),
        }))
        row = {
            "query_digest": digest, "canonical_event_id": str(event.get("canonical_event_id") or ""),
            "query_url": canonicalize_url(url), "provider": "GOOGLE_NEWS_PUBLIC_RSS",
            "status": status, "items_accepted": count, "raw_capture_path": raw_path,
            "raw_capture_sha256": raw_sha, "detail": detail[:240],
            "updated_at_utc": iso_utc(utc_now()),
        }
        append_jsonl(self.discovery_query_ledger, row)
        self.discovery_query_latest[digest] = row
        return digest

    def _discover_from_event_master(self, fetcher: PublicFetcher, max_queries: int) -> tuple[int, int, int]:
        """Bounded label-blind public RSS discovery after local exhaustion.

        Returns (queries_attempted, articles_discovered, parse_operations).
        """
        event_path = self.output_root / "EVENT_MASTER.parquet"
        if not event_path.exists():
            self.state["discovery_expansion_pending"] = True
            self.state["discovery_expansion_reason"] = "EVENT_MASTER_NOT_AVAILABLE"
            return 0, 0, 0
        if fetcher.requests_used >= fetcher.max_requests or max_queries <= 0:
            self.state["discovery_expansion_pending"] = True
            self.state["discovery_expansion_reason"] = "REQUEST_OR_QUERY_BUDGET_ZERO"
            return 0, 0, 0
        columns = ["canonical_event_id", "market", "ticker", "issuer_name", "event_time_utc"]
        events = pd.read_parquet(event_path, columns=columns)
        events = events.sort_values("event_time_utc", ascending=False, na_position="last")
        attempts = accepted = operations = 0
        for event in events.to_dict("records"):
            if attempts >= max_queries or fetcher.requests_used >= fetcher.max_requests:
                break
            url = self._rss_query_url(event)
            digest = sha256_bytes(canonical_json_bytes({
                "event": event.get("canonical_event_id"), "provider": "GOOGLE_NEWS_PUBLIC_RSS",
                "url": canonicalize_url(url),
            }))
            prior = self.discovery_query_latest.get(digest)
            if prior and prior.get("status") in {"SUCCESS", "TERMINAL", "QUARANTINED"}:
                continue
            attempts += 1
            document = fetcher.document(url, "DISCOVERY_RSS", {
                "application/rss+xml", "application/xml", "text/xml", "application/atom+xml",
            })
            if not document:
                req = self.request_registry.latest.get(self.request_registry.digest("DISCOVERY_RSS", url), {})
                status = str(req.get("status") or "PENDING")
                if status not in {"TERMINAL", "QUARANTINED"}:
                    status = "PENDING"
                self._record_discovery_query(event, url, status, detail=str(req.get("error_class") or "DEFERRED_OR_ROBOTS"))
                continue
            raw, _mime = document
            raw_rel, raw_sha = raw_store(self.output_root, raw, "xml")
            try:
                tree = ET.fromstring(raw)
            except ET.ParseError:
                self._record_discovery_query(event, url, "TERMINAL", raw_path=raw_rel,
                                             raw_sha=raw_sha, detail="MALFORMED_XML")
                continue
            event_time, _, _, _ = parse_published(event.get("event_time_utc"))
            event_accepted = 0
            for item in tree.findall(".//item"):
                title = normalize_text(item.findtext("title") or "")
                link = str(item.findtext("link") or "").strip()
                pub_raw = str(item.findtext("pubDate") or "").strip()
                pub_time, _, _, _ = parse_published(pub_raw, "GOOGLE_NEWS_PUBLIC_RSS")
                if not title or not link or not pub_time or not event_time:
                    continue
                if pub_time < event_time - timedelta(hours=24) or pub_time > event_time + timedelta(hours=24):
                    continue
                ticker = str(event.get("ticker") or "").strip()
                issuer = normalize_text(event.get("issuer_name") or "")
                issuer_tokens = [x.casefold() for x in re.findall(r"[A-Za-z0-9\uac00-\ud7a3]+", issuer) if len(x) >= 4]
                title_fold = title.casefold()
                high_match = bool(ticker and re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker.casefold())}(?![A-Za-z0-9])", title_fold)) or any(x in title_fold for x in issuer_tokens[:4])
                source_node = item.find("source")
                publisher = normalize_text(source_node.text if source_node is not None else "")
                window = "CAUSAL_T2_CANDIDATE" if pub_time <= event_time + timedelta(minutes=2) else "RETROSPECTIVE_SUPPORT"
                material = {
                    "event_id": event.get("canonical_event_id"), "market": event.get("market"),
                    "ticker": ticker, "company": issuer, "event_time_utc": event.get("event_time_utc"),
                    "source": "GOOGLE_NEWS_PUBLIC_RSS", "source_type": "PUBLIC_RSS_DISCOVERY",
                    "title": title, "text": "", "pub_time": pub_raw, "url": link,
                    "publisher": publisher, "timestamp_quality": "EXACT",
                    "event_match_quality": "HIGH" if high_match else "LOW",
                    "discovery_window": window,
                    "article_id": "GNEWS_DISC:" + sha256_bytes(link.encode("utf-8"))[:24],
                }
                candidate = source_candidate(material, "GOOGLE_NEWS_PUBLIC_RSS")
                local_raw = canonical_json_bytes(material)
                row = build_normalized(candidate, self.output_root, raw=local_raw,
                                       raw_mime="application/json", raw_kind="PUBLIC_RSS_ITEM")
                uid = row["article_uid"]
                self._event_source(candidate, uid, "PUBLIC_RSS_DISCOVERED")
                if self._append_article(row):
                    accepted += 1
                    event_accepted += 1
                    operations += 1
                if len(str(self.article_latest.get(uid, row).get("body") or "")) < 80:
                    self._record_backlog(candidate, "PENDING", "RSS_ITEM_BODY_RECOVERY")
            self._record_discovery_query(event, url, "SUCCESS", count=event_accepted,
                                         raw_path=raw_rel, raw_sha=raw_sha)
        self.state["discovery_expansion_pending"] = True
        self.state["discovery_expansion_reason"] = (
            "MORE_UNQUERIED_EVENTS_REMAIN" if attempts else "QUERY_SPACE_EXHAUSTED_OR_DEFERRED"
        )
        return attempts, accepted, operations

    def run_once(self, *, resume: bool, max_requests: int, max_local_items: int,
                 max_discovery_queries: int = 5, workers: int = RUNTIME_CPU_THREADS) -> dict[str, Any]:
        if self.state_path.exists() and not resume:
            raise RuntimeError("state exists; pass --resume to continue without overwriting")
        self.local_audit()
        started = time.monotonic()
        fetcher = PublicFetcher(self.output_root, self.request_registry, max_requests)
        processed = discovered = appended = downloaded = parsed = 0
        source_errors: list[dict[str, str]] = []
        remaining = max(0, int(max_local_items))
        source_cursors = self.state.setdefault("source_cursors", {})
        source_exhausted = self.state.setdefault("source_exhausted", {})
        workers = max(1, min(int(workers), RUNTIME_CPU_THREADS))
        workers_used = 0

        # Persistent recovery backlog is serviced before discovering more URLs.
        # Exhausted request budgets leave entries pending without causing failure.
        for item in list(self.fetch_backlog_latest.values()):
            if fetcher.requests_used >= fetcher.max_requests:
                break
            if item.get("status") != "PENDING" or not isinstance(item.get("candidate"), Mapping):
                continue
            got, operations = self._recover_candidate(item["candidate"], fetcher)
            downloaded += got
            parsed += operations

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hist-news-local") as pool:
            for spec in self.specs:
                if remaining <= 0:
                    break
                if not spec.path.exists():
                    continue
                cursor = int(source_cursors.get(spec.name, 0))
                ordinal = 0
                exhausted_this_run = False
                try:
                    iterator = iter(iter_source(spec))
                    while ordinal < cursor:
                        try:
                            next(iterator)
                        except StopIteration:
                            exhausted_this_run = True
                            break
                        ordinal += 1
                    while not exhausted_this_run and remaining > 0:
                        batch: list[dict[str, Any]] = []
                        cap = min(remaining, max(workers * 4, 32))
                        for _ in range(cap):
                            try:
                                batch.append(next(iterator))
                            except StopIteration:
                                exhausted_this_run = True
                                break
                        if not batch:
                            break
                        remaining -= len(batch)
                        workers_used = max(workers_used, min(workers, len(batch)))
                        rows = list(pool.map(self._normalize_local_candidate, batch))
                        for candidate, row in zip(batch, rows):
                            ordinal += 1
                            processed += 1
                            discovered += 1
                            uid = row["article_uid"]
                            self._event_source(candidate, uid, "LOCAL_DISCOVERED")
                            added = self._append_article(row)
                            appended += int(added)
                            parsed += int(added)
                            current = self.article_latest.get(uid, row)
                            if len(str(current.get("body") or "")) < 80 and candidate.get("raw_url"):
                                self._record_backlog(candidate, "PENDING", "MISSING_SUBSTANTIVE_BODY")
                                if fetcher.requests_used < fetcher.max_requests:
                                    got, operations = self._recover_candidate(candidate, fetcher)
                                    downloaded += got
                                    parsed += operations
                            source_cursors[spec.name] = ordinal
                    if exhausted_this_run:
                        source_exhausted[spec.name] = True
                except Exception as exc:
                    source_exhausted[spec.name] = False
                    source_errors.append({"source": spec.name, "error_class": type(exc).__name__, "message": str(exc)[:300]})
                    if spec.name not in self.state["failed_sources"]:
                        self.state["failed_sources"].append(spec.name)
                    continue

        local_exhausted = all(source_exhausted.get(s.name) is True for s in self.specs if s.path.exists())
        discovery_queries = 0
        if local_exhausted:
            discovery_queries, rss_discovered, rss_operations = self._discover_from_event_master(
                fetcher, max_discovery_queries,
            )
            discovered += rss_discovered
            appended += rss_discovered
            parsed += rss_operations
        else:
            self.state["discovery_expansion_pending"] = True
            self.state["discovery_expansion_reason"] = "LOCAL_SOURCES_NOT_EXHAUSTED"

        self._write_snapshots()
        duration = time.monotonic() - started
        self.state.update({
            "script_version": SCRIPT_VERSION,
            "productive_seconds": round(float(self.state.get("productive_seconds", 0.0)) + duration, 6),
            "current_phase": "ARTICLE_BACKFILL_BOUNDED_BATCH_COMPLETE",
            "events_processed": int(self.state.get("events_processed", 0)) + processed,
            "events_total": max(int(self.state.get("events_total", 0)), len(self.event_source_seen)),
            "articles_discovered": int(self.state.get("articles_discovered", 0)) + discovered,
            "articles_downloaded": int(self.state.get("articles_downloaded", 0)) + downloaded,
            "articles_parsed": len(self.article_latest),
            "causal_tier_a": sum(x.get("tier") == TIER_A for x in self.article_latest.values()),
            "causal_tier_b": sum(x.get("tier") in {"B", TIER_B_FULL} for x in self.article_latest.values()),
            "tier_b_full": sum(x.get("tier") in {"B", TIER_B_FULL} for x in self.article_latest.values()),
            "tier_b_headline": sum(x.get("tier") == TIER_B_HEADLINE for x in self.article_latest.values()),
            "aux_tier_c": sum(x.get("tier") == TIER_C for x in self.article_latest.values()),
            "article_candidates_scanned": int(self.state.get("article_candidates_scanned", self.state.get("events_processed", 0))) + processed,
            "canonical_articles_count": len(self.article_latest),
            "pending_sources": [s.name for s in self.specs if s.path.exists()],
            "last_checkpoint": iso_utc(utc_now()),
            "last_run_status": "COMPLETED_WITH_SOURCE_QUARANTINE" if source_errors else ("NO_PROGRESS" if processed == 0 else "COMPLETED"),
            "requests_total": int(self.state.get("requests_total", 0)) + fetcher.requests_used,
            "request_failures": sum(x.get("status") in {"FAILED", "RETRY"} for x in self.request_registry.latest.values()),
            "quarantined_requests": sum(x.get("status") == "QUARANTINED" for x in self.request_registry.latest.values()),
            "pending_article_fetches": sum(x.get("status") == "PENDING" for x in self.fetch_backlog_latest.values()),
            "discovery_queries_attempted": int(self.state.get("discovery_queries_attempted", 0)) + discovery_queries,
            "cpu_worker_count": workers, "cpu_workers_used_last_run": workers_used,
            "cpu_ids": RUNTIME_CPU_IDS, "gpu_used": False,
        })
        atomic_json(self.state_path, self.state)
        self._merge_canonical_state()
        summary = {
            "schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
            "status": self.state["last_run_status"], "bounded": True,
            "local_items_processed": processed, "articles_discovered": discovered,
            "new_canonical_articles": appended, "articles_downloaded": downloaded,
            "articles_parsed_operations": parsed, "requests_used": fetcher.requests_used,
            "max_requests": max_requests, "max_local_items": max_local_items,
            "max_discovery_queries": max_discovery_queries,
            "cpu_worker_count": workers, "cpu_workers_used": workers_used,
            "gpu_used": False,
            "source_errors": source_errors, "state_path": str(self.canonical_state_path),
            "worker_state_path": str(self.state_path),
            "articles_path": str(self.output_root / "ARTICLES_NORMALIZED.parquet"),
            "first_seen_policy": "NULL_UNLESS_EXPLICIT_GENUINE_HISTORICAL_PROVENANCE",
            "outcome_data_read": False, "completed_at_utc": iso_utc(utc_now()),
        }
        append_jsonl(self.run_ledger, summary)
        atomic_json(self.output_root / "LAST_RUN_STATUS.json", summary)
        return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one bounded batch; no daemon mode exists.")
    parser.add_argument("--resume", action="store_true", help="Resume persistent cursors and registries.")
    parser.add_argument("--max-requests", type=int, default=0, help="Hard cap including robots.txt and redirects.")
    parser.add_argument("--max-local-items", type=int, default=2500, help="Hard cap on local records in this batch.")
    parser.add_argument("--max-discovery-queries", type=int, default=5, help="Hard cap after local sources are exhausted.")
    parser.add_argument("--workers", type=int, default=RUNTIME_CPU_THREADS,
                        help="Bounded local parse workers; commits remain deterministic and single-writer.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.once:
        parser.error("--once is required; unbounded daemon mode is intentionally unsupported")
    if args.max_requests < 0 or args.max_local_items < 0 or args.max_discovery_queries < 0 or args.workers < 1:
        parser.error("request and local-item bounds must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        engine = BackfillEngine(args.root, args.output_root)
        summary = engine.run_once(resume=args.resume, max_requests=args.max_requests,
                                  max_local_items=args.max_local_items,
                                  max_discovery_queries=args.max_discovery_queries,
                                  workers=args.workers)
    except Exception as exc:
        print(json.dumps({"status": "FATAL", "error_class": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    # Bounded source exhaustion, rate limiting, and no-progress are legitimate
    # resumable states and intentionally return success.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
