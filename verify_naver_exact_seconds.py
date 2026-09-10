"""Verify Naver publication seconds without reading roles, selections, or labels.

The Naver ticker-news API used by the historical collectors exposes only a
minute timestamp.  The public article page exposes the publication timestamp
to the second in the ``_ARTICLE_DATE_TIME`` element.  This standalone worker
reads only event metadata, verifies the page and its identifiers, and writes a
new content-addressed snapshot.  It never mutates an input snapshot.

Rows that fail any input, HTTP, identifier, timestamp, or minute-floor check
are omitted from the verified snapshot and recorded in a quarantine report.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import requests

import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POOL = DATA / "untouched_pool"
DEFAULT_V35 = DATA / "events_naver_kr_v35_raw.csv"
KR_TZ = "Asia/Seoul"

# Deliberately excludes every role, selection, price, return, and label field.
# ``selection_uses_labels`` is not loaded even when it exists in an input file.
SAFE_INPUT_COLUMNS = (
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
)
REQUIRED_INPUT_COLUMNS = {
    "event_id",
    "market",
    "ticker",
    "event_time_utc",
    "source",
    "url",
    "article_id",
}
URL_RE = re.compile(r"^/(?:mnews/)?article/(?P<office>[0-9]{3})/(?P<article>[0-9]{10})/?$")
EVENT_RE = re.compile(r"^NAVER:(?P<ticker>[0-9]{6}):(?P<article_id>[0-9]{13})$")
OFFICE_RE = re.compile(r"\bofficeId\s*:\s*[\"'](?P<value>[0-9]{3})[\"']")
ARTICLE_RE = re.compile(r"\barticleId\s*:\s*[\"'](?P<value>[0-9]{10})[\"']")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
USER_AGENT = "Mozilla/5.0 MARKET_BIO Naver publication-time verifier/1.0"
_THREAD_LOCAL = threading.local()


class PublicationTimeParser(HTMLParser):
    """Extract publication timestamps, excluding modification timestamps."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if "_ARTICLE_DATE_TIME" in classes and attr.get("data-date-time"):
            self.values.append(attr["data-date-time"].strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode(
        "utf-8"
    )


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_naver_url(value: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.hostname != "n.news.naver.com" or parsed.query or parsed.fragment:
        return None
    match = URL_RE.fullmatch(parsed.path)
    if not match:
        return None
    return match.group("office"), match.group("article")


def quarantine_entry(row: pd.Series, reason: str, detail: str = "", **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reason": reason,
        "detail": detail[:500],
        "event_id": str(row.get("event_id", "")),
        "article_id": str(row.get("article_id", "")),
        "ticker": str(row.get("ticker", "")),
        "url": str(row.get("url", "")),
        "input_path": str(row.get("_input_path", "")),
        "input_row": int(row.get("_input_row", -1)),
    }
    result.update(extra)
    return result


def validate_input_row(row: pd.Series) -> tuple[bool, str, str]:
    market = str(row.get("market", "")).strip().upper()
    ticker = str(row.get("ticker", "")).strip().zfill(6)
    event_id = str(row.get("event_id", "")).strip()
    article_id = str(row.get("article_id", "")).strip()
    source = str(row.get("source", "")).strip().upper()
    if market != "KR":
        return False, "INPUT_MARKET_NOT_KR", market
    if not source.startswith("NAVER"):
        return False, "INPUT_SOURCE_NOT_NAVER", source
    if not re.fullmatch(r"[0-9]{6}", ticker):
        return False, "INPUT_TICKER_INVALID", ticker
    event_match = EVENT_RE.fullmatch(event_id)
    if not event_match:
        return False, "INPUT_EVENT_ID_INVALID", event_id
    if event_match.group("ticker") != ticker or event_match.group("article_id") != article_id:
        return False, "INPUT_EVENT_ID_MISMATCH", f"ticker={ticker};article_id={article_id}"
    if not re.fullmatch(r"[0-9]{13}", article_id):
        return False, "INPUT_ARTICLE_ID_INVALID", article_id
    url_parts = parse_naver_url(str(row.get("url", "")).strip())
    if not url_parts:
        return False, "INPUT_URL_INVALID", str(row.get("url", ""))
    office_id, page_article_id = url_parts
    if office_id + page_article_id != article_id:
        return False, "INPUT_URL_ID_MISMATCH", f"url={office_id}{page_article_id};row={article_id}"
    stamp = pd.to_datetime(row.get("event_time_utc", ""), utc=True, errors="coerce")
    if pd.isna(stamp):
        return False, "INPUT_TIMESTAMP_INVALID", str(row.get("event_time_utc", ""))
    return True, "", ""


def load_metadata(path: Path, input_priority: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(header))
    if missing:
        raise ValueError(f"{path}: missing required metadata columns: {missing}")
    allowed = [column for column in SAFE_INPUT_COLUMNS if column in header]
    frame = pd.read_csv(path, usecols=allowed, dtype=str, keep_default_na=False)
    for column in SAFE_INPUT_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame = frame.loc[:, SAFE_INPUT_COLUMNS].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.zfill(6)
    frame["_input_path"] = str(path.resolve())
    frame["_input_row"] = range(2, len(frame) + 2)  # CSV line number, including header.
    frame["_input_priority"] = input_priority
    return frame


def load_and_deduplicate(paths: Iterable[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        # Prefer prospective untouched-pool metadata to V35 when the same
        # article is present in both.  All remaining ties use the collector's
        # original outcome-blind article_id|ticker hash.
        priority = 0 if path.parent.resolve() == POOL.resolve() else 1
        frames.append(load_metadata(path, priority))
    if not frames:
        raise ValueError("no input metadata files")
    combined = pd.concat(frames, ignore_index=True)
    quarantine: list[dict[str, Any]] = []
    valid_mask: list[bool] = []
    for _, row in combined.iterrows():
        valid, reason, detail = validate_input_row(row)
        valid_mask.append(valid)
        if not valid:
            quarantine.append(quarantine_entry(row, reason, detail))
    strict = combined.loc[valid_mask].copy()
    strict["_selection_hash"] = [
        hashlib.sha256(f"{article_id}|{ticker}".encode("utf-8")).hexdigest()
        for article_id, ticker in zip(strict["article_id"], strict["ticker"])
    ]
    strict = strict.sort_values(
        ["event_id", "_input_priority", "_selection_hash", "_input_path", "_input_row"], kind="stable"
    )
    before_event = len(strict)
    strict = strict.drop_duplicates("event_id", keep="first")
    deduplicated_event = before_event - len(strict)
    strict = strict.sort_values(
        ["article_id", "_input_priority", "_selection_hash", "event_id", "_input_path", "_input_row"], kind="stable"
    )
    before_article = len(strict)
    strict = strict.drop_duplicates("article_id", keep="first")
    deduplicated_article = before_article - len(strict)
    strict = strict.sort_values(["article_id", "event_id"], kind="stable").reset_index(drop=True)
    counts = {
        "input_rows": len(combined),
        "input_invalid_rows": len(combined) - sum(valid_mask),
        "strict_rows_before_deduplication": sum(valid_mask),
        "deduplicated_event_id_rows": deduplicated_event,
        "deduplicated_article_id_rows": deduplicated_article,
        "strict_unique_article_rows": len(strict),
    }
    return strict, quarantine, counts


def session() -> requests.Session:
    value = getattr(_THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
        _THREAD_LOCAL.session = value
    return value


def page_identity(html: str) -> tuple[set[str], set[str]]:
    return ({match.group("value") for match in OFFICE_RE.finditer(html)}, {match.group("value") for match in ARTICLE_RE.finditer(html)})


def publication_values(html: str) -> list[str]:
    parser = PublicationTimeParser()
    parser.feed(html)
    return sorted(set(parser.values))


def fetch_and_verify(row: pd.Series, timeout: float, retries: int) -> dict[str, Any]:
    url = str(row["url"])
    requested_office, requested_article = parse_naver_url(url) or ("", "")
    response: requests.Response | None = None
    error = ""
    for attempt in range(retries + 1):
        try:
            response = session().get(url, timeout=timeout, allow_redirects=True)
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.5 * (2**attempt)
            time.sleep(min(delay, 10.0))
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    if response is None:
        return {"ok": False, "reason": "REQUEST_ERROR", "detail": error}
    if response.status_code == 404:
        return {"ok": False, "reason": "HTTP_404", "detail": "article page not found", "http_status": 404}
    if response.status_code != 200:
        return {
            "ok": False,
            "reason": "HTTP_STATUS",
            "detail": f"status={response.status_code}",
            "http_status": response.status_code,
        }
    final_parts = parse_naver_url(response.url)
    if final_parts != (requested_office, requested_article):
        return {"ok": False, "reason": "FINAL_URL_MISMATCH", "detail": response.url, "http_status": 200}
    offices, articles = page_identity(response.text)
    if offices != {requested_office} or articles != {requested_article}:
        return {
            "ok": False,
            "reason": "PAGE_ID_MISMATCH" if offices and articles else "PAGE_ID_MISSING",
            "detail": f"offices={sorted(offices)};articles={sorted(articles)}",
            "http_status": 200,
        }
    values = publication_values(response.text)
    if not values:
        return {"ok": False, "reason": "PUBLICATION_TIME_MISSING", "detail": "", "http_status": 200}
    if len(values) != 1:
        return {
            "ok": False,
            "reason": "PUBLICATION_TIME_AMBIGUOUS",
            "detail": repr(values),
            "http_status": 200,
        }
    try:
        local_time = pd.Timestamp(datetime.strptime(values[0], DATE_FORMAT), tz=KR_TZ)
    except ValueError as exc:
        return {"ok": False, "reason": "PUBLICATION_TIME_INVALID", "detail": str(exc), "http_status": 200}
    original = pd.to_datetime(row["event_time_utc"], utc=True, errors="coerce")
    if original.tz_convert(KR_TZ).floor("min") != local_time.floor("min"):
        return {
            "ok": False,
            "reason": "MINUTE_FLOOR_MISMATCH",
            "detail": f"input={original.isoformat()};page={local_time.isoformat()}",
            "http_status": 200,
        }
    return {
        "ok": True,
        "event_time_utc": local_time.tz_convert("UTC").isoformat(),
        "publication_time_kst": local_time.isoformat(),
        "verified_office_id": requested_office,
        "verified_page_article_id": requested_article,
        "http_status": 200,
    }


def output_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_gzip_deterministic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def date_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return "EMPTY", "EMPTY"
    stamps = pd.to_datetime(frame["event_time_utc"], utc=True).dt.tz_convert(KR_TZ)
    return stamps.min().date().isoformat(), stamps.max().date().isoformat()


def outcome_blind_candidate_scope(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce page requests using only the frozen event-metadata contract."""
    prior = exact.normalized_events(DATA / "events_exact_v35.csv.gz")
    candidate = frame.copy()
    candidate["ticker"] = candidate["ticker"].astype(str).str.zfill(6)
    candidate["event_time_utc"] = pd.to_datetime(
        candidate["event_time_utc"], utc=True, errors="coerce", format="mixed"
    )
    candidate = candidate.dropna(subset=["event_time_utc"]).copy()
    candidate, _ = exact.outside_prior(candidate, prior)
    universe = pd.read_csv(DATA / "universe_v29_kr_health.csv", dtype={"ticker": str})
    approved = set(universe["ticker"].astype(str).str.zfill(6))
    candidate = candidate[
        candidate["ticker"].isin(approved)
        & candidate["event_time_utc"].ge(pd.Timestamp("2025-08-20T00:00:00Z"))
    ].copy()
    candidate = exact.regular_session(candidate, "KR")
    candidate = exact.space_events(candidate)
    keep = set(candidate["event_id"].astype(str))
    return frame[frame["event_id"].astype(str).isin(keep)].copy().reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v35-input", type=Path, default=DEFAULT_V35)
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--output-dir", type=Path, default=POOL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Deterministic test limit; zero verifies every strict candidate")
    parser.add_argument("--candidate-only", action="store_true",
                        help="Pre-filter with outcome-blind V35 overlap, KR universe, session, retention, and 35m rules")
    parser.add_argument("--article-id", action="append", default=[], help="Verify only this 13-digit article ID (repeatable)")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.limit < 0:
        parser.error("workers/timeout must be positive and retries/limit must be non-negative")

    prospective = sorted(path for path in args.pool_dir.glob("naver_kr_*.csv.gz") if path.is_file())
    paths = ([args.v35_input] if args.v35_input.is_file() else []) + prospective
    if not paths:
        parser.error("no V35 or prospective Naver inputs found")
    strict, quarantine, counts = load_and_deduplicate(paths)
    all_strict_count = len(strict)
    if args.candidate_only:
        strict = outcome_blind_candidate_scope(strict)
    counts["candidate_scope_rows"] = len(strict)
    requested_ids = {str(value).strip() for value in args.article_id}
    if requested_ids:
        invalid_requested = sorted(value for value in requested_ids if not re.fullmatch(r"[0-9]{13}", value))
        if invalid_requested:
            parser.error(f"invalid --article-id values: {invalid_requested}")
        strict = strict.loc[strict["article_id"].isin(requested_ids)].copy()
        missing_requested = sorted(requested_ids - set(strict["article_id"]))
        if missing_requested:
            parser.error(f"requested article IDs not found among strict candidates: {missing_requested}")
    if args.limit:
        strict = strict.head(args.limit).copy()
    selected_count = len(strict)
    if selected_count == 0:
        parser.error("the selected verification scope contains zero strict candidates")

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, selected_count)) as pool:
        futures = {
            pool.submit(fetch_and_verify, row, args.timeout, args.retries): int(index)
            for index, row in strict.iterrows()
        }
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # Per-row quarantine; never lose the whole audit.
                results[index] = {"ok": False, "reason": "UNEXPECTED_ERROR", "detail": f"{type(exc).__name__}: {exc}"}
            if done % 100 == 0 or done == selected_count:
                print(f"[NAVER SECOND VERIFY] {done}/{selected_count}", flush=True)

    verified_rows: list[dict[str, Any]] = []
    for index, row in strict.iterrows():
        result = results[int(index)]
        if not result.get("ok"):
            quarantine.append(
                quarantine_entry(
                    row,
                    str(result.get("reason", "UNKNOWN_FAILURE")),
                    str(result.get("detail", "")),
                    http_status=result.get("http_status"),
                )
            )
            continue
        record = {column: str(row.get(column, "")) for column in SAFE_INPUT_COLUMNS}
        record["original_event_time_utc"] = record["event_time_utc"]
        record["event_time_utc"] = result["event_time_utc"]
        record["timestamp_quality"] = "EXACT_SECOND_NAVER_PAGE"
        record["timestamp_verification_source"] = "NAVER_ARTICLE_PAGE_DATA_DATE_TIME"
        record["publication_time_kst"] = result["publication_time_kst"]
        record["verified_office_id"] = result["verified_office_id"]
        record["verified_page_article_id"] = result["verified_page_article_id"]
        record["selection_uses_labels"] = False
        verified_rows.append(record)

    output_columns = list(SAFE_INPUT_COLUMNS) + [
        "original_event_time_utc",
        "timestamp_verification_source",
        "publication_time_kst",
        "verified_office_id",
        "verified_page_article_id",
        "selection_uses_labels",
    ]
    verified = pd.DataFrame(verified_rows, columns=output_columns)
    if not verified.empty:
        verified = verified.sort_values(["event_time_utc", "article_id", "event_id"], kind="stable").reset_index(drop=True)
    csv_payload = output_csv_bytes(verified)
    content_sha = hashlib.sha256(csv_payload).hexdigest()
    start_date, end_date = date_bounds(verified)
    scope = "ALL" if not requested_ids and not args.limit else f"LIMIT_{selected_count}_OF_{all_strict_count}"
    output_name = f"naver_verified_kr_{scope}_{start_date}_{end_date}_{content_sha[:12]}.csv.gz"
    output_path = args.output_dir / output_name
    write_gzip_deterministic(output_path, csv_payload)

    quarantine = sorted(
        quarantine,
        key=lambda item: (
            str(item.get("reason", "")),
            str(item.get("article_id", "")),
            str(item.get("event_id", "")),
            str(item.get("input_path", "")),
            int(item.get("input_row", -1)),
        ),
    )
    reason_counts: dict[str, int] = {}
    for item in quarantine:
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    quarantine_payload = {
        "contract": "NAVER_EXACT_SECOND_QUARANTINE_V1",
        "selection_uses_labels": False,
        "scope": scope,
        "candidate_only": bool(args.candidate_only),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": quarantine,
    }
    quarantine_bytes = canonical_json_bytes(quarantine_payload)
    quarantine_sha = hashlib.sha256(quarantine_bytes).hexdigest()
    quarantine_path = args.output_dir / f"naver_verified_kr_quarantine_{scope}_{quarantine_sha[:12]}.json"
    atomic_bytes(quarantine_path, quarantine_bytes)

    input_manifest = [
        {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in paths
    ]
    manifest = {
        "contract": "NAVER_EXACT_SECOND_VERIFICATION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_uses_labels": False,
        "loaded_columns": list(SAFE_INPUT_COLUMNS),
        "input_files": input_manifest,
        "scope": scope,
        "candidate_only": bool(args.candidate_only),
        "requested_article_ids": sorted(requested_ids),
        "counts": {
            **counts,
            "verification_scope_rows": selected_count,
            "verified_rows": len(verified),
            "page_quarantined_rows": selected_count - len(verified),
            "all_quarantine_rows_in_report": len(quarantine),
        },
        "verified_snapshot": {
            "path": str(output_path.resolve()),
            "content_sha256_uncompressed_csv": content_sha,
            "file_sha256": file_sha256(output_path),
            "bytes": output_path.stat().st_size,
            "timestamp_quality": "EXACT_SECOND_NAVER_PAGE",
        },
        "quarantine_report": {
            "path": str(quarantine_path.resolve()),
            "content_sha256": quarantine_sha,
            "file_sha256": file_sha256(quarantine_path),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = args.output_dir / f"naver_verified_kr_manifest_{scope}_{content_sha[:12]}.json"
    atomic_bytes(manifest_path, manifest_bytes)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
