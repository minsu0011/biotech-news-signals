"""Build an outcome-blind historical US/KR news event inventory.

This module deliberately does not import model code or inspect labels, returns,
predictions, Research Seal, Final Meta, certificates, or V224 prospective data.
Only explicitly allow-listed event metadata columns participate in membership,
deduplication, and ordering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import runtime_limits
except ImportError:  # pragma: no cover - only relevant when copied elsewhere
    runtime_limits = None


SCRIPT_VERSION = "HIST_NEWS_EVENT_INVENTORY_V1"
DEFAULT_OUTPUT_REL = Path("data/historical_news_backfill")
DEFAULT_STATE_REL = Path("state/HIST_NEWS_BACKFILL_STATE.json")

# Reading is restricted to these metadata fields. In particular, no outcome,
# return, correctness, prediction, strategy, role, or split field is present.
ALLOWED_INPUT_COLUMNS = (
    "event_id",
    "event_group_id",
    "market",
    "ticker",
    "company",
    "event_time_utc",
    "source",
    "source_family",
    "form",
    "event_type",
    "timestamp_quality",
    "url",
    "article_id",
    "origin_file",
    "selection_uses_labels",
)

FORBIDDEN_COLUMN_NAMES = {
    "y",
    "label",
    "target",
    "fwd_ret",
    "fwd_return",
    "forward_return",
    "prediction",
    "prediction_correct",
    "correctness",
    "strategy_return",
    "p_correct",
    "role",
    "fold",
}
FORBIDDEN_PATH_TOKENS = (
    "cert",
    "research_seal",
    "final_meta",
    "v224_prospective",
    "prospective_confirmation",
)
FORBIDDEN_PROVENANCE_TOKENS = (
    "seal",
    "final_meta",
    "prospective",
    "cert",
)
ALLOWED_SOURCE_FAMILIES = {"US_NEWS", "KR_NEWS"}
MARKET_TIMEZONES = {"US": "America/New_York", "KR": "Asia/Seoul"}

FIXED_EVENT_SOURCES = (
    ("data/dev_contract_v36_events.csv.gz", "EXACT_DEV_CONTRACT"),
    ("data/events_us_news_real.csv", "CURATED_LOCAL"),
    ("data/events_naver_news_real.csv", "CURATED_LOCAL"),
    ("data/events_google_news_v33_raw.csv", "LOCAL_RAW_TABLE"),
    ("data/events_google_us_v33_raw.csv", "LOCAL_RAW_TABLE"),
    ("data/events_google_us_v34_raw.csv", "LOCAL_RAW_TABLE"),
    ("data/events_google_kr_v34_raw.csv", "LOCAL_RAW_TABLE"),
    ("data/events_naver_kr_v35_raw.csv", "LOCAL_RAW_TABLE"),
)

LOCAL_RAW_AUDIT_DIRS = (
    ("cache/sources/QiuStockNews/json_files", "REUTERS_QIU_RAW"),
    ("cache/naver_news_by_ticker", "NAVER_NEWS_RAW"),
    ("cache/kind_days", "KIND_STRUCTURED_RAW"),
    ("cache/sec_events_by_ticker", "SEC_STRUCTURED_RAW"),
    ("data/untouched_pool", "UNTOUCHED_NEWS_POOL"),
)

OUTPUT_COLUMNS = (
    "canonical_event_id",
    "existing_event_id",
    "event_group_id",
    "market",
    "ticker",
    "cik",
    "issuer_id",
    "issuer_name",
    "event_time_utc",
    "event_time_local",
    "event_timezone",
    "decision_cutoff_utc",
    "source_family",
    "source_provider",
    "source_file",
    "source_file_sha256",
    "source_row_number",
    "source_row_sha256",
    "existing_article_id",
    "existing_url",
    "event_type",
    "timestamp_quality",
    "exact_price_eligible",
    "inventory_priority_year",
    "inventory_priority_rank",
    "join_key_event_id",
    "join_key_market_ticker_event_time",
    "provenance_count",
    "provenance_json",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_forbidden_path(path: Path, root: Path) -> bool:
    try:
        rel = relative_posix(path, root).casefold()
    except ValueError:
        return True
    parts = rel.replace("-", "_").split("/")
    return any(token in part for part in parts for token in FORBIDDEN_PATH_TOKENS)


def has_forbidden_provenance(*values: Any) -> bool:
    text = " ".join("" if pd.isna(v) else str(v) for v in values).casefold()
    return any(token in text for token in FORBIDDEN_PROVENANCE_TOKENS)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False, engine="pyarrow", compression="zstd")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def merge_inventory_state(state_path: Path, update: dict[str, Any]) -> None:
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state = loaded
        except (OSError, json.JSONDecodeError):
            # The inventory owns only its namespace; a malformed shared state is
            # not overwritten because that could erase another worker's state.
            raise RuntimeError(f"shared state is not valid JSON: {state_path}")
    prior = state.get("event_inventory")
    if not isinstance(prior, dict):
        prior = {}
    prior.update(update)
    state["event_inventory"] = prior
    atomic_write_json(state_path, state)


def discover_event_sources(root: Path) -> list[tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    for rel, source_role in FIXED_EVENT_SOURCES:
        path = root / rel
        if path.is_file() and not is_forbidden_path(path, root):
            found[relative_posix(path, root)] = (path, source_role)
    untouched = root / "data/untouched_pool"
    if untouched.is_dir():
        for pattern in ("gnews_us_*.csv.gz", "gnews_kr_*.csv.gz"):
            for path in untouched.glob(pattern):
                lowered = path.name.casefold()
                if any(token in lowered for token in ("audit", "quarantine")):
                    continue
                if not is_forbidden_path(path, root):
                    found[relative_posix(path, root)] = (path, "UNTOUCHED_LABEL_BLIND_POOL")
    return [found[key] for key in sorted(found)]


def quick_source_manifest(sources: Iterable[tuple[Path, str]], root: Path) -> list[dict[str, Any]]:
    result = []
    for path, source_role in sources:
        stat = path.stat()
        result.append(
            {
                "path": relative_posix(path, root),
                "source_role": source_role,
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return result


def audit_local_raw_sources(root: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for rel, provider in LOCAL_RAW_AUDIT_DIRS:
        directory = root / rel
        if not directory.is_dir() or is_forbidden_path(directory, root):
            continue
        file_count = 0
        total_bytes = 0
        extensions: Counter[str] = Counter()
        snapshot_rows: list[tuple[str, int, int]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or is_forbidden_path(path, root):
                continue
            lowered = path.as_posix().casefold()
            if any(token in lowered for token in ("research_seal", "final_meta", "v224_prospective")):
                continue
            stat = path.stat()
            file_count += 1
            total_bytes += int(stat.st_size)
            extensions[path.suffix.casefold() or "<none>"] += 1
            snapshot_rows.append((relative_posix(path, root), int(stat.st_size), int(stat.st_mtime_ns)))
        audits.append(
            {
                "provider": provider,
                "path": rel,
                "files": file_count,
                "bytes": total_bytes,
                "extensions": dict(sorted(extensions.items())),
                "metadata_snapshot_sha256": sha256_json(snapshot_rows),
                "content_inspected": False,
            }
        )
    return audits


def safe_csv_metadata(path: Path) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in ALLOWED_INPUT_COLUMNS if column in header]
    lowered = {column.casefold() for column in usecols}
    forbidden = lowered & FORBIDDEN_COLUMN_NAMES
    if forbidden:
        raise RuntimeError(f"forbidden columns requested from {path}: {sorted(forbidden)}")
    required = {"event_id", "market", "ticker", "event_time_utc"}
    if not required.issubset(usecols):
        raise RuntimeError(f"missing required metadata columns in {path}: {sorted(required - set(usecols))}")
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    return frame, header


def clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def canonical_event_id(row: pd.Series) -> str:
    event_id = clean_string(row.get("event_id"))
    if event_id:
        return event_id
    identity = {
        "market": clean_string(row.get("market")).upper(),
        "ticker": clean_string(row.get("ticker")).upper(),
        "event_time_utc": str(row.get("event_time_utc")),
        "article_id": clean_string(row.get("article_id")),
        "url": clean_string(row.get("url")),
    }
    return f"HIST_NEWS:{sha256_json(identity)[:32]}"


def event_local_iso(timestamp: pd.Timestamp, market: str) -> tuple[str, str]:
    timezone_name = MARKET_TIMEZONES[market]
    local = timestamp.to_pydatetime().astimezone(ZoneInfo(timezone_name))
    return local.isoformat(), timezone_name


def _label_blind_pool_is_safe(frame: pd.DataFrame, source_path: str) -> None:
    if "selection_uses_labels" not in frame:
        return
    values = frame["selection_uses_labels"].dropna().astype(str).str.strip().str.casefold()
    unsafe = ~values.isin({"false", "0", "no", "n"})
    if bool(unsafe.any()):
        raise RuntimeError(f"fail-closed: outcome-aware selection declared by {source_path}")


def normalize_source(
    frame: pd.DataFrame,
    *,
    path: Path,
    root: Path,
    source_role: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rel = relative_posix(path, root)
    _label_blind_pool_is_safe(frame, rel)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    for index, row in frame.iterrows():
        market = clean_string(row.get("market")).upper()
        if market not in MARKET_TIMEZONES:
            rejected["MARKET_NOT_US_OR_KR"] += 1
            continue
        explicit_family = clean_string(row.get("source_family")).upper()
        family = explicit_family or ("US_NEWS" if market == "US" else "KR_NEWS")
        if family not in ALLOWED_SOURCE_FAMILIES:
            rejected["NOT_NEWS_SOURCE_FAMILY"] += 1
            continue
        source_provider = clean_string(row.get("source"))
        origin_file = clean_string(row.get("origin_file"))
        if has_forbidden_provenance(source_provider, origin_file, rel):
            rejected["FORBIDDEN_PROVENANCE"] += 1
            continue
        event_time = pd.to_datetime(row.get("event_time_utc"), utc=True, errors="coerce")
        if pd.isna(event_time):
            rejected["INVALID_EVENT_TIME"] += 1
            continue
        ticker = clean_string(row.get("ticker")).upper()
        if not ticker:
            rejected["MISSING_TICKER"] += 1
            continue
        event_id = canonical_event_id(row)
        local_iso, timezone_name = event_local_iso(event_time, market)
        row_identity = {
            "event_id": event_id,
            "market": market,
            "ticker": ticker,
            "event_time_utc": event_time.isoformat(),
            "source": source_provider,
            "article_id": clean_string(row.get("article_id")),
            "url": clean_string(row.get("url")),
        }
        source_row_number = int(index) + 2
        record = {
            "canonical_event_id": event_id,
            "existing_event_id": event_id,
            "event_group_id": clean_string(row.get("event_group_id")) or event_id,
            "market": market,
            "ticker": ticker,
            "cik": None,
            "issuer_id": f"{market}:{ticker}",
            "issuer_name": clean_string(row.get("company")) or None,
            "event_time_utc": event_time,
            "event_time_local": local_iso,
            "event_timezone": timezone_name,
            "decision_cutoff_utc": event_time + pd.Timedelta(minutes=2),
            "source_family": family,
            "source_provider": source_provider or None,
            "source_file": rel,
            "source_file_sha256": source_sha256,
            "source_row_number": source_row_number,
            "source_row_sha256": sha256_json(row_identity),
            "existing_article_id": clean_string(row.get("article_id")) or None,
            "existing_url": clean_string(row.get("url")) or None,
            "event_type": clean_string(row.get("event_type")) or None,
            "timestamp_quality": clean_string(row.get("timestamp_quality")) or None,
            "exact_price_eligible": source_role == "EXACT_DEV_CONTRACT",
            "inventory_priority_year": int(event_time.year),
            "join_key_event_id": event_id,
            "join_key_market_ticker_event_time": (
                f"{market}|{ticker}|{event_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"
            ),
            "_source_role": source_role,
        }
        accepted.append(record)
    return accepted, {
        "path": rel,
        "source_role": source_role,
        "source_file_sha256": source_sha256,
        "bytes": int(path.stat().st_size),
        "rows_scanned": int(len(frame)),
        "rows_accepted": int(len(accepted)),
        "rows_rejected": int(len(frame) - len(accepted)),
        "rejection_reasons": dict(sorted(rejected.items())),
    }


def collapse_candidates(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not records:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, {"duplicate_records_collapsed": 0, "canonical_id_conflicts": 0}

    # A provider event ID is not always a unique event-to-issuer key: syndicated
    # articles can be mapped to several tickers or timestamps. Preserve every
    # mapping and disambiguate only those collided canonical IDs. The raw ID is
    # retained separately for exact downstream joins.
    raw_join_keys: dict[str, set[str]] = defaultdict(set)
    for record in records:
        raw_join_keys[record["existing_event_id"]].add(record["join_key_market_ticker_event_time"])
    collided_raw_ids = {event_id for event_id, keys in raw_join_keys.items() if len(keys) > 1}
    for record in records:
        raw_id = record["existing_event_id"]
        if raw_id in collided_raw_ids:
            suffix = hashlib.sha256(record["join_key_market_ticker_event_time"].encode("utf-8")).hexdigest()[:16]
            record["canonical_event_id"] = f"{raw_id}::{suffix}"

    provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_join_keys: dict[str, set[str]] = defaultdict(set)
    exact_ids: set[str] = set()
    for record in records:
        event_id = record["canonical_event_id"]
        provenance[event_id].append(
            {
                "source_file": record["source_file"],
                "source_file_sha256": record["source_file_sha256"],
                "source_row_number": record["source_row_number"],
                "source_row_sha256": record["source_row_sha256"],
                "source_role": record["_source_role"],
            }
        )
        canonical_join_keys[event_id].add(record["join_key_market_ticker_event_time"])
        if record["exact_price_eligible"]:
            exact_ids.add(event_id)

    frame = pd.DataFrame(records)
    role_rank = {
        "EXACT_DEV_CONTRACT": 0,
        "CURATED_LOCAL": 1,
        "LOCAL_RAW_TABLE": 2,
        "UNTOUCHED_LABEL_BLIND_POOL": 3,
    }
    frame["_role_rank"] = frame["_source_role"].map(role_rank).fillna(9).astype(int)
    frame["_metadata_missing"] = (
        frame[["existing_article_id", "existing_url", "issuer_name", "timestamp_quality"]]
        .isna()
        .sum(axis=1)
    )
    frame = frame.sort_values(
        ["canonical_event_id", "_role_rank", "_metadata_missing", "source_file", "source_row_number"],
        kind="mergesort",
    ).drop_duplicates("canonical_event_id", keep="first")
    frame["exact_price_eligible"] = frame["canonical_event_id"].isin(exact_ids)
    frame["provenance_count"] = frame["canonical_event_id"].map(lambda value: len(provenance[value]))
    frame["provenance_json"] = frame["canonical_event_id"].map(
        lambda value: json.dumps(
            sorted(provenance[value], key=lambda item: (item["source_file"], item["source_row_number"])),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    # Recent-to-old is the primary order. All tie-breaks are source metadata;
    # no label, return, or prediction is available to this function.
    frame = frame.sort_values(
        [
            "inventory_priority_year",
            "event_time_utc",
            "exact_price_eligible",
            "market",
            "ticker",
            "canonical_event_id",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["inventory_priority_rank"] = pd.Series(range(1, len(frame) + 1), dtype="int64")
    frame = frame.drop(columns=["_source_role", "_role_rank", "_metadata_missing"])
    frame = frame.loc[:, OUTPUT_COLUMNS]
    return frame, {
        "duplicate_records_collapsed": int(len(records) - len(frame)),
        "raw_event_id_multi_join_conflicts_resolved": int(len(collided_raw_ids)),
        "canonical_id_conflicts": int(sum(len(values) > 1 for values in canonical_join_keys.values())),
    }


def build_markdown_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Historical News Event Inventory Audit",
        "",
        f"- Script: `{audit['script_version']}`",
        f"- Generated UTC: `{audit['generated_at_utc']}`",
        f"- Outcome-blind gate: **{audit['gates']['outcome_blind_membership']}**",
        f"- Protected namespace isolation: **{audit['gates']['protected_namespaces_excluded']}**",
        f"- Total canonical events: **{audit['events_total']:,}**",
        f"- Exact price eligible: **{audit['exact_price_eligible']:,}**",
        "",
        "## Coverage",
        "",
    ]
    for market, count in audit["market_counts"].items():
        lines.append(f"- {market}: {count:,}")
    lines.extend(["", "## Year coverage", ""])
    for year, count in audit["year_counts"].items():
        lines.append(f"- {year}: {count:,}")
    lines.extend(["", "## Input sources", ""])
    for source in audit["event_sources"]:
        lines.append(
            f"- `{source['path']}`: {source['rows_accepted']:,}/{source['rows_scanned']:,} rows accepted"
        )
    lines.extend(
        [
            "",
            "## Leakage boundary",
            "",
            "Membership and priority use only event identity, market, ticker, publication/event time,",
            "source provenance, article identifiers, and exact-contract availability. No outcome,",
            "forward return, prediction correctness, strategy return, role, fold, Research Seal,",
            "Final Meta, certificate, or V224 prospective content is read.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_inventory(
    frame: pd.DataFrame,
    *,
    source_audits: list[dict[str, Any]],
    raw_audits: list[dict[str, Any]],
    collapse_audit: dict[str, Any],
    input_digest: str,
    runtime_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "input_digest_sha256": input_digest,
        "events_total": int(len(frame)),
        "market_counts": {str(k): int(v) for k, v in frame["market"].value_counts().sort_index().items()},
        "source_family_counts": {
            str(k): int(v) for k, v in frame["source_family"].value_counts().sort_index().items()
        },
        "year_counts": {
            str(int(k)): int(v)
            for k, v in frame["inventory_priority_year"].value_counts().sort_index(ascending=False).items()
        },
        "exact_price_eligible": int(frame["exact_price_eligible"].sum()),
        "event_time_min_utc": frame["event_time_utc"].min().isoformat() if len(frame) else None,
        "event_time_max_utc": frame["event_time_utc"].max().isoformat() if len(frame) else None,
        "event_sources": source_audits,
        "local_raw_source_audit": raw_audits,
        "deduplication": collapse_audit,
        "output_schema": list(OUTPUT_COLUMNS),
        "runtime_limits": runtime_status,
        "gates": {
            "outcome_blind_membership": "PASS",
            "protected_namespaces_excluded": "PASS",
            "seal_provenance_rows_excluded": "PASS",
            "recent_to_old_priority": "PASS",
            "model_training_performed": False,
        },
    }


def build_inventory(
    root: Path,
    output_dir: Path,
    state_path: Path,
    *,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    state_path = state_path.resolve()
    event_master_path = output_dir / "EVENT_MASTER.parquet"
    audit_path = output_dir / "EVENT_INVENTORY_AUDIT.json"
    report_path = output_dir / "EVENT_INVENTORY_REPORT.md"
    manifest_path = output_dir / "EVENT_INVENTORY_SOURCE_MANIFEST.csv"

    sources = discover_event_sources(root)
    if not sources:
        raise RuntimeError("no allow-listed historical news sources were found")
    quick_manifest = quick_source_manifest(sources, root)
    quick_digest = sha256_json(quick_manifest)

    prior_inventory: dict[str, Any] = {}
    if state_path.exists():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("event_inventory"), dict):
            prior_inventory = loaded["event_inventory"]
    if resume and not force and prior_inventory.get("source_quick_digest_sha256") == quick_digest:
        prior_output_sha = prior_inventory.get("event_master_sha256")
        if event_master_path.is_file() and prior_output_sha == sha256_file(event_master_path):
            update = {
                "status": "NO_INPUT_CHANGE",
                "script_version": SCRIPT_VERSION,
                "last_checked_utc": utc_now_iso(),
                "source_quick_digest_sha256": quick_digest,
            }
            merge_inventory_state(state_path, update)
            return {"action": "NO_INPUT_CHANGE", "event_master": str(event_master_path), **update}

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_audits = audit_local_raw_sources(root)
    source_audits: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    content_manifest: list[dict[str, Any]] = []
    for path, source_role in sources:
        source_sha = sha256_file(path)
        frame, _header = safe_csv_metadata(path)
        normalized, source_audit = normalize_source(
            frame,
            path=path,
            root=root,
            source_role=source_role,
            source_sha256=source_sha,
        )
        records.extend(normalized)
        source_audits.append(source_audit)
        content_manifest.append(
            {
                "path": relative_posix(path, root),
                "source_role": source_role,
                "sha256": source_sha,
                "bytes": int(path.stat().st_size),
            }
        )

    event_master, collapse_audit = collapse_candidates(records)
    if event_master.empty:
        raise RuntimeError("no eligible US_NEWS/KR_NEWS events survived leakage guards")
    input_digest = sha256_json(content_manifest)
    runtime_status = runtime_limits.status() if runtime_limits is not None else {"available": False}
    audit = summarize_inventory(
        event_master,
        source_audits=source_audits,
        raw_audits=raw_audits,
        collapse_audit=collapse_audit,
        input_digest=input_digest,
        runtime_status=runtime_status,
    )

    atomic_write_parquet(event_master_path, event_master)
    event_master_sha = sha256_file(event_master_path)
    audit["event_master_sha256"] = event_master_sha
    audit["event_master_path"] = relative_posix(event_master_path, root)
    atomic_write_json(audit_path, audit)
    atomic_write_text(report_path, build_markdown_report(audit))

    manifest_rows = [
        {
            "source_file": item["path"],
            "source_role": item["source_role"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in content_manifest
    ]
    manifest_text_buffer: list[str] = []
    if manifest_rows:
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["source_file", "source_role", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        manifest_text_buffer.append(buffer.getvalue())
    atomic_write_text(manifest_path, "".join(manifest_text_buffer))

    update = {
        "status": "COMPLETE",
        "script_version": SCRIPT_VERSION,
        "last_checkpoint": utc_now_iso(),
        "last_checked_utc": utc_now_iso(),
        "events_total": int(len(event_master)),
        "events_processed": int(len(event_master)),
        "source_quick_digest_sha256": quick_digest,
        "input_digest_sha256": input_digest,
        "event_master_path": relative_posix(event_master_path, root),
        "event_master_sha256": event_master_sha,
        "audit_path": relative_posix(audit_path, root),
        "report_path": relative_posix(report_path, root),
        "manifest_path": relative_posix(manifest_path, root),
        "protected_namespaces_touched": [],
        "outcome_fields_read": [],
    }
    merge_inventory_state(state_path, update)
    return {
        "action": "BUILT",
        "event_master": str(event_master_path),
        "event_master_sha256": event_master_sha,
        "events_total": int(len(event_master)),
        "market_counts": audit["market_counts"],
        "exact_price_eligible": audit["exact_price_eligible"],
        "audit": str(audit_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "state": str(state_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / DEFAULT_OUTPUT_REL)).resolve()
    state_path = (args.state or (root / DEFAULT_STATE_REL)).resolve()
    try:
        result = build_inventory(
            root,
            output_dir,
            state_path,
            resume=bool(args.resume),
            force=bool(args.force),
        )
    except Exception as exc:
        try:
            merge_inventory_state(
                state_path,
                {
                    "status": "FAILED",
                    "script_version": SCRIPT_VERSION,
                    "last_checkpoint": utc_now_iso(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
