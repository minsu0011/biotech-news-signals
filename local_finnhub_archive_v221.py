"""Parity-gated extraction from the pinned 67GB Finnhub/HF minute archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import alpaca_backfill_v221 as alpaca_backfill
import alpaca_provider_acquisition as alpaca
import historical_symbol_resolver as historical
import local_finnhub_fallback_v221 as parity_core
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
RAW = ROOT / "cache" / "hf_months"
CACHE = ROOT / "price_data" / "local_finnhub_archive_1m"
POLICY_PATH = DATA / "LOCAL_FINNHUB_ARCHIVE_PROVIDER_PARITY_POLICY_V1.json"
SEMANTICS_PATH = DATA / "BAR_SEMANTICS_AUDIT_V36.json"
US_QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
HISTORICAL_QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
NO_BAR_PATH = DATA / "HISTORICAL_NO_BAR_CLASSIFICATION.parquet"
SAMPLE_PATH = RESEARCH / "LOCAL_FINNHUB_ARCHIVE_PARITY_SAMPLE.csv"
DETAIL_PATH = RESEARCH / "LOCAL_FINNHUB_ARCHIVE_PARITY_DETAIL.csv"
SUMMARY_PATH = RESEARCH / "LOCAL_FINNHUB_ARCHIVE_PARITY_SUMMARY.json"
BACKFILL_REPORT_PATH = RESEARCH / "LOCAL_FINNHUB_ARCHIVE_BACKFILL_REPORT.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "policy_name": "MASSIVE_ALPACA_LOCAL_FINNHUB_ARCHIVE_EXACT_T2_30M_PARITY_V1",
        "frozen_before_archive_candidate_results": True,
        "selection_uses_outcomes": False,
        "eligible_role_for_label_level_parity": "DEV_EXTENSION",
        "eligible_source_family_for_parity": "US_SEC",
        "candidate_provider": "LOCAL_FINNHUB_ARCHIVE_1M",
        "provider_choice_uses_outcomes": False,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("LOCAL_FINNHUB_ARCHIVE_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


def verify_semantics() -> str:
    return parity_core.verify_semantics()


def raw_month_path(month: str) -> Path:
    if not str(month).replace("-", "").isdigit() or len(str(month)) != 7:
        raise ValueError("ARCHIVE_MONTH_REJECTED")
    return RAW / f"ohlcv_{month}.parquet"


def slice_path(symbol: str, date: str) -> Path:
    value = str(symbol).upper().strip()
    if not value or not value.replace(".", "").replace("-", "").isalnum():
        raise ValueError("ARCHIVE_SYMBOL_REJECTED")
    return CACHE / str(date)[:4] / value / f"{date}.parquet"


def read_month_symbols(path: Path, symbols: set[str]) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
    if not path.is_file() or not symbols:
        return pd.DataFrame(columns=columns)
    wanted = sorted({str(value).upper() for value in symbols})
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("ticker", "in", wanted)])
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame.ticker.astype(str).isin(wanted)].copy()
    if frame.empty:
        return frame
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame.sort_values(["ticker", "timestamp"]).drop_duplicates(["ticker", "timestamp"])


def event_window(month_frame: pd.DataFrame, row: Any) -> pd.DataFrame:
    start = pd.Timestamp(row.required_start_utc)
    end = pd.Timestamp(row.required_end_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    symbol = str(row.provider_symbol).upper()
    frame = month_frame.loc[
        month_frame.ticker.eq(symbol)
        & month_frame.timestamp.ge(start)
        & month_frame.timestamp.le(end),
        ["timestamp", "open", "high", "low", "close", "volume"],
    ].copy()
    frame["datetime"] = pd.to_datetime(frame.timestamp, utc=True)
    frame.attrs.update({
        "price_source": "LOCAL_FINNHUB_ARCHIVE_1M",
        "price_cadence_sec": 60,
        "bar_timestamp_semantics": "OPEN",
        "official_exact_eligible": True,
    })
    return frame


def save_slice(frame: pd.DataFrame, symbol: str, date: str) -> tuple[Path, str]:
    path = slice_path(symbol, date)
    stored = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    atomic_parquet(stored, path)
    return path, sha256(path)


def source_frame() -> pd.DataFrame:
    queue = pd.read_parquet(US_QUEUE_PATH)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "labels_opened"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if frame.role.isna().any():
        raise RuntimeError("ARCHIVE_PARITY_EVENT_WITHOUT_FROZEN_ROLE")
    if frame.loc[frame.role.ne("DEV_EXTENSION"), "labels_opened"].fillna(False).astype(bool).any():
        raise RuntimeError("SEALED_ROLE_OPEN_STATE_UNEXPECTED")
    frame = frame.loc[
        frame.role.astype(str).eq("DEV_EXTENSION")
        & frame.source_family.astype(str).eq("US_SEC")
    ].copy()
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["trade_date"] = frame.event_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    frame["month"] = frame.event_time_utc.dt.strftime("%Y-%m")
    historical_queue = pd.read_parquet(HISTORICAL_QUEUE_PATH)
    historical_queue = historical_queue.loc[
        historical_queue.resolution_confidence.astype(str).isin({"A", "B"})
        & historical_queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
        & historical_queue.resolved_ticker.astype(str).ne("")
    ]
    resolved = historical_queue.set_index("event_id").resolved_ticker.astype(str).to_dict()
    frame["provider_symbol"] = [
        resolved.get(str(event_id), str(ticker))
        for event_id, ticker in zip(frame.event_id, frame.ticker)
    ]
    frame["selection_hash"] = frame.event_id.astype(str).map(
        lambda value: hashlib.sha256(
            f"V221_LOCAL_FINNHUB_ARCHIVE_PARITY_V1|{value}".encode()
        ).hexdigest()
    )
    return frame


def build_parity_sample(max_events: int) -> pd.DataFrame:
    policy, _ = load_policy()
    frame = source_frame()
    year_rank = {int(value): rank for rank, value in enumerate(policy["priority_years"])}
    frame["_year_rank"] = frame.event_time_utc.dt.year.map(
        lambda value: year_rank.get(int(value), len(year_rank))
    )
    frame = frame.loc[frame.event_time_utc.le(pd.Timestamp("2025-05-31 23:59:59Z"))]
    frame = frame.sort_values(["_year_rank", "selection_hash", "event_time_utc"])
    rows: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    raw_hashes: dict[str, str] = {}
    for month, group in frame.groupby("month", sort=False):
        raw_path = raw_month_path(str(month))
        if not raw_path.is_file():
            continue
        month_bars = read_month_symbols(raw_path, set(group.provider_symbol.astype(str)))
        if month_bars.empty:
            continue
        raw_hash = raw_hashes.setdefault(str(raw_path), sha256(raw_path))
        for row in group.itertuples(index=False):
            key = (str(row.provider_symbol), str(row.trade_date))
            if key in used:
                continue
            bars = event_window(month_bars, row)
            if bars.empty or not alpaca.bar_qa(bars).get("pass"):
                continue
            exact, _, _ = alpaca_backfill.timing_only_eligibility(
                pd.Timestamp(row.event_time_utc), bars.timestamp
            )
            if not exact:
                continue
            cache_path, cache_hash = save_slice(bars, str(row.provider_symbol), str(row.trade_date))
            values = row._asdict()
            values.update({
                "reference_provider": "ALPACA_SIP",
                "reference_raw_file": "",
                "candidate_provider": "LOCAL_FINNHUB_ARCHIVE_1M",
                "candidate_raw_file": str(cache_path.relative_to(ROOT)),
                "candidate_raw_sha256": cache_hash,
                "source_archive_file": str(raw_path.relative_to(ROOT)),
                "source_archive_sha256": raw_hash,
            })
            rows.append(values)
            used.add(key)
            if len(rows) >= max_events:
                break
        if len(rows) >= max_events:
            break
    sample = pd.DataFrame(rows)
    if not sample.empty:
        columns = [
            "event_id", "ticker", "provider_symbol", "issuer", "event_time_utc",
            "required_start_utc", "required_end_utc", "source_family", "role",
            "trade_date", "reference_provider", "reference_raw_file",
            "candidate_provider", "candidate_raw_file", "candidate_raw_sha256",
            "source_archive_file", "source_archive_sha256", "selection_hash",
        ]
        sample = sample[columns]
    atomic_csv(sample, SAMPLE_PATH)
    return sample


def failed_reference_row(sample: pd.Series, error: str) -> dict[str, Any]:
    row = parity_core.failed_reference_row(sample, error)
    row["candidate_provider"] = "LOCAL_FINNHUB_ARCHIVE_1M"
    return row


def run_parity(max_events: int = 120, max_requests: int | None = None) -> dict[str, Any]:
    policy, policy_hash = load_policy()
    semantics_hash = verify_semantics()
    sample = build_parity_sample(max_events)
    # The sample and all source hashes are frozen before any reference outcome is opened.
    frozen_sample_hash = sha256(SAMPLE_PATH)
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = alpaca_backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    client = alpaca.AlpacaClient()
    requests_used = 0
    detail_rows: list[dict[str, Any]] = []
    updated = sample.copy()
    for position, row in enumerate(sample.itertuples(index=False)):
        if max_requests is not None and requests_used >= max_requests:
            break
        sample_row = pd.Series(row._asdict())
        reference_path = provider._cache_path(
            provider.ALPACA_CACHE / "sip", str(row.provider_symbol), row.trade_date
        )
        reference = pd.DataFrame()
        status: dict[str, Any] = {"ok": True, "requests": 0}
        if reference_path.is_file():
            reference = parity_core.load_window(
                reference_path, row.required_start_utc, row.required_end_utc, "ALPACA_SIP"
            )
        else:
            reference, status = client.get_bars(
                str(row.provider_symbol), pd.Timestamp(row.required_start_utc),
                pd.Timestamp(row.required_end_utc), "sip"
            )
            requests_used += int(status.get("requests", 0))
            if status.get("ok") and not reference.empty and alpaca.bar_qa(reference).get("pass"):
                alpaca.atomic_parquet(reference, reference_path)
                reference = parity_core.load_window(
                    reference_path, row.required_start_utc, row.required_end_utc, "ALPACA_SIP"
                )
        if not status.get("ok") or reference.empty or not alpaca.bar_qa(reference).get("pass"):
            detail_rows.append(failed_reference_row(
                sample_row, str(status.get("error") or "EMPTY_OR_QA_FAIL")
            ))
            continue
        updated.loc[position, "reference_raw_file"] = str(reference_path.relative_to(ROOT))
        candidate = parity_core.load_window(
            ROOT / row.candidate_raw_file, row.required_start_utc,
            row.required_end_utc, "LOCAL_FINNHUB_ARCHIVE_1M",
        )
        result = parity_core.compare_pair(sample_row, reference, candidate)
        result["candidate_provider"] = "LOCAL_FINNHUB_ARCHIVE_1M"
        detail_rows.append(result)
    atomic_csv(updated, SAMPLE_PATH)
    detail = pd.DataFrame(detail_rows)
    atomic_csv(detail, DETAIL_PATH)
    result = parity_core.summarize(detail, policy)
    result["provider"] = "LOCAL_FINNHUB_ARCHIVE_1M"
    summary = {
        "schema_version": 1,
        "audited_at": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "semantics_audit_path": str(SEMANTICS_PATH.relative_to(ROOT)),
        "semantics_audit_sha256": semantics_hash,
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "sample_frozen_before_reference_sha256": frozen_sample_hash,
        "sample_final_sha256": sha256(SAMPLE_PATH),
        "detail_path": str(DETAIL_PATH.relative_to(ROOT)),
        "detail_sha256": sha256(DETAIL_PATH),
        "selection_uses_outcomes": False,
        "sample_selection_uses_timestamp_availability_only": True,
        "label_level_parity_role": "DEV_EXTENSION",
        "label_level_parity_source_family": "US_SEC",
        "sealed_role_labels_opened": False,
        "reference_network_requests": int(requests_used),
        "result": result,
        "backfill_authorized": result["status"] == "PASS",
    }
    atomic_json(summary, SUMMARY_PATH)
    return summary


def authorize_backfill() -> tuple[dict[str, Any], str]:
    policy, policy_hash = load_policy()
    verify_semantics()
    if not SUMMARY_PATH.is_file():
        raise RuntimeError("LOCAL_FINNHUB_ARCHIVE_PARITY_MISSING")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("policy_sha256") != policy_hash:
        raise RuntimeError("LOCAL_FINNHUB_ARCHIVE_PARITY_POLICY_HASH_MISMATCH")
    if summary.get("backfill_authorized") is not True:
        raise RuntimeError("LOCAL_FINNHUB_ARCHIVE_PARITY_FAIL")
    if summary.get("sealed_role_labels_opened") is not False:
        raise RuntimeError("SEALED_ROLE_LABEL_GUARD_FAIL")
    return policy, policy_hash


def ensure_columns(queue: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "local_archive_backfill_status": "NOT_ATTEMPTED",
        "local_archive_policy_sha256": "",
        "local_archive_attempt_count": 0,
        "local_archive_source_file": "",
        "local_archive_source_sha256": "",
    }
    for column, default in defaults.items():
        if column not in queue:
            queue[column] = default
    return queue


def run_backfill(years: set[int] | None = None) -> dict[str, Any]:
    policy, policy_hash = authorize_backfill()
    queue = ensure_columns(pd.read_parquet(HISTORICAL_QUEUE_PATH))
    audit = pd.read_parquet(NO_BAR_PATH)
    if audit.selection_uses_labels.astype(bool).any():
        raise RuntimeError("NO_BAR_AUDIT_LABEL_GUARD_FAIL")
    eligible_ids = set(
        audit.loc[audit.high_confidence_fallback_candidate.astype(bool), "event_id"].astype(str)
    )
    event_time = pd.to_datetime(queue.event_time_utc, utc=True)
    event_year = event_time.dt.year
    pending = queue.event_id.astype(str).isin(eligible_ids)
    pending &= queue.role.astype(str).eq("DEV_EXTENSION")
    pending &= queue.source_family.astype(str).eq("US_SEC")
    pending &= queue.fetch_status.astype(str).eq("ALPACA_NO_BAR")
    pending &= queue.resolution_confidence.astype(str).isin({"A", "B"})
    pending &= queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
    pending &= queue.resolved_ticker.astype(str).ne("")
    pending &= pd.to_numeric(queue.local_archive_attempt_count, errors="coerce").fillna(0).eq(0)
    pending &= event_time.le(pd.Timestamp("2025-05-31 23:59:59Z"))
    if years is not None:
        pending &= event_year.isin(years)
    candidates = queue.loc[pending].copy()
    candidates["provider_symbol"] = candidates.resolved_ticker.astype(str)
    candidates["month"] = event_time.loc[candidates.index].dt.strftime("%Y-%m")
    priority = {int(value): rank for rank, value in enumerate(policy["priority_years"])}
    candidates["_year_rank"] = event_year.loc[candidates.index].map(
        lambda value: priority.get(int(value), len(priority))
    )
    candidates = candidates.sort_values(["_year_rank", "event_time_utc", "event_id"])
    status = pd.read_parquet(STATUS_PATH).set_index("event_id")
    for column, default in (
        ("provider_symbol", ""), ("symbol_resolution_method", ""),
        ("symbol_resolution_confidence", ""), ("resolution_asof_mode", ""),
    ):
        if column not in status:
            status[column] = default
    result_counts: Counter[str] = Counter()
    by_year: Counter[int] = Counter()
    eligible_ids_added: list[str] = []
    raw_hashes: dict[str, str] = {}
    for month, group in candidates.groupby("month", sort=False):
        raw_path = raw_month_path(str(month))
        month_bars = read_month_symbols(raw_path, set(group.provider_symbol.astype(str)))
        raw_hash = sha256(raw_path) if raw_path.is_file() and not month_bars.empty else ""
        if raw_hash:
            raw_hashes[str(raw_path)] = raw_hash
        for index, row in group.iterrows():
            queue.loc[index, "local_archive_attempt_count"] = 1
            queue.loc[index, "local_archive_policy_sha256"] = policy_hash
            if not raw_path.is_file():
                result = "ARCHIVE_MONTH_MISSING"
            elif month_bars.empty:
                result = "ARCHIVE_SYMBOL_MISSING"
            else:
                bars = event_window(month_bars, row)
                if bars.empty or not alpaca.bar_qa(bars).get("pass"):
                    result = "NO_BAR_OR_QA_FAIL"
                else:
                    exact, timing_reason, _ = alpaca_backfill.timing_only_eligibility(
                        pd.Timestamp(row.event_time_utc), bars.timestamp
                    )
                    result = "EXACT_TIMING_ELIGIBLE" if exact else timing_reason
                    if exact:
                        cache_path, cache_hash = save_slice(
                            bars, str(row.provider_symbol), str(row.event_date)
                        )
                        queue.loc[index, "fetch_status"] = "LOCAL_FINNHUB_ARCHIVE_FETCHED_VALID"
                        queue.loc[index, "fetch_error"] = ""
                        queue.loc[index, "exact_eligible"] = True
                        queue.loc[index, "provider"] = "LOCAL_FINNHUB_ARCHIVE_1M"
                        queue.loc[index, "provider_symbol"] = str(row.provider_symbol)
                        queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT))
                        queue.loc[index, "raw_sha256"] = cache_hash
                        queue.loc[index, "local_archive_source_file"] = str(raw_path.relative_to(ROOT))
                        queue.loc[index, "local_archive_source_sha256"] = raw_hash
                        event_id = str(row.event_id)
                        if event_id not in status.index:
                            raise RuntimeError("ARCHIVE_EVENT_MISSING_PROVIDER_STATUS")
                        status.loc[event_id, "selected_price_provider"] = "LOCAL_FINNHUB_ARCHIVE_1M"
                        status.loc[event_id, "provider_parity_eligible"] = True
                        status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
                        status.loc[event_id, "provider_symbol"] = str(row.provider_symbol)
                        status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
                        status.loc[event_id, "symbol_resolution_confidence"] = str(row.resolution_confidence)
                        status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
                        eligible_ids_added.append(event_id)
                        by_year[int(event_year.loc[index])] += 1
            queue.loc[index, "local_archive_backfill_status"] = result
            result_counts[result] += 1
    atomic_parquet(queue, HISTORICAL_QUEUE_PATH)
    atomic_parquet(status.reset_index(), STATUS_PATH)
    historical.write_aliases_and_timeline(queue)
    report = {
        "schema_version": 1,
        "timestamp": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "parity_summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
        "parity_summary_sha256": sha256(SUMMARY_PATH),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "candidate_rows": int(len(candidates)),
        "result_counts": dict(result_counts),
        "exact_rows_added": int(len(eligible_ids_added)),
        "exact_rows_added_by_year": {str(key): int(value) for key, value in sorted(by_year.items())},
        "source_months_hashed": int(len(raw_hashes)),
        "source_month_hashes": {
            str(Path(key).relative_to(ROOT)): value for key, value in sorted(raw_hashes.items())
        },
        "eligible_event_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(eligible_ids_added)).encode()
        ).hexdigest(),
        "historical_queue_sha256": sha256(HISTORICAL_QUEUE_PATH),
        "provider_status_sha256": sha256(STATUS_PATH),
    }
    atomic_json(report, BACKFILL_REPORT_PATH)
    return report


def parse_years(value: str | None) -> set[int] | None:
    if not value:
        return None
    output: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        elif part:
            output.add(int(part))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--years")
    args = parser.parse_args()
    output: dict[str, Any] = {}
    if args.parity:
        output["parity"] = run_parity(args.max_events, args.max_requests)
    if args.backfill:
        output["backfill"] = run_backfill(parse_years(args.years))
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
