"""Fail-closed parity and local historical 1-minute fallback for V221+.

The candidate store is the already pinned ``price_data/US`` copy of
``ggaddam/OHLCV-1m`` (originally Finnhub).  It is never authorized merely
because it exists: frozen DEV-only label parity against Massive/Alpaca must
pass first.  Backfill selection and timing QA do not load outcome columns.
"""
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
import bio_news_30m_v3 as app
import historical_symbol_resolver as historical
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
LOCAL_STORE = ROOT / "price_data" / "US"
POLICY_PATH = DATA / "LOCAL_FINNHUB_PROVIDER_PARITY_POLICY_V1.json"
SEMANTICS_PATH = DATA / "BAR_SEMANTICS_AUDIT_V36.json"
QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
HISTORICAL_QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
NO_BAR_PATH = DATA / "HISTORICAL_NO_BAR_CLASSIFICATION.parquet"
SAMPLE_PATH = RESEARCH / "LOCAL_FINNHUB_PARITY_OVERLAP_SAMPLE.csv"
DETAIL_PATH = RESEARCH / "LOCAL_FINNHUB_PROVIDER_PARITY_DETAIL.csv"
SUMMARY_PATH = RESEARCH / "LOCAL_FINNHUB_PROVIDER_PARITY_SUMMARY.json"
BACKFILL_REPORT_PATH = RESEARCH / "LOCAL_FINNHUB_HISTORICAL_BACKFILL_REPORT.json"
INITIAL_PARITY_PATH = RESEARCH / "LOCAL_FINNHUB_PROVIDER_PARITY_INITIAL_EXISTING_CACHE_RESULT.json"


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
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    guard = {
        "schema_version": 1,
        "policy_name": "MASSIVE_ALPACA_LOCAL_FINNHUB_EXACT_T2_30M_PARITY_V1",
        "frozen_before_local_overlap_results": True,
        "selection_uses_outcomes": False,
        "eligible_role_for_label_level_parity": "DEV_EXTENSION",
        "eligible_source_family_for_parity": "US_SEC",
        "candidate_provider": "LOCAL_FINNHUB_1M",
        "provider_choice_uses_outcomes": False,
    }
    if any(policy.get(key) != value for key, value in guard.items()):
        raise RuntimeError("LOCAL_FINNHUB_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


def verify_semantics() -> str:
    audit = json.loads(SEMANTICS_PATH.read_text(encoding="utf-8"))
    stores = {row.get("directory"): row for row in audit.get("stores", [])}
    row = stores.get("price_data/US")
    definition = (row or {}).get("definition", {})
    if not row or definition.get("official_exact_eligible") is not True:
        raise RuntimeError("LOCAL_FINNHUB_SEMANTICS_NOT_EXACT_ELIGIBLE")
    if definition.get("documented_cadence_sec") != 60:
        raise RuntimeError("LOCAL_FINNHUB_CADENCE_GUARD_FAIL")
    if definition.get("timestamp_semantics") != "OPEN" or definition.get("timezone") != "UTC":
        raise RuntimeError("LOCAL_FINNHUB_TIMESTAMP_SEMANTICS_GUARD_FAIL")
    return sha256(SEMANTICS_PATH)


def local_path(symbol: str) -> Path:
    value = str(symbol).upper().strip()
    if not value or not value.replace(".", "").replace("-", "").isalnum():
        raise ValueError("LOCAL_FINNHUB_SYMBOL_REJECTED")
    return LOCAL_STORE / f"{value}.parquet"


def load_window(path: Path, start: Any, end: Any, provider_name: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("timestamp", ">=", start_ts), ("timestamp", "<=", end_ts)],
        )
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        timestamps = pd.to_datetime(frame.timestamp, utc=True)
        frame = frame.loc[(timestamps >= start_ts) & (timestamps <= end_ts)].copy()
    frame["datetime"] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.sort_values("datetime").drop_duplicates("datetime")
    frame.attrs.update({
        "price_source": provider_name,
        "price_cadence_sec": 60,
        "bar_timestamp_semantics": "OPEN",
        "official_exact_eligible": True,
    })
    return frame


def exact_row(sample: pd.Series, bars: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    event = pd.Series({
        "event_id": sample.event_id,
        "market": "US",
        "ticker": sample.provider_symbol,
        "event_time_utc": pd.Timestamp(sample.event_time_utc),
        "source": sample.source_family,
        "source_family": sample.source_family,
        "headline": "",
        "body": "",
        "form": "",
        "event_type": "OTHER",
    })
    return app.event_to_row_exact_contract(event, bars, app.Config(official_exact_contract=True))


def compare_pair(sample: pd.Series, reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    start = pd.Timestamp(sample.required_start_utc)
    end = pd.Timestamp(sample.required_end_utc)
    reference_window = reference[(reference.datetime >= start) & (reference.datetime <= end)]
    candidate_window = candidate[(candidate.datetime >= start) & (candidate.datetime <= end)]
    reference_times = set(reference_window.datetime.astype("int64"))
    candidate_times = set(candidate_window.datetime.astype("int64"))
    union = reference_times | candidate_times
    intersection = reference_times & candidate_times
    reference_row, reference_reason = exact_row(sample, reference)
    candidate_row, candidate_reason = exact_row(sample, candidate)
    result: dict[str, Any] = {
        "event_id": sample.event_id,
        "ticker": sample.ticker,
        "provider_symbol": sample.provider_symbol,
        "source_family": sample.source_family,
        "role": sample.role,
        "reference_provider": sample.reference_provider,
        "candidate_provider": "LOCAL_FINNHUB_1M",
        "bar_coverage_ratio": len(intersection) / len(union) if union else 0.0,
        "timestamp_overlap": len(intersection),
        "reference_window_bars": len(reference_times),
        "candidate_window_bars": len(candidate_times),
        "reference_exact_reason": reference_reason,
        "candidate_exact_reason": candidate_reason,
        "both_exact": reference_row is not None and candidate_row is not None,
        "candidate_fetch_usable": not candidate.empty and bool(alpaca.bar_qa(candidate).get("pass")),
    }
    if reference_row is None or candidate_row is None:
        result.update({
            "entry_timestamp_agreement": False,
            "exit_timestamp_agreement": False,
            "entry_price_relative_diff": np.nan,
            "exit_price_relative_diff": np.nan,
            "fwd_ret_30m_reference": np.nan,
            "fwd_ret_30m_candidate": np.nan,
            "label_agreement": False,
            "provider_disposition": "EXACT_CONTRACT_FAIL",
        })
        return result
    entry_diff = abs(float(candidate_row["entry_price"]) / float(reference_row["entry_price"]) - 1.0)
    exit_diff = abs(float(candidate_row["exit_price"]) / float(reference_row["exit_price"]) - 1.0)
    label_agreement = int(reference_row["y"]) == int(candidate_row["y"])
    result.update({
        "entry_timestamp_agreement": pd.Timestamp(reference_row["actual_entry_time_utc"]) == pd.Timestamp(candidate_row["actual_entry_time_utc"]),
        "exit_timestamp_agreement": pd.Timestamp(reference_row["actual_exit_time_utc"]) == pd.Timestamp(candidate_row["actual_exit_time_utc"]),
        "entry_price_relative_diff": entry_diff,
        "exit_price_relative_diff": exit_diff,
        "fwd_ret_30m_reference": float(reference_row["fwd_ret_30m"]),
        "fwd_ret_30m_candidate": float(candidate_row["fwd_ret_30m"]),
        "label_agreement": label_agreement,
        "provider_disposition": "PAIR_OK" if label_agreement else "PROVIDER_LABEL_DISAGREEMENT",
    })
    return result


def summarize(detail: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    thresholds = policy["thresholds"]
    exact = detail.loc[detail.both_exact.astype(bool)].copy() if not detail.empty else detail.copy()
    correlation = (
        float(exact.fwd_ret_30m_reference.corr(exact.fwd_ret_30m_candidate))
        if len(exact) >= 2 else None
    )
    reference_eligible = detail.reference_exact_reason.eq("OK") if len(detail) else pd.Series(dtype=bool)
    candidate_eligible = detail.candidate_exact_reason.eq("OK") if len(detail) else pd.Series(dtype=bool)
    usable = detail.candidate_fetch_usable.astype(bool) if len(detail) else pd.Series(dtype=bool)
    metrics = {
        "n": int(len(detail)),
        "exact_pair_n": int(len(exact)),
        "exact_pair_rate": float(len(exact) / len(detail)) if len(detail) else 0.0,
        "candidate_fetch_usable_rate": float(usable.mean()) if len(detail) else 0.0,
        "exact_eligibility_agreement_on_usable": (
            float((reference_eligible[usable] == candidate_eligible[usable]).mean())
            if usable.any() else 0.0
        ),
        "bar_coverage": float(detail.bar_coverage_ratio.mean()) if len(detail) else 0.0,
        "entry_timestamp_agreement": float(exact.entry_timestamp_agreement.mean()) if len(exact) else 0.0,
        "exit_timestamp_agreement": float(exact.exit_timestamp_agreement.mean()) if len(exact) else 0.0,
        "timing_contract_agreement": float(
            (exact.entry_timestamp_agreement & exact.exit_timestamp_agreement).mean()
        ) if len(exact) else 0.0,
        "median_absolute_entry_price_relative_diff": float(exact.entry_price_relative_diff.median()) if len(exact) else None,
        "p95_absolute_entry_price_relative_diff": float(exact.entry_price_relative_diff.quantile(0.95)) if len(exact) else None,
        "return_correlation": correlation,
        "up_down_label_agreement": float(exact.label_agreement.mean()) if len(exact) else 0.0,
        "provider_label_disagreement_n": int((exact.provider_disposition == "PROVIDER_LABEL_DISAGREEMENT").sum()) if len(exact) else 0,
    }
    checks = {
        "minimum_exact_pair_rows": metrics["exact_pair_n"] >= thresholds["minimum_exact_pair_rows"],
        "timing_contract_agreement": metrics["timing_contract_agreement"] >= thresholds["timing_contract_agreement_min"],
        "up_down_label_agreement": metrics["up_down_label_agreement"] >= thresholds["up_down_label_agreement_min"],
        "return_correlation": correlation is not None and correlation >= thresholds["return_correlation_min"],
        "median_entry_price_diff": metrics["median_absolute_entry_price_relative_diff"] is not None and metrics["median_absolute_entry_price_relative_diff"] <= thresholds["median_absolute_entry_price_relative_diff_max"],
        "p95_entry_price_diff": metrics["p95_absolute_entry_price_relative_diff"] is not None and metrics["p95_absolute_entry_price_relative_diff"] <= thresholds["p95_absolute_entry_price_relative_diff_max"],
    }
    return {
        "provider": "LOCAL_FINNHUB_1M",
        "reference_providers": sorted(set(detail.reference_provider.astype(str))) if len(detail) else [],
        "metrics": metrics,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def build_overlap_sample(max_events: int = 200) -> pd.DataFrame:
    queue = pd.read_parquet(QUEUE_PATH)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "labels_opened"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if frame.role.isna().any():
        raise RuntimeError("LOCAL_PARITY_EVENT_WITHOUT_FROZEN_ROLE")
    if frame.loc[frame.role.ne("DEV_EXTENSION"), "labels_opened"].fillna(False).astype(bool).any():
        raise RuntimeError("SEALED_ROLE_OPEN_STATE_UNEXPECTED")
    frame = frame.loc[
        frame.role.astype(str).eq("DEV_EXTENSION")
        & frame.source_family.astype(str).eq("US_SEC")
    ].copy()
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["trade_date"] = frame.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)
    status = provider.load_us_provider_status()
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        authorities = provider.resolve_us_price_cache_candidates(
            str(row.event_id), str(row.ticker), row.trade_date, status
        )
        if not authorities:
            continue
        reference_path, reference_provider, reference_feed, provider_symbol = authorities[0]
        candidate_path = local_path(provider_symbol)
        if not candidate_path.is_file():
            continue
        values = row._asdict()
        values.update({
            "provider_symbol": str(provider_symbol),
            "reference_provider": f"{reference_provider}_{reference_feed}",
            "reference_raw_file": str(reference_path.relative_to(ROOT)),
            "candidate_raw_file": str(candidate_path.relative_to(ROOT)),
            "selection_hash": hashlib.sha256(
                f"V221_LOCAL_FINNHUB_PARITY_V1|{row.event_id}".encode()
            ).hexdigest(),
        })
        rows.append(values)
    selected = pd.DataFrame(rows)
    if not selected.empty:
        selected = selected.sort_values("selection_hash").head(max_events).copy()
        columns = [
            "event_id", "ticker", "provider_symbol", "issuer", "event_time_utc",
            "required_start_utc", "required_end_utc", "source_family", "role",
            "trade_date", "reference_provider", "reference_raw_file",
            "candidate_raw_file", "selection_hash",
        ]
        selected = selected[columns]
    atomic_csv(selected, SAMPLE_PATH)
    return selected


def run_parity(max_events: int = 200) -> dict[str, Any]:
    policy, policy_hash = load_policy()
    semantics_hash = verify_semantics()
    sample = build_overlap_sample(max_events)
    detail_rows: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        sample_row = pd.Series(row._asdict())
        reference = load_window(
            ROOT / row.reference_raw_file, row.required_start_utc,
            row.required_end_utc, row.reference_provider,
        )
        candidate = load_window(
            ROOT / row.candidate_raw_file, row.required_start_utc,
            row.required_end_utc, "LOCAL_FINNHUB_1M",
        )
        detail_rows.append(compare_pair(sample_row, reference, candidate))
    detail = pd.DataFrame(detail_rows)
    atomic_csv(detail, DETAIL_PATH)
    result = summarize(detail, policy)
    summary = {
        "schema_version": 1,
        "audited_at": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "semantics_audit_path": str(SEMANTICS_PATH.relative_to(ROOT)),
        "semantics_audit_sha256": semantics_hash,
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "sample_sha256": sha256(SAMPLE_PATH),
        "detail_path": str(DETAIL_PATH.relative_to(ROOT)),
        "detail_sha256": sha256(DETAIL_PATH),
        "selection_uses_outcomes": False,
        "label_level_parity_role": "DEV_EXTENSION",
        "label_level_parity_source_family": "US_SEC",
        "sealed_role_labels_opened": False,
        "result": result,
        "backfill_authorized": result["status"] == "PASS",
    }
    atomic_json(summary, SUMMARY_PATH)
    return summary


def build_reference_acquisition_sample(max_events: int = 160) -> pd.DataFrame:
    """Freeze local-exact DEV windows before acquiring independent references."""
    policy, _ = load_policy()
    queue = pd.read_parquet(QUEUE_PATH)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "labels_opened"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if frame.role.isna().any():
        raise RuntimeError("LOCAL_PARITY_EVENT_WITHOUT_FROZEN_ROLE")
    if frame.loc[frame.role.ne("DEV_EXTENSION"), "labels_opened"].fillna(False).astype(bool).any():
        raise RuntimeError("SEALED_ROLE_OPEN_STATE_UNEXPECTED")
    frame = frame.loc[
        frame.role.astype(str).eq("DEV_EXTENSION")
        & frame.source_family.astype(str).eq("US_SEC")
    ].copy()
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["trade_date"] = frame.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)
    status = provider.load_us_provider_status()
    historical_queue = pd.read_parquet(HISTORICAL_QUEUE_PATH)
    historical_queue = historical_queue.loc[
        historical_queue.resolution_confidence.astype(str).isin({"A", "B"})
        & historical_queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
        & historical_queue.resolved_ticker.astype(str).ne("")
    ]
    historical_symbols = historical_queue.set_index("event_id").resolved_ticker.astype(str).to_dict()
    status_symbols = {}
    if not status.empty and "provider_symbol" in status:
        status_symbols = {
            str(key): "" if pd.isna(value) else str(value).strip()
            for key, value in status.provider_symbol.items()
        }
    frame["provider_symbol"] = [
        historical_symbols.get(str(event_id))
        or status_symbols.get(str(event_id), "")
        or str(ticker)
        for event_id, ticker in zip(frame.event_id, frame.ticker)
    ]
    frame["selection_hash"] = frame.event_id.astype(str).map(
        lambda value: hashlib.sha256(
            f"V221_LOCAL_FINNHUB_ACQUIRED_PARITY_V1|{value}".encode()
        ).hexdigest()
    )
    year = frame.event_time_utc.dt.year
    priority = {int(value): rank for rank, value in enumerate(policy["priority_years"])}
    frame["_year_rank"] = year.map(lambda value: priority.get(int(value), len(priority)))
    frame = frame.sort_values(["_year_rank", "selection_hash", "event_time_utc"])
    rows: list[dict[str, Any]] = []
    used_symbol_dates: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        symbol = str(row.provider_symbol).upper().strip()
        key = (symbol, str(row.trade_date))
        if key in used_symbol_dates:
            continue
        path = local_path(symbol)
        if not path.is_file():
            continue
        candidate = load_window(
            path, row.required_start_utc, row.required_end_utc, "LOCAL_FINNHUB_1M"
        )
        if candidate.empty or not alpaca.bar_qa(candidate).get("pass"):
            continue
        exact, _, _ = alpaca_backfill.timing_only_eligibility(
            pd.Timestamp(row.event_time_utc), candidate.timestamp
        )
        if not exact:
            continue
        values = row._asdict()
        values.update({
            "reference_provider": "ALPACA_SIP",
            "reference_raw_file": "",
            "candidate_raw_file": str(path.relative_to(ROOT)),
        })
        rows.append(values)
        used_symbol_dates.add(key)
        if len(rows) >= max_events:
            break
    selected = pd.DataFrame(rows)
    if not selected.empty:
        columns = [
            "event_id", "ticker", "provider_symbol", "issuer", "event_time_utc",
            "required_start_utc", "required_end_utc", "source_family", "role",
            "trade_date", "reference_provider", "reference_raw_file",
            "candidate_raw_file", "selection_hash",
        ]
        selected = selected[columns]
    atomic_csv(selected, SAMPLE_PATH)
    return selected


def failed_reference_row(sample: pd.Series, error: str) -> dict[str, Any]:
    return {
        "event_id": sample.event_id,
        "ticker": sample.ticker,
        "provider_symbol": sample.provider_symbol,
        "source_family": sample.source_family,
        "role": sample.role,
        "reference_provider": "ALPACA_SIP",
        "candidate_provider": "LOCAL_FINNHUB_1M",
        "bar_coverage_ratio": 0.0,
        "timestamp_overlap": 0,
        "reference_window_bars": 0,
        "candidate_window_bars": 0,
        "reference_exact_reason": str(error)[:240],
        "candidate_exact_reason": "NOT_EVALUATED",
        "both_exact": False,
        "candidate_fetch_usable": True,
        "entry_timestamp_agreement": False,
        "exit_timestamp_agreement": False,
        "entry_price_relative_diff": np.nan,
        "exit_price_relative_diff": np.nan,
        "fwd_ret_30m_reference": np.nan,
        "fwd_ret_30m_candidate": np.nan,
        "label_agreement": False,
        "provider_disposition": "REFERENCE_FETCH_FAIL",
    }


def run_parity_with_reference_acquisition(
    max_events: int = 160,
    max_requests: int | None = None,
) -> dict[str, Any]:
    policy, policy_hash = load_policy()
    semantics_hash = verify_semantics()
    if SUMMARY_PATH.is_file() and not INITIAL_PARITY_PATH.is_file():
        atomic_json(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")), INITIAL_PARITY_PATH)
    sample = build_reference_acquisition_sample(max_events)
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = alpaca_backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    client = alpaca.AlpacaClient()
    requests_used = 0
    detail_rows: list[dict[str, Any]] = []
    updated_sample = sample.copy()
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
            reference = load_window(
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
                reference = load_window(
                    reference_path, row.required_start_utc, row.required_end_utc, "ALPACA_SIP"
                )
        if not status.get("ok") or reference.empty or not alpaca.bar_qa(reference).get("pass"):
            detail_rows.append(failed_reference_row(
                sample_row, str(status.get("error") or "EMPTY_OR_QA_FAIL")
            ))
            continue
        updated_sample.loc[position, "reference_raw_file"] = str(reference_path.relative_to(ROOT))
        candidate = load_window(
            ROOT / row.candidate_raw_file, row.required_start_utc,
            row.required_end_utc, "LOCAL_FINNHUB_1M",
        )
        detail_rows.append(compare_pair(sample_row, reference, candidate))
    atomic_csv(updated_sample, SAMPLE_PATH)
    detail = pd.DataFrame(detail_rows)
    atomic_csv(detail, DETAIL_PATH)
    result = summarize(detail, policy)
    summary = {
        "schema_version": 1,
        "audited_at": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "semantics_audit_path": str(SEMANTICS_PATH.relative_to(ROOT)),
        "semantics_audit_sha256": semantics_hash,
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "sample_sha256": sha256(SAMPLE_PATH),
        "detail_path": str(DETAIL_PATH.relative_to(ROOT)),
        "detail_sha256": sha256(DETAIL_PATH),
        "selection_uses_outcomes": False,
        "sample_selection_requires_local_exact_timing_only": True,
        "label_level_parity_role": "DEV_EXTENSION",
        "label_level_parity_source_family": "US_SEC",
        "sealed_role_labels_opened": False,
        "reference_network_requests": int(requests_used),
        "result": result,
        "backfill_authorized": result["status"] == "PASS",
    }
    atomic_json(summary, SUMMARY_PATH)
    return summary


def ensure_backfill_columns(queue: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "local_finnhub_backfill_status": "NOT_ATTEMPTED",
        "local_finnhub_policy_sha256": "",
        "local_finnhub_attempt_count": 0,
    }
    for column, default in defaults.items():
        if column not in queue:
            queue[column] = default
    return queue


def authorize_backfill() -> tuple[dict[str, Any], str]:
    policy, policy_hash = load_policy()
    verify_semantics()
    if not SUMMARY_PATH.is_file():
        raise RuntimeError("LOCAL_FINNHUB_PARITY_MISSING")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("policy_sha256") != policy_hash:
        raise RuntimeError("LOCAL_FINNHUB_PARITY_POLICY_HASH_MISMATCH")
    if summary.get("backfill_authorized") is not True:
        raise RuntimeError("LOCAL_FINNHUB_PARITY_FAIL")
    if summary.get("sealed_role_labels_opened") is not False:
        raise RuntimeError("SEALED_ROLE_LABEL_GUARD_FAIL")
    return policy, policy_hash


def run_backfill(years: set[int] | None = None) -> dict[str, Any]:
    policy, policy_hash = authorize_backfill()
    queue = ensure_backfill_columns(pd.read_parquet(HISTORICAL_QUEUE_PATH))
    audit = pd.read_parquet(NO_BAR_PATH)
    if audit.selection_uses_labels.astype(bool).any():
        raise RuntimeError("NO_BAR_AUDIT_LABEL_GUARD_FAIL")
    eligible_ids = set(
        audit.loc[audit.high_confidence_fallback_candidate.astype(bool), "event_id"].astype(str)
    )
    event_year = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    pending = queue.event_id.astype(str).isin(eligible_ids)
    pending &= queue.role.astype(str).eq("DEV_EXTENSION")
    pending &= queue.source_family.astype(str).eq("US_SEC")
    pending &= queue.fetch_status.astype(str).eq("ALPACA_NO_BAR")
    pending &= queue.resolution_confidence.astype(str).isin({"A", "B"})
    pending &= queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
    pending &= queue.resolved_ticker.astype(str).ne("")
    pending &= pd.to_numeric(queue.local_finnhub_attempt_count, errors="coerce").fillna(0).eq(0)
    if years is not None:
        pending &= event_year.isin(years)
    priority = {int(year): rank for rank, year in enumerate(policy["priority_years"])}
    candidates = queue.loc[pending].copy()
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
    raw_hashes: dict[str, str] = {}
    eligible_ids_added: list[str] = []
    by_year: Counter[int] = Counter()
    for index, row in candidates.iterrows():
        symbol = str(row.resolved_ticker)
        path = local_path(symbol)
        queue.loc[index, "local_finnhub_attempt_count"] = 1
        queue.loc[index, "local_finnhub_policy_sha256"] = policy_hash
        if not path.is_file():
            result = "LOCAL_SYMBOL_FILE_MISSING"
        else:
            bars = load_window(path, row.required_start_utc, row.required_end_utc, "LOCAL_FINNHUB_1M")
            if bars.empty or not alpaca.bar_qa(bars).get("pass"):
                result = "NO_BAR_OR_QA_FAIL"
            else:
                exact, timing_reason, _ = alpaca_backfill.timing_only_eligibility(
                    pd.Timestamp(row.event_time_utc), bars.timestamp
                )
                result = "EXACT_TIMING_ELIGIBLE" if exact else timing_reason
                if exact:
                    raw_hash = raw_hashes.setdefault(str(path), sha256(path))
                    queue.loc[index, "fetch_status"] = "LOCAL_FINNHUB_FETCHED_VALID"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = True
                    queue.loc[index, "provider"] = "LOCAL_FINNHUB_1M"
                    queue.loc[index, "provider_symbol"] = symbol
                    queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
                    queue.loc[index, "raw_sha256"] = raw_hash
                    event_id = str(row.event_id)
                    if event_id not in status.index:
                        raise RuntimeError("LOCAL_FINNHUB_EVENT_MISSING_PROVIDER_STATUS")
                    status.loc[event_id, "selected_price_provider"] = "LOCAL_FINNHUB_1M"
                    status.loc[event_id, "provider_parity_eligible"] = True
                    status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
                    status.loc[event_id, "provider_symbol"] = symbol
                    status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
                    status.loc[event_id, "symbol_resolution_confidence"] = str(row.resolution_confidence)
                    status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
                    eligible_ids_added.append(event_id)
                    by_year[int(event_year.loc[index])] += 1
        queue.loc[index, "local_finnhub_backfill_status"] = result
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
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        elif part:
            years.add(int(part))
    return years


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--parity-acquire-reference", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--years")
    args = parser.parse_args()
    output: dict[str, Any] = {}
    if args.parity:
        output["parity"] = run_parity(args.max_events)
    if args.parity_acquire_reference:
        output["parity"] = run_parity_with_reference_acquisition(
            args.max_events, args.max_requests
        )
    if args.backfill:
        output["backfill"] = run_backfill(parse_years(args.years))
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
