from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpaca_backfill_v221 import timing_only_eligibility
from provider_env_runtime import import_user_environment_if_missing
import v59_provider_acquisition as provider


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "research" / "multisource_v2" / "COLLECTION_POLICY_FREEZE.json"
EVENT_TIME_MANIFEST = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING" / "V2_EVENT_TIME_MANIFEST.parquet"
CAUSAL_FEATURES = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING" / "V2_CAUSAL_TPLUS2_FEATURES.parquet"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
V2_RAW = WORKING / "V2_PRICE_RAW"
AVAILABILITY = WORKING / "V2_PRICE_AVAILABILITY.parquet"
PRE_EVENT = WORKING / "V2_PRE_EVENT_PRICE_FEATURES.parquet"
REPORT = ROOT / "research" / "multisource_v2" / "PRICE_COVERAGE_AUDIT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

TIMEZONES = {"US": "America/New_York", "KR": "Asia/Seoul"}
PROVIDERS = {"US": "MASSIVE_SIP", "KR": "KIS"}
PRIMARY_CACHE = {
    "US": ROOT / "price_data" / "US_V59_MASSIVE_1M",
    "KR": ROOT / "price_data" / "KR_V59_KIS_1M",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def cache_path(market: str, ticker: str, event_time: pd.Timestamp) -> tuple[Path, str]:
    local = pd.Timestamp(event_time).tz_convert(TIMEZONES[market])
    date = local.date().isoformat()
    return PRIMARY_CACHE[market] / str(local.year) / str(ticker) / f"{date}.parquet", date


def v2_raw_path(market: str, ticker: str, date: str) -> Path:
    return V2_RAW / market / PROVIDERS[market] / date[:4] / str(ticker) / f"{date}.parquet"


def copy_raw_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"V2 raw price hash mismatch: {destination}")
        return
    temp = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temp)
    os.replace(temp, destination)


def exact_times(event_time: pd.Timestamp, timestamps: pd.Series) -> tuple[bool, str, dict[str, Any]]:
    eligible, reason, timing = timing_only_eligibility(event_time, timestamps)
    result = dict(timing)
    event = pd.Timestamp(event_time).tz_convert("UTC")
    target = event + pd.Timedelta(minutes=2)
    values = pd.Series(pd.to_datetime(timestamps, utc=True)).sort_values().drop_duplicates().reset_index(drop=True)
    entry_candidates = values.loc[values >= target]
    if entry_candidates.empty:
        return eligible, reason, result
    entry = pd.Timestamp(entry_candidates.iloc[0])
    exit_target = entry + pd.Timedelta(minutes=30)
    exit_candidates = values.loc[values >= exit_target]
    result.update({
        "entry_target_utc": target,
        "actual_entry_time_utc": entry,
        "exit_target_utc": exit_target,
        "actual_exit_time_utc": pd.Timestamp(exit_candidates.iloc[0]) if not exit_candidates.empty else pd.NaT,
    })
    return eligible, reason, result


def asof_completed_close(completed: pd.DataFrame, target: pd.Timestamp) -> float:
    eligible = completed.loc[completed["bar_end_utc"] <= target]
    if eligible.empty:
        return math.nan
    return float(pd.to_numeric(eligible.iloc[-1]["close"], errors="coerce"))


def pre_event_features(raw_path: Path, actual_entry: pd.Timestamp) -> dict[str, Any]:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    bars = pd.read_parquet(raw_path, columns=columns, filters=[("timestamp", "<", actual_entry)])
    if bars.empty:
        return {"pre_event_feature_status": "NO_PRE_ENTRY_BARS"}
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    bars["bar_end_utc"] = bars["timestamp"] + pd.Timedelta(minutes=1)
    completed = bars.loc[bars["bar_end_utc"] <= actual_entry].copy()
    if len(completed) < 2:
        return {"pre_event_feature_status": "INSUFFICIENT_COMPLETED_PRE_ENTRY_BARS"}
    reference_close = float(pd.to_numeric(completed.iloc[-1]["close"], errors="coerce"))

    def ret_back(minutes: int) -> float:
        prior = asof_completed_close(completed, actual_entry - pd.Timedelta(minutes=minutes))
        return float(reference_close / prior - 1.0) if np.isfinite(reference_close) and np.isfinite(prior) and prior > 0 else math.nan

    def window(minutes: int) -> pd.DataFrame:
        return completed.loc[completed["bar_end_utc"] > actual_entry - pd.Timedelta(minutes=minutes)].copy()

    returns_30 = pd.to_numeric(window(30)["close"], errors="coerce").pct_change().dropna()
    volumes_30 = pd.to_numeric(window(30)["volume"], errors="coerce").dropna()
    latest_volume = float(pd.to_numeric(completed.iloc[-1]["volume"], errors="coerce"))
    high_60 = float(pd.to_numeric(window(60)["high"], errors="coerce").max())
    low_60 = float(pd.to_numeric(window(60)["low"], errors="coerce").min())
    session_open = float(pd.to_numeric(completed.iloc[0]["open"], errors="coerce"))
    intraday_position = (
        float((reference_close - low_60) / (high_60 - low_60))
        if np.isfinite(reference_close) and np.isfinite(high_60) and np.isfinite(low_60) and high_60 > low_60
        else math.nan
    )
    return {
        "pre_event_feature_status": "PASS",
        "latest_completed_bar_end_utc": completed["bar_end_utc"].max(),
        "completed_pre_entry_bar_count": int(len(completed)),
        **{f"return_{minutes}m": ret_back(minutes) for minutes in (1, 5, 15, 30, 60)},
        "realized_vol_30m": float(returns_30.std()) if len(returns_30) >= 2 else math.nan,
        "latest_completed_volume": latest_volume,
        "relative_volume_1_30": (
            float(latest_volume / volumes_30.mean()) if len(volumes_30) and volumes_30.mean() > 0 else math.nan
        ),
        "intraday_position_60m": intraday_position,
        "intraday_return_from_open": (
            float(reference_close / session_open - 1.0)
            if np.isfinite(reference_close) and np.isfinite(session_open) and session_open > 0 else math.nan
        ),
        "market_return_status": "NOT_AVAILABLE_IN_SYMBOL_ONLY_RAW",
        "sector_return_status": "NOT_AVAILABLE_IN_SYMBOL_ONLY_RAW",
    }


def selected_events() -> pd.DataFrame:
    features = pd.read_parquet(CAUSAL_FEATURES)
    selected = features.loc[
        features["multisource_exact_ge2_t2"].fillna(False).astype(bool)
        | features["event_time_changed"].fillna(False).astype(bool)
    ].copy()
    return selected.sort_values(["market", "v2_resolved_event_time_utc", "canonical_event_id"], kind="mergesort")


def run(fetch_missing: bool, max_api_requests: int) -> dict[str, Any]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind_collection") or policy.get("post_entry_prices_namespace") != "LABEL_ONLY":
        raise RuntimeError("price collection policy guard failed")
    events = selected_events()
    import_user_environment_if_missing()
    kis_client = None
    requests_used = 0
    availability_rows: list[dict[str, Any]] = []
    pre_rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        market = str(event.market)
        event_time = pd.Timestamp(event.v2_resolved_event_time_utc)
        source_path, date = cache_path(market, str(event.ticker), event_time)
        fetch_status = "CACHE_PRESENT" if source_path.is_file() else "CACHE_MISSING"
        if not source_path.is_file() and fetch_missing and market == "KR" and requests_used < max_api_requests:
            if kis_client is None:
                kis_client = provider.KISClient()
            remaining_pages = max_api_requests - requests_used
            bars, status = kis_client.get_day(
                str(event.ticker), date.replace("-", ""), max_pages=min(8, remaining_pages)
            )
            requests_used += int(status.get("api_requests", 1))
            qa = provider.bar_qa(bars, "KR") if not bars.empty else {"pass": False}
            if status.get("ok") and not bars.empty and qa.get("pass"):
                provider.atomic_parquet(bars, source_path)
                fetch_status = "FETCHED_VALID"
            else:
                fetch_status = "PROVIDER_FAILED_OR_EMPTY"
        raw_path = v2_raw_path(market, str(event.ticker), date)
        row: dict[str, Any] = {
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_PRICE_AVAILABILITY_V1",
            "canonical_event_id": str(event.canonical_event_id),
            "market": market,
            "ticker": str(event.ticker),
            "v2_resolved_event_time_utc": event_time,
            "event_time_changed": bool(event.event_time_changed),
            "multisource_exact_ge2_t2": bool(event.multisource_exact_ge2_t2),
            "provider": PROVIDERS[market],
            "trade_date": date,
            "provider_cache_status": fetch_status,
            "provider_cache_path": str(source_path.relative_to(ROOT)) if source_path.is_file() else "",
            "v2_raw_path": "",
            "raw_sha256": "",
            "exact_timing_eligible": False,
            "timing_reason": "RAW_PRICE_MISSING",
            "entry_target_utc": pd.NaT,
            "actual_entry_time_utc": pd.NaT,
            "entry_slippage_sec": math.nan,
            "exit_target_utc": pd.NaT,
            "actual_exit_time_utc": pd.NaT,
            "exit_slippage_sec": math.nan,
            "actual_hold_seconds": math.nan,
            "price_value_columns_read_for_availability": False,
            "outcome_fields_read": "",
        }
        if source_path.is_file():
            copy_raw_immutable(source_path, raw_path)
            timestamps = pd.read_parquet(raw_path, columns=["timestamp"])["timestamp"]
            eligible, reason, timing = exact_times(event_time, timestamps)
            row.update({
                "v2_raw_path": str(raw_path.relative_to(ROOT)),
                "raw_sha256": sha256_file(raw_path),
                "exact_timing_eligible": bool(eligible),
                "timing_reason": reason,
                **timing,
            })
            if eligible:
                safe = pre_event_features(raw_path, pd.Timestamp(timing["actual_entry_time_utc"]))
                pre_rows.append({
                    "schema_version": "HIST_NEWS_MULTISOURCE_V2_PRE_EVENT_PRICE_FEATURES_V1",
                    "canonical_event_id": str(event.canonical_event_id),
                    "market": market,
                    "ticker": str(event.ticker),
                    "v2_resolved_event_time_utc": event_time,
                    "decision_cutoff_utc": timing["entry_target_utc"],
                    "actual_entry_time_utc": timing["actual_entry_time_utc"],
                    "feature_price_cutoff_rule": "BAR_END_UTC_LE_ACTUAL_ENTRY; NO_POST_ENTRY_PRICE_VALUES_READ",
                    **safe,
                    "outcome_fields_read": "",
                })
        availability_rows.append(row)

    availability = pd.DataFrame(availability_rows)
    pre = pd.DataFrame(pre_rows)
    atomic_parquet(AVAILABILITY, availability)
    atomic_parquet(PRE_EVENT, pre)
    exact = availability.loc[availability["exact_timing_eligible"].fillna(False).astype(bool)]
    exact_multi = exact.loc[exact["multisource_exact_ge2_t2"].fillna(False).astype(bool)]
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_PRICE_COVERAGE_AUDIT_V1",
        "generated_at_utc": utc_now(),
        "scope": "V2_EVENT_TIME_CHANGED_OR_MULTISOURCE_EXACT_EVENTS",
        "events_audited": int(len(availability)),
        "exact_timing_eligible_events": int(len(exact)),
        "exact_price_multisource_events": int(len(exact_multi)),
        "pre_event_feature_rows": int(len(pre)),
        "network_api_requests": int(requests_used),
        "credential_presence": {
            "KIS": "PRESENT" if provider.credential_status()["KIS_APP_KEY"] == "PRESENT" else "MISSING",
            "MASSIVE": "PRESENT" if provider.credential_status()["MASSIVE_API_KEY"] == "PRESENT" else "MISSING",
        },
        "price_value_columns_read_for_availability": False,
        "post_entry_price_values_read_for_features": False,
        "labels_joined": False,
        "outcome_fields_read": [],
        "entry_slippage_violations": int((
            exact["entry_slippage_sec"].lt(0) | exact["entry_slippage_sec"].gt(60)
        ).sum()) if not exact.empty else 0,
        "exit_slippage_violations": int((
            exact["exit_slippage_sec"].lt(0) | exact["exit_slippage_sec"].gt(60)
        ).sum()) if not exact.empty else 0,
        "hold_violations": int((
            exact["actual_hold_seconds"].lt(1800) | exact["actual_hold_seconds"].gt(1860)
        ).sum()) if not exact.empty else 0,
        "five_minute_fallback": 0,
        "daily_fallback": 0,
        "synthetic_interpolation": 0,
        "source_freeze_ready": False,
        "model_training_allowed": False,
        "base_v1_mutated": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {
            AVAILABILITY.name: sha256_file(AVAILABILITY),
            PRE_EVENT.name: sha256_file(PRE_EVENT),
        },
    }
    atomic_json(REPORT, report)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update({
        "updated_at_utc": utc_now(),
        "status": "PRICE_AVAILABILITY_AUDITED_COLLECTION_CONTINUES",
        "phase": "OUTCOME_BLIND_MULTISOURCE_AND_EXACT_PRICE_COLLECTION",
        "v2_price_events_audited": report["events_audited"],
        "v2_exact_price_multisource_events": report["exact_price_multisource_events"],
        "v2_pre_event_price_feature_rows": report["pre_event_feature_rows"],
        "labels_read": False,
        "source_frozen": False,
        "split_frozen": False,
        "model_training_allowed": False,
        "base_v1_membership_mutable": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
    })
    atomic_json(STATE, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--max-api-requests", type=int, default=20)
    args = parser.parse_args()
    if args.max_api_requests < 0:
        raise SystemExit("--max-api-requests must be non-negative")
    print(json.dumps(run(args.fetch_missing, args.max_api_requests), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
