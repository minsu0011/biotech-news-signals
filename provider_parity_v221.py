"""Outcome-safe Massive/Alpaca parity harness for V221+.

Only frozen DEV_EXTENSION rows may enter label-level parity.  Research seal and
final reserve rows remain unopened.  Thresholds are loaded from a policy file
that was frozen before any Alpaca overlap result was available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import alpaca_provider_acquisition as alpaca
import bio_news_30m_v3 as app
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "ALPACA_PROVIDER_PARITY_POLICY_V1.json"
SAMPLE_PATH = RESEARCH / "PROVIDER_PARITY_OVERLAP_SAMPLE.csv"
DETAIL_PATH = RESEARCH / "PROVIDER_PARITY_MASSIVE_ALPACA.csv"
SUMMARY_PATH = RESEARCH / "PROVIDER_PARITY_SUMMARY.json"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
BACKFILL_QUEUE_PATH = DATA / "ALPACA_BACKFILL_QUEUE.parquet"
BACKFILL_AUDIT_PATH = DATA / "ALPACA_BACKFILL_QUEUE_AUDIT.json"
US_QUEUE = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
MASSIVE_CACHE = ROOT / "price_data" / "US_V59_MASSIVE_1M"
ALPACA_CACHE = ROOT / "price_data" / "alpaca_1m"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    for secret in alpaca.alpaca_secret_values(required=False):
        if secret and secret in serialized:
            raise RuntimeError("ALPACA_SECRET_SERIALIZATION_REJECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
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


def cache_path(root: Path, feed: str, ticker: str, date: str) -> Path:
    return root / feed.lower() / date[:4] / str(ticker) / f"{date}.parquet"


def massive_path(ticker: str, date: str) -> Path:
    return MASSIVE_CACHE / date[:4] / str(ticker) / f"{date}.parquet"


def load_bars(path: Path, provider: str, feed: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "timestamp" not in frame:
        raise RuntimeError("CACHE_TIMESTAMP_MISSING")
    frame["datetime"] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.sort_values("datetime").drop_duplicates("datetime")
    frame.attrs.update(
        {
            "price_source": f"{provider}_{feed}_1M",
            "price_cadence_sec": 60,
            "bar_timestamp_semantics": "OPEN",
            "official_exact_eligible": True,
        }
    )
    return frame


def _pre_event_volume(path: Path, event_time: pd.Timestamp) -> float:
    try:
        bars = pd.read_parquet(path, columns=["timestamp", "volume"])
        timestamp = pd.to_datetime(bars.timestamp, utc=True)
        selected = bars.loc[
            (timestamp >= event_time - pd.Timedelta(minutes=120))
            & (timestamp <= event_time - pd.Timedelta(minutes=5)),
            "volume",
        ]
        value = pd.to_numeric(selected, errors="coerce").mean()
        return float(value) if np.isfinite(value) and value >= 0 else 0.0
    except Exception:
        return 0.0


def build_overlap_sample(max_events: int = 120) -> pd.DataFrame:
    queue = pd.read_parquet(US_QUEUE)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "labels_opened"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if frame.role.isna().any():
        raise RuntimeError("PARITY_EVENT_WITHOUT_FROZEN_ROLE")
    if (frame.loc[frame.role.ne("DEV_EXTENSION"), "labels_opened"].fillna(False)).any():
        raise RuntimeError("SEALED_ROLE_OPEN_STATE_UNEXPECTED")
    frame = frame[
        frame.role.eq("DEV_EXTENSION")
        & frame.status.isin(["CACHED_VALID", "FETCHED_VALID"])
        & frame.source_family.isin(["US_SEC", "US_NEWS"])
    ].copy()
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["trade_date"] = frame.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)
    frame["massive_raw_file"] = [
        str(massive_path(ticker, date).relative_to(ROOT))
        for ticker, date in zip(frame.ticker.astype(str), frame.trade_date)
    ]
    frame = frame[[Path(ROOT / path).is_file() for path in frame.massive_raw_file]].copy()
    frame["pre_event_mean_minute_volume"] = [
        _pre_event_volume(ROOT / path, event_time)
        for path, event_time in zip(frame.massive_raw_file, frame.event_time_utc)
    ]
    ranked = frame.pre_event_mean_minute_volume.rank(method="first", pct=True)
    frame["liquidity_bucket"] = pd.cut(
        ranked,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["MICRO", "SMALL", "MID", "LARGE"],
        include_lowest=True,
    ).astype(str)
    frame["selection_hash"] = frame.event_id.astype(str).map(
        lambda value: hashlib.sha256(f"V221_PARITY_SAMPLE_V1|{value}".encode()).hexdigest()
    )
    # Round-robin source and liquidity strata without reading outcomes.
    selected = []
    groups = [
        group.sort_values("selection_hash")
        for _, group in frame.groupby(["source_family", "liquidity_bucket"], observed=True)
    ]
    offset = 0
    while len(selected) < min(max_events, len(frame)):
        changed = False
        for group in groups:
            if offset < len(group):
                selected.append(group.iloc[offset])
                changed = True
                if len(selected) >= min(max_events, len(frame)):
                    break
        if not changed:
            break
        offset += 1
    output = pd.DataFrame(selected).sort_values(["source_family", "liquidity_bucket", "selection_hash"])
    columns = [
        "event_id", "ticker", "issuer", "event_time_utc", "required_start_utc",
        "required_end_utc", "source_family", "role", "trade_date",
        "liquidity_bucket", "pre_event_mean_minute_volume", "massive_raw_file",
        "selection_hash",
    ]
    output = output[columns].reset_index(drop=True)
    atomic_csv(output, SAMPLE_PATH)
    return output


def build_provider_status(auth_error: str = "NOT_ATTEMPTED") -> pd.DataFrame:
    queue = pd.read_parquet(US_QUEUE)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    frame["massive_status"] = frame.status.astype(str)
    frame["alpaca_sip_status"] = auth_error
    frame["alpaca_iex_status"] = auth_error
    valid = frame.status.isin(["CACHED_VALID", "FETCHED_VALID"])
    frame["selected_price_provider"] = np.where(valid, "MASSIVE_SIP", "NONE")
    frame["provider_parity_eligible"] = False
    frame["exact_contract_status"] = np.where(valid, "MASSIVE_CACHE_REQUIRES_EVENT_TIMING_QA", "NOT_EXACT_ELIGIBLE")
    columns = [
        "event_id", "ticker", "event_time_utc", "source_family", "role",
        "massive_status", "alpaca_sip_status", "alpaca_iex_status",
        "selected_price_provider", "provider_parity_eligible", "exact_contract_status",
    ]
    output = frame[columns].copy()
    atomic_parquet(output, STATUS_PATH)
    return output


def build_backfill_queue(blocked_status: str = "BLOCKED_AUTH_HTTP_401") -> pd.DataFrame:
    queue = pd.read_parquet(US_QUEUE)
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "role_hash", "role_frozen_at"])
    frame = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if frame.role.isna().any():
        raise RuntimeError("ALPACA_BACKFILL_EVENT_WITHOUT_FROZEN_ROLE")
    frame = frame[
        frame.status.eq("DEFERRED_OUTSIDE_VERIFIED_ENTITLEMENT_ANCHOR")
        & frame.source_family.eq("US_SEC")
    ].copy()
    role_priority = {
        "DEV_EXTENSION": 0,
        "RESEARCH_SEAL_POOL": 1,
        "FINAL_META_RESERVE": 2,
    }
    frame["role_priority"] = frame.role.map(role_priority)
    if frame.role_priority.isna().any():
        raise RuntimeError("UNKNOWN_FROZEN_ROLE_IN_ALPACA_BACKFILL")
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["trade_date"] = frame.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)
    frame["selection_hash"] = frame.event_id.astype(str).map(
        lambda value: hashlib.sha256(f"V221_ALPACA_BACKFILL_V1|{value}".encode()).hexdigest()
    )
    frame = frame.sort_values(
        ["role_priority", "event_time_utc", "selection_hash"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    frame["backfill_rank"] = np.arange(1, len(frame) + 1)
    frame["alpaca_sip_status"] = blocked_status
    frame["alpaca_iex_status"] = blocked_status
    frame["selected_price_provider"] = "NONE"
    frame["provider_parity_eligible"] = False
    frame["exact_contract_status"] = "NOT_ATTEMPTED"
    frame["attempt_count"] = 0
    frame["last_error"] = blocked_status
    columns = [
        "backfill_rank", "event_id", "ticker", "issuer", "event_time_utc",
        "required_start_utc", "required_end_utc", "source_family", "role",
        "role_hash", "role_frozen_at", "role_priority", "trade_date",
        "selection_hash", "alpaca_sip_status", "alpaca_iex_status",
        "selected_price_provider", "provider_parity_eligible",
        "exact_contract_status", "attempt_count", "last_error",
    ]
    output = frame[columns].copy()
    atomic_parquet(output, BACKFILL_QUEUE_PATH)
    counts = {
        str(key): int(value)
        for key, value in output.groupby("role").size().to_dict().items()
    }
    inventory = {
        "by_year": {
            str(key): int(value)
            for key, value in output.groupby(output.event_time_utc.dt.year).size().to_dict().items()
        },
        "by_role": counts,
        "by_source_family": {
            str(key): int(value)
            for key, value in output.groupby("source_family").size().to_dict().items()
        },
        "unique_tickers": int(output.ticker.nunique()),
        "ticker_dates": int(output[["ticker", "trade_date"]].drop_duplicates().shape[0]),
    }
    audit = {
        "schema_version": 1,
        "built_at": now(),
        "queue_path": str(BACKFILL_QUEUE_PATH.relative_to(ROOT)),
        "queue_sha256": sha256(BACKFILL_QUEUE_PATH),
        "rows": int(len(output)),
        "selection_uses_outcomes": False,
        "existing_frozen_roles_preserved": True,
        "priority": ["DEV_EXTENSION", "RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"],
        "inventory": inventory,
        "network_requests": 0,
        "blocked_status": blocked_status,
    }
    atomic_json(audit, BACKFILL_AUDIT_PATH)
    # Extend the representative historical coverage audit without changing its
    # read-only probe outcome.
    coverage_path = DATA / "ALPACA_HISTORICAL_COVERAGE_AUDIT.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    coverage["deferred_candidate_inventory"] = inventory
    coverage["backfill_queue"] = {
        "path": str(BACKFILL_QUEUE_PATH.relative_to(ROOT)),
        "sha256": sha256(BACKFILL_QUEUE_PATH),
        "rows": int(len(output)),
    }
    coverage["full_deferred_candidate_backfill_started"] = False
    atomic_json(coverage, coverage_path)
    return output


def exact_row(sample: pd.Series, bars: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    event = pd.Series(
        {
            "event_id": sample.event_id,
            "market": "US",
            "ticker": sample.ticker,
            "event_time_utc": pd.Timestamp(sample.event_time_utc),
            "source": sample.source_family,
            "source_family": sample.source_family,
            "headline": "",
            "body": "",
            "form": "",
            "event_type": "OTHER",
        }
    )
    return app.event_to_row_exact_contract(event, bars, app.Config(official_exact_contract=True))


def compare_exact_pair(
    sample: pd.Series,
    massive_bars: pd.DataFrame,
    alpaca_bars: pd.DataFrame,
    feed: str,
) -> dict[str, Any]:
    start = pd.Timestamp(sample.required_start_utc)
    end = pd.Timestamp(sample.required_end_utc)
    massive_window = massive_bars[(massive_bars.datetime >= start) & (massive_bars.datetime <= end)]
    alpaca_window = alpaca_bars[(alpaca_bars.datetime >= start) & (alpaca_bars.datetime <= end)]
    massive_times = set(massive_window.datetime.astype("int64"))
    alpaca_times = set(alpaca_window.datetime.astype("int64"))
    union = massive_times | alpaca_times
    intersection = massive_times & alpaca_times
    massive_row, massive_reason = exact_row(sample, massive_bars)
    alpaca_row, alpaca_reason = exact_row(sample, alpaca_bars)
    result: dict[str, Any] = {
        "event_id": sample.event_id,
        "ticker": sample.ticker,
        "source_family": sample.source_family,
        "liquidity_bucket": sample.liquidity_bucket,
        "role": sample.role,
        "provider": "ALPACA",
        "feed": feed.upper(),
        "bar_coverage_ratio": len(intersection) / len(union) if union else 0.0,
        "timestamp_overlap": len(intersection),
        "massive_window_bars": len(massive_times),
        "alpaca_window_bars": len(alpaca_times),
        "missing_minute_rate": 1.0 - (len(intersection) / len(massive_times) if massive_times else 0.0),
        "massive_exact_reason": massive_reason,
        "alpaca_exact_reason": alpaca_reason,
        "both_exact": massive_row is not None and alpaca_row is not None,
        "alpaca_fetch_usable": True,
    }
    if massive_row is None or alpaca_row is None:
        result.update(
            {
                "entry_timestamp_agreement": False,
                "exit_timestamp_agreement": False,
                "entry_price_relative_diff": np.nan,
                "exit_price_relative_diff": np.nan,
                "fwd_ret_30m_massive": np.nan,
                "fwd_ret_30m_alpaca": np.nan,
                "label_agreement": False,
                "provider_disposition": "EXACT_CONTRACT_FAIL",
            }
        )
        return result
    entry_diff = abs(float(alpaca_row["entry_price"]) / float(massive_row["entry_price"]) - 1.0)
    exit_diff = abs(float(alpaca_row["exit_price"]) / float(massive_row["exit_price"]) - 1.0)
    label_agreement = int(massive_row["y"]) == int(alpaca_row["y"])
    result.update(
        {
            "entry_timestamp_agreement": pd.Timestamp(massive_row["actual_entry_time_utc"]) == pd.Timestamp(alpaca_row["actual_entry_time_utc"]),
            "exit_timestamp_agreement": pd.Timestamp(massive_row["actual_exit_time_utc"]) == pd.Timestamp(alpaca_row["actual_exit_time_utc"]),
            "entry_price_relative_diff": entry_diff,
            "exit_price_relative_diff": exit_diff,
            "fwd_ret_30m_massive": float(massive_row["fwd_ret_30m"]),
            "fwd_ret_30m_alpaca": float(alpaca_row["fwd_ret_30m"]),
            "label_agreement": label_agreement,
            "provider_disposition": "PAIR_OK" if label_agreement else "PROVIDER_LABEL_DISAGREEMENT",
        }
    )
    return result


def summarize(detail: pd.DataFrame, feed: str, policy: dict[str, Any]) -> dict[str, Any]:
    thresholds = policy["thresholds"]
    exact = detail[detail.both_exact].copy() if not detail.empty else detail.copy()
    correlation = (
        float(exact.fwd_ret_30m_massive.corr(exact.fwd_ret_30m_alpaca))
        if len(exact) >= 2
        else None
    )
    timing_agreement = (
        float((exact.entry_timestamp_agreement & exact.exit_timestamp_agreement).mean())
        if len(exact)
        else 0.0
    )
    massive_eligible = detail.massive_exact_reason.eq("OK") if len(detail) else pd.Series(dtype=bool)
    alpaca_eligible = detail.alpaca_exact_reason.eq("OK") if len(detail) else pd.Series(dtype=bool)
    fetched = (
        detail.alpaca_fetch_usable.astype(bool)
        if len(detail) and "alpaca_fetch_usable" in detail
        else pd.Series(False, index=detail.index)
    )
    eligibility_agreement = (
        float((massive_eligible[fetched] == alpaca_eligible[fetched]).mean())
        if fetched.any()
        else 0.0
    )
    metrics = {
        "n": int(len(detail)),
        "exact_pair_n": int(len(exact)),
        "exact_pair_rate": float(len(exact) / len(detail)) if len(detail) else 0.0,
        "alpaca_fetch_usable_rate": float(fetched.mean()) if len(detail) else 0.0,
        "exact_eligibility_agreement_on_fetched": eligibility_agreement,
        "bar_coverage": float(detail.bar_coverage_ratio.mean()) if len(detail) else 0.0,
        "entry_timestamp_agreement": float(exact.entry_timestamp_agreement.mean()) if len(exact) else 0.0,
        "exit_timestamp_agreement": float(exact.exit_timestamp_agreement.mean()) if len(exact) else 0.0,
        "timing_contract_agreement": timing_agreement,
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
        "provider": "ALPACA",
        "feed": feed.upper(),
        "metrics": metrics,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def run_parity(feed: str, max_events: int = 120, max_requests: int | None = None) -> dict[str, Any]:
    refresh_user_environment(ALPACA_ENV_NAMES)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("frozen_before_alpaca_overlap_results") is not True:
        raise RuntimeError("PARITY_POLICY_NOT_FROZEN")
    sample = build_overlap_sample(max_events=max_events)
    client = alpaca.AlpacaClient()
    detail_rows: list[dict[str, Any]] = []
    requests_used = 0
    for (ticker, date), group in sample.groupby(["ticker", "trade_date"], sort=True):
        if max_requests is not None and requests_used >= max_requests:
            break
        path = cache_path(ALPACA_CACHE, feed, str(ticker), str(date))
        if not path.is_file():
            start = pd.to_datetime(group.required_start_utc, utc=True).min()
            end = pd.to_datetime(group.required_end_utc, utc=True).max()
            bars, status = client.get_bars(str(ticker), start, end, feed)
            requests_used += int(status.get("requests", 0))
            if not status.get("ok") or bars.empty or not alpaca.bar_qa(bars).get("pass"):
                for row in group.itertuples(index=False):
                    detail_rows.append(
                        {
                            "event_id": row.event_id,
                            "ticker": row.ticker,
                            "source_family": row.source_family,
                            "liquidity_bucket": row.liquidity_bucket,
                            "role": row.role,
                            "provider": "ALPACA",
                            "feed": feed.upper(),
                            "bar_coverage_ratio": 0.0,
                            "timestamp_overlap": 0,
                            "massive_window_bars": 0,
                            "alpaca_window_bars": 0,
                            "missing_minute_rate": 1.0,
                            "massive_exact_reason": "NOT_EVALUATED",
                            "alpaca_exact_reason": status.get("error", "EMPTY_OR_QA_FAIL"),
                            "both_exact": False,
                            "alpaca_fetch_usable": False,
                            "entry_timestamp_agreement": False,
                            "exit_timestamp_agreement": False,
                            "entry_price_relative_diff": np.nan,
                            "exit_price_relative_diff": np.nan,
                            "fwd_ret_30m_massive": np.nan,
                            "fwd_ret_30m_alpaca": np.nan,
                            "label_agreement": False,
                            "provider_disposition": "ALPACA_FETCH_FAIL",
                        }
                    )
                if status.get("http_status") == 401:
                    break
                continue
            alpaca.atomic_parquet(bars, path)
        alpaca_bars = load_bars(path, "ALPACA", feed.upper())
        for row in group.itertuples(index=False):
            massive_bars = load_bars(ROOT / row.massive_raw_file, "MASSIVE", "SIP")
            detail_rows.append(compare_exact_pair(pd.Series(row._asdict()), massive_bars, alpaca_bars, feed))
    detail = pd.DataFrame(detail_rows)
    atomic_csv(detail, DETAIL_PATH)
    summary = {
        "schema_version": 1,
        "audited_at": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": sha256(POLICY_PATH),
        "selection_uses_outcomes": False,
        "label_level_parity_role": "DEV_EXTENSION",
        "sealed_role_labels_opened": False,
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "sample_rows": int(len(sample)),
        "detail_rows": int(len(detail)),
        "requests_used": requests_used,
        "result": summarize(detail, feed, policy),
        "backfill_authorized": False,
    }
    summary["backfill_authorized"] = summary["result"]["status"] == "PASS"
    atomic_json(summary, SUMMARY_PATH)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-sample", action="store_true")
    parser.add_argument("--build-provider-status", action="store_true")
    parser.add_argument("--build-backfill-queue", action="store_true")
    parser.add_argument("--auth-error", default="NOT_ATTEMPTED")
    parser.add_argument("--run-parity", choices=("sip", "iex"))
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--max-requests", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {}
    if args.build_sample:
        sample = build_overlap_sample(args.max_events)
        result["sample"] = {"rows": len(sample), "path": str(SAMPLE_PATH)}
    if args.build_provider_status:
        status = build_provider_status(args.auth_error)
        result["provider_status"] = {"rows": len(status), "path": str(STATUS_PATH)}
    if args.build_backfill_queue:
        backfill = build_backfill_queue(args.auth_error)
        result["backfill_queue"] = {"rows": len(backfill), "path": str(BACKFILL_QUEUE_PATH)}
    if args.run_parity:
        result["parity"] = run_parity(args.run_parity, args.max_events, args.max_requests)
    if not result:
        raise SystemExit("choose an action")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
