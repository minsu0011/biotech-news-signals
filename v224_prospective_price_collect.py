"""Bounded, resume-safe raw Massive collector for V224 Track A.

Only the independent outcome-blind prospective role ledger is read.  This
collector writes raw one-minute bars to the independent V224 cache and never
computes or opens labels, returns, predictions, Seal, Final Meta, or central
provider state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import bio_news_30m_v3 as app
import v224_prospective_causal_one_shot as track_a
import v224_prospective_locked_evaluation as locked


ROOT = Path(__file__).resolve().parent
STATUS_REL = track_a.TRACK_REL / "PRICE_COLLECTION_STATUS.json"
ITEM_STATUS_REL = track_a.TRACK_REL / "price_collection"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def bar_qa(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        return {"pass": False, "reason": "EMPTY_OR_SCHEMA", "rows": int(len(frame))}
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work.timestamp, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    duplicate = int(work.timestamp.duplicated().sum())
    invalid_timestamp = int(work.timestamp.isna().sum())
    invalid_price = int((work[["open", "high", "low", "close"]].le(0) | work[["open", "high", "low", "close"]].isna()).any(axis=1).sum())
    bad_ohlc = int(((work.high < work[["open", "close", "low"]].max(axis=1)) | (work.low > work[["open", "close", "high"]].min(axis=1))).sum())
    negative_volume = int(work.volume.lt(0).sum())
    result = {
        "rows": int(len(work)), "duplicates": duplicate,
        "invalid_timestamp": invalid_timestamp, "invalid_price": invalid_price,
        "bad_ohlc": bad_ohlc, "negative_volume": negative_volume,
    }
    result["pass"] = not any((duplicate, invalid_timestamp, invalid_price, bad_ohlc, negative_volume))
    return result


def fetch_massive_day(ticker: str, trade_date: str, api_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import requests

    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"
    try:
        response = requests.get(
            url,
            params={"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=45,
        )
        payload = response.json()
        if response.status_code >= 400 or str(payload.get("status", "")).upper() not in {"OK", "DELAYED"}:
            return pd.DataFrame(), {
                "ok": False, "http_status": int(response.status_code),
                "error": str(payload.get("error") or payload.get("message") or "PROVIDER_ERROR")[:200],
            }
        raw = pd.DataFrame(payload.get("results") or [])
        if raw.empty:
            return pd.DataFrame(), {"ok": False, "http_status": int(response.status_code), "error": "EMPTY"}
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(raw.t, unit="ms", utc=True),
            "open": pd.to_numeric(raw.o, errors="coerce"),
            "high": pd.to_numeric(raw.h, errors="coerce"),
            "low": pd.to_numeric(raw.l, errors="coerce"),
            "close": pd.to_numeric(raw.c, errors="coerce"),
            "volume": pd.to_numeric(raw.v, errors="coerce"),
            "transactions": pd.to_numeric(raw.n, errors="coerce") if "n" in raw else 0,
            "vwap": pd.to_numeric(raw.vw, errors="coerce") if "vw" in raw else pd.NA,
        }).sort_values("timestamp").drop_duplicates("timestamp")
        qa = bar_qa(frame)
        return frame, {"ok": bool(qa["pass"]), "http_status": int(response.status_code), "qa": qa}
    except Exception as error:
        return pd.DataFrame(), {"ok": False, "error": f"{type(error).__name__}:{error}"[:200]}


def timing_coverage(timestamps: pd.Series, events: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    covered = 0
    for event_time in events.event_time_utc:
        target = pd.Timestamp(event_time).tz_convert("UTC") + pd.Timedelta(minutes=2)
        candidates = values[(values >= target) & (values <= target + pd.Timedelta(seconds=60))]
        if candidates.empty:
            continue
        entry = candidates.iloc[0]
        exit_target = entry + pd.Timedelta(minutes=30)
        exits = values[(values >= exit_target) & (values <= exit_target + pd.Timedelta(seconds=60))]
        if not exits.empty:
            covered += 1
    return {
        "events": int(len(events)),
        "exact_timestamp_windows_covered": int(covered),
        "all_exact_timestamp_windows_covered": bool(covered == len(events)),
        "price_values_loaded": False,
        "outcomes_computed": False,
    }


def item_status_path(root: Path, ticker: str, trade_date: str) -> Path:
    safe_ticker = "".join(character for character in ticker if character.isalnum() or character == "-")
    return root / ITEM_STATUS_REL / safe_ticker / f"{trade_date}.json"


def run(
    root: Path = ROOT,
    max_requests: int = 200,
    fetcher: Callable[[str, str, str], tuple[pd.DataFrame, dict[str, Any]]] = fetch_massive_day,
) -> dict[str, Any]:
    if not 0 <= max_requests <= 200:
        raise ValueError("TRACK_A_MAX_REQUESTS_MUST_BE_0_TO_200")
    policy, policy_sha = locked.verify_policy(root)
    locked.verify_frozen_files(root)
    roles = track_a.load_roles(root, policy_sha)
    exact = policy["exact_price_contract"]
    if exact.get("price_provider") != "MASSIVE" or exact.get("bar_cadence_seconds") != 60 or exact.get("bar_timestamp_semantics") != "OPEN":
        raise RuntimeError("TRACK_A_FROZEN_PROVIDER_POLICY_MISMATCH")
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    used = 0
    fetched = 0
    cached = 0
    deferred = 0
    failed = 0
    items: list[dict[str, Any]] = []
    if not roles.empty:
        local = roles.event_time_utc.dt.tz_convert(app.US_TZ)
        work = roles.assign(trade_date=local.dt.date.astype(str))
        groups = list(work.groupby(["ticker", "trade_date"], sort=True))
    else:
        groups = []
    for (ticker, trade_date), events in groups:
        ticker, trade_date = str(ticker), str(trade_date)
        cache = root / exact["independent_cache_root"] / trade_date[:4] / ticker / f"{trade_date}.parquet"
        coverage: dict[str, Any] | None = None
        cache_valid = False
        if cache.is_file():
            try:
                timestamps = pd.read_parquet(cache, columns=["timestamp"])["timestamp"]
                coverage = timing_coverage(timestamps, events)
                cache_valid = bool(coverage["all_exact_timestamp_windows_covered"])
            except Exception:
                cache_valid = False
        action = "CACHED_COMPLETE" if cache_valid else ""
        provider_result: dict[str, Any] = {}
        if cache_valid:
            cached += 1
        elif used >= max_requests:
            action = "DEFERRED_REQUEST_BUDGET"
            deferred += 1
        elif not key:
            action = "DEFERRED_CREDENTIAL_MISSING"
            deferred += 1
        else:
            bars, provider_result = fetcher(ticker, trade_date, key)
            used += 1
            qa = bar_qa(bars)
            if provider_result.get("ok") and qa["pass"]:
                new_coverage = timing_coverage(bars.timestamp, events)
                atomic_parquet(bars, cache)
                coverage = new_coverage
                fetched += 1
                action = "FETCHED_RAW_COMPLETE" if new_coverage["all_exact_timestamp_windows_covered"] else "FETCHED_RAW_TIMING_GAPS"
            else:
                failed += 1
                action = "PROVIDER_OR_QA_FAILED"
        item = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "ticker": ticker,
            "trade_date": trade_date,
            "role_event_ids": events.event_id.astype(str).tolist(),
            "action": action,
            "cache_path": str(cache.relative_to(root)).replace("\\", "/"),
            "cache_sha256": locked.sha256_file(cache) if cache.is_file() else None,
            "timing_coverage": coverage,
            "api_request_used": bool(action.startswith("FETCHED") or action == "PROVIDER_OR_QA_FAILED"),
            "provider_error": provider_result.get("error"),
            "credential_value_logged": False,
            "price_values_loaded_for_selection": False,
            "labels_or_returns_loaded": False,
            "outcomes_computed": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
        }
        convergent = track_a.convergent_json(item_status_path(root, ticker, trade_date), item)
        item["item_status_sha256"] = convergent
        items.append(item)
    status_name = (
        "ZERO_ROLE_ROWS_NO_REQUESTS" if roles.empty else
        "BOUNDED_COLLECTION_COMPLETE" if not deferred and not failed else
        "BOUNDED_COLLECTION_PARTIAL"
    )
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "phase": "RAW_PRICE_COLLECTION",
        "status": status_name,
        "role_rows": int(len(roles)),
        "ticker_days": int(len(groups)),
        "max_requests": int(max_requests),
        "requests_used": int(used),
        "raw_ticker_days_fetched": int(fetched),
        "cached_complete": int(cached),
        "deferred": int(deferred),
        "failed": int(failed),
        "items": items,
        "provider": "MASSIVE_READ_ONLY_AGGREGATES",
        "provider_policy_sha256": policy_sha,
        "credential_present": bool(key),
        "credential_value_logged": False,
        "one_shot": True,
        "poll_or_sleep": False,
        "labels_or_returns_loaded": False,
        "outcomes_computed": False,
        "predictions_loaded": False,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "central_queue_written": False,
        "central_epoch_written": False,
    }
    track_a.convergent_json(root / STATUS_REL, status)
    return status


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        return int(subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT, env=environment, check=False).returncode)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--max-requests", type=int, default=200)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    try:
        status = run(root, max_requests=args.max_requests)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "phase": "RAW_PRICE_COLLECTION",
            "status": "FAILED_CLOSED",
            "blocker": f"{type(error).__name__}:{error}",
            "labels_or_returns_loaded": False,
            "outcomes_computed": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
        }
        track_a.convergent_json(root / track_a.TRACK_REL / "PRICE_COLLECTION_FAILED_CLOSED.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
