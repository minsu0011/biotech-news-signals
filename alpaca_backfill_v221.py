"""Persistent, fail-closed Alpaca backfill worker for frozen V221+ US_SEC roles."""
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

import alpaca_provider_acquisition as alpaca
import bio_news_30m_v3 as app
import provider_parity_v221 as parity
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
QUEUE_PATH = DATA / "ALPACA_BACKFILL_QUEUE.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
PARITY_PATH = RESEARCH / "PROVIDER_PARITY_SUMMARY.json"
LEDGER_PATH = RESEARCH / "ALPACA_ACQUISITION_LEDGER.csv"
STATE_PATH = ROOT / "state" / "ALPACA_BACKFILL_STATE.json"
CONTRACT_VERSION = "EXACT_T2_30M_V36"
ALPACA_SIP_HISTORY_START_UTC = pd.Timestamp("2016-01-01T00:00:00Z")


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def append_ledger(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "timestamp", "provider", "feed", "ticker_date", "roles", "events_requested",
        "requests", "success", "cached", "denied", "rate_limited", "bars",
        "exact_eligible", "timing_rejected", "parity_rejected", "raw_sha256",
    ]
    new = pd.DataFrame(rows, columns=columns)
    old = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.is_file() else pd.DataFrame(columns=columns)
    combined = pd.concat([old, new], ignore_index=True)
    temporary = LEDGER_PATH.with_name(LEDGER_PATH.name + f".tmp.{os.getpid()}")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, LEDGER_PATH)


def timing_only_eligibility(
    event_time_utc: pd.Timestamp,
    timestamps: pd.Series,
) -> tuple[bool, str, dict[str, Any]]:
    """Apply t+2/+30m using timestamps only; no prices or outcomes are read."""
    values = pd.to_datetime(timestamps, utc=True).sort_values().drop_duplicates().array.asi8
    event = pd.Timestamp(event_time_utc)
    event = event.tz_localize("UTC") if event.tzinfo is None else event.tz_convert("UTC")
    target = event + pd.Timedelta(minutes=2)
    position = int(np.searchsorted(values, int(target.value)))
    if position >= len(values):
        return False, "BAR_MISSING", {}
    entry = pd.Timestamp(values[position], tz="UTC")
    entry_slip = float((entry - target).total_seconds())
    if not 0 <= entry_slip <= 60:
        return False, "ENTRY_TIMING_MISS", {"entry_slippage_sec": entry_slip}
    exit_target = entry + pd.Timedelta(minutes=30)
    exit_position = int(np.searchsorted(values, int(exit_target.value)))
    if exit_position >= len(values):
        return False, "BAR_MISSING", {"entry_slippage_sec": entry_slip}
    exit_time = pd.Timestamp(values[exit_position], tz="UTC")
    exit_slip = float((exit_time - exit_target).total_seconds())
    hold = float((exit_time - entry).total_seconds())
    if not 0 <= exit_slip <= 60:
        return False, "EXIT_TIMING_MISS", {
            "entry_slippage_sec": entry_slip,
            "exit_slippage_sec": exit_slip,
            "hold_seconds": hold,
        }
    if not 1800 <= hold <= 1860:
        return False, "HOLD_TIMING_MISS", {
            "entry_slippage_sec": entry_slip,
            "exit_slippage_sec": exit_slip,
            "hold_seconds": hold,
        }
    return True, "EXACT_TIMING_ELIGIBLE", {
        "entry_slippage_sec": entry_slip,
        "exit_slippage_sec": exit_slip,
        "hold_seconds": hold,
    }


def authorized_feed(feed: str) -> tuple[bool, str]:
    if not PARITY_PATH.is_file():
        return False, "PROVIDER_PARITY_REPORT_MISSING"
    report = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    result = report.get("result", {})
    if str(result.get("feed", "")).lower() != feed.lower():
        return False, "PROVIDER_PARITY_FEED_MISMATCH"
    if result.get("status") != "PASS" or report.get("backfill_authorized") is not True:
        return False, "PROVIDER_PARITY_FAIL"
    if report.get("sealed_role_labels_opened") is not False:
        return False, "SEALED_ROLE_LABEL_GUARD_FAIL"
    return True, "PASS"


def absolute_alpaca_targets(
    us_dev_total: int,
    eligible_by_role: dict[str, Any],
    alpaca_achieved: Counter[str],
) -> dict[str, int]:
    """Return absolute Alpaca contributions needed for each frozen role.

    The readiness/timing totals already include successful Alpaca rows.  A
    dynamic shortfall (for example ``120 - current_total``) therefore cannot
    be compared directly with Alpaca's cumulative contribution: doing so
    double-counts earlier Alpaca successes and may skip remaining roles.  The
    stable target is the required total minus the non-Alpaca baseline.
    """
    current_totals = {
        "DEV_EXTENSION": int(us_dev_total),
        "RESEARCH_SEAL_POOL": int(eligible_by_role.get("US:RESEARCH_SEAL_POOL", 0)),
        "FINAL_META_RESERVE": int(eligible_by_role.get("US:FINAL_META_RESERVE", 0)),
    }
    required_totals = {
        "DEV_EXTENSION": 120,
        "RESEARCH_SEAL_POOL": 250,
        "FINAL_META_RESERVE": 250,
    }
    return {
        role: max(0, required_totals[role] - (current_totals[role] - int(alpaca_achieved[role])))
        for role in required_totals
    }


def role_targets(queue: pd.DataFrame) -> dict[str, int]:
    readiness_path = RESEARCH / "V70_NEW_DATA_EPOCH_READINESS.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    us_dev = int(readiness["counts"]["by_source_family"]["US_SEC"]["n"])
    timing_path = DATA / "V59_EXACT_TIMING_ELIGIBILITY.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    alpaca_achieved = Counter(
        queue.loc[queue.exact_contract_status.eq("EXACT_TIMING_ELIGIBLE"), "role"].astype(str)
    )
    return absolute_alpaca_targets(us_dev, timing.get("eligible_by_role", {}), alpaca_achieved)


def group_is_settled(group: pd.DataFrame, feed: str, retry_asof: bool = False) -> bool:
    column = f"alpaca_{feed.lower()}_status"
    settled = {
        "CACHED_VALID",
        "FETCHED_VALID",
        "PROVIDER_FAILED",
        "DEFERRED_OUTSIDE_ALPACA_COVERAGE_ANCHOR",
    }
    if retry_asof:
        retryable = group[column].astype(str).eq("PROVIDER_FAILED") & group.last_error.astype(str).eq("NO_BAR_OR_QA_FAIL")
        if retryable.any():return False
    return bool(len(group) and group[column].astype(str).isin(settled).all())


def defer_outside_alpaca_anchor(queue: pd.DataFrame, feed: str) -> tuple[pd.DataFrame, int]:
    result = queue.copy()
    column = f"alpaca_{feed.lower()}_status"
    event_time = pd.to_datetime(result.event_time_utc, utc=True)
    unsettled = ~result[column].astype(str).isin(
        {"CACHED_VALID", "FETCHED_VALID", "PROVIDER_FAILED", "DEFERRED_OUTSIDE_ALPACA_COVERAGE_ANCHOR"}
    )
    selected = unsettled & event_time.lt(ALPACA_SIP_HISTORY_START_UTC)
    result.loc[selected, column] = "DEFERRED_OUTSIDE_ALPACA_COVERAGE_ANCHOR"
    result.loc[selected, "last_error"] = "ALPACA_SIP_HISTORY_START_2016_01_01"
    return result, int(selected.sum())


def _update_status_view(queue: pd.DataFrame, feed: str) -> None:
    if not STATUS_PATH.is_file():
        parity.build_provider_status("NOT_ATTEMPTED")
    status = pd.read_parquet(STATUS_PATH).set_index("event_id")
    queue_indexed = queue.set_index("event_id")
    common = status.index.intersection(queue_indexed.index)
    feed_column = f"alpaca_{feed.lower()}_status"
    status.loc[common, feed_column] = queue_indexed.loc[common, feed_column].astype(str)
    eligible = queue_indexed.loc[common, "exact_contract_status"].eq("EXACT_TIMING_ELIGIBLE")
    eligible_ids = common[eligible.to_numpy()]
    status.loc[eligible_ids, "selected_price_provider"] = f"ALPACA_{feed.upper()}"
    status.loc[eligible_ids, "provider_parity_eligible"] = True
    status.loc[eligible_ids, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
    atomic_parquet(status.reset_index(), STATUS_PATH)


def run_backfill(feed: str, max_requests: int | None = None, retry_asof: bool = False) -> dict[str, Any]:
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = authorized_feed(feed)
    if not allowed:
        report = {
            "schema_version": 1,
            "updated_at": now(),
            "status": "BLOCKED",
            "reason": reason,
            "feed": feed.upper(),
            "network_requests": 0,
            "values_serialized": False,
        }
        atomic_json(report, STATE_PATH)
        return report
    queue = pd.read_parquet(QUEUE_PATH)
    queue, deferred = defer_outside_alpaca_anchor(queue, feed)
    if deferred:
        atomic_parquet(queue, QUEUE_PATH)
        _update_status_view(queue, feed)
    targets = role_targets(queue)
    achieved = Counter(
        queue.loc[queue.exact_contract_status.eq("EXACT_TIMING_ELIGIBLE"), "role"].astype(str)
    )
    client = alpaca.AlpacaClient()
    requests_used = 0
    ledger: list[dict[str, Any]] = []
    for (ticker, date), group in queue.sort_values("backfill_rank").groupby(
        ["ticker", "trade_date"], sort=False
    ):
        roles = set(group.role.astype(str))
        if all(achieved[role] >= targets[role] for role in targets):
            break
        if not any(achieved[role] < targets[role] for role in roles):
            continue
        if group_is_settled(group, feed, retry_asof):
            continue
        if max_requests is not None and requests_used >= max_requests:
            break
        cache = parity.cache_path(alpaca.ALPACA_CACHE, feed, str(ticker), str(date))
        cached = cache.is_file()
        bars = pd.DataFrame()
        status: dict[str, Any] = {"ok": True, "requests": 0, "feed": feed}
        if cached:
            try:
                bars = pd.read_parquet(cache)
            except Exception:
                cached = False
        if not cached:
            start = pd.to_datetime(group.required_start_utc, utc=True).min()
            end = pd.to_datetime(group.required_end_utc, utc=True).max()
            bars, status = client.get_bars(str(ticker), start, end, feed, asof=str(date) if retry_asof else None)
            requests_used += int(status.get("requests", 0))
            if status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache)
        qa_ok = bool(status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"))
        indices = group.index
        queue.loc[indices, "attempt_count"] = queue.loc[indices, "attempt_count"].astype(int) + (0 if cached else 1)
        feed_column = f"alpaca_{feed.lower()}_status"
        exact_count = 0
        reason_counts = Counter()
        if qa_ok:
            timestamps = bars["timestamp"]
            raw_hash = sha256(cache)
            for index in indices:
                ok, timing_reason, _ = timing_only_eligibility(queue.loc[index, "event_time_utc"], timestamps)
                reason_counts[timing_reason] += 1
                queue.loc[index, feed_column] = "FETCHED_VALID" if not cached else "CACHED_VALID"
                queue.loc[index, "provider_parity_eligible"] = True
                queue.loc[index, "selected_price_provider"] = f"ALPACA_{feed.upper()}" if ok else "NONE"
                queue.loc[index, "exact_contract_status"] = timing_reason
                queue.loc[index, "last_error"] = "" if ok else timing_reason
                if ok:
                    achieved[str(queue.loc[index, "role"])] += 1
                    exact_count += 1
            raw_sha = raw_hash
        else:
            failure = str(status.get("error", "NO_BAR_OR_QA_FAIL"))[:240]
            if retry_asof and failure=="NO_BAR_OR_QA_FAIL":failure="ASOF_NO_BAR_OR_QA_FAIL"
            queue.loc[indices, feed_column] = "PROVIDER_FAILED"
            queue.loc[indices, "last_error"] = failure
            reason_counts[failure] += len(indices)
            raw_sha = ""
        atomic_parquet(queue, QUEUE_PATH)
        _update_status_view(queue, feed)
        ledger.append(
            {
                "timestamp": now(),
                "provider": "ALPACA",
                "feed": feed.upper(),
                "ticker_date": f"{ticker}/{date}",
                "roles": "|".join(sorted(roles)),
                "events_requested": len(group),
                "requests": int(status.get("requests", 0)),
                "success": int(qa_ok),
                "cached": int(cached),
                "denied": int(status.get("http_status") in {401, 403}),
                "rate_limited": int(status.get("rate_limited", 0)),
                "bars": len(bars),
                "exact_eligible": exact_count,
                "timing_rejected": len(group) - exact_count if qa_ok else 0,
                "parity_rejected": 0,
                "raw_sha256": raw_sha,
            }
        )
        atomic_json(
            {
                "schema_version": 1,
                "updated_at": now(),
                "status": "RUNNING",
                "feed": feed.upper(),
                "network_requests": requests_used,
                "deferred_outside_coverage_anchor": deferred,
                "corporate_action_asof_retry": retry_asof,
                "last_ticker_date": f"{ticker}/{date}",
                "targets": targets,
                "achieved_this_queue": dict(achieved),
                "values_serialized": False,
            },
            STATE_PATH,
        )
        if status.get("http_status") == 401:
            break
    append_ledger(ledger)
    final = {
        "schema_version": 1,
        "updated_at": now(),
        "status": "TARGETS_MET" if all(achieved[role] >= targets[role] for role in targets) else "INCOMPLETE",
        "feed": feed.upper(),
        "network_requests": requests_used,
        "deferred_outside_coverage_anchor": deferred,
        "corporate_action_asof_retry": retry_asof,
        "targets": targets,
        "achieved_this_queue": dict(achieved),
        "queue_sha256": sha256(QUEUE_PATH),
        "values_serialized": False,
    }
    atomic_json(final, STATE_PATH)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", choices=("sip", "iex"), required=True)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--retry-asof", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_backfill(args.feed, args.max_requests, args.retry_asof)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"TARGETS_MET", "INCOMPLETE"} else 20


if __name__ == "__main__":
    raise SystemExit(main())
