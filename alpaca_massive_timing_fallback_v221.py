"""Parity-approved Alpaca SIP fallback for every DEV Massive timing miss."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import alpaca_backfill_v221 as backfill
import alpaca_provider_acquisition as alpaca
import alpaca_recent_timing_fallback_v221 as recent
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "ALPACA_ALL_DEV_MASSIVE_TIMING_FALLBACK_POLICY_V1.json"
QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
REPORT_PATH = RESEARCH / "ALPACA_ALL_DEV_MASSIVE_TIMING_FALLBACK_REPORT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "policy_name": "ALPACA_SIP_ALL_DEV_AFTER_MASSIVE_EVENT_TIMING_MISS_V1",
        "frozen_before_fallback_results": True,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "eligible_role": "DEV_EXTENSION",
        "eligible_source_family": "US_SEC",
        "required_primary_provider": "MASSIVE_SIP",
        "required_primary_cache_qa": "PASS",
        "required_primary_event_timing": "NOT_EXACT",
        "fallback_provider": "ALPACA_SIP",
        "required_fallback_parity": "PASS",
        "one_request_per_symbol_date": True,
        "request_mode": "NO_ASOF_PARAMETER",
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("ALL_DEV_ALPACA_FALLBACK_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


def main() -> int:
    policy, policy_hash = load_policy()
    allowed, reason = backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    queue = pd.read_parquet(QUEUE_PATH)
    roles = pd.read_parquet(ROLE_PATH)
    queue["role"] = queue.event_id.astype(str).map(
        roles.set_index("event_id").role.astype(str).to_dict()
    )
    queue["event_time_utc"] = pd.to_datetime(queue.event_time_utc, utc=True)
    queue["trade_date"] = queue.event_time_utc.dt.tz_convert(
        "America/New_York"
    ).dt.date.astype(str)
    candidates = queue.loc[
        queue.role.astype(str).eq(policy["eligible_role"])
        & queue.source_family.astype(str).eq(policy["eligible_source_family"])
    ].copy()
    primary_reasons: Counter[str] = Counter()
    fallback_indices: list[int] = []
    for index, row in candidates.iterrows():
        path = provider._cache_path(provider.MASSIVE_CACHE, str(row.ticker), row.trade_date)
        if not path.is_file():
            primary_reasons["MASSIVE_CACHE_MISSING"] += 1
            continue
        bars = pd.read_parquet(path)
        if bars.empty or not provider.bar_qa(bars, "US").get("pass"):
            primary_reasons["MASSIVE_QA_FAIL"] += 1
            continue
        exact, timing_reason, _ = backfill.timing_only_eligibility(
            pd.Timestamp(row.event_time_utc), bars.timestamp
        )
        primary_reasons["MASSIVE_EXACT" if exact else str(timing_reason)] += 1
        if not exact:
            fallback_indices.append(index)
    fallback = candidates.loc[fallback_indices].sort_values(["event_time_utc", "event_id"])
    status = recent.extend_status(
        queue.drop(columns=["role", "trade_date"]), roles
    ).set_index("event_id")
    already_exact = status.exact_contract_status.astype(str).eq("EXACT_TIMING_ELIGIBLE")
    fallback = fallback.loc[~fallback.event_id.astype(str).isin(set(status.index[already_exact]))]
    refresh_user_environment(ALPACA_ENV_NAMES)
    client = alpaca.AlpacaClient()
    result_counts: Counter[str] = Counter()
    requests_used = 0
    exact_ids: list[str] = []
    for (ticker, date), group in fallback.groupby(["ticker", "trade_date"], sort=False):
        path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(ticker), date)
        cached = path.is_file()
        bars = pd.DataFrame()
        call_status: dict[str, Any] = {"ok": True, "requests": 0}
        if cached:
            try:
                bars = pd.read_parquet(path)
            except Exception:
                cached = False
        if not cached:
            bars, call_status = client.get_bars(
                str(ticker),
                pd.to_datetime(group.required_start_utc, utc=True).min(),
                pd.to_datetime(group.required_end_utc, utc=True).max(),
                "sip",
            )
            requests_used += int(call_status.get("requests", 0))
            if call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, path)
        qa_ok = bool(
            call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass")
        )
        for _, row in group.iterrows():
            event_id = str(row.event_id)
            if not qa_ok:
                result = "ALPACA_NO_BAR_OR_QA_FAIL"
                status.loc[event_id, "alpaca_sip_status"] = result
            else:
                exact, timing_reason, _ = backfill.timing_only_eligibility(
                    pd.Timestamp(row.event_time_utc), bars.timestamp
                )
                result = "EXACT_TIMING_ELIGIBLE" if exact else str(timing_reason)
                status.loc[event_id, "alpaca_sip_status"] = (
                    "FETCHED_VALID" if exact else result
                )
                if exact:
                    status.loc[event_id, "selected_price_provider"] = "ALPACA_SIP"
                    status.loc[event_id, "provider_parity_eligible"] = True
                    status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
                    status.loc[event_id, "provider_symbol"] = str(ticker)
                    status.loc[event_id, "symbol_resolution_method"] = "CURRENT_EVENT_TICKER_EXACT_SEC"
                    status.loc[event_id, "symbol_resolution_confidence"] = "A"
                    status.loc[event_id, "resolution_asof_mode"] = "EVENT_DATE_CURRENT_LISTING"
                    exact_ids.append(event_id)
            result_counts[result] += 1
    atomic_parquet(status.reset_index(), STATUS_PATH)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "provider_parity_path": str(backfill.PARITY_PATH.relative_to(ROOT)),
        "provider_parity_sha256": sha256(backfill.PARITY_PATH),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "dev_source_candidates": int(len(candidates)),
        "primary_timing_counts": dict(primary_reasons),
        "fallback_candidate_rows": int(len(fallback)),
        "fallback_symbol_dates": int(
            fallback[["ticker", "trade_date"]].drop_duplicates().shape[0]
        ),
        "network_requests": int(requests_used),
        "fallback_result_counts": dict(result_counts),
        "exact_rows_added": int(len(exact_ids)),
        "exact_event_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(exact_ids)).encode()
        ).hexdigest(),
        "provider_status_sha256": sha256(STATUS_PATH),
        "sealed_role_labels_opened": False,
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
