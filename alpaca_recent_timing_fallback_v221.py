"""Use parity-approved Alpaca SIP after a recent Massive event-timing miss."""
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
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "ALPACA_RECENT_MASSIVE_TIMING_FALLBACK_POLICY_V1.json"
COLLECTION_STATUS_PATH = DATA / "V221_FRESH_SEC_COLLECTION_STATUS.json"
QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
REPORT_PATH = RESEARCH / "ALPACA_RECENT_MASSIVE_TIMING_FALLBACK_REPORT.json"


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
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "policy_name": "ALPACA_SIP_AFTER_MASSIVE_EVENT_TIMING_MISS_V1",
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
        raise RuntimeError("RECENT_ALPACA_FALLBACK_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


def extend_status(queue: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    status = pd.read_parquet(STATUS_PATH)
    existing = set(status.event_id.astype(str))
    missing = queue.loc[~queue.event_id.astype(str).isin(existing)].merge(
        roles[["event_id", "role"]], on="event_id", how="left", validate="one_to_one"
    )
    if missing.empty:
        return status
    additions = pd.DataFrame({
        "event_id": missing.event_id.astype(str),
        "ticker": missing.ticker.astype(str),
        "event_time_utc": missing.event_time_utc,
        "source_family": missing.source_family.astype(str),
        "role": missing.role.astype(str),
        "massive_status": missing.status.astype(str),
        "alpaca_sip_status": "NOT_ATTEMPTED",
        "alpaca_iex_status": "NOT_ATTEMPTED",
        "selected_price_provider": "NONE",
        "provider_parity_eligible": False,
        "exact_contract_status": "NOT_EXACT_ELIGIBLE",
        "provider_symbol": "",
        "symbol_resolution_method": "",
        "symbol_resolution_confidence": "",
        "resolution_asof_mode": "",
    })
    return pd.concat([status, additions], ignore_index=True, sort=False)


def main() -> int:
    policy, policy_hash = load_policy()
    expected_collection_hash = str(policy.get("eligible_collection_artifact_sha256", ""))
    if expected_collection_hash and sha256(COLLECTION_STATUS_PATH) != expected_collection_hash:
        raise RuntimeError("RECENT_ALPACA_FALLBACK_COLLECTION_HASH_MISMATCH")
    allowed, reason = backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    collection = json.loads(COLLECTION_STATUS_PATH.read_text(encoding="utf-8"))
    fresh_path = ROOT / str(collection["output_path"])
    fresh_ids = set(pd.read_csv(fresh_path, usecols=["event_id"]).event_id.astype(str))
    queue = pd.read_parquet(QUEUE_PATH)
    roles = pd.read_parquet(ROLE_PATH)
    role_map = roles.set_index("event_id").role.astype(str).to_dict()
    queue["role"] = queue.event_id.astype(str).map(role_map)
    queue["event_time_utc"] = pd.to_datetime(queue.event_time_utc, utc=True)
    queue["trade_date"] = queue.event_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    candidates = queue.loc[
        queue.event_id.astype(str).isin(fresh_ids)
        & queue.role.astype(str).eq(policy["eligible_role"])
        & queue.source_family.astype(str).eq(policy["eligible_source_family"])
    ].copy()
    primary_reasons: Counter[str] = Counter()
    eligible_indices: list[int] = []
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
        primary_reasons["MASSIVE_EXACT" if exact else timing_reason] += 1
        if not exact:
            eligible_indices.append(index)
    fallback = candidates.loc[eligible_indices].sort_values(["event_time_utc", "event_id"])
    status = extend_status(queue.drop(columns=["role", "trade_date"]), roles).set_index("event_id")
    refresh_user_environment(ALPACA_ENV_NAMES)
    client = alpaca.AlpacaClient()
    result_counts: Counter[str] = Counter()
    requests_used = 0
    exact_ids: list[str] = []
    for (ticker, date), group in fallback.groupby(["ticker", "trade_date"], sort=False):
        path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(ticker), date)
        bars = pd.DataFrame()
        call_status: dict[str, Any] = {"ok": True, "requests": 0}
        if path.is_file():
            bars = pd.read_parquet(path)
        else:
            bars, call_status = client.get_bars(
                str(ticker), group.required_start_utc.min(), group.required_end_utc.max(), "sip"
            )
            requests_used += int(call_status.get("requests", 0))
            if call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, path)
        qa_ok = bool(call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"))
        for _, row in group.iterrows():
            event_id = str(row.event_id)
            if not qa_ok:
                result = "ALPACA_NO_BAR_OR_QA_FAIL"
            else:
                exact, timing_reason, _ = backfill.timing_only_eligibility(
                    pd.Timestamp(row.event_time_utc), bars.timestamp
                )
                result = "EXACT_TIMING_ELIGIBLE" if exact else timing_reason
                if exact:
                    status.loc[event_id, "alpaca_sip_status"] = "FETCHED_VALID"
                    status.loc[event_id, "selected_price_provider"] = "ALPACA_SIP"
                    status.loc[event_id, "provider_parity_eligible"] = True
                    status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
                    status.loc[event_id, "provider_symbol"] = str(ticker)
                    status.loc[event_id, "symbol_resolution_method"] = str(
                        policy.get("symbol_resolution_method", "CURRENT_EVENT_TICKER_RECENT_FILING")
                    )
                    status.loc[event_id, "symbol_resolution_confidence"] = "A"
                    status.loc[event_id, "resolution_asof_mode"] = str(
                        policy.get("resolution_asof_mode", "EVENT_DATE_CURRENT_LISTING")
                    )
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
        "fresh_dev_candidates": int(len(candidates)),
        "primary_timing_counts": dict(primary_reasons),
        "fallback_candidate_rows": int(len(fallback)),
        "fallback_symbol_dates": int(fallback[["ticker", "trade_date"]].drop_duplicates().shape[0]),
        "network_requests": int(requests_used),
        "fallback_result_counts": dict(result_counts),
        "exact_rows_added": int(len(exact_ids)),
        "exact_event_id_set_sha256": hashlib.sha256("\n".join(sorted(exact_ids)).encode()).hexdigest(),
        "provider_status_sha256": sha256(STATUS_PATH),
        "sealed_role_labels_opened": False,
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
