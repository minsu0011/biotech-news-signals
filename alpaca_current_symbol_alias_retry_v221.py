"""One-shot Alpaca current-symbol provider-alias retry for resolved history."""
from __future__ import annotations

import argparse
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
import historical_symbol_resolver as historical
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "ALPACA_CURRENT_SYMBOL_ALIAS_RETRY_POLICY_V1.json"
QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
NO_BAR_PATH = DATA / "HISTORICAL_NO_BAR_CLASSIFICATION.parquet"
STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
REPORT_PATH = RESEARCH / "ALPACA_CURRENT_SYMBOL_ALIAS_RETRY_REPORT.json"


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
        "policy_id": "ALPACA_CURRENT_SYMBOL_ALIAS_RETRY_V1",
        "frozen_before_retry_results": True,
        "selection_uses_labels": False,
        "outcome_columns_permitted": False,
        "provider": "ALPACA",
        "feed": "SIP",
        "request_mode": "CURRENT_CIK_SYMBOL_NO_ASOF",
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("ALPACA_CURRENT_ALIAS_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


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


def run(max_requests: int | None, years: set[int] | None) -> dict[str, Any]:
    policy, policy_hash = load_policy()
    allowed, reason = backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    queue = pd.read_parquet(QUEUE_PATH)
    audit = pd.read_parquet(NO_BAR_PATH)
    if audit.selection_uses_labels.astype(bool).any():
        raise RuntimeError("NO_BAR_AUDIT_LABEL_GUARD_FAIL")
    for column, default in (
        ("alpaca_current_alias_retry_status", "NOT_ATTEMPTED"),
        ("alpaca_current_alias_retry_policy_sha256", ""),
        ("alpaca_current_alias_retry_attempt_count", 0),
    ):
        if column not in queue:
            queue[column] = default
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
    pending &= queue.resolved_ticker.astype(str).ne(queue.ticker.astype(str))
    pending &= pd.to_numeric(
        queue.alpaca_current_alias_retry_attempt_count, errors="coerce"
    ).fillna(0).eq(0)
    if years is not None:
        pending &= event_year.isin(years)
    rank = {int(year): position for position, year in enumerate(policy["priority_years"])}
    candidates = queue.loc[pending].copy()
    candidates["_year_rank"] = event_year.loc[candidates.index].map(
        lambda year: rank.get(int(year), len(rank))
    )
    candidates = candidates.sort_values(["_year_rank", "event_time_utc", "event_id"])
    status = pd.read_parquet(STATUS_PATH).set_index("event_id")
    refresh_user_environment(ALPACA_ENV_NAMES)
    client = alpaca.AlpacaClient()
    requests_used = 0
    symbol_dates_attempted = 0
    results: Counter[str] = Counter()
    exact_ids: list[str] = []
    for (symbol, date), group in candidates.groupby(["ticker", "event_date"], sort=False):
        if max_requests is not None and requests_used >= max_requests:
            break
        cache_path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(symbol), str(date))
        cached = cache_path.is_file()
        bars = pd.DataFrame()
        call_status: dict[str, Any] = {"ok": True, "requests": 0}
        if cached:
            try:
                bars = pd.read_parquet(cache_path)
            except Exception:
                cached = False
        if not cached:
            bars, call_status = client.get_bars(
                str(symbol),
                pd.to_datetime(group.required_start_utc, utc=True).min(),
                pd.to_datetime(group.required_end_utc, utc=True).max(),
                "sip",
            )
            requests_used += int(call_status.get("requests", 0))
            symbol_dates_attempted += 1
            if call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache_path)
        qa_ok = bool(
            call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass")
        )
        for index, row in group.iterrows():
            queue.loc[index, "alpaca_current_alias_retry_attempt_count"] = 1
            queue.loc[index, "alpaca_current_alias_retry_policy_sha256"] = policy_hash
            if not qa_ok:
                result = "NO_BAR"
            else:
                exact, reason, _ = backfill.timing_only_eligibility(
                    pd.Timestamp(row.event_time_utc), bars.timestamp
                )
                result = "EXACT_TIMING_ELIGIBLE" if exact else str(reason)
                if exact:
                    raw_hash = sha256(cache_path)
                    queue.loc[index, "fetch_status"] = "ALPACA_FETCHED_VALID"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = True
                    queue.loc[index, "provider"] = "ALPACA_SIP_CURRENT_SYMBOL_ALIAS"
                    queue.loc[index, "provider_symbol"] = str(symbol)
                    queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT))
                    queue.loc[index, "raw_sha256"] = raw_hash
                    event_id = str(row.event_id)
                    if event_id not in status.index:
                        raise RuntimeError("CURRENT_ALIAS_EVENT_MISSING_PROVIDER_STATUS")
                    status.loc[event_id, "alpaca_sip_status"] = "FETCHED_VALID"
                    status.loc[event_id, "selected_price_provider"] = "ALPACA_SIP"
                    status.loc[event_id, "provider_parity_eligible"] = True
                    status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
                    status.loc[event_id, "provider_symbol"] = str(symbol)
                    status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
                    status.loc[event_id, "symbol_resolution_confidence"] = str(row.resolution_confidence)
                    status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
                    exact_ids.append(event_id)
            queue.loc[index, "alpaca_current_alias_retry_status"] = result
            results[result] += 1
        atomic_parquet(queue, QUEUE_PATH)
        if call_status.get("http_status") == 401:
            break
    atomic_parquet(status.reset_index(), STATUS_PATH)
    historical.write_aliases_and_timeline(queue)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "provider_parity_path": str(backfill.PARITY_PATH.relative_to(ROOT)),
        "provider_parity_sha256": sha256(backfill.PARITY_PATH),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "candidate_rows": int(len(candidates)),
        "candidate_symbol_dates": int(
            candidates[["ticker", "event_date"]].drop_duplicates().shape[0]
        ),
        "network_requests": int(requests_used),
        "symbol_dates_attempted": int(symbol_dates_attempted),
        "result_counts": dict(results),
        "exact_rows_added": int(len(exact_ids)),
        "exact_event_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(exact_ids)).encode()
        ).hexdigest(),
        "historical_queue_sha256": sha256(QUEUE_PATH),
        "provider_status_sha256": sha256(STATUS_PATH),
        "event_date_identity_preserved": True,
        "sealed_role_labels_opened": False,
    }
    atomic_json(report, REPORT_PATH)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--years")
    args = parser.parse_args()
    print(json.dumps(run(args.max_requests, parse_years(args.years)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
