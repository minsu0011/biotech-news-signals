"""Frozen Alpaca SIP retry for annual-plus-6-K-exhibit candidate."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import alpaca_backfill_v221 as alpaca_backfill
import alpaca_provider_acquisition as alpaca
import annual_plus_sec_symbol_recovery_v221 as hybrid
import historical_symbol_resolver as resolver
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "ALPACA_ANNUAL_PLUS_6K_EXHIBIT_RETRY_POLICY_V1.json"
IDENTITY_POLICY_PATH = DATA / "HISTORICAL_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_POLICY_V1.json"
IDENTITY_AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_AUDIT.json"
PARITY_PATH = RESEARCH / "PROVIDER_PARITY_SUMMARY.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"
REPORT_PATH = RESEARCH / "ALPACA_ANNUAL_PLUS_6K_EXHIBIT_RETRY_REPORT.json"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "policy_id": "ALPACA_SIP_ANNUAL_PLUS_6K_EXHIBIT_RETRY_V1",
        "frozen_before_provider_results": True,
        "selection_uses_labels": False,
        "outcome_columns_loaded_for_selection": False,
        "selection_uses_price_values_or_timing_results": False,
        "provider": "ALPACA",
        "feed": "SIP",
        "exact_timing_contract_unchanged": True,
        "retry_each_symbol_date_once": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("6K_EXHIBIT_ALPACA_RETRY_POLICY_GUARD_FAIL")
    bindings = {
        resolver.QUEUE_PATH: policy["activation_queue_sha256"],
        DEV_PATH: policy["activation_dev_extension_sha256"],
        IDENTITY_AUDIT_PATH: policy["activation_identity_audit_sha256"],
        PARITY_PATH: policy["activation_parity_summary_sha256"],
        IDENTITY_POLICY_PATH: policy["required_identity_policy_sha256"],
    }
    for path, expected in bindings.items():
        if resolver.sha256(path) != expected:
            raise RuntimeError(f"6K_EXHIBIT_ALPACA_BINDING_MISMATCH:{path.name}")
    return policy, resolver.sha256(POLICY_PATH)


def select_candidates(queue: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    official_ids = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    mask = queue.source_family.astype(str).eq(str(policy["source_family"]))
    mask &= queue.role.astype(str).eq(str(policy["role"]))
    mask &= queue.annual_plus_6k_exhibit_status.astype(str).eq(
        str(policy["required_identity_status"])
    )
    mask &= queue.annual_plus_6k_exhibit_contract_sha256.astype(str).eq(
        str(policy["required_identity_policy_sha256"])
    )
    mask &= queue.resolution_confidence.astype(str).eq(
        str(policy["required_resolution_confidence"])
    )
    mask &= queue.resolution_asof_mode.astype(str).eq(
        str(policy["required_resolution_asof_mode"])
    )
    mask &= ~queue.event_id.astype(str).isin(official_ids)
    return queue.loc[mask].sort_values(["priority", "event_time_utc", "event_id"]).copy()


def main() -> int:
    policy, policy_sha = load_policy()
    queue = pd.read_parquet(resolver.QUEUE_PATH)
    candidates = select_candidates(queue, policy)
    digest = hybrid.event_ids_sha256(candidates.event_id.astype(str).tolist())
    if (
        len(candidates) != int(policy["candidate_count"])
        or digest != policy["candidate_event_ids_sha256"]
    ):
        raise RuntimeError("6K_EXHIBIT_ALPACA_CANDIDATE_SET_MISMATCH")
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = alpaca_backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    client = alpaca.AlpacaClient()
    results: Counter[str] = Counter()
    exact_indices: list[Any] = []
    requests_used = 0
    attempted_groups = 0
    for (symbol, date), group in candidates.groupby(
        ["resolved_ticker", "event_date"], sort=False
    ):
        if attempted_groups >= int(policy["maximum_symbol_date_requests"]):
            break
        attempted_groups += 1
        cache_path = provider._cache_path(
            provider.ALPACA_CACHE / "sip", str(symbol), date
        )
        cached = cache_path.is_file()
        bars = pd.DataFrame()
        status: dict[str, Any] = {"ok": True, "requests": 0}
        if cached:
            try:
                bars = pd.read_parquet(cache_path)
            except Exception:
                cached = False
        if not cached:
            start = pd.to_datetime(group.required_start_utc, utc=True).min()
            end = pd.to_datetime(group.required_end_utc, utc=True).max()
            bars, status = client.get_bars(
                str(symbol), start, end, "sip", asof=str(date)
            )
            requests_used += int(status.get("requests", 0))
            if status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache_path)
        qa_ok = bool(
            status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass")
        )
        raw_hash = resolver.sha256(cache_path) if qa_ok and cache_path.is_file() else ""
        for index in group.index:
            queue.loc[index, "annual_plus_6k_exhibit_alpaca_retry_policy_sha256"] = policy_sha
            queue.loc[index, "annual_plus_6k_exhibit_alpaca_retry_attempt_count"] = 1
            queue.loc[index, "provider"] = "ALPACA_SIP"
            queue.loc[index, "provider_symbol"] = str(symbol)
            if not qa_ok:
                result = "ALPACA_NO_BAR"
                queue.loc[index, "fetch_error"] = str(
                    status.get("error") or "NO_BAR_OR_QA_FAIL"
                )[:240]
                queue.loc[index, "exact_eligible"] = False
            else:
                exact, timing_reason, _ = alpaca_backfill.timing_only_eligibility(
                    pd.Timestamp(queue.loc[index, "event_time_utc"]), bars["timestamp"]
                )
                result = "ALPACA_FETCHED_VALID" if exact else f"TIMING_FAIL:{timing_reason}"
                queue.loc[index, "fetch_error"] = "" if exact else timing_reason
                queue.loc[index, "exact_eligible"] = bool(exact)
                queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT))
                queue.loc[index, "raw_sha256"] = raw_hash
                if exact:
                    exact_indices.append(index)
            queue.loc[index, "fetch_status"] = result
            results[result] += 1
        resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.update_provider_status(queue)
    resolver.write_aliases_and_timeline(queue)
    recovery = resolver.recovery_report(queue)
    report = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "policy_frozen_before_provider_results": True,
        "candidate_count": len(candidates),
        "candidate_event_ids_sha256": digest,
        "symbol_date_groups_attempted": attempted_groups,
        "network_requests": requests_used,
        "result_counts": dict(results),
        "new_exact_rows": len(exact_indices),
        "selection_uses_labels": False,
        "outcome_columns_loaded_for_selection": False,
        "exact_contract_relaxed": False,
        "roles_reassigned": False,
        "seal_or_final_labels_opened": False,
        "queue_sha256": recovery["queue_sha256"],
    }
    resolver.atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
