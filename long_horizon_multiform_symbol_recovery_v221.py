"""Frozen, label-blind long-horizon SEC symbol corroboration for V221.

Identity is decided for the complete preregistered candidate set before price
cache availability or provider responses are inspected.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import alpaca_backfill_v221 as timing
import alpaca_provider_acquisition as alpaca
import historical_symbol_resolver as resolver
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_LONG_HORIZON_MULTIFORM_SYMBOL_POLICY_V1.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_LONG_HORIZON_MULTIFORM_SYMBOL_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def ids_sha256(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "policy_name": "CIK_PRIOR_LONG_HORIZON_MULTIFORM_SYMBOL_CONSENSUS_V1",
        "frozen_before_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "current_or_future_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("LONG_HORIZON_MULTIFORM_POLICY_GUARD_FAIL")
    if int(policy.get("minimum_independent_prior_filings", 0)) < 3:
        raise RuntimeError("LONG_HORIZON_REQUIRES_THREE_FILINGS")
    if int(policy.get("minimum_distinct_form_families", 0)) < 2:
        raise RuntimeError("LONG_HORIZON_REQUIRES_TWO_FORM_FAMILIES")
    return policy, resolver.sha256(POLICY_PATH)


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "long_horizon_multiform_status": "NOT_ATTEMPTED",
        "long_horizon_multiform_contract_sha256": "",
        "long_horizon_multiform_accessions": "[]",
        "long_horizon_multiform_evidence_files": "[]",
        "long_horizon_multiform_evidence_sha256": "[]",
        "long_horizon_multiform_evidence_count": 0,
        "long_horizon_multiform_form_families": "[]",
        "long_horizon_alpaca_retry_status": "NOT_ATTEMPTED",
        "long_horizon_alpaca_retry_attempt_count": 0,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def candidate_mask(queue: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    definition = policy["candidate_definition"]
    years = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    mask = queue.role.astype(str).eq(str(definition["role"]))
    mask &= queue.source_family.astype(str).eq(str(definition["source_family"]))
    mask &= years.isin({int(value) for value in definition["years"]})
    mask &= queue.resolution_confidence.astype(str).isin(
        {str(value) for value in definition["resolution_confidence_before"]}
    )
    mask &= queue.resolution_status.astype(str).isin(
        {str(value) for value in definition["resolution_status_before"]}
    )
    excluded = tuple(str(value).upper() for value in definition["event_registration_forms_excluded"])
    mask &= ~queue.form.astype(str).str.upper().str.startswith(excluded)
    return mask


def form_family(form: str, policy: dict[str, Any]) -> str:
    value = str(form).upper().strip()
    if value.endswith("/A"):
        value = value[:-2]
    for family, prefixes in policy["allowed_prior_form_families"].items():
        if any(value.startswith(str(prefix).upper()) for prefix in prefixes):
            return str(family)
    return ""


def evidence_record(
    cik: str,
    filing: dict[str, Any],
    days_before_event: float,
    family: str,
    client: resolver.SECArchiveClient,
    allowed_methods: set[str],
) -> tuple[dict[str, Any] | None, bool]:
    accession = str(filing.get("accessionNumber") or "")
    try:
        evidence, path, digest = resolver.inspect_filing_evidence(cik, accession, filing, client)
    except Exception:
        return None, False
    evidence = [item for item in evidence if str(item.get("method", "")) in allowed_methods]
    symbol, ambiguous = resolver.choose_symbol(evidence, "")
    if ambiguous:
        return None, True
    if not symbol or not path.is_file() or not digest:
        return None, False
    return {
        "accession": accession,
        "form": str(filing.get("form") or "").upper(),
        "form_family": family,
        "filing_time_utc": str(resolver.filing_time(filing) or ""),
        "days_before_event": round(float(days_before_event), 6),
        "symbol": resolver.normalize_symbol(symbol),
        "evidence_file": str(path.relative_to(ROOT)),
        "evidence_sha256": digest,
        "evidence_methods": sorted({str(item.get("method", "")) for item in evidence}),
        "evidence": evidence,
    }, False


def evaluate_records(
    records: list[dict[str, Any]], ambiguous_filings: int, policy: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    unique = {str(row["accession"]): row for row in records if row.get("accession")}
    records = sorted(unique.values(), key=lambda row: float(row["days_before_event"]))
    if ambiguous_filings and policy["ambiguous_symbol_evidence_forbidden"]:
        return "", "AMBIGUOUS_PRIOR_SYMBOL_EVIDENCE", records
    if len(records) < int(policy["minimum_independent_prior_filings"]):
        return "", "INSUFFICIENT_INDEPENDENT_PRIOR_FILINGS", records
    symbols = sorted({resolver.normalize_symbol(row.get("symbol")) for row in records} - {""})
    if len(symbols) > 1 and policy["contradictory_symbol_evidence_forbidden"]:
        return "", "CONTRADICTORY_PRIOR_SYMBOL_EVIDENCE", records
    if len(symbols) != 1:
        return "", "NO_SINGLE_MULTIFORM_CONSENSUS_SYMBOL", records
    families = {str(row["form_family"]) for row in records}
    if len(families) < int(policy["minimum_distinct_form_families"]):
        return "", "INSUFFICIENT_DISTINCT_FORM_FAMILIES", records
    days = [float(row["days_before_event"]) for row in records]
    if min(days) > float(policy["maximum_nearest_prior_days"]):
        return "", "NEAREST_PRIOR_OUTSIDE_WINDOW", records
    if max(days) > float(policy["maximum_oldest_prior_days"]):
        return "", "OLDEST_PRIOR_OUTSIDE_WINDOW", records
    if max(days) - min(days) < float(policy["minimum_observed_symbol_span_days"]):
        return "", "OBSERVED_SYMBOL_SPAN_TOO_SHORT", records
    if policy["evidence_file_sha256_required"] and any(
        not str(row.get("evidence_sha256", "")) for row in records
    ):
        return "", "EVIDENCE_SHA_MISSING", records
    return symbols[0], "PASS", records


def corroborate_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_hash: str,
) -> dict[str, Any]:
    cik, event_accession = resolver.parse_event_id(str(row.event_id))
    base = {
        "promoted": False,
        "long_horizon_multiform_status": "NOT_ELIGIBLE",
        "long_horizon_multiform_contract_sha256": policy_hash,
        "long_horizon_multiform_accessions": "[]",
        "long_horizon_multiform_evidence_files": "[]",
        "long_horizon_multiform_evidence_sha256": "[]",
        "long_horizon_multiform_evidence_count": 0,
        "long_horizon_multiform_form_families": "[]",
    }
    if not cik or not event_accession:
        return {**base, "long_horizon_multiform_status": "CIK_OR_ACCESSION_MISSING"}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {**base, "long_horizon_multiform_status": f"SEC_SUBMISSIONS_{type(error).__name__}"}
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    prior: list[tuple[float, dict[str, Any], str]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        stamp = resolver.filing_time(filing)
        family = form_family(str(filing.get("form") or ""), policy)
        if not accession or accession == event_accession or stamp is None or stamp >= event_time or not family:
            continue
        days = (event_time - stamp).total_seconds() / 86400.0
        if 0 <= days <= float(policy["maximum_oldest_prior_days"]):
            prior.append((days, filing, family))
    prior = sorted(prior, key=lambda item: item[0])[: int(policy["maximum_filings_scanned"])]
    allowed_methods = {str(value) for value in policy["allowed_explicit_symbol_methods"]}
    records: list[dict[str, Any]] = []
    ambiguous_filings = 0
    for days, filing, family in prior:
        record, ambiguous = evidence_record(
            cik, filing, days, family, client, allowed_methods
        )
        ambiguous_filings += int(ambiguous)
        if record:
            records.append(record)
    symbol, reason, audit_records = evaluate_records(records, ambiguous_filings, policy)
    result = {
        **base,
        "long_horizon_multiform_status": reason,
        "long_horizon_multiform_accessions": json.dumps(
            [record["accession"] for record in audit_records], ensure_ascii=False
        ),
        "long_horizon_multiform_evidence_files": json.dumps(
            [record["evidence_file"] for record in audit_records], ensure_ascii=False
        ),
        "long_horizon_multiform_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in audit_records], ensure_ascii=False
        ),
        "long_horizon_multiform_evidence_count": len(audit_records),
        "long_horizon_multiform_form_families": json.dumps(
            sorted({record["form_family"] for record in audit_records}), ensure_ascii=False
        ),
    }
    if not symbol:
        return result
    nearest = audit_records[0]
    return {
        **result,
        "promoted": True,
        "resolved_ticker": symbol,
        "ticker_candidates": json.dumps(nearest["evidence"], ensure_ascii=False, sort_keys=True),
        "resolution_method": policy["promoted_resolution_method"],
        "resolution_confidence": policy["promoted_resolution_confidence"],
        "resolution_asof_mode": policy["promoted_resolution_asof_mode"],
        "evidence_file": nearest["evidence_file"],
        "evidence_sha256": nearest["evidence_sha256"],
        "evidence_accession": nearest["accession"],
        "evidence_filing_time_utc": nearest["filing_time_utc"],
        "resolution_error": "",
    }


def update_provider_status(queue: pd.DataFrame, exact_indices: list[Any]) -> None:
    if not exact_indices:
        return
    status = pd.read_parquet(resolver.PROVIDER_STATUS_PATH).set_index("event_id")
    for index in exact_indices:
        row = queue.loc[index]
        event_id = str(row.event_id)
        if event_id not in status.index:
            raise RuntimeError("LONG_HORIZON_EVENT_MISSING_PROVIDER_STATUS")
        status.loc[event_id, "alpaca_sip_status"] = "FETCHED_VALID"
        status.loc[event_id, "selected_price_provider"] = "ALPACA_SIP"
        status.loc[event_id, "provider_parity_eligible"] = True
        status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
        status.loc[event_id, "provider_symbol"] = str(row.resolved_ticker)
        status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
        status.loc[event_id, "symbol_resolution_confidence"] = str(row.resolution_confidence)
        status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
    resolver.atomic_parquet(status.reset_index(), resolver.PROVIDER_STATUS_PATH)


def retry_prices_after_identity(
    queue: pd.DataFrame, promoted_indices: list[Any], policy_hash: str
) -> tuple[list[Any], Counter[str], int]:
    if not promoted_indices:
        return [], Counter(), 0
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = timing.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    client = alpaca.AlpacaClient()
    promoted = queue.loc[promoted_indices].copy()
    promoted["event_date"] = pd.to_datetime(
        promoted.event_time_utc, utc=True
    ).dt.tz_convert("America/New_York").dt.date.astype(str)
    exact_indices: list[Any] = []
    counts: Counter[str] = Counter()
    requests_used = 0
    for (symbol, date), group in promoted.groupby(["resolved_ticker", "event_date"], sort=False):
        cache_path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(symbol), date)
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
            if call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache_path)
        qa_ok = bool(call_status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"))
        raw_hash = resolver.sha256(cache_path) if qa_ok and cache_path.is_file() else ""
        for index in group.index:
            queue.loc[index, "long_horizon_alpaca_retry_attempt_count"] = 1
            queue.loc[index, "long_horizon_multiform_contract_sha256"] = policy_hash
            queue.loc[index, "attempt_count"] = int(queue.loc[index, "attempt_count"]) + (0 if cached else 1)
            queue.loc[index, "provider"] = "ALPACA_SIP"
            queue.loc[index, "provider_symbol"] = str(symbol)
            if not qa_ok:
                result = "ALPACA_NO_BAR"
                queue.loc[index, "fetch_status"] = result
                queue.loc[index, "fetch_error"] = str(call_status.get("error") or "NO_BAR_OR_QA_FAIL")[:240]
                queue.loc[index, "exact_eligible"] = False
                queue.loc[index, "raw_file"] = ""
                queue.loc[index, "raw_sha256"] = ""
            else:
                exact, timing_reason, _ = timing.timing_only_eligibility(
                    pd.Timestamp(queue.loc[index, "event_time_utc"]), bars.timestamp
                )
                result = "EXACT_TIMING_ELIGIBLE" if exact else str(timing_reason)
                queue.loc[index, "fetch_status"] = "ALPACA_FETCHED_VALID" if exact else "TIMING_FAIL"
                queue.loc[index, "fetch_error"] = "" if exact else str(timing_reason)
                queue.loc[index, "exact_eligible"] = bool(exact)
                queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT))
                queue.loc[index, "raw_sha256"] = raw_hash
                if exact:
                    exact_indices.append(index)
            queue.loc[index, "long_horizon_alpaca_retry_status"] = result
            counts[result] += 1
    return exact_indices, counts, requests_used


def main() -> int:
    policy, policy_hash = load_policy()
    if resolver.sha256(resolver.QUEUE_PATH) != str(policy["activation_queue_sha256"]):
        raise RuntimeError("LONG_HORIZON_ACTIVATION_QUEUE_HASH_MISMATCH")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    selected = candidate_mask(queue, policy)
    indices = list(queue.loc[selected].sort_values(["priority", "event_time_utc", "event_id"]).index)
    selected_ids = queue.loc[indices, "event_id"].astype(str).tolist()
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("LONG_HORIZON_CANDIDATE_COUNT_MISMATCH")
    if ids_sha256(selected_ids) != str(policy["candidate_event_ids_sha256"]):
        raise RuntimeError("LONG_HORIZON_CANDIDATE_HASH_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    grouped: dict[str, list[Any]] = {}
    for index in indices:
        grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
    status_counts: Counter[str] = Counter()
    promoted_indices: list[Any] = []
    sec_requests = 0
    sec_cache_hits = 0

    def group_work(group: list[Any]) -> tuple[list[tuple[Any, dict[str, Any]]], int, int]:
        client = resolver.SECArchiveClient(min_interval=0.55)
        rows = [
            (index, corroborate_one(queue.loc[index].copy(), client, policy, policy_hash))
            for index in group
        ]
        return rows, client.network_requests, client.cache_hits

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(grouped)))) as executor:
        futures = [executor.submit(group_work, group) for group in grouped.values()]
        for future in as_completed(futures):
            rows, requests_used, hits = future.result()
            sec_requests += requests_used
            sec_cache_hits += hits
            for index, result in rows:
                promoted = bool(result.pop("promoted", False))
                status_counts[str(result.get("long_horizon_multiform_status", "UNKNOWN"))] += 1
                for key, value in result.items():
                    queue.loc[index, key] = value
                if promoted:
                    symbol = resolver.normalize_symbol(result["resolved_ticker"])
                    queue.loc[index, "resolution_status"] = (
                        "RESOLVED_SAME_TICKER"
                        if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                        else "RESOLVED_NEW_TICKER"
                    )
                    queue.loc[index, "fetch_status"] = "LONG_HORIZON_IDENTITY_PRICE_PENDING"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = False
                    promoted_indices.append(index)

    # Identity selection is now complete for every frozen candidate. Price
    # caches and the network provider are first consulted below this boundary.
    exact_indices, price_counts, alpaca_requests = retry_prices_after_identity(
        queue, promoted_indices, policy_hash
    )
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    update_provider_status(queue, exact_indices)
    resolver.write_aliases_and_timeline(queue)
    recovery = resolver.recovery_report(queue)
    promoted = queue.loc[promoted_indices] if promoted_indices else queue.iloc[0:0]
    exact_new = [
        index for index in exact_indices
        if str(queue.loc[index, "event_id"]) not in official_before
    ]
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "policy_frozen_before_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded_for_identity": False,
        "selected_rows": len(indices),
        "selected_ciks": len(grouped),
        "selected_event_ids_sha256": ids_sha256(selected_ids),
        "identity_status_counts": dict(status_counts),
        "promoted_to_B": len(promoted_indices),
        "promoted_same_ticker": int((promoted.resolved_ticker.astype(str) == promoted.ticker.astype(str)).sum()) if len(promoted) else 0,
        "promoted_changed_ticker": int((promoted.resolved_ticker.astype(str) != promoted.ticker.astype(str)).sum()) if len(promoted) else 0,
        "post_identity_price_result_counts": dict(price_counts),
        "new_exact_rows_after_promotion": len(exact_new),
        "new_exact_event_id_set_sha256": ids_sha256(
            [str(queue.loc[index, "event_id"]) for index in exact_new]
        ),
        "SEC_network_requests": sec_requests,
        "SEC_cache_hits": sec_cache_hits,
        "alpaca_network_requests": alpaca_requests,
        "queue_sha256": recovery["queue_sha256"],
        "exact_contract_relaxed": False,
        "roles_reassigned": False,
        "seal_or_final_labels_opened": False,
    }
    resolver.atomic_json(audit, AUDIT_PATH)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
