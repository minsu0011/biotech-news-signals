"""Recover frozen DEV symbols from prior annual + SEC prospectus evidence."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import annual_plus_sec_symbol_recovery_v221 as hybrid
import annual_report_symbol_recovery_v221 as annual
import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_ANNUAL_PLUS_PROSPECTUS_SYMBOL_CONSENSUS_POLICY_V1.json"
PRIOR_AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_PLUS_SEC_SYMBOL_CONSENSUS_AUDIT.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_PLUS_PROSPECTUS_SYMBOL_CONSENSUS_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "frozen_before_prospectus_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "current_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_POLICY_GUARD_FAIL")
    if int(policy.get("minimum_prior_annual_filings", 0)) != 1:
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_REQUIRES_ONE_ANNUAL")
    if int(policy.get("minimum_distinct_prior_prospectus_filings", 0)) < 1:
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_REQUIRES_PROSPECTUS")
    return policy, resolver.sha256(POLICY_PATH)


def columns() -> dict[str, Any]:
    return {
        "annual_plus_prospectus_status": "NOT_ATTEMPTED",
        "annual_plus_prospectus_contract_sha256": "",
        "annual_plus_prospectus_accessions": "[]",
        "annual_plus_prospectus_evidence_files": "[]",
        "annual_plus_prospectus_evidence_sha256": "[]",
        "annual_plus_prospectus_evidence_count": 0,
        "annual_plus_prospectus_exact_provider": "",
        "annual_plus_prospectus_exact_feed": "",
    }


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column, default in columns().items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def candidate_mask(queue: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    mask = queue.source_family.astype(str).eq(str(policy["source_family"]))
    mask &= queue.role.astype(str).isin(
        {str(value) for value in policy["eligible_frozen_roles"]}
    )
    mask &= queue.annual_plus_sec_status.astype(str).eq(
        str(policy["required_prior_hybrid_status"])
    )
    mask &= queue.annual_plus_sec_contract_sha256.astype(str).eq(
        str(policy["required_prior_hybrid_policy_sha256"])
    )
    return mask


def corroborate_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha256: str,
) -> dict[str, Any]:
    cik, event_accession = resolver.parse_event_id(str(row.event_id))
    base = {
        "promoted": False,
        "annual_plus_prospectus_status": "NOT_ELIGIBLE",
        "annual_plus_prospectus_contract_sha256": policy_sha256,
        "annual_plus_prospectus_accessions": "[]",
        "annual_plus_prospectus_evidence_files": "[]",
        "annual_plus_prospectus_evidence_sha256": "[]",
        "annual_plus_prospectus_evidence_count": 0,
    }
    if not cik or not event_accession:
        return {**base, "annual_plus_prospectus_status": "CIK_OR_ACCESSION_MISSING"}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {
            **base,
            "annual_plus_prospectus_status": f"SEC_SUBMISSIONS_{type(error).__name__}",
        }
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = (
        event_time.tz_localize("UTC")
        if event_time.tzinfo is None
        else event_time.tz_convert("UTC")
    )
    annual_forms = {str(value).upper() for value in policy["allowed_prior_annual_forms"]}
    prospectus_forms = {
        str(value).upper()
        for value in policy["allowed_independent_prior_prospectus_forms"]
    }
    prior_annual: list[tuple[float, dict[str, Any]]] = []
    prior_prospectus: list[tuple[float, dict[str, Any]]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        form = str(filing.get("form") or "").upper()
        stamp = resolver.filing_time(filing)
        if (
            not accession
            or accession == event_accession
            or stamp is None
            or stamp >= event_time
        ):
            continue
        days = (event_time - stamp).total_seconds() / 86400.0
        if not 0 <= days <= float(policy["maximum_oldest_prior_days"]):
            continue
        if form in annual_forms:
            prior_annual.append((days, filing))
        elif form in prospectus_forms:
            prior_prospectus.append((days, filing))
    prior_annual.sort(key=lambda item: item[0])
    prior_prospectus = sorted(prior_prospectus, key=lambda item: item[0])[
        : int(policy["maximum_prospectus_filings_scanned"])
    ]
    annual_methods = {
        str(value) for value in policy["allowed_annual_explicit_symbol_methods"]
    }
    prospectus_methods = {
        str(value) for value in policy["allowed_prospectus_explicit_symbol_methods"]
    }
    annual_records: list[dict[str, Any]] = []
    prospectus_records: list[dict[str, Any]] = []
    ambiguous_filings = 0
    for days, filing in prior_annual:
        record, ambiguous = hybrid.evidence_record(
            cik, filing, days, client, annual_methods
        )
        ambiguous_filings += int(ambiguous)
        if record:
            annual_records.append(record)
    for days, filing in prior_prospectus:
        record, ambiguous = hybrid.evidence_record(
            cik, filing, days, client, prospectus_methods
        )
        ambiguous_filings += int(ambiguous)
        if record:
            prospectus_records.append(record)
    evaluation_policy = {
        **policy,
        "minimum_distinct_nonannual_sec_filings": policy[
            "minimum_distinct_prior_prospectus_filings"
        ],
    }
    symbol, reason, audit_records = hybrid.evaluate_records(
        annual_records, prospectus_records, ambiguous_filings, evaluation_policy
    )
    if reason == "INDEPENDENT_PRIOR_SEC_EXPLICIT_EVIDENCE_MISSING":
        reason = "PRIOR_PROSPECTUS_EXPLICIT_EVIDENCE_MISSING"
    result = {
        **base,
        "annual_plus_prospectus_status": reason,
        "annual_plus_prospectus_accessions": json.dumps(
            [record["accession"] for record in audit_records], ensure_ascii=False
        ),
        "annual_plus_prospectus_evidence_files": json.dumps(
            [record["evidence_file"] for record in audit_records], ensure_ascii=False
        ),
        "annual_plus_prospectus_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in audit_records], ensure_ascii=False
        ),
        "annual_plus_prospectus_evidence_count": len(audit_records),
    }
    if not symbol:
        return result
    nearest = audit_records[0]
    return {
        **result,
        "promoted": True,
        "resolved_ticker": symbol,
        "ticker_candidates": json.dumps(
            nearest["evidence"], ensure_ascii=False, sort_keys=True
        ),
        "resolution_method": policy["promoted_resolution_method"],
        "resolution_confidence": policy["promoted_resolution_confidence"],
        "resolution_asof_mode": policy["promoted_resolution_asof_mode"],
        "evidence_file": nearest["evidence_file"],
        "evidence_sha256": nearest["evidence_sha256"],
        "evidence_accession": nearest["accession"],
        "evidence_filing_time_utc": nearest["filing_time_utc"],
        "resolution_error": "",
    }


def update_provider_status(exact_rows: pd.DataFrame) -> None:
    if exact_rows.empty:
        return
    status = pd.read_parquet(resolver.PROVIDER_STATUS_PATH).set_index("event_id")
    indexed = exact_rows.set_index("event_id")
    for event_id in status.index.intersection(indexed.index):
        row = indexed.loc[event_id]
        source = str(row.annual_plus_prospectus_exact_provider)
        feed = str(row.annual_plus_prospectus_exact_feed)
        status.loc[event_id, "selected_price_provider"] = f"{source}_{feed}"
        status.loc[event_id, "provider_parity_eligible"] = True
        status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
        status.loc[event_id, "provider_symbol"] = str(row.resolved_ticker)
        status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
        status.loc[event_id, "symbol_resolution_confidence"] = str(
            row.resolution_confidence
        )
        status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
        if source == "MASSIVE":
            status.loc[event_id, "massive_status"] = "CACHED_VALID"
        else:
            status.loc[event_id, "alpaca_sip_status"] = "CACHED_VALID"
    resolver.atomic_parquet(status.reset_index(), resolver.PROVIDER_STATUS_PATH)


def main() -> int:
    policy, policy_sha256 = load_policy()
    if resolver.sha256(resolver.QUEUE_PATH) != policy["activation_queue_sha256"]:
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_ACTIVATION_QUEUE_HASH_MISMATCH")
    if resolver.sha256(PRIOR_AUDIT_PATH) != policy["activation_prior_hybrid_audit_sha256"]:
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_ACTIVATION_AUDIT_HASH_MISMATCH")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    indices = list(
        queue.loc[candidate_mask(queue, policy)]
        .sort_values(["priority", "event_time_utc", "event_id"])
        .index
    )
    selected_ids = queue.loc[indices, "event_id"].astype(str).tolist()
    selection_digest = hybrid.event_ids_sha256(selected_ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_CANDIDATE_COUNT_MISMATCH")
    if selection_digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("ANNUAL_PLUS_PROSPECTUS_CANDIDATE_HASH_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    grouped: dict[str, list[Any]] = {}
    for index in indices:
        grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
    status_counts: Counter[str] = Counter()
    promoted_indices: list[Any] = []
    network_requests = 0
    cache_hits = 0

    def group_work(group: list[Any]) -> tuple[list[tuple[Any, dict[str, Any]]], int, int]:
        client = resolver.SECArchiveClient(min_interval=0.55)
        rows = [
            (index, corroborate_one(queue.loc[index].copy(), client, policy, policy_sha256))
            for index in group
        ]
        return rows, client.network_requests, client.cache_hits

    with ThreadPoolExecutor(max_workers=min(2, max(1, len(grouped)))) as executor:
        futures = [executor.submit(group_work, group) for group in grouped.values()]
        for future in as_completed(futures):
            rows, requests_used, hits = future.result()
            network_requests += requests_used
            cache_hits += hits
            for index, result in rows:
                promoted = bool(result.pop("promoted", False))
                status_counts[str(result.get("annual_plus_prospectus_status", "UNKNOWN"))] += 1
                for key, value in result.items():
                    queue.loc[index, key] = value
                if promoted:
                    symbol = resolver.normalize_symbol(result["resolved_ticker"])
                    queue.loc[index, "resolution_status"] = (
                        "RESOLVED_SAME_TICKER"
                        if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                        else "RESOLVED_NEW_TICKER"
                    )
                    queue.loc[index, "fetch_status"] = "ANNUAL_PLUS_PROSPECTUS_PRICE_REUSE_PENDING"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = False
                    promoted_indices.append(index)

    exact_new_indices: list[Any] = []
    exact_existing_official = 0
    for index in promoted_indices:
        event_id = str(queue.loc[index, "event_id"])
        if event_id in official_before:
            exact_existing_official += 1
            queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_PROSPECTUS_PROMOTION"
            continue
        authority = annual.exact_cache_candidate(queue.loc[index])
        if authority is None:
            queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_PROSPECTUS_PROMOTION"
            continue
        path, source, feed = authority
        queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_ANNUAL_PLUS_PROSPECTUS"
        queue.loc[index, "exact_eligible"] = True
        queue.loc[index, "provider"] = f"{source}_{feed}"
        queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
        queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
        queue.loc[index, "raw_sha256"] = resolver.sha256(path)
        queue.loc[index, "annual_plus_prospectus_exact_provider"] = source
        queue.loc[index, "annual_plus_prospectus_exact_feed"] = feed
        exact_new_indices.append(index)

    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    exact_rows = queue.loc[exact_new_indices].copy() if exact_new_indices else queue.iloc[0:0].copy()
    update_provider_status(exact_rows)
    recovery = resolver.recovery_report(queue)
    promoted = queue.loc[promoted_indices].copy() if promoted_indices else queue.iloc[0:0].copy()
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha256,
        "policy_frozen_before_prospectus_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded_for_promotion": False,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": selection_digest,
        "selected_ciks": len(grouped),
        "promoted_to_B": len(promoted_indices),
        "promoted_same_ticker": int((promoted.resolved_ticker.astype(str) == promoted.ticker.astype(str)).sum()) if not promoted.empty else 0,
        "promoted_changed_ticker": int((promoted.resolved_ticker.astype(str) != promoted.ticker.astype(str)).sum()) if not promoted.empty else 0,
        "already_in_official_dev_before_promotion": exact_existing_official,
        "new_exact_cache_rows_after_promotion": len(exact_new_indices),
        "consensus_status_counts": dict(status_counts),
        "SEC_network_requests": network_requests,
        "SEC_cache_hits": cache_hits,
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
