"""Recover frozen DEV symbols from two prior 6-K archive exhibits."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import annual_plus_6k_exhibit_symbol_recovery_v221 as exhibit
import annual_plus_sec_symbol_recovery_v221 as hybrid
import annual_report_symbol_recovery_v221 as annual
import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_6K_EXHIBIT_MULTI_FILING_SYMBOL_CONSENSUS_POLICY_V1.json"
PREVIOUS_REPORT_PATH = RESEARCH / "ALPACA_ANNUAL_PLUS_6K_EXHIBIT_RETRY_REPORT.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_6K_EXHIBIT_MULTI_FILING_SYMBOL_CONSENSUS_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "frozen_before_exhibit_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "current_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_POLICY_GUARD_FAIL")
    if int(policy.get("minimum_distinct_prior_filings", 0)) < 2:
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_REQUIRES_TWO_FILINGS")
    return policy, resolver.sha256(POLICY_PATH)


def columns() -> dict[str, Any]:
    return {
        "sixk_exhibit_consensus_status": "NOT_ATTEMPTED",
        "sixk_exhibit_consensus_contract_sha256": "",
        "sixk_exhibit_consensus_accessions": "[]",
        "sixk_exhibit_consensus_evidence_files": "[]",
        "sixk_exhibit_consensus_evidence_sha256": "[]",
        "sixk_exhibit_consensus_evidence_count": 0,
        "sixk_exhibit_exact_provider": "",
        "sixk_exhibit_exact_feed": "",
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
    mask &= queue.resolution_confidence.astype(str).isin(
        {str(value) for value in policy["eligible_resolution_confidence_before"]}
    )
    mask &= queue.annual_consensus_status.astype(str).isin(
        {str(value) for value in policy["eligible_prior_annual_statuses"]}
    )
    excluded = tuple(str(value).upper() for value in policy["registration_forms_excluded"])
    mask &= ~queue.form.astype(str).str.upper().str.startswith(excluded)
    return mask


def evaluate_records(
    records: list[dict[str, Any]], ambiguous_filings: int, policy: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    unique = {str(record["accession"]): record for record in records}
    records = list(unique.values())
    if ambiguous_filings and policy["ambiguous_symbol_evidence_forbidden"]:
        return "", "AMBIGUOUS_PRIOR_6K_EXHIBIT_EVIDENCE", records
    symbols = sorted(
        {resolver.normalize_symbol(record.get("symbol")) for record in records} - {""}
    )
    if len(symbols) > 1 and policy["contradictory_symbol_evidence_forbidden"]:
        return "", "CONTRADICTORY_PRIOR_6K_EXHIBIT_EVIDENCE", records
    if len(symbols) != 1:
        return "", "NO_SINGLE_6K_EXHIBIT_CONSENSUS_SYMBOL", records
    matching = sorted(
        [
            record
            for record in records
            if resolver.normalize_symbol(record.get("symbol")) == symbols[0]
        ],
        key=lambda record: float(record["days_before_event"]),
    )
    if len(matching) < int(policy["minimum_distinct_prior_filings"]):
        return "", "INSUFFICIENT_DISTINCT_PRIOR_6K_EXHIBIT_FILINGS", matching
    days = [float(record["days_before_event"]) for record in matching]
    if min(days) > float(policy["maximum_nearest_prior_days"]):
        return "", "NEAREST_PRIOR_6K_EXHIBIT_OUTSIDE_WINDOW", matching
    if max(days) > float(policy["maximum_oldest_prior_days"]):
        return "", "OLDEST_PRIOR_6K_EXHIBIT_OUTSIDE_WINDOW", matching
    if policy["evidence_file_sha256_required"] and any(
        not str(record.get("evidence_sha256", "")) for record in matching
    ):
        return "", "6K_EXHIBIT_EVIDENCE_SHA_MISSING", matching
    return symbols[0], "PASS", matching


def inspect_filing(
    cik: str,
    accession: str,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    memo: dict[str, tuple[dict[str, Any] | None, bool]],
) -> tuple[dict[str, Any] | None, bool]:
    if accession in memo:
        return memo[accession]
    try:
        names = exhibit.archive_document_names(cik, accession, client, policy)
    except Exception:
        memo[accession] = (None, False)
        return memo[accession]
    allowed_methods = {
        str(value) for value in policy["allowed_exhibit_explicit_symbol_methods"]
    }
    document_records: list[dict[str, Any]] = []
    ambiguous = False
    for name in names:
        try:
            record, document_ambiguous = exhibit.inspect_archive_document(
                cik, accession, name, client, allowed_methods
            )
        except Exception:
            continue
        ambiguous |= document_ambiguous
        if record:
            document_records.append(record)
    symbols = {
        resolver.normalize_symbol(record.get("symbol")) for record in document_records
    } - {""}
    if ambiguous or len(symbols) > 1:
        memo[accession] = (None, True)
    elif not document_records:
        memo[accession] = (None, False)
    else:
        first = document_records[0]
        combined_evidence = [
            item for record in document_records for item in record.get("evidence", [])
        ]
        memo[accession] = ({
            **first,
            "symbol": next(iter(symbols)),
            "evidence": combined_evidence,
            "supporting_files": [record["evidence_file"] for record in document_records],
            "supporting_hashes": [record["evidence_sha256"] for record in document_records],
        }, False)
    return memo[accession]


def corroborate_one(
    row: pd.Series,
    filings: list[dict[str, Any]],
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha: str,
    memo: dict[str, tuple[dict[str, Any] | None, bool]],
) -> dict[str, Any]:
    base = {
        "promoted": False,
        "sixk_exhibit_consensus_status": "NOT_ELIGIBLE",
        "sixk_exhibit_consensus_contract_sha256": policy_sha,
        "sixk_exhibit_consensus_accessions": "[]",
        "sixk_exhibit_consensus_evidence_files": "[]",
        "sixk_exhibit_consensus_evidence_sha256": "[]",
        "sixk_exhibit_consensus_evidence_count": 0,
    }
    cik, event_accession = resolver.parse_event_id(str(row.event_id))
    if not cik or not event_accession:
        return {**base, "sixk_exhibit_consensus_status": "CIK_OR_ACCESSION_MISSING"}
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = (
        event_time.tz_localize("UTC")
        if event_time.tzinfo is None
        else event_time.tz_convert("UTC")
    )
    forms = {str(value).upper() for value in policy["independent_filing_forms"]}
    prior: list[tuple[float, dict[str, Any]]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        stamp = resolver.filing_time(filing)
        if (
            str(filing.get("form") or "").upper() not in forms
            or not accession
            or accession == event_accession
            or stamp is None
            or stamp >= event_time
        ):
            continue
        days = (event_time - stamp).total_seconds() / 86400.0
        if 0 <= days <= float(policy["maximum_oldest_prior_days"]):
            prior.append((days, filing))
    prior = sorted(prior, key=lambda item: item[0])[
        : int(policy["maximum_independent_filings_scanned"])
    ]
    records: list[dict[str, Any]] = []
    ambiguous_filings = 0
    for days, filing in prior:
        accession = str(filing.get("accessionNumber") or "")
        record, ambiguous = inspect_filing(cik, accession, client, policy, memo)
        ambiguous_filings += int(ambiguous)
        if record:
            records.append({**record, "days_before_event": round(days, 6)})
    symbol, reason, audit_records = evaluate_records(
        records, ambiguous_filings, policy
    )
    result = {
        **base,
        "sixk_exhibit_consensus_status": reason,
        "sixk_exhibit_consensus_accessions": json.dumps(
            [record["accession"] for record in audit_records], ensure_ascii=False
        ),
        "sixk_exhibit_consensus_evidence_files": json.dumps(
            [record["evidence_file"] for record in audit_records], ensure_ascii=False
        ),
        "sixk_exhibit_consensus_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in audit_records], ensure_ascii=False
        ),
        "sixk_exhibit_consensus_evidence_count": len(audit_records),
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
        "evidence_filing_time_utc": "",
        "resolution_error": "",
    }


def main() -> int:
    policy, policy_sha = load_policy()
    if resolver.sha256(resolver.QUEUE_PATH) != policy["activation_queue_sha256"]:
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_QUEUE_HASH_MISMATCH")
    if resolver.sha256(PREVIOUS_REPORT_PATH) != policy["activation_previous_provider_report_sha256"]:
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_REPORT_HASH_MISMATCH")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    indices = list(
        queue.loc[candidate_mask(queue, policy)]
        .sort_values(["priority", "event_time_utc", "event_id"])
        .index
    )
    ids = queue.loc[indices, "event_id"].astype(str).tolist()
    digest = hybrid.event_ids_sha256(ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_CANDIDATE_COUNT_MISMATCH")
    if digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_CANDIDATE_HASH_MISMATCH")
    grouped: dict[str, list[Any]] = {}
    for index in indices:
        grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
    if len(grouped) != int(policy["candidate_cik_count"]):
        raise RuntimeError("6K_EXHIBIT_CONSENSUS_CIK_COUNT_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    status_counts: Counter[str] = Counter()
    promoted_indices: list[Any] = []
    network_requests = 0
    cache_hits = 0

    def group_work(cik: str, group: list[Any]):
        client = resolver.SECArchiveClient(min_interval=0.55)
        try:
            _, filings = client.filing_rows(cik)
        except Exception:
            filings = []
        memo: dict[str, tuple[dict[str, Any] | None, bool]] = {}
        rows = [
            (
                index,
                corroborate_one(
                    queue.loc[index].copy(), filings, client, policy, policy_sha, memo
                ),
            )
            for index in group
        ]
        return rows, client.network_requests, client.cache_hits

    with ThreadPoolExecutor(max_workers=min(4, len(grouped))) as executor:
        futures = [
            executor.submit(group_work, cik, group) for cik, group in grouped.items()
        ]
        for future in as_completed(futures):
            rows, requests_used, hits = future.result()
            network_requests += requests_used
            cache_hits += hits
            for index, result in rows:
                promoted = bool(result.pop("promoted", False))
                status_counts[str(result["sixk_exhibit_consensus_status"])] += 1
                for key, value in result.items():
                    queue.loc[index, key] = value
                if promoted:
                    symbol = resolver.normalize_symbol(result["resolved_ticker"])
                    queue.loc[index, "resolution_status"] = (
                        "RESOLVED_SAME_TICKER"
                        if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                        else "RESOLVED_NEW_TICKER"
                    )
                    queue.loc[index, "fetch_status"] = "6K_EXHIBIT_CONSENSUS_PRICE_REUSE_PENDING"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = False
                    promoted_indices.append(index)

    exact_indices: list[Any] = []
    already_official = 0
    for index in promoted_indices:
        if str(queue.loc[index, "event_id"]) in official_before:
            already_official += 1
            queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_6K_CONSENSUS"
            continue
        authority = annual.exact_cache_candidate(queue.loc[index])
        if authority is None:
            queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_6K_CONSENSUS"
            continue
        path, source, feed = authority
        queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_6K_EXHIBIT_CONSENSUS"
        queue.loc[index, "exact_eligible"] = True
        queue.loc[index, "provider"] = f"{source}_{feed}"
        queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
        queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
        queue.loc[index, "raw_sha256"] = resolver.sha256(path)
        queue.loc[index, "sixk_exhibit_exact_provider"] = source
        queue.loc[index, "sixk_exhibit_exact_feed"] = feed
        exact_indices.append(index)
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    if exact_indices:
        annual.update_provider_status(queue.loc[exact_indices].rename(columns={
            "sixk_exhibit_exact_provider": "annual_exact_provider",
            "sixk_exhibit_exact_feed": "annual_exact_feed",
        }))
    recovery = resolver.recovery_report(queue)
    promoted = queue.loc[promoted_indices] if promoted_indices else queue.iloc[0:0]
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "policy_frozen_before_exhibit_results": True,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": digest,
        "selected_ciks": len(grouped),
        "promoted_to_B": len(promoted_indices),
        "promoted_same_ticker": int((promoted.resolved_ticker.astype(str) == promoted.ticker.astype(str)).sum()) if not promoted.empty else 0,
        "promoted_changed_ticker": int((promoted.resolved_ticker.astype(str) != promoted.ticker.astype(str)).sum()) if not promoted.empty else 0,
        "already_in_official_dev_before_promotion": already_official,
        "new_exact_cache_rows_after_promotion": len(exact_indices),
        "status_counts": dict(status_counts),
        "SEC_network_requests": network_requests,
        "SEC_cache_hits": cache_hits,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded_for_promotion": False,
        "exact_contract_relaxed": False,
        "roles_reassigned": False,
        "seal_or_final_labels_opened": False,
        "queue_sha256": recovery["queue_sha256"],
    }
    resolver.atomic_json(audit, AUDIT_PATH)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
