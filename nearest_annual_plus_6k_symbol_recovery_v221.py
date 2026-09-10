"""Resolve ticker transitions with nearest annual + nearest prior 6-K exhibit."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import annual_plus_sec_symbol_recovery_v221 as hybrid
import annual_report_symbol_recovery_v221 as annual
import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_NEAREST_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_POLICY_V1.json"
PRIOR_AUDIT_PATH = RESEARCH / "HISTORICAL_6K_EXHIBIT_MULTI_FILING_SYMBOL_CONSENSUS_AUDIT.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_NEAREST_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "frozen_before_evidence_comparison_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "current_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
        "nearest_explicit_annual_only": True,
        "nearest_explicit_6k_exhibit_only": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("NEAREST_ANNUAL_PLUS_6K_POLICY_GUARD_FAIL")
    return policy, resolver.sha256(POLICY_PATH)


def columns() -> dict[str, Any]:
    return {
        "nearest_annual_plus_6k_status": "NOT_ATTEMPTED",
        "nearest_annual_plus_6k_contract_sha256": "",
        "nearest_annual_plus_6k_accessions": "[]",
        "nearest_annual_plus_6k_evidence_files": "[]",
        "nearest_annual_plus_6k_evidence_sha256": "[]",
        "nearest_annual_plus_6k_evidence_count": 0,
        "nearest_annual_plus_6k_exact_provider": "",
        "nearest_annual_plus_6k_exact_feed": "",
    }


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column, default in columns().items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def candidate_mask(queue: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    mask = queue.source_family.astype(str).eq(str(policy["source_family"]))
    mask &= queue.role.astype(str).eq(str(policy["role"]))
    mask &= queue.sixk_exhibit_consensus_status.astype(str).eq(
        str(policy["required_prior_6k_consensus_status"])
    )
    mask &= queue.sixk_exhibit_consensus_contract_sha256.astype(str).eq(
        str(policy["required_prior_6k_consensus_policy_sha256"])
    )
    mask &= queue.annual_consensus_status.astype(str).eq(
        str(policy["required_prior_annual_status"])
    )
    return mask


def parse_frozen_records(
    accessions_json: str,
    files_json: str,
    hashes_json: str,
    by_accession: dict[str, dict[str, Any]],
    event_time: pd.Timestamp,
    allowed_methods: set[str],
) -> tuple[list[dict[str, Any]], bool, str]:
    accessions = json.loads(str(accessions_json))
    files = json.loads(str(files_json))
    hashes = json.loads(str(hashes_json))
    if not (len(accessions) == len(files) == len(hashes)):
        return [], False, "FROZEN_EVIDENCE_CARDINALITY_FAIL"
    records: list[dict[str, Any]] = []
    ambiguous = False
    for accession, relative, expected_hash in zip(accessions, files, hashes):
        path = ROOT / str(relative)
        if not path.is_file() or resolver.sha256(path) != str(expected_hash):
            return [], False, "FROZEN_EVIDENCE_HASH_FAIL"
        filing = by_accession.get(str(accession))
        stamp = resolver.filing_time(filing or {})
        if stamp is None or stamp >= event_time:
            return [], False, "FROZEN_EVIDENCE_POINT_IN_TIME_FAIL"
        evidence = [
            item
            for item in resolver.extract_trading_symbols(
                path.read_text(encoding="utf-8", errors="replace")
            )
            if str(item.get("method", "")) in allowed_methods
        ]
        symbol, is_ambiguous = resolver.choose_symbol(evidence, "")
        ambiguous |= is_ambiguous
        if not symbol:
            continue
        records.append({
            "accession": str(accession),
            "symbol": symbol,
            "filing_time_utc": str(stamp),
            "days_before_event": (event_time - stamp).total_seconds() / 86400.0,
            "evidence_file": str(relative),
            "evidence_sha256": str(expected_hash),
            "evidence": evidence,
        })
    return records, ambiguous, "PASS"


def corroborate_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha: str,
) -> dict[str, Any]:
    base = {
        "promoted": False,
        "nearest_annual_plus_6k_status": "NOT_ELIGIBLE",
        "nearest_annual_plus_6k_contract_sha256": policy_sha,
        "nearest_annual_plus_6k_accessions": "[]",
        "nearest_annual_plus_6k_evidence_files": "[]",
        "nearest_annual_plus_6k_evidence_sha256": "[]",
        "nearest_annual_plus_6k_evidence_count": 0,
    }
    cik, _ = resolver.parse_event_id(str(row.event_id))
    if not cik:
        return {**base, "nearest_annual_plus_6k_status": "CIK_MISSING"}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {
            **base,
            "nearest_annual_plus_6k_status": f"SEC_SUBMISSIONS_{type(error).__name__}",
        }
    by_accession = {
        str(filing.get("accessionNumber")): filing for filing in filings
    }
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = (
        event_time.tz_localize("UTC")
        if event_time.tzinfo is None
        else event_time.tz_convert("UTC")
    )
    annual_methods = {
        str(value) for value in policy["allowed_annual_explicit_symbol_methods"]
    }
    sixk_methods = {
        str(value) for value in policy["allowed_6k_exhibit_explicit_symbol_methods"]
    }
    annual_records, annual_ambiguous, annual_status = parse_frozen_records(
        row.annual_corroborating_accessions,
        row.annual_corroborating_evidence_files,
        row.annual_corroborating_evidence_sha256,
        by_accession,
        event_time,
        annual_methods,
    )
    sixk_records, sixk_ambiguous, sixk_status = parse_frozen_records(
        row.sixk_exhibit_consensus_accessions,
        row.sixk_exhibit_consensus_evidence_files,
        row.sixk_exhibit_consensus_evidence_sha256,
        by_accession,
        event_time,
        sixk_methods,
    )
    all_records = annual_records + sixk_records
    if annual_status != "PASS":
        reason = annual_status
    elif sixk_status != "PASS":
        reason = sixk_status
    elif annual_ambiguous or sixk_ambiguous:
        reason = "AMBIGUOUS_NEAREST_EVIDENCE"
    elif not annual_records:
        reason = "NEAREST_ANNUAL_EXPLICIT_EVIDENCE_MISSING"
    elif not sixk_records:
        reason = "NEAREST_6K_EXHIBIT_EXPLICIT_EVIDENCE_MISSING"
    else:
        nearest_annual = min(
            annual_records, key=lambda record: float(record["days_before_event"])
        )
        nearest_sixk = min(
            sixk_records, key=lambda record: float(record["days_before_event"])
        )
        compared = [nearest_annual, nearest_sixk]
        symbols = {
            resolver.normalize_symbol(record["symbol"]) for record in compared
        } - {""}
        if float(nearest_annual["days_before_event"]) > float(
            policy["maximum_nearest_annual_days"]
        ):
            reason = "NEAREST_ANNUAL_OUTSIDE_WINDOW"
        elif float(nearest_sixk["days_before_event"]) > float(
            policy["maximum_nearest_6k_exhibit_days"]
        ):
            reason = "NEAREST_6K_EXHIBIT_OUTSIDE_WINDOW"
        elif len(symbols) != 1:
            reason = "NEAREST_ANNUAL_AND_6K_SYMBOL_DISAGREE"
        else:
            reason = "PASS"
            all_records = compared
    result = {
        **base,
        "nearest_annual_plus_6k_status": reason,
        "nearest_annual_plus_6k_accessions": json.dumps(
            [record["accession"] for record in all_records], ensure_ascii=False
        ),
        "nearest_annual_plus_6k_evidence_files": json.dumps(
            [record["evidence_file"] for record in all_records], ensure_ascii=False
        ),
        "nearest_annual_plus_6k_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in all_records], ensure_ascii=False
        ),
        "nearest_annual_plus_6k_evidence_count": len(all_records),
    }
    if reason != "PASS":
        return result
    nearest = min(all_records, key=lambda record: float(record["days_before_event"]))
    symbol = resolver.normalize_symbol(nearest["symbol"])
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


def main() -> int:
    policy, policy_sha = load_policy()
    if resolver.sha256(resolver.QUEUE_PATH) != policy["activation_queue_sha256"]:
        raise RuntimeError("NEAREST_ANNUAL_PLUS_6K_QUEUE_HASH_MISMATCH")
    if resolver.sha256(PRIOR_AUDIT_PATH) != policy["activation_6k_consensus_audit_sha256"]:
        raise RuntimeError("NEAREST_ANNUAL_PLUS_6K_AUDIT_HASH_MISMATCH")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    indices = list(queue.loc[candidate_mask(queue, policy)].index)
    ids = queue.loc[indices, "event_id"].astype(str).tolist()
    digest = hybrid.event_ids_sha256(ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("NEAREST_ANNUAL_PLUS_6K_CANDIDATE_COUNT_MISMATCH")
    if digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("NEAREST_ANNUAL_PLUS_6K_CANDIDATE_HASH_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    clients: dict[str, resolver.SECArchiveClient] = {}
    status_counts: Counter[str] = Counter()
    promoted_indices: list[Any] = []
    for index in indices:
        cik = str(queue.loc[index, "CIK"])
        client = clients.setdefault(cik, resolver.SECArchiveClient(min_interval=0.55))
        result = corroborate_one(queue.loc[index].copy(), client, policy, policy_sha)
        promoted = bool(result.pop("promoted", False))
        status_counts[str(result["nearest_annual_plus_6k_status"])] += 1
        for key, value in result.items():
            queue.loc[index, key] = value
        if promoted:
            symbol = resolver.normalize_symbol(result["resolved_ticker"])
            queue.loc[index, "resolution_status"] = (
                "RESOLVED_SAME_TICKER"
                if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                else "RESOLVED_NEW_TICKER"
            )
            queue.loc[index, "fetch_status"] = "NEAREST_ANNUAL_PLUS_6K_PRICE_REUSE_PENDING"
            queue.loc[index, "fetch_error"] = ""
            queue.loc[index, "exact_eligible"] = False
            promoted_indices.append(index)
    exact_indices: list[Any] = []
    already_official = 0
    for index in promoted_indices:
        if str(queue.loc[index, "event_id"]) in official_before:
            already_official += 1
            queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_NEAREST_PROMOTION"
            continue
        authority = annual.exact_cache_candidate(queue.loc[index])
        if authority is None:
            queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_NEAREST_PROMOTION"
            continue
        path, source, feed = authority
        queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_NEAREST_ANNUAL_PLUS_6K"
        queue.loc[index, "exact_eligible"] = True
        queue.loc[index, "provider"] = f"{source}_{feed}"
        queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
        queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
        queue.loc[index, "raw_sha256"] = resolver.sha256(path)
        queue.loc[index, "nearest_annual_plus_6k_exact_provider"] = source
        queue.loc[index, "nearest_annual_plus_6k_exact_feed"] = feed
        exact_indices.append(index)
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    if exact_indices:
        annual.update_provider_status(queue.loc[exact_indices].rename(columns={
            "nearest_annual_plus_6k_exact_provider": "annual_exact_provider",
            "nearest_annual_plus_6k_exact_feed": "annual_exact_feed",
        }))
    recovery = resolver.recovery_report(queue)
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "policy_frozen_before_evidence_comparison_results": True,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": digest,
        "promoted_to_B": len(promoted_indices),
        "already_in_official_dev_before_promotion": already_official,
        "new_exact_cache_rows_after_promotion": len(exact_indices),
        "status_counts": dict(status_counts),
        "SEC_network_requests": sum(client.network_requests for client in clients.values()),
        "SEC_cache_hits": sum(client.cache_hits for client in clients.values()),
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
