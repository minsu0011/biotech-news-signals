"""Recover symbols from documents inside the exact SEC event accession."""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

import annual_plus_sec_symbol_recovery_v221 as hybrid
import annual_report_symbol_recovery_v221 as annual
import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_EVENT_ACCESSION_ARCHIVE_SYMBOL_POLICY_V1.json"
PREVIOUS_AUDIT_PATH = RESEARCH / "HISTORICAL_NEAREST_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_AUDIT.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_EVENT_ACCESSION_ARCHIVE_SYMBOL_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "frozen_before_event_archive_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "exact_event_accession_required": True,
        "future_filing_evidence_forbidden": True,
        "current_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
        "full_submission_text_excluded": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_POLICY_GUARD_FAIL")
    return policy, resolver.sha256(POLICY_PATH)


def columns() -> dict[str, Any]:
    return {
        "event_archive_symbol_status": "NOT_ATTEMPTED",
        "event_archive_symbol_contract_sha256": "",
        "event_archive_symbol_evidence_files": "[]",
        "event_archive_symbol_evidence_sha256": "[]",
        "event_archive_symbol_evidence_count": 0,
        "event_archive_exact_provider": "",
        "event_archive_exact_feed": "",
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
    excluded = tuple(str(value).upper() for value in policy["registration_forms_excluded"])
    mask &= ~queue.form.astype(str).str.upper().str.startswith(excluded)
    return mask


def archive_names(
    cik: str,
    accession: str,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
) -> list[str]:
    items = (client.archive_index(cik, accession).get("directory", {}) or {}).get(
        "item", []
    ) or []
    extensions = {str(value).lower() for value in policy["allowed_archive_document_extensions"]}
    excluded = {
        str(value).lower() for value in policy["archive_support_documents_excluded_by_name"]
    }
    compact = re.sub(r"[^0-9]", "", accession)
    names: list[str] = []
    for item in items:
        name = str(item.get("name") or "")
        lower = name.lower()
        if not name or Path(lower).suffix not in extensions:
            continue
        if lower in {f"{compact}.txt", f"{accession.lower()}.txt"}:
            continue
        if any(token in lower for token in excluded):
            continue
        names.append(name)
    return sorted(set(names))[: int(policy["maximum_archive_documents_per_event_accession"])]


def inspect_document(
    cik: str,
    accession: str,
    name: str,
    client: resolver.SECArchiveClient,
    allowed_methods: set[str],
) -> tuple[dict[str, Any] | None, bool]:
    compact = re.sub(r"[^0-9]", "", accession)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    path = resolver.CACHE / "documents" / f"{int(cik)}_{compact}_{safe}"
    raw = client.get(
        resolver.ARCHIVE.format(
            cik=int(cik), accession=compact, document=quote(name)
        ),
        path,
    ).decode("utf-8", errors="replace")
    evidence = [
        item
        for item in resolver.extract_trading_symbols(raw)
        if str(item.get("method", "")) in allowed_methods
    ]
    symbol, ambiguous = resolver.choose_symbol(evidence, "")
    if ambiguous:
        return None, True
    if not symbol:
        return None, False
    return {
        "symbol": symbol,
        "evidence_file": str(path.relative_to(ROOT)),
        "evidence_sha256": resolver.sha256(path),
        "evidence": evidence,
    }, False


def evaluate_records(
    records: list[dict[str, Any]], ambiguous_documents: int, policy: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    if ambiguous_documents and policy["ambiguous_event_accession_symbol_evidence_forbidden"]:
        return "", "AMBIGUOUS_EVENT_ACCESSION_ARCHIVE_SYMBOL", records
    symbols = sorted(
        {resolver.normalize_symbol(record.get("symbol")) for record in records} - {""}
    )
    if len(symbols) > 1 and policy["contradictory_event_accession_symbol_evidence_forbidden"]:
        return "", "CONTRADICTORY_EVENT_ACCESSION_ARCHIVE_SYMBOLS", records
    if len(symbols) != 1:
        return "", "NO_SINGLE_EVENT_ACCESSION_ARCHIVE_SYMBOL", records
    matching = [
        record
        for record in records
        if resolver.normalize_symbol(record.get("symbol")) == symbols[0]
    ]
    if policy["evidence_file_sha256_required"] and any(
        not str(record.get("evidence_sha256", "")) for record in matching
    ):
        return "", "EVENT_ACCESSION_ARCHIVE_EVIDENCE_SHA_MISSING", matching
    return symbols[0], "PASS", matching


def recover_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha: str,
) -> dict[str, Any]:
    base = {
        "promoted": False,
        "event_archive_symbol_status": "NOT_ELIGIBLE",
        "event_archive_symbol_contract_sha256": policy_sha,
        "event_archive_symbol_evidence_files": "[]",
        "event_archive_symbol_evidence_sha256": "[]",
        "event_archive_symbol_evidence_count": 0,
    }
    cik, accession = resolver.parse_event_id(str(row.event_id))
    if not cik or not accession:
        return {**base, "event_archive_symbol_status": "CIK_OR_ACCESSION_MISSING"}
    try:
        names = archive_names(cik, accession, client, policy)
    except Exception as error:
        return {
            **base,
            "event_archive_symbol_status": f"EVENT_ARCHIVE_INDEX_{type(error).__name__}",
        }
    allowed_methods = {
        str(value) for value in policy["allowed_explicit_symbol_methods"]
    }
    records: list[dict[str, Any]] = []
    ambiguous_documents = 0
    for name in names:
        try:
            record, ambiguous = inspect_document(
                cik, accession, name, client, allowed_methods
            )
        except Exception:
            continue
        ambiguous_documents += int(ambiguous)
        if record:
            records.append(record)
    symbol, reason, audit_records = evaluate_records(
        records, ambiguous_documents, policy
    )
    result = {
        **base,
        "event_archive_symbol_status": reason,
        "event_archive_symbol_evidence_files": json.dumps(
            [record["evidence_file"] for record in audit_records], ensure_ascii=False
        ),
        "event_archive_symbol_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in audit_records], ensure_ascii=False
        ),
        "event_archive_symbol_evidence_count": len(audit_records),
    }
    if not symbol:
        return result
    first = audit_records[0]
    return {
        **result,
        "promoted": True,
        "resolved_ticker": symbol,
        "ticker_candidates": json.dumps(
            first["evidence"], ensure_ascii=False, sort_keys=True
        ),
        "resolution_method": policy["promoted_resolution_method"],
        "resolution_confidence": policy["promoted_resolution_confidence"],
        "resolution_asof_mode": policy["promoted_resolution_asof_mode"],
        "evidence_file": first["evidence_file"],
        "evidence_sha256": first["evidence_sha256"],
        "evidence_accession": accession,
        "evidence_filing_time_utc": str(row.event_time_utc),
        "resolution_error": "",
    }


def main() -> int:
    policy, policy_sha = load_policy()
    if resolver.sha256(resolver.QUEUE_PATH) != policy["activation_queue_sha256"]:
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_QUEUE_HASH_MISMATCH")
    if resolver.sha256(PREVIOUS_AUDIT_PATH) != policy["activation_previous_audit_sha256"]:
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_PREVIOUS_AUDIT_HASH_MISMATCH")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    indices = list(
        queue.loc[candidate_mask(queue, policy)]
        .sort_values(["priority", "event_time_utc", "event_id"])
        .index
    )
    ids = queue.loc[indices, "event_id"].astype(str).tolist()
    digest = hybrid.event_ids_sha256(ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_CANDIDATE_COUNT_MISMATCH")
    if digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_CANDIDATE_HASH_MISMATCH")
    grouped: dict[str, list[Any]] = {}
    for index in indices:
        grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
    if len(grouped) != int(policy["candidate_cik_count"]):
        raise RuntimeError("EVENT_ARCHIVE_SYMBOL_CIK_COUNT_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    status_counts: Counter[str] = Counter()
    promoted_indices: list[Any] = []
    network_requests = 0
    cache_hits = 0

    def group_work(group: list[Any]):
        client = resolver.SECArchiveClient(min_interval=0.75)
        rows = [
            (index, recover_one(queue.loc[index].copy(), client, policy, policy_sha))
            for index in group
        ]
        return rows, client.network_requests, client.cache_hits

    with ThreadPoolExecutor(max_workers=min(6, len(grouped))) as executor:
        futures = [executor.submit(group_work, group) for group in grouped.values()]
        for future in as_completed(futures):
            rows, requests_used, hits = future.result()
            network_requests += requests_used
            cache_hits += hits
            for index, result in rows:
                promoted = bool(result.pop("promoted", False))
                status_counts[str(result["event_archive_symbol_status"])] += 1
                for key, value in result.items():
                    queue.loc[index, key] = value
                if promoted:
                    symbol = resolver.normalize_symbol(result["resolved_ticker"])
                    queue.loc[index, "resolution_status"] = (
                        "RESOLVED_SAME_TICKER"
                        if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                        else "RESOLVED_NEW_TICKER"
                    )
                    queue.loc[index, "fetch_status"] = "EVENT_ARCHIVE_SYMBOL_PRICE_REUSE_PENDING"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = False
                    promoted_indices.append(index)

    exact_indices: list[Any] = []
    already_official = 0
    for index in promoted_indices:
        if str(queue.loc[index, "event_id"]) in official_before:
            already_official += 1
            queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_EVENT_ARCHIVE_PROMOTION"
            continue
        authority = annual.exact_cache_candidate(queue.loc[index])
        if authority is None:
            queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_EVENT_ARCHIVE_PROMOTION"
            continue
        path, source, feed = authority
        queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_EVENT_ARCHIVE_SYMBOL"
        queue.loc[index, "exact_eligible"] = True
        queue.loc[index, "provider"] = f"{source}_{feed}"
        queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
        queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
        queue.loc[index, "raw_sha256"] = resolver.sha256(path)
        queue.loc[index, "event_archive_exact_provider"] = source
        queue.loc[index, "event_archive_exact_feed"] = feed
        exact_indices.append(index)
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    if exact_indices:
        annual.update_provider_status(queue.loc[exact_indices].rename(columns={
            "event_archive_exact_provider": "annual_exact_provider",
            "event_archive_exact_feed": "annual_exact_feed",
        }))
    recovery = resolver.recovery_report(queue)
    promoted = queue.loc[promoted_indices] if promoted_indices else queue.iloc[0:0]
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "policy_frozen_before_event_archive_results": True,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": digest,
        "selected_ciks": len(grouped),
        "promoted_to_A": len(promoted_indices),
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
