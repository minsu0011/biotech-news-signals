"""Point-in-time annual-report plus prior 6-K exhibit symbol recovery."""
from __future__ import annotations

import json
import os
import re
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
POLICY_PATH = DATA / "HISTORICAL_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_POLICY_V1.json"
PROSPECTUS_AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_PLUS_PROSPECTUS_SYMBOL_CONSENSUS_AUDIT.json"
PROVIDER_RETRY_PATH = RESEARCH / "ALPACA_ANNUAL_PLUS_PROSPECTUS_RETRY_REPORT.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_PLUS_6K_EXHIBIT_SYMBOL_AUDIT.json"
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
        raise RuntimeError("ANNUAL_PLUS_6K_EXHIBIT_POLICY_GUARD_FAIL")
    return policy, resolver.sha256(POLICY_PATH)


def columns() -> dict[str, Any]:
    return {
        "annual_plus_6k_exhibit_status": "NOT_ATTEMPTED",
        "annual_plus_6k_exhibit_contract_sha256": "",
        "annual_plus_6k_exhibit_accessions": "[]",
        "annual_plus_6k_exhibit_evidence_files": "[]",
        "annual_plus_6k_exhibit_evidence_sha256": "[]",
        "annual_plus_6k_exhibit_evidence_count": 0,
        "annual_plus_6k_exhibit_exact_provider": "",
        "annual_plus_6k_exhibit_exact_feed": "",
    }


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column, default in columns().items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def candidate_mask(queue: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    mask = queue.source_family.astype(str).eq(str(policy["source_family"]))
    mask &= queue.role.astype(str).eq(str(policy["role"]))
    mask &= queue.annual_plus_prospectus_status.astype(str).eq(
        str(policy["required_prior_prospectus_status"])
    )
    mask &= queue.annual_plus_prospectus_contract_sha256.astype(str).eq(
        str(policy["required_prior_prospectus_policy_sha256"])
    )
    mask &= queue.annual_consensus_status.astype(str).eq(
        str(policy["required_prior_annual_status"])
    )
    mask &= pd.to_numeric(
        queue.annual_consensus_evidence_count, errors="coerce"
    ).fillna(0).eq(int(policy["required_prior_annual_evidence_count"]))
    return mask


def load_frozen_annual_record(
    row: pd.Series, policy: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    accessions = json.loads(str(row.annual_corroborating_accessions))
    files = json.loads(str(row.annual_corroborating_evidence_files))
    hashes = json.loads(str(row.annual_corroborating_evidence_sha256))
    if not (len(accessions) == len(files) == len(hashes) == 1):
        return None, "FROZEN_ANNUAL_EVIDENCE_CARDINALITY_FAIL"
    path = ROOT / files[0]
    if not path.is_file() or resolver.sha256(path) != hashes[0]:
        return None, "FROZEN_ANNUAL_EVIDENCE_HASH_FAIL"
    evidence = resolver.extract_trading_symbols(
        path.read_text(encoding="utf-8", errors="replace")
    )
    allowed = {
        str(value) for value in policy["allowed_annual_explicit_symbol_methods"]
    }
    evidence = [item for item in evidence if str(item.get("method", "")) in allowed]
    symbol, ambiguous = resolver.choose_symbol(evidence, "")
    if ambiguous:
        return None, "AMBIGUOUS_FROZEN_ANNUAL_EVIDENCE"
    if not symbol:
        return None, "FROZEN_ANNUAL_EXPLICIT_SYMBOL_MISSING"
    return {
        "accession": accessions[0],
        "symbol": symbol,
        "evidence_file": files[0],
        "evidence_sha256": hashes[0],
        "evidence": evidence,
        "evidence_kind": "ANNUAL_PRIMARY_DOCUMENT",
    }, "PASS"


def archive_document_names(
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
    names = []
    for item in items:
        name = str(item.get("name") or "")
        lower = name.lower()
        if not name or Path(lower).suffix not in extensions:
            continue
        if any(token in lower for token in excluded):
            continue
        names.append(name)
    return sorted(set(names))[: int(policy["maximum_archive_documents_per_filing"])]


def inspect_archive_document(
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
        "accession": accession,
        "symbol": symbol,
        "evidence_file": str(path.relative_to(ROOT)),
        "evidence_sha256": resolver.sha256(path),
        "evidence": evidence,
        "evidence_kind": "PRIOR_6K_ARCHIVE_DOCUMENT",
    }, False


def corroborate_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha256: str,
) -> dict[str, Any]:
    base = {
        "promoted": False,
        "annual_plus_6k_exhibit_status": "NOT_ELIGIBLE",
        "annual_plus_6k_exhibit_contract_sha256": policy_sha256,
        "annual_plus_6k_exhibit_accessions": "[]",
        "annual_plus_6k_exhibit_evidence_files": "[]",
        "annual_plus_6k_exhibit_evidence_sha256": "[]",
        "annual_plus_6k_exhibit_evidence_count": 0,
    }
    cik, event_accession = resolver.parse_event_id(str(row.event_id))
    if not cik or not event_accession:
        return {**base, "annual_plus_6k_exhibit_status": "CIK_OR_ACCESSION_MISSING"}
    annual_record, annual_status = load_frozen_annual_record(row, policy)
    if annual_record is None:
        return {**base, "annual_plus_6k_exhibit_status": annual_status}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {
            **base,
            "annual_plus_6k_exhibit_status": f"SEC_SUBMISSIONS_{type(error).__name__}",
        }
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = (
        event_time.tz_localize("UTC")
        if event_time.tzinfo is None
        else event_time.tz_convert("UTC")
    )
    allowed_forms = {str(value).upper() for value in policy["independent_filing_forms"]}
    prior: list[tuple[float, dict[str, Any]]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        stamp = resolver.filing_time(filing)
        if (
            str(filing.get("form") or "").upper() not in allowed_forms
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
    allowed_methods = {
        str(value) for value in policy["allowed_exhibit_explicit_symbol_methods"]
    }
    exhibit_records: list[dict[str, Any]] = []
    ambiguous = False
    for _, filing in prior:
        accession = str(filing.get("accessionNumber") or "")
        filing_records: list[dict[str, Any]] = []
        try:
            names = archive_document_names(cik, accession, client, policy)
        except Exception:
            continue
        for name in names:
            try:
                record, document_ambiguous = inspect_archive_document(
                    cik, accession, name, client, allowed_methods
                )
            except Exception:
                continue
            ambiguous |= document_ambiguous
            if record:
                filing_records.append(record)
        filing_symbols = {
            resolver.normalize_symbol(record["symbol"]) for record in filing_records
        } - {""}
        if len(filing_symbols) > 1:
            ambiguous = True
        elif filing_records:
            exhibit_records.extend(filing_records)

    all_records = [annual_record, *exhibit_records]
    symbols = {
        resolver.normalize_symbol(record["symbol"]) for record in all_records
    } - {""}
    if ambiguous:
        reason = "AMBIGUOUS_PRIOR_6K_EXHIBIT_SYMBOL_EVIDENCE"
    elif not exhibit_records:
        reason = "PRIOR_6K_EXHIBIT_EXPLICIT_SYMBOL_MISSING"
    elif len(symbols) > 1:
        reason = "CONTRADICTORY_ANNUAL_AND_6K_EXHIBIT_SYMBOLS"
    elif len(symbols) != 1:
        reason = "NO_SINGLE_ANNUAL_AND_6K_EXHIBIT_SYMBOL"
    elif any(not record.get("evidence_sha256") for record in all_records):
        reason = "EVIDENCE_SHA_MISSING"
    else:
        reason = "PASS"
    result = {
        **base,
        "annual_plus_6k_exhibit_status": reason,
        "annual_plus_6k_exhibit_accessions": json.dumps(
            [record["accession"] for record in all_records], ensure_ascii=False
        ),
        "annual_plus_6k_exhibit_evidence_files": json.dumps(
            [record["evidence_file"] for record in all_records], ensure_ascii=False
        ),
        "annual_plus_6k_exhibit_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in all_records], ensure_ascii=False
        ),
        "annual_plus_6k_exhibit_evidence_count": len(all_records),
    }
    if reason != "PASS":
        return result
    symbol = next(iter(symbols))
    nearest = exhibit_records[0]
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
    bindings = {
        resolver.QUEUE_PATH: policy["activation_queue_sha256"],
        PROSPECTUS_AUDIT_PATH: policy["activation_prospectus_audit_sha256"],
        PROVIDER_RETRY_PATH: policy["activation_provider_retry_report_sha256"],
    }
    for path, expected in bindings.items():
        if resolver.sha256(path) != expected:
            raise RuntimeError(f"ANNUAL_PLUS_6K_ACTIVATION_BINDING_MISMATCH:{path.name}")
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    indices = list(queue.loc[candidate_mask(queue, policy)].index)
    ids = queue.loc[indices, "event_id"].astype(str).tolist()
    digest = hybrid.event_ids_sha256(ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("ANNUAL_PLUS_6K_CANDIDATE_COUNT_MISMATCH")
    if digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("ANNUAL_PLUS_6K_CANDIDATE_HASH_MISMATCH")
    official_before = set(pd.read_csv(DEV_PATH, usecols=["event_id"]).event_id.astype(str))
    client = resolver.SECArchiveClient(min_interval=0.55)
    promoted_indices: list[Any] = []
    status_counts: dict[str, int] = {}
    for index in indices:
        result = corroborate_one(queue.loc[index].copy(), client, policy, policy_sha)
        promoted = bool(result.pop("promoted", False))
        status = str(result["annual_plus_6k_exhibit_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        for key, value in result.items():
            queue.loc[index, key] = value
        if promoted:
            queue.loc[index, "resolution_status"] = "RESOLVED_SAME_TICKER"
            queue.loc[index, "fetch_status"] = "ANNUAL_PLUS_6K_EXHIBIT_PRICE_REUSE_PENDING"
            queue.loc[index, "fetch_error"] = ""
            queue.loc[index, "exact_eligible"] = False
            promoted_indices.append(index)

    exact_indices: list[Any] = []
    for index in promoted_indices:
        if str(queue.loc[index, "event_id"]) in official_before:
            queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_6K_PROMOTION"
            continue
        authority = annual.exact_cache_candidate(queue.loc[index])
        if authority is None:
            queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_6K_PROMOTION"
            continue
        path, source, feed = authority
        queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_ANNUAL_PLUS_6K_EXHIBIT"
        queue.loc[index, "exact_eligible"] = True
        queue.loc[index, "provider"] = f"{source}_{feed}"
        queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
        queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
        queue.loc[index, "raw_sha256"] = resolver.sha256(path)
        queue.loc[index, "annual_plus_6k_exhibit_exact_provider"] = source
        queue.loc[index, "annual_plus_6k_exhibit_exact_feed"] = feed
        exact_indices.append(index)
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    if exact_indices:
        annual.update_provider_status(queue.loc[exact_indices].rename(columns={
            "annual_plus_6k_exhibit_exact_provider": "annual_exact_provider",
            "annual_plus_6k_exhibit_exact_feed": "annual_exact_feed",
        }))
    recovery = resolver.recovery_report(queue)
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "policy_frozen_before_exhibit_results": True,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": digest,
        "promoted_to_B": len(promoted_indices),
        "new_exact_cache_rows_after_promotion": len(exact_indices),
        "status_counts": status_counts,
        "SEC_network_requests": client.network_requests,
        "SEC_cache_hits": client.cache_hits,
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
