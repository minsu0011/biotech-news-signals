"""Point-in-time 20-F/40-F symbol corroboration for frozen V221 DEV rows.

Promotion is identity-only and follows a policy frozen before annual-document
results were downloaded. Price/cache availability is consulted only after all
promotions are complete, and only to measure incremental exact eligibility.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import alpaca_backfill_v221 as alpaca_backfill
import bio_news_30m_v3 as app
import historical_symbol_resolver as resolver
import v59_provider_acquisition as provider


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "HISTORICAL_ANNUAL_REPORT_SYMBOL_CONSENSUS_POLICY_V1.json"
AUDIT_PATH = RESEARCH / "HISTORICAL_ANNUAL_REPORT_SYMBOL_CONSENSUS_AUDIT.json"
DEV_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 1,
        "frozen_before_annual_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "current_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("ANNUAL_REPORT_POLICY_GUARD_FAIL")
    if int(policy.get("minimum_independent_prior_annual_filings", 0)) < 2:
        raise RuntimeError("ANNUAL_REPORT_POLICY_REQUIRES_TWO_FILINGS")
    return policy, resolver.sha256(POLICY_PATH)


def annual_columns() -> dict[str, Any]:
    return {
        "annual_consensus_status": "NOT_ATTEMPTED",
        "annual_consensus_contract_sha256": "",
        "annual_corroborating_accessions": "[]",
        "annual_corroborating_evidence_files": "[]",
        "annual_corroborating_evidence_sha256": "[]",
        "annual_consensus_evidence_count": 0,
    }


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column, default in annual_columns().items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def evaluate_records(
    records: list[dict[str, Any]], ambiguous_filings: int, policy: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    if ambiguous_filings and policy["ambiguous_annual_symbol_evidence_forbidden"]:
        return "", "AMBIGUOUS_ANNUAL_SYMBOL_EVIDENCE", []
    unique = {str(row["accession"]): row for row in records if row.get("accession")}
    records = list(unique.values())
    symbols = sorted({resolver.normalize_symbol(row.get("symbol")) for row in records} - {""})
    if len(symbols) > 1 and policy["contradictory_annual_symbol_evidence_forbidden"]:
        return "", "CONTRADICTORY_ANNUAL_SYMBOL_EVIDENCE", []
    if len(symbols) != 1:
        return "", "NO_SINGLE_ANNUAL_CONSENSUS_SYMBOL", []
    matching = sorted(
        [row for row in records if resolver.normalize_symbol(row.get("symbol")) == symbols[0]],
        key=lambda row: float(row["days_before_event"]),
    )
    if len(matching) < int(policy["minimum_independent_prior_annual_filings"]):
        return "", "INSUFFICIENT_INDEPENDENT_PRIOR_ANNUAL_FILINGS", matching
    days = [float(row["days_before_event"]) for row in matching]
    if min(days) > float(policy["maximum_nearest_prior_days"]):
        return "", "NEAREST_PRIOR_ANNUAL_OUTSIDE_WINDOW", matching
    if max(days) > float(policy["maximum_oldest_prior_days"]):
        return "", "OLDEST_PRIOR_ANNUAL_OUTSIDE_WINDOW", matching
    if policy["evidence_file_sha256_required"] and any(
        not str(row.get("evidence_sha256", "")) for row in matching
    ):
        return "", "ANNUAL_EVIDENCE_SHA_MISSING", matching
    return symbols[0], "PASS", matching


def corroborate_one(
    row: pd.Series,
    client: resolver.SECArchiveClient,
    policy: dict[str, Any],
    policy_sha256: str,
) -> dict[str, Any]:
    cik, event_accession = resolver.parse_event_id(str(row.event_id))
    base = {
        "promoted": False,
        "annual_consensus_status": "NOT_ELIGIBLE",
        "annual_consensus_contract_sha256": policy_sha256,
        "annual_corroborating_accessions": "[]",
        "annual_corroborating_evidence_files": "[]",
        "annual_corroborating_evidence_sha256": "[]",
        "annual_consensus_evidence_count": 0,
    }
    if not cik or not event_accession:
        return {**base, "annual_consensus_status": "CIK_OR_ACCESSION_MISSING"}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {**base, "annual_consensus_status": f"SEC_SUBMISSIONS_{type(error).__name__}"}

    event_time = pd.Timestamp(row.event_time_utc)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    annual_forms = {str(value).upper() for value in policy["allowed_prior_annual_forms"]}
    prior: list[tuple[float, dict[str, Any]]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        form = str(filing.get("form") or "").upper()
        stamp = resolver.filing_time(filing)
        if (
            form not in annual_forms or not accession or accession == event_accession
            or stamp is None or stamp >= event_time
        ):
            continue
        days = (event_time - stamp).total_seconds() / 86400.0
        if 0 <= days <= float(policy["maximum_oldest_prior_days"]):
            prior.append((days, filing))
    prior = sorted(prior, key=lambda item: item[0])[: int(policy["maximum_annual_filings_scanned"])]
    if not prior:
        return {**base, "annual_consensus_status": "NO_PRIOR_ANNUAL_FILINGS_IN_WINDOW"}

    allowed_methods = {str(value) for value in policy["allowed_explicit_symbol_methods"]}
    records: list[dict[str, Any]] = []
    ambiguous_filings = 0
    for days, filing in prior:
        accession = str(filing.get("accessionNumber"))
        try:
            evidence, path, digest = resolver.inspect_filing_evidence(cik, accession, filing, client)
        except Exception:
            continue
        evidence = [item for item in evidence if str(item.get("method")) in allowed_methods]
        # The policy forbids using the mutable/current queue ticker as a tie-break.
        symbol, ambiguous = resolver.choose_symbol(evidence, "")
        if ambiguous:
            ambiguous_filings += 1
            continue
        if not symbol or not path.is_file() or not digest:
            continue
        records.append({
            "accession": accession,
            "filing_time_utc": str(resolver.filing_time(filing) or ""),
            "days_before_event": round(days, 6),
            "symbol": symbol,
            "evidence_file": str(path.relative_to(ROOT)),
            "evidence_sha256": digest,
            "evidence_methods": sorted({str(item.get("method", "")) for item in evidence}),
            "evidence": evidence,
        })

    symbol, reason, matching = evaluate_records(records, ambiguous_filings, policy)
    audit_records = matching if matching else records
    result = {
        **base,
        "annual_consensus_status": reason,
        "annual_corroborating_accessions": json.dumps(
            [record["accession"] for record in audit_records], ensure_ascii=False
        ),
        "annual_corroborating_evidence_files": json.dumps(
            [record["evidence_file"] for record in audit_records], ensure_ascii=False
        ),
        "annual_corroborating_evidence_sha256": json.dumps(
            [record["evidence_sha256"] for record in audit_records], ensure_ascii=False
        ),
        "annual_consensus_evidence_count": len(matching),
    }
    if not symbol:
        return result
    nearest = matching[0]
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


def candidate_mask(queue: pd.DataFrame, policy: dict[str, Any], policy_sha256: str) -> pd.Series:
    mask = queue.source_family.astype(str).eq(str(policy["source_family"]))
    mask &= queue.role.astype(str).isin({str(value) for value in policy["eligible_frozen_roles"]})
    mask &= queue.resolution_confidence.astype(str).isin(
        {str(value) for value in policy["eligible_resolution_confidence_before"]}
    )
    mask &= queue.resolution_status.astype(str).isin(
        {str(value) for value in policy["eligible_prior_resolution_status"]}
    )
    excluded = tuple(str(value).upper() for value in policy["registration_forms_excluded"])
    mask &= ~queue.form.astype(str).str.upper().str.startswith(excluded)
    mask &= queue.annual_consensus_contract_sha256.astype(str).ne(policy_sha256)
    return mask


def exact_cache_candidate(row: pd.Series) -> tuple[Path, str, str] | None:
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    date = event_time.tz_convert(app.US_TZ).date()
    symbol = resolver.normalize_symbol(row.resolved_ticker)
    if not symbol:
        return None
    massive = provider._cache_path(provider.MASSIVE_CACHE, symbol, date)
    candidates = [(massive, "MASSIVE", "SIP")]
    if provider.authorized_alpaca_feed() == "SIP":
        candidates.append((provider._cache_path(provider.ALPACA_CACHE / "sip", symbol, date), "ALPACA", "SIP"))
    for path, source, feed in candidates:
        if not path.is_file():
            continue
        try:
            bars = pd.read_parquet(path, columns=["timestamp"])
            exact, _, _ = alpaca_backfill.timing_only_eligibility(event_time, bars["timestamp"])
        except Exception:
            exact = False
        if exact:
            return path, source, feed
    return None


def update_provider_status(exact_rows: pd.DataFrame) -> None:
    if exact_rows.empty:
        return
    status = pd.read_parquet(resolver.PROVIDER_STATUS_PATH).set_index("event_id")
    indexed = exact_rows.set_index("event_id")
    common = status.index.intersection(indexed.index)
    for event_id in common:
        row = indexed.loc[event_id]
        source = str(row.annual_exact_provider)
        feed = str(row.annual_exact_feed)
        if source == "MASSIVE":
            status.loc[event_id, "massive_status"] = "CACHED_VALID"
        else:
            status.loc[event_id, "alpaca_sip_status"] = "CACHED_VALID"
        status.loc[event_id, "selected_price_provider"] = f"{source}_{feed}"
        status.loc[event_id, "provider_parity_eligible"] = True
        status.loc[event_id, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
        status.loc[event_id, "provider_symbol"] = str(row.resolved_ticker)
        status.loc[event_id, "symbol_resolution_method"] = str(row.resolution_method)
        status.loc[event_id, "symbol_resolution_confidence"] = str(row.resolution_confidence)
        status.loc[event_id, "resolution_asof_mode"] = str(row.resolution_asof_mode)
    resolver.atomic_parquet(status.reset_index(), resolver.PROVIDER_STATUS_PATH)


def main() -> int:
    policy, policy_sha256 = load_policy()
    queue = ensure_columns(pd.read_parquet(resolver.QUEUE_PATH))
    official_before = set(pd.read_csv(DEV_PATH).event_id.astype(str))
    selected = candidate_mask(queue, policy, policy_sha256)
    indices = list(queue.loc[selected].sort_values(["priority", "event_time_utc", "event_id"]).index)
    selection_digest = hashlib.sha256(
        "\n".join(sorted(queue.loc[indices, "event_id"].astype(str))).encode()
    ).hexdigest()

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

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(grouped)))) as executor:
        futures = [executor.submit(group_work, group) for group in sorted(grouped.values(), key=lambda x: (len(x), x[0]))]
        for future in as_completed(futures):
            rows, requests_used, hits = future.result()
            network_requests += requests_used
            cache_hits += hits
            for index, result in rows:
                promoted = bool(result.pop("promoted", False))
                status_counts[str(result.get("annual_consensus_status", "UNKNOWN"))] += 1
                for key, value in result.items():
                    queue.loc[index, key] = value
                if promoted:
                    symbol = resolver.normalize_symbol(result["resolved_ticker"])
                    queue.loc[index, "resolution_status"] = (
                        "RESOLVED_SAME_TICKER"
                        if symbol == resolver.normalize_symbol(queue.loc[index, "ticker"])
                        else "RESOLVED_NEW_TICKER"
                    )
                    queue.loc[index, "fetch_status"] = "ANNUAL_CONSENSUS_PRICE_REUSE_PENDING"
                    queue.loc[index, "fetch_error"] = ""
                    queue.loc[index, "exact_eligible"] = False
                    promoted_indices.append(index)

    promoted = queue.loc[promoted_indices].copy() if promoted_indices else queue.iloc[0:0].copy()
    exact_new_indices: list[Any] = []
    exact_existing_official = 0
    if not promoted.empty:
        queue["annual_exact_provider"] = queue.get("annual_exact_provider", "")
        queue["annual_exact_feed"] = queue.get("annual_exact_feed", "")
        for index in promoted_indices:
            event_id = str(queue.loc[index, "event_id"])
            if event_id in official_before:
                exact_existing_official += 1
                queue.loc[index, "fetch_status"] = "ALREADY_OFFICIAL_DEV_BEFORE_ANNUAL_PROMOTION"
                continue
            authority = exact_cache_candidate(queue.loc[index])
            if authority is None:
                queue.loc[index, "fetch_status"] = "NO_EXISTING_EXACT_CACHE_AFTER_ANNUAL_PROMOTION"
                continue
            path, source, feed = authority
            queue.loc[index, "fetch_status"] = f"{source}_CACHED_EXACT_ANNUAL_CONSENSUS"
            queue.loc[index, "exact_eligible"] = True
            queue.loc[index, "provider"] = f"{source}_{feed}"
            queue.loc[index, "provider_symbol"] = str(queue.loc[index, "resolved_ticker"])
            queue.loc[index, "raw_file"] = str(path.relative_to(ROOT))
            queue.loc[index, "raw_sha256"] = resolver.sha256(path)
            queue.loc[index, "annual_exact_provider"] = source
            queue.loc[index, "annual_exact_feed"] = feed
            exact_new_indices.append(index)

    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    exact_rows = queue.loc[exact_new_indices].copy() if exact_new_indices else queue.iloc[0:0].copy()
    update_provider_status(exact_rows)
    recovery = resolver.recovery_report(queue)
    audit = {
        "schema_version": 1,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha256,
        "policy_frozen_before_annual_document_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded_for_promotion": False,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": selection_digest,
        "selected_ciks": len(grouped),
        "promoted_to_B": len(promoted_indices),
        "promoted_same_ticker": int(
            (promoted.resolved_ticker.astype(str) == promoted.ticker.astype(str)).sum()
        ) if not promoted.empty else 0,
        "promoted_changed_ticker": int(
            (promoted.resolved_ticker.astype(str) != promoted.ticker.astype(str)).sum()
        ) if not promoted.empty else 0,
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
