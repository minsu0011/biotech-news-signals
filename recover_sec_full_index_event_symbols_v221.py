"""Recover novel historical SEC events from the official quarterly full index.

The candidate and identity pass is label/role/price blind.  A ticker is
accepted only when the event accession's own primary filing document contains
one unambiguous explicit common-equity trading symbol.  Current submissions
ticker metadata is deliberately never used as historical identity evidence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import historical_symbol_resolver as resolver
import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "v221_sec_full_index"
POLICY_PATH = DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_POLICY_V1.json"
UNIVERSE_PATH = DATA / "universe_v221_fresh_sec_expanded.csv"
PRIOR_PATH = DATA / "events_exact_v35.csv.gz"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
OUTPUT_PATH = DATA / "events_sec_v221_legacy_gap_full_index_20240903_20250531.csv"
IDENTITY_AUDIT_PATH = DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_IDENTITY_AUDIT.parquet"
REPORT_PATH = DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_STATUS_V1.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.gz"
FORBIDDEN_OUTCOME_COLUMNS = {
    "y", "label", "target", "fwd_ret_30m", "entry_price", "exit_price",
    "actual_entry_time_utc", "actual_exit_time_utc", "prediction", "probability",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace(temporary: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "frozen_before_collection_identity_or_price_results": True,
        "selection_uses_roles": False,
        "selection_uses_labels": False,
        "selection_uses_prices": False,
        "outcome_columns_loaded": False,
        "same_accession_primary_document_required": True,
        "current_submissions_ticker_as_identity_forbidden": True,
        "queue_ticker_tie_break_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("SEC_FULL_INDEX_POLICY_GUARD_FAIL")
    if policy.get("policy_name") not in {
        "V221_SEC_FULL_INDEX_SAME_ACCESSION_SYMBOL_RECOVERY_V1",
        "V221_SEC_FULL_INDEX_SAME_ACCESSION_SYMBOL_RECOVERY_V2",
        "V221_SEC_FULL_INDEX_SAME_ACCESSION_SYMBOL_RECOVERY_V3",
    }:
        raise RuntimeError("SEC_FULL_INDEX_POLICY_NAME_GUARD_FAIL")
    if sha256(UNIVERSE_PATH) != policy["current_healthcare_universe_sha256"]:
        raise RuntimeError("SEC_FULL_INDEX_UNIVERSE_HASH_MISMATCH")
    return policy, sha256(POLICY_PATH)


def parse_master(raw: bytes, allowed_forms: set[str], allowed_ciks: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    text = gzip.decompress(raw).decode("latin-1", errors="replace")
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, filed, filename = (value.strip() for value in parts)
        normalized_cik = str(int(cik)) if cik.isdigit() else ""
        if not normalized_cik or normalized_cik not in allowed_ciks or form not in allowed_forms:
            continue
        accession = Path(filename).stem
        rows.append({
            "cik": normalized_cik,
            "index_company": company,
            "index_form": form,
            "filed": filed,
            "filename": filename,
            "accession": accession,
        })
    columns = ["cik", "index_company", "index_form", "filed", "filename", "accession"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(["cik", "accession"])


def event_symbol(
    evidence: list[dict[str, Any]], allowed_methods: set[str]
) -> tuple[str, str, list[dict[str, Any]]]:
    usable = []
    for item in evidence:
        method = str(item.get("method", ""))
        symbol = resolver.normalize_symbol(item.get("symbol"))
        title = str(item.get("security_title", "") or "")
        if method not in allowed_methods or not symbol:
            continue
        if resolver.EXCLUDED_TITLE.search(title):
            continue
        usable.append({**item, "symbol": symbol})
    symbols = sorted({str(item["symbol"]) for item in usable})
    if not symbols:
        return "", "NO_ALLOWED_EXPLICIT_COMMON_SYMBOL", usable
    if len(symbols) != 1:
        return "", "AMBIGUOUS_EXPLICIT_COMMON_SYMBOLS", usable
    return symbols[0], "PASS", usable


def outside_active_spacing(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty or not QUEUE_PATH.exists():
        return frame.copy()
    queue = pd.read_parquet(QUEUE_PATH, columns=["ticker", "event_time_utc"])
    active_times = {
        str(ticker): np.sort(pd.to_datetime(group.event_time_utc, utc=True).astype("int64").to_numpy())
        for ticker, group in queue.groupby("ticker")
    }
    embargo = int(pd.Timedelta(minutes=minutes).value)
    keep: list[int] = []
    for index, row in frame.iterrows():
        values = active_times.get(str(row.ticker))
        if values is None or not len(values):
            keep.append(index)
            continue
        target = int(pd.Timestamp(row.event_time_utc).value)
        position = int(np.searchsorted(values, target))
        distances = []
        if position < len(values):
            distances.append(abs(int(values[position]) - target))
        if position:
            distances.append(abs(int(values[position - 1]) - target))
        if not distances or min(distances) > embargo:
            keep.append(index)
    return frame.loc[keep].copy()


def main() -> int:
    policy, policy_hash = load_policy()
    start = pd.Timestamp(policy["start"], tz="UTC")
    end = pd.Timestamp(policy["end"], tz="UTC") + pd.Timedelta(days=1)
    universe = pd.read_csv(UNIVERSE_PATH, dtype=str)
    if set(universe.columns) & FORBIDDEN_OUTCOME_COLUMNS:
        raise RuntimeError("SEC_FULL_INDEX_UNIVERSE_OUTCOME_COLUMN_REJECTED")
    universe = universe.loc[universe.market.eq("US") & universe.cik.fillna("").str.fullmatch(r"\d+")].copy()
    allowed_ciks = {str(int(value)) for value in universe.cik}
    client = resolver.SECArchiveClient(min_interval=0.16)
    index_frames: list[pd.DataFrame] = []
    index_hashes: dict[str, str] = {}
    for item in policy["quarterly_indexes"]:
        year_text, quarter_text = str(item).split("/QTR")
        year, quarter = int(year_text), int(quarter_text)
        cache_path = CACHE / "full-index" / f"{year}_QTR{quarter}_master.gz"
        raw = client.get(INDEX_URL.format(year=year, quarter=quarter), cache_path)
        index_hashes[str(cache_path.relative_to(ROOT))] = sha256(cache_path)
        index_frames.append(parse_master(raw, set(app.Config().sec_forms), allowed_ciks))
        print(f"INDEX {year} QTR{quarter}: {len(index_frames[-1])} universe/form rows", flush=True)
    index = pd.concat(index_frames, ignore_index=True).drop_duplicates(["cik", "accession"])
    filed = pd.to_datetime(index.filed, utc=True, errors="coerce")
    index = index.loc[(filed >= start) & (filed < end)].copy()
    stages: dict[str, int] = {
        "healthcare_universe_ciks": len(allowed_ciks),
        "index_form_rows_in_range": len(index),
    }
    prior = exact.normalized_events(PRIOR_PATH)
    excluded_ids = set(prior.event_id.astype(str))
    if ROLE_PATH.exists():
        roles = pd.read_parquet(ROLE_PATH, columns=["event_id"])
        excluded_ids.update(roles.event_id.astype(str))
    index["event_id"] = index.apply(
        lambda row: f"SEC:{int(row.cik)}:{row.accession}", axis=1
    )
    index = index.loc[~index.event_id.astype(str).isin(excluded_ids)].copy()
    stages["novel_vs_v35_and_frozen_roles"] = len(index)
    metadata_candidates: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    grouped = list(index.groupby("cik", sort=True))
    for ordinal, (cik, group) in enumerate(grouped, start=1):
        try:
            submission, filings = client.filing_rows(str(cik))
            filing_map = {str(row.get("accessionNumber") or ""): row for row in filings}
        except Exception as error:
            statuses[f"SUBMISSIONS_{type(error).__name__}"] += len(group)
            continue
        sic = str(submission.get("sic", "") or "")
        company = str(submission.get("name", "") or group.iloc[0].index_company)
        for source in group.itertuples(index=False):
            filing = filing_map.get(str(source.accession))
            if filing is None:
                statuses["ACCESSION_METADATA_MISSING"] += 1
                continue
            raw_acceptance = str(filing.get("acceptanceDateTime") or "")
            if "T" not in raw_acceptance:
                statuses["EXACT_ACCEPTANCE_TIME_MISSING"] += 1
                continue
            stamp = resolver.filing_time(filing)
            if stamp is None or not (start <= stamp < end):
                statuses["ACCEPTANCE_OUTSIDE_RANGE"] += 1
                continue
            local, session_open, session_close = app.session_bounds(stamp, "US")
            if not (session_open <= local <= session_close):
                statuses["OUTSIDE_REGULAR_SESSION"] += 1
                continue
            if local + pd.Timedelta(minutes=int(policy["minimum_exit_headroom_minutes"])) > session_close:
                statuses["INSUFFICIENT_EXIT_HEADROOM"] += 1
                continue
            metadata_candidates.append({
                "event_id": source.event_id,
                "cik": str(cik),
                "accession": source.accession,
                "event_time_utc": stamp,
                "form": str(filing.get("form") or source.index_form),
            })
            try:
                evidence, evidence_path, digest = resolver.inspect_filing_evidence(
                    str(cik), str(source.accession), filing, client
                )
                symbol, status, usable = event_symbol(
                    evidence, set(policy["allowed_identity_methods"])
                )
            except Exception as error:
                evidence, evidence_path, digest, usable = [], Path(), "", []
                symbol, status = "", f"EVENT_DOCUMENT_{type(error).__name__}"
            statuses[status] += 1
            identity_rows.append({
                "event_id": source.event_id,
                "cik": str(cik),
                "accession": source.accession,
                "event_time_utc": stamp,
                "form": str(filing.get("form") or source.index_form),
                "sic": sic,
                "identity_status": status,
                "resolved_ticker": symbol,
                "evidence_methods": json.dumps(
                    sorted({str(item.get("method", "")) for item in usable}), ensure_ascii=False
                ),
                "evidence_symbols": json.dumps(
                    sorted({str(item.get("symbol", "")) for item in usable}), ensure_ascii=False
                ),
                "evidence_file": str(evidence_path.relative_to(ROOT)) if evidence_path.is_file() else "",
                "evidence_sha256": digest,
                "current_submissions_ticker_used": False,
            })
            if status != "PASS" or not evidence_path.is_file() or not digest:
                continue
            form = str(filing.get("form") or source.index_form)
            items = str(filing.get("items") or "")
            headline = f"{form} {items}".strip()
            identity_rows[-1]["event_row"] = {
                "event_id": source.event_id,
                "market": "US",
                "ticker": symbol,
                "company": company,
                "event_time_utc": stamp.isoformat(),
                "source": "SEC_V221_FULL_INDEX_SAME_ACCESSION",
                "form": form,
                "headline": headline,
                "body": "",
                "event_type": app.infer_event_type(headline, form),
                "timestamp_quality": "EXACT",
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{str(source.accession).replace('-', '')}/"
                ),
                "article_id": "",
                "cik": str(int(cik)),
                "sic": sic,
                "identity_confidence": "A",
                "identity_method": "SAME_ACCESSION_EXPLICIT_TRADING_SYMBOL",
                "identity_evidence_sha256": digest,
            }
        if ordinal % 25 == 0 or ordinal == len(grouped):
            print(
                f"CIK {ordinal}/{len(grouped)} metadata={len(metadata_candidates)} "
                f"identity_pass={statuses['PASS']}",
                flush=True,
            )
    audit = pd.DataFrame([{key: value for key, value in row.items() if key != "event_row"}
                          for row in identity_rows])
    if audit.empty:
        audit = pd.DataFrame(columns=[
            "event_id", "cik", "accession", "event_time_utc", "form", "sic",
            "identity_status", "resolved_ticker", "evidence_methods", "evidence_symbols",
            "evidence_file", "evidence_sha256", "current_submissions_ticker_used",
        ])
    atomic_parquet(audit, IDENTITY_AUDIT_PATH)
    events = pd.DataFrame([row["event_row"] for row in identity_rows if "event_row" in row])
    stages["exact_timestamp_regular_session_metadata_candidates"] = len(metadata_candidates)
    stages["same_accession_identity_pass"] = len(events)
    if not events.empty:
        events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
        events, prior_stages = exact.outside_prior(events, prior)
        stages.update({f"v35_{key}": int(value) for key, value in prior_stages.items()})
        events = outside_active_spacing(
            events, int(policy["minimum_same_ticker_distance_from_active_queue_minutes"])
        )
        stages["outside_active_queue_spacing"] = len(events)
        events = exact.space_events(events).sort_values(["ticker", "event_time_utc", "event_id"])
    stages["final_frozen_identity_candidates"] = len(events)
    atomic_csv(events, OUTPUT_PATH)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "selection_uses_roles": False,
        "selection_uses_labels": False,
        "selection_uses_prices": False,
        "outcome_columns_loaded": False,
        "current_submissions_ticker_used": False,
        "index_file_hashes": index_hashes,
        "stages": stages,
        "identity_status_counts": dict(statuses),
        "sec_network_requests": client.network_requests,
        "sec_cache_hits": client.cache_hits,
        "identity_audit_path": str(IDENTITY_AUDIT_PATH.relative_to(ROOT)),
        "identity_audit_sha256": sha256(IDENTITY_AUDIT_PATH),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_PATH),
        "output_rows": len(events),
        "output_tickers": int(events.ticker.nunique()) if not events.empty else 0,
        "candidate_set_frozen_before_role_assignment_or_price_acquisition": True,
        "existing_roles_opened_or_changed": False,
        "exact_price_contract_unchanged": True,
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
