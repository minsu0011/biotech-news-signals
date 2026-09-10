"""Recover event-date ticker identity from the complete SEC submission text."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
CACHE = ROOT / "cache" / "historical_symbol_resolver" / "complete_submissions"
QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
POLICY_PATH = DATA / "SAME_ACCESSION_SUBMISSION_SYMBOL_POLICY_V1.json"
AUDIT_PATH = RESEARCH / "SAME_ACCESSION_SUBMISSION_SYMBOL_AUDIT.json"
ARCHIVE_SUBMISSION = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{accession}.txt"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "policy_name": "SAME_ACCESSION_FULL_SUBMISSION_EXPLICIT_SYMBOL_V1",
        "frozen_before_resolution_results": True,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "same_accession_required": True,
        "same_event_filing_time_required": True,
        "future_evidence_forbidden": True,
        "eligible_role": "DEV_EXTENSION",
        "multiple_strong_common_equity_symbols_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("SAME_ACCESSION_POLICY_GUARD_FAIL")
    return policy, sha256(POLICY_PATH)


def complete_submission(
    client: resolver.SECArchiveClient,
    cik: str,
    accession: str,
) -> tuple[str, Path, str]:
    accession_name = str(accession).strip()
    compact = re.sub(r"[^0-9]", "", accession_name)
    path = CACHE / f"{int(cik)}_{compact}.txt"
    raw = client.get(
        ARCHIVE_SUBMISSION.format(
            cik=int(cik), compact=compact, accession=quote(accession_name)
        ),
        path,
    )
    return raw.decode("utf-8", errors="replace"), path, sha256(path)


def is_warrant_target(symbol: str) -> bool:
    return bool(re.search(r"(?:WS|W)$", resolver.normalize_symbol(symbol)))


def strong_symbol_evidence(
    evidence: list[dict[str, Any]],
    queue_ticker: str,
    minimum_score: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    warrant_target = is_warrant_target(queue_ticker)
    usable: list[dict[str, Any]] = []
    for row in evidence:
        symbol = resolver.normalize_symbol(row.get("symbol"))
        title = str(row.get("security_title", ""))
        if not symbol or int(row.get("score", 0)) < minimum_score:
            continue
        excluded = bool(resolver.EXCLUDED_TITLE.search(title))
        if warrant_target:
            if not (re.search(r"(?i)warrant", title) or re.search(r"(?:WS|W)$", symbol)):
                continue
        elif excluded:
            continue
        usable.append({**row, "symbol": symbol})
    symbols = sorted({str(row["symbol"]) for row in usable})
    if not symbols:
        return "", "NO_STRONG_EXPLICIT_SYMBOL", usable
    if len(symbols) > 1:
        return "", "CONFLICTING_STRONG_SYMBOLS", usable
    return symbols[0], "PASS", usable


def run(
    years: set[int] | None,
    max_events: int | None,
    min_interval: float,
) -> dict[str, Any]:
    policy, policy_hash = load_policy()
    queue = pd.read_parquet(QUEUE_PATH)
    event_year = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    eligible = queue.role.astype(str).eq(str(policy["eligible_role"]))
    eligible &= queue.resolution_confidence.astype(str).isin(set(policy["eligible_prior_confidence"]))
    eligible &= ~queue.resolution_status.astype(str).isin(set(policy["excluded_resolution_statuses"]))
    eligible &= ~queue.form.astype(str).str.upper().isin(set(policy["excluded_forms"]))
    eligible &= queue.CIK.astype(str).ne("") & queue.accession.astype(str).ne("")
    if years is not None:
        eligible &= event_year.isin(years)
    candidates = queue.loc[eligible].copy()
    priority_years = [2023, 2024, 2019, 2022, 2021, 2020, 2018, 2017, 2016, 2025, 2026]
    rank = {year: index for index, year in enumerate(priority_years)}
    candidates["_year_rank"] = event_year.loc[candidates.index].map(
        lambda value: rank.get(int(value), len(rank))
    )
    candidates = candidates.sort_values(["_year_rank", "event_time_utc", "event_id"])
    if max_events is not None:
        candidates = candidates.head(max_events)
    client = resolver.SECArchiveClient(min_interval=min_interval)
    results: Counter[str] = Counter()
    promotions: Counter[int] = Counter()
    audit_rows: list[dict[str, Any]] = []
    minimum_score = int(policy["minimum_explicit_evidence_score"])
    for index, row in candidates.iterrows():
        try:
            raw, evidence_path, evidence_hash = complete_submission(
                client, str(row.CIK), str(row.accession)
            )
            evidence = resolver.extract_trading_symbols(raw)
            symbol, status, strong = strong_symbol_evidence(
                evidence, str(row.ticker), minimum_score
            )
        except Exception as error:
            symbol = ""
            status = f"FETCH_OR_PARSE_ERROR:{type(error).__name__}"
            strong = []
            evidence_path = Path()
            evidence_hash = ""
        results[status] += 1
        promoted = status == "PASS" and bool(symbol) and bool(evidence_hash)
        if promoted:
            queue_ticker = resolver.normalize_symbol(row.ticker)
            queue.loc[index, "resolved_ticker"] = symbol
            queue.loc[index, "resolution_method"] = policy["promotion"]["resolution_method"]
            queue.loc[index, "resolution_confidence"] = policy["promotion"]["resolution_confidence"]
            queue.loc[index, "resolution_asof_mode"] = policy["promotion"]["resolution_asof_mode"]
            queue.loc[index, "resolution_status"] = (
                "RESOLVED_SAME_TICKER" if symbol == queue_ticker else "RESOLVED_NEW_TICKER"
            )
            queue.loc[index, "resolution_error"] = ""
            queue.loc[index, "evidence_file"] = str(evidence_path.relative_to(ROOT))
            queue.loc[index, "evidence_sha256"] = evidence_hash
            queue.loc[index, "evidence_accession"] = str(row.accession)
            queue.loc[index, "evidence_filing_time_utc"] = str(row.event_time_utc)
            queue.loc[index, "ticker_candidates"] = json.dumps(
                strong, ensure_ascii=False, sort_keys=True, default=str
            )
            queue.loc[index, "fetch_status"] = "ALPACA_RETRY_PENDING"
            queue.loc[index, "fetch_error"] = ""
            queue.loc[index, "exact_eligible"] = False
            promotions[int(event_year.loc[index])] += 1
        audit_rows.append({
            "event_id": str(row.event_id),
            "CIK": str(row.CIK),
            "accession": str(row.accession),
            "event_date": str(row.event_date),
            "year": int(event_year.loc[index]),
            "form": str(row.form),
            "prior_confidence": str(row.resolution_confidence),
            "queue_ticker": str(row.ticker),
            "resolved_ticker": symbol if promoted else "",
            "status": status,
            "strong_symbols": sorted({str(value.get("symbol", "")) for value in strong}),
            "strong_methods": sorted({str(value.get("method", "")) for value in strong}),
            "evidence_file": str(evidence_path.relative_to(ROOT)) if evidence_path.is_file() else "",
            "evidence_sha256": evidence_hash,
            "promoted": promoted,
            "selection_uses_labels": False,
        })
        atomic_parquet(queue, QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    audit_frame = pd.DataFrame(audit_rows)
    audit_csv = RESEARCH / "SAME_ACCESSION_SUBMISSION_SYMBOL_DETAIL.csv"
    resolver.atomic_csv(audit_frame, audit_csv)
    report = {
        "schema_version": 1,
        "timestamp": now(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "candidate_rows": int(len(candidates)),
        "result_counts": dict(results),
        "promoted_rows": int(sum(promotions.values())),
        "promoted_by_year": {str(key): int(value) for key, value in sorted(promotions.items())},
        "sec_network_requests": int(client.network_requests),
        "sec_cache_hits": int(client.cache_hits),
        "detail_path": str(audit_csv.relative_to(ROOT)),
        "detail_sha256": sha256(audit_csv),
        "queue_sha256": sha256(QUEUE_PATH),
    }
    atomic_json(report, AUDIT_PATH)
    return report


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--sec-min-interval", type=float, default=0.16)
    args = parser.parse_args()
    print(json.dumps(
        run(parse_years(args.years), args.max_events, args.sec_min_interval),
        ensure_ascii=False, indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
