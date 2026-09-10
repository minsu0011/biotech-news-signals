"""Collect recent healthcare-SIC SEC events from a daily master index."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import bio_news_30m_v3 as app
import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "v221_daily_sic"
POLICY_PATH = DATA / "V221_FRESH_SEC_SIC_COLLECTION_POLICY_V1.json"
OUTPUT_PATH = DATA / "events_sec_v221_sic_fresh_20260828.csv"
REPORT_PATH = DATA / "V221_FRESH_SEC_SIC_COLLECTION_STATUS.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{compact}.idx"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fetch(
    session: requests.Session, url: str, path: Path, *, reuse_existing: bool = False
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and path.exists():
        return path.read_bytes()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(response.content)
    os.replace(temporary, path)
    time.sleep(0.12)
    return response.content


def parse_index(raw: str, allowed_forms: set[str]) -> pd.DataFrame:
    rows = []
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, filed, filename = (value.strip() for value in parts)
        if not cik.isdigit() or form not in allowed_forms:
            continue
        accession = Path(filename).stem
        rows.append({
            "cik": str(int(cik)), "company": company, "form": form,
            "filed": filed, "filename": filename, "accession": accession,
        })
    return pd.DataFrame(rows).drop_duplicates(["cik", "accession"])


def primary_symbol(submission: dict[str, Any]) -> str:
    tickers = [str(value).upper().replace(".", "-").strip() for value in submission.get("tickers", [])]
    exchanges = [str(value).upper().strip() for value in submission.get("exchanges", [])]
    ranked = []
    for index, ticker in enumerate(tickers):
        if (
            not re.fullmatch(r"[A-Z][A-Z0-9-]{0,4}", ticker)
            or ticker.endswith(("W", "P", "R", "U"))
        ):
            continue
        exchange = exchanges[index] if index < len(exchanges) else ""
        priority = 0 if any(name in exchange for name in ("NASDAQ", "NYSE")) else 1
        ranked.append((priority, ticker))
    return sorted(ranked)[0][1] if ranked else ""


def filing_metadata(submission: dict[str, Any], accession: str) -> dict[str, Any] | None:
    recent = ((submission.get("filings") or {}).get("recent") or {})
    accessions = [str(value) for value in recent.get("accessionNumber", [])]
    try:
        index = accessions.index(accession)
    except ValueError:
        return None
    return {
        key: values[index] if index < len(values) else ""
        for key, values in recent.items()
        if isinstance(values, list)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-08-28")
    args = parser.parse_args()
    day = pd.Timestamp(args.date)
    compact_day = day.strftime("%Y%m%d")
    policy_path = (
        POLICY_PATH if compact_day == "20260828"
        else DATA / f"V221_FRESH_SEC_SIC_COLLECTION_POLICY_{compact_day}_V1.json"
    )
    output_path = DATA / f"events_sec_v221_sic_fresh_{compact_day}.csv"
    report_path = DATA / f"V221_FRESH_SEC_SIC_COLLECTION_STATUS_{compact_day}.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("frozen_before_collection_results") is not True
        or policy.get("selection_uses_labels") is not False
        or policy.get("exact_price_contract_unchanged") is not True
    ):
        raise RuntimeError("FRESH_SIC_COLLECTION_POLICY_GUARD_FAIL")
    session = requests.Session()
    session.headers.update(app.SEC_HEADERS)
    if str(policy.get("date")) != day.strftime("%Y-%m-%d"):
        raise RuntimeError("FRESH_SIC_COLLECTION_DATE_POLICY_MISMATCH")
    quarter = (int(day.month) - 1) // 3 + 1
    index_url = INDEX_URL.format(
        year=day.year, quarter=quarter, compact=compact_day
    )
    index_raw = fetch(session, index_url, CACHE / f"master.{compact_day}.idx").decode(
        "latin-1", errors="replace"
    )
    candidates = parse_index(index_raw, set(app.Config().sec_forms))
    rows: list[dict[str, Any]] = []
    fetched_ciks = 0
    healthcare_ciks = 0
    for cik, group in candidates.groupby("cik", sort=True):
        try:
            raw = fetch(
                session, SUBMISSIONS_URL.format(cik=int(cik)),
                CACHE / "submissions" / f"CIK{int(cik):010d}.json",
                reuse_existing=True,
            )
            submission = json.loads(raw.decode("utf-8"))
            fetched_ciks += 1
        except Exception:
            continue
        sic = str(submission.get("sic", "") or "")
        if sic not in exact.APPROVED_HEALTHCARE_SIC:
            continue
        ticker = primary_symbol(submission)
        if not ticker:
            continue
        healthcare_ciks += 1
        for source in group.itertuples(index=False):
            metadata = filing_metadata(submission, str(source.accession))
            if metadata is None:
                continue
            raw_time = str(metadata.get("acceptanceDateTime", ""))
            if not raw_time or "T" not in raw_time:
                continue
            timestamp = pd.Timestamp(raw_time)
            timestamp = timestamp.tz_localize(app.US_TZ).tz_convert("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
            local, session_open, session_close = app.session_bounds(timestamp, "US")
            if not session_open <= local <= session_close:
                continue
            if local + pd.Timedelta(minutes=33) > session_close:
                continue
            form = str(metadata.get("form") or source.form)
            items = str(metadata.get("items") or "")
            headline = f"{form} {items}".strip()
            rows.append({
                "event_id": f"SEC:{int(cik)}:{source.accession}",
                "market": "US",
                "ticker": ticker,
                "company": str(submission.get("name") or source.company),
                "event_time_utc": timestamp.isoformat(),
                "source": "SEC_V221_SIC_FRESH",
                "form": form,
                "headline": headline,
                "body": "",
                "event_type": app.infer_event_type(headline, form),
                "timestamp_quality": "EXACT",
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{str(source.accession).replace('-', '')}/",
                "cik": str(int(cik)),
                "sic": sic,
            })
    output = pd.DataFrame(rows).drop_duplicates("event_id").sort_values("event_time_utc")
    output.to_csv(output_path, index=False)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy_path": str(policy_path.relative_to(ROOT)),
        "policy_sha256": sha256(policy_path),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "daily_index_allowed_form_rows": int(len(candidates)),
        "candidate_ciks": int(candidates.cik.nunique()),
        "submissions_fetched": int(fetched_ciks),
        "healthcare_sic_ciks_with_primary_symbol": int(healthcare_ciks),
        "events": int(len(output)),
        "unique_tickers": int(output.ticker.nunique()) if not output.empty else 0,
        "output_path": str(output_path.relative_to(ROOT)),
        "output_sha256": sha256(output_path),
    }
    atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
