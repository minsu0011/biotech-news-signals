"""Collect one outcome-blind V224 prospective SEC day with exact filing text."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import bio_news_30m_v3 as app
import collect_sec_daily_sic_v221 as daily
import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
POLICY_REL = Path("research/V224_PROSPECTIVE_COLLECTION_POLICY_V1.json")
ROLE_POLICY_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION_POLICY_V1.json")
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "v224_prospective_sec"
INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{compact}.idx"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
OUTPUT_COLUMNS = [
    "event_id", "market", "ticker", "company", "event_time_utc", "source",
    "source_family", "form", "headline", "body", "event_type",
    "timestamp_quality", "url", "cik", "sic", "primary_document",
    "collection_time_utc", "collection_policy_sha256",
]


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_contract(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], str]:
    collection_path = root / POLICY_REL
    role_path = root / ROLE_POLICY_REL
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    role = json.loads(role_path.read_text(encoding="utf-8"))
    if not (
        collection.get("status") == "FROZEN_BEFORE_FIRST_PROSPECTIVE_ROW"
        and collection.get("selection_uses_labels") is False
        and collection.get("selection_uses_prices") is False
        and collection.get("exact_accession_body_required") is True
        and role.get("status") == "FROZEN_BEFORE_PROSPECTIVE_RESULTS"
    ):
        raise RuntimeError("V224_PROSPECTIVE_COLLECTION_POLICY_GUARD_FAIL")
    return collection, role, sha256(collection_path)


def collect_day(day_text: str, root: Path = ROOT) -> dict[str, Any]:
    collection, role_policy, policy_sha = load_contract(root)
    day = pd.Timestamp(day_text)
    if day.strftime("%Y-%m-%d") < str(collection["first_eligible_date_et"]):
        raise ValueError("V224_DATE_PRECEDES_PROSPECTIVE_COLLECTION_START")
    if day.weekday() >= 5:
        raise ValueError("V224_COLLECTION_DATE_NOT_BUSINESS_DAY")
    compact = day.strftime("%Y%m%d")
    output_path = root / "data" / f"events_sec_v224_prospective_{compact}.csv"
    status_path = root / "research" / "V224_PROSPECTIVE_CONFIRMATION" / f"COLLECTION_{compact}.json"
    allowed_forms = set(role_policy["role_contract"]["allowed_forms"])
    quarter = (int(day.month) - 1) // 3 + 1
    url = INDEX_URL.format(year=day.year, quarter=quarter, compact=compact)
    session = requests.Session()
    session.headers.update(app.SEC_HEADERS)
    index_bytes = daily.fetch(session, url, root / "cache" / "v224_prospective_sec" / f"master.{compact}.idx")
    candidates = daily.parse_index(index_bytes.decode("latin-1", errors="replace"), allowed_forms)
    collector = app.SECCollector(app.Config())
    rows: list[dict[str, Any]] = []
    submission_failures = 0
    text_failures = 0
    collected_at = datetime.now(timezone.utc).isoformat()
    for cik, group in candidates.groupby("cik", sort=True):
        try:
            payload = daily.fetch(
                session,
                SUBMISSIONS_URL.format(cik=int(cik)),
                root / "cache" / "v224_prospective_sec" / "submissions" / f"CIK{int(cik):010d}.json",
                reuse_existing=False,
            )
            submission = json.loads(payload.decode("utf-8"))
        except Exception:
            submission_failures += 1
            continue
        sic = str(submission.get("sic", "") or "")
        if sic not in exact.APPROVED_HEALTHCARE_SIC:
            continue
        ticker = daily.primary_symbol(submission)
        if not ticker:
            continue
        for source in group.itertuples(index=False):
            metadata = daily.filing_metadata(submission, str(source.accession))
            if metadata is None:
                continue
            raw_time = str(metadata.get("acceptanceDateTime", ""))
            if not raw_time or "T" not in raw_time:
                continue
            timestamp = pd.Timestamp(raw_time)
            timestamp = (
                timestamp.tz_localize(app.US_TZ).tz_convert("UTC")
                if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
            )
            local, session_open, session_close = app.session_bounds(timestamp, "US")
            if not session_open <= local <= session_close:
                continue
            if local + pd.Timedelta(minutes=33) > session_close:
                continue
            form = str(metadata.get("form") or source.form)
            items = str(metadata.get("items") or "")
            primary = str(metadata.get("primaryDocument") or "")
            try:
                body = collector.filing_text(int(cik), str(source.accession), primary, form)
            except Exception:
                text_failures += 1
                continue
            if not str(body).strip():
                text_failures += 1
                continue
            headline = f"{form} {items}".strip()
            rows.append({
                "event_id": f"SEC:{int(cik)}:{source.accession}",
                "market": "US", "ticker": ticker,
                "company": str(submission.get("name") or source.company),
                "event_time_utc": timestamp.isoformat(),
                "source": "SEC_V224_PROSPECTIVE", "source_family": "US_SEC",
                "form": form, "headline": headline, "body": body,
                "event_type": app.infer_event_type(f"{headline} {body}", form),
                "timestamp_quality": "EXACT",
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{str(source.accession).replace('-', '')}/",
                "cik": str(int(cik)), "sic": sic, "primary_document": primary,
                "collection_time_utc": collected_at,
                "collection_policy_sha256": policy_sha,
            })
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(output):
        output = output.drop_duplicates("event_id").sort_values(["event_time_utc", "event_id"])
    atomic_csv(output, output_path)
    status = {
        "schema_version": 1, "track": "V224_PROSPECTIVE_CONFIRMATION",
        "phase": "OUTCOME_BLIND_SEC_COLLECTION", "status": "COMPLETE",
        "collection_date_et": day.strftime("%Y-%m-%d"),
        "collection_policy_path": str(POLICY_REL).replace("\\", "/"),
        "collection_policy_sha256": policy_sha,
        "role_policy_sha256": sha256(root / ROLE_POLICY_REL),
        "daily_index_allowed_form_rows": int(len(candidates)),
        "events": int(len(output)),
        "unique_tickers": int(output.ticker.nunique()) if len(output) else 0,
        "submission_failures": submission_failures, "text_failures": text_failures,
        "exact_accession_body_required": True,
        "outcome_columns_loaded": False, "price_columns_loaded": False,
        "central_role_table_written": False, "central_queue_written": False,
        "central_epoch_written": False, "credentials_logged": False,
        "output_path": str(output_path.relative_to(root)).replace("\\", "/"),
        "output_sha256": sha256(output_path), "completed_at_utc": collected_at,
    }
    atomic_json(status, status_path)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="America/New_York filing date, YYYY-MM-DD")
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = collect_day(args.date, args.workspace_root.resolve())
    except Exception as error:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
