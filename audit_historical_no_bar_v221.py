"""Label-blind cause classification for high-confidence Alpaca no-bar rows."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
OUTPUT_PATH = DATA / "HISTORICAL_NO_BAR_CLASSIFICATION.parquet"
REPORT_PATH = RESEARCH / "HISTORICAL_NO_BAR_ANALYSIS.json"
TIINGO_PATH = DATA / "TIINGO_PREFLIGHT_STATUS.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def evidence_text(relative: str) -> str:
    path = ROOT / str(relative)
    if not path.is_file():
        return ""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="replace"))


def classify(row: pd.Series, text: str, cik_symbols: set[str]) -> tuple[str, str]:
    symbol = str(row.resolved_ticker).upper()
    form = str(row.form).upper()
    if form.startswith(("S-1", "F-1")):
        return "NOT_LISTED", "registration filing can precede actual first trade"
    if re.search(r"(?:WS|WT|W|U|R)$", symbol):
        return "SYMBOL_NOT_SUPPORTED", "warrant/unit/right share class"
    local = text.lower()
    if re.search(r"\botc(?:qb|qx|bb)?\b|pink\s+sheets?", local):
        return "OTC", "explicit OTC/Pink evidence in SEC filing"
    if symbol.endswith(("F", "Y")):
        return "OTC_OR_UNSUPPORTED", "foreign/ADR-style suffix; exchange support not established"
    if len(cik_symbols - {symbol}) > 0 or symbol != str(row.ticker).upper():
        return "CORPORATE_ACTION", "same-CIK symbol timeline or queue/resolved symbol differs"
    if re.search(r"\b(?:nasdaq|nyse|nyse\s+american)\b", local):
        return "PROVIDER_MISSING", "explicit listed-exchange evidence but Alpaca returned no usable bars"
    if "no trading symbol" in local:
        return "NOT_LISTED", "SEC no-trading-symbol evidence"
    return "UNKNOWN", "explicit identity exists but listing/trade/provider cause is not provable"


def main() -> int:
    queue = pd.read_parquet(QUEUE_PATH)
    forbidden = {"y", "fwd_ret_30m", "entry_price", "exit_price"}
    if forbidden & set(queue.columns):
        raise RuntimeError("NO_BAR_AUDIT_OUTCOME_COLUMN_GUARD_FAIL")
    subset = queue.loc[
        queue.fetch_status.astype(str).eq("ALPACA_NO_BAR")
        & queue.resolution_confidence.astype(str).isin({"A", "B"})
    ].copy()
    cik_timeline = {
        str(cik): set(group.resolved_ticker.astype(str).str.upper())
        for cik, group in queue.loc[queue.resolved_ticker.astype(str).ne("")].groupby("CIK")
    }
    rows: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        series = pd.Series(row._asdict())
        text = evidence_text(str(series.evidence_file))
        cause, rationale = classify(series, text, cik_timeline.get(str(series.CIK), set()))
        high_confidence_fallback_candidate = (
            str(series.role) == "DEV_EXTENSION"
            and str(series.source_family) == "US_SEC"
            and cause in {"PROVIDER_MISSING", "UNKNOWN", "CORPORATE_ACTION"}
            and str(series.resolution_asof_mode) == "EVENT_DATE_METADATA_ONLY"
        )
        rows.append({
            "event_id": str(series.event_id),
            "CIK": str(series.CIK),
            "event_date": str(series.event_date),
            "year": int(pd.Timestamp(series.event_time_utc).year),
            "queue_ticker": str(series.ticker),
            "resolved_ticker": str(series.resolved_ticker),
            "role": str(series.role),
            "source_family": str(series.source_family),
            "form": str(series.form),
            "resolution_method": str(series.resolution_method),
            "resolution_confidence": str(series.resolution_confidence),
            "resolution_asof_mode": str(series.resolution_asof_mode),
            "cause": cause,
            "rationale": rationale,
            "high_confidence_fallback_candidate": high_confidence_fallback_candidate,
            "selection_uses_labels": False,
        })
    frame = pd.DataFrame(rows)
    atomic_parquet(frame, OUTPUT_PATH)
    tiingo = json.loads(TIINGO_PATH.read_text(encoding="utf-8")) if TIINGO_PATH.is_file() else {}
    target = frame.loc[frame.high_confidence_fallback_candidate] if not frame.empty else frame
    report = {
        "schema_version": 1,
        "timestamp": now(),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "provider_requests_executed": 0,
        "rows": len(frame),
        "cause_counts": frame.cause.value_counts().to_dict() if not frame.empty else {},
        "by_role": frame.role.value_counts().to_dict() if not frame.empty else {},
        "high_confidence_fallback_candidates": len(target),
        "fallback_candidates_by_year": (
            target.year.value_counts().sort_index().to_dict() if not target.empty else {}
        ),
        "tiingo_status": tiingo.get("status", "UNKNOWN"),
        "tiingo_network_attempted": tiingo.get("network_attempted", False),
        "tiingo_backfill_authorized": tiingo.get("promotion", {}).get(
            "exact_contract_backfill_authorized", False
        ),
        "classification_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "classification_sha256": sha256(OUTPUT_PATH),
        "limitations": [
            "NO_BAR alone cannot distinguish no trades from provider omission.",
            "OTC/foreign suffix classification is identity/provider triage, not exact eligibility.",
            "No class is promoted without an independent parity-authorized provider fetch.",
        ],
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
