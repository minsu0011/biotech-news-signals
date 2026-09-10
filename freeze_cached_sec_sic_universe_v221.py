"""Freeze a label-blind SEC healthcare-CIK universe from cached submissions."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "v221_daily_sic" / "submissions"
BASE_PATH = DATA / "universe_v221_fresh_sec_expanded.csv"
OUTPUT_PATH = DATA / "universe_v221_cached_official_sic_expanded.csv"
REPORT_PATH = DATA / "V221_CACHED_OFFICIAL_SEC_SIC_UNIVERSE_FREEZE_STATUS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def primary_symbol(payload: dict[str, Any]) -> str:
    for value in payload.get("tickers", []) or []:
        symbol = str(value).upper().replace(".", "-").strip()
        if re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", symbol):
            return symbol
    return ""


def main() -> int:
    base = pd.read_csv(BASE_PATH, dtype=str).fillna("")
    base["universe_evidence"] = "PINNED_V221_EXPANDED_UNIVERSE"
    base["sic"] = ""
    additions: list[dict[str, str]] = []
    source_records: list[str] = []
    parsed = 0
    for path in sorted(CACHE.glob("CIK*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed += 1
        except Exception:
            continue
        cik_value = payload.get("cik")
        if not str(cik_value or "").isdigit():
            match = re.fullmatch(r"CIK(\d+)\.json", path.name)
            cik_value = match.group(1) if match else ""
        if not str(cik_value or "").isdigit():
            continue
        cik = str(int(cik_value))
        sic = str(payload.get("sic", "") or "")
        if sic not in exact.APPROVED_HEALTHCARE_SIC:
            continue
        digest = sha256(path)
        source_records.append(f"{path.name}|{digest}")
        additions.append({
            "market": "US",
            "ticker": primary_symbol(payload),
            "company": str(payload.get("name", "") or ""),
            "cik": cik,
            "universe_evidence": "SEC_SUBMISSIONS_APPROVED_HEALTHCARE_SIC",
            "sic": sic,
        })
    added = pd.DataFrame(additions)
    combined = pd.concat([base, added], ignore_index=True, sort=False)
    combined["cik"] = combined.cik.astype(str).map(lambda value: str(int(value)) if value.isdigit() else "")
    combined = combined.loc[combined.cik.ne("")].copy()
    combined["_priority"] = combined.universe_evidence.eq(
        "SEC_SUBMISSIONS_APPROVED_HEALTHCARE_SIC"
    ).astype(int)
    combined = (
        combined.sort_values(["cik", "_priority", "ticker"], ascending=[True, False, True])
        .drop_duplicates("cik", keep="first")
        .drop(columns="_priority")
        .sort_values(["cik", "ticker"])
    )
    atomic_csv(combined, OUTPUT_PATH)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "selection_uses_labels": False,
        "selection_uses_roles": False,
        "selection_uses_prices": False,
        "outcome_columns_loaded": False,
        "source": "cached official SEC submissions JSON",
        "source_cache_files_parsed": parsed,
        "approved_sic_source_ciks": len(added),
        "base_universe_ciks": int(base.cik.nunique()),
        "combined_unique_ciks": int(combined.cik.nunique()),
        "new_approved_sic_ciks_vs_base": int(len(set(added.cik) - set(base.cik))),
        "approved_sic_list": sorted(exact.APPROVED_HEALTHCARE_SIC),
        "source_snapshot_sha256": hashlib.sha256(
            "\n".join(sorted(source_records)).encode()
        ).hexdigest(),
        "base_path": str(BASE_PATH.relative_to(ROOT)),
        "base_sha256": sha256(BASE_PATH),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_PATH),
        "frozen_before_event_candidate_or_price_results": True,
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
