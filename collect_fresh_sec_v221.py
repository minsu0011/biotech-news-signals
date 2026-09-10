"""Collect never-seen recent regular-session SEC events for the V221 epoch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import bio_news_30m_v3 as app
import prepare_real_data as prepare
import v36_exact_pipeline as exact
from audit_v36_static import biomedical_name_evidence


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRICE = ROOT / "price_data" / "US"
UNIVERSE_PATH = DATA / "universe_v221_fresh_sec.csv"
STATUS_PATH = DATA / "V221_FRESH_SEC_COLLECTION_STATUS.json"


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


def build_universe(expanded: bool = False) -> tuple[pd.DataFrame, Path]:
    collector = app.SECCollector(app.Config())
    raw = collector.get_json(collector.TICKER_MAP)
    ticker_rows = {
        str(item.get("ticker", "")).upper().replace(".", "-").strip(): item
        for item in raw.values()
    }
    metadata_payload = json.loads((DATA / "V36_SEC_ISSUER_METADATA.json").read_text(encoding="utf-8"))
    metadata = {
        str(int(row["cik"])): row
        for row in metadata_payload.get("issuers", [])
        if str(row.get("cik", "")).isdigit()
    }
    seed = app.load_seed_universe()
    approved = set(seed.loc[seed.market.eq("US"), "ticker"].astype(str))
    rows: list[dict[str, Any]] = []
    candidate_tickers = (
        sorted(ticker_rows)
        if expanded
        else [path.stem.upper() for path in sorted(PRICE.glob("*.parquet"))]
    )
    for ticker in candidate_tickers:
        item = ticker_rows.get(ticker)
        if not item:
            continue
        # Current-provider expansion admits primary common symbols only.
        # Suffix warrants/rights remain excluded from the fresh epoch universe.
        if expanded and (
            not ticker.replace("-", "").isalnum()
            or len(ticker) > 5
            or ticker.endswith(("W", "P", "R", "U"))
        ):
            continue
        cik = str(int(item.get("cik_str")))
        company = str(item.get("title", "")).strip()
        issuer = metadata.get(cik, {})
        # SIC metadata is complete only for the legacy local-price universe;
        # current expansion therefore relies on the same audited biomedical
        # company-name rule (plus the frozen seed), never on outcomes.
        health = ticker in approved or biomedical_name_evidence(company)["name_rule_pass"]
        if not expanded:
            health = health or str(issuer.get("sic", "") or "") in exact.APPROVED_HEALTHCARE_SIC
        if health:
            rows.append({"market": "US", "ticker": ticker, "company": company, "cik": cik})
    universe = pd.DataFrame(rows).drop_duplicates("ticker").sort_values("ticker")
    universe_path = (
        DATA / "universe_v221_fresh_sec_expanded.csv" if expanded else UNIVERSE_PATH
    )
    universe.to_csv(universe_path, index=False)
    return universe, universe_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--expanded-universe", action="store_true")
    parser.add_argument("--policy")
    args = parser.parse_args()
    policy_path = Path(args.policy).resolve() if args.policy else None
    policy: dict[str, Any] | None = None
    if policy_path is not None:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if (
            policy.get("frozen_before_collection_results") is not True
            or policy.get("selection_uses_labels") is not False
            or policy.get("selection_uses_prices") is not False
            or policy.get("exact_price_contract_unchanged") is not True
            or str(policy.get("start")) != args.start
            or str(policy.get("end")) != args.end
            or bool(policy.get("expanded_universe")) != bool(args.expanded_universe)
        ):
            raise RuntimeError("FRESH_SEC_COLLECTION_POLICY_GUARD_FAIL")
    universe, universe_path = build_universe(args.expanded_universe)
    flavor = "expanded_" if args.expanded_universe else ""
    output_name = f"events_sec_v221_{flavor}fresh_{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
    events = prepare.collect_sec(
        args.start,
        args.end,
        force=True,
        output_name=output_name,
        source_name="SEC_V221_FRESH",
        one_per_ticker_day=False,
        max_events_per_ticker=None,
        universe_path=universe_path,
        material_only=False,
        fetch_body=False,
    )
    output_path = DATA / output_name
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "start": args.start,
        "end": args.end,
        "expanded_universe": bool(args.expanded_universe),
        "policy_path": str(policy_path.relative_to(ROOT)) if policy_path else None,
        "policy_sha256": sha256(policy_path) if policy_path else None,
        "universe_rows": int(len(universe)),
        "universe_path": str(universe_path.relative_to(ROOT)),
        "universe_sha256": sha256(universe_path),
        "events": int(len(events)),
        "unique_tickers": int(events.ticker.astype(str).nunique()),
        "output_path": str(output_path.relative_to(ROOT)),
        "output_sha256": sha256(output_path),
    }
    atomic_json(report, STATUS_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
