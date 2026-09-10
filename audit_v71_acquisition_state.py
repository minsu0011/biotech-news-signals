"""Credential-free, label-blind audit of the V71 acquisition queues.

This audit never instantiates a provider client, loads outcome columns, or
changes queue/role/cache state.  It is intended to make a long-run resume
decision from the files that actually exist now instead of relying on a
coverage report that may predate the latest queue rebuild.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRICE = ROOT / "price_data"
RESEARCH = ROOT / "research"

ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
US_QUEUE = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
KR_QUEUE = DATA / "PRICE_ACQUISITION_QUEUE_KR.parquet"
US_CACHE = PRICE / "US_V59_MASSIVE_1M"
KR_CACHE = PRICE / "KR_V59_KIS_1M"
TIMING_AUDIT = DATA / "V59_EXACT_TIMING_ELIGIBILITY.json"
READINESS_AUDIT = RESEARCH / "V70_NEW_DATA_EPOCH_READINESS.json"
LEDGER = RESEARCH / "DATA_ACQUISITION_LEDGER.csv"
OUTPUT = RESEARCH / "V71_ACQUISITION_STATE_AUDIT.json"

VALID_STATUSES = {"CACHED_VALID", "FETCHED_VALID"}
ROLE_ORDER = {
    "DEV_EXTENSION": 0,
    "RESEARCH_SEAL_POOL": 1,
    "FINAL_META_RESERVE": 2,
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def cache_path(base: Path, ticker: str, trade_date: str) -> Path:
    stamp = pd.Timestamp(trade_date)
    return base / f"{stamp.year:04d}" / str(ticker) / f"{stamp.date().isoformat()}.parquet"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def queue_audit(
    market: str,
    queue_path: Path,
    cache_root: Path,
    timezone_name: str,
    roles: pd.DataFrame,
) -> dict[str, Any]:
    queue = pd.read_parquet(queue_path)
    if queue["event_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{market}_QUEUE_DUPLICATE_EVENT_ID")
    queue["event_id"] = queue["event_id"].astype(str)
    merged = queue.merge(roles, on="event_id", how="left", validate="one_to_one")
    if merged["role"].isna().any():
        raise RuntimeError(f"{market}_QUEUE_EVENT_MISSING_FROZEN_ROLE")
    if merged["role_status"].ne("ACTIVE").any():
        raise RuntimeError(f"{market}_QUEUE_EVENT_ROLE_NOT_ACTIVE")

    local = pd.to_datetime(merged["event_time_local"], utc=True).dt.tz_convert(timezone_name)
    merged["trade_date"] = local.dt.date.astype(str)
    merged["cache_exists"] = [
        cache_path(cache_root, ticker, trade_date).exists()
        for ticker, trade_date in zip(merged["ticker"].astype(str), merged["trade_date"])
    ]

    status_counts = merged["status"].astype(str).value_counts().sort_index()
    role_status = (
        merged.groupby(["role", "status"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["role", "status"])
    )
    source_role_status = (
        merged.groupby(["source_family", "role", "status"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["source_family", "role", "status"])
    )
    required_source = "US_SEC" if market == "US" else "KR_NEWS"
    required_dev = merged[
        merged["source_family"].eq(required_source) & merged["role"].eq("DEV_EXTENSION")
    ]
    required_dev_status = required_dev["status"].astype(str).value_counts().sort_index()
    failure_class = merged.loc[merged["status"].eq("PROVIDER_FAILED"), "last_error"].fillna("").map(
        lambda value: "HTTP_403" if str(value).startswith("HTTP_403") else (str(value) or "EMPTY_ERROR")
    )
    deferred_rows = merged["status"].astype(str).str.startswith("DEFERRED_")

    group_rows: list[dict[str, Any]] = []
    for (ticker, trade_date), group in merged.groupby(["ticker", "trade_date"], sort=False):
        statuses = sorted(set(group["status"].astype(str)))
        sources = sorted(set(group["source_family"].astype(str)))
        roles_in_group = sorted(set(group["role"].astype(str)), key=lambda value: ROLE_ORDER.get(value, 99))
        group_rows.append(
            {
                "ticker": str(ticker),
                "trade_date": str(trade_date),
                "rows": int(len(group)),
                "status_set": "|".join(statuses),
                "all_valid": all(status in VALID_STATUSES for status in statuses),
                "has_pending": "PENDING" in statuses,
                "has_failed": "PROVIDER_FAILED" in statuses,
                "cache_exists": bool(group["cache_exists"].any()),
                "sources": sources,
                "roles": roles_in_group,
                "priority_role": roles_in_group[0],
            }
        )
    groups = pd.DataFrame(group_rows)
    combination_counts = groups["status_set"].value_counts().sort_index().to_dict()

    pending = groups[groups["has_pending"]].copy()
    if not pending.empty:
        pending["required_source"] = pending["sources"].map(lambda values: required_source in values)
        pending["dev"] = pending["roles"].map(lambda values: "DEV_EXTENSION" in values)
        pending["role_rank"] = pending["priority_role"].map(lambda value: ROLE_ORDER.get(value, 99))
        pending = pending.sort_values(
            ["required_source", "dev", "trade_date", "role_rank", "ticker"],
            ascending=[False, False, False, True, True],
        )
        priority_counts = (
            pending.groupby(["required_source", "dev"], observed=True)
            .size()
            .rename("ticker_dates")
            .reset_index()
            .sort_values(["required_source", "dev"], ascending=[False, False])
        )
        next_dates = records(pending.head(20)[["ticker", "trade_date", "sources", "roles"]])
    else:
        priority_counts = pd.DataFrame(columns=["required_source", "dev", "ticker_dates"])
        next_dates = []

    return {
        "queue_path": str(queue_path.relative_to(ROOT)),
        "queue_sha256": sha256(queue_path),
        "event_rows": int(len(merged)),
        "ticker_dates": int(len(groups)),
        "status_event_rows": {str(key): int(value) for key, value in status_counts.items()},
        "valid_event_rows": int(merged["status"].isin(VALID_STATUSES).sum()),
        "pending_event_rows": int(merged["status"].eq("PENDING").sum()),
        "failed_event_rows": int(merged["status"].eq("PROVIDER_FAILED").sum()),
        "deferred_event_rows": int(deferred_rows.sum()),
        "provider_failure_classes": {
            str(key): int(value) for key, value in failure_class.value_counts().sort_index().items()
        },
        "ticker_date_status_combinations": {str(key): int(value) for key, value in combination_counts.items()},
        "valid_ticker_dates": int(groups["all_valid"].sum()),
        "pending_ticker_dates": int(groups["has_pending"].sum()),
        "failed_ticker_dates": int(groups["has_failed"].sum()),
        "cache_backed_event_rows": int(merged["cache_exists"].sum()),
        "pending_event_rows_with_existing_cache": int(
            (merged["status"].eq("PENDING") & merged["cache_exists"]).sum()
        ),
        "pending_required_dev_rows_with_existing_cache": int(
            (
                merged["status"].eq("PENDING")
                & merged["cache_exists"]
                & merged["role"].eq("DEV_EXTENSION")
                & merged["source_family"].eq(required_source)
            ).sum()
        ),
        "role_binding": {
            "missing": 0,
            "non_active": 0,
            "labels_opened_true": int(merged["labels_opened"].fillna(False).sum()),
        },
        "by_role_status": records(role_status),
        "by_source_role_status": records(source_role_status),
        "required_source_dev": {
            "source_family": required_source,
            "event_rows": int(len(required_dev)),
            "ticker_dates": int(required_dev.groupby(["ticker", "trade_date"]).ngroups),
            "status_event_rows": {str(key): int(value) for key, value in required_dev_status.items()},
            "valid_event_rows": int(required_dev["status"].isin(VALID_STATUSES).sum()),
            "pending_event_rows": int(required_dev["status"].eq("PENDING").sum()),
            "failed_event_rows": int(required_dev["status"].eq("PROVIDER_FAILED").sum()),
        },
        "credential_free_priority_plan": {
            "policy": "REQUIRED_SOURCE_THEN_DEV_THEN_DATE_DESC_LABEL_BLIND_V1",
            "required_source": required_source,
            "ticker_date_counts": records(priority_counts),
            "next_20_ticker_dates": next_dates,
            "network_requests_executed": 0,
        },
    }


def ledger_audit() -> dict[str, Any]:
    if not LEDGER.exists():
        return {"exists": False}
    ledger = pd.read_csv(LEDGER)
    if ledger.empty:
        return {"exists": True, "rows": 0}
    summary = (
        ledger.groupby("provider", observed=True)
        .agg(
            ledger_rows=("provider", "size"),
            api_requests=("api_requests", "sum"),
            bars_downloaded=("bars_downloaded", "sum"),
            failed_ticker_dates=("failed", "sum"),
            cache_reuses=("cached", "sum"),
        )
        .reset_index()
    )
    return {
        "exists": True,
        "path": str(LEDGER.relative_to(ROOT)),
        "sha256": sha256(LEDGER),
        "rows": int(len(ledger)),
        "last_timestamp": str(ledger["timestamp"].max()),
        "by_provider": records(summary),
        "exact_eligible_column_sum": int(pd.to_numeric(ledger["exact_eligible"], errors="coerce").fillna(0).sum()),
        "exact_eligible_authority": str(TIMING_AUDIT.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    roles = pd.read_parquet(
        ROLE_PATH,
        columns=["event_id", "role", "role_status", "labels_opened"],
    )
    roles["event_id"] = roles["event_id"].astype(str)
    active_roles = roles[roles["role_status"].eq("ACTIVE")].copy()
    if active_roles["event_id"].duplicated().any():
        raise RuntimeError("DUPLICATE_EVENT_ID_IN_FROZEN_ROLE_ASSIGNMENT")

    timing = read_json(TIMING_AUDIT)
    readiness = read_json(READINESS_AUDIT)
    credential_presence = {
        name: "PRESENT" if os.getenv(name) else "MISSING"
        for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "KIS_APP_KEY", "KIS_APP_SECRET")
    }
    massive_available = credential_presence["MASSIVE_API_KEY"] == "PRESENT" or credential_presence["POLYGON_API_KEY"] == "PRESENT"
    kis_available = credential_presence["KIS_APP_KEY"] == "PRESENT" and credential_presence["KIS_APP_SECRET"] == "PRESENT"

    markets = {
        "US": queue_audit("US", US_QUEUE, US_CACHE, app.US_TZ, active_roles),
        "KR": queue_audit("KR", KR_QUEUE, KR_CACHE, app.KR_TZ, active_roles),
    }
    blockers: list[str] = []
    if not massive_available:
        blockers.append("MASSIVE_CREDENTIAL_NOT_VISIBLE_TO_PROCESS")
    if not kis_available:
        blockers.append("KIS_CREDENTIAL_PAIR_NOT_VISIBLE_TO_PROCESS")
    blockers.extend(str(value) for value in readiness.get("blocking_reasons", []))

    seal_or_final_opened = bool(readiness.get("seal_or_final_labels_opened"))
    report = {
        "schema_version": 1,
        "audited_at": now(),
        "audit_mode": "CREDENTIAL_FREE_LABEL_BLIND_READ_ONLY",
        "credentials": credential_presence,
        "provider_request_authorization": {
            "MASSIVE": massive_available,
            "KIS": kis_available,
            "requests_executed_by_this_audit": 0,
        },
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "seal_or_final_labels_opened": seal_or_final_opened,
        "role_assignment": {
            "path": str(ROLE_PATH.relative_to(ROOT)),
            "sha256": sha256(ROLE_PATH),
            "rows": int(len(roles)),
            "active_rows": int(len(active_roles)),
            "inactive_rows": int(len(roles) - len(active_roles)),
            "labels_opened_true": int(roles["labels_opened"].fillna(False).sum()),
        },
        "markets": markets,
        "exact_timing_authority": {
            "path": str(TIMING_AUDIT.relative_to(ROOT)),
            "sha256": sha256(TIMING_AUDIT),
            "eligible_total": int(timing.get("eligible_total", 0)),
            "eligible_by_role": timing.get("eligible_by_role", {}),
            "reason_counts": timing.get("reason_counts", {}),
        },
        "new_data_epoch_authority": {
            "path": str(READINESS_AUDIT.relative_to(ROOT)),
            "sha256": sha256(READINESS_AUDIT),
            "status": readiness.get("status"),
            "counts": readiness.get("counts", {}).get("observed_source_family_counts", {}),
            "required_support": {
                source: {
                    "observed": int(readiness.get("counts", {}).get("by_source_family", {}).get(source, {}).get("n", 0)),
                    "required": int(minimum),
                    "shortfall": max(
                        0,
                        int(minimum)
                        - int(readiness.get("counts", {}).get("by_source_family", {}).get(source, {}).get("n", 0)),
                    ),
                }
                for source, minimum in readiness.get("minimum_support", {}).get("required_source_families", {}).items()
            },
            "blocking_reasons": readiness.get("blocking_reasons", []),
        },
        "ledger": ledger_audit(),
        "blockers": list(dict.fromkeys(blockers)),
        "safe_resume": {
            "queue_preparation_ready": True,
            "network_fetch_ready": bool(massive_available or kis_available),
            "massive_fetch_ready": massive_available,
            "kis_fetch_ready": kis_available,
            "note": "Provider fetch must not be simulated when the relevant credential presence check fails.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "blockers": report["blockers"],
        "US": {
            "event_rows": markets["US"]["event_rows"],
            "pending": markets["US"]["pending_event_rows"],
            "valid": markets["US"]["valid_event_rows"],
            "failed": markets["US"]["failed_event_rows"],
        },
        "KR": {
            "event_rows": markets["KR"]["event_rows"],
            "pending": markets["KR"]["pending_event_rows"],
            "valid": markets["KR"]["valid_event_rows"],
            "failed": markets["KR"]["failed_event_rows"],
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
