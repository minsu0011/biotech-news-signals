"""Consolidate the current V221+ data authority and supersession metadata."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
STATE = ROOT / "state"
AUTHORITY_PATH = STATE / "CURRENT_DATA_AUTHORITY.json"
SUPERSEDED_PATH = STATE / "SUPERSEDED_DATA_ARTIFACTS.json"
EXPECTED_HASHES = {
    "v36_contract": "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    "v69_selected_oof": "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
    "role_assignment_baseline": "00cbe558b11a3d198988425ebb2d2c98195ab00d508216c29cce1eb5dec71cc6",
}
ROLE_SEED = "MARKET_BIO_V59_ROLE_V1"
FILES = {
    "v36_contract": DATA / "dev_contract_v36_labeled.csv.gz",
    "v69_selected_oof": ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz",
    "role_assignment": DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet",
    "role_assignment_baseline": (
        ROOT / "reference_inputs" / "v221plus_20260829" / "data"
        / "V59_DATA_ROLE_ASSIGNMENT.parquet"
    ),
    "new_role_policy": DATA / "V70_NEW_ROLE_POLICY.json",
    "dev_extension": DATA / "V59_DEV_EXTENSION_LABELED.csv.gz",
    "dev_manifest": DATA / "V59_DEV_EXTENSION_MANIFEST.json",
    "timing": DATA / "V59_EXACT_TIMING_ELIGIBILITY.json",
    "readiness": RESEARCH / "V70_NEW_DATA_EPOCH_READINESS.json",
    "historical_queue": DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet",
    "historical_aliases": DATA / "HISTORICAL_TICKER_ALIAS_REGISTRY.parquet",
    "cik_symbol_timeline": DATA / "CIK_SYMBOL_TIMELINE.parquet",
    "historical_report": RESEARCH / "HISTORICAL_SYMBOL_RECOVERY_REPORT.json",
    "historical_ledger": RESEARCH / "HISTORICAL_RECOVERY_LEDGER.csv",
    "massive_audit": DATA / "MASSIVE_COVERAGE_AUDIT.json",
    "alpaca_preflight": DATA / "ALPACA_PREFLIGHT_STATUS.json",
    "alpaca_parity": RESEARCH / "PROVIDER_PARITY_SUMMARY.json",
    "tiingo_preflight": DATA / "TIINGO_PREFLIGHT_STATUS.json",
    "tiingo_plan": RESEARCH / "TIINGO_FALLBACK_PLAN.json",
    "champion": STATE / "CURRENT_CHAMPION.json",
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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("/", "\\"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().isoformat(),
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def role_count(timing: dict[str, Any], market: str, role: str) -> int:
    return int(timing["eligible_by_role"].get(f"{market}:{role}", 0))


def nested_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if len(columns) == 1:
        return {str(key): int(value) for key, value in frame[columns[0]].value_counts(dropna=False).items()}
    result: dict[str, Any] = {}
    for key, group in frame.groupby(columns[0], dropna=False):
        result[str(key)] = nested_counts(group, columns[1:])
    return result


def validate_append_only_roles(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Prove frozen historical roles are unchanged and new roles follow policy.

    The role-assignment parquet is intentionally append-only, so its complete
    file hash changes when legitimate new events arrive.  The immutable anchor
    is the preserved pre-fresh-data baseline; all baseline role fields must be
    byte-for-byte equivalent after scalar normalization, while every appended
    event must satisfy the predeclared deterministic role policy.
    """
    required = {"event_id", "role", "role_hash", "role_frozen_at", "labels_opened"}
    for name, frame in (("current", current), ("baseline", baseline)):
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise RuntimeError(f"ROLE_{name.upper()}_COLUMNS_MISSING")
        if frame.event_id.astype(str).duplicated().any():
            raise RuntimeError(f"ROLE_{name.upper()}_DUPLICATE_EVENT_ID")

    current_indexed = current.assign(event_id=current.event_id.astype(str)).set_index("event_id")
    baseline_indexed = baseline.assign(event_id=baseline.event_id.astype(str)).set_index("event_id")
    missing_ids = baseline_indexed.index.difference(current_indexed.index)
    if len(missing_ids):
        raise RuntimeError("FROZEN_ROLE_BASELINE_EVENT_REMOVED")

    frozen_columns = ["role", "role_hash", "role_frozen_at", "labels_opened"]
    baseline_ids = baseline_indexed.index
    changed = {
        column: int(
            (
                baseline_indexed.loc[baseline_ids, column].astype(str)
                != current_indexed.loc[baseline_ids, column].astype(str)
            ).sum()
        )
        for column in frozen_columns
    }
    if any(changed.values()):
        raise RuntimeError("FROZEN_ROLE_BASELINE_CHANGED")

    expected_hash = current_indexed.index.to_series().map(
        lambda value: hashlib.sha256(f"{ROLE_SEED}|{value}".encode()).hexdigest()
    )
    invalid_role_hashes = int((current_indexed.role_hash.astype(str) != expected_hash).sum())
    if invalid_role_hashes:
        raise RuntimeError("DETERMINISTIC_ROLE_HASH_MISMATCH")

    allocation = policy.get("new_row_allocation", {})
    dev_share = int(allocation.get("DEV_EXTENSION", -1))
    research_share = int(allocation.get("RESEARCH_SEAL_POOL", -1))
    final_share = int(allocation.get("FINAL_META_RESERVE", -1))
    if (
        policy.get("selection_uses_labels") is not False
        or policy.get("existing_roles_must_remain_unchanged") is not True
        or min(dev_share, research_share, final_share) < 0
        or dev_share + research_share + final_share != 100
    ):
        raise RuntimeError("INVALID_NEW_ROLE_POLICY_ARTIFACT")

    new_ids = current_indexed.index.difference(baseline_indexed.index)
    if len(new_ids):
        new_hashes = expected_hash.loc[new_ids]
        buckets = new_hashes.str[:8].map(lambda value: int(value, 16) % 100)
        expected_roles = buckets.map(
            lambda bucket: (
                "DEV_EXTENSION" if bucket < dev_share else
                "RESEARCH_SEAL_POOL" if bucket < dev_share + research_share else
                "FINAL_META_RESERVE"
            )
        )
        if not current_indexed.loc[new_ids, "role"].astype(str).equals(expected_roles.astype(str)):
            raise RuntimeError("APPENDED_ROLE_POLICY_MISMATCH")

    return {
        "mode": "IMMUTABLE_BASELINE_PLUS_DETERMINISTIC_APPEND_ONLY",
        "baseline_rows": int(len(baseline_indexed)),
        "current_rows": int(len(current_indexed)),
        "appended_rows": int(len(new_ids)),
        "baseline_events_missing": 0,
        "baseline_frozen_field_changes": changed,
        "baseline_roles_unchanged": True,
        "all_role_hashes_valid": True,
        "appended_roles_match_frozen_policy": True,
        "current_assignment_sha256": sha256(FILES["role_assignment"]),
        "role_seed_sha256": hashlib.sha256(ROLE_SEED.encode()).hexdigest(),
        "new_role_policy_sha256": sha256(FILES["new_role_policy"]),
    }


def main() -> int:
    missing = [name for name, path in FILES.items() if not path.is_file()]
    if missing:
        raise RuntimeError("MISSING_AUTHORITY_INPUTS:" + ",".join(missing))
    readiness = read_json(FILES["readiness"])
    timing = read_json(FILES["timing"])
    historical = read_json(FILES["historical_report"])
    massive = read_json(FILES["massive_audit"])
    alpaca = read_json(FILES["alpaca_preflight"])
    parity = read_json(FILES["alpaca_parity"])
    tiingo = read_json(FILES["tiingo_preflight"])
    tiingo_plan = read_json(FILES["tiingo_plan"])
    champion = read_json(FILES["champion"])
    roles = pd.read_parquet(FILES["role_assignment"])
    role_baseline = pd.read_parquet(FILES["role_assignment_baseline"])
    role_invariant = validate_append_only_roles(
        roles, role_baseline, read_json(FILES["new_role_policy"])
    )
    queue = pd.read_parquet(FILES["historical_queue"])
    dev = pd.read_csv(FILES["dev_extension"])
    dev["event_time_utc"] = pd.to_datetime(dev["event_time_utc"], utc=True, errors="raise")
    us_sec = dev.loc[dev.source_family.astype(str).eq("US_SEC")].copy()
    target_years = list(range(2018, 2025))
    year_counts = Counter(int(year) for year in us_sec.event_time_utc.dt.year)
    all_years = range(int(us_sec.event_time_utc.dt.year.min()), int(us_sec.event_time_utc.dt.year.max()) + 1)
    official_year_counts = {str(year): int(year_counts.get(year, 0)) for year in all_years}
    target_year_counts = {str(year): int(year_counts.get(year, 0)) for year in target_years}
    source_counts = readiness["counts"]["observed_source_family_counts"]
    us_sec_count = int(source_counts.get("US_SEC", 0))
    us_required = int(readiness["minimum_support"]["required_source_families"]["US_SEC"])
    observed_hashes = {name: sha256(FILES[name]) for name in EXPECTED_HASHES}
    invariants = {
        name: {
            "expected_sha256": EXPECTED_HASHES[name],
            "observed_sha256": observed_hashes[name],
            "match": observed_hashes[name] == EXPECTED_HASHES[name],
        }
        for name in EXPECTED_HASHES
    }
    if not all(item["match"] for item in invariants.values()):
        raise RuntimeError("IMMUTABLE_HASH_MISMATCH")
    total_us = sum(role_count(timing, "US", role) for role in (
        "DEV_EXTENSION", "RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"
    ))
    total_kr = sum(role_count(timing, "KR", role) for role in (
        "DEV_EXTENSION", "RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"
    ))
    authority = {
        "schema_version": 2,
        "timestamp": now(),
        "status": readiness["status"],
        "authority_precedence": "THIS_FILE_SUPERSEDES_OLDER_DATA_COUNT_AND_PROVIDER_STATUS_SUMMARIES",
        "source_of_truth": {
            "counts": str(FILES["readiness"].relative_to(ROOT)),
            "timing_eligibility": str(FILES["timing"].relative_to(ROOT)),
            "historical_resolution": str(FILES["historical_report"].relative_to(ROOT)),
            "supersession_registry": str(SUPERSEDED_PATH.relative_to(ROOT)),
        },
        "counts": {
            "exact_total": int(timing["eligible_total"]),
            "US_exact": total_us,
            "KR_exact": total_kr,
            "DEV_EXTENSION": {
                "US_exact_all_source_families": role_count(timing, "US", "DEV_EXTENSION"),
                "KR_exact_all_source_families": role_count(timing, "KR", "DEV_EXTENSION"),
                "US_SEC": us_sec_count,
                "US_SEC_required": us_required,
                "US_SEC_shortfall": max(0, us_required - us_sec_count),
                "KR_NEWS": int(source_counts.get("KR_NEWS", 0)),
            },
            "RESEARCH_SEAL_POOL": {
                "US": role_count(timing, "US", "RESEARCH_SEAL_POOL"),
                "KR": role_count(timing, "KR", "RESEARCH_SEAL_POOL"),
                "labels_opened": False,
            },
            "FINAL_META_RESERVE": {
                "US": role_count(timing, "US", "FINAL_META_RESERVE"),
                "KR": role_count(timing, "KR", "FINAL_META_RESERVE"),
                "US_required": 250,
                "US_ready_by_count": role_count(timing, "US", "FINAL_META_RESERVE") >= 250,
                "labels_opened": False,
            },
        },
        "current_historical_coverage": {
            "official_US_SEC_DEV_by_year": official_year_counts,
            "official_US_SEC_DEV_2018_2024": target_year_counts,
            "official_US_SEC_DEV_2018_2024_total": sum(target_year_counts.values()),
            "resolver_queue_2018_2024": historical.get("year_distribution", []),
            "resolver_exact_by_role": historical.get("exact_by_role", {}),
            "historical_ticker_changed_count": int(historical.get("historical_ticker_changed_count", 0)),
            "new_ticker_resolved": int(historical.get("new_ticker_resolved", 0)),
        },
        "latest_queue_status": {
            "rows": len(queue),
            "resolution_status_counts": nested_counts(queue, ["resolution_status"]),
            "priority_resolution_status_counts": nested_counts(queue, ["priority", "resolution_status"]),
            "exact_eligible": int(queue.exact_eligible.astype(bool).sum()),
            "label_columns_absent": not bool({"y", "fwd_ret_30m", "entry_price", "exit_price"} & set(queue.columns)),
        },
        "provider_statuses": {
            "priority_policy": ["P1_MASSIVE_SIP", "P2_ALPACA_SIP", "P3_TIINGO_IEX_PARITY_ELIGIBLE_ONLY"],
            "MASSIVE": {
                "history_entitlement": massive.get("history_entitlement"),
                "oldest_successful_anchor": massive.get("oldest_successful_anchor"),
                "exact_contract_eligible": massive.get("exact_contract_eligible"),
            },
            "ALPACA": {
                "status": alpaca.get("status"),
                "historical_sip_usable": alpaca.get("historical_sip_usable"),
                "parity_status": parity.get("result", {}).get("status"),
                "parity_backfill_authorized": parity.get("backfill_authorized"),
            },
            "TIINGO": {
                "status": tiingo.get("status"),
                "credential_present": tiingo.get("credential_present"),
                "network_attempted": tiingo.get("network_attempted"),
                "parity_executed": tiingo_plan.get("parity", {}).get("executed"),
                "backfill_authorized": tiingo_plan.get("backfill_authorized"),
            },
            "synthetic_or_interpolated_prices_used": False,
        },
        "new_data_epoch": {
            "status": readiness["status"],
            "ready": readiness["status"] in {"READY", "READY_TO_FREEZE", "NEW_DATA_EPOCH_READY"},
            "blocking_reasons": readiness.get("blocking_reasons", []),
            "activation_contract_written": readiness.get("activation_contract_written", False),
        },
        "model_state": {
            "champion": f"V{champion['version']}",
            "hypothesis": champion.get("hypothesis"),
            "model_restart_executed": False,
            "zero_tune_executed": False,
            "reason": "NEW_DATA_EPOCH_NOT_READY" if readiness["status"] == "NOT_READY" else "AWAITING_EXPLICIT_PHASE_TRANSITION",
            "FINAL_PASS": False,
        },
        "protections": {
            "selection_uses_labels": False,
            "historical_resolution_uses_labels": historical.get("selection_uses_labels", False),
            "seal_or_final_labels_opened": readiness.get("seal_or_final_labels_opened", False),
            "exact_contract_relaxed": False,
            "roles_reassigned": False,
            "credentials_serialized": False,
        },
        "invariants": invariants,
        "role_assignment_invariant": role_invariant,
        "artifacts": {name: binding(path) for name, path in FILES.items()},
    }
    atomic_json(authority, AUTHORITY_PATH)
    stale = [
        {
            "path": "research/V221_ALPACA_CONTINUATION_STATUS.json",
            "stale_claims": ["US_SEC DEV=78", "Final Meta US=245"],
            "reason": "Historical symbol recovery and resolved-ticker Alpaca rebuild increased official counts.",
        },
        {
            "path": "research/V221_US_SEC_DEV_REACHABILITY_AUDIT.json",
            "stale_claims": ["US_SEC exact=43", "strict upper bound=110 as a future-data reachability estimate"],
            "reason": "The old estimate predates CIK/event-date symbol recovery; 110 is now an observed official count, not an upper-bound authority.",
        },
        {
            "path": "research/V221PLUS_RESEARCH_STATUS.json",
            "stale_claims": ["US_SEC DEV=43"],
            "reason": "Research findings remain historical diagnostics, but its data-count snapshot is stale.",
        },
        {
            "path": "research/V221_POST_TRIAL_ACQUISITION_AUDIT.json",
            "stale_claims": ["pre-Alpaca provider queue counts", "pre-historical-symbol readiness counts"],
            "reason": "Later Massive/Alpaca and historical-symbol artifacts are authoritative.",
        },
    ]
    superseded = {
        "schema_version": 1,
        "timestamp": now(),
        "policy": "FILES_ARE_PRESERVED_FOR_AUDIT; ONLY_THE_LISTED_SNAPSHOT_CLAIMS_ARE_SUPERSEDED",
        "SUPERSEDED_BY": str(AUTHORITY_PATH.relative_to(ROOT)).replace("/", "\\"),
        "current_authority_sha256": sha256(AUTHORITY_PATH),
        "artifacts": [],
    }
    for item in stale:
        path = ROOT / item["path"]
        if path.is_file():
            superseded["artifacts"].append({
                **item,
                "SUPERSEDED_BY": str(AUTHORITY_PATH.relative_to(ROOT)).replace("/", "\\"),
                "preserved": True,
                "file_sha256": sha256(path),
            })
    atomic_json(superseded, SUPERSEDED_PATH)
    print(json.dumps({
        "status": authority["status"],
        "US_SEC_DEV": us_sec_count,
        "US_SEC_shortfall": max(0, us_required - us_sec_count),
        "KR_NEWS_DEV": int(source_counts.get("KR_NEWS", 0)),
        "Final_Meta_US": role_count(timing, "US", "FINAL_META_RESERVE"),
        "Tiingo": tiingo.get("status"),
        "model_restart_executed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
