"""Synchronize stale state projections to the current V221 data authority."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
RESEARCH = ROOT / "research"
DATA = ROOT / "data"
AUTHORITY_PATH = STATE / "CURRENT_DATA_AUTHORITY.json"
EPOCH_PATH = STATE / "CURRENT_DATA_EPOCH.json"
ACQUISITION_PATH = STATE / "ACQUISITION_STATE.json"
AUTONOMOUS_PATH = STATE / "AUTONOMOUS_V71PLUS_STATE.json"
MILESTONES_PATH = STATE / "V221_FINAL_MILESTONES.json"
SUPERSEDED_PATH = STATE / "SUPERSEDED_DATA_ARTIFACTS.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    authority = read_json(AUTHORITY_PATH)
    readiness_path = RESEARCH / "V70_NEW_DATA_EPOCH_READINESS.json"
    timing_path = DATA / "V59_EXACT_TIMING_ELIGIBILITY.json"
    historical_path = RESEARCH / "HISTORICAL_SYMBOL_RECOVERY_REPORT.json"
    readiness = read_json(readiness_path)
    timing = read_json(timing_path)
    historical = read_json(historical_path)
    if not authority or not readiness or not timing or not historical:
        raise RuntimeError("STATE_SYNC_AUTHORITY_INPUT_MISSING")
    authority_us = int(authority["counts"]["DEV_EXTENSION"]["US_SEC"])
    readiness_us = int(readiness["counts"]["observed_source_family_counts"]["US_SEC"])
    if authority_us != readiness_us:
        raise RuntimeError("STATE_SYNC_AUTHORITY_COUNT_MISMATCH")

    old_epoch = read_json(EPOCH_PATH)
    old_acquisition = read_json(ACQUISITION_PATH)
    old_autonomous = read_json(AUTONOMOUS_PATH)
    old_epoch_us = int(old_epoch.get("counts", {}).get("observed_source_family_counts", {}).get("US_SEC", -1))
    synced_at = now()
    backup_entries: list[dict[str, Any]] = []
    if old_epoch_us != authority_us:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = STATE / "superseded" / f"PRE_FINAL_DATA_GAP_SYNC_{stamp}"
        for name, payload, source_path in (
            ("CURRENT_DATA_EPOCH.json", old_epoch, EPOCH_PATH),
            ("ACQUISITION_STATE.json", old_acquisition, ACQUISITION_PATH),
            ("AUTONOMOUS_V71PLUS_STATE.json", old_autonomous, AUTONOMOUS_PATH),
        ):
            backup = backup_root / name
            atomic_json({
                **payload,
                "SUPERSEDED_BY": str(AUTHORITY_PATH.relative_to(ROOT)),
                "superseded_at": synced_at,
                "supersession_reason": "State projection predated historical symbol recovery and official 110-row rebuild.",
            }, backup)
            backup_entries.append({
                "path": str(source_path.relative_to(ROOT)).replace("/", "\\"),
                "preserved_snapshot": str(backup.relative_to(ROOT)).replace("/", "\\"),
                "stale_claims": [f"US_SEC DEV={old_epoch_us}" if old_epoch_us >= 0 else "stale acquisition projection"],
                "SUPERSEDED_BY": str(AUTHORITY_PATH.relative_to(ROOT)).replace("/", "\\"),
                "preserved": True,
                "file_sha256": sha256(backup),
            })

    epoch = {
        "schema_version": readiness.get("schema_version", 1),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": readiness["status"],
        "activation_contract_written": bool(readiness.get("activation_contract_written", False)),
        "seal_or_final_labels_opened": bool(readiness.get("seal_or_final_labels_opened", False)),
        "counts": readiness["counts"],
        "checks": readiness["checks"],
        "blocking_reasons": readiness.get("blocking_reasons", []),
        "data_epoch_sha256": readiness.get("data_epoch_sha256"),
        "activation_contract_path": readiness.get("activation_contract_path"),
        "activation_contract_sha256": readiness.get("activation_contract_sha256"),
        "source": str(readiness_path.relative_to(ROOT)).replace("/", "\\"),
        "current_data_authority": str(AUTHORITY_PATH.relative_to(ROOT)).replace("/", "\\"),
        "current_data_authority_sha256": sha256(AUTHORITY_PATH),
        "readiness_sha256": sha256(readiness_path),
    }
    atomic_json(epoch, EPOCH_PATH)

    acquisition = {
        **old_acquisition,
        "timestamp": synced_at,
        "phase": "FINAL_DATA_GAP_CLOSURE",
        "last_progress_at": synced_at,
        "current_data_authority": {
            "path": str(AUTHORITY_PATH.relative_to(ROOT)).replace("/", "\\"),
            "sha256": sha256(AUTHORITY_PATH),
        },
        "new_data_epoch": {
            "status": readiness["status"],
            "US_SEC_DEV": authority_us,
            "US_SEC_required": int(authority["counts"]["DEV_EXTENSION"]["US_SEC_required"]),
            "US_SEC_shortfall": int(authority["counts"]["DEV_EXTENSION"]["US_SEC_shortfall"]),
            "KR_NEWS_DEV": int(authority["counts"]["DEV_EXTENSION"]["KR_NEWS"]),
            "blocking_reasons": readiness.get("blocking_reasons", []),
        },
        "exact_eligibility": {
            "eligible_total": int(timing["eligible_total"]),
            "eligible_by_role": timing["eligible_by_role"],
            "sha256": sha256(timing_path),
        },
        "historical_recovery": {
            "queue_rows": int(historical["queue_rows"]),
            "new_ticker_resolved": int(historical["new_ticker_resolved"]),
            "new_exact_rows": int(historical["new_exact_rows"]),
            "queue_sha256": historical["queue_sha256"],
        },
        "provider_statuses": authority["provider_statuses"],
        "selection_uses_labels": False,
        "seal_or_final_labels_opened": False,
    }
    atomic_json(acquisition, ACQUISITION_PATH)

    if old_autonomous:
        old_autonomous.update({
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "FINAL_DATA_GAP_CLOSURE",
            "data_epoch": epoch,
            "active_children": [],
        })
        atomic_json(old_autonomous, AUTONOMOUS_PATH)

    old_milestones = read_json(MILESTONES_PATH)
    milestone_names = (
        "US_SEC_DEV_115", "US_SEC_DEV_119", "US_SEC_DEV_120", "NEW_DATA_EPOCH_READY",
        "V69_ZERO_TUNE_COMPLETE", "HISTORICAL_CHALLENGER_TRANSFER_COMPLETE",
        "POLARITY_TRANSFER_COMPLETE", "DATA_ONLY_RETRAIN_COMPLETE",
        "FIRST_10_OF_14", "FIRST_11_OF_14", "FIRST_12_OF_14", "FIRST_13_OF_14",
        "FIRST_14_OF_14", "RESEARCH_SEAL_PASS", "FINAL_META_PASS",
    )
    milestone_values = old_milestones.get("milestones", {})
    for name in milestone_names:
        milestone_values.setdefault(name, {"achieved": False, "achieved_at": None})
    milestone_values["US_SEC_DEV_115"]["achieved"] = authority_us >= 115
    milestone_values["US_SEC_DEV_119"]["achieved"] = authority_us >= 119
    if authority_us >= 119 and not milestone_values["US_SEC_DEV_119"].get("achieved_at"):
        milestone_values["US_SEC_DEV_119"]["achieved_at"] = synced_at
    milestone_values["US_SEC_DEV_120"]["achieved"] = authority_us >= 120
    milestone_values["NEW_DATA_EPOCH_READY"]["achieved"] = readiness["status"] in {
        "READY", "READY_TO_FREEZE", "NEW_DATA_EPOCH_READY"
    }
    for name in ("US_SEC_DEV_115", "US_SEC_DEV_120", "NEW_DATA_EPOCH_READY"):
        if milestone_values[name]["achieved"] and not milestone_values[name].get("achieved_at"):
            milestone_values[name]["achieved_at"] = synced_at
    atomic_json({
        "schema_version": 1,
        "updated_at": synced_at,
        "US_SEC_DEV": authority_us,
        "milestones": milestone_values,
    }, MILESTONES_PATH)

    if backup_entries:
        superseded = read_json(SUPERSEDED_PATH)
        superseded.setdefault("artifacts", []).extend(backup_entries)
        superseded["timestamp"] = synced_at
        superseded["current_authority_sha256"] = sha256(AUTHORITY_PATH)
        atomic_json(superseded, SUPERSEDED_PATH)

    print(json.dumps({
        "status": readiness["status"],
        "US_SEC_DEV": authority_us,
        "previous_epoch_US_SEC_DEV": old_epoch_us,
        "state_files_synchronized": [
            str(EPOCH_PATH.relative_to(ROOT)),
            str(ACQUISITION_PATH.relative_to(ROOT)),
            str(AUTONOMOUS_PATH.relative_to(ROOT)),
            str(MILESTONES_PATH.relative_to(ROOT)),
        ],
        "superseded_snapshots_written": len(backup_entries),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
