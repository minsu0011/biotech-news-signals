"""Freeze the V221 exact DEV_EXTENSION epoch after every readiness gate passes."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import autonomous_v37plus as controller


ROOT = Path(__file__).resolve().parent
READINESS_PATH = ROOT / "research" / "V70_NEW_DATA_EPOCH_READINESS.json"
ACTIVATION_PATH = ROOT / "research" / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
AUTHORITY_PATH = ROOT / "state" / "CURRENT_DATA_AUTHORITY.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"MISSING_ACTIVATION_INPUT:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_ACTIVATION_INPUT:{path.name}")
    return value


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_contract(
    readiness: dict[str, Any],
    authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if readiness.get("status") not in {"READY_TO_FREEZE", "NEW_DATA_EPOCH_READY"}:
        raise RuntimeError("EPOCH_ACTIVATION_NOT_READY")
    if readiness.get("seal_or_final_labels_opened") is not False:
        raise RuntimeError("RESERVED_LABELS_ALREADY_OPENED")
    if readiness.get("blocking_reasons"):
        raise RuntimeError("EPOCH_ACTIVATION_HAS_BLOCKERS")
    checks = readiness.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise RuntimeError("EPOCH_ACTIVATION_CHECK_FAILED")
    if int(authority["counts"]["DEV_EXTENSION"]["US_SEC"]) < 120:
        raise RuntimeError("AUTHORITY_US_SEC_BELOW_120")
    invariant = authority.get("role_assignment_invariant", {})
    if not (
        invariant.get("baseline_roles_unchanged") is True
        and invariant.get("all_role_hashes_valid") is True
        and invariant.get("appended_roles_match_frozen_policy") is True
    ):
        raise RuntimeError("APPEND_ONLY_ROLE_AUTHORITY_INVALID")
    if not all(item.get("match") is True for item in authority.get("invariants", {}).values()):
        raise RuntimeError("IMMUTABLE_MODEL_OR_CONTRACT_AUTHORITY_INVALID")

    assets_before = controller.extension_asset_metadata()
    if set(assets_before) != set(controller.DATA_EPOCH_ASSETS):
        raise RuntimeError("EPOCH_ASSET_MISSING")
    for name, metadata in readiness.get("files", {}).items():
        if assets_before.get(name) != metadata:
            raise RuntimeError(f"READINESS_ASSET_BINDING_MISMATCH:{name}")

    counts, support, issues = controller.inspect_extension_support(
        controller.DATA_EPOCH_ASSETS["extension_csv"]
    )
    support_ok = (
        support.get("total_n_ge_240") is True
        and support.get("only_required_source_families") is True
        and set(support.get("by_source_family", {}))
        == set(controller.DATA_EPOCH_MINIMUM["required_source_families"])
        and all(
            all(source_checks.values())
            for source_checks in support.get("by_source_family", {}).values()
        )
    )
    if issues or not support_ok:
        raise RuntimeError("CONTROLLER_SUPPORT_AUDIT_FAILED")
    assets_after = controller.extension_asset_metadata()
    if assets_before != assets_after:
        raise RuntimeError("EPOCH_ASSET_CHANGED_DURING_FREEZE")

    prefreeze = {
        "family": controller.NEW_EXACT_DEV_EXTENSION_FAMILY,
        "counts": counts,
        "minimum_support": controller.DATA_EPOCH_MINIMUM,
        "data_contract": controller.DATA_EPOCH_EXACT_CONTRACT,
        "files": assets_after,
        "readiness_sha256": controller.sha256(READINESS_PATH),
    }
    freeze_digest = hashlib.sha256(controller.canonical(prefreeze)).hexdigest()
    freeze_id = f"V221-EPOCH-{freeze_digest[:24]}"
    epoch_payload = {
        "schema_version": 1,
        "family": controller.NEW_EXACT_DEV_EXTENSION_FAMILY,
        "freeze_id": freeze_id,
        "role": "DEV_EXTENSION",
        "counts": counts,
        "minimum_support": controller.DATA_EPOCH_MINIMUM,
        "data_contract": controller.DATA_EPOCH_EXACT_CONTRACT,
        "files": assets_after,
        "selection_uses_labels": False,
    }
    epoch_sha = controller.digest_bytes(controller.canonical(epoch_payload))
    frozen_at = datetime.now(timezone.utc).isoformat()
    contract = {
        **epoch_payload,
        "status": "FROZEN",
        "frozen": True,
        "frozen_at": frozen_at,
        "data_epoch_sha256": epoch_sha,
        "activation_path": str(ACTIVATION_PATH.relative_to(ROOT)).replace("/", "\\"),
        "activation_rule": (
            "All readiness, immutable-authority, append-only role, exact-data, "
            "and file-binding checks passed before this atomic freeze."
        ),
        "reserved_labels_opened": False,
        "credentials_serialized": False,
    }
    return contract, {"support": support, "epoch_payload": epoch_payload}


def main() -> int:
    readiness = read_json(READINESS_PATH)
    authority = read_json(AUTHORITY_PATH)
    if ACTIVATION_PATH.is_file() and readiness.get("activation_contract_written") is True:
        audit = controller.inspect_data_epoch_contract()
        if audit.get("ready") is not True:
            raise RuntimeError("EXISTING_EPOCH_ACTIVATION_INVALID")
        print(json.dumps({
            "status": "NEW_DATA_EPOCH_READY",
            "idempotent": True,
            "data_epoch_sha256": audit["data_epoch_sha256"],
        }, indent=2))
        return 0

    contract, _audit = build_contract(readiness, authority)
    atomic_json(contract, ACTIVATION_PATH)
    controller_audit = controller.inspect_data_epoch_contract()
    if controller_audit.get("ready") is not True:
        raise RuntimeError("POST_WRITE_EPOCH_CONTRACT_AUDIT_FAILED")

    activated = {
        **readiness,
        "status": "NEW_DATA_EPOCH_READY",
        "activation_contract_written": True,
        "activation_contract_path": str(ACTIVATION_PATH.relative_to(ROOT)).replace("/", "\\"),
        "activation_contract_sha256": controller.sha256(ACTIVATION_PATH),
        "data_epoch_sha256": contract["data_epoch_sha256"],
        "frozen_at_utc": contract["frozen_at"],
        "blocking_reasons": [],
    }
    atomic_json(activated, READINESS_PATH)
    print(json.dumps({
        "status": activated["status"],
        "US_SEC_DEV": int(activated["counts"]["by_source_family"]["US_SEC"]["n"]),
        "data_epoch_sha256": contract["data_epoch_sha256"],
        "activation_contract_sha256": controller.sha256(ACTIVATION_PATH),
        "seal_or_final_labels_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
