"""Record whether a genuine pre-epoch deployable V69 scorer is available."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "V69_NEW_DATA_ZERO_TUNE.json"
V69 = ROOT / "output_V69"
EPOCH = ROOT / "research" / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
PREDICTION_MANIFEST = ROOT / "research" / "zero_tune_inputs" / "V221" / "manifest.json"
SERIALIZED_SUFFIXES = {
    ".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth", ".safetensors",
    ".ubj", ".cbm", ".model",
}


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    epoch = json.loads(EPOCH.read_text(encoding="utf-8"))
    manifest_path = V69 / "ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(str(name) for name in manifest.get("files", {}))
    serialized = sorted(
        name for name in files if Path(name).suffix.lower() in SERIALIZED_SUFFIXES
    )
    scorer_named = sorted(
        name for name in files
        if any(token in name.lower() for token in ("scorer", "estimator", "booster", "weights", "coefficients"))
    )
    bundle_present = PREDICTION_MANIFEST.is_file()
    deployable = bool(serialized or scorer_named) and bundle_present
    payload = {
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": (
            "READY_FOR_LOCKED_ZERO_TUNE"
            if deployable else "BLOCKED_FAIL_CLOSED_NO_PRE_EPOCH_DEPLOYABLE_SCORER"
        ),
        "data_epoch_status": epoch.get("status"),
        "data_epoch_sha256": epoch.get("data_epoch_sha256"),
        "v69_commit_sha256": sha256(V69 / "COMMIT.json"),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_manifest_files": files,
        "serialized_scorer_candidates": serialized,
        "scorer_named_candidates": scorer_named,
        "frozen_prediction_bundle_present": bundle_present,
        "zero_tune_predictions_created": False,
        "zero_tune_metrics_computed": False,
        "fit_executed_for_epoch": False,
        "retrained_to_imitate_v69": False,
        "threshold_changed": False,
        "calibration_changed": False,
        "polarity_changed": False,
        "seal_or_final_rows_read": False,
        "reason": (
            "V69 stores outer-OOF evidence and reports, but no deployable fitted scorer. "
            "Training after the epoch freeze cannot be represented as locked zero-tune."
        ),
        "predictions_artifact": None,
        "required_unlock": (
            "A scorer serialized before this epoch, bound to the pinned V69 commit, plus a "
            "label-blind prediction manifest matching the frozen epoch."
        ),
        "productive_continuation": [
            "predeclared causal data-only retrain/adaptation on V36 DEV plus frozen DEV_EXTENSION",
            "new-information research without opening Research Seal or Final Meta",
        ],
    }
    atomic_json(payload, OUTPUT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if deployable else 2


if __name__ == "__main__":
    raise SystemExit(main())
