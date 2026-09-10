"""V222: frozen-policy V64 causal-adapter transfer on the V221 data epoch.

This is a material new-data diagnostic, not V69 zero-tune and not a champion.
It reuses the already-declared V64 adapter architecture and grids, reads only
V36 DEV plus the frozen DEV_EXTENSION, and never opens reserved roles.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import experiment_v64_new_data_retrain as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_V222"
EPOCH_PATH = ROOT / "research" / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
EXPECTED_EXTENSION_SHA256 = "e11c60ef38c5b65e676bab79c96cd21e44fdcb6d604e02a550ffc49003ed311b"
EXPECTED_EXTENSION_ROWS = 430


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


def atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)
    os.replace(temporary, path)


def main() -> int:
    epoch = json.loads(EPOCH_PATH.read_text(encoding="utf-8"))
    if epoch.get("status") != "FROZEN" or epoch.get("reserved_labels_opened") is not False:
        raise RuntimeError("V222_FROZEN_EPOCH_GUARD_FAIL")
    if sha256(base.EXTENSION_PATH) != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError("V222_EXTENSION_HASH_MISMATCH")
    base.EXPECTED_SHA256[base.EXTENSION_PATH.name] = EXPECTED_EXTENSION_SHA256
    base.EXPECTED_EXTENSION_ROWS = EXPECTED_EXTENSION_ROWS
    base.EXTENSION_CLUSTER_POLICY = "LABEL_BLIND_FIRST_EVENT_PER_CLUSTER"
    base.ALLOW_UNSCORABLE_PREFIX_QUARANTINE = True
    payload = base.execute(smoke=False)
    selected = payload["robustness"]["selected"]
    extension = payload["robustness"]["extension_diagnostic"]
    report = {
        "version": 222,
        "hypothesis": "CURRENT_EPOCH_FROZEN_V64_CAUSAL_ADAPTER_TRANSFER",
        "status": "RESEARCH_FAIL" if payload["status"] != "ROBUST_SURVIVOR" else "ROBUST_SURVIVOR",
        "material_change": "frozen V221 exact DEV_EXTENSION data",
        "evaluation_type": "DATA_ONLY_RETRAIN_HISTORICAL_ARCHITECTURE_TRANSFER",
        "is_v69_locked_zero_tune": False,
        "v69_champion_changed": False,
        "data_epoch_sha256": epoch.get("data_epoch_sha256"),
        "extension_sha256": EXPECTED_EXTENSION_SHA256,
        "extension_input_rows": EXPECTED_EXTENSION_ROWS,
        "predicted_extension_rows": int(len(payload["extension_oof"])),
        "combined_metrics": selected["metrics"],
        "canonical_research_gate": selected["research_gate"],
        "extension_metrics": extension["metrics"],
        "extension_by_market_source": extension["by_market_source_even_when_small"],
        "fold_audits": payload["comparison"]["fold_audits"],
        "source_audit": payload["source_audit"],
        "predeclared_architecture": payload["comparison"]["configuration"],
        "caveats": [
            "V69 has no pre-epoch deployable scorer, so this is not zero-tune.",
            "Two repeated text-cluster rows were label-blind quarantined.",
            "The earliest KR_NEWS block lacks causal past training history and was quarantined.",
            "No Research Seal or Final Meta row was read.",
        ],
        "seal_authorized": False,
        "final_meta_authorized": False,
        "research_seal_or_final_rows_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(report, OUT / "V222_CAUSAL_ADAPTER_TRANSFER_REPORT.json")
    atomic_csv_gz(payload["extension_oof"], OUT / "V222_EXTENSION_PREQUENTIAL_PREDICTIONS.csv.gz")
    run_status = {
        "version": 222,
        "status": report["status"],
        "phase": "NEW_DATA_RESEARCH",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seal_state": "UNOPENED",
        "final_status": "CONTINUE",
    }
    atomic_json(run_status, OUT / "RUN_STATUS.json")
    files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}
    }
    manifest = {
        "version": 222,
        "hypothesis": report["hypothesis"],
        "files": files,
        "research_seal_or_final_rows_read": False,
    }
    atomic_json(manifest, OUT / "ARTIFACT_MANIFEST.json")
    commit = {
        "version": 222,
        "hypothesis": report["hypothesis"],
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
        "data_epoch_sha256": epoch.get("data_epoch_sha256"),
        "extension_sha256": EXPECTED_EXTENSION_SHA256,
        "runner_sha256": sha256(Path(__file__)),
    }
    atomic_json(commit, OUT / "COMMIT.json")
    print(json.dumps({
        "version": 222,
        "status": report["status"],
        "predicted_extension_rows": report["predicted_extension_rows"],
        "canonical_gate": report["canonical_research_gate"],
        "extension_metrics": report["extension_metrics"],
        "commit_sha256": sha256(OUT / "COMMIT.json"),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
