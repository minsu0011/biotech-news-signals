"""Fail-closed readiness audit for the V70 exact DEV_EXTENSION data epoch."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
TEMPLATE = RESEARCH / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.template.json"
DEV = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"
MANIFEST = DATA / "V59_DEV_EXTENSION_MANIFEST.json"
ROLE_JSON = DATA / "V59_DATA_ROLE_ASSIGNMENT.json"
ROLE_PARQUET = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
OUTPUT = RESEARCH / "V70_NEW_DATA_EPOCH_READINESS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def chronological_blocks(frame: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    ordered = frame.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
    blocks = []
    for index, positions in enumerate(np.array_split(np.arange(len(ordered)), count), 1):
        block = ordered.iloc[positions]
        blocks.append({
            "block": index,
            "n": len(block),
            "y_0": int(block.y.eq(0).sum()),
            "y_1": int(block.y.eq(1).sum()),
            "start_utc": None if block.empty else str(block.event_time_utc.iloc[0]),
            "end_utc": None if block.empty else str(block.event_time_utc.iloc[-1]),
        })
    return blocks


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    minimum = template["minimum_support"]
    required = minimum["required_source_families"]
    per_source = minimum["per_source"]
    block_rule = minimum["chronological_blocks"]
    frame = pd.read_csv(DEV)
    if not frame.event_id.is_unique:
        raise RuntimeError("DUPLICATE_DEV_EVENT_ID")
    if set(frame.data_role.astype(str)) != {"DEV_EXTENSION"}:
        raise RuntimeError("NON_DEV_ROLE_ENTERED_READINESS_AUDIT")
    if not frame.y.isin([0, 1]).all():
        raise RuntimeError("NON_BINARY_DEV_LABEL")
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)

    observed_sources = sorted(set(frame.source_family.astype(str)))
    unexpected_sources = sorted(set(observed_sources) - set(required))
    checks: dict[str, bool] = {
        "total_n": len(frame) >= int(minimum["total"]),
        "only_required_source_families": not unexpected_sources,
    }
    by_source: dict[str, Any] = {}
    for source, required_n in required.items():
        group = frame[frame.source_family.astype(str).eq(source)].copy()
        blocks = chronological_blocks(group, int(block_rule["count"]))
        stats = {
            "n": len(group),
            "y_0": int(group.y.eq(0).sum()),
            "y_1": int(group.y.eq(1).sum()),
            "unique_dates": int(group.event_time_utc.dt.date.nunique()),
            "unique_tickers": int(group.ticker.astype(str).nunique()),
            "chronological_blocks": blocks,
        }
        source_checks = {
            "n": stats["n"] >= int(required_n),
            "y_0": stats["y_0"] >= int(per_source["y_0"]),
            "y_1": stats["y_1"] >= int(per_source["y_1"]),
            "unique_dates": stats["unique_dates"] >= int(per_source["unique_dates"]),
            "unique_tickers": stats["unique_tickers"] >= int(per_source["unique_tickers"]),
            "chronological_blocks": all(
                block["n"] >= int(block_rule["minimum_n"])
                and (not block_rule["require_both_classes"] or (block["y_0"] > 0 and block["y_1"] > 0))
                for block in blocks
            ),
        }
        stats["checks"] = source_checks
        stats["ready"] = all(source_checks.values())
        by_source[source] = stats
        checks[f"source:{source}"] = stats["ready"]

    payload = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_TO_FREEZE" if all(checks.values()) else "NOT_READY",
        "activation_contract_written": False,
        "seal_or_final_labels_opened": False,
        "role_assignment_selection_uses_labels": False,
        "dev_support_audit_reads_labels": True,
        "role": "DEV_EXTENSION",
        "counts": {
            "total": len(frame),
            "observed_source_family_counts": frame.source_family.astype(str).value_counts().to_dict(),
            "unexpected_source_families": unexpected_sources,
            "by_source_family": by_source,
        },
        "checks": checks,
        "minimum_support": minimum,
        "files": {
            "extension_csv": file_binding(DEV),
            "extension_manifest": file_binding(MANIFEST),
            "role_assignment_json": file_binding(ROLE_JSON),
            "role_assignment_parquet": file_binding(ROLE_PARQUET),
        },
        "blocking_reasons": [name for name, passed in checks.items() if not passed],
    }
    atomic_json(payload, OUTPUT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
