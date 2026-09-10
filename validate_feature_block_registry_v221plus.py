from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "research" / "FEATURE_BLOCK_REGISTRY_V221PLUS.json"
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_PATH = ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    authority = registry["authority"]
    if sha256(DEV_PATH) != authority["immutable_v36_dev_sha256"]:
        raise RuntimeError("V36 authority mismatch")
    if sha256(V69_PATH) != authority["atomic_v69_oof_sha256"]:
        raise RuntimeError("V69 authority mismatch")
    inventory_source = ROOT / authority["inventory_source"]
    if sha256(inventory_source) != authority["inventory_source_sha256"]:
        raise RuntimeError("V64 inventory source mismatch")

    v36_columns = set(pd.read_csv(DEV_PATH, nrows=0).columns)
    v69_columns = set(pd.read_csv(V69_PATH, nrows=0).columns)
    derived = set(registry["derived_strict_past_columns"])
    forbidden = set(registry["forbidden_label_or_future"])
    excluded = set(registry["identifier_or_provenance_excluded"])
    seen: set[str] = set()
    block_counts: dict[str, int] = {}
    for name, block in registry["blocks"].items():
        columns = list(block["columns"])
        if not columns or len(columns) != len(set(columns)):
            raise RuntimeError(f"empty or duplicate columns in block {name}")
        overlap = seen.intersection(columns)
        if overlap:
            raise RuntimeError(f"cross-block overlap: {name}: {sorted(overlap)}")
        seen.update(columns)
        block_counts[name] = len(columns)
        unknown = set(columns) - v36_columns - derived
        if unknown:
            raise RuntimeError(f"unknown columns in {name}: {sorted(unknown)}")
    if seen.intersection(forbidden | excluded):
        raise RuntimeError("model blocks contain forbidden/future/identifier columns")
    if set(registry["source_families"]) != {"US_SEC", "US_NEWS", "KR_NEWS", "KR_KIND"}:
        raise RuntimeError("source-family scope mismatch")
    for role in registry["roles"].values():
        if role["anchor_column"] not in v69_columns:
            raise RuntimeError(f"missing V69 anchor: {role['anchor_column']}")
    if registry["counterfactual"]["raw_feature_sign_flip_allowed"] is not False:
        raise RuntimeError("raw sign flip must remain forbidden")
    if registry["nested_selection"]["outer_result_reselection_allowed"] is not False:
        raise RuntimeError("outer-result reselection must remain forbidden")
    if registry["nested_selection"]["embargo_minutes"] != 35:
        raise RuntimeError("embargo changed")
    print(
        json.dumps(
            {
                "registry": str(REGISTRY_PATH.relative_to(ROOT)),
                "registry_sha256": sha256(REGISTRY_PATH),
                "block_count": len(registry["blocks"]),
                "block_column_counts": block_counts,
                "unique_model_inputs": len(seen),
                "v36_columns": len(v36_columns),
                "v69_anchor_columns_verified": True,
                "forbidden_overlap": 0,
                "verification": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
