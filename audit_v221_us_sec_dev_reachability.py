"""Label-blind reachability audit for the V221 US_SEC DEV floor.

This audit does not change assignments, fetch prices, or read outcomes.  Its
upper bound deliberately assumes every entitlement-addressable DEV event can
pass the exact timing contract; if even that bound is below the frozen floor,
the current queue cannot open the new data epoch without genuinely new event
IDs or a new historical provider authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import v59_provider_acquisition as provider


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "V221_US_SEC_DEV_REACHABILITY_AUDIT.json"
DEV_FLOOR = 120


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def predicted_role(event_id: str, seed: str, dev_share: int, research_share: int) -> str:
    digest = hashlib.sha256(f"{seed}|{event_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < dev_share:
        return "DEV_EXTENSION"
    if bucket < dev_share + research_share:
        return "RESEARCH_SEAL_POOL"
    return "FINAL_META_RESERVE"


def main() -> None:
    queue = pd.read_parquet(provider.US_QUEUE)
    roles = pd.read_parquet(provider.ROLE_PATH)
    if roles.event_id.astype(str).duplicated().any():
        raise RuntimeError("duplicate frozen event_id")
    assignment = json.loads((provider.DATA / "V59_DATA_ROLE_ASSIGNMENT.json").read_text(encoding="utf-8"))
    policy = json.loads(provider.NEW_ROLE_POLICY_PATH.read_text(encoding="utf-8"))
    allocation = policy["new_row_allocation"]
    if policy.get("selection_uses_labels") is not False or sum(allocation.values()) != 100:
        raise RuntimeError("invalid frozen new-row role policy")

    merged = queue.merge(
        roles[["event_id", "role"]], on="event_id", how="left", validate="one_to_one"
    )
    if merged.role.isna().any():
        raise RuntimeError("queue row lacks frozen role")
    merged["event_time_utc"] = pd.to_datetime(merged.event_time_utc, utc=True)
    sec_dev = merged[
        merged.source_family.eq("US_SEC") & merged.role.eq("DEV_EXTENSION")
    ].copy()
    anchor = provider.massive_verified_entitlement_anchor()
    sec_dev["trade_date"] = sec_dev.event_time_utc.dt.floor("D")
    sec_dev["entitlement_addressable"] = sec_dev.trade_date.ge(anchor)

    candidates = provider._us_candidates()
    candidates["event_time_utc"] = pd.to_datetime(candidates.event_time_utc, utc=True)
    frozen_ids = set(roles.event_id.astype(str))
    unassigned = candidates[~candidates.event_id.astype(str).isin(frozen_ids)].copy()
    unassigned = unassigned[
        unassigned.source.astype(str).str.contains("SEC", case=False, na=False)
    ]
    unassigned["predicted_role"] = unassigned.event_id.astype(str).map(
        lambda value: predicted_role(
            value,
            provider.ROLE_SEED,
            int(allocation["DEV_EXTENSION"]),
            int(allocation["RESEARCH_SEAL_POOL"]),
        )
    )
    unassigned["entitlement_addressable"] = (
        unassigned.event_time_utc.dt.floor("D").ge(anchor)
    )
    predicted_new_dev = unassigned[
        unassigned.predicted_role.eq("DEV_EXTENSION")
        & unassigned.entitlement_addressable
    ]

    timing = json.loads((provider.DATA / "V59_EXACT_TIMING_ELIGIBILITY.json").read_text(encoding="utf-8"))
    exact_now = int(timing.get("eligible_by_role", {}).get("US:DEV_EXTENSION", 0))
    addressable_current = sec_dev[sec_dev.entitlement_addressable]
    strict_upper = int(len(addressable_current) + len(predicted_new_dev))
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CURRENT_QUEUE_STRICT_UPPER_BOUND_BELOW_FROZEN_FLOOR" if strict_upper < DEV_FLOOR else "UPPER_BOUND_CAN_REACH_FLOOR",
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "existing_roles_changed": False,
        "frozen_floor": {"source_family": "US_SEC", "role": "DEV_EXTENSION", "n": DEV_FLOOR},
        "current_exact_timing_eligible": exact_now,
        "current_exact_gap": max(0, DEV_FLOOR - exact_now),
        "massive_oldest_verified_entitlement_anchor": anchor.date().isoformat(),
        "current_frozen_us_sec_dev": {
            "rows": int(len(sec_dev)),
            "by_status": {str(key): int(value) for key, value in sec_dev.status.value_counts().to_dict().items()},
            "entitlement_addressable_rows": int(len(addressable_current)),
            "entitlement_addressable_by_status": {
                str(key): int(value) for key, value in addressable_current.status.value_counts().to_dict().items()
            },
            "outside_verified_entitlement_rows": int((~sec_dev.entitlement_addressable).sum()),
        },
        "new_unassigned_us_sec": {
            "rows": int(len(unassigned)),
            "predicted_role_counts": {
                str(key): int(value) for key, value in unassigned.predicted_role.value_counts().to_dict().items()
            },
            "predicted_entitlement_addressable_dev_rows": int(len(predicted_new_dev)),
            "event_id_set_sha256": hashlib.sha256(
                "\n".join(sorted(unassigned.event_id.astype(str))).encode("utf-8")
            ).hexdigest(),
        },
        "strict_no_new_future_event_upper_bound": strict_upper,
        "strict_upper_bound_gap": max(0, DEV_FLOOR - strict_upper),
        "upper_bound_assumption": "Every current or newly unassigned entitlement-addressable frozen/predicted DEV event passes exact timing; this is deliberately optimistic.",
        "required_resolution": [
            "genuinely new unassigned US_SEC event_ids under the frozen 80/10/10 label-blind allocation",
            "or an independently verified historical one-minute OPEN-bar provider covering the old frozen DEV dates",
        ] if strict_upper < DEV_FLOOR else [],
        "forbidden_shortcuts": [
            "reassign existing research/final rows",
            "lower the 120-row floor",
            "relax exact entry/exit timing",
            "substitute US_NEWS for the frozen US_SEC source family",
        ],
        "authority": {
            "queue_sha256": sha256(provider.US_QUEUE),
            "role_assignment_sha256": sha256(provider.ROLE_PATH),
            "role_report_existing_roles_unchanged": assignment.get("existing_roles_unchanged"),
            "new_role_policy_sha256": sha256(provider.NEW_ROLE_POLICY_PATH),
            "timing_audit_sha256": sha256(provider.DATA / "V59_EXACT_TIMING_ELIGIBILITY.json"),
        },
    }
    atomic_json(report, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT), "output_sha256": sha256(OUTPUT),
        "status": report["status"], "exact_now": exact_now,
        "strict_upper_bound": strict_upper, "strict_upper_bound_gap": report["strict_upper_bound_gap"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
