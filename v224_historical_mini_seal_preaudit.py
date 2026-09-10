"""Metadata-only preflight for the user-authorized V224 historical mini-seal.

Never loads price values, targets, or outcome columns. Does not run scoring,
training, acquisition, or the full-seal opener. All new files are isolated.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import v224_prospective_locked_evaluation as frozen
import v59_provider_acquisition as provider

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cert_research/V224_HISTORICAL_MINI_SEAL"
ROLE_COLUMNS = ["event_id", "market", "ticker", "event_time_utc", "source_family",
                "role", "labels_opened"]


def write_new(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def timestamp_eligibility(path, event_time):
    # Column projection is mandatory: never load OHLCV for this preflight.
    stamps = pd.read_parquet(path, columns=["timestamp"])
    values = pd.to_datetime(stamps.timestamp, utc=True).sort_values().drop_duplicates().array.asi8
    target = event_time + pd.Timedelta(minutes=2)
    i = int(np.searchsorted(values, target.value))
    if i >= len(values):
        return "BAR_MISSING"
    entry = pd.Timestamp(values[i], tz="UTC")
    if not 0 <= (entry - target).total_seconds() <= 60:
        return "ENTRY_TIMING_MISS"
    exit_target = entry + pd.Timedelta(minutes=30)
    j = int(np.searchsorted(values, exit_target.value))
    if j >= len(values):
        return "BAR_MISSING"
    exit_time = pd.Timestamp(values[j], tz="UTC")
    if not (0 <= (exit_time - exit_target).total_seconds() <= 60
            and 1800 <= (exit_time - entry).total_seconds() <= 1860):
        return "EXIT_TIMING_MISS"
    local, session_open, session_close = provider.app.session_bounds(event_time, "US")
    if local.weekday() >= 5 or not session_open <= local <= session_close:
        return "OFF_HOURS"
    if entry < session_open.tz_convert("UTC") or exit_time > session_close.tz_convert("UTC"):
        return "CROSSES_SESSION"
    if ((values >= session_open.tz_convert("UTC").value)
            & (values <= (entry - pd.Timedelta(minutes=1)).value)).sum() < 2:
        return "NO_PREHISTORY"
    return "OK"


def main():
    if (OUT / "METADATA_PREFLIGHT.json").exists():
        raise RuntimeError("PREAUDIT_ALREADY_EXISTS_INSPECT_DO_NOT_OVERWRITE")
    manifest, spec, state = frozen.verify_frozen_files(ROOT)
    before = {name: frozen.sha256_file(ROOT / frozen.FROZEN_REL / name)
              for name in manifest["files"]}
    write_new("PRE_CERT_INTEGRITY.json", {
        "files": before, "frozen_manifest_sha256": frozen.sha256_file(ROOT / frozen.FROZEN_REL / frozen.FROZEN_SHA_NAME),
        "source_dependencies": {p: {"match": frozen.sha256_file(ROOT / p) == d["sha256"]}
                                for p, d in manifest["source_dependencies"].items()},
        "training": spec["training"], "outcome_rows_read": 0,
    })
    roles = pd.read_parquet(provider.ROLE_PATH, columns=ROLE_COLUMNS)
    assert not roles.event_id.duplicated().any()
    inventory = roles.groupby(["role", "source_family"]).size().reset_index(name="rows")
    queue = pd.read_parquet(provider.US_QUEUE,
                           columns=["event_id", "issuer", "selection_uses_labels"])
    assert not queue.event_id.duplicated().any()
    assert not queue.selection_uses_labels.astype(bool).any()
    candidates = roles.loc[roles.role.eq("RESEARCH_SEAL_POOL") & roles.source_family.eq("US_SEC")].copy()
    candidates = candidates.merge(queue, on="event_id", how="left", validate="one_to_one")
    candidates["event_time_utc"] = pd.to_datetime(candidates.event_time_utc, utc=True, errors="raise")
    candidates = candidates.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
    status = provider.load_us_provider_status()  # schema inspected: metadata only
    rows = []
    for index, row in enumerate(candidates.itertuples(index=False)):
        event_time = row.event_time_utc
        local = event_time.tz_convert(provider.app.US_TZ)
        authorities = provider.resolve_us_price_cache_candidates(row.event_id, row.ticker, local.date(), status)
        reason, selected = "NO_AUTHORIZED_CACHE", None
        for authority in authorities:
            reason = timestamp_eligibility(authority[0], event_time)
            if reason == "OK":
                selected = authority
                break
        rows.append({
            "canonical_event_id": row.event_id, "event_time": event_time.isoformat(),
            "ticker": str(row.ticker), "CIK": row.event_id.split(":")[1] if row.event_id.startswith("SEC:") else "",
            "issuer": row.issuer, "source_family": row.source_family, "role": row.role,
            "labels_opened_metadata": bool(row.labels_opened),
            "exact_eligible": reason == "OK", "eligibility_reason": reason,
            "price_path": str(selected[0].relative_to(ROOT)) if selected else "",
            "price_provider": selected[1] if selected else "",
            "provider_symbol": selected[3] if selected else "",
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"metadata_rows_audited": index + 1, "outcome_rows_read": 0}), flush=True)
    metadata = pd.DataFrame(rows)
    metadata.to_csv(OUT / "US_SEC_ELIGIBILITY_METADATA.csv", index=False, lineterminator="\n", mode="x")
    eligible = metadata.loc[metadata.exact_eligible].copy()
    selected = eligible.head(60).copy()
    selected.insert(0, "cert_row_index", range(1, len(selected) + 1))
    selected.insert(1, "block_id", (selected.cert_row_index - 1) // 20 + 1)
    # Candidate selection only; final membership freeze follows contamination checks.
    selected.to_csv(OUT / "CANDIDATE_MEMBERSHIP_METADATA.csv", index=False, lineterminator="\n", mode="x")
    wanted = set(selected.canonical_event_id)
    exposures = []
    for relative in ["data/dev_contract_v36_labeled.csv.gz", "data/V59_DEV_EXTENSION_LABELED.csv.gz"]:
        ids = pd.read_csv(ROOT / relative, usecols=["event_id"], dtype=str)
        overlap = sorted(wanted & set(ids.event_id))
        exposures.append({"path": relative, "id_column_only": True, "overlap": overlap,
                          "row_count": len(ids), "sha256": frozen.sha256_file(ROOT / relative)})
    state_ids = set(state["machine_state"]["cumulative_raw_history"]["event_id"])
    state_overlap = sorted(wanted & state_ids)
    report = {
        "status": "INSUFFICIENT_MINI_SEAL_ROWS" if len(eligible) < 60 else "METADATA_FIRST_60_SELECTED_PENDING_FULL_PREAUDIT",
        "role_mapping": {"RESEARCH_SEAL": "RESEARCH_SEAL_POOL", "FINAL_META": "FINAL_META_RESERVE"},
        "role_assignment_sha256": frozen.sha256_file(provider.ROLE_PATH),
        "inventory_metadata_only": inventory.to_dict("records"),
        "research_seal_us_sec_metadata_rows": len(metadata), "exact_eligible_us_sec_rows": len(eligible),
        "candidate_rows": len(selected), "eligibility_reasons": dict(Counter(metadata.eligibility_reason)),
        "membership_selection_uses_outcomes": False, "price_columns_loaded": ["timestamp"],
        "outcome_rows_read": 0, "final_meta_outcome_rows_read": 0, "non_us_sec_outcome_rows_read": 0,
        "initial_training_overlap_audit": exposures, "frozen_state_event_id_overlap": state_overlap,
        "selected_date_min": selected.event_time.min() if len(selected) else None,
        "selected_date_max": selected.event_time.max() if len(selected) else None,
        "frozen_training_time_max": spec["training"]["training_time_max_utc"],
        "frozen_artifacts_unchanged": all(frozen.sha256_file(ROOT / frozen.FROZEN_REL / n) == h for n, h in before.items()),
    }
    write_new("METADATA_PREFLIGHT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
