"""Recover label-blind raw SEC events formerly blocked by local-file gating."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import v36_exact_pipeline as exact


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POLICY_PATH = DATA / "V221_LEGACY_RAW_MASSIVE_GAP_RECOVERY_POLICY_V3.json"
UNIVERSE_PATH = DATA / "universe_v221_fresh_sec_expanded.csv"
PRIOR_PATH = DATA / "events_exact_v35.csv.gz"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
OUTPUT_PATH = DATA / "events_sec_v221_legacy_gap_20250601_20260827.csv"
REPORT_PATH = DATA / "V221_LEGACY_RAW_MASSIVE_GAP_RECOVERY_STATUS_V3.json"
FORBIDDEN_OUTCOME_COLUMNS = {
    "y", "label", "target", "fwd_ret_30m", "entry_price", "exit_price",
    "actual_entry_time_utc", "actual_exit_time_utc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_policy() -> tuple[dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 3,
        "policy_name": "V221_LEGACY_RAW_SEC_MASSIVE_ENTITLEMENT_GAP_RECOVERY_V3",
        "frozen_before_candidate_role_distribution_or_price_results": True,
        "selection_uses_roles": False,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "selection_uses_prices": False,
        "provider": "MASSIVE_SIP",
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("LEGACY_RAW_GAP_POLICY_GUARD_FAIL")
    if sha256(UNIVERSE_PATH) != policy["current_healthcare_universe_sha256"]:
        raise RuntimeError("LEGACY_RAW_GAP_UNIVERSE_HASH_MISMATCH")
    return policy, sha256(POLICY_PATH)


def outside_active_spacing(frame: pd.DataFrame, queue: pd.DataFrame, minutes: int) -> pd.DataFrame:
    active_times = {
        str(ticker): np.sort(pd.to_datetime(group.event_time_utc, utc=True).astype("int64").to_numpy())
        for ticker, group in queue.groupby("ticker")
    }
    embargo = int(pd.Timedelta(minutes=minutes).value)
    keep: list[int] = []
    for index, row in frame.iterrows():
        values = active_times.get(str(row.ticker))
        if values is None or not len(values):
            keep.append(index)
            continue
        target = int(pd.Timestamp(row.event_time_utc).value)
        position = int(np.searchsorted(values, target))
        distances: list[int] = []
        if position < len(values):
            distances.append(abs(int(values[position]) - target))
        if position:
            distances.append(abs(int(values[position - 1]) - target))
        if not distances or min(distances) > embargo:
            keep.append(index)
    return frame.loc[keep].copy()


def main() -> int:
    policy, policy_hash = load_policy()
    paths = [
        path for path in sorted(DATA.glob("events_sec_v*raw*.csv"))
        if "v221" not in path.name.lower()
        and "dev" not in path.name.lower()
        and "seal" not in path.name.lower()
    ]
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for path in paths:
        columns = {str(value).lower() for value in pd.read_csv(path, nrows=0).columns}
        if columns & FORBIDDEN_OUTCOME_COLUMNS:
            raise RuntimeError(f"RAW_SOURCE_OUTCOME_COLUMN_REJECTED:{path.name}")
        frame = exact.normalized_events(path)
        frame["discovery_raw_file"] = path.name
        frames.append(frame)
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    raw = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("event_id")
    stages: dict[str, int] = {"raw_unique_events": int(len(raw))}
    event_time = pd.to_datetime(raw.event_time_utc, utc=True)
    start = pd.Timestamp(policy["start"], tz="UTC")
    end = pd.Timestamp(policy["end"], tz="UTC") + pd.Timedelta(days=1)
    candidate = raw.loc[(event_time >= start) & (event_time < end)].copy()
    stages["entitlement_date_range"] = int(len(candidate))
    prior = exact.normalized_events(PRIOR_PATH)
    candidate, prior_stages = exact.outside_prior(candidate, prior)
    stages.update({f"prior_{key}": int(value) for key, value in prior_stages.items()})
    candidate = candidate.loc[
        candidate.form.fillna("").astype(str).isin(set(app.Config().sec_forms))
    ].copy()
    stages["allowed_forms"] = int(len(candidate))
    candidate = exact.regular_session(candidate, "US")
    stages["regular_session"] = int(len(candidate))
    universe = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    approved = set(universe.ticker.astype(str))
    candidate = candidate.loc[candidate.ticker.astype(str).isin(approved)].copy()
    stages["current_healthcare_primary_symbol"] = int(len(candidate))
    roles = pd.read_parquet(ROLE_PATH, columns=["event_id", "role", "role_hash"])
    assigned = set(roles.event_id.astype(str))
    stages["previous_role_available_for_exact_preservation"] = int(
        candidate.event_id.astype(str).isin(assigned).sum()
    )
    stages["previously_unassigned_for_frozen_v70_allocation"] = int(
        (~candidate.event_id.astype(str).isin(assigned)).sum()
    )
    queue = pd.read_parquet(QUEUE_PATH, columns=["event_id", "ticker", "event_time_utc"])
    candidate = outside_active_spacing(
        candidate, queue, int(policy["minimum_same_ticker_distance_from_active_queue_minutes"])
    )
    stages["outside_active_queue_spacing"] = int(len(candidate))
    candidate = exact.space_events(candidate).sort_values(["ticker", "event_time_utc", "event_id"])
    stages["candidate_internal_spacing"] = int(len(candidate))
    candidate["source"] = "SEC_V221_LEGACY_RAW_GAP"
    candidate["body"] = ""
    atomic_csv(candidate, OUTPUT_PATH)
    report = {
        "schema_version": 3,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "selection_uses_labels": False,
        "selection_uses_roles": False,
        "outcome_columns_loaded": False,
        "selection_uses_prices": False,
        "source_files": int(len(paths)),
        "source_file_hashes": source_hashes,
        "stages": stages,
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "output_rows": int(len(candidate)),
        "output_tickers": int(candidate.ticker.nunique()) if not candidate.empty else 0,
        "output_sha256": sha256(OUTPUT_PATH),
        "existing_roles_opened_or_changed": False,
        "existing_roles_to_be_preserved_on_queue_reactivation": True,
    }
    atomic_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
