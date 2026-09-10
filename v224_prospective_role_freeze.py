"""Append-only, outcome-blind role freeze for V224 prospective US SEC rows.

This module deliberately does not import the V59 role/queue code and never
loads the central role parquet.  The protected authority is byte-hashed only.
All assignments live in a V224-specific namespace and are strictly later than
the protocol freeze, so an existing RESEARCH_SEAL_POOL or FINAL_META_RESERVE
row cannot be reclassified by this runner.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
POLICY_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION_POLICY_V1.json")
STATUS_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION/ROLE_FREEZE_STATUS.json")
ROLE = "V224_PROSPECTIVE_CONFIRMATION"
RESERVED_ROLES = {"RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"}
FORBIDDEN_OUTCOME_COLUMNS = {
    "y", "label", "target", "fwd_ret_30m", "return", "net_return",
    "prediction", "probability", "correct", "direction_correct",
}
RAW_COLUMNS = {
    "event_id", "market", "ticker", "company", "event_time_utc", "source",
    "form", "headline", "body", "event_type", "timestamp_quality", "url",
}
LEDGER_COLUMNS = [
    "assignment_order", "event_id", "market", "ticker", "company",
    "event_time_utc", "source", "source_family", "form", "headline", "body",
    "event_type", "timestamp_quality", "url", "data_role", "role_namespace",
    "role_frozen_at_utc", "role_policy_sha256", "raw_event_file",
    "raw_event_file_sha256",
]
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9-]{0,9}$")
NY = ZoneInfo("America/New_York")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                frame.to_csv(text_handle, index=False, lineterminator="\n")
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(temporary, path)


def load_policy(root: Path = ROOT) -> tuple[dict[str, Any], Path, str]:
    path = root / POLICY_REL
    policy = json.loads(path.read_text(encoding="utf-8"))
    role = policy.get("role_contract", {})
    exact = policy.get("exact_price_contract", {})
    decision = policy.get("decision_contract", {})
    execution = policy.get("execution_contract", {})
    failures: list[str] = []
    if policy.get("status") != "FROZEN_BEFORE_PROSPECTIVE_RESULTS":
        failures.append("POLICY_NOT_FROZEN")
    if policy.get("frozen_before_prospective_rows") is not True:
        failures.append("NOT_FROZEN_BEFORE_ROWS")
    if policy.get("selection_uses_labels") is not False:
        failures.append("ROLE_SELECTION_LABEL_DEPENDENT")
    expected_role = {
        "new_role": ROLE,
        "eligible_market": "US",
        "eligible_source_family": "US_SEC",
        "target_rows": 100,
        "preliminary_rows": 60,
        "checkpoint_block_rows": 20,
        "checkpoint_rows": [20, 40, 60, 80, 100],
        "existing_central_roles_may_be_reclassified": False,
        "central_role_table_may_be_written": False,
        "central_queue_may_be_written": False,
        "central_epoch_may_be_written": False,
        "role_assignment_uses_outcomes": False,
        "role_assignment_uses_prices": False,
    }
    for key, expected in expected_role.items():
        if role.get(key) != expected:
            failures.append(f"ROLE_CONTRACT_{key.upper()}")
    expected_exact = {
        "contract_version": "EXACT_T2_30M_V36",
        "bar_cadence_seconds": 60,
        "bar_timestamp_semantics": "OPEN",
        "entry_lag_minutes": 2,
        "horizon_minutes_from_actual_entry": 30,
        "entry_slippage_seconds_inclusive": [0, 60],
        "exit_slippage_seconds_inclusive": [0, 60],
        "actual_hold_seconds_inclusive": [1800, 1860],
        "missing_bar_policy": "NO_FILL_NO_INTERPOLATION",
    }
    for key, expected in expected_exact.items():
        if exact.get(key) != expected:
            failures.append(f"EXACT_CONTRACT_{key.upper()}")
    for key in (
        "checkpoint_results_may_change_protocol", "v224_results_may_change_protocol",
        "watcher_may_open_research_seal", "watcher_may_open_final_meta",
        "watcher_may_promote_champion", "watcher_may_tune_model_or_threshold",
    ):
        if decision.get(key) is not False:
            failures.append(f"DECISION_CONTRACT_{key.upper()}")
    if execution.get("one_shot") is not True or execution.get("poll_or_sleep") is not False:
        failures.append("NOT_ONE_SHOT")
    if failures:
        raise RuntimeError("V224_POLICY_GUARD_FAIL:" + ",".join(failures))
    return policy, path, sha256(path)


def protected_hashes(root: Path, policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, contract in policy["protected_authorities"].items():
        path = root / contract["path"]
        observed[name] = {
            "path": contract["path"],
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "freeze_sha256": contract["freeze_sha256"],
            "matches_freeze": bool(path.is_file() and sha256(path) == contract["freeze_sha256"]),
            "content_loaded": False,
            "written": False,
        }
    return observed


def validate_authorities(root: Path, policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed = protected_hashes(root, policy)
    role = observed["central_role_assignment"]
    if not role["exists"]:
        raise RuntimeError("CENTRAL_ROLE_AUTHORITY_MISSING")
    # The central role table is never opened.  A byte-level change after this
    # freeze is ambiguous, so fail closed instead of risking a role collision.
    if not role["matches_freeze"]:
        raise RuntimeError("CENTRAL_ROLE_AUTHORITY_CHANGED_AFTER_V224_FREEZE")
    for name, contract in policy["v224_authority"].items():
        if not name.endswith("_path"):
            continue
        prefix = name.removesuffix("_path")
        path = root / contract
        expected = policy["v224_authority"][prefix + "_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"V224_AUTHORITY_HASH_MISMATCH:{prefix}")
    return observed


def _load_existing(path: Path, policy_sha: str, target: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in LEDGER_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError("ROLE_LEDGER_SCHEMA_MISMATCH:" + ",".join(missing))
    frame = frame[LEDGER_COLUMNS].copy()
    if len(frame) > target or frame.event_id.duplicated().any():
        raise RuntimeError("ROLE_LEDGER_CARDINALITY_INVALID")
    orders = pd.to_numeric(frame.assignment_order, errors="coerce").tolist()
    if orders != list(range(1, len(frame) + 1)):
        raise RuntimeError("ROLE_LEDGER_ORDER_INVALID")
    if len(frame) and (
        set(frame.data_role) != {ROLE}
        or set(frame.role_namespace) != {"INDEPENDENT_APPEND_ONLY_V224"}
        or set(frame.role_policy_sha256) != {policy_sha}
        or RESERVED_ROLES.intersection(set(frame.data_role))
    ):
        raise RuntimeError("ROLE_LEDGER_IMMUTABILITY_GUARD_FAIL")
    return frame


def _candidate_frame(path: Path, policy: dict[str, Any], policy_sha: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    forbidden = FORBIDDEN_OUTCOME_COLUMNS.intersection(header)
    if forbidden:
        raise RuntimeError(f"RAW_CANDIDATE_HAS_OUTCOME_COLUMNS:{path.name}:{sorted(forbidden)}")
    needed = {"event_id", "market", "ticker", "event_time_utc", "source", "form", "timestamp_quality"}
    if not needed.issubset(header):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    usecols = [column for column in header if column in RAW_COLUMNS]
    frame = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False)
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = ""
    timestamp = pd.to_datetime(frame.event_time_utc, utc=True, errors="coerce")
    frozen = pd.Timestamp(policy["frozen_at_utc"])
    local = timestamp.dt.tz_convert(NY)
    local_time = local.dt.time
    allowed_forms = set(policy["role_contract"]["allowed_forms"])
    ticker = frame.ticker.str.upper().str.replace(".", "-", regex=False).str.strip()
    source = frame.source.str.upper().str.strip()
    eligible = (
        timestamp.notna()
        & timestamp.gt(frozen)
        & frame.market.str.upper().eq("US")
        & frame.timestamp_quality.str.upper().eq("EXACT")
        & source.str.startswith("SEC")
        & frame.form.str.upper().isin(allowed_forms)
        & ticker.map(lambda value: bool(TICKER_RE.fullmatch(value)))
        & local.dt.weekday.lt(5)
        & local_time.map(lambda value: time(9, 30) <= value < time(15, 28) if pd.notna(value) else False)
    )
    frame = frame.loc[eligible].copy()
    if frame.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame["ticker"] = ticker.loc[frame.index]
    frame["market"] = "US"
    frame["event_time_utc"] = timestamp.loc[frame.index].map(lambda value: value.isoformat())
    frame["source_family"] = "US_SEC"
    frame["data_role"] = ROLE
    frame["role_namespace"] = "INDEPENDENT_APPEND_ONLY_V224"
    frame["role_frozen_at_utc"] = utc_now()
    frame["role_policy_sha256"] = policy_sha
    frame["raw_event_file"] = str(path.relative_to(path.parents[1])).replace("\\", "/")
    frame["raw_event_file_sha256"] = sha256(path)
    frame["assignment_order"] = 0
    return frame[LEDGER_COLUMNS]


def discover_candidates(root: Path, policy: dict[str, Any], policy_sha: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths: dict[str, Path] = {}
    for pattern in policy["role_contract"]["raw_candidate_globs"]:
        for path in root.glob(pattern):
            paths[str(path.resolve()).lower()] = path
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(paths.values()):
        candidate = _candidate_frame(path, policy, policy_sha)
        frames.append(candidate)
        sources.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256(path),
            "eligible_rows_after_freeze": int(len(candidate)),
            "outcome_columns_loaded": False,
        })
    if not frames:
        return pd.DataFrame(columns=LEDGER_COLUMNS), sources
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS), sources
    combined = combined.sort_values(["event_time_utc", "event_id", "raw_event_file"])
    combined = combined.drop_duplicates("event_id", keep="first")
    return combined[LEDGER_COLUMNS].reset_index(drop=True), sources


def status_name(rows: int, appended: int, target: int, preliminary: int) -> str:
    if rows == 0:
        return "BOOTSTRAPPED_ZERO_PROSPECTIVE_ROWS"
    if rows >= target:
        return "ROLE_TARGET_100_FROZEN"
    if rows >= preliminary:
        return "PRELIMINARY_60_ROLE_ROWS_FROZEN_NON_DECISIONAL"
    return "PROSPECTIVE_ROLE_COLLECTION_ACTIVE"


def run(root: Path = ROOT) -> dict[str, Any]:
    started = utc_now()
    policy, policy_path, policy_sha = load_policy(root)
    protected_before = validate_authorities(root, policy)
    role_contract = policy["role_contract"]
    target = int(role_contract["target_rows"])
    preliminary = int(role_contract["preliminary_rows"])
    ledger_path = root / role_contract["ledger_path"]
    previous = _load_existing(ledger_path, policy_sha, target)
    previous_records_sha = canonical_sha256(previous.to_dict(orient="records"))
    candidates, sources = discover_candidates(root, policy, policy_sha)
    existing_ids = set(previous.event_id.astype(str))
    additions = candidates.loc[~candidates.event_id.astype(str).isin(existing_ids)].copy()
    additions = additions.head(max(0, target - len(previous)))
    if not additions.empty:
        additions["assignment_order"] = range(len(previous) + 1, len(previous) + len(additions) + 1)
    combined = pd.concat([previous, additions], ignore_index=True)[LEDGER_COLUMNS]
    if canonical_sha256(combined.iloc[: len(previous)].to_dict(orient="records")) != previous_records_sha:
        raise RuntimeError("APPEND_ONLY_PREFIX_CHANGED")
    if not ledger_path.exists() or len(additions):
        atomic_csv_gzip(combined, ledger_path)
    protected_after = protected_hashes(root, policy)
    changed = [
        name for name in protected_before
        if protected_before[name]["sha256"] != protected_after[name]["sha256"]
    ]
    if changed:
        raise RuntimeError("PROTECTED_AUTHORITY_CHANGED_DURING_RUN:" + ",".join(changed))
    checkpoints = role_contract["checkpoint_rows"]
    next_checkpoint = next((value for value in checkpoints if value > len(combined)), None)
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "phase": "ROLE_FREEZE",
        "status": status_name(len(combined), len(additions), target, preliminary),
        "run_started_utc": started,
        "run_completed_utc": utc_now(),
        "one_shot": True,
        "poll_or_sleep": False,
        "exited_without_wait": True,
        "policy_path": str(policy_path.relative_to(root)).replace("\\", "/"),
        "policy_sha256": policy_sha,
        "frozen_at_utc": policy["frozen_at_utc"],
        "role": ROLE,
        "role_namespace": "INDEPENDENT_APPEND_ONLY_V224",
        "rows_before": int(len(previous)),
        "rows_appended": int(len(additions)),
        "rows_frozen": int(len(combined)),
        "target_rows": target,
        "preliminary_rows": preliminary,
        "checkpoint_block_rows": int(role_contract["checkpoint_block_rows"]),
        "next_checkpoint_rows": next_checkpoint,
        "candidate_files": sources,
        "eligible_unique_candidates": int(len(candidates)),
        "ledger_path": str(ledger_path.relative_to(root)).replace("\\", "/"),
        "ledger_sha256": sha256(ledger_path),
        "logical_append_only": True,
        "previous_prefix_records_sha256": previous_records_sha,
        "central_role_table_content_loaded": False,
        "central_role_rows_read": 0,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "outcome_columns_loaded": False,
        "price_columns_loaded": False,
        "existing_roles_reclassified": False,
        "protected_authorities_before": protected_before,
        "protected_authorities_after": protected_after,
        "protected_authorities_changed": changed,
        "protocol_change_authority": False,
        "seal_or_final_open_authority": False,
    }
    atomic_json(status, root / STATUS_REL)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    try:
        status = run(root)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CONFIRMATION",
            "phase": "ROLE_FREEZE",
            "status": "FAILED_CLOSED",
            "timestamp_utc": utc_now(),
            "one_shot": True,
            "poll_or_sleep": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "central_role_table_content_loaded": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "outcome_columns_loaded": False,
            "existing_roles_reclassified": False,
            "protocol_change_authority": False,
        }
        atomic_json(failure, root / STATUS_REL)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
