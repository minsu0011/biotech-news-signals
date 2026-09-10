"""One-shot V224 prospective exact-row collector and 20-row checkpoint watcher.

The default execution performs no network requests.  It freezes newly present
raw US SEC events, labels only that independent prospective role from already
available exact bars, writes a concrete zero-row bootstrap status when needed,
and exits.  ``--collect-prices`` explicitly permits a bounded read-only Massive
aggregate fetch into the independent V224 cache.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import v224_prospective_role_freeze as role_freeze


ROOT = Path(__file__).resolve().parent
STATUS_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION/WATCH_STATUS.json")
CHECKPOINT_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION/checkpoints")
ROLE = role_freeze.ROLE
EMPTY_EXACT_COLUMNS = [
    "prospective_observation_order", "prospective_role_assignment_order",
    "event_id", "market", "ticker", "event_time_utc", "source", "source_family",
    "form", "event_type", "headline", "body", "text", "contract_version",
    "price_source", "price_cadence_sec", "bar_timestamp_semantics",
    "entry_target_utc", "actual_entry_time_utc", "entry_slippage_sec",
    "exit_target_utc", "actual_exit_time_utc", "exit_slippage_sec",
    "actual_hold_seconds", "entry_price", "exit_price", "fwd_ret_30m", "y",
    "data_role", "role_policy_sha256", "raw_price_file", "raw_price_file_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _cache_path(root: Path, cache_rel: str, ticker: str, trade_date: str) -> Path:
    return root / cache_rel / trade_date[:4] / ticker / f"{trade_date}.parquet"


def _read_role_ledger(root: Path, policy: dict[str, Any], policy_sha: str) -> pd.DataFrame:
    path = root / policy["role_contract"]["ledger_path"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) and (
        set(frame.data_role) != {ROLE}
        or set(frame.role_policy_sha256) != {policy_sha}
        or frame.event_id.duplicated().any()
    ):
        raise RuntimeError("PROSPECTIVE_ROLE_LEDGER_INVALID")
    return frame


def _load_exact(path: Path, policy_sha: str, target: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EMPTY_EXACT_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "prospective_observation_order", "event_id", "data_role", "role_policy_sha256",
        "entry_slippage_sec", "exit_slippage_sec", "actual_hold_seconds",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("EXACT_LEDGER_SCHEMA_INVALID")
    if len(frame) > target or frame.event_id.astype(str).duplicated().any():
        raise RuntimeError("EXACT_LEDGER_CARDINALITY_INVALID")
    observed_order = pd.to_numeric(frame.prospective_observation_order, errors="coerce").tolist()
    if observed_order != list(range(1, len(frame) + 1)):
        raise RuntimeError("EXACT_LEDGER_ORDER_INVALID")
    if len(frame) and (set(frame.data_role.astype(str)) != {ROLE} or set(frame.role_policy_sha256.astype(str)) != {policy_sha}):
        raise RuntimeError("EXACT_LEDGER_ROLE_INVALID")
    entry = pd.to_numeric(frame.entry_slippage_sec, errors="coerce")
    exit_slip = pd.to_numeric(frame.exit_slippage_sec, errors="coerce")
    hold = pd.to_numeric(frame.actual_hold_seconds, errors="coerce")
    if len(frame) and not (entry.between(0, 60).all() and exit_slip.between(0, 60).all() and hold.between(1800, 1860).all()):
        raise RuntimeError("EXACT_LEDGER_TIMING_CONTRACT_INVALID")
    return frame


def bar_qa(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        return {"pass": False, "reason": "EMPTY_OR_SCHEMA", "rows": int(len(frame))}
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work.timestamp, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.sort_values("timestamp")
    invalid_timestamp = int(work.timestamp.isna().sum())
    duplicate = int(work.timestamp.duplicated().sum())
    invalid_price = int((work[["open", "high", "low", "close"]].le(0) | work[["open", "high", "low", "close"]].isna()).any(axis=1).sum())
    bad_ohlc = int(((work.high < work[["open", "close", "low"]].max(axis=1)) | (work.low > work[["open", "close", "high"]].min(axis=1))).sum())
    negative_volume = int(work.volume.lt(0).sum())
    result = {
        "rows": int(len(work)), "invalid_timestamp": invalid_timestamp,
        "duplicates": duplicate, "invalid_price": invalid_price,
        "bad_ohlc": bad_ohlc, "negative_volume": negative_volume,
        "missing_bar_policy": "NO_FILL_NO_INTERPOLATION",
    }
    result["pass"] = not any((invalid_timestamp, duplicate, invalid_price, bad_ohlc, negative_volume))
    return result


def _fetch_massive_day(ticker: str, trade_date: str, api_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import requests

    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"
    try:
        response = requests.get(
            url,
            params={"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=45,
        )
        payload = response.json()
        if response.status_code >= 400 or str(payload.get("status", "")).upper() not in {"OK", "DELAYED"}:
            return pd.DataFrame(), {"ok": False, "http_status": response.status_code, "error": str(payload.get("error") or payload.get("message") or "PROVIDER_ERROR")[:240]}
        raw = pd.DataFrame(payload.get("results") or [])
        if raw.empty:
            return pd.DataFrame(), {"ok": False, "http_status": response.status_code, "error": "EMPTY"}
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(raw["t"], unit="ms", utc=True),
            "open": pd.to_numeric(raw["o"], errors="coerce"),
            "high": pd.to_numeric(raw["h"], errors="coerce"),
            "low": pd.to_numeric(raw["l"], errors="coerce"),
            "close": pd.to_numeric(raw["c"], errors="coerce"),
            "volume": pd.to_numeric(raw["v"], errors="coerce"),
            "transactions": pd.to_numeric(raw["n"], errors="coerce") if "n" in raw else 0,
            "vwap": pd.to_numeric(raw["vw"], errors="coerce") if "vw" in raw else pd.NA,
        }).sort_values("timestamp").drop_duplicates("timestamp")
        qa = bar_qa(frame)
        return frame, {"ok": bool(qa["pass"]), "http_status": response.status_code, "qa": qa}
    except Exception as error:
        return pd.DataFrame(), {"ok": False, "error": f"{type(error).__name__}:{error}"[:240]}


def collect_prices(root: Path, roles: pd.DataFrame, policy: dict[str, Any], max_requests: int) -> dict[str, Any]:
    exact = policy["exact_price_contract"]
    independent = exact["independent_cache_root"]
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    used = 0
    fetched = 0
    failures: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    if max_requests == 0:
        return {
            "enabled": False,
            "provider": "MASSIVE_READ_ONLY_AGGREGATES",
            "credential_present": bool(key),
            "credential_value_logged": False,
            "request_budget": 0,
            "requests_used": 0,
            "ticker_days_fetched": 0,
            "failure_counts": {},
            "cache_root": independent,
            "central_queue_written": False,
            "central_cache_written": False,
        }
    for row in roles.itertuples(index=False):
        timestamp = pd.Timestamp(row.event_time_utc).tz_convert("America/New_York")
        trade_date = str(timestamp.date())
        pair = (str(row.ticker), trade_date)
        if pair in seen:
            continue
        seen.add(pair)
        path = _cache_path(root, independent, pair[0], pair[1])
        if path.exists():
            continue
        if used >= max_requests:
            failures["REQUEST_BUDGET_EXHAUSTED"] += 1
            continue
        if not key:
            failures["MASSIVE_CREDENTIAL_MISSING"] += 1
            continue
        bars, result = _fetch_massive_day(pair[0], pair[1], key)
        used += 1
        if result.get("ok") and bar_qa(bars)["pass"]:
            atomic_parquet(bars, path)
            fetched += 1
        else:
            failures[str(result.get("error") or "QA_FAIL")] += 1
    return {
        "enabled": max_requests > 0,
        "provider": "MASSIVE_READ_ONLY_AGGREGATES",
        "credential_present": bool(key),
        "credential_value_logged": False,
        "request_budget": int(max_requests),
        "requests_used": int(used),
        "ticker_days_fetched": int(fetched),
        "failure_counts": dict(failures),
        "cache_root": independent,
        "central_queue_written": False,
        "central_cache_written": False,
    }


def _resolve_cache(root: Path, policy: dict[str, Any], ticker: str, trade_date: str) -> tuple[Path | None, str]:
    exact = policy["exact_price_contract"]
    for key, authority in (
        ("independent_cache_root", "V224_INDEPENDENT_MASSIVE_1M"),
        ("read_only_compatible_cache_root", "V59_MASSIVE_1M_READ_ONLY"),
    ):
        path = _cache_path(root, exact[key], ticker, trade_date)
        if path.is_file():
            return path, authority
    return None, "NO_CACHE"


def label_event(event: pd.Series, raw_bars: pd.DataFrame, price_source: str) -> tuple[dict[str, Any] | None, str]:
    """Apply the project's locked V36 exact contract to one prospective row."""
    import bio_news_30m_v3 as app

    qa = bar_qa(raw_bars)
    if not qa["pass"]:
        return None, "BAR_QA_FAIL"
    bars = raw_bars.rename(columns={"timestamp": "datetime"}).copy()
    bars["datetime"] = pd.to_datetime(bars.datetime, utc=True, errors="coerce")
    bars = bars.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    bars.attrs.update({
        "price_source": price_source,
        "price_cadence_sec": 60,
        "bar_timestamp_semantics": "OPEN",
        "official_exact_eligible": True,
    })
    cfg = app.Config(
        official_exact_contract=True,
        entry_lag_min=2,
        horizon_min=30,
        max_official_cadence_sec=60,
        max_entry_slippage_sec=60,
        max_exit_slippage_sec=60,
        min_hold_seconds=1800,
        max_hold_seconds=1860,
    )
    return app.event_to_row_exact_contract(event, bars, cfg)


def append_exact_rows(root: Path, roles: pd.DataFrame, policy: dict[str, Any], policy_sha: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    exact_contract = policy["exact_price_contract"]
    target = int(policy["role_contract"]["target_rows"])
    path = root / exact_contract["labeled_ledger_path"]
    previous = _load_exact(path, policy_sha, target)
    previous_hash = role_freeze.canonical_sha256(previous.to_dict(orient="records"))
    opened = set(previous.event_id.astype(str)) if len(previous) else set()
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    price_files: list[dict[str, Any]] = []
    for _, source_row in roles.iterrows():
        if len(previous) + len(rows) >= target:
            break
        if str(source_row.event_id) in opened:
            continue
        local = pd.Timestamp(source_row.event_time_utc).tz_convert("America/New_York")
        cache, authority = _resolve_cache(root, policy, str(source_row.ticker), str(local.date()))
        if cache is None:
            reasons["NO_EXACT_PRICE_CACHE"] += 1
            continue
        try:
            raw = pd.read_parquet(cache)
            row, reason = label_event(source_row, raw, authority)
        except Exception as error:
            row, reason = None, f"{type(error).__name__}:{error}"[:180]
        reasons[reason] += 1
        if row is None:
            continue
        raw_sha = role_freeze.sha256(cache)
        row.update({
            "prospective_observation_order": len(previous) + len(rows) + 1,
            "prospective_role_assignment_order": int(source_row.assignment_order),
            "data_role": ROLE,
            "role_policy_sha256": policy_sha,
            "raw_price_file": str(cache.relative_to(root)).replace("\\", "/"),
            "raw_price_file_sha256": raw_sha,
        })
        rows.append(row)
        price_files.append({
            "path": str(cache.relative_to(root)).replace("\\", "/"),
            "sha256": raw_sha,
            "authority": authority,
            "event_id": str(source_row.event_id),
        })
    if rows:
        combined = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True, sort=False)
    else:
        combined = previous
    if role_freeze.canonical_sha256(combined.iloc[: len(previous)].to_dict(orient="records")) != previous_hash:
        raise RuntimeError("EXACT_APPEND_ONLY_PREFIX_CHANGED")
    if not path.exists() or rows:
        role_freeze.atomic_csv_gzip(combined, path)
    return combined, {
        "rows_before": int(len(previous)),
        "rows_appended": int(len(rows)),
        "reason_counts": dict(reasons),
        "price_files_opened": price_files,
        "outcomes_opened_only_after_prospective_role_freeze": True,
    }


def create_checkpoints(root: Path, exact_rows: pd.DataFrame, policy: dict[str, Any], policy_sha: str) -> list[dict[str, Any]]:
    directory = root / CHECKPOINT_REL
    created_or_verified: list[dict[str, Any]] = []
    block_rows = int(policy["role_contract"]["checkpoint_block_rows"])
    for threshold in policy["role_contract"]["checkpoint_rows"]:
        if len(exact_rows) < threshold:
            continue
        snapshot_path = directory / f"V224_PROSPECTIVE_EXACT_BLOCK_{threshold:03d}.csv.gz"
        manifest_path = directory / f"CHECKPOINT_{threshold:03d}.json"
        prefix = exact_rows.iloc[:threshold].copy()
        block = exact_rows.iloc[threshold - block_rows : threshold].copy()
        expected_prefix_sha = role_freeze.canonical_sha256(prefix.to_dict(orient="records"))
        expected_block_sha = role_freeze.canonical_sha256(block.to_dict(orient="records"))
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("rows") != block_rows
                or manifest.get("cumulative_rows") != threshold
                or manifest.get("policy_sha256") != policy_sha
                or manifest.get("prefix_records_sha256") != expected_prefix_sha
                or manifest.get("block_records_sha256") != expected_block_sha
                or not snapshot_path.exists()
                or manifest.get("snapshot_sha256") != role_freeze.sha256(snapshot_path)
            ):
                raise RuntimeError(f"CHECKPOINT_IMMUTABILITY_FAIL:{threshold}")
            action = "VERIFIED_EXISTING"
        else:
            role_freeze.atomic_csv_gzip(block, snapshot_path)
            manifest = {
                "schema_version": 1,
                "track": "V224_PROSPECTIVE_CONFIRMATION",
                "checkpoint_rows": threshold,
                "rows": block_rows,
                "cumulative_rows": threshold,
                "block_start_observation_order": threshold - block_rows + 1,
                "block_end_observation_order": threshold,
                "created_at_utc": utc_now(),
                "policy_sha256": policy_sha,
                "prefix_records_sha256": expected_prefix_sha,
                "block_records_sha256": expected_block_sha,
                "snapshot_path": str(snapshot_path.relative_to(root)).replace("\\", "/"),
                "snapshot_sha256": role_freeze.sha256(snapshot_path),
                "source_family": "US_SEC",
                "contract_version": "EXACT_T2_30M_V36",
                "entry_lag_minutes": 2,
                "horizon_minutes_from_actual_entry": 30,
                "entry_slippage_seconds_inclusive": [0, 60],
                "exit_slippage_seconds_inclusive": [0, 60],
                "actual_hold_seconds_inclusive": [1800, 1860],
                "preliminary": threshold == 60,
                "preliminary_is_non_decisional": threshold == 60,
                "protocol_change_authority": False,
                "seal_or_final_open_authority": False,
                "v224_evaluation_run_by_watcher": False,
                "next_action": "SEPARATE_LOCKED_EVALUATION" if threshold == 100 else "CONTINUE_ONE_SHOT_COLLECTION",
            }
            role_freeze.atomic_json(manifest, manifest_path)
            action = "CREATED"
        created_or_verified.append({"rows": threshold, "action": action, "manifest_sha256": role_freeze.sha256(manifest_path)})
    return created_or_verified


def watcher_state(rows: int, role_rows: int) -> str:
    if role_rows == 0:
        return "BOOTSTRAPPED_ZERO_PROSPECTIVE_ROWS"
    if rows >= 100:
        return "TARGET_100_EXACT_ROWS_READY_FOR_SEPARATE_LOCKED_EVALUATION"
    if rows >= 60:
        return "PRELIMINARY_60_EXACT_ROWS_READY_NON_DECISIONAL"
    return "COLLECTING_EXACT_PROSPECTIVE_ROWS"


def run(root: Path = ROOT, collect: bool = False, max_requests: int = 0) -> dict[str, Any]:
    started = utc_now()
    if max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if not collect and max_requests != 0:
        raise ValueError("max_requests requires collect=True")
    role_status = role_freeze.run(root)
    policy, policy_path, policy_sha = role_freeze.load_policy(root)
    protected_before = role_freeze.protected_hashes(root, policy)
    roles = _read_role_ledger(root, policy, policy_sha)
    acquisition = collect_prices(root, roles, policy, max_requests if collect else 0)
    exact_rows, labeling = append_exact_rows(root, roles, policy, policy_sha)
    checkpoints = create_checkpoints(root, exact_rows, policy, policy_sha)
    protected_after = role_freeze.protected_hashes(root, policy)
    changed = [name for name in protected_before if protected_before[name]["sha256"] != protected_after[name]["sha256"]]
    if changed:
        raise RuntimeError("PROTECTED_AUTHORITY_CHANGED_DURING_WATCHER:" + ",".join(changed))
    exact_path = root / policy["exact_price_contract"]["labeled_ledger_path"]
    threshold_list = policy["role_contract"]["checkpoint_rows"]
    next_checkpoint = next((value for value in threshold_list if value > len(exact_rows)), None)
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "phase": "ONE_SHOT_EXACT_CHECKPOINT_WATCHER",
        "status": watcher_state(len(exact_rows), len(roles)),
        "run_started_utc": started,
        "run_completed_utc": utc_now(),
        "one_shot": True,
        "poll_or_sleep": False,
        "exited_without_wait": True,
        "default_network_requests": 0,
        "network_collection_explicitly_enabled": bool(collect),
        "policy_path": str(policy_path.relative_to(root)).replace("\\", "/"),
        "policy_sha256": policy_sha,
        "role_freeze_status": role_status["status"],
        "prospective_role_rows": int(len(roles)),
        "exact_rows": int(len(exact_rows)),
        "target_rows": 100,
        "preliminary_rows": 60,
        "checkpoint_block_rows": 20,
        "completed_checkpoint_rows": [int(item["rows"]) for item in checkpoints],
        "next_checkpoint_rows": next_checkpoint,
        "rows_to_preliminary": max(0, 60 - len(exact_rows)),
        "rows_to_target": max(0, 100 - len(exact_rows)),
        "exact_contract": policy["exact_price_contract"],
        "exact_ledger_path": str(exact_path.relative_to(root)).replace("\\", "/"),
        "exact_ledger_sha256": role_freeze.sha256(exact_path),
        "acquisition": acquisition,
        "labeling": labeling,
        "checkpoints": checkpoints,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_role_table_content_loaded": False,
        "central_role_rows_read": 0,
        "protected_authorities_changed": changed,
        "existing_roles_reclassified": False,
        "protocol_changed": False,
        "protocol_change_authority": False,
        "seal_or_final_open_authority": False,
        "v224_evaluation_run": False,
        "preliminary_is_non_decisional": True,
        "next_action": (
            "SEPARATE_LOCKED_V224_EVALUATION" if len(exact_rows) >= 100
            else "RUN_THIS_ONE_SHOT_WATCHER_AGAIN_AFTER_NEW_RAW_SEC_OR_PRICE_BARS_ARRIVE"
        ),
    }
    role_freeze.atomic_json(status, root / STATUS_REL)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--collect-prices", action="store_true")
    parser.add_argument("--max-requests", type=int, default=0)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    try:
        status = run(root, collect=args.collect_prices, max_requests=args.max_requests)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CONFIRMATION",
            "phase": "ONE_SHOT_EXACT_CHECKPOINT_WATCHER",
            "status": "FAILED_CLOSED",
            "timestamp_utc": utc_now(),
            "one_shot": True,
            "poll_or_sleep": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_role_table_content_loaded": False,
            "existing_roles_reclassified": False,
            "protocol_changed": False,
            "protocol_change_authority": False,
        }
        role_freeze.atomic_json(failure, root / STATUS_REL)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
