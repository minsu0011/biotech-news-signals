"""Track A: genuinely prediction-before-outcome V224 prospective runner.

This independent namespace consumes only the outcome-blind V224 role ledger
and raw Massive minute bars.  It projects causal features through the actual
t+2 entry using no future candle fields, durably freezes a full chronological
20-row prediction block, and only then opens exit-price projections.  The old
outcome-first watcher/checkpoint lineage is intentionally not imported.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import v224_prospective_locked_evaluation as locked


ROOT = Path(__file__).resolve().parent
TRACK_REL = Path("research/V224_PROSPECTIVE_CAUSAL_CONFIRMATION")
STATUS_REL = TRACK_REL / "TRACK_A_STATUS.json"
ROLE_LEDGER_REL = Path("data/V224_PROSPECTIVE_CONFIRMATION_ROLE_FREEZE.csv.gz")
ROLE = "V224_PROSPECTIVE_CONFIRMATION"
BLOCK_SIZE = 20
TARGET = 100
CHECKPOINTS = (20, 40, 60, 80, 100)
OUTCOME_NAMES = {"y", "fwd_ret_30m", "exit_price", "exit_time_utc", "actual_exit_time_utc"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def immutable_json(path: Path, value: dict[str, Any]) -> str:
    return locked.write_immutable(path, locked.pretty_json_bytes(clean_json(value)))


def convergent_json(path: Path, value: dict[str, Any]) -> str:
    return locked.write_convergent(path, locked.pretty_json_bytes(clean_json(value)))


def event_feature_path(root: Path, order: int) -> Path:
    return root / TRACK_REL / "events" / f"EVENT_{order:06d}_FEATURE.json"


def event_prediction_audit_path(root: Path, order: int) -> Path:
    return root / TRACK_REL / "events" / f"EVENT_{order:06d}_PREDICTION_AUDIT.json"


def event_outcome_path(root: Path, order: int) -> Path:
    return root / TRACK_REL / "events" / f"EVENT_{order:06d}_OUTCOME_AUDIT.json"


def block_paths(root: Path, block: int) -> dict[str, Path]:
    directory = root / TRACK_REL / "blocks"
    return {
        "features": directory / f"BLOCK_{block:03d}_CAUSAL_FEATURES.csv.gz",
        "feature_manifest": directory / f"BLOCK_{block:03d}_CAUSAL_FEATURES_MANIFEST.json",
        "predictions": directory / f"BLOCK_{block:03d}_PREDICTIONS.csv.gz",
        "prediction_manifest": directory / f"BLOCK_{block:03d}_PREDICTIONS_MANIFEST.json",
        "state": directory / f"STATE_AFTER_BLOCK_{block:03d}.json",
        "completion": directory / f"BLOCK_{block:03d}_COMPLETION.json",
    }


def load_roles(root: Path, policy_sha: str) -> pd.DataFrame:
    path = root / ROLE_LEDGER_REL
    if not path.is_file():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if OUTCOME_NAMES.intersection(header) or {"label", "return", "prediction"}.intersection(header):
        raise RuntimeError("TRACK_A_ROLE_LEDGER_CONTAINS_OUTCOME_OR_PREDICTION")
    roles = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "assignment_order", "event_id", "market", "ticker", "event_time_utc",
        "source_family", "form", "headline", "body", "event_type", "data_role",
        "role_policy_sha256",
    }
    if not required.issubset(roles.columns):
        raise RuntimeError("TRACK_A_ROLE_LEDGER_SCHEMA_MISMATCH")
    orders = pd.to_numeric(roles.assignment_order, errors="raise").astype(int).tolist()
    if (
        orders != list(range(1, len(roles) + 1))
        or len(roles) > TARGET
        or roles.event_id.duplicated().any()
        or (len(roles) and set(roles.data_role) != {ROLE})
        or (len(roles) and set(roles.role_policy_sha256) != {policy_sha})
        or (len(roles) and not roles.source_family.eq("US_SEC").all())
        or (len(roles) and not roles.market.eq("US").all())
    ):
        raise RuntimeError("TRACK_A_ROLE_LEDGER_GUARD_FAIL")
    roles["assignment_order"] = orders
    roles["event_time_utc"] = pd.to_datetime(roles.event_time_utc, utc=True, errors="raise")
    ordered = roles.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
    if ordered.event_id.astype(str).tolist() != roles.event_id.astype(str).tolist():
        raise RuntimeError("TRACK_A_ROLE_LEDGER_NOT_CHRONOLOGICAL")
    return roles


def resolve_price_path(root: Path, policy: dict[str, Any], ticker: str, trade_date: str) -> Path | None:
    exact = policy["exact_price_contract"]
    for key in ("independent_cache_root", "read_only_compatible_cache_root"):
        path = root / exact[key] / trade_date[:4] / ticker / f"{trade_date}.parquet"
        if path.is_file():
            return path
    return None


def projected_parquet(path: Path, columns: list[str], filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
    """Read only the declared columns and causal timestamp interval."""
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    if "timestamp" not in frame:
        raise RuntimeError("TRACK_A_PRICE_TIMESTAMP_MISSING")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _safe_returns(window: pd.DataFrame) -> pd.Series:
    if len(window) < 2:
        return pd.Series(dtype=float)
    return pd.to_numeric(window.close, errors="coerce").pct_change().dropna()


def causal_feature_snapshot(event: pd.Series, price_path: Path) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Build exact decision-time features without loading any exit/future field."""
    eutc = pd.Timestamp(event.event_time_utc).tz_convert("UTC")
    local, session_open, session_close = app.session_bounds(eutc, "US")
    if not session_open <= local <= session_close:
        return None, "OFF_HOURS", {}
    entry_target = eutc + pd.Timedelta(minutes=2)
    entry_limit = entry_target + pd.Timedelta(seconds=60)
    # Crucial projection: the current entry candle exposes timestamp and OPEN
    # only. Its high/low/close/volume are not decision-time information.
    entry_candidates = projected_parquet(
        price_path, ["timestamp", "open"],
        [("timestamp", ">=", entry_target), ("timestamp", "<=", entry_limit)],
    )
    if entry_candidates.empty:
        return None, "ENTRY_BAR_MISSING", {}
    entry_row = entry_candidates.iloc[0]
    actual_entry = pd.Timestamp(entry_row.timestamp).tz_convert("UTC")
    entry_slippage = float((actual_entry - entry_target).total_seconds())
    if not 0 <= entry_slippage <= 60:
        return None, "ENTRY_TIMING_MISS", {}
    entry_local = actual_entry.tz_convert(app.US_TZ)
    if entry_local.date() != local.date() or not session_open <= entry_local <= session_close:
        return None, "ENTRY_CROSSES_SESSION", {}
    entry_price = float(entry_row.open)
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None, "BAD_ENTRY_OPEN", {}
    session_start_utc = session_open.tz_convert("UTC")
    # Strictly pre-entry candles only; no field from the current/future bar.
    pre = projected_parquet(
        price_path, ["timestamp", "open", "high", "low", "close", "volume"],
        [("timestamp", ">=", session_start_utc), ("timestamp", "<", actual_entry)],
    )
    if len(pre) < 2:
        return None, "NO_PREHISTORY", {}
    for column in ("open", "high", "low", "close", "volume"):
        pre[column] = pd.to_numeric(pre[column], errors="coerce")
    if pre[["open", "high", "low", "close"]].le(0).any().any() or pre[["open", "high", "low", "close"]].isna().any().any():
        return None, "PREHISTORY_PRICE_QA_FAIL", {}
    pre["bar_end_utc"] = pre.timestamp + pd.Timedelta(seconds=60)
    completed = pre[pre.bar_end_utc <= actual_entry].copy()
    if len(completed) < 2 or completed.timestamp.max() >= actual_entry:
        return None, "CAUSAL_PREHISTORY_GUARD_FAIL", {}

    def clock_window(minutes: int) -> pd.DataFrame:
        start = actual_entry - pd.Timedelta(minutes=minutes)
        return completed[(completed.timestamp >= start) & (completed.bar_end_utc <= actual_entry)]

    def asof_close(target: pd.Timestamp) -> float:
        eligible = completed[completed.bar_end_utc <= target]
        if eligible.empty:
            return np.nan
        last = eligible.iloc[-1]
        stale = float((target - last.bar_end_utc).total_seconds())
        value = float(last.close)
        return value if 0 <= stale <= 60 and value > 0 else np.nan

    def ret_back(minutes: int) -> float:
        value = asof_close(actual_entry - pd.Timedelta(minutes=minutes))
        return float(entry_price / value - 1) if np.isfinite(value) and value > 0 else np.nan

    def volume_mean(minutes: int) -> float:
        values = pd.to_numeric(clock_window(minutes).volume, errors="coerce")
        return float(values.mean()) if len(values) and values.notna().any() else np.nan

    def rolling_range(minutes: int) -> float:
        window = clock_window(minutes)
        if window.empty:
            return np.nan
        return float((window.high.max() - window.low.min()) / entry_price)

    def close_position(minutes: int) -> float:
        window = clock_window(minutes)
        if window.empty:
            return np.nan
        low, high = float(window.low.min()), float(window.high.max())
        return float((entry_price - low) / (high - low)) if high > low else 0.5

    def trend_slope(minutes: int) -> float:
        window = clock_window(minutes)
        values = pd.to_numeric(window.close, errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & (values > 0)
        if valid.sum() < 5:
            return np.nan
        elapsed = (window.timestamp.astype("int64").to_numpy() - int(window.timestamp.iloc[0].value)) / float(pd.Timedelta(minutes=1).value)
        return float(np.polyfit(elapsed[valid], np.log(values[valid]), 1)[0])

    windows = {minutes: clock_window(minutes) for minutes in (1, 2, 5, 10, 30, 60)}
    returns = {minutes: _safe_returns(window) for minutes, window in windows.items()}
    mean30, mean60 = volume_mean(30), volume_mean(60)
    vwap_window = windows[30]
    weights = pd.to_numeric(vwap_window.volume, errors="coerce").fillna(0).to_numpy(float)
    closes = pd.to_numeric(vwap_window.close, errors="coerce").to_numpy(float)
    valid = np.isfinite(closes)
    if valid.any() and weights[valid].sum() > 0:
        vwap = float(np.average(closes[valid], weights=weights[valid]))
    elif valid.any():
        vwap = float(np.nanmean(closes[valid]))
    else:
        vwap = np.nan
    previous = completed.iloc[-1]
    text = (str(event.get("headline", "")) + " " + str(event.get("body", ""))).strip()
    feature: dict[str, Any] = {
        "event_id": str(event.event_id),
        "market": "US",
        "ticker": str(event.ticker),
        "event_time_utc": eutc,
        "source": str(event.get("source", "")),
        "source_family": "US_SEC",
        "form": str(event.get("form", "")),
        "event_type": str(event.get("event_type", "OTHER")),
        "headline": str(event.get("headline", "")),
        "body": str(event.get("body", "")),
        "text": text,
        "contract_version": "EXACT_T2_30M_V36_CAUSAL_FEATURE_ONLY",
        "price_source": "MASSIVE_1M_CAUSAL_PROJECTION",
        "price_cadence_sec": 60,
        "bar_timestamp_semantics": "OPEN",
        "entry_target_utc": entry_target,
        "actual_entry_time_utc": actual_entry,
        "entry_slippage_sec": entry_slippage,
        "entry_price": entry_price,
        "log_entry_price": float(np.log(entry_price)),
        "minutes_from_open": float((local - session_open).total_seconds() / 60),
        "minutes_to_close": float((session_close - local).total_seconds() / 60),
        "regular_session_eligible": True,
        **{f"pre_ret_{minutes}": ret_back(minutes) for minutes in (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120)},
        **{f"pre_vol_{minutes}": float(returns[minutes].std()) if len(returns[minutes]) >= 2 else np.nan for minutes in (5, 10, 30, 60)},
        "volume_ratio_1_30": float(volume_mean(1) / max(mean30, 1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_2_30": float(volume_mean(2) / max(mean30, 1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_5_30": float(volume_mean(5) / max(mean30, 1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_10_60": float(volume_mean(10) / max(mean60, 1.0)) if np.isfinite(mean60) else np.nan,
        **{f"range_{minutes}m": rolling_range(minutes) for minutes in (5, 10, 30, 60)},
        **{f"close_position_{minutes}": close_position(minutes) for minutes in (10, 30, 60)},
        "entry_bar_ret": float(previous.close / previous.open - 1),
        "entry_bar_range": float((previous.high - previous.low) / previous.close),
        "intraday_ret_open": float(entry_price / float(completed.iloc[0].open) - 1),
        "return_autocorr_30": float(returns[30].autocorr(1)) if len(returns[30]) >= 3 else np.nan,
        "up_fraction_10": float((returns[10] > 0).mean()) if len(returns[10]) else np.nan,
        "up_fraction_30": float((returns[30] > 0).mean()) if len(returns[30]) else np.nan,
        "trend_slope_30": trend_slope(30),
        "trend_slope_60": trend_slope(60),
        "vwap_distance_30": float(entry_price / vwap - 1) if np.isfinite(vwap) and vwap > 0 else np.nan,
        "event_positive_kw": app.count_kw(text, app.POSITIVE_KW),
        "event_negative_kw": app.count_kw(text, app.NEGATIVE_KW),
        "event_financing_kw": app.count_kw(text, app.FINANCING_KW),
        "event_trial_kw": app.count_kw(text, app.TRIAL_KW),
        "event_regulatory_kw": app.count_kw(text, app.REGULATORY_KW),
        "event_ma_kw": app.count_kw(text, app.MA_KW),
        "event_earnings_kw": app.count_kw(text, app.EARNINGS_KW),
        "headline_len": len(str(event.get("headline", ""))),
        "body_len": len(str(event.get("body", ""))),
    }
    clean_feature = clean_json(feature)
    projection = {
        "entry": clean_json(entry_candidates[["timestamp", "open"]].to_dict(orient="records")),
        "prehistory": clean_json(pre[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")),
    }
    guards = {
        "entry_projection_columns": ["timestamp", "open"],
        "prehistory_projection_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        "future_bar_rows_loaded": 0,
        "entry_candle_future_fields_loaded": False,
        "outcome_columns_loaded": False,
        "latest_prehistory_bar_utc": pd.Timestamp(pre.timestamp.max()).isoformat(),
        "decision_time_utc": actual_entry.isoformat(),
        "entry_slippage_sec": entry_slippage,
        "causal_bar_projection_sha256": locked.sha256_bytes(locked.canonical_json_bytes(projection)),
    }
    return clean_feature, "OK", guards


def create_or_load_features(root: Path, roles: pd.DataFrame, policy: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    artifacts: dict[int, dict[str, Any]] = {}
    reasons: dict[str, int] = {}
    for _, event in roles.iterrows():
        order = int(event.assignment_order)
        path = event_feature_path(root, order)
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            feature = artifact.get("feature_row", {})
            if (
                artifact.get("event_id") != str(event.event_id)
                or artifact.get("assignment_order") != order
                or OUTCOME_NAMES.intersection(feature)
                or artifact.get("feature_row_sha256") != locked.sha256_bytes(locked.canonical_json_bytes(feature))
            ):
                raise RuntimeError(f"TRACK_A_IMMUTABLE_FEATURE_INVALID:{order}")
            artifacts[order] = artifact
            continue
        local = pd.Timestamp(event.event_time_utc).tz_convert(app.US_TZ)
        price_path = resolve_price_path(root, policy, str(event.ticker), str(local.date()))
        if price_path is None:
            reasons["NO_PRICE_CACHE"] = reasons.get("NO_PRICE_CACHE", 0) + 1
            continue
        feature, reason, guards = causal_feature_snapshot(event, price_path)
        if reason != "OK":
            reasons[reason] = reasons.get(reason, 0) + 1
        if feature is None:
            continue
        artifact = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "artifact_role": "IMMUTABLE_CAUSAL_EVENT_FEATURE_SNAPSHOT",
            "assignment_order": order,
            "event_id": str(event.event_id),
            "feature_created_at_utc": utc_now(),
            "feature_row": feature,
            "feature_row_sha256": locked.sha256_bytes(locked.canonical_json_bytes(feature)),
            "price_path": str(price_path.relative_to(root)).replace("\\", "/"),
            "causal_guards": guards,
            "prediction_created": False,
            "outcome_opened": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
        }
        immutable_json(path, artifact)
        artifacts[order] = artifact
    return artifacts, reasons


def feature_block(root: Path, block: int, artifacts: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    start = (block - 1) * BLOCK_SIZE + 1
    orders = list(range(start, start + BLOCK_SIZE))
    if not all(order in artifacts for order in orders):
        return None
    paths = block_paths(root, block)
    frame = pd.DataFrame([artifacts[order]["feature_row"] for order in orders])
    # Event JSON is key-sorted on disk.  A canonical column order makes the
    # first in-memory run and every reload byte-identical.
    frame = frame.reindex(sorted(frame.columns), axis=1)
    feature_bytes = locked.deterministic_csv_gzip_bytes(frame)
    feature_sha = locked.write_immutable(paths["features"], feature_bytes)
    manifest = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "block": block,
        "rows": BLOCK_SIZE,
        "assignment_orders": orders,
        "event_ids": frame.event_id.astype(str).tolist(),
        "feature_file_sha256": feature_sha,
        "event_feature_file_sha256": {
            str(order): locked.sha256_file(event_feature_path(root, order)) for order in orders
        },
        "outcome_columns_present": False,
        "future_bar_rows_loaded": 0,
        "chronological": True,
    }
    immutable_json(paths["feature_manifest"], manifest)
    return frame, manifest


def score_block_durable(root: Path, block: int, features: pd.DataFrame, state: dict[str, Any], runtime: locked.FrozenRuntime) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = block_paths(root, block)
    forbidden = set(features.columns).intersection(runtime.scorer.FORBIDDEN_SCORE_COLUMNS | OUTCOME_NAMES)
    if forbidden:
        raise RuntimeError("TRACK_A_OUTCOME_AT_SCORE_TIME:" + ",".join(sorted(forbidden)))
    scored = runtime.scorer.score_block(features, runtime.bundle, state)
    prediction_bytes = locked.deterministic_csv_gzip_bytes(scored)
    prediction_sha = locked.sha256_bytes(prediction_bytes)
    if paths["predictions"].exists():
        if paths["predictions"].read_bytes() != prediction_bytes:
            raise RuntimeError(f"TRACK_A_PREDICTION_IMMUTABILITY_FAIL:{block}")
        manifest = json.loads(paths["prediction_manifest"].read_text(encoding="utf-8"))
        static = {
            "prediction_file_sha256": prediction_sha,
            "score_digest": runtime.scorer.score_digest(scored),
            "state_before_fingerprint": runtime.scorer.state_fingerprint(state),
        }
        if any(manifest.get(key) != value for key, value in static.items()):
            raise RuntimeError(f"TRACK_A_PREDICTION_MANIFEST_FAIL:{block}")
    else:
        if paths["prediction_manifest"].exists():
            raise RuntimeError(f"TRACK_A_INCOMPLETE_PREDICTION_PAIR:{block}")
        locked.write_immutable(paths["predictions"], prediction_bytes)
        manifest = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "artifact_role": "IMMUTABLE_20_ROW_PREDICTIONS_BEFORE_OUTCOMES",
            "block": block,
            "rows": BLOCK_SIZE,
            "prediction_created_at_utc": utc_now(),
            "prediction_file_sha256": prediction_sha,
            "score_digest": runtime.scorer.score_digest(scored),
            "state_before_fingerprint": runtime.scorer.state_fingerprint(state),
            "frozen_scorer_sha256": runtime.freeze_manifest["files"][locked.FROZEN_SCORER_NAME]["sha256"],
            "frozen_model_sha256": runtime.freeze_manifest["files"][locked.FROZEN_MODEL_NAME]["sha256"],
            "labels_or_returns_loaded_at_score_time": False,
            "all_20_predictions_durable_before_outcome_open": True,
        }
        immutable_json(paths["prediction_manifest"], manifest)
    for index, row in scored.iterrows():
        order = (block - 1) * BLOCK_SIZE + index + 1
        fields = {
            "event_id": str(row.event_id),
            "raw_probability": float(row.raw_probability),
            "final_probability": float(row.probability),
            "raw_direction": bool(row.raw_probability >= 0.5),
            "final_direction": bool(row.prediction),
            "polarity_reversed": bool(row.polarity_reversed),
            "direction_cutoff": float(row.direction_cutoff),
            "confidence_polarity": str(row.confidence_polarity),
            "confidence_threshold": float(row.confidence_threshold),
            "confidence_signal": float(row.confidence_signal),
            "high_conf": bool(row.high_conf),
        }
        audit = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "artifact_role": "IMMUTABLE_EVENT_PREDICTION_AUDIT",
            "assignment_order": order,
            "block": block,
            "event_id": str(row.event_id),
            "feature_file_sha256": locked.sha256_file(event_feature_path(root, order)),
            "feature_row_sha256": json.loads(event_feature_path(root, order).read_text(encoding="utf-8"))["feature_row_sha256"],
            "frozen_scorer_sha256": manifest["frozen_scorer_sha256"],
            "frozen_model_sha256": manifest["frozen_model_sha256"],
            "block_prediction_file_sha256": prediction_sha,
            "event_prediction_sha256": locked.sha256_bytes(locked.canonical_json_bytes(fields)),
            "prediction_created_at_utc": manifest["prediction_created_at_utc"],
            "outcome_opened_at_utc": None,
            **fields,
        }
        immutable_json(event_prediction_audit_path(root, order), audit)
    # Every event audit and the block pair must exist before outcome projection.
    start = (block - 1) * BLOCK_SIZE + 1
    if not all(event_prediction_audit_path(root, order).is_file() for order in range(start, start + BLOCK_SIZE)):
        raise RuntimeError(f"TRACK_A_EVENT_PREDICTION_AUDIT_INCOMPLETE:{block}")
    return scored, manifest


def open_exact_outcome(feature_artifact: dict[str, Any], price_path: Path) -> tuple[dict[str, Any] | None, str]:
    feature = feature_artifact["feature_row"]
    actual_entry = pd.Timestamp(feature["actual_entry_time_utc"])
    exit_target = actual_entry + pd.Timedelta(minutes=30)
    exit_limit = exit_target + pd.Timedelta(seconds=60)
    candidates = projected_parquet(
        price_path, ["timestamp", "open"],
        [("timestamp", ">=", exit_target), ("timestamp", "<=", exit_limit)],
    )
    if candidates.empty:
        return None, "EXIT_BAR_MISSING"
    row = candidates.iloc[0]
    actual_exit = pd.Timestamp(row.timestamp).tz_convert("UTC")
    exit_slippage = float((actual_exit - exit_target).total_seconds())
    hold = float((actual_exit - actual_entry).total_seconds())
    if not 0 <= exit_slippage <= 60:
        return None, "EXIT_TIMING_MISS"
    if not 1800 <= hold <= 1860:
        return None, "HOLD_TIMING_MISS"
    event_local, _, session_close = app.session_bounds(pd.Timestamp(feature["event_time_utc"]), "US")
    exit_local = actual_exit.tz_convert(app.US_TZ)
    if exit_local.date() != event_local.date() or exit_local > session_close:
        return None, "EXIT_CROSSES_SESSION"
    entry, exit_price = float(feature["entry_price"]), float(row.open)
    if not (np.isfinite(entry) and np.isfinite(exit_price) and entry > 0 and exit_price > 0):
        return None, "BAD_OUTCOME_OPEN"
    result = {
        "event_id": feature["event_id"],
        "entry_price": entry,
        "exit_target_utc": exit_target,
        "actual_exit_time_utc": actual_exit,
        "exit_slippage_sec": exit_slippage,
        "actual_hold_seconds": hold,
        "exit_price": exit_price,
        "fwd_ret_30m": float(exit_price / entry - 1),
        "y": int(exit_price > entry),
    }
    return clean_json(result), "OK"


def open_block_outcomes(root: Path, block: int, artifacts: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame | None, dict[str, int]]:
    paths = block_paths(root, block)
    if not paths["predictions"].is_file() or not paths["prediction_manifest"].is_file():
        raise RuntimeError(f"TRACK_A_OUTCOME_BEFORE_BLOCK_PREDICTION:{block}")
    prediction_manifest = json.loads(paths["prediction_manifest"].read_text(encoding="utf-8"))
    if locked.sha256_file(paths["predictions"]) != prediction_manifest["prediction_file_sha256"]:
        raise RuntimeError(f"TRACK_A_PREDICTION_NOT_DURABLE:{block}")
    start = (block - 1) * BLOCK_SIZE + 1
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for order in range(start, start + BLOCK_SIZE):
        outcome_path = event_outcome_path(root, order)
        if outcome_path.exists():
            audit = json.loads(outcome_path.read_text(encoding="utf-8"))
            rows.append(audit["outcome"])
            continue
        feature_artifact = artifacts[order]
        price_path = root / feature_artifact["price_path"]
        outcome, reason = open_exact_outcome(feature_artifact, price_path)
        if reason != "OK":
            reasons[reason] = reasons.get(reason, 0) + 1
        if outcome is None:
            continue
        prediction_audit = json.loads(event_prediction_audit_path(root, order).read_text(encoding="utf-8"))
        opened_at = utc_now()
        if pd.Timestamp(opened_at) < pd.Timestamp(prediction_audit["prediction_created_at_utc"]):
            raise RuntimeError(f"TRACK_A_TIMESTAMP_ORDER_FAIL:{order}")
        audit = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "artifact_role": "IMMUTABLE_EVENT_OUTCOME_OPEN_AUDIT",
            "assignment_order": order,
            "block": block,
            "event_id": outcome["event_id"],
            "feature_file_sha256": locked.sha256_file(event_feature_path(root, order)),
            "frozen_scorer_sha256": prediction_audit["frozen_scorer_sha256"],
            "block_prediction_file_sha256": prediction_audit["block_prediction_file_sha256"],
            "event_prediction_sha256": prediction_audit["event_prediction_sha256"],
            "feature_created_at_utc": feature_artifact["feature_created_at_utc"],
            "prediction_created_at_utc": prediction_audit["prediction_created_at_utc"],
            "outcome_opened_at_utc": opened_at,
            "outcome_opened_after_full_block_prediction_durable": True,
            "raw_price_file_sha256_at_outcome_open": locked.sha256_file(price_path),
            "outcome": outcome,
        }
        immutable_json(outcome_path, audit)
        rows.append(outcome)
    if len(rows) != BLOCK_SIZE:
        return None, reasons
    frame = pd.DataFrame(rows)
    expected_ids = [artifacts[order]["event_id"] for order in range(start, start + BLOCK_SIZE)]
    if frame.event_id.astype(str).tolist() != expected_ids:
        raise RuntimeError(f"TRACK_A_OUTCOME_ORDER_FAIL:{block}")
    return frame[["event_id", "y", "fwd_ret_30m"]], reasons


def complete_state(root: Path, block: int, state: dict[str, Any], scored: pd.DataFrame, outcomes: pd.DataFrame, runtime: locked.FrozenRuntime) -> dict[str, Any]:
    paths = block_paths(root, block)
    next_state = runtime.scorer.complete_block(state, scored, outcomes)
    expected_rows = block * BLOCK_SIZE
    if int(next_state["machine_state"]["confirmation_rows_completed"]) != expected_rows:
        raise RuntimeError(f"TRACK_A_STATE_COUNT_FAIL:{block}")
    state_bytes = runtime.scorer.canonical_json_bytes(next_state)
    state_sha = locked.write_immutable(paths["state"], state_bytes)
    completion = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "block": block,
        "cumulative_n": expected_rows,
        "state_before_fingerprint": runtime.scorer.state_fingerprint(state),
        "state_after_fingerprint": runtime.scorer.state_fingerprint(next_state),
        "state_file_sha256": state_sha,
        "event_prediction_audit_sha256": {
            str(order): locked.sha256_file(event_prediction_audit_path(root, order))
            for order in range((block - 1) * BLOCK_SIZE + 1, block * BLOCK_SIZE + 1)
        },
        "event_outcome_audit_sha256": {
            str(order): locked.sha256_file(event_outcome_path(root, order))
            for order in range((block - 1) * BLOCK_SIZE + 1, block * BLOCK_SIZE + 1)
        },
        "complete_block_called_only_after_20_outcomes": True,
        "prediction_before_outcomes": True,
        "state_append_only": True,
    }
    immutable_json(paths["completion"], completion)
    return next_state


def raw_vs_adapted_metrics(frame: pd.DataFrame, runtime: locked.FrozenRuntime) -> dict[str, Any]:
    adapted = frame.copy()
    raw = frame.copy()
    raw["probability"] = pd.to_numeric(raw.raw_probability, errors="raise")
    raw["prediction"] = raw.probability >= 0.5
    margin = (raw.probability - 0.5).abs()
    raw["confidence_signal"] = margin
    raw["high_conf"] = margin >= float(margin.quantile(0.8))
    raw_metrics = locked.metrics_for(raw, float(runtime.scorer.COST), include_bootstrap=True)
    adapted_metrics = locked.metrics_for(adapted, float(runtime.scorer.COST), include_bootstrap=True)
    deltas = {}
    for key in ("accuracy", "balanced_accuracy", "auc", "edge_vs_naive", "highconf_accuracy", "highconf_net"):
        left, right = raw_metrics.get(key), adapted_metrics.get(key)
        deltas[key] = None if left is None or right is None else float(right - left)
    return {"raw": raw_metrics, "adapted": adapted_metrics, "adapted_minus_raw": deltas}


def write_metrics(root: Path, n: int, combined: pd.DataFrame, state: dict[str, Any], runtime: locked.FrozenRuntime) -> dict[str, str]:
    comparison = raw_vs_adapted_metrics(combined, runtime)
    classification = locked.checkpoint_classification(n)
    metrics = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "cumulative_n": n,
        "comparison": comparison,
        "classification": classification,
        "state_fingerprint": runtime.scorer.state_fingerprint(state),
        "prediction_before_outcomes": True,
        "protocol_changed": False,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
    }
    directory = root / TRACK_REL
    metrics_path = directory / f"N{n:03d}_RAW_VS_ADAPTED_METRICS.json"
    metrics_sha = immutable_json(metrics_path, metrics)
    raw, adapted = comparison["raw"], comparison["adapted"]
    lines = [
        f"# V224 Track A Raw vs Adapted - N={n}", "",
        f"Status: `{classification['classification']}`", "",
        "| Metric | Raw | Adapted | Delta |", "|---|---:|---:|---:|",
    ]
    for key in ("accuracy", "balanced_accuracy", "auc", "edge_vs_naive", "highconf_accuracy", "highconf_net"):
        lines.append(f"| {key} | {locked._fmt(raw[key])} | {locked._fmt(adapted[key])} | {locked._fmt(comparison['adapted_minus_raw'][key])} |")
    lines.extend([
        "",
        f"Raw bootstrap BA lower95: {locked._fmt(raw['bootstrap']['balanced_accuracy_lower95'])}",
        f"Adapted bootstrap BA lower95: {locked._fmt(adapted['bootstrap']['balanced_accuracy_lower95'])}",
        f"Raw bootstrap HC net lower95: {locked._fmt(raw['bootstrap']['highconf_net_lower95'])}",
        f"Adapted bootstrap HC net lower95: {locked._fmt(adapted['bootstrap']['highconf_net_lower95'])}",
        "",
        "All 20 predictions were durable before any outcome in their block was opened.",
        "This research checkpoint has no protocol, Seal, Final Meta, or champion authority.", "",
    ])
    report_path = directory / f"N{n:03d}_RAW_VS_ADAPTED_REPORT.md"
    report_sha = locked.write_immutable(report_path, "\n".join(lines).encode("utf-8"))
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "cumulative_n": n,
        **classification,
        "metrics_sha256": metrics_sha,
        "report_sha256": report_sha,
        "prediction_before_outcomes": True,
        "protocol_changed": False,
    }
    status_sha = immutable_json(directory / f"N{n:03d}_STATUS.json", status)
    return {"metrics_sha256": metrics_sha, "report_sha256": report_sha, "status_sha256": status_sha}


def run(root: Path = ROOT, runtime: locked.FrozenRuntime | None = None) -> dict[str, Any]:
    policy, policy_sha = locked.verify_policy(root)
    frozen_manifest, _, _ = locked.verify_frozen_files(root)
    if runtime is None:
        runtime = locked.load_frozen_runtime(root)
    roles = load_roles(root, policy_sha)
    if roles.empty:
        status = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "status": "ZERO_PROSPECTIVE_ROWS_BOOTSTRAPPED",
            "role_rows": 0,
            "causal_feature_rows": 0,
            "prediction_rows": 0,
            "outcome_rows_opened": 0,
            "completed_rows": 0,
            "next_block_rows": 20,
            "one_shot": True,
            "poll_or_sleep": False,
            "old_outcome_first_watcher_used": False,
            "legacy_locked_evaluator_entrypoint_run_used": False,
            "shared_frozen_verification_metric_helpers_imported": True,
            "frozen_sha_manifest_sha256": locked.PINNED_FROZEN_SHA256,
            "prediction_before_outcomes": True,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
            "protocol_changed": False,
        }
        convergent_json(root / STATUS_REL, status)
        return status
    artifacts, feature_reasons = create_or_load_features(root, roles, policy)
    state = runtime.initial_state
    combined_frames: list[pd.DataFrame] = []
    report_hashes: dict[str, dict[str, str]] = {}
    outcome_reasons: dict[str, int] = {}
    completed_rows = 0
    predictions_durable = 0
    blocked_reason = None
    max_blocks = min(5, math.ceil(len(roles) / BLOCK_SIZE))
    for block in range(1, max_blocks + 1):
        block_feature = feature_block(root, block, artifacts)
        if block_feature is None:
            blocked_reason = f"BLOCK_{block:03d}_NEEDS_20_CONTIGUOUS_CAUSAL_FEATURES"
            break
        features, _ = block_feature
        scored, prediction_manifest = score_block_durable(root, block, features, state, runtime)
        predictions_durable = block * BLOCK_SIZE
        outcomes, reasons = open_block_outcomes(root, block, artifacts)
        for reason, count in reasons.items():
            outcome_reasons[reason] = outcome_reasons.get(reason, 0) + count
        if outcomes is None:
            blocked_reason = f"BLOCK_{block:03d}_PREDICTIONS_DURABLE_AWAITING_ALL_EXACT_OUTCOMES"
            break
        state = complete_state(root, block, state, scored, outcomes, runtime)
        merged = scored.merge(outcomes, on="event_id", how="left", validate="one_to_one")
        combined_frames.append(merged)
        completed_rows = block * BLOCK_SIZE
        combined = pd.concat(combined_frames, ignore_index=True, sort=False)
        report_hashes[str(completed_rows)] = write_metrics(root, completed_rows, combined, state, runtime)
    outcome_rows = sum(1 for order in range(1, len(roles) + 1) if event_outcome_path(root, order).is_file())
    if completed_rows == 100:
        status_name = "FINAL_PROSPECTIVE_CONFIRMATION_REPORT_READY"
    elif blocked_reason and "OUTCOMES" in blocked_reason:
        status_name = "PREDICTIONS_DURABLE_AWAITING_EXACT_OUTCOMES"
    elif blocked_reason:
        status_name = "CAUSAL_FEATURE_BLOCK_NOT_READY"
    else:
        status_name = "ONE_SHOT_COMPLETE"
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
        "status": status_name,
        "blocker": blocked_reason,
        "role_rows": int(len(roles)),
        "causal_feature_rows": int(len(artifacts)),
        "prediction_rows": predictions_durable,
        "outcome_rows_opened": outcome_rows,
        "completed_rows": completed_rows,
        "next_block_rows": next((value for value in CHECKPOINTS if value > completed_rows), None),
        "feature_reason_counts": feature_reasons,
        "outcome_reason_counts": outcome_reasons,
        "report_hashes": report_hashes,
        "one_shot": True,
        "poll_or_sleep": False,
        "old_outcome_first_watcher_used": False,
        "legacy_locked_evaluator_entrypoint_run_used": False,
        "shared_frozen_verification_metric_helpers_imported": True,
        "frozen_sha_manifest_sha256": locked.PINNED_FROZEN_SHA256,
        "freeze_set_sha256": frozen_manifest["freeze_set_sha256"],
        "prediction_before_outcomes": True,
        "state_append_only": True,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "protocol_changed": False,
        "protocol_change_authority": False,
    }
    convergent_json(root / STATUS_REL, status)
    return status


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        return int(subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT, env=environment, check=False).returncode)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    try:
        status = run(root)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
            "status": "FAILED_CLOSED",
            "blocker": f"{type(error).__name__}:{error}",
            "one_shot": True,
            "poll_or_sleep": False,
            "prediction_before_outcomes": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
            "protocol_changed": False,
        }
        convergent_json(root / TRACK_REL / "FAILED_CLOSED_STATUS.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
