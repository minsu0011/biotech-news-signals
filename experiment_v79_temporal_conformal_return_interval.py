"""V79 temporal split-conformal signed-return interval direction challenger.

This runner changes direction through calibrated return intervals.  It is not
V45 conditional-mean regression, V75 return-bin classification, or V76's
frozen-direction downside-quantile allocator.  A proper-fit chronological
slice trains fixed q20/q50/q80 models for raw 30-minute return.  A later, still
past-only calibration slice supplies interval nonconformity and median-bias
residuals.  The calibrated interval's signed center relative to its half-width
becomes P(up), with an optional fixed width-reliability blend.

Authority is immutable V36 DEV plus the atomic output_V69 commit.  Every outer,
inner, and proper-fit/calibration boundary has a strict 35-minute embargo and
event-group exclusion.  Inner-past OOF selects only among exact V69 and three
fixed architectures.  Outer outcomes are evaluation-only.  V69 confidence and
high-confidence membership remain frozen; material failure restores the exact
complete V69 frame.  Canonical 14-gate results are recomputed by the controller.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import autonomous_v37plus as controller
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
V69_COMMIT_PATH = V69_DIR / "COMMIT.json"
V69_MANIFEST_PATH = V69_DIR / "ARTIFACT_MANIFEST.json"
V69_ROBUSTNESS_PATH = V69_DIR / "DEV_ROBUSTNESS_REPORT.json"
OUT = Path(
    os.environ.get(
        "MARKET_BIO_VERSION_OUTPUT",
        str(ROOT / "staging" / "V79_TEMPORAL_CONFORMAL_RETURN_INTERVAL_V1"),
    )
)

VERSION = 79
HYPOTHESIS = "TEMPORAL_CONFORMAL_RETURN_INTERVAL_DIRECTION_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V69_PATH: "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
}
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_ROBUSTNESS_SHA256 = "778c5a87e4be328d702f85eb7298bcbb70e8b54d3429b19a0dc6e2c7e156cce1"
EXPECTED_DEV_ROWS = 9462
EXPECTED_OOF_ROWS = 7568

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
CALIBRATION_FRACTION = 0.25
QUANTILES = (0.20, 0.50, 0.80)
TARGET_INTERVAL_COVERAGE = 0.80
MIN_HALF_WIDTH = 0.002
SEED = 7901
BOOTSTRAP_DRAWS = 2000

NUMERIC_COLUMNS = (
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_ret_90", "pre_ret_120", "pre_vol_5", "pre_vol_10", "pre_vol_30",
    "pre_vol_60", "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "volume_ratio_10_60", "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "close_position_60", "entry_bar_ret",
    "entry_bar_range", "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
    "up_fraction_30", "trend_slope_30", "trend_slope_60", "vwap_distance_30",
    "log_entry_price", "minutes_from_open", "minutes_to_close", "event_positive_kw",
    "event_negative_kw", "event_financing_kw", "event_trial_kw", "event_regulatory_kw",
    "event_ma_kw", "event_earnings_kw", "headline_len", "body_len", "benchmark_ret_2",
    "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30", "benchmark_ret_60",
)
CATEGORICAL_COLUMNS = ("source_family", "event_type", "form")

ARCHITECTURES = (
    {"name": "CONFORMAL_INTERVAL_LOGIT_W25", "mode": "fixed", "weight": 0.25},
    {"name": "CONFORMAL_INTERVAL_LOGIT_W50", "mode": "fixed", "weight": 0.50},
    {"name": "CONFORMAL_WIDTH_RELIABILITY_W50", "mode": "reliability", "weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    raw = (json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(raw, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.name.endswith(".gz"):
        with temporary.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                compressed.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary.write_bytes(raw)
    os.replace(temporary, path)


def metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int | None]:
    y = frame.y.to_numpy(int)
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(prediction == y)),
        "pred_up": float(prediction.mean()),
    }


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V79 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V79 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    require(sha256(V69_COMMIT_PATH) == EXPECTED_V69_COMMIT_SHA256, "V69 COMMIT changed")
    require(sha256(V69_MANIFEST_PATH) == EXPECTED_V69_MANIFEST_SHA256, "V69 manifest changed")
    require(sha256(V69_ROBUSTNESS_PATH) == EXPECTED_V69_ROBUSTNESS_SHA256, "V69 robustness changed")
    commit = json.loads(V69_COMMIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(V69_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment mismatch")
    require(commit.get("manifest_sha256") == EXPECTED_V69_MANIFEST_SHA256, "V69 COMMIT binding mismatch")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment mismatch")
    files = manifest.get("files", {})
    require(files.get(V69_PATH.name, {}).get("sha256") == EXPECTED_SHA256[V69_PATH], "V69 OOF unbound")
    require(
        files.get(V69_ROBUSTNESS_PATH.name, {}).get("sha256") == EXPECTED_V69_ROBUSTNESS_SHA256,
        "V69 robustness unbound",
    )
    data = pd.read_csv(DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    champion = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(len(data) == EXPECTED_DEV_ROWS and data.event_id.is_unique, "immutable DEV changed")
    require(len(champion) == EXPECTED_OOF_ROWS and champion.event_id.is_unique, "V69 OOF changed")
    require(set(champion.event_id).issubset(set(data.event_id)), "V69 event outside immutable DEV")
    missing = [column for column in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS) if column not in data]
    require(not missing, f"immutable DEV missing conformal features: {missing}")
    require(np.isfinite(data.fwd_ret_30m.to_numpy(float)).all(), "non-finite return target")
    data["event_time_utc"] = pd.to_datetime(data.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    audit = {
        "input_sha256": hashes,
        "authorized_data_inputs": [str(DEV_PATH.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT))],
        "authorized_v69_integrity_inputs": [
            str(V69_COMMIT_PATH.relative_to(ROOT)),
            str(V69_MANIFEST_PATH.relative_to(ROOT)),
            str(V69_ROBUSTNESS_PATH.relative_to(ROOT)),
        ],
        "immutable_dev_only": True,
        "extension_data_opened": False,
        "assignment_file_opened": False,
        "reserved_evaluation_opened": False,
        "failed_v71_v78_results_opened": False,
        "dev_rows": len(data),
        "v69_oof_rows": len(champion),
        "v69_atomic_commit_verified": True,
        "return_target": "raw fwd_ret_30m",
        "direction_label_used_to_fit_quantile_models": False,
    }
    return data, champion, audit


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)].copy()
    output[list(NUMERIC_COLUMNS)] = (
        output[list(NUMERIC_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].fillna("").astype(str)
    return output


def make_quantile_model(quantile: float) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
            ),
        ]
    )
    processor = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=4.0,
        early_stopping=False,
        random_state=SEED + int(round(100 * quantile)),
    )
    return Pipeline([("features", processor), ("model", model)])


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) > 0 and len(valid) > 0, f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: 35-minute embargo failed")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label}: event-group overlap")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end.isoformat(),
        "valid_start": valid_start.isoformat(),
        "strict_35m_embargo": True,
        "event_group_overlap": 0,
    }


def proper_calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable")
    split = max(150, int(math.floor(len(ordered) * (1.0 - CALIBRATION_FRACTION))))
    require(split < len(ordered) - 40, "calibration slice too small")
    calibration = ordered.iloc[split:].copy()
    calibration_start = pd.Timestamp(calibration.event_time_utc.min())
    proper = ordered.loc[ordered.event_time_utc < calibration_start - EMBARGO].copy()
    calibration_groups = set(calibration.event_group_id.astype(str))
    proper = proper.loc[~proper.event_group_id.astype(str).isin(calibration_groups)].copy()
    require(len(proper) >= 150 and len(calibration) >= 40, "proper/calibration support insufficient")
    audit = chronology(proper, calibration, "proper-fit/calibration")
    audit.update(
        {
            "calibration_fraction_target": CALIBRATION_FRACTION,
            "proper_fit_labels_only": True,
            "calibration_labels_used_only_for_conformal_residual": True,
        }
    )
    return proper, calibration, audit


def higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def conformal_interval_probability(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    proper, calibration, split_audit = proper_calibration_split(train)
    proper_x = feature_frame(proper)
    calibration_x = feature_frame(calibration)
    target_x = feature_frame(target)
    calibration_predictions: list[np.ndarray] = []
    target_predictions: list[np.ndarray] = []
    iterations: dict[str, int] = {}
    for quantile in QUANTILES:
        model = make_quantile_model(quantile)
        model.fit(proper_x, proper.fwd_ret_30m.to_numpy(float))
        calibration_predictions.append(model.predict(calibration_x))
        target_predictions.append(model.predict(target_x))
        iterations[str(quantile)] = int(model.named_steps["model"].n_iter_)
    calibration_quantiles = np.sort(np.column_stack(calibration_predictions), axis=1)
    target_quantiles = np.sort(np.column_stack(target_predictions), axis=1)
    calibration_y = calibration.fwd_ret_30m.to_numpy(float)
    nonconformity = np.maximum.reduce(
        [calibration_quantiles[:, 0] - calibration_y,
         calibration_y - calibration_quantiles[:, 2],
         np.zeros(len(calibration), dtype=float)]
    )
    conformal_level = min(
        1.0,
        math.ceil((len(nonconformity) + 1) * TARGET_INTERVAL_COVERAGE) / len(nonconformity),
    )
    interval_adjustment = higher_quantile(nonconformity, conformal_level)
    median_bias = float(np.median(calibration_y - calibration_quantiles[:, 1]))
    calibrated_lower = target_quantiles[:, 0] - interval_adjustment
    calibrated_upper = target_quantiles[:, 2] + interval_adjustment
    calibrated_center = np.clip(
        target_quantiles[:, 1] + median_bias,
        calibrated_lower,
        calibrated_upper,
    )
    width = np.maximum(calibrated_upper - calibrated_lower, 2.0 * MIN_HALF_WIDTH)
    half_width = np.maximum(0.5 * width, MIN_HALF_WIDTH)
    interval_probability = expit(calibrated_center / half_width)
    calibration_lower = calibration_quantiles[:, 0] - interval_adjustment
    calibration_upper = calibration_quantiles[:, 2] + interval_adjustment
    raw_coverage = float(
        np.mean((calibration_y >= calibration_quantiles[:, 0]) & (calibration_y <= calibration_quantiles[:, 2]))
    )
    calibrated_coverage = float(
        np.mean((calibration_y >= calibration_lower) & (calibration_y <= calibration_upper))
    )
    calibration_width = np.maximum(calibration_upper - calibration_lower, 2.0 * MIN_HALF_WIDTH)
    reference_width = max(float(np.median(calibration_width)), 2.0 * MIN_HALF_WIDTH)
    coverage_reliability = max(
        0.10,
        1.0 - abs(calibrated_coverage - TARGET_INTERVAL_COVERAGE) / TARGET_INTERVAL_COVERAGE,
    )
    width_reliability = 1.0 / (1.0 + width / reference_width)
    reliability = np.clip(coverage_reliability * 2.0 * width_reliability, 0.10, 1.0)
    require(
        np.isfinite(interval_probability).all() and np.isfinite(reliability).all(),
        "conformal interval produced non-finite output",
    )
    return {
        "interval_probability": interval_probability,
        "reliability": reliability,
        "lower": calibrated_lower,
        "center": calibrated_center,
        "upper": calibrated_upper,
        "width": width,
    }, {
        "proper_calibration_chronology": split_audit,
        "quantiles": list(QUANTILES),
        "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
        "conformal_level": conformal_level,
        "interval_adjustment": interval_adjustment,
        "median_bias": median_bias,
        "raw_calibration_coverage": raw_coverage,
        "calibrated_calibration_coverage": calibrated_coverage,
        "calibration_n": len(calibration),
        "proper_fit_n": len(proper),
        "reference_interval_width": reference_width,
        "mean_target_interval_width": float(np.mean(width)),
        "mean_target_reliability": float(np.mean(reliability)),
        "model_iterations": iterations,
        "target_labels_used": False,
        "direction_labels_used_to_fit": False,
    }


def architecture_probability(
    baseline: np.ndarray,
    interval: dict[str, np.ndarray],
    architecture: dict[str, Any],
) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    interval_logit = logit(np.clip(interval["interval_probability"], 1e-5, 1.0 - 1e-5))
    if architecture["mode"] == "fixed":
        weight: float | np.ndarray = float(architecture["weight"])
    elif architecture["mode"] == "reliability":
        weight = float(architecture["weight"]) * interval["reliability"]
    else:
        raise ValueError(f"unknown architecture mode: {architecture['mode']}")
    probability = expit((1.0 - weight) * base_logit + weight * interval_logit)
    require(np.isfinite(probability).all(), "architecture returned non-finite probability")
    return probability


def inner_partition(
    data_market: pd.DataFrame,
    champion_market: pd.DataFrame,
    outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = champion_market.loc[champion_market.event_time_utc < outer_start - EMBARGO].copy()
    eligible = eligible.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(eligible) >= 250, "insufficient prior market OOF evidence")
    boundary = max(100, int(math.floor(len(eligible) * (1.0 - INNER_VALID_FRACTION))))
    valid_ids = set(eligible.iloc[boundary:].event_id)
    lookup = data_market.set_index("event_id")
    inner_valid = lookup.loc[list(valid_ids)].reset_index()
    inner_valid = inner_valid.sort_values(["event_time_utc", "event_id"], kind="stable")
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = data_market.loc[data_market.event_time_utc < inner_start - EMBARGO].copy()
    valid_groups = set(inner_valid.event_group_id.astype(str))
    inner_train = inner_train.loc[~inner_train.event_group_id.astype(str).isin(valid_groups)].copy()
    audit = chronology(inner_train, inner_valid, "inner")
    require(inner_valid.y.nunique() == 2, "inner valid is single-class")
    audit["inner_valid_fraction_target"] = INNER_VALID_FRACTION
    audit["inner_valid_is_prior_chronological_oof"] = True
    return inner_train, inner_valid, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    champion_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    baseline = champion_lookup.set_index("event_id").loc[inner_valid.event_id, "prob"].to_numpy(float)
    baseline_metrics = metric(inner_valid, baseline)
    interval, interval_audit = conformal_interval_probability(inner_train, inner_valid)
    trials: list[dict[str, Any]] = [
        {
            "name": "V69_NOOP",
            "architecture": None,
            "metrics": baseline_metrics,
            "auc_delta": 0.0,
            "balanced_accuracy_delta": 0.0,
            "eligible": True,
            "score": float(2.0 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]),
        }
    ]
    predictions: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        probability = architecture_probability(baseline, interval, architecture)
        predictions[architecture["name"]] = probability
        values = metric(inner_valid, probability)
        auc_delta = float(values["auc"] - baseline_metrics["auc"])
        ba_delta = float(values["balanced_accuracy"] - baseline_metrics["balanced_accuracy"])
        trials.append(
            {
                "name": architecture["name"],
                "architecture": architecture,
                "metrics": values,
                "auc_delta": auc_delta,
                "balanced_accuracy_delta": ba_delta,
                "eligible": ba_delta >= -0.010,
                "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
            }
        )
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past max 2*AUC+BA, BA delta >= -0.010; fixed interval blends only",
        "baseline": baseline_metrics,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "no_quantile_or_blend_micro_tuning": True,
    }, predictions, interval_audit


def run_nested(
    data: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v79_applied"] = False
    diagnostic["v79_architecture"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    stop = False
    for market in ("US", "KR"):
        data_market = data.loc[data.market.eq(market)].copy()
        champion_market = champion.loc[champion.market.eq(market)].copy()
        data_lookup = data_market.set_index("event_id")
        for fold in (2, 3, 4):
            outer_champion = champion_market.loc[champion_market.fold.eq(fold)].copy()
            outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
            require(len(outer_champion) >= 100, f"{market} fold {fold}: outer OOF support insufficient")
            outer_valid = data_lookup.loc[outer_champion.event_id].reset_index()
            outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
            outer_train = data_market.loc[data_market.event_time_utc < outer_start - EMBARGO].copy()
            valid_groups = set(outer_valid.event_group_id.astype(str))
            outer_train = outer_train.loc[~outer_train.event_group_id.astype(str).isin(valid_groups)].copy()
            outer_audit = chronology(outer_train, outer_valid, f"outer {market} fold {fold}")
            inner_train, inner_valid, inner_audit = inner_partition(
                data_market, champion_market, outer_start
            )
            policy, _, inner_interval_audit = choose_inner(inner_train, inner_valid, champion_market)
            selected = policy["selected"]
            baseline_outer = outer_champion.prob.to_numpy(float)
            outer_interval_audit: dict[str, Any] = {"skipped_for_exact_noop": True}
            interval: dict[str, np.ndarray] | None = None
            if selected["architecture"] is None:
                outer_probability = baseline_outer.copy()
            else:
                interval, outer_interval_audit = conformal_interval_probability(outer_train, outer_valid)
                outer_probability = architecture_probability(
                    baseline_outer, interval, selected["architecture"]
                )
            baseline_metrics = metric(outer_valid, baseline_outer)
            candidate_metrics = metric(outer_valid, outer_probability)
            probability_map = dict(zip(outer_valid.event_id, outer_probability))
            positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
            diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
            diagnostic.loc[positions, "v79_applied"] = selected["architecture"] is not None
            diagnostic.loc[positions, "v79_architecture"] = selected["name"]
            evidence = outer_champion[
                [
                    "event_id", "event_group_id", "event_time_utc", "market", "ticker",
                    "source_family", "fold", "y", "fwd_ret_30m", "high_conf",
                ]
            ].copy()
            evidence["baseline_prob"] = baseline_outer
            evidence["candidate_prob"] = outer_probability
            evidence["baseline_direction"] = baseline_outer >= 0.5
            evidence["candidate_direction"] = outer_probability >= 0.5
            evidence["v79_architecture"] = selected["name"]
            if interval is None:
                evidence["conformal_lower"] = np.nan
                evidence["conformal_center"] = np.nan
                evidence["conformal_upper"] = np.nan
                evidence["conformal_width"] = np.nan
                evidence["conformal_reliability"] = np.nan
            else:
                evidence["conformal_lower"] = interval["lower"]
                evidence["conformal_center"] = interval["center"]
                evidence["conformal_upper"] = interval["upper"]
                evidence["conformal_width"] = interval["width"]
                evidence["conformal_reliability"] = interval["reliability"]
            evidence_parts.append(evidence)
            outer_coverage = None
            if interval is not None:
                returns = outer_valid.fwd_ret_30m.to_numpy(float)
                outer_coverage = float(
                    np.mean((returns >= interval["lower"]) & (returns <= interval["upper"]))
                )
            audits.append(
                {
                    "market": market,
                    "fold": fold,
                    "outer_chronology": outer_audit,
                    "inner_chronology": inner_audit,
                    "policy": policy,
                    "inner_interval_audit": inner_interval_audit,
                    "outer_interval_audit": outer_interval_audit,
                    "outer_interval_coverage_evaluation_only": outer_coverage,
                    "outer_baseline": baseline_metrics,
                    "outer_candidate": candidate_metrics,
                    "outer_auc_delta": candidate_metrics["auc"] - baseline_metrics["auc"],
                    "outer_balanced_accuracy_delta": (
                        candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
                    ),
                    "policy_locked_before_outer_evaluation": True,
                    "outer_labels_used_for_selection": False,
                }
            )
            print(
                f"[V79] market={market} fold={fold} architecture={selected['name']} "
                f"inner_auc_delta={selected['auc_delta']:+.6f} "
                f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
                flush=True,
            )
            if smoke:
                stop = True
                break
        if stop:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_date_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["market", "fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few outer date blocks")
    generator = np.random.default_rng(SEED)
    auc_deltas: list[float] = []
    ba_deltas: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_deltas.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)))
        ba_deltas.append(
            float(
                balanced_accuracy_score(y, candidate >= 0.5)
                - balanced_accuracy_score(y, baseline >= 0.5)
            )
        )
    require(len(auc_deltas) >= int(0.95 * draws), "bootstrap lost too many samples")
    return {
        "method": "paired market/fold/date-block bootstrap over locked conformal architectures",
        "seed": SEED,
        "requested_draws": draws,
        "effective_draws": len(auc_deltas),
        "date_blocks": len(blocks),
        "auc_delta_probability_gt_zero": float(np.mean(np.asarray(auc_deltas) > 0.0)),
        "auc_delta_quantiles": {
            str(q): float(np.quantile(auc_deltas, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
        "balanced_accuracy_delta_probability_gt_zero": float(np.mean(np.asarray(ba_deltas) > 0.0)),
        "balanced_accuracy_delta_quantiles": {
            str(q): float(np.quantile(ba_deltas, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads(V69_ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_summary = v44.summarize(diagnostic)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V79 canonical gate mismatch")
    outer_baseline = metric(evidence, evidence.baseline_prob.to_numpy(float))
    outer_candidate = metric(evidence, evidence.candidate_prob.to_numpy(float))
    bootstrap = paired_date_bootstrap(evidence, draws)
    fold_auc_deltas = np.asarray([audit["outer_auc_delta"] for audit in audits], dtype=float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(
        champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)
    ) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool))
    chronology_ok = all(
        audit["outer_chronology"]["strict_35m_embargo"]
        and audit["inner_chronology"]["strict_35m_embargo"]
        and audit["inner_interval_audit"]["proper_calibration_chronology"]["strict_35m_embargo"]
        and (
            audit["outer_interval_audit"].get("skipped_for_exact_noop", False)
            or audit["outer_interval_audit"]["proper_calibration_chronology"]["strict_35m_embargo"]
        )
        and not audit["outer_labels_used_for_selection"]
        for audit in audits
    )
    material_checks = {
        "outer_evidence_auc_delta_gt_0_003": outer_candidate["auc"] - outer_baseline["auc"] > 0.003,
        "outer_evidence_balanced_accuracy_delta_gt_0": (
            outer_candidate["balanced_accuracy"] - outer_baseline["balanced_accuracy"] > 0.0
        ),
        "positive_outer_block_auc_fraction_ge_2_of_3": float(np.mean(fold_auc_deltas > 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta_probability_gt_zero"] >= 0.75,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": (
            candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001
        ),
        "hc_accuracy_delta_ge_minus_0_005": (
            candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005
        ),
        "hc_net_delta_ge_minus_0_0005": (
            candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005
        ),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_outer_inner_and_conformal_chronology": chronology_ok,
        "quantile_models_use_raw_return_not_direction_label": all(
            audit["inner_interval_audit"]["direction_labels_used_to_fit"] is False for audit in audits
        ),
        "outer_labels_evaluation_only": all(not audit["outer_labels_used_for_selection"] for audit in audits),
    }
    material_pass = bool(all(material_checks.values()))
    selected = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(selected_summary["research_gate"] == canonical_selected, "selected canonical gate mismatch")
    exact_entire_frame = selected.equals(champion)
    require(material_pass or exact_entire_frame, "V79 failure did not restore complete V69 frame")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_V69_NOOP",
        "material_pass": material_pass,
        "material_checks": material_checks,
        "outer_baseline": outer_baseline,
        "outer_candidate": outer_candidate,
        "outer_auc_delta": outer_candidate["auc"] - outer_baseline["auc"],
        "outer_balanced_accuracy_delta": (
            outer_candidate["balanced_accuracy"] - outer_baseline["balanced_accuracy"]
        ),
        "positive_outer_block_auc_fraction": float(np.mean(fold_auc_deltas > 0.0)),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "direction_change": {
            "probability_changed_n": int(
                np.sum(champion.prob.to_numpy(float) != diagnostic.prob.to_numpy(float))
            ),
            "direction_changed_n": int(
                np.sum(
                    (champion.prob.to_numpy(float) >= 0.5)
                    != (diagnostic.prob.to_numpy(float) >= 0.5)
                )
            ),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected.prob.to_numpy(float)),
            "explicit_direction_challenger": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact complete V69 frame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or exact_entire_frame),
        },
        "diagnostic_frame": diagnostic,
        "selected_frame": selected,
    }


def build_reports(
    champion: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected_contract = dict(evaluation["selected_summary"])
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
    }
    validation = (
        "Immutable V36 DEV and atomic V69 only; q20/q50/q80 raw-return models; chronological split conformal "
        "residual calibration; interval center/width direction probability; market-specific outer folds 2-4; "
        "strict 35-minute outer, inner, and proper/calibration embargoes; inner-only fixed architecture choice."
    )
    report = {
        "version": "V79",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "material_change": "temporal split-conformal raw-return interval converted to direction probability",
        "disjoint_from": {
            "V45": "not a conditional-mean signed-return regressor",
            "V75": "not a return-bin classifier",
            "V76": "changes direction from a two-sided calibrated interval; not frozen-direction downside allocation",
        },
        "quantiles": list(QUANTILES),
        "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
        "architectures": list(ARCHITECTURES),
        "fold_audits": audits,
        "input_audit": input_audit,
        "evaluation": compact,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V79",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "opened_dev_only": True,
        "reserved_outcomes_loaded": False,
        "selected": selected_contract,
        "candidate": evaluation["candidate_summary"],
        "baseline": evaluation["baseline_summary"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    gate_audit = {
        "version": "V79",
        "status": "MATCH",
        "reported": evaluation["selected_summary"]["research_gate"],
        "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        "material_gate": selected_contract["material_gate"],
    }
    model_comparison = {
        "version": "V79",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "models": {
            "V69_CHAMPION": evaluation["baseline_summary"],
            "V79_CONFORMAL_INTERVAL_DIAGNOSTIC": evaluation["candidate_summary"],
            "V79_FAIL_CLOSED_SELECTED": selected_contract,
        },
        "evaluation": compact,
        "input_audit": input_audit,
        "fold_audits": audits,
        "seal_authorized": False,
    }
    source_transfer = {
        "version": "V79",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
        "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
        "selected_by_market": selected_contract["metrics"]["by_market"],
        "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
        "outer": {
            "auc_delta": evaluation["outer_auc_delta"],
            "balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
            "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        },
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    fold_rows = []
    for audit in audits:
        chosen = audit["policy"]["selected"]
        fold_rows.append(
            {
                "market": audit["market"],
                "fold": audit["fold"],
                "outer_train_n": audit["outer_chronology"]["train_n"],
                "outer_valid_n": audit["outer_chronology"]["valid_n"],
                "inner_train_n": audit["inner_chronology"]["train_n"],
                "inner_valid_n": audit["inner_chronology"]["valid_n"],
                "selected_architecture": chosen["name"],
                "inner_auc_delta": chosen["auc_delta"],
                "inner_ba_delta": chosen["balanced_accuracy_delta"],
                "outer_auc_delta": audit["outer_auc_delta"],
                "outer_ba_delta": audit["outer_balanced_accuracy_delta"],
                "outer_interval_coverage": audit["outer_interval_coverage_evaluation_only"],
                "strict_35m_embargo": True,
                "outer_labels_used_for_selection": False,
            }
        )
    json_reports = {
        "MODEL_COMPARISON.json": model_comparison,
        "V79_TEMPORAL_CONFORMAL_INTERVAL_REPORT.json": report,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": source_transfer,
        "CANONICAL_RESEARCH_GATE_AUDIT.json": gate_audit,
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": evaluation["bootstrap"],
    }
    selected_columns = list(champion.columns)
    table_reports = {
        "V79_CONFORMAL_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "V79_CONFORMAL_DIRECTION_EVIDENCE_OOF.csv.gz": evidence,
        "V79_DIAGNOSTIC_CHALLENGER_OOF.csv.gz": evaluation["diagnostic_frame"],
        "V79_FAIL_CLOSED_SELECTED_OOF.csv.gz": evaluation["selected_frame"][selected_columns],
    }
    return json_reports, table_reports


def write_outputs(
    json_reports: dict[str, Any],
    table_reports: dict[str, pd.DataFrame],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in json_reports.items():
        atomic_json(payload, OUT / name)
    for name, frame in table_reports.items():
        atomic_csv(frame, OUT / name)
    material_spec = {
        "hypothesis": HYPOTHESIS,
        "quantiles": QUANTILES,
        "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
        "calibration_fraction": CALIBRATION_FRACTION,
        "architectures": ARCHITECTURES,
        "embargo_seconds": int(EMBARGO.total_seconds()),
        "validation": "nested temporal split-conformal OOF",
    }
    experiment_id = hashlib.sha256(
        (
            "|".join(input_audit["input_sha256"].values())
            + "|"
            + json.dumps(material_spec, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    status = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "phase": (
            "ROBUST_SURVIVOR"
            if evaluation["selected_summary"]["research_gate"]["robust_survivor"]
            else "RESEARCH_FAIL"
        ),
        "canonical_gate_passed": evaluation["selected_summary"]["research_gate"]["passed"],
        "canonical_gate_total": evaluation["selected_summary"]["research_gate"]["total"],
        "material_pass": evaluation["material_pass"],
        "champion_changed": evaluation["material_pass"],
        "completed_at": now(),
        "experiment_id": experiment_id,
        "reserved_state": "UNOPENED",
        "seal_authorized": False,
        "output_scope": str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT),
    }
    atomic_json(status, OUT / "RUN_STATUS.json")
    files: dict[str, Any] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"version": VERSION, "hypothesis": HYPOTHESIS, "experiment_id": experiment_id, "files": files}
    atomic_json(manifest, OUT / "ARTIFACT_MANIFEST.json")
    commit = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "committed_at": now(),
        "experiment_id": experiment_id,
        "input_sha256": input_audit["input_sha256"],
        "material_fingerprint": hashlib.sha256(
            json.dumps(material_spec, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
    }
    atomic_json(commit, OUT / "COMMIT.json")
    return {"experiment_id": experiment_id, "artifact_count": len(files) + 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    data, champion, input_audit = verify_and_load()
    if args.audit_only:
        print(
            json.dumps(
                clean(
                    {
                        "status": "AUDIT_OK",
                        "hypothesis": HYPOTHESIS,
                        "input_audit": input_audit,
                        "quantiles": list(QUANTILES),
                        "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
                        "calibration_fraction": CALIBRATION_FRACTION,
                        "architectures": list(ARCHITECTURES),
                        "output_written": False,
                    }
                ),
                indent=2,
            )
        )
        return
    diagnostic, evidence, audits = run_nested(data, champion, smoke=args.smoke_test)
    evaluation = evaluate(
        champion,
        diagnostic,
        evidence,
        audits,
        draws=250 if args.smoke_test else BOOTSTRAP_DRAWS,
    )
    summary = {
        "status": evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "material_pass": evaluation["material_pass"],
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "inner_trials": [audit["policy"]["trials"] for audit in audits],
        "interval_audits": [
            {
                "market": audit["market"],
                "fold": audit["fold"],
                "inner": audit["inner_interval_audit"],
                "outer": audit["outer_interval_audit"],
                "outer_coverage": audit["outer_interval_coverage_evaluation_only"],
            }
            for audit in audits
        ],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    if args.smoke_test:
        print(json.dumps(clean({**summary, "status": "SMOKE_OK", "output_written": False}), indent=2))
        return
    json_reports, table_reports = build_reports(champion, evidence, audits, evaluation, input_audit)
    if args.dry_run:
        print(json.dumps(clean({**summary, "status": "DRY_RUN_OK", "output_written": False}), indent=2))
        return
    write_result = write_outputs(json_reports, table_reports, evaluation, input_audit)
    print(json.dumps(clean({**summary, "output_dir": str(OUT), "write": write_result}), indent=2))


if __name__ == "__main__":
    main()
