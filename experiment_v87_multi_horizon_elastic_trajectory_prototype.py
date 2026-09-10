"""V87 multi-horizon elastic trajectory-prototype direction challenger.

The material hypothesis is that the *shape across causal lookback horizons*,
rather than an unordered scalar-state density, carries direction information.
For every past-only training cut, an earlier proper-fit segment supplies robust
channel scaling and duplicate-weighted up/down median trajectory prototypes.
Rigid profile distance and radius-one Sakoe-Chiba dynamic-time-warping distance
produce two direction margins.  A later, separately embargoed past segment fits
only one-dimensional Platt maps.  Thus target rows never define scales,
prototypes, distance calibration, policies, or preprocessing.

Only immutable V36 DEV and atomic output_V69 are authoritative.  Folds 2--4
use nested expanding chronology, strict 35-minute embargoes, event/duplicate
purges, inner-only selection among exact V69 and three fixed trajectory blends,
and outer evaluation-only outcomes.  V69 confidence/high-confidence membership
remain exact.  Material failure restores the exact entire V69 frame, and the
controller's canonical 14 checks are recomputed from raw selected metrics.
Preparation permits authority audit and one bounded single-fold smoke only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import autonomous_v37plus as controller
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V87_MULTI_HORIZON_ELASTIC_TRAJECTORY_V1"

VERSION = 87
HYPOTHESIS = "MULTI_HORIZON_ELASTIC_TRAJECTORY_PROTOTYPE_DIRECTION_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_OOF_SHA256 = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_DEV_ROWS = 9462
EXPECTED_V69_ROWS = 7568

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
PROPER_FIT_FRACTION = 0.70
DTW_RADIUS = 1
PLATT_C = 1.0
PROFILE_BLEND_WEIGHT = 0.30
CONSENSUS_BLEND_WEIGHT = 0.50
COST = 0.002
SEED = 8701
BOOTSTRAP_DRAWS = 2000

# Every field is a pre-decision market-state measurement.  Columns remain in
# chronological horizon order inside each channel.  No source/category/text,
# issuer/ticker, event semantics, historical response, or target feature exists.
TRAJECTORY_CHANNELS = {
    "RETURN": (
        "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
        "pre_ret_90", "pre_ret_120",
    ),
    "VOLATILITY": ("pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60"),
    "VOLUME": (
        "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
        "volume_ratio_10_60",
    ),
    "RANGE": ("range_5m", "range_10m", "range_30m", "range_60m"),
    "CLOSE_POSITION": ("close_position_10", "close_position_30", "close_position_60"),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
    "TREND": ("trend_slope_30", "trend_slope_60"),
}
NUMERIC_COLUMNS = tuple(column for columns in TRAJECTORY_CHANNELS.values() for column in columns)

POLICIES = (
    {"name": "RIGID_TRAJECTORY_W0.30", "mode": "rigid", "weight": PROFILE_BLEND_WEIGHT},
    {"name": "ELASTIC_DTW_W0.30", "mode": "elastic", "weight": PROFILE_BLEND_WEIGHT},
    {"name": "RIGID_DTW_CONSENSUS_W0.50", "mode": "consensus", "weight": CONSENSUS_BLEND_WEIGHT},
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
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
    atomic_bytes(
        (json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        path,
    )


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else None


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: float) -> np.ndarray:
    base = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    challenge = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    return expit((1.0 - float(weight)) * base + float(weight) * challenge)


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "V36 authority changed")
    require(sha256(commit_path) == EXPECTED_V69_COMMIT_SHA256, "V69 COMMIT changed")
    require(sha256(manifest_path) == EXPECTED_V69_MANIFEST_SHA256, "V69 manifest changed")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == EXPECTED_V69_MANIFEST_SHA256, "V69 binding changed")
    require(commit.get("data_sha") == EXPECTED_V36_SHA256, "V69/V36 binding changed")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest identity changed")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 file missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 size mismatch: {name}")
        require(sha256(path) == item["sha256"], f"V69 hash mismatch: {name}")
    require(
        manifest["files"].get(V69_PATH.name, {}).get("sha256") == EXPECTED_V69_OOF_SHA256,
        "V69 selected OOF binding changed",
    )
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
        ],
        "v36_dev_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": EXPECTED_V69_COMMIT_SHA256,
        "v69_manifest_sha256": EXPECTED_V69_MANIFEST_SHA256,
        "v69_selected_oof_sha256": EXPECTED_V69_OOF_SHA256,
        "v69_experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "manifest_files_verified": len(manifest.get("files", {})),
    }


def prepare_dev(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["event_time_utc"] = pd.to_datetime(output.event_time_utc, utc=True)
    output["headline"] = output.headline.fillna("").astype(str)
    output["body"] = output.body.fillna("").astype(str)
    output["text_cluster_id"] = output.apply(v37.normalized_cluster, axis=1)
    output["cluster_weight"] = 1.0 / output.groupby("text_cluster_id").event_id.transform("size")
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return output.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    champion = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_DEV_ROWS and len(champion) == EXPECTED_V69_ROWS, "row-count authority changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 event outside V36")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(
        np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15),
        "V36/V69 returns differ",
    )
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    return prepare_dev(dev), champion


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: embargo failed")
    require(not (set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))), f"{label}: group leakage")
    require(not (set(train.text_cluster_id.astype(str)) & set(valid.text_cluster_id.astype(str))), f"{label}: duplicate leakage")
    return {
        "train_n": len(train), "valid_n": len(valid), "train_end": train_end,
        "valid_start": valid_start, "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0, "duplicate_cluster_overlap_n": 0,
        "strict_35m_embargo": True,
    }


def leakage_filtered_train(dev: pd.DataFrame, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = dev.loc[dev.event_time_utc < boundary - EMBARGO].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    train = train.loc[~train.text_cluster_id.astype(str).isin(set(valid.text_cluster_id.astype(str)))].copy()
    return train


def aligned_dev(dev: pd.DataFrame, champion_rows: pd.DataFrame) -> pd.DataFrame:
    aligned = dev.set_index("event_id").loc[champion_rows.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion_rows.y.to_numpy(int)), "aligned labels differ")
    return aligned


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.event_time_utc < outer_start - EMBARGO].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= 1000, "insufficient prior V69 OOF rows")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_train = leakage_filtered_train(dev, pd.Timestamp(inner_valid.event_time_utc.min()), inner_valid)
    require(len(inner_train) >= 700, "inner training cut too small")
    audit = chronology(inner_train, inner_valid, "V87 inner")
    audit["validation_source"] = "earlier committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def internal_proper_calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable")
    split = int(math.floor(len(ordered) * PROPER_FIT_FRACTION))
    require(split >= 700 and len(ordered) - split >= 250, "internal trajectory split too small")
    calibration = ordered.iloc[split:].copy()
    proper = leakage_filtered_train(ordered, pd.Timestamp(calibration.event_time_utc.min()), calibration)
    require(len(proper) >= 650, "proper prototype segment too small after purge")
    audit = chronology(proper, calibration, "V87 proper-to-calibration")
    audit["proper_fit_fraction_requested"] = PROPER_FIT_FRACTION
    audit["prototype_labels_from_proper_only"] = True
    audit["platt_labels_from_later_calibration_only"] = True
    return proper, calibration, audit


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * sorted_weights.sum()
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


class TrajectoryState:
    def __init__(self) -> None:
        self.median: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "TrajectoryState":
        for channel, columns in TRAJECTORY_CHANNELS.items():
            values = frame.loc[:, columns].to_numpy(float)
            median = np.nanmedian(values, axis=0)
            median = np.where(np.isfinite(median), median, 0.0)
            filled = np.where(np.isfinite(values), values, median)
            q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
            standard = np.std(filled, axis=0)
            scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
            self.median[channel] = median
            self.scale[channel] = scale
        require(len(self.median) == len(TRAJECTORY_CHANNELS), "trajectory scaler fit failed")
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        output = {}
        for channel, columns in TRAJECTORY_CHANNELS.items():
            values = frame.loc[:, columns].to_numpy(float)
            filled = np.where(np.isfinite(values), values, self.median[channel])
            state = np.clip((filled - self.median[channel]) / self.scale[channel], -8.0, 8.0)
            require(np.isfinite(state).all(), f"non-finite trajectory channel: {channel}")
            output[channel] = state
        return output


def class_prototypes(
    state: dict[str, np.ndarray], labels: np.ndarray, weights: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {0: {}, 1: {}}
    for label in (0, 1):
        mask = np.asarray(labels, int) == label
        require(mask.sum() >= 200, f"too few proper rows for class {label}")
        for channel, values in state.items():
            result[label][channel] = np.asarray([
                weighted_median(values[mask, position], weights[mask])
                for position in range(values.shape[1])
            ])
    return result


def rigid_distance(path: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(path, float) - np.asarray(prototype, float))))


def dtw_distance(path: np.ndarray, prototype: np.ndarray, radius: int = DTW_RADIUS) -> float:
    path = np.asarray(path, float)
    prototype = np.asarray(prototype, float)
    n, m = len(path), len(prototype)
    radius = max(int(radius), abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lower = max(1, i - radius)
        upper = min(m, i + radius)
        for j in range(lower, upper + 1):
            local = abs(path[i - 1] - prototype[j - 1])
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    require(math.isfinite(cost[n, m]), "DTW path unreachable")
    return float(cost[n, m] / max(n + m, 1))


def distance_margin(
    state: dict[str, np.ndarray], prototypes: dict[int, dict[str, np.ndarray]], method: str,
) -> np.ndarray:
    require(method in {"rigid", "elastic"}, "unknown trajectory distance")
    n = len(next(iter(state.values())))
    down = np.zeros(n, dtype=float)
    up = np.zeros(n, dtype=float)
    distance_function = rigid_distance if method == "rigid" else dtw_distance
    for channel, values in state.items():
        down += np.asarray([distance_function(row, prototypes[0][channel]) for row in values])
        up += np.asarray([distance_function(row, prototypes[1][channel]) for row in values])
    margin = (down - up) / len(TRAJECTORY_CHANNELS)
    require(np.isfinite(margin).all() and float(np.std(margin)) > 1e-10, f"degenerate {method} margin")
    return margin


def fit_platt(score: np.ndarray, calibration: pd.DataFrame, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        C=PLATT_C, penalty="l2", solver="lbfgs", max_iter=300,
        tol=1e-8, random_state=seed,
    )
    model.fit(
        np.asarray(score, float).reshape(-1, 1), calibration.y.to_numpy(int),
        sample_weight=calibration.cluster_weight.to_numpy(float),
    )
    require(int(model.n_iter_[0]) < model.max_iter, "Platt mapping did not converge")
    return model


def fit_trajectory_prototype(
    train: pd.DataFrame, target: pd.DataFrame, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    proper, calibration, internal_audit = internal_proper_calibration_split(train)
    transform = TrajectoryState().fit(proper)
    proper_state = transform.transform(proper)
    calibration_state = transform.transform(calibration)
    target_state = transform.transform(target)
    prototypes = class_prototypes(
        proper_state, proper.y.to_numpy(int), proper.cluster_weight.to_numpy(float)
    )
    calibration_rigid = distance_margin(calibration_state, prototypes, "rigid")
    calibration_elastic = distance_margin(calibration_state, prototypes, "elastic")
    target_rigid = distance_margin(target_state, prototypes, "rigid")
    target_elastic = distance_margin(target_state, prototypes, "elastic")
    rigid_platt = fit_platt(calibration_rigid, calibration, seed)
    elastic_platt = fit_platt(calibration_elastic, calibration, seed + 1)
    rigid_probability = rigid_platt.predict_proba(target_rigid.reshape(-1, 1))[:, 1]
    elastic_probability = elastic_platt.predict_proba(target_elastic.reshape(-1, 1))[:, 1]
    consensus_probability = expit(
        0.5 * logit(np.clip(rigid_probability, 1e-6, 1.0 - 1e-6))
        + 0.5 * logit(np.clip(elastic_probability, 1e-6, 1.0 - 1e-6))
    )
    disagreement = np.abs(rigid_probability - elastic_probability)
    predictions = {
        "rigid_probability": np.clip(rigid_probability, 1e-6, 1.0 - 1e-6),
        "elastic_probability": np.clip(elastic_probability, 1e-6, 1.0 - 1e-6),
        "consensus_probability": np.clip(consensus_probability, 1e-6, 1.0 - 1e-6),
        "rigid_margin": target_rigid,
        "elastic_margin": target_elastic,
        "method_disagreement": disagreement,
    }
    audit = {
        "architecture": "proper-past weighted-median class trajectory prototypes plus later-past Platt maps",
        "channels": {name: list(columns) for name, columns in TRAJECTORY_CHANNELS.items()},
        "raw_causal_numeric_feature_n": len(NUMERIC_COLUMNS),
        "distance_methods": ["rigid mean absolute profile", "radius-one Sakoe-Chiba DTW"],
        "dtw_radius": DTW_RADIUS,
        "prototype": "duplicate-weighted coordinate median by direction class",
        "internal_chronology": internal_audit,
        "proper_fit_n": len(proper), "calibration_n": len(calibration), "target_n": len(target),
        "proper_class_counts": proper.y.value_counts().sort_index().to_dict(),
        "calibration_class_counts": calibration.y.value_counts().sort_index().to_dict(),
        "rigid_platt_coefficient": float(rigid_platt.coef_[0, 0]),
        "rigid_platt_intercept": float(rigid_platt.intercept_[0]),
        "rigid_platt_iterations": int(rigid_platt.n_iter_[0]),
        "elastic_platt_coefficient": float(elastic_platt.coef_[0, 0]),
        "elastic_platt_intercept": float(elastic_platt.intercept_[0]),
        "elastic_platt_iterations": int(elastic_platt.n_iter_[0]),
        "target_rigid_margin_quantiles": {
            str(q): float(np.quantile(target_rigid, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_elastic_margin_quantiles": {
            str(q): float(np.quantile(target_elastic, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_method_disagreement_quantiles": {
            str(q): float(np.quantile(disagreement, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_rows_used_for_scaling_prototypes_or_platt": False,
        "source_category_issuer_ticker_text_used": False,
        "kernel_or_covariance_density_used": False,
        "event_neighbor_retrieval_used": False,
        "device": "CPU", "gpu_used": False,
        "cpu_reason": "bounded deterministic median prototypes and short radius-one dynamic programs",
    }
    return predictions, audit


def metric(
    frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray, high: np.ndarray,
) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    high = np.asarray(high, bool)
    correct = (prediction == y).astype(int)
    net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return {
        "n": len(frame), "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(correct.mean()), "pred_up": float(prediction.mean()),
        "all_trade_mean_signed_net": float(net.mean()),
        "confidence_correctness_auc": safe_auc(correct, np.asarray(confidence, float)),
        "highconf_n": int(high.sum()),
        "highconf_accuracy": float(correct[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(net[high].mean()) if high.any() else None,
    }


def apply_policy(
    baseline: np.ndarray, predictions: dict[str, np.ndarray], policy: dict[str, Any],
) -> np.ndarray:
    if policy["mode"] == "noop":
        return np.asarray(baseline, float).copy()
    key = {
        "rigid": "rigid_probability",
        "elastic": "elastic_probability",
        "consensus": "consensus_probability",
    }[policy["mode"]]
    return probability_blend(baseline, predictions[key], float(policy["weight"]))


def choose_inner(
    train: pd.DataFrame, valid: pd.DataFrame, champion_valid: pd.DataFrame, fold: int,
) -> dict[str, Any]:
    predictions, model_audit = fit_trajectory_prototype(train, valid, SEED + fold * 10)
    baseline = champion_valid.prob.to_numpy(float)
    confidence = champion_valid.confidence_signal.to_numpy(float)
    high = bool_series(champion_valid.high_conf).to_numpy(bool)
    baseline_metric = metric(valid, baseline, confidence, high)
    noop = {"name": "V69_NOOP", "mode": "noop", "weight": 0.0}
    trials = []
    for policy in (noop, *POLICIES):
        probability = apply_policy(baseline, predictions, policy)
        values = metric(valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            **policy, "metrics": values,
            "auc_delta": auc_delta, "ba_delta": ba_delta, "net_delta": net_delta,
            "eligible": bool(policy["mode"] == "noop" or (ba_delta >= -0.005 and net_delta >= -0.0005)),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["ba_delta"], -trial["weight"]),
    )
    return {
        "selection_rule": "maximize 2*AUC+BA on inner-past OOF; BA delta>=-0.005 and net delta>=-0.0005; exact no-op plus three fixed trajectory policies only",
        "selected": selected, "trials": trials,
        "trajectory_model_audit": model_audit,
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v87_model"] = "V69_NOOP"
    diagnostic["v87_blend_weight"] = 0.0
    evidence_parts = []
    audits = []
    for fold in (2, 3, 4):
        outer_champion = champion.loc[champion.fold.eq(fold)].copy().sort_values(
            ["event_time_utc", "event_id"], kind="stable"
        )
        require(len(outer_champion) == 1892, f"fold {fold} count changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_train = leakage_filtered_train(dev, pd.Timestamp(outer_valid.event_time_utc.min()), outer_valid)
        require(len(outer_train) >= 2500, "outer training cut too small")
        outer_audit = chronology(outer_train, outer_valid, f"V87 outer fold {fold}")
        inner_train, inner_valid, inner_champion, inner_audit = inner_partition(
            dev, champion, pd.Timestamp(outer_valid.event_time_utc.min())
        )
        policy = choose_inner(inner_train, inner_valid, inner_champion, fold)
        selected = policy["selected"]
        predictions, outer_model_audit = fit_trajectory_prototype(
            outer_train, outer_valid, SEED + 1000 + fold
        )
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        candidate = apply_policy(baseline, predictions, selected)
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        probability_map = dict(zip(outer_valid.event_id, candidate))
        positions = diagnostic.index[diagnostic.event_id.isin(probability_map)]
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v87_model"] = selected["name"]
        diagnostic.loc[positions, "v87_blend_weight"] = float(selected["weight"])
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["rigid_trajectory_prob"] = predictions["rigid_probability"]
        evidence["elastic_dtw_prob"] = predictions["elastic_probability"]
        evidence["trajectory_consensus_prob"] = predictions["consensus_probability"]
        evidence["rigid_margin"] = predictions["rigid_margin"]
        evidence["elastic_margin"] = predictions["elastic_margin"]
        evidence["method_disagreement"] = predictions["method_disagreement"]
        evidence["candidate_prob"] = candidate
        evidence["v87_model"] = selected["name"]
        evidence["v87_blend_weight"] = float(selected["weight"])
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold, "inner_chronology": inner_audit, "outer_chronology": outer_audit,
            "policy": policy, "outer_trajectory_model_audit": outer_model_audit,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "outer_mean_method_disagreement": float(np.mean(predictions["method_disagreement"])),
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V87 TRAJECTORY] fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f} "
            f"method_disagreement={audits[-1]['outer_mean_method_disagreement']:.4f}",
            flush=True,
        )
        if smoke:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few bootstrap blocks")
    rng = np.random.default_rng(SEED + 99)
    auc_values = []
    net_values = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        returns = sample.fwd_ret_30m.to_numpy(float)
        auc_values.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        base_net = np.where(base >= 0.5, 1.0, -1.0) * returns - COST
        candidate_net = np.where(candidate >= 0.5, 1.0, -1.0) * returns - COST
        net_values.append(float(candidate_net.mean() - base_net.mean()))

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        require(len(array) > 0, "empty bootstrap")
        return {
            "effective_draws": len(array), "probability_gt_zero": float(np.mean(array > 0)),
            "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
        }
    return {
        "method": "paired fold/date block bootstrap over locked V87 trajectory policies",
        "auc_delta": interval(auc_values), "net_delta": interval(net_values),
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], smoke: bool,
) -> dict[str, Any]:
    baseline_summary = v44.summarize(champion)
    candidate_summary = v44.summarize(diagnostic)
    canonical_base = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_base["total"] == 14 and canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_base, "baseline canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "candidate canonical mismatch")
    base_simple = {
        "auc": float(roc_auc_score(evidence.y, evidence.baseline_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(evidence.y, evidence.baseline_prob >= 0.5)),
    }
    candidate_simple = {
        "auc": float(roc_auc_score(evidence.y, evidence.candidate_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(evidence.y, evidence.candidate_prob >= 0.5)),
    }
    returns = evidence.fwd_ret_30m.to_numpy(float)
    base_net = np.where(evidence.baseline_prob.to_numpy(float) >= 0.5, 1.0, -1.0) * returns - COST
    candidate_net = np.where(evidence.candidate_prob.to_numpy(float) >= 0.5, 1.0, -1.0) * returns - COST
    bootstrap = None if smoke else paired_bootstrap(evidence, BOOTSTRAP_DRAWS)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits])
    fold_ba = np.asarray([audit["outer_ba_delta"] for audit in audits])
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits])
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float))
        and np.array_equal(bool_series(champion.high_conf).to_numpy(bool), bool_series(diagnostic.high_conf).to_numpy(bool))
    )
    model_audits = [audit["policy"]["trajectory_model_audit"] for audit in audits] + [
        audit["outer_trajectory_model_audit"] for audit in audits
    ]
    checks = {
        "outer_auc_delta_gt_0_003": candidate_simple["auc"] - base_simple["auc"] > 0.003,
        "outer_ba_delta_gt_0": candidate_simple["balanced_accuracy"] - base_simple["balanced_accuracy"] > 0.0,
        "outer_net_delta_gt_0": float(candidate_net.mean() - base_net.mean()) > 0.0,
        "positive_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2 / 3,
        "nonnegative_fold_ba_fraction_ge_2_of_3": float(np.mean(fold_ba >= 0.0)) >= 2 / 3,
        "positive_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net > 0.0)) >= 2 / 3,
        "bootstrap_auc_probability_gt_zero_ge_0_80": bool(not smoke and bootstrap["auc_delta"]["probability_gt_zero"] >= 0.80),
        "bootstrap_net_probability_gt_zero_ge_0_80": bool(not smoke and bootstrap["net_delta"]["probability_gt_zero"] >= 0.80),
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "proper_calibration_embargo_all_fits": all(
            audit["internal_chronology"]["strict_35m_embargo"]
            and audit["internal_chronology"]["prototype_labels_from_proper_only"]
            and audit["internal_chronology"]["platt_labels_from_later_calibration_only"]
            for audit in model_audits
        ),
        "trajectory_dtw_architecture_all_fits": all(
            audit["dtw_radius"] == DTW_RADIUS
            and not audit["kernel_or_covariance_density_used"]
            and not audit["event_neighbor_retrieval_used"] for audit in model_audits
        ),
        "causal_numeric_inputs_only": all(
            not audit["source_category_issuer_ticker_text_used"]
            and not audit["target_rows_used_for_scaling_prototypes_or_platt"] for audit in model_audits
        ),
        "rigid_elastic_disagreement_nonconstant": bool(
            np.isfinite(evidence.method_disagreement.to_numpy(float)).all()
            and float(evidence.method_disagreement.std()) > 1e-10
        ),
        "strict_nested_chronology": all(
            audit["inner_chronology"]["strict_35m_embargo"]
            and audit["outer_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    selected_canonical = controller.canonical_research_gate(selected_summary)
    require(selected_canonical["total"] == 14, "selected canonical gate count changed")
    require(selected_summary["research_gate"] == selected_canonical, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": {**base_simple, "all_trade_mean_signed_net": float(base_net.mean())},
        "outer_candidate": {**candidate_simple, "all_trade_mean_signed_net": float(candidate_net.mean())},
        "outer_auc_delta": candidate_simple["auc"] - base_simple["auc"],
        "outer_ba_delta": candidate_simple["balanced_accuracy"] - base_simple["balanced_accuracy"],
        "outer_net_delta": float(candidate_net.mean() - base_net.mean()),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "trajectory_diagnostics": {
            "rigid_margin_quantiles": {
                str(q): float(np.quantile(evidence.rigid_margin, q)) for q in (0.1, 0.5, 0.9)
            },
            "elastic_margin_quantiles": {
                str(q): float(np.quantile(evidence.elastic_margin, q)) for q in (0.1, 0.5, 0.9)
            },
            "method_disagreement_quantiles": {
                str(q): float(np.quantile(evidence.method_disagreement, q)) for q in (0.1, 0.5, 0.9)
            },
        },
        "canonical_gate_audit": {
            "baseline": canonical_base, "candidate": canonical_candidate,
            "selected": selected_canonical, "reported_equals_controller_recomputed": True,
        },
        "direction_change": {
            "probability_changed_n": int(np.sum(champion.prob.to_numpy(float) != diagnostic.prob.to_numpy(float))),
            "direction_changed_n": int(np.sum((champion.prob.to_numpy(float) >= 0.5) != (diagnostic.prob.to_numpy(float) >= 0.5))),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected_frame.prob.to_numpy(float)),
        },
        "fallback": {
            "activated": not material_pass,
            "exact_full_v69_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic, "selected_frame": selected_frame,
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(not out.name.startswith("output_V87"), "runner must not write output_V87 directly")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
    }
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V87", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V87_TRAJECTORY_DIAGNOSTIC": evaluation["candidate_summary"],
                "V87_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V87", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V87", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V87_ELASTIC_TRAJECTORY_PROTOTYPE_REPORT.json": {
            "version": "V87", "hypothesis": HYPOTHESIS,
            "channels": TRAJECTORY_CHANNELS, "dtw_radius": DTW_RADIUS,
            "policies": POLICIES, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V87_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V87_TRAJECTORY_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V87_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V87_TRAJECTORY_OUTER_OOF.csv.gz", "V87_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        ])
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    authority = verify_authority()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "model_contract": {
                "objective": "up/down class trajectory-prototype distance with later-past Platt mapping",
                "trajectory_channels": list(TRAJECTORY_CHANNELS),
                "distance_methods": ["rigid profile", "radius-one Sakoe-Chiba DTW"],
                "model_inputs": "causal pre-event numeric horizon profiles only",
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V86 root runners plus fixed V85/V86 declarations",
                "trajectory_dtw_prototype_found": False,
                "platt_prior_art": "V62 calibrates existing OOF probabilities; V87 applies a subordinate scalar map to newly constructed proper-past trajectory-distance margins",
                "disjoint_from_v84": "no RFF, kernel, Laplace posterior, or Bayesian ensemble",
                "disjoint_from_v85": "no entropy tilting, regime moment matching, or weighted ridge logistic",
                "disjoint_from_v86": "no copula rank-normal transform, Gaussian density, or class covariance QDA",
            },
            "canonical_controller_gate_checks": 14, "output_written": False,
        }), indent=2))
        return
    dev, champion = load_authorized()
    diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, smoke=args.smoke_test)
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_completed": len(audits),
        "prototype_fit_count": 2 * len(audits),
        "platt_fit_count": 4 * len(audits),
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "trajectory_diagnostics": evaluation["trajectory_diagnostics"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "output_written": False,
    }
    if not args.dry_run and args.full_run:
        summary.update(write_outputs(args.output, authority, evidence, audits, evaluation))
        summary["output_written"] = True
        summary["output"] = str(args.output)
    print(json.dumps(clean(summary), indent=2))


if __name__ == "__main__":
    main()
