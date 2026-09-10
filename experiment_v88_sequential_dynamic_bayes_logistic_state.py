"""V88 sequential dynamic Bayesian-logistic state challenger.

Within every embargoed past cut, causal numeric pre-event state is robustly
scaled and processed exactly once in chronological order.  A diagonal
assumed-density logistic filter carries a coefficient mean and variance,
injects fixed process variance between events, and updates the posterior only
after observing each past label.  The terminal posterior is frozen before any
inner-validation or outer row is scored.  This represents coefficient drift;
it is not V84's static RFF/Laplace kernel, V85's entropy reweighting, V86's
copula QDA class density, or V87's DTW trajectory prototype.

Inner-past OOF selects exact V69 or fixed 0.25/0.50 logit blends.  Outer labels
are evaluation-only, every boundary has a strict 35-minute embargo and event-
group purge, and V69 confidence/high-confidence membership remain exact.
Authority is immutable V36 DEV plus atomic output_V69.  Material failure
restores the exact entire V69 frame.  Audit and one two-thread CPU smoke write
nothing and are designed not to disturb a concurrent GPU full run.
"""
from __future__ import annotations

import os

_thread_default = "32" if os.environ.get("MARKET_BIO_VERSION_OUTPUT") else "2"
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = _thread_default

import argparse
import ctypes
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import autonomous_v37plus as controller
import experiment_v44_microstructure as v44


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "research" / "staging" / "V88_SEQUENTIAL_DYNAMIC_BAYES_LOGISTIC_STATE_V1"
)

VERSION = 88
HYPOTHESIS = "SEQUENTIAL_DYNAMIC_DIAGONAL_BAYES_LOGISTIC_STATE_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_MARKET_FOLD_ROWS = {"US": 1595, "KR": 297}
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
SEED = 8801
BOOTSTRAP_DRAWS = 2000
COST = 0.002

ROBUST_CLIP = 6.0
PRIOR_VARIANCE = 0.50
PROCESS_VARIANCE = 0.00020
MAX_POSTERIOR_VARIANCE = 1.50
MIN_LOGISTIC_CURVATURE = 0.02
RIDGE_PRECISION_INCREMENT = 0.00002
MIN_CLASS_ROWS = 120

# Strictly pre-decision price/volume/clock/benchmark state.  No entry-bar,
# source, category, issuer, ticker, text, keyword, response, or target feature.
FEATURES = (
    "pre_ret_1", "pre_ret_2", "pre_ret_5", "pre_ret_10", "pre_ret_15",
    "pre_ret_30", "pre_ret_60", "pre_ret_120",
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "volume_ratio_10_60", "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "close_position_60",
    "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
    "up_fraction_30", "trend_slope_30", "trend_slope_60",
    "vwap_distance_30", "minutes_from_open", "minutes_to_close",
    "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
    "benchmark_ret_30", "benchmark_ret_60",
)

ARCHITECTURES = (
    {"name": "DYNAMIC_BAYES_STATE_W0.25", "weight": 0.25},
    {"name": "DYNAMIC_BAYES_STATE_W0.50", "weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_bounded_runtime() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    controller_full = bool(os.environ.get("MARKET_BIO_VERSION_OUTPUT"))
    cpu_ids = list(range(logical)) if controller_full else list(range(max(0, logical - 2), logical))
    if os.name == "nt":
        mask = sum(1 << cpu_id for cpu_id in cpu_ids)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        require(bool(kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(mask))), "bounded CPU affinity failed")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
    return {
        "logical_cpu_count": logical,
        "affinity_cpu_ids": cpu_ids,
        "numeric_thread_cap": logical if controller_full else 2,
        "gpu_used": False,
        "concurrent_full_run_interference_guard": not controller_full,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, np.asarray(score, float))) if np.unique(target).size == 2 else None


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


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    robustness_path = V69_DIR / "DEV_ROBUSTNESS_REPORT.json"
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit_path.is_file() and manifest_path.is_file(), "V69 atomic authority incomplete")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == sha256(manifest_path), "V69 manifest binding changed")
    require(commit.get("data_sha") == EXPECTED_V36_SHA256, "V69 not bound to immutable V36 DEV")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment changed")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 artifact hash changed: {name}")
    require(V69_PATH.name in manifest.get("files", {}), "V69 selected OOF not manifested")
    require(robustness_path.name in manifest.get("files", {}), "V69 robustness not manifested")
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
            str(robustness_path.relative_to(ROOT)),
        ],
        "v36_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": sha256(commit_path),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_oof_sha256": sha256(V69_PATH),
        "v69_robustness_sha256": sha256(robustness_path),
        "manifest_files_verified": len(manifest.get("files", {})),
        "failed_output_loaded": False,
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_reserve_loaded": False,
    }


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    champion = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_V36_ROWS and len(champion) == EXPECTED_V69_ROWS, "authority row count changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 is not a V36 subset")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15), "V36/V69 returns differ")
    require(all(column in dev.columns for column in FEATURES), "V88 causal numeric feature missing")
    dev["event_time_utc"] = pd.to_datetime(dev.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    return dev.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True), champion


class TrainingCutRobustScaler:
    def fit(self, frame: pd.DataFrame) -> "TrainingCutRobustScaler":
        raw = frame[list(FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        self.median = np.nanmedian(raw, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        q25 = np.nanquantile(raw, 0.25, axis=0)
        q75 = np.nanquantile(raw, 0.75, axis=0)
        self.scale = q75 - q25
        self.scale[~np.isfinite(self.scale) | (self.scale < 1e-8)] = 1.0
        self.fit_rows = len(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raw = frame[list(FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        scaled = np.clip((filled - self.median) / self.scale, -ROBUST_CLIP, ROBUST_CLIP)
        # Missing indicators are deterministic functions of authorized numeric
        # inputs, not additional data or categories.
        matrix = np.column_stack([
            np.ones(len(frame), dtype=float),
            scaled,
            (~observed).astype(float),
        ]).astype(np.float64)
        require(np.isfinite(matrix).all(), "non-finite V88 numeric state")
        return matrix, {
            "rows": len(frame),
            "matrix_columns": matrix.shape[1],
            "missing_rate_mean": float((~observed).mean()),
            "missing_rate_max": float((~observed).mean(axis=0).max()),
        }


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    return 1.0 / np.maximum(count, 1.0)


class DynamicDiagonalBayesLogistic:
    """One-pass chronological diagonal assumed-density logistic filter."""

    def fit(self, frame: pd.DataFrame) -> "DynamicDiagonalBayesLogistic":
        train = frame.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
        require(len(train) >= 400 and train.y.nunique() == 2, "V88 past support insufficient")
        class_count = train.y.value_counts()
        require(int(class_count.min()) >= MIN_CLASS_ROWS, "V88 class support insufficient")
        self.scaler = TrainingCutRobustScaler().fit(train)
        matrix, transform_audit = self.scaler.transform(train)
        target = train.y.to_numpy(int)
        weights = duplicate_weights(train)
        times = pd.to_datetime(train.event_time_utc, utc=True)
        dimension = matrix.shape[1]
        mean = np.zeros(dimension, dtype=np.float64)
        variance = np.full(dimension, PRIOR_VARIANCE, dtype=np.float64)
        innovations = np.zeros(len(train), dtype=float)
        predictive = np.zeros(len(train), dtype=float)
        snapshots: dict[str, np.ndarray] = {}
        snapshot_positions = {
            max(0, int(len(train) * 0.25) - 1): "quarter",
            max(0, int(len(train) * 0.50) - 1): "half",
            max(0, int(len(train) * 0.75) - 1): "three_quarter",
            len(train) - 1: "terminal",
        }
        previous = pd.Timestamp(times.iloc[0])
        total_process_variance = 0.0
        for position, (state, label, weight) in enumerate(zip(matrix, target, weights)):
            timestamp = pd.Timestamp(times.iloc[position])
            gap_hours = max((timestamp - previous).total_seconds() / 3600.0, 0.0)
            process = PROCESS_VARIANCE * (1.0 + min(gap_hours, 24.0) / 24.0)
            variance = np.minimum(variance + process, MAX_POSTERIOR_VARIANCE)
            total_process_variance += process
            score_mean = float(state @ mean)
            score_variance = float(np.sum(state * state * variance))
            attenuation = math.sqrt(1.0 + math.pi * score_variance / 8.0)
            probability = float(expit(np.clip(score_mean / attenuation, -12.0, 12.0)))
            predictive[position] = probability
            innovations[position] = label - probability
            curvature = max(probability * (1.0 - probability), MIN_LOGISTIC_CURVATURE)
            posterior_precision = (
                1.0 / np.maximum(variance, 1e-10)
                + weight * curvature * state * state
                + RIDGE_PRECISION_INCREMENT
            )
            posterior_variance = 1.0 / posterior_precision
            gradient = weight * (label - probability) * state - RIDGE_PRECISION_INCREMENT * mean
            mean = mean + posterior_variance * gradient
            variance = np.clip(posterior_variance, 1e-8, MAX_POSTERIOR_VARIANCE)
            if position in snapshot_positions:
                snapshots[snapshot_positions[position]] = mean.copy()
            previous = timestamp
        require(np.isfinite(mean).all() and np.isfinite(variance).all(), "V88 posterior is non-finite")
        require(len(snapshots) == 4, "V88 coefficient snapshots incomplete")
        self.mean = mean
        self.variance = variance
        first = slice(0, max(1, len(train) // 4))
        last = slice(len(train) - max(1, len(train) // 4), len(train))

        def log_loss(section: slice) -> float:
            p = np.clip(predictive[section], 1e-6, 1.0 - 1e-6)
            y = target[section]
            return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

        self.audit = {
            "train_n": len(train),
            "feature_n": len(FEATURES),
            "state_dimension": dimension,
            "numeric_state_only": True,
            "categorical_text_source_ticker_features": 0,
            "chronological_one_pass": True,
            "posterior_update_count": len(train),
            "training_rows_revisited": False,
            "target_rows_used_in_scaler_or_filter": False,
            "terminal_posterior_frozen_before_target": True,
            "diagonal_assumed_density_filter": True,
            "fixed_parameters": {
                "prior_variance": PRIOR_VARIANCE,
                "process_variance": PROCESS_VARIANCE,
                "max_posterior_variance": MAX_POSTERIOR_VARIANCE,
                "min_logistic_curvature": MIN_LOGISTIC_CURVATURE,
                "ridge_precision_increment": RIDGE_PRECISION_INCREMENT,
            },
            "train_transform": transform_audit,
            "class_counts": {str(key): int(value) for key, value in class_count.sort_index().items()},
            "duplicate_balanced": True,
            "prequential_auc": safe_auc(target, predictive),
            "prequential_first_quarter_log_loss": log_loss(first),
            "prequential_last_quarter_log_loss": log_loss(last),
            "innovation_mean": float(innovations.mean()),
            "innovation_std": float(innovations.std()),
            "total_process_variance_injected_per_coefficient": total_process_variance,
            "terminal_coefficient_l2": float(np.linalg.norm(mean)),
            "terminal_posterior_variance_mean": float(variance.mean()),
            "terminal_posterior_variance_max": float(variance.max()),
            "coefficient_drift_quarter_to_terminal_l2": float(np.linalg.norm(snapshots["terminal"] - snapshots["quarter"])),
            "terminal_coefficient_sha256": array_sha256(mean),
            "terminal_variance_sha256": array_sha256(variance),
        }
        return self

    def predict_probability(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        matrix, target_transform = self.scaler.transform(frame)
        score_mean = matrix @ self.mean
        score_variance = np.sum(matrix * matrix * self.variance[None, :], axis=1)
        attenuation = np.sqrt(1.0 + math.pi * score_variance / 8.0)
        probability = np.clip(expit(np.clip(score_mean / attenuation, -12.0, 12.0)), 1e-5, 1.0 - 1e-5)
        require(np.isfinite(probability).all(), "V88 target probability is non-finite")
        return probability, {
            "target_n": len(frame),
            "target_transform": target_transform,
            "target_rows_used_in_fit_or_update": False,
            "posterior_updates_during_target_scoring": 0,
            "terminal_posterior_frozen": True,
            "score_mean": float(score_mean.mean()),
            "score_std": float(score_mean.std()),
            "posterior_score_variance_mean": float(score_variance.mean()),
            "probability_mean": float(probability.mean()),
            "probability_sha256": array_sha256(probability),
        }


def dynamic_predictions(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model = DynamicDiagonalBayesLogistic().fit(train)
    probability, prediction_audit = model.predict_probability(target)
    return probability, {**model.audit, "prediction": prediction_audit}


def architecture_probability(baseline: np.ndarray, expert: np.ndarray, architecture: dict[str, Any]) -> np.ndarray:
    weight = float(architecture["weight"])
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    expert_logit = logit(np.clip(np.asarray(expert, float), 1e-5, 1.0 - 1e-5))
    probability = expit((1.0 - weight) * base_logit + weight * expert_logit)
    require(np.isfinite(probability).all(), "V88 blend returned non-finite probability")
    return probability


def metric(frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    target = frame.y.to_numpy(int)
    confidence = np.asarray(confidence, float)
    high = np.asarray(high, bool)
    correct = (prediction == target).astype(int)
    signed_net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "accuracy": float(correct.mean()),
        "pred_up": float(prediction.mean()),
        "all_trade_mean_signed_net": float(signed_net.mean()),
        "confidence_correctness_auc": safe_auc(correct, confidence),
        "highconf_n": int(high.sum()),
        "highconf_accuracy": float(correct[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(signed_net[high].mean()) if high.any() else None,
    }


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) and len(valid), f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: strict 35-minute embargo failed")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label}: event-group leakage")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end,
        "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0,
        "strict_35m_embargo": True,
    }


def aligned_dev(dev: pd.DataFrame, champion_rows: pd.DataFrame) -> pd.DataFrame:
    aligned = dev.set_index("event_id").loc[champion_rows.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion_rows.y.to_numpy(int)), "aligned labels differ")
    return aligned


def prior_train(dev: pd.DataFrame, market: str, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = dev.loc[dev.market.eq(market) & (dev.event_time_utc < boundary - EMBARGO)].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    return train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)


def inner_partition(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    market: str,
    outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)
    ].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= 400, "inner V88 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V88 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expert, model_audit = dynamic_predictions(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline_metric,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]),
    }]
    for architecture in ARCHITECTURES:
        probability = architecture_probability(baseline, expert, architecture)
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"],
            "architecture": architecture,
            "metrics": values,
            "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; maximize 2*AUC+BA subject to BA delta>=-0.005 and net delta>=-0.0005; exact no-op and two fixed dynamic-state blends",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "process_variance_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v88_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in fold_specs[:1] if smoke else fold_specs:
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V88 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        expert, outer_model_audit = dynamic_predictions(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = (
            baseline.copy()
            if selected["architecture"] is None
            else architecture_probability(baseline, expert, selected["architecture"])
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v88_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                architecture_probability(baseline, expert, architecture),
                confidence,
                high,
            )
            for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["dynamic_state_prob"] = expert
        evidence["candidate_prob"] = candidate
        evidence["v88_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market,
            "fold": fold,
            "inner_chronology": inner_chronology,
            "outer_chronology": outer_chronology,
            "policy": policy,
            "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V88 DYNAMIC] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V88 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V88 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V88 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d")
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V88 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        ba_delta.append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)))
    require(len(auc_delta) >= int(0.90 * draws), "V88 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked dynamic-state blend",
        "seed": SEED,
        "requested_draws": draws,
        "blocks": len(blocks),
        "auc_delta": interval(auc_delta),
        "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V88/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    dynamic_valid = all(
        audit["inner_model_audit"]["chronological_one_pass"]
        and audit["outer_model_audit"]["chronological_one_pass"]
        and audit["inner_model_audit"]["posterior_update_count"] == audit["inner_model_audit"]["train_n"]
        and audit["outer_model_audit"]["posterior_update_count"] == audit["outer_model_audit"]["train_n"]
        and audit["inner_model_audit"]["target_rows_used_in_scaler_or_filter"] is False
        and audit["outer_model_audit"]["target_rows_used_in_scaler_or_filter"] is False
        and audit["inner_model_audit"]["prediction"]["posterior_updates_during_target_scoring"] == 0
        and audit["outer_model_audit"]["prediction"]["posterior_updates_during_target_scoring"] == 0
        for audit in audits
    )
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "bootstrap_net_delta_probability_gt_zero_ge_0_65": bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "sequential_dynamic_posterior_and_frozen_target_verified": dynamic_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V88 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": base,
        "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
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
        "immutability": {
            "confidence_exact": confidence_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V69 entire frame" if not material_pass else None,
            "exact_frame_verified": bool(material_pass or selected.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected,
    }


def write_outputs(
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected_contract = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
    }
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V88", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V88_DIAGNOSTIC_DYNAMIC_STATE": evaluation["candidate_summary"],
                "V88_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V88", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V88", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected_contract["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "net_delta": evaluation["outer_net_delta"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SEQUENTIAL_DYNAMIC_BAYES_STATE_REPORT.json": {
            "version": "V88", "hypothesis": HYPOTHESIS, "features": FEATURES,
            "architectures": ARCHITECTURES,
            "filter_parameters": {
                "prior_variance": PRIOR_VARIANCE, "process_variance": PROCESS_VARIANCE,
                "max_posterior_variance": MAX_POSTERIOR_VARIANCE,
                "min_logistic_curvature": MIN_LOGISTIC_CURVATURE,
                "ridge_precision_increment": RIDGE_PRECISION_INCREMENT,
            },
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V88", "status": "MATCH", "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"], "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "coefficient_drift_l2": audit["outer_model_audit"]["coefficient_drift_quarter_to_terminal_l2"],
        "terminal_variance_mean": audit["outer_model_audit"]["terminal_posterior_variance_mean"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V88_DYNAMIC_STATE_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V88_DYNAMIC_STATE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V88_DIAGNOSTIC_DYNAMIC_STATE_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V88_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out), "artifact_count": len(files) + 1,
        "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = configure_bounded_runtime()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "feature_n": len(FEATURES),
            "causal_numeric_pre_event_only": True,
            "dynamic_diagonal_assumed_density_filter": True,
            "chronological_training_passes": 1,
            "target_batch_posterior_updates": 0,
            "fixed_parameters": {
                "prior_variance": PRIOR_VARIANCE, "process_variance": PROCESS_VARIANCE,
                "max_posterior_variance": MAX_POSTERIOR_VARIANCE,
                "min_logistic_curvature": MIN_LOGISTIC_CURVATURE,
                "ridge_precision_increment": RIDGE_PRECISION_INCREMENT,
            },
            "architectures": ARCHITECTURES,
            "canonical_controller_gate_checks": 14,
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "runtime": runtime,
        "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["inner_model_audit"] = audits[0]["inner_model_audit"]
        summary["outer_model_audit"] = audits[0]["outer_model_audit"]
        summary["bootstrap"] = evaluation["bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
