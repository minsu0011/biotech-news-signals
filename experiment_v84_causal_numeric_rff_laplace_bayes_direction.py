"""V84 causal-numeric RFF Laplace-Bayesian direction challenger.

The material hypothesis is smooth nonlinear similarity in causal pre-event
numeric state, not another linear, tree, deep, ranking, retrieval, graph, or
shape-additive learner.  Every past-only training cut is robustly standardized,
then mapped through deterministic random Fourier features approximating an RBF
kernel at two fixed median-heuristic length scales.  L2 logistic fits receive a
full Laplace covariance audit; posterior predictive variance and cross-scale
disagreement define one fixed reliability-weighted ensemble candidate.

Only immutable V36 DEV and atomic output_V69 are authoritative.  Folds 2--4
use nested expanding chronology, a strict 35-minute embargo, event/duplicate
purge, inner-only selection among exact V69 and three predeclared kernel
policies, and outer evaluation-only outcomes.  V69 confidence/high-confidence
membership are immutable.  A failed material gate returns the exact entire
V69 frame, and the controller's canonical 14 checks are recomputed from raw
selected metrics.  Preparation permits audit and one single-fold smoke only.
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
DEFAULT_OUT = ROOT / "research" / "staging" / "V84_CAUSAL_NUMERIC_RFF_LAPLACE_BAYES_V1"

VERSION = 84
HYPOTHESIS = "CAUSAL_NUMERIC_RFF_LAPLACE_BAYES_DIRECTION_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_OOF_SHA256 = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_DEV_ROWS = 9462
EXPECTED_V69_ROWS = 7568

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
COST = 0.002
SEED = 8401
RFF_COMPONENTS = 192
LENGTH_MULTIPLIERS = (1.0, 2.0)
LOGISTIC_C = 1.0
SCALE_BLEND_WEIGHT = 0.30
ENSEMBLE_BLEND_WEIGHT = 0.50
DISAGREEMENT_DECAY = 4.0
BOOTSTRAP_DRAWS = 2000

# Pure causal numeric state.  No source, issuer, ticker, text representation,
# event taxonomy, form, response prior, or target-derived feature enters the
# kernel.  Missingness flags are created from these same training-cut columns.
NUMERIC_COLUMNS = (
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_ret_90", "pre_ret_120", "pre_vol_5", "pre_vol_10", "pre_vol_30",
    "pre_vol_60", "volume_ratio_1_30", "volume_ratio_2_30",
    "volume_ratio_5_30", "volume_ratio_10_60", "range_5m", "range_10m",
    "range_30m", "range_60m", "close_position_10", "close_position_30",
    "close_position_60", "entry_bar_ret", "entry_bar_range",
    "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
    "up_fraction_30", "trend_slope_30", "trend_slope_60",
    "vwap_distance_30", "log_entry_price", "minutes_from_open",
    "minutes_to_close", "benchmark_ret_2", "benchmark_ret_5",
    "benchmark_ret_15", "benchmark_ret_30", "benchmark_ret_60",
)

POLICIES = (
    {"name": "RFF_MEDIAN_W0.30", "mode": "scale", "scale_index": 0, "weight": SCALE_BLEND_WEIGHT},
    {"name": "RFF_BROAD_W0.30", "mode": "scale", "scale_index": 1, "weight": SCALE_BLEND_WEIGHT},
    {"name": "RFF_POSTERIOR_ENSEMBLE_W0.50", "mode": "ensemble", "scale_index": None, "weight": ENSEMBLE_BLEND_WEIGHT},
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


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: np.ndarray | float) -> np.ndarray:
    base = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    challenge = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    local = np.asarray(weight, float)
    return expit((1.0 - local) * base + local * challenge)


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


class RobustNumericState:
    """Training-cut-only median/IQR preprocessing with missingness state."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "RobustNumericState":
        values = frame.loc[:, NUMERIC_COLUMNS].to_numpy(float)
        self.median = np.nanmedian(values, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(values), values, self.median)
        q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
        scale = q75 - q25
        standard = np.std(filled, axis=0)
        self.scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "numeric state fit failed")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        require(self.median is not None and self.scale is not None, "numeric state not fitted")
        values = frame.loc[:, NUMERIC_COLUMNS].to_numpy(float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        standardized = np.clip((filled - self.median) / self.scale, -8.0, 8.0)
        output = np.column_stack([standardized, missing.astype(float)]).astype(np.float64)
        require(np.isfinite(output).all(), "non-finite robust numeric state")
        return output


def median_length_scale(state: np.ndarray) -> float:
    require(len(state) >= 100, "too few rows for length-scale heuristic")
    indexes = np.linspace(0, len(state) - 1, min(len(state), 512), dtype=int)
    sample = state[indexes]
    squared_norm = np.sum(sample * sample, axis=1)
    distance_squared = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * sample @ sample.T,
        0.0,
    )
    upper = distance_squared[np.triu_indices(len(sample), k=1)]
    positive = upper[upper > 1e-12]
    require(len(positive) >= 100, "degenerate causal-state distances")
    result = float(np.sqrt(np.median(positive)))
    require(math.isfinite(result) and result > 1e-6, "invalid median length scale")
    return result


def rff_map(state: np.ndarray, base_frequency: np.ndarray, phase: np.ndarray, length_scale: float) -> np.ndarray:
    projection = state @ (base_frequency / float(length_scale)) + phase
    features = math.sqrt(2.0 / base_frequency.shape[1]) * np.cos(projection)
    require(features.shape[1] == RFF_COMPONENTS and np.isfinite(features).all(), "RFF map failed")
    return features.astype(np.float64)


def laplace_variance(features: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([features, np.ones(len(features), dtype=float)])
    variance = np.sum((augmented @ covariance) * augmented, axis=1)
    return np.maximum(np.asarray(variance, float), 1e-10)


def posterior_predictive(mean_logit: np.ndarray, variance: np.ndarray) -> np.ndarray:
    shrink = np.sqrt(1.0 + math.pi * np.asarray(variance, float) / 8.0)
    return np.clip(expit(np.asarray(mean_logit, float) / shrink), 1e-6, 1.0 - 1e-6)


def fit_one_scale(
    train_features: np.ndarray, target_features: np.ndarray, train: pd.DataFrame,
    length_scale: float, multiplier: float, seed: int,
) -> dict[str, Any]:
    model = LogisticRegression(
        penalty="l2", C=LOGISTIC_C, solver="lbfgs", max_iter=350,
        tol=1e-7, fit_intercept=True, random_state=seed,
    )
    sample_weight = train.cluster_weight.to_numpy(float)
    model.fit(train_features, train.y.to_numpy(int), sample_weight=sample_weight)
    train_mean = model.decision_function(train_features).astype(float)
    target_mean = model.decision_function(target_features).astype(float)
    fitted = expit(train_mean)
    curvature = sample_weight * fitted * (1.0 - fitted)
    augmented = np.column_stack([train_features, np.ones(len(train_features), dtype=float)])
    prior = np.full(augmented.shape[1], 1.0 / LOGISTIC_C, dtype=float)
    prior[-1] = 1e-6
    hessian = augmented.T @ (curvature[:, None] * augmented)
    hessian.flat[:: hessian.shape[0] + 1] += prior + 1e-7
    covariance = np.linalg.inv(hessian)
    require(np.isfinite(covariance).all(), "non-finite Laplace covariance")
    train_variance = laplace_variance(train_features, covariance)
    target_variance = laplace_variance(target_features, covariance)
    probability = posterior_predictive(target_mean, target_variance)
    return {
        "probability": probability,
        "mean_logit": target_mean,
        "variance": target_variance,
        "train_mean_logit": train_mean,
        "train_variance": train_variance,
        "audit": {
            "model": "L2 logistic on deterministic RBF random Fourier features",
            "posterior": "full-Hessian Laplace covariance with logistic-normal predictive approximation",
            "length_multiplier": float(multiplier),
            "length_scale": float(length_scale),
            "rff_components": RFF_COMPONENTS,
            "input_state_n": train_features.shape[1],
            "training_n": len(train),
            "iterations": int(model.n_iter_[0]),
            "converged": bool(model.n_iter_[0] < model.max_iter),
            "coefficient_l2": float(np.linalg.norm(model.coef_)),
            "train_posterior_std_quantiles": {
                str(q): float(np.quantile(np.sqrt(train_variance), q)) for q in (0.1, 0.5, 0.9)
            },
            "target_posterior_std_quantiles": {
                str(q): float(np.quantile(np.sqrt(target_variance), q)) for q in (0.1, 0.5, 0.9)
            },
        },
    }


def combine_scales(scale_results: list[dict[str, Any]], training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_key = "train_mean_logit" if training else "mean_logit"
    variance_key = "train_variance" if training else "variance"
    means = np.column_stack([item[mean_key] for item in scale_results])
    variances = np.column_stack([item[variance_key] for item in scale_results])
    precision = 1.0 / np.maximum(variances, 1e-10)
    weights = precision / precision.sum(axis=1, keepdims=True)
    ensemble_mean = np.sum(weights * means, axis=1)
    ensemble_variance = np.sum(weights * (variances + (means - ensemble_mean[:, None]) ** 2), axis=1)
    probability = posterior_predictive(ensemble_mean, ensemble_variance)
    disagreement = np.abs(expit(means[:, 0]) - expit(means[:, 1]))
    return probability, np.maximum(ensemble_variance, 1e-10), disagreement


def fit_rff_bayes(
    train: pd.DataFrame, target: pd.DataFrame, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    processor = RobustNumericState().fit(train)
    train_state = processor.transform(train)
    target_state = processor.transform(target)
    base_length = median_length_scale(train_state)
    rng = np.random.default_rng(seed)
    base_frequency = rng.normal(size=(train_state.shape[1], RFF_COMPONENTS))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=RFF_COMPONENTS)
    scale_results = []
    for multiplier in LENGTH_MULTIPLIERS:
        length = base_length * multiplier
        train_features = rff_map(train_state, base_frequency, phase, length)
        target_features = rff_map(target_state, base_frequency, phase, length)
        scale_results.append(
            fit_one_scale(train_features, target_features, train, length, multiplier, seed)
        )
    ensemble_probability, ensemble_variance, disagreement = combine_scales(scale_results, training=False)
    _, train_ensemble_variance, _ = combine_scales(scale_results, training=True)
    reference_std = float(np.median(np.sqrt(train_ensemble_variance)))
    require(reference_std > 0.0 and math.isfinite(reference_std), "invalid training posterior reference")
    reliability = np.clip(reference_std / np.sqrt(ensemble_variance), 0.0, 1.0)
    reliability *= np.exp(-DISAGREEMENT_DECAY * disagreement)
    require(np.isfinite(reliability).all(), "non-finite posterior reliability")
    predictions = {
        "median_probability": scale_results[0]["probability"],
        "broad_probability": scale_results[1]["probability"],
        "ensemble_probability": ensemble_probability,
        "ensemble_variance": ensemble_variance,
        "disagreement": disagreement,
        "reliability": reliability,
    }
    audit = {
        "architecture": "two-scale deterministic RBF RFF plus Laplace Bayesian logistic ensemble",
        "smooth_nonlinear_kernel": True,
        "kernel": "RBF approximated by random Fourier features",
        "median_length_scale": base_length,
        "length_multipliers": LENGTH_MULTIPLIERS,
        "rff_components": RFF_COMPONENTS,
        "raw_numeric_feature_n": len(NUMERIC_COLUMNS),
        "state_feature_n": train_state.shape[1],
        "categorical_feature_n": 0,
        "source_feature_used": False,
        "issuer_or_ticker_feature_used": False,
        "text_feature_used": False,
        "labels_used_for_preprocessing_or_kernel_scale": False,
        "posterior_reference_std": reference_std,
        "target_reliability_quantiles": {
            str(q): float(np.quantile(reliability, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_disagreement_quantiles": {
            str(q): float(np.quantile(disagreement, q)) for q in (0.1, 0.5, 0.9)
        },
        "scale_models": [item["audit"] for item in scale_results],
        "training_n": len(train),
        "target_n": len(target),
        "seed": seed,
        "device": "CPU",
        "gpu_used": False,
        "cpu_reason": "small deterministic RFF maps and exact 193x193 Laplace covariance inversions",
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
    audit = chronology(inner_train, inner_valid, "V84 inner")
    audit["validation_source"] = "earlier committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def apply_policy(
    baseline: np.ndarray, predictions: dict[str, np.ndarray], policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if policy["mode"] == "noop":
        weight = np.zeros(len(baseline), dtype=float)
        return np.asarray(baseline, float).copy(), weight
    if policy["mode"] == "scale":
        key = "median_probability" if int(policy["scale_index"]) == 0 else "broad_probability"
        weight = np.full(len(baseline), float(policy["weight"]), dtype=float)
        return probability_blend(baseline, predictions[key], weight), weight
    require(policy["mode"] == "ensemble", "unknown V84 policy mode")
    weight = float(policy["weight"]) * predictions["reliability"]
    return probability_blend(baseline, predictions["ensemble_probability"], weight), weight


def choose_inner(
    train: pd.DataFrame, valid: pd.DataFrame, champion_valid: pd.DataFrame, fold: int,
) -> dict[str, Any]:
    predictions, model_audit = fit_rff_bayes(train, valid, SEED + fold * 10)
    baseline = champion_valid.prob.to_numpy(float)
    confidence = champion_valid.confidence_signal.to_numpy(float)
    high = bool_series(champion_valid.high_conf).to_numpy(bool)
    baseline_metric = metric(valid, baseline, confidence, high)
    noop = {"name": "V69_NOOP", "mode": "noop", "scale_index": None, "weight": 0.0}
    trials = []
    for policy in (noop, *POLICIES):
        probability, local_weight = apply_policy(baseline, predictions, policy)
        values = metric(valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            **policy, "metrics": values,
            "auc_delta": auc_delta, "ba_delta": ba_delta, "net_delta": net_delta,
            "mean_effective_weight": float(np.mean(local_weight)),
            "eligible": bool(policy["mode"] == "noop" or (ba_delta >= -0.005 and net_delta >= -0.0005)),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["ba_delta"], -trial["mean_effective_weight"]),
    )
    return {
        "selection_rule": "maximize 2*AUC+BA on inner-past OOF; BA delta>=-0.005 and net delta>=-0.0005; exact no-op plus three fixed RFF policies only",
        "selected": selected, "trials": trials,
        "rff_bayes_model_audit": model_audit,
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v84_model"] = "V69_NOOP"
    diagnostic["v84_effective_weight"] = 0.0
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
        outer_audit = chronology(outer_train, outer_valid, f"V84 outer fold {fold}")
        inner_train, inner_valid, inner_champion, inner_audit = inner_partition(
            dev, champion, pd.Timestamp(outer_valid.event_time_utc.min())
        )
        policy = choose_inner(inner_train, inner_valid, inner_champion, fold)
        selected = policy["selected"]
        predictions, outer_model_audit = fit_rff_bayes(outer_train, outer_valid, SEED + 1000 + fold)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        candidate, local_weight = apply_policy(baseline, predictions, selected)
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        probability_map = dict(zip(outer_valid.event_id, candidate))
        weight_map = dict(zip(outer_valid.event_id, local_weight))
        positions = diagnostic.index[diagnostic.event_id.isin(probability_map)]
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v84_model"] = selected["name"]
        diagnostic.loc[positions, "v84_effective_weight"] = diagnostic.loc[positions, "event_id"].map(weight_map)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["rff_median_prob"] = predictions["median_probability"]
        evidence["rff_broad_prob"] = predictions["broad_probability"]
        evidence["rff_ensemble_prob"] = predictions["ensemble_probability"]
        evidence["posterior_variance"] = predictions["ensemble_variance"]
        evidence["scale_disagreement"] = predictions["disagreement"]
        evidence["posterior_reliability"] = predictions["reliability"]
        evidence["candidate_prob"] = candidate
        evidence["v84_model"] = selected["name"]
        evidence["v84_effective_weight"] = local_weight
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold, "inner_chronology": inner_audit, "outer_chronology": outer_audit,
            "policy": policy, "outer_rff_bayes_model_audit": outer_model_audit,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "outer_mean_posterior_variance": float(np.mean(predictions["ensemble_variance"])),
            "outer_mean_scale_disagreement": float(np.mean(predictions["disagreement"])),
            "outer_mean_posterior_reliability": float(np.mean(predictions["reliability"])),
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V84 RFF-BAYES] fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f} "
            f"posterior_reliability={audits[-1]['outer_mean_posterior_reliability']:.4f}",
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
        "method": "paired fold/date block bootstrap over locked RFF-Bayesian policies",
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
    model_audits = [audit["policy"]["rff_bayes_model_audit"] for audit in audits] + [
        audit["outer_rff_bayes_model_audit"] for audit in audits
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
        "all_fits_are_smooth_nonlinear_rff_laplace": all(
            audit["smooth_nonlinear_kernel"]
            and audit["kernel"] == "RBF approximated by random Fourier features"
            and all(scale["converged"] for scale in audit["scale_models"])
            for audit in model_audits
        ),
        "pure_causal_numeric_model_inputs": all(
            not audit["source_feature_used"]
            and not audit["issuer_or_ticker_feature_used"]
            and not audit["text_feature_used"]
            and audit["categorical_feature_n"] == 0
            for audit in model_audits
        ),
        "posterior_uncertainty_finite_nonconstant": bool(
            np.isfinite(evidence.posterior_variance.to_numpy(float)).all()
            and float(evidence.posterior_variance.std()) > 1e-10
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
        "uncertainty": {
            "posterior_variance_quantiles": {
                str(q): float(np.quantile(evidence.posterior_variance, q)) for q in (0.1, 0.5, 0.9)
            },
            "scale_disagreement_quantiles": {
                str(q): float(np.quantile(evidence.scale_disagreement, q)) for q in (0.1, 0.5, 0.9)
            },
            "posterior_reliability_quantiles": {
                str(q): float(np.quantile(evidence.posterior_reliability, q)) for q in (0.1, 0.5, 0.9)
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
    require(not out.name.startswith("output_V84"), "runner must not write output_V84 directly")
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
            "version": "V84", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V84_RFF_LAPLACE_BAYES_DIAGNOSTIC": evaluation["candidate_summary"],
                "V84_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V84", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V84", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V84_RFF_LAPLACE_BAYES_REPORT.json": {
            "version": "V84", "hypothesis": HYPOTHESIS,
            "kernel": "RBF random Fourier features",
            "length_multipliers": LENGTH_MULTIPLIERS,
            "policies": POLICIES,
            "authority_audit": authority, "nested_fold_audits": audits, "evaluation": compact,
        },
        "V84_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V84_RFF_BAYES_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V84_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V84_RFF_BAYES_OUTER_OOF.csv.gz", "V84_FAIL_CLOSED_SELECTED_OOF.csv.gz",
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
                "objective": "Bayesian logistic direction over smooth nonlinear RBF random Fourier features",
                "kernel_scales": LENGTH_MULTIPLIERS,
                "posterior": "full-Hessian Laplace covariance and predictive uncertainty",
                "model_inputs": "causal numeric state only",
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V83 root runners",
                "kernel_svm_rff_nystrom_found": False,
                "disjoint_from_v82": "no deep masked-state representation",
                "disjoint_from_v83": "no source-specific additive shape functions",
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
        "rff_logistic_fit_count": 4 * len(audits),
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "uncertainty": evaluation["uncertainty"],
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
