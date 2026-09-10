"""V86 robust Gaussian-copula QDA causal-state direction challenger.

Every embargoed training cut learns empirical marginal CDFs from causal
pre-event numeric market state only and maps each marginal to a clipped normal
score.  Duplicate-balanced class means and two fixed-shrinkage class covariance
matrices then define a generative quadratic log-density ratio.  This is neither
a discriminative logistic/tree/neural learner nor V84's RFF/Laplace posterior,
V85's recent-regime entropy balancing, V83's additive shapes, or V81 Group-DRO.

Inner-past evidence chooses exact V69 or one of two fixed logit blends.  Outer
outcomes are evaluation-only, every nested boundary has a strict 35-minute
embargo and event-group purge, and V69 confidence/high-confidence membership
stay frozen.  Inputs are immutable V36 DEV and atomic output_V69 only.  Any
material failure restores the exact entire V69 frame.  Audit and one bounded
two-thread CPU smoke are preparation modes; no output is written by either.
"""
from __future__ import annotations

import os

# Keep preparation smoke bounded, but allow a controller-authorized full run
# to use the configured 32-thread host.
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
from scipy.special import expit, logit, ndtri
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
    else ROOT / "research" / "staging" / "V86_ROBUST_GAUSSIAN_COPULA_QDA_V1"
)

VERSION = 86
HYPOTHESIS = "ROBUST_GAUSSIAN_COPULA_QDA_DIRECTION_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_MARKET_FOLD_ROWS = {"US": 1595, "KR": 297}
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
SEED = 8601
BOOTSTRAP_DRAWS = 2000
COST = 0.002

# Fixed covariance mixture: full dependence + diagonal robustness + spherical
# floor.  These are architecture constants, not validation-tuned parameters.
FULL_COVARIANCE_WEIGHT = 0.55
DIAGONAL_COVARIANCE_WEIGHT = 0.30
SPHERICAL_COVARIANCE_WEIGHT = 0.15
COPULA_QUANTILE_FLOOR = 0.005
COPULA_SCORE_CLIP = 3.5

FEATURES = (
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

ARCHITECTURES = (
    {"name": "COPULA_QDA_W0.25", "weight": 0.25},
    {"name": "COPULA_QDA_W0.50", "weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_bounded_runtime() -> dict[str, Any]:
    """Bound preparation smoke; use all logical CPUs for controller full runs."""
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
        "concurrent_v78_interference_guard": not controller_full,
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
    require(all(column in dev.columns for column in FEATURES), "V86 causal numeric feature missing")
    dev["event_time_utc"] = pd.to_datetime(dev.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    return dev.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True), champion


class EmpiricalGaussianizer:
    """Training-cut empirical copula map; target batches never update marginals."""

    def fit(self, frame: pd.DataFrame) -> "EmpiricalGaussianizer":
        self.sorted_values: list[np.ndarray] = []
        self.medians: list[float] = []
        self.unique_counts: list[int] = []
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite_values = values[np.isfinite(values)]
            # The earliest KR inner cut has 241 finite 120-minute-return rows;
            # 150 remains above the class-covariance support floor and avoids
            # silently dropping this predeclared causal channel by market.
            require(len(finite_values) >= 150, f"{column}: insufficient finite past support")
            ordered = np.sort(finite_values, kind="stable")
            self.sorted_values.append(ordered)
            self.medians.append(float(np.median(ordered)))
            self.unique_counts.append(int(np.unique(ordered).size))
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.zeros((len(frame), len(FEATURES)), dtype=np.float64)
        missing_rates: dict[str, float] = {}
        for position, column in enumerate(FEATURES):
            raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            missing = ~np.isfinite(raw)
            values = np.where(missing, self.medians[position], raw)
            ordered = self.sorted_values[position]
            left = np.searchsorted(ordered, values, side="left")
            right = np.searchsorted(ordered, values, side="right")
            quantile = (left + right + 1.0) / (2.0 * (len(ordered) + 1.0))
            quantile = np.clip(quantile, COPULA_QUANTILE_FLOOR, 1.0 - COPULA_QUANTILE_FLOOR)
            matrix[:, position] = np.clip(ndtri(quantile), -COPULA_SCORE_CLIP, COPULA_SCORE_CLIP)
            missing_rates[column] = float(missing.mean())
        require(np.isfinite(matrix).all(), "non-finite Gaussian copula state")
        return matrix, {
            "rows": len(frame),
            "missing_rate_mean": float(np.mean(list(missing_rates.values()))),
            "missing_rate_max": float(np.max(list(missing_rates.values()))),
        }


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    group_count = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    return 1.0 / np.maximum(group_count, 1.0)


def covariance_components(matrix: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weight = np.asarray(weight, float)
    weight = weight / max(weight.sum(), 1e-12)
    mean = np.sum(matrix * weight[:, None], axis=0)
    centered = matrix - mean
    covariance = (centered * weight[:, None]).T @ centered
    diagonal = np.diag(np.diag(covariance))
    average_variance = max(float(np.trace(covariance) / matrix.shape[1]), 1e-6)
    spherical = np.eye(matrix.shape[1], dtype=float) * average_variance
    shrunk = (
        FULL_COVARIANCE_WEIGHT * covariance
        + DIAGONAL_COVARIANCE_WEIGHT * diagonal
        + SPHERICAL_COVARIANCE_WEIGHT * spherical
    )
    shrunk += np.eye(matrix.shape[1], dtype=float) * 1e-7
    sign, log_determinant = np.linalg.slogdet(shrunk)
    require(sign > 0 and math.isfinite(float(log_determinant)), "non-positive shrunk covariance")
    inverse = np.linalg.inv(shrunk)
    condition = float(np.linalg.cond(shrunk))
    require(np.isfinite(inverse).all() and condition < 1e8, "unstable shrunk covariance")
    effective_n = float(1.0 / np.sum(weight * weight))
    return mean, inverse, {
        "log_determinant": float(log_determinant),
        "condition_number": condition,
        "effective_duplicate_balanced_n": effective_n,
        "average_variance": average_variance,
    }


class RobustCopulaQDA:
    def fit(self, frame: pd.DataFrame) -> "RobustCopulaQDA":
        require(len(frame) >= 400 and frame.y.nunique() == 2, "V86 past cut support insufficient")
        self.gaussianizer = EmpiricalGaussianizer().fit(frame)
        matrix, transform_audit = self.gaussianizer.transform(frame)
        target = frame.y.to_numpy(int)
        weights = duplicate_weights(frame)
        self.class_models: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        class_audits: dict[str, Any] = {}
        class_mass = []
        for label in (0, 1):
            mask = target == label
            require(int(mask.sum()) >= 120, f"V86 class {label} support insufficient")
            mean, inverse, audit = covariance_components(matrix[mask], weights[mask])
            covariance_sign, log_inverse_determinant = np.linalg.slogdet(inverse)
            require(covariance_sign > 0, "inverse covariance sign failed")
            log_determinant = -float(log_inverse_determinant)
            self.class_models[label] = (mean, inverse, log_determinant)
            mass = float(weights[mask].sum())
            class_mass.append(mass)
            class_audits[str(label)] = {
                **audit,
                "raw_n": int(mask.sum()),
                "duplicate_weight_mass": mass,
            }
        total_mass = max(sum(class_mass), 1e-12)
        self.log_prior_odds = float(math.log((class_mass[1] + 0.5) / (class_mass[0] + 0.5)))
        training_raw = self.raw_score(matrix)
        self.score_scale = max(float(np.std(training_raw)), 1.0)
        self.audit = {
            "train_n": len(frame),
            "feature_n": len(FEATURES),
            "numeric_state_only": True,
            "categorical_features": 0,
            "text_features": 0,
            "source_or_ticker_features": 0,
            "empirical_marginals_fit_on_training_cut_only": True,
            "target_rows_used_in_transform_fit": False,
            "duplicate_balanced": True,
            "fixed_covariance_weights": {
                "full": FULL_COVARIANCE_WEIGHT,
                "diagonal": DIAGONAL_COVARIANCE_WEIGHT,
                "spherical": SPHERICAL_COVARIANCE_WEIGHT,
            },
            "train_transform": transform_audit,
            "class_audits": class_audits,
            "weighted_positive_prevalence": float(class_mass[1] / total_mass),
            "log_prior_odds": self.log_prior_odds,
            "score_scale": self.score_scale,
            "generative_class_log_density_ratio": True,
        }
        return self

    def raw_score(self, matrix: np.ndarray) -> np.ndarray:
        log_density: dict[int, np.ndarray] = {}
        for label in (0, 1):
            mean, inverse, log_determinant = self.class_models[label]
            centered = matrix - mean
            quadratic = np.einsum("ij,jk,ik->i", centered, inverse, centered, optimize=True)
            log_density[label] = -0.5 * (log_determinant + quadratic)
        return log_density[1] - log_density[0] + self.log_prior_odds

    def predict_probability(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        matrix, target_transform = self.gaussianizer.transform(frame)
        raw = self.raw_score(matrix)
        standardized = np.clip(raw / self.score_scale, -10.0, 10.0)
        probability = np.clip(expit(standardized), 1e-5, 1.0 - 1e-5)
        require(np.isfinite(probability).all(), "V86 non-finite probability")
        return probability, {
            "target_n": len(frame),
            "target_transform": target_transform,
            "target_rows_used_in_fit": False,
            "raw_score_mean": float(np.mean(raw)),
            "raw_score_std": float(np.std(raw)),
            "probability_mean": float(np.mean(probability)),
            "probability_sha256": array_sha256(probability),
        }


def copula_qda_predictions(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model = RobustCopulaQDA().fit(train)
    probability, prediction_audit = model.predict_probability(target)
    return probability, {**model.audit, "prediction": prediction_audit}


def architecture_probability(baseline: np.ndarray, expert: np.ndarray, architecture: dict[str, Any]) -> np.ndarray:
    weight = float(architecture["weight"])
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    expert_logit = logit(np.clip(np.asarray(expert, float), 1e-5, 1.0 - 1e-5))
    probability = expit((1.0 - weight) * base_logit + weight * expert_logit)
    require(np.isfinite(probability).all(), "V86 blend returned non-finite probability")
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
    require(len(inner_train) >= 400, "inner V86 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V86 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expert, model_audit = copula_qda_predictions(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; maximize 2*AUC+BA subject to BA delta>=-0.005 and net delta>=-0.0005; exact no-op and two fixed blends",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "covariance_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v86_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V86 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        expert, outer_model_audit = copula_qda_predictions(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v86_model"] = selected["name"]
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
        evidence["copula_qda_prob"] = expert
        evidence["candidate_prob"] = candidate
        evidence["v86_model"] = selected["name"]
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
            f"[V86 COPULA-QDA] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V86 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V86 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V86 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d")
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V86 bootstrap blocks")
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
    require(len(auc_delta) >= int(0.90 * draws), "V86 bootstrap lost too many draws")

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
        "method": "paired market-time block bootstrap over inner-locked copula-QDA blend",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V86/controller gate mismatch")
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
    generative_valid = all(
        audit["inner_model_audit"]["generative_class_log_density_ratio"]
        and audit["outer_model_audit"]["generative_class_log_density_ratio"]
        and audit["inner_model_audit"]["target_rows_used_in_transform_fit"] is False
        and audit["outer_model_audit"]["target_rows_used_in_transform_fit"] is False
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
        "copula_transform_and_generative_qda_verified": generative_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V86 fallback is not exact complete V69")
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
            "version": "V86",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V86_DIAGNOSTIC_COPULA_QDA": evaluation["candidate_summary"],
                "V86_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact,
            "authority_audit": authority,
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V86",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "opened_dev_only": True,
            "selected": selected_contract,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V86",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
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
        "ROBUST_GAUSSIAN_COPULA_QDA_REPORT.json": {
            "version": "V86",
            "hypothesis": HYPOTHESIS,
            "features": FEATURES,
            "architectures": ARCHITECTURES,
            "covariance_weights": {
                "full": FULL_COVARIANCE_WEIGHT,
                "diagonal": DIAGONAL_COVARIANCE_WEIGHT,
                "spherical": SPHERICAL_COVARIANCE_WEIGHT,
            },
            "nested_fold_audits": audits,
            "evaluation": compact,
            "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V86",
            "status": "MATCH",
            "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION,
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED",
            "seal_authorized": False,
            "completed_at": now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"],
        "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "max_covariance_condition": max(
            class_audit["condition_number"]
            for class_audit in audit["outer_model_audit"]["class_audits"].values()
        ),
        "strict_35m_embargo": True,
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V86_COPULA_QDA_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V86_COPULA_QDA_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V86_DIAGNOSTIC_COPULA_QDA_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V86_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out),
        "artifact_count": len(files) + 1,
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
            "status": "AUDIT_OK",
            "hypothesis": HYPOTHESIS,
            "authority": authority,
            "runtime": runtime,
            "dev_rows": len(dev),
            "v69_rows": len(champion),
            "features": FEATURES,
            "feature_n": len(FEATURES),
            "causal_numeric_state_only": True,
            "categorical_or_text_features": False,
            "empirical_copula_training_cut_only": True,
            "generative_qda": True,
            "fixed_covariance_weights": {
                "full": FULL_COVARIANCE_WEIGHT,
                "diagonal": DIAGONAL_COVARIANCE_WEIGHT,
                "spherical": SPHERICAL_COVARIANCE_WEIGHT,
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
        "hypothesis": HYPOTHESIS,
        "runtime": runtime,
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
