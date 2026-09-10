"""V83 source-specific shape-constrained additive direction challenger.

This model is intentionally transparent and interaction-free.  A small set of
causal numeric features is expanded into at most four saturating piecewise-
linear ramps plus a missing indicator.  Economically defensible signs are
enforced for benchmark returns and explicit positive/negative event flags;
other features receive low-degree unconstrained additive shapes.  Shared
training-cut knots support pooled and source-family heads without ticker,
categorical interactions, text, retrieval, trees, or neural representations.

A second fit on the earlier portion of the same past cut measures temporal
shape extrapolation stability.  Inner-past OOF evidence chooses exact V69, one
of two fixed additive blends, or a fixed stability-gated blend.  Every split
has a strict 35-minute embargo; outer labels are evaluation-only.  Inputs are
immutable V36 DEV and atomic output_V69 only.  Material failure restores the
exact entire V69 frame and the controller recomputes all 14 canonical gates.
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
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import autonomous_v37plus as controller
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "staging" / "V83_SOURCE_SHAPE_CONSTRAINED_ADDITIVE_V1"
)

VERSION = 83
HYPOTHESIS = "SOURCE_SHAPE_CONSTRAINED_ADDITIVE_DIRECTION_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_MARKET_FOLD_ROWS = {"US": 1595, "KR": 297}
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
EARLY_STABILITY_FRACTION = 0.70
SOURCE_MIN_ROWS = 220
L2_PENALTY = 0.08
SHAPE_SMOOTH_PENALTY = 0.04
STABILITY_LOGIT_SCALE = 0.50
SEED = 8301
BOOTSTRAP_DRAWS = 2000
COST = 0.002

MONOTONIC_SIGNS = {
    "event_positive_kw": 1,
    "event_negative_kw": -1,
    "benchmark_ret_5": 1,
    "benchmark_ret_15": 1,
    "benchmark_ret_30": 1,
}

UNCONSTRAINED_FEATURES = (
    "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
    "pre_vol_5", "pre_vol_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "range_10m", "range_30m", "close_position_30", "intraday_ret_open",
    "return_autocorr_30", "up_fraction_30", "trend_slope_30",
    "trend_slope_60", "vwap_distance_30", "minutes_from_open",
    "minutes_to_close", "event_financing_kw", "event_trial_kw",
    "event_regulatory_kw", "event_ma_kw", "event_earnings_kw",
)

FEATURES = tuple(MONOTONIC_SIGNS) + UNCONSTRAINED_FEATURES

ARCHITECTURES = (
    {"name": "SOURCE_ADDITIVE_W0.25", "mode": "fixed", "weight": 0.25},
    {"name": "SOURCE_ADDITIVE_W0.50", "mode": "fixed", "weight": 0.50},
    {"name": "SHAPE_STABILITY_GATED_W0.50", "mode": "stability", "weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
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
    atomic_bytes((json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"), path)


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
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
        ],
        "v36_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": sha256(commit_path),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_oof_sha256": sha256(V69_PATH),
        "manifest_files_verified": len(manifest.get("files", {})),
        "failed_output_loaded": False, "dev_extension_loaded": False,
        "role_assignment_loaded": False, "research_seal_loaded": False,
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
    dev["event_time_utc"] = pd.to_datetime(dev.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    require(all(column in dev.columns for column in FEATURES), "V83 causal feature missing")
    return dev.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True), champion


class AdditiveBasis:
    def __init__(self):
        self.edges: dict[str, list[float]] = {}
        self.ramp_indices: dict[str, list[int]] = {}
        self.missing_indices: dict[str, int] = {}
        self.feature_names: list[str] = ["INTERCEPT"]
        self.bounds: list[tuple[float | None, float | None]] = [(None, None)]

    def fit(self, frame: pd.DataFrame) -> "AdditiveBasis":
        position = 1
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite_values = values[np.isfinite(values)]
            require(len(finite_values) >= 200, f"{column} finite support insufficient")
            unique = np.unique(finite_values)
            if len(unique) <= 4:
                edge = [float(unique.min()), float(unique.max())]
            else:
                edge = list(dict.fromkeys(float(value) for value in np.quantile(finite_values, [0.02, 0.25, 0.50, 0.75, 0.98])))
            if len(edge) < 2 or not edge[-1] > edge[0]:
                edge = [float(finite_values.min()), float(finite_values.max() + 1e-12)]
            self.edges[column] = edge
            indices = []
            sign = MONOTONIC_SIGNS.get(column, 0)
            for interval in range(len(edge) - 1):
                indices.append(position)
                self.feature_names.append(f"{column}::ramp_{interval}")
                self.bounds.append((0.0, None) if sign > 0 else ((None, 0.0) if sign < 0 else (None, None)))
                position += 1
            self.ramp_indices[column] = indices
            self.missing_indices[column] = position
            self.feature_names.append(f"{column}::missing")
            self.bounds.append((None, None))
            position += 1
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.zeros((len(frame), len(self.feature_names)), dtype=np.float64)
        matrix[:, 0] = 1.0
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.edges[column][0], values)
            for index, (left, right) in zip(self.ramp_indices[column], zip(self.edges[column][:-1], self.edges[column][1:])):
                matrix[:, index] = np.clip((filled - left) / max(right - left, 1e-12), 0.0, 1.0)
            matrix[:, self.missing_indices[column]] = missing.astype(float)
        return matrix

    def smooth_pairs(self) -> list[tuple[int, int]]:
        return [(left, right) for indices in self.ramp_indices.values() for left, right in zip(indices[:-1], indices[1:])]


def fit_logistic_additive(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    basis: AdditiveBasis,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(target, float)
    weight = np.asarray(weight, float)
    class_count = np.bincount(target.astype(int), minlength=2).astype(float)
    require((class_count > 0).all(), "source additive fit has a single class")
    class_weight = np.where(target > 0.5, len(target) / (2.0 * class_count[1]), len(target) / (2.0 * class_count[0]))
    weight = weight * class_weight
    weight = weight / max(weight.sum(), 1e-12)
    smooth_pairs = basis.smooth_pairs()

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = matrix @ coefficient
        loss = float(np.sum(weight * (np.logaddexp(0.0, score) - target * score)))
        gradient = matrix.T @ (weight * (expit(score) - target))
        loss += 0.5 * L2_PENALTY * float(np.dot(coefficient[1:], coefficient[1:]))
        gradient[1:] += L2_PENALTY * coefficient[1:]
        for left, right in smooth_pairs:
            difference = coefficient[right] - coefficient[left]
            loss += 0.5 * SHAPE_SMOOTH_PENALTY * difference * difference
            gradient[right] += SHAPE_SMOOTH_PENALTY * difference
            gradient[left] -= SHAPE_SMOOTH_PENALTY * difference
        return loss, gradient

    initial = np.zeros(matrix.shape[1], dtype=float)
    prevalence = float(np.average(target, weights=weight))
    initial[0] = float(logit(np.clip(prevalence, 1e-5, 1.0 - 1e-5)))
    result = minimize(
        objective, initial, method="L-BFGS-B", jac=True, bounds=basis.bounds,
        options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
    )
    require(np.isfinite(result.x).all() and math.isfinite(float(result.fun)), "additive optimization failed")
    violations = 0
    for column, sign in MONOTONIC_SIGNS.items():
        coefficients = result.x[basis.ramp_indices[column]]
        violations += int(np.sum(coefficients < -1e-10)) if sign > 0 else int(np.sum(coefficients > 1e-10))
    return result.x, {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
        "objective": float(result.fun), "coefficient_n": len(result.x),
        "monotonic_constraint_violations": violations,
    }


def causal_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    return 1.0 / np.maximum(count, 1.0)


def fit_heads(
    train: pd.DataFrame,
    matrix: np.ndarray,
    basis: AdditiveBasis,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    heads: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    pooled, pooled_audit = fit_logistic_additive(matrix, train.y.to_numpy(int), causal_weights(train), basis)
    heads["__POOLED__"] = pooled
    audits["__POOLED__"] = {**pooled_audit, "train_n": len(train), "fallback": False}
    for source, positions in train.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, int)
        source_frame = train.iloc[index]
        if len(index) < SOURCE_MIN_ROWS or source_frame.y.nunique() < 2:
            audits[str(source)] = {"train_n": len(index), "fallback": True, "fallback_to": "__POOLED__"}
            continue
        coefficient, audit = fit_logistic_additive(matrix[index], source_frame.y.to_numpy(int), causal_weights(source_frame), basis)
        heads[str(source)] = coefficient
        audits[str(source)] = {**audit, "train_n": len(index), "fallback": False}
    return heads, audits


def predict_heads(frame: pd.DataFrame, matrix: np.ndarray, heads: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    probability = np.zeros(len(frame), dtype=float)
    routing: dict[str, int] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, int)
        coefficient = heads.get(str(source), heads["__POOLED__"])
        probability[index] = expit(matrix[index] @ coefficient)
        routing[str(source)] = len(index)
    require(np.isfinite(probability).all(), "non-finite additive probability")
    return np.clip(probability, 1e-5, 1.0 - 1e-5), routing


def additive_predictions(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    basis = AdditiveBasis().fit(train)
    train_matrix = basis.transform(train)
    target_matrix = basis.transform(target)
    full_heads, full_audits = fit_heads(train, train_matrix, basis)
    full_probability, routing = predict_heads(target, target_matrix, full_heads)
    split = max(500, int(math.floor(len(train) * EARLY_STABILITY_FRACTION)))
    early = train.iloc[:split].copy()
    early_matrix = train_matrix[:split]
    early_heads, early_audits = fit_heads(early, early_matrix, basis)
    early_probability, _ = predict_heads(target, target_matrix, early_heads)
    logit_distance = np.abs(logit(full_probability) - logit(early_probability))
    stability = np.exp(-logit_distance / STABILITY_LOGIT_SCALE)
    violations = sum(audit.get("monotonic_constraint_violations", 0) for audit in full_audits.values())
    violations += sum(audit.get("monotonic_constraint_violations", 0) for audit in early_audits.values())
    require(violations == 0, "monotonic coefficient bound violated")
    return full_probability, early_probability, stability, {
        "train_n": len(train), "early_train_n": len(early), "target_n": len(target),
        "feature_n": len(FEATURES), "basis_column_n": len(basis.feature_names),
        "maximum_ramps_per_feature": 4, "additive_only": True,
        "interaction_terms": 0, "ticker_feature": False, "text_feature": False,
        "monotonic_signs": MONOTONIC_SIGNS,
        "monotonic_constraint_violations": violations,
        "source_head_audits": full_audits, "early_source_head_audits": early_audits,
        "target_source_routing": routing,
        "mean_shape_stability": float(stability.mean()),
        "p10_shape_stability": float(np.quantile(stability, 0.10)),
        "full_probability_sha256": array_sha256(full_probability),
        "early_probability_sha256": array_sha256(early_probability),
    }


def architecture_probability(
    baseline: np.ndarray,
    additive: np.ndarray,
    stability: np.ndarray,
    architecture: dict[str, Any],
) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    additive_logit = logit(np.clip(np.asarray(additive, float), 1e-5, 1.0 - 1e-5))
    row_weight = np.full(len(baseline), float(architecture["weight"]))
    if architecture["mode"] == "stability":
        row_weight *= np.asarray(stability, float)
    return expit((1.0 - row_weight) * base_logit + row_weight * additive_logit)


def metric(frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    y = frame.y.to_numpy(int)
    confidence = np.asarray(confidence, float)
    high = np.asarray(high, bool)
    correct = (prediction == y).astype(int)
    signed_net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return {
        "n": len(frame), "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(correct.mean()), "pred_up": float(prediction.mean()),
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
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end, "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0, "strict_35m_embargo": True,
    }


def prior_train(dev: pd.DataFrame, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = dev.loc[dev.event_time_utc < boundary - EMBARGO].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    return train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)


def aligned_dev(dev: pd.DataFrame, champion_rows: pd.DataFrame) -> pd.DataFrame:
    aligned = dev.set_index("event_id").loc[champion_rows.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion_rows.y.to_numpy(int)), "aligned labels differ")
    return aligned


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, inner_start, inner_valid)
    require(len(inner_train) >= 700, "inner additive training cut too small")
    audit = chronology(inner_train, inner_valid, f"V83 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    additive, _, stability, model_audit = additive_predictions(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"]),
    }]
    for architecture in ARCHITECTURES:
        probability = architecture_probability(baseline, additive, stability, architecture)
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture,
            "metrics": values, "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; maximize 2*AUC+BA+10*all-trade-net with BA delta>=-0.005 and net delta>=-0.0005; exact no-op and three fixed additive blends",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v83_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in fold_specs[:1] if smoke else fold_specs:
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V83 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        additive, early, stability, outer_model_audit = additive_predictions(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else architecture_probability(baseline, additive, stability, selected["architecture"])
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v83_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                architecture_probability(baseline, additive, stability, architecture),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["source_additive_prob"] = additive
        evidence["early_shape_prob"] = early
        evidence["shape_stability"] = stability
        evidence["candidate_prob"] = candidate
        evidence["v83_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V83 SHAPE] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V83 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V83 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V83 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V83 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        ba_delta.append(float(balanced_accuracy_score(y, candidate >= 0.5) - balanced_accuracy_score(y, base >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(base >= 0.5, 1.0, -1.0) * returns)))
    require(len(auc_delta) >= int(0.90 * draws), "V83 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked additive architecture",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V83/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, evidence.high_conf)
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, evidence.high_conf)
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    shape_valid = all(
        audit["outer_model_audit"]["monotonic_constraint_violations"] == 0
        and audit["outer_model_audit"]["additive_only"]
        and audit["outer_model_audit"]["interaction_terms"] == 0
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
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "shape_constraints_and_additivity_verified": shape_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V83 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "immutability": {
            "confidence_exact": confidence_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {"activated": not material_pass, "policy": "exact V69 entire frame" if not material_pass else None, "exact_frame_verified": bool(material_pass or selected.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def write_outputs(authority: dict[str, Any], champion: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any], out: Path) -> dict[str, Any]:
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
            "version": "V83", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V83_DIAGNOSTIC_ADDITIVE": evaluation["candidate_summary"], "V83_FAIL_CLOSED_SELECTED": selected_contract},
            "evaluation": compact, "authority_audit": authority,
            "validation": "source-specific interaction-free additive shapes; fixed monotonic signs; shared cut-local knots; early/full shape stability; nested 35-minute chronology",
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V83", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V83", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected_contract["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "net_delta": evaluation["outer_net_delta"],
                "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta"]["probability_gt_zero"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_SHAPE_CONSTRAINED_ADDITIVE_REPORT.json": {
            "version": "V83", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "monotonic_signs": MONOTONIC_SIGNS,
            "architectures": ARCHITECTURES, "interaction_terms": 0,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V83", "status": "MATCH", "canonical_check_count": 14,
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
        "mean_shape_stability": audit["outer_model_audit"]["mean_shape_stability"],
        "monotonic_violations": audit["outer_model_audit"]["monotonic_constraint_violations"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V83_SHAPE_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V83_ADDITIVE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V83_DIAGNOSTIC_ADDITIVE_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V83_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


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
    runtime_limits.configure()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "monotonic_signs": MONOTONIC_SIGNS,
            "source_specific_heads": True, "maximum_ramps_per_feature": 4,
            "interaction_terms": 0, "ticker_feature": False, "text_feature": False,
            "canonical_controller_gate_checks": 14, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
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
        summary["shape_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, champion, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
