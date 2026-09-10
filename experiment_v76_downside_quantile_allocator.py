"""V76 downside-quantile economic allocator over the frozen V69 direction.

This hypothesis is deliberately distributional and decision-focused.  It does
not learn a direction, P(correct), P(material move), or a mean signed return.
Instead, pooled causal history estimates the 10th, 25th, and 50th conditional
quantiles of net return obtained by following the already outer-OOF V69
direction.  Fixed lower-tail utility formulas allocate high-confidence trades.

The only data authorities are immutable V36 DEV and the atomic output_V69
commit.  Every split is chronological with a strict 35-minute embargo;
near-event groups are excluded.  A small fixed policy set is chosen on an
inner-past slice.  Outer labels are evaluation-only.  Probability and direction
remain byte-identical to V69.  Unless every material check passes, the complete
selected frame is an exact V69 fallback.  The controller recomputes its 14
canonical checks from raw selected-frame metrics.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    else ROOT / "staging" / "V76_DOWNSIDE_QUANTILE_ALLOCATOR_V1"
)

VERSION = 76
HYPOTHESIS = "FROZEN_DIRECTION_DOWNSIDE_QUANTILE_ALLOCATOR_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_MARKET_FOLD_ROWS = {"US": 1595, "KR": 297}
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.25
QUANTILES = (0.10, 0.25, 0.50)
COVERAGE_LEVELS = (0.10, 0.15, 0.20)
SEED = 7601
BOOTSTRAP_DRAWS = 2000
COST = 0.002

STATE_COLUMNS = (
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "volume_ratio_10_60", "range_5m", "range_10m", "range_30m",
    "range_60m", "close_position_10", "close_position_30",
    "close_position_60", "entry_bar_ret", "entry_bar_range",
    "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
    "up_fraction_30", "trend_slope_30", "trend_slope_60",
    "vwap_distance_30", "minutes_from_open", "minutes_to_close",
    "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30",
    "event_positive_kw", "event_negative_kw", "event_financing_kw",
    "event_trial_kw", "event_regulatory_kw", "event_ma_kw",
    "event_earnings_kw", "headline_len", "body_len",
)

FROZEN_EXPERT_COLUMNS = (
    "prob", "confidence_signal", "opportunity_confidence",
    "prob_opportunity_direction", "confidence_opportunity_direction",
    "prob_event_form_direction", "confidence_event_form_direction",
    "prob_fixed_numeric_direction", "confidence_fixed_numeric_direction",
)

CATEGORICAL_COLUMNS = (
    "market", "source_family", "form", "event_type", "source",
    "volatility_regime", "activity_regime", "trend_regime",
    "source_volatility_regime", "source_activity_regime",
)

UTILITY_SPECS = (
    {"name": "Q10_FLOOR", "weights": {0.10: 1.0}},
    {"name": "LOWER_TAIL_BAND", "weights": {0.10: 0.65, 0.25: 0.35}},
    {"name": "MEDIAN_DOWNSIDE_PENALTY", "weights": {0.10: 0.75, 0.50: 0.25}},
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


def number_or(value: Any, default: float) -> float:
    result = finite(value)
    return float(default) if result is None else result


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda item: str(item).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, np.asarray(score, float))) if np.unique(target).size == 2 else None


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit_path.is_file() and manifest_path.is_file(), "V69 atomic authority incomplete")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == sha256(manifest_path), "V69 COMMIT/manifest mismatch")
    require(commit.get("data_sha") == EXPECTED_V36_SHA256, "V69 is not bound to immutable V36 DEV")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment mismatch")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 artifact hash changed: {name}")
    require(V69_PATH.name in manifest.get("files", {}), "V69 selected OOF is not manifested")
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
        ],
        "v36_dev_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": sha256(commit_path),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_selected_oof_sha256": sha256(V69_PATH),
        "experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "manifest_files_verified": len(manifest.get("files", {})),
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_reserve_loaded": False,
    }


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    champion = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_V36_ROWS and len(champion) == EXPECTED_V69_ROWS, "authority row count changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 is not a V36 DEV subset")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15), "V36/V69 returns differ")
    require(aligned.source_family.astype(str).equals(champion.source_family.astype(str)), "V36/V69 source differs")
    work = champion.copy()
    for column in ("source", "event_type", *STATE_COLUMNS):
        work[column] = aligned[column].to_numpy()
    work["event_time_utc"] = pd.to_datetime(work.event_time_utc, utc=True)
    work["high_conf"] = bool_series(work.high_conf)
    work["frozen_direction"] = work.prob.to_numpy(float) >= 0.5
    work["correct_target"] = (work.frozen_direction.to_numpy(bool) == work.y.to_numpy(int)).astype(int)
    work["signed_net"] = np.where(work.frozen_direction, 1.0, -1.0) * work.fwd_ret_30m.to_numpy(float) - COST
    work["frozen_margin"] = np.abs(work.prob.to_numpy(float) - 0.5)
    probabilities = [column for column in FROZEN_EXPERT_COLUMNS if column.startswith("prob_") and column in work]
    if probabilities:
        matrix = np.column_stack([pd.to_numeric(work[column], errors="coerce").fillna(0.5) for column in probabilities])
        directions = matrix >= 0.5
        work["expert_direction_agreement"] = np.mean(directions == work.frozen_direction.to_numpy(bool)[:, None], axis=1)
        work["expert_probability_dispersion"] = np.std(matrix, axis=1)
    else:
        work["expert_direction_agreement"] = 1.0
        work["expert_probability_dispersion"] = 0.0
    require(work.signed_net.notna().all(), "non-finite signed-net target")
    work = work.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    return work, champion


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) and len(valid), f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: strict 35-minute embargo failed")
    group_overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    cluster_overlap = set(train.text_cluster_id.astype(str)) & set(valid.text_cluster_id.astype(str))
    require(not group_overlap, f"{label}: event-group leakage")
    require(not cluster_overlap, f"{label}: text-cluster leakage")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end, "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0, "text_cluster_overlap_n": 0,
        "strict_35m_embargo": True,
    }


def prior_rows(frame: pd.DataFrame, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = frame.loc[frame.event_time_utc < boundary - EMBARGO].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    train = train.loc[~train.text_cluster_id.astype(str).isin(set(valid.text_cluster_id.astype(str)))].copy()
    return train


def outer_folds(frame: pd.DataFrame, smoke: bool) -> list[tuple[pd.DataFrame, pd.DataFrame, str, int]]:
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str, int]] = []
    for market in ("US", "KR"):
        for fold in (2, 3, 4):
            valid = frame.loc[frame.market.eq(market) & frame.fold.eq(fold)].copy()
            valid = valid.sort_values(["event_time_utc", "event_id"], kind="stable")
            require(len(valid) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
            boundary = pd.Timestamp(valid.event_time_utc.min())
            train = prior_rows(frame, boundary, valid)
            require(len(train) >= 500 and train.signed_net.nunique() > 10, f"{market} fold {fold} prior support insufficient")
            chronology(train, valid, f"V76 outer {market} fold {fold}")
            folds.append((train, valid, market, fold))
    return folds[:1] if smoke else folds


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable")
    split = max(350, int(math.floor(len(ordered) * (1.0 - INNER_VALID_FRACTION))))
    valid = ordered.iloc[split:].copy()
    boundary = pd.Timestamp(valid.event_time_utc.min())
    base = prior_rows(ordered, boundary, valid)
    require(len(base) >= 300 and len(valid) >= 100, "V76 inner split insufficient")
    return base, valid, chronology(base, valid, "V76 inner")


class FeatureBuilder:
    def __init__(self):
        self.regime_edges: dict[str, tuple[float, float]] = {}
        self.numeric_columns = list(STATE_COLUMNS) + list(FROZEN_EXPERT_COLUMNS) + [
            "frozen_margin", "expert_direction_agreement", "expert_probability_dispersion",
        ]
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=False)),
        ])
        self.transformer = ColumnTransformer([
            ("numeric", numeric, self.numeric_columns),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ], sparse_threshold=0.0)

    def add_regimes(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        output = frame.copy()
        definitions = {
            "volatility_regime": pd.to_numeric(output.pre_vol_30, errors="coerce"),
            "activity_regime": pd.to_numeric(output.volume_ratio_5_30, errors="coerce"),
            "trend_regime": pd.to_numeric(output.trend_slope_30, errors="coerce"),
        }
        for name, values in definitions.items():
            if fit:
                finite_values = values[np.isfinite(values)]
                require(len(finite_values) >= 100, f"{name} has insufficient finite support")
                self.regime_edges[name] = tuple(float(value) for value in np.quantile(finite_values, [1 / 3, 2 / 3]))
            low, high = self.regime_edges[name]
            output[name] = np.where(values <= low, "LOW", np.where(values <= high, "MID", "HIGH"))
            output.loc[~np.isfinite(values), name] = "MISSING"
        output["source_volatility_regime"] = output.source_family.astype(str) + "|" + output.volatility_regime.astype(str)
        output["source_activity_regime"] = output.source_family.astype(str) + "|" + output.activity_regime.astype(str)
        return output

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.transformer.fit_transform(self.add_regimes(frame, True)), dtype=np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.transformer.transform(self.add_regimes(frame, False)), dtype=np.float32)


class DownsideQuantileModel:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def fit_predict(self, train: pd.DataFrame, target: pd.DataFrame) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[str, Any]]:
        features = FeatureBuilder()
        train_x = features.fit_transform(train)
        target_x = features.transform(target)
        weight = pd.to_numeric(train.cluster_weight, errors="coerce").fillna(1.0).to_numpy(float)
        train_predictions: dict[float, np.ndarray] = {}
        target_predictions: dict[float, np.ndarray] = {}
        iterations: dict[str, int] = {}
        for offset, quantile in enumerate(QUANTILES):
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, learning_rate=0.055,
                max_iter=140, max_leaf_nodes=15, min_samples_leaf=45,
                l2_regularization=1.0, early_stopping=False,
                random_state=self.seed + offset,
            )
            model.fit(train_x, train.signed_net.to_numpy(float), sample_weight=weight)
            train_predictions[quantile] = model.predict(train_x)
            target_predictions[quantile] = model.predict(target_x)
            iterations[str(quantile)] = int(model.n_iter_)
        train_stack = np.sort(np.column_stack([train_predictions[q] for q in QUANTILES]), axis=1)
        target_stack = np.sort(np.column_stack([target_predictions[q] for q in QUANTILES]), axis=1)
        train_predictions = {q: train_stack[:, index] for index, q in enumerate(QUANTILES)}
        target_predictions = {q: target_stack[:, index] for index, q in enumerate(QUANTILES)}
        require(all(np.isfinite(values).all() for values in target_predictions.values()), "non-finite quantile prediction")
        return train_predictions, target_predictions, {
            "train_n": len(train), "target_n": len(target), "feature_n": train_x.shape[1],
            "quantiles": QUANTILES, "iterations": iterations,
            "target": "V69 frozen-direction signed net return",
            "pooled_source_regime_features": True,
            "direction_model": False, "correctness_model": False,
            "material_move_model": False, "mean_return_model": False,
        }


def utility(predictions: dict[float, np.ndarray], spec: dict[str, Any]) -> np.ndarray:
    return sum(float(weight) * np.asarray(predictions[float(q)], float) for q, weight in spec["weights"].items())


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, float))
    return np.searchsorted(ordered, np.asarray(values, float), side="right") / max(1, len(ordered))


def lower_tail_mean(values: np.ndarray, fraction: float = 0.20) -> float | None:
    values = np.asarray(values, float)
    if not len(values):
        return None
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.sort(values)[:count].mean())


def mask_metrics(frame: pd.DataFrame, high: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    high = np.asarray(high, bool)
    net = frame.signed_net.to_numpy(float)
    correct = frame.correct_target.to_numpy(int)
    selected = net[high]
    return {
        "n": len(frame), "selected_n": int(high.sum()), "coverage": float(high.mean()),
        "accuracy": float(correct[high].mean()) if high.any() else None,
        "mean_signed_net": float(selected.mean()) if high.any() else None,
        "median_signed_net": float(np.median(selected)) if high.any() else None,
        "realized_lower_tail_20_mean": lower_tail_mean(selected),
        "realized_p10": float(np.quantile(selected, 0.10)) if high.any() else None,
        "positive_net_rate": float(np.mean(selected > 0.0)) if high.any() else None,
        "confidence_correctness_auc": safe_auc(correct, np.asarray(confidence, float)),
    }


def choose_inner(train: pd.DataFrame, valid: pd.DataFrame, train_q: dict[float, np.ndarray], valid_q: dict[float, np.ndarray]) -> dict[str, Any]:
    baseline = mask_metrics(valid, valid.high_conf.to_numpy(bool), valid.confidence_signal.to_numpy(float))
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "spec": None, "coverage_target": None,
        "metrics": baseline, "net_delta": 0.0, "lower_tail_delta": 0.0,
        "accuracy_delta": 0.0, "eligible": True,
        "objective": float(number_or(baseline["realized_lower_tail_20_mean"], -1.0) + 0.5 * number_or(baseline["mean_signed_net"], -1.0)),
    }]
    for spec in UTILITY_SPECS:
        train_score = utility(train_q, spec)
        valid_score = utility(valid_q, spec)
        confidence = percentile_against(train_score, valid_score)
        for coverage in COVERAGE_LEVELS:
            high = confidence >= 1.0 - float(coverage)
            values = mask_metrics(valid, high, confidence)
            net_delta = number_or(values["mean_signed_net"], -1.0) - number_or(baseline["mean_signed_net"], -1.0)
            tail_delta = number_or(values["realized_lower_tail_20_mean"], -1.0) - number_or(baseline["realized_lower_tail_20_mean"], -1.0)
            accuracy_delta = number_or(values["accuracy"], 0.0) - number_or(baseline["accuracy"], 0.0)
            eligible = bool(values["selected_n"] >= 25 and 0.06 <= values["coverage"] <= 0.30 and accuracy_delta >= -0.03)
            trials.append({
                "name": f"{spec['name']}_C{coverage:.2f}", "spec": spec,
                "coverage_target": float(coverage), "metrics": values,
                "net_delta": net_delta, "lower_tail_delta": tail_delta,
                "accuracy_delta": accuracy_delta, "eligible": eligible,
                "objective": float(number_or(values["realized_lower_tail_20_mean"], -1.0) + 0.5 * number_or(values["mean_signed_net"], -1.0)),
            })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (
            trial["lower_tail_delta"] >= 0.0, trial["net_delta"] >= 0.0,
            trial["objective"], trial["accuracy_delta"],
            -abs((trial["metrics"]["coverage"] or 0.0) - 0.15),
        ),
    )
    return {
        "selection_rule": "inner-past only: prefer nonnegative realized lower-tail and mean-net deltas, then maximize lower-tail+0.5*mean; fixed 3x3 grid plus exact no-op",
        "baseline": baseline, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }


def run_nested(frame: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = frame.copy()
    diagnostic["v76_applied"] = False
    diagnostic["v76_policy"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for train, valid, market, fold in outer_folds(frame, smoke):
        outer_chronology = chronology(train, valid, f"V76 outer {market} fold {fold}")
        inner_train, inner_valid, inner_chronology = inner_split(train)
        inner_train_q, inner_valid_q, inner_model = DownsideQuantileModel(SEED + fold).fit_predict(inner_train, inner_valid)
        policy = choose_inner(inner_train, inner_valid, inner_train_q, inner_valid_q)
        selected = policy["selected"]
        candidate_high = valid.high_conf.to_numpy(bool).copy()
        candidate_confidence = valid.confidence_signal.to_numpy(float).copy()
        outer_model = None
        predicted_quantiles = {q: np.full(len(valid), np.nan) for q in QUANTILES}
        if selected["spec"] is not None:
            train_q, valid_q, outer_model = DownsideQuantileModel(SEED + 100 + fold).fit_predict(train, valid)
            score_train = utility(train_q, selected["spec"])
            score_valid = utility(valid_q, selected["spec"])
            candidate_confidence = percentile_against(score_train, score_valid)
            candidate_high = candidate_confidence >= 1.0 - float(selected["coverage_target"])
            predicted_quantiles = valid_q
            positions = diagnostic.index[diagnostic.event_id.isin(set(valid.event_id))]
            confidence_map = dict(zip(valid.event_id, candidate_confidence))
            high_map = dict(zip(valid.event_id, candidate_high))
            diagnostic.loc[positions, "confidence_signal"] = diagnostic.loc[positions, "event_id"].map(confidence_map)
            diagnostic.loc[positions, "high_conf"] = diagnostic.loc[positions, "event_id"].map(high_map).astype(bool)
            diagnostic.loc[positions, "family"] = "V76_DOWNSIDE_QUANTILE_ALLOCATOR"
            diagnostic.loc[positions, "v76_applied"] = True
            diagnostic.loc[positions, "v76_policy"] = selected["name"]
        evidence = valid[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "signed_net",
            "correct_target", "high_conf", "confidence_signal",
        ]].copy()
        evidence["candidate_high"] = candidate_high
        evidence["candidate_confidence"] = candidate_confidence
        for q in QUANTILES:
            evidence[f"predicted_net_q{int(q * 100):02d}"] = predicted_quantiles[q]
        evidence["v76_policy"] = selected["name"]
        evidence_parts.append(evidence)
        baseline_metrics = mask_metrics(valid, valid.high_conf, valid.confidence_signal)
        candidate_metrics = mask_metrics(valid, candidate_high, candidate_confidence)
        audits.append({
            "market": market, "fold": fold,
            "outer_chronology": outer_chronology, "inner_chronology": inner_chronology,
            "inner_model": inner_model, "outer_model": outer_model,
            "policy": policy, "outer_baseline": baseline_metrics,
            "outer_candidate": candidate_metrics,
            "outer_net_delta": number_or(candidate_metrics["mean_signed_net"], -1.0) - number_or(baseline_metrics["mean_signed_net"], -1.0),
            "outer_lower_tail_delta": number_or(candidate_metrics["realized_lower_tail_20_mean"], -1.0) - number_or(baseline_metrics["realized_lower_tail_20_mean"], -1.0),
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V76] market={market} fold={fold} policy={selected['name']} "
            f"inner_tail_delta={selected['lower_tail_delta']:+.6f} "
            f"outer_tail_delta={audits[-1]['outer_lower_tail_delta']:+.6f} "
            f"outer_net_delta={audits[-1]['outer_net_delta']:+.6f}", flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V76 evidence repeats an event")
    require(np.array_equal(frame.prob.to_numpy(float), diagnostic.prob.to_numpy(float)), "V76 changed V69 probability")
    require(np.array_equal(frame.prob.to_numpy(float) >= 0.5, diagnostic.prob.to_numpy(float) >= 0.5), "V76 changed V69 direction")
    return diagnostic, evidence, audits


def bootstrap_economics(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V76 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    candidate_net: list[float] = []
    net_delta: list[float] = []
    tail_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        candidate = sample.candidate_high.to_numpy(bool)
        baseline = sample.high_conf.to_numpy(bool)
        if candidate.sum() < 10 or baseline.sum() < 10:
            continue
        candidate_values = sample.loc[candidate, "signed_net"].to_numpy(float)
        baseline_values = sample.loc[baseline, "signed_net"].to_numpy(float)
        candidate_net.append(float(candidate_values.mean()))
        net_delta.append(float(candidate_values.mean() - baseline_values.mean()))
        tail_delta.append(float(number_or(lower_tail_mean(candidate_values), -1.0) - number_or(lower_tail_mean(baseline_values), -1.0)))
    require(len(candidate_net) >= int(0.90 * draws), "V76 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "mean": float(array.mean()),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over locked downside-quantile policies",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "candidate_net": interval(candidate_net), "net_delta": interval(net_delta),
        "lower_tail_20_delta": interval(tail_delta),
    }


def evaluate(champion: pd.DataFrame, frame: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_frame = champion.copy()
    diagnostic_lookup = diagnostic.set_index("event_id")
    require(set(candidate_frame.event_id) == set(diagnostic_lookup.index), "V76 diagnostic/champion event set mismatch")
    for column in ("prob", "confidence_signal", "high_conf", "family"):
        candidate_frame[column] = diagnostic_lookup.loc[candidate_frame.event_id, column].to_numpy()
    candidate_frame["high_conf"] = bool_series(candidate_frame.high_conf)
    candidate_summary = v44.summarize(candidate_frame)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V76/controller gate mismatch")
    baseline = mask_metrics(evidence, evidence.high_conf, evidence.confidence_signal)
    candidate = mask_metrics(evidence, evidence.candidate_high, evidence.candidate_confidence)
    bootstrap = bootstrap_economics(evidence, draws)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    fold_tail = np.asarray([audit["outer_lower_tail_delta"] for audit in audits], float)
    direction_exact = np.array_equal(champion.prob.to_numpy(float), candidate_frame.prob.to_numpy(float)) and np.array_equal(champion.prob.to_numpy(float) >= 0.5, candidate_frame.prob.to_numpy(float) >= 0.5)
    checks = {
        "all_six_predeclared_market_folds_evaluated": len(audits) == 6,
        "outer_candidate_mean_net_gt_0": candidate["mean_signed_net"] is not None and candidate["mean_signed_net"] > 0.0,
        "outer_mean_net_gt_v69": candidate["mean_signed_net"] is not None and baseline["mean_signed_net"] is not None and candidate["mean_signed_net"] > baseline["mean_signed_net"],
        "outer_lower_tail_20_gt_v69": candidate["realized_lower_tail_20_mean"] is not None and baseline["realized_lower_tail_20_mean"] is not None and candidate["realized_lower_tail_20_mean"] > baseline["realized_lower_tail_20_mean"],
        "bootstrap_candidate_net_lower95_gt_0": bootstrap["candidate_net"]["lower95"] > 0.0,
        "bootstrap_net_delta_probability_gt_zero_ge_0_75": bootstrap["net_delta"]["probability_gt_zero"] >= 0.75,
        "bootstrap_tail_delta_probability_gt_zero_ge_0_70": bootstrap["lower_tail_20_delta"]["probability_gt_zero"] >= 0.70,
        "positive_net_delta_fold_fraction_ge_2_of_3": float(np.mean(fold_net > 0.0)) >= 2.0 / 3.0,
        "nonnegative_tail_delta_fold_fraction_ge_2_of_3": float(np.mean(fold_tail >= 0.0)) >= 2.0 / 3.0,
        "outer_accuracy_delta_ge_minus_0_005": candidate["accuracy"] is not None and baseline["accuracy"] is not None and candidate["accuracy"] - baseline["accuracy"] >= -0.005,
        "confidence_correctness_auc_delta_ge_minus_0_005": candidate["confidence_correctness_auc"] is not None and baseline["confidence_correctness_auc"] is not None and candidate["confidence_correctness_auc"] - baseline["confidence_correctness_auc"] >= -0.005,
        "candidate_coverage_between_0_10_and_0_22": 0.10 <= candidate["coverage"] <= 0.22,
        "probability_and_direction_exact_v69": direction_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "distributional_tail_models_only": all(audit["inner_model"]["direction_model"] is False and audit["inner_model"]["mean_return_model"] is False for audit in audits),
    }
    material_pass = bool(all(checks.values()))
    selected = candidate_frame if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V76 fail-closed frame is not exact V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_mean_net_delta": number_or(candidate["mean_signed_net"], -1.0) - number_or(baseline["mean_signed_net"], -1.0),
        "outer_lower_tail_20_delta": number_or(candidate["realized_lower_tail_20_mean"], -1.0) - number_or(baseline["realized_lower_tail_20_mean"], -1.0),
        "outer_accuracy_delta": number_or(candidate["accuracy"], 0.0) - number_or(baseline["accuracy"], 0.0),
        "outer_confidence_correctness_auc_delta": number_or(candidate["confidence_correctness_auc"], 0.0) - number_or(baseline["confidence_correctness_auc"], 0.0),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "immutability": {
            "probability_sha256_before": array_sha256(champion.prob.to_numpy(float)),
            "probability_sha256_after": array_sha256(candidate_frame.prob.to_numpy(float)),
            "direction_sha256_before": array_sha256(champion.prob.to_numpy(float) >= 0.5),
            "direction_sha256_after": array_sha256(candidate_frame.prob.to_numpy(float) >= 0.5),
            "probability_exact": direction_exact, "direction_exact": direction_exact,
        },
        "fallback": {"activated": not material_pass, "policy": "exact V69 full frame" if not material_pass else None, "exact_frame_verified": bool(material_pass or selected.equals(champion))},
        "diagnostic_frame": candidate_frame, "selected_frame": selected,
    }


def write_outputs(authority: dict[str, Any], frame: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any], out: Path) -> dict[str, Any]:
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
            "version": "V76", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V76_DIAGNOSTIC": evaluation["candidate_summary"], "V76_FAIL_CLOSED_SELECTED": selected_contract},
            "evaluation": compact, "authority_audit": authority,
            "validation": "pooled causal history; nested chronology; 35-minute embargo; inner-only fixed coarse lower-tail allocation; outer labels evaluation-only",
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V76", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V76", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected_contract["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer_economic_deltas": {
                "mean_signed_net": evaluation["outer_mean_net_delta"],
                "lower_tail_20_mean": evaluation["outer_lower_tail_20_delta"],
                "accuracy": evaluation["outer_accuracy_delta"],
                "confidence_correctness_auc": evaluation["outer_confidence_correctness_auc_delta"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "DOWNSIDE_QUANTILE_ALLOCATION_REPORT.json": {
            "version": "V76", "hypothesis": HYPOTHESIS,
            "target": "net return from immutable V69 outer-OOF direction",
            "quantiles": QUANTILES, "utility_specs": UTILITY_SPECS,
            "coverage_levels": COVERAGE_LEVELS, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V76", "status": "MATCH", "canonical_check_count": 14,
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
    fold_rows = [{
        "market": audit["market"], "fold": audit["fold"],
        "selected_policy": audit["policy"]["selected"]["name"],
        "outer_net_delta": audit["outer_net_delta"],
        "outer_lower_tail_20_delta": audit["outer_lower_tail_delta"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]
    atomic_csv(pd.DataFrame(fold_rows), out / "V76_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V76_OUTER_ECONOMIC_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V76_DIAGNOSTIC_ALLOCATOR_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"][list(frame.columns)], out / "V76_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=CONTROLLER_OUTPUT is None)
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        if args.audit_only or args.smoke_test or args.full_run:
            parser.error("explicit modes cannot be combined with MARKET_BIO_VERSION_OUTPUT")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    return args


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    authority = verify_authority()
    frame, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "v69_rows": len(frame),
            "source_counts": frame.source_family.value_counts().sort_index().to_dict(),
            "market_fold_counts": frame.groupby(["market", "fold"]).size().to_dict(),
            "probability_and_direction_frozen": True,
            "canonical_controller_gate_checks": 14, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    diagnostic, evidence, audits = run_nested(frame, smoke=args.smoke_test)
    evaluation = evaluate(champion, frame, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "selected_policies": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_mean_net_delta": evaluation["outer_mean_net_delta"],
        "outer_lower_tail_20_delta": evaluation["outer_lower_tail_20_delta"],
        "outer_accuracy_delta": evaluation["outer_accuracy_delta"],
        "outer_confidence_correctness_auc_delta": evaluation["outer_confidence_correctness_auc_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_outer_baseline"] = evaluation["outer_baseline"]
        summary["raw_outer_candidate"] = evaluation["outer_candidate"]
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, champion, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
