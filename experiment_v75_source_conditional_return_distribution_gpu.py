"""V75 source-conditional return-distribution direction challenger on GPU.

Material novelty is a fixed four-class 30-minute return distribution, not
another binary direction classifier, signed-return regressor, confidence
router, text retriever, or causal response-prior model.  A CatBoost MultiClass
model uses native categorical source/market/event/issuer interactions and
causal pre-event numeric state.  P(up) is reconstructed as the probability of
the two positive-return classes, then an inner-past split chooses only among
exact V69, a 0.25 logit blend, and a 0.50 logit blend.

The only data authority is immutable V36 DEV plus atomic output_V69.  Folds
2--4 use an expanding chronology, strict 35-minute embargo, event/duplicate
purge, inner-only selection, and outer evaluation-only outcomes.  V69
confidence and high-confidence membership remain immutable.  A failed
material gate returns the exact full V69 frame.  This runner never performs a
full run implicitly; the preparation contract permits audit and one-fold GPU
smoke before a later explicit controller run.
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

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
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
DEFAULT_OUT = ROOT / "research" / "staging" / "V75_SOURCE_CONDITIONAL_RETURN_DISTRIBUTION_GPU_V1"

VERSION = 75
HYPOTHESIS = "SOURCE_CONDITIONAL_RETURN_DISTRIBUTION_CATBOOST_GPU_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_OOF_SHA256 = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_DEV_ROWS = 9462
EXPECTED_V69_ROWS = 7568

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
RETURN_BOUNDARY = 0.006
COST = 0.002
SEED = 7501
FULL_ITERATIONS = 360
SMOKE_ITERATIONS = 72
BLEND_WEIGHTS = (0.25, 0.50)
BOOTSTRAP_DRAWS = 2000

CATEGORICAL_COLUMNS = (
    "market", "source_family", "source", "form", "event_type", "ticker",
    "source_event", "market_event", "family_form", "market_session_bucket",
)
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
    "minutes_to_close", "event_positive_kw", "event_negative_kw",
    "event_financing_kw", "event_trial_kw", "event_regulatory_kw",
    "event_ma_kw", "event_earnings_kw", "headline_len", "body_len",
    "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
    "benchmark_ret_30", "benchmark_ret_60",
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
    return series.map(lambda item: str(item).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else None


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: float) -> np.ndarray:
    base = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    challenge = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    return expit((1.0 - float(weight)) * base + float(weight) * challenge)


def return_class(values: pd.Series | np.ndarray) -> np.ndarray:
    returns = np.asarray(values, float)
    # 0 strong down, 1 mild/non-positive, 2 mild positive, 3 strong up.
    return np.select(
        [returns <= -RETURN_BOUNDARY, returns <= 0.0, returns < RETURN_BOUNDARY],
        [0, 1, 2], default=3,
    ).astype(int)


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(DEV_PATH.is_file(), "immutable V36 DEV is missing")
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit_path.is_file() and manifest_path.is_file(), "V69 atomic authority is incomplete")
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
        require(path.is_file(), f"V69 manifested file missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 file size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 file hash changed: {name}")
    require(manifest["files"].get(V69_PATH.name, {}).get("sha256") == EXPECTED_V69_OOF_SHA256,
            "V69 selected OOF is not manifest-bound")
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


def prepare_dev(dev: pd.DataFrame) -> pd.DataFrame:
    output = dev.copy()
    output["event_time_utc"] = pd.to_datetime(output.event_time_utc, utc=True)
    output["headline"] = output.headline.fillna("").astype(str)
    output["body"] = output.body.fillna("").astype(str)
    output["text_cluster_id"] = output.apply(v37.normalized_cluster, axis=1)
    output["cluster_weight"] = 1.0 / output.groupby("text_cluster_id").event_id.transform("size")
    output["return_class"] = return_class(output.fwd_ret_30m)
    output["source_event"] = output.source_family.fillna("UNKNOWN").astype(str) + "|" + output.event_type.fillna("UNKNOWN").astype(str)
    output["market_event"] = output.market.fillna("UNKNOWN").astype(str) + "|" + output.event_type.fillna("UNKNOWN").astype(str)
    output["family_form"] = output.source_family.fillna("UNKNOWN").astype(str) + "|" + output.form.fillna("UNKNOWN").astype(str)
    minutes = pd.to_numeric(output.minutes_from_open, errors="coerce")
    output["market_session_bucket"] = pd.cut(
        minutes, bins=[-np.inf, 30, 120, 270, np.inf],
        labels=["OPEN", "MORNING", "MIDDAY", "CLOSE"],
    ).astype(str)
    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].fillna("UNKNOWN").astype(str)
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
    require(len(dev) == EXPECTED_DEV_ROWS, "V36 DEV row count changed")
    require(len(champion) == EXPECTED_V69_ROWS, "V69 OOF row count changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 contains an event outside V36")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15),
            "V36/V69 return outcomes differ")
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    dev = prepare_dev(dev)
    require(set(dev.return_class.unique()) == {0, 1, 2, 3}, "return distribution lacks a class")
    return dev, champion


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(CATEGORICAL_COLUMNS) + list(NUMERIC_COLUMNS)].copy()


def class_weights(labels: np.ndarray) -> list[float]:
    counts = np.bincount(np.asarray(labels, int), minlength=4).astype(float)
    require(np.all(counts > 0), "training fold lacks a return class")
    weights = len(labels) / (4.0 * counts)
    return (weights / weights.mean()).tolist()


def fit_gpu_distribution(
    train: pd.DataFrame, target: pd.DataFrame, seed: int, smoke: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels = train.return_class.to_numpy(int)
    train_pool = Pool(
        feature_frame(train), label=labels,
        cat_features=list(CATEGORICAL_COLUMNS), weight=train.cluster_weight.to_numpy(float),
    )
    target_pool = Pool(feature_frame(target), cat_features=list(CATEGORICAL_COLUMNS))
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        classes_count=4,
        iterations=SMOKE_ITERATIONS if smoke else FULL_ITERATIONS,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=8.0,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        boosting_type="Plain",
        border_count=64,
        class_weights=class_weights(labels),
        task_type="GPU",
        devices="0",
        thread_count=runtime_limits.THREAD_COUNT,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(train_pool)
    require(str(model.get_param("task_type")).upper() == "GPU", "CatBoost did not use GPU task type")
    probability = np.asarray(model.predict_proba(target_pool), float)
    classes = np.asarray(model.classes_, int)
    require(set(classes.tolist()) == {0, 1, 2, 3}, "CatBoost class order changed")
    ordered = np.zeros((len(target), 4), dtype=float)
    for position, label in enumerate(classes):
        ordered[:, int(label)] = probability[:, position]
    require(np.isfinite(ordered).all(), "non-finite class probability")
    require(np.allclose(ordered.sum(axis=1), 1.0, rtol=0.0, atol=1e-6), "class probabilities do not sum to one")
    p_up = np.clip(ordered[:, 2] + ordered[:, 3], 1e-6, 1.0 - 1e-6)
    return p_up, ordered, {
        "library": "catboost",
        "catboost_version": catboost.__version__,
        "task_type": model.get_param("task_type"),
        "devices": model.get_param("devices"),
        "actual_gpu_fit_completed": True,
        "iterations": model.get_param("iterations"),
        "depth": model.get_param("depth"),
        "return_class_counts": {str(index): int(count) for index, count in enumerate(np.bincount(labels, minlength=4))},
        "class_weights": class_weights(labels),
        "categorical_feature_n": len(CATEGORICAL_COLUMNS),
        "numeric_feature_n": len(NUMERIC_COLUMNS),
        "train_n": len(train),
        "target_n": len(target),
    }


def metric(frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    confidence = np.asarray(confidence, float)
    high = np.asarray(high, bool)
    signed_net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    correctness = (prediction == y).astype(int)
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(correctness)),
        "pred_up": float(prediction.mean()),
        "all_trade_mean_signed_net": float(signed_net.mean()),
        "confidence_correctness_auc": safe_auc(correctness, confidence),
        "highconf_n": int(high.sum()),
        "highconf_accuracy": float(correctness[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(signed_net[high].mean()) if high.any() else None,
    }


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: 35-minute embargo failed")
    event_overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    cluster_overlap = set(train.text_cluster_id.astype(str)) & set(valid.text_cluster_id.astype(str))
    require(not event_overlap, f"{label}: event-group leakage")
    require(not cluster_overlap, f"{label}: duplicate-cluster leakage")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end, "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
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
    dev: pd.DataFrame, champion: pd.DataFrame, outer_start: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.event_time_utc < outer_start - EMBARGO].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= 1000, "insufficient prior V69 OOF rows for inner selection")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = leakage_filtered_train(dev, inner_start, inner_valid)
    require(len(inner_train) >= 700, "inner train is too small")
    require(set(inner_train.return_class.unique()) == {0, 1, 2, 3}, "inner train lacks a return class")
    audit = chronology(inner_train, inner_valid, "V75 inner")
    audit["validation_source"] = "earlier committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
    fold: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    p_up, distribution, model_audit = fit_gpu_distribution(
        inner_train, inner_valid, SEED + fold * 10, smoke
    )
    trials = [{
        "name": "V69_NOOP", "challenger_weight": 0.0,
        "metrics": baseline_metric, "auc_delta": 0.0, "ba_delta": 0.0,
        "net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]
                       + 10.0 * baseline_metric["all_trade_mean_signed_net"]),
    }]
    for weight in BLEND_WEIGHTS:
        probability = probability_blend(baseline, p_up, weight)
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": f"RETURN_DISTRIBUTION_W{weight:.2f}",
            "challenger_weight": float(weight), "metrics": values,
            "auc_delta": auc_delta, "ba_delta": ba_delta, "net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]
                           + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["net_delta"], -trial["challenger_weight"]),
    )
    return {
        "selection_rule": "maximize 2*AUC + BA + 10*all-trade-net on inner-past OOF; BA delta>=-0.005 and net delta>=-0.0005; exact V69 and fixed 0.25/0.50 blends only",
        "return_classes": {
            "0": f"return <= {-RETURN_BOUNDARY}",
            "1": f"{-RETURN_BOUNDARY} < return <= 0",
            "2": f"0 < return < {RETURN_BOUNDARY}",
            "3": f"return >= {RETURN_BOUNDARY}",
        },
        "p_up_formula": "P(class=2)+P(class=3)",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "gpu_model_audit": model_audit,
        "inner_distribution_probability_sha256": array_sha256(distribution),
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v75_model"] = "V69_NOOP"
    diagnostic["v75_challenger_weight"] = 0.0
    evidence_parts = []
    audits = []
    for fold in (2, 3, 4):
        outer_champion = champion.loc[champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == 1892, f"V69 fold {fold} row count changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = leakage_filtered_train(dev, outer_start, outer_valid)
        require(len(outer_train) >= 2500, f"outer fold {fold} train too small")
        require(set(outer_train.return_class.unique()) == {0, 1, 2, 3}, "outer train lacks a class")
        outer_audit = chronology(outer_train, outer_valid, f"V75 outer fold {fold}")
        inner_train, inner_valid, inner_champion, inner_audit = inner_partition(dev, champion, outer_start)
        policy = choose_inner(inner_train, inner_valid, inner_champion, fold, smoke)
        selected = policy["selected"]
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        p_up, distribution, outer_model_audit = fit_gpu_distribution(
            outer_train, outer_valid, SEED + 1000 + fold, smoke
        )
        weight = float(selected["challenger_weight"])
        candidate = baseline.copy() if weight == 0.0 else probability_blend(baseline, p_up, weight)
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        probability_map = dict(zip(outer_valid.event_id, candidate))
        positions = diagnostic.index[diagnostic.event_id.isin(probability_map)]
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v75_model"] = selected["name"]
        diagnostic.loc[positions, "v75_challenger_weight"] = weight
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["distribution_p_up"] = p_up
        for label in range(4):
            evidence[f"return_class_probability_{label}"] = distribution[:, label]
        evidence["candidate_prob"] = candidate
        evidence["v75_model"] = selected["name"]
        evidence["v75_challenger_weight"] = weight
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold,
            "inner_chronology": inner_audit,
            "outer_chronology": outer_audit,
            "policy": policy,
            "outer_gpu_model_audit": outer_model_audit,
            "outer_baseline": base_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V75 GPU] fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
        if smoke:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V75 evidence repeats an event")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few date blocks")
    rng = np.random.default_rng(SEED + 99)
    auc_delta = []
    net_delta = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        returns = sample.fwd_ret_30m.to_numpy(float)
        auc_delta.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        base_net = np.where(base >= 0.5, 1.0, -1.0) * returns - COST
        candidate_net = np.where(candidate >= 0.5, 1.0, -1.0) * returns - COST
        net_delta.append(float(candidate_net.mean() - base_net.mean()))

    def interval(values: list[float]) -> dict[str, Any]:
        require(values, "empty bootstrap")
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array),
            "probability_gt_zero": float(np.mean(array > 0.0)),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
        }
    return {
        "method": "paired outer-fold/date-block bootstrap over locked nested policies",
        "requested_draws": draws,
        "auc_delta": interval(auc_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    smoke: bool,
) -> dict[str, Any]:
    baseline_summary = v44.summarize(champion)
    candidate_summary = v44.summarize(diagnostic)
    canonical_base = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(baseline_summary["research_gate"] == canonical_base, "V69 canonical gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V75 canonical gate mismatch")
    base = metric(
        evidence, evidence.baseline_prob, evidence.confidence_signal,
        bool_series(evidence.high_conf),
    )
    candidate = metric(
        evidence, evidence.candidate_prob, evidence.confidence_signal,
        bool_series(evidence.high_conf),
    )
    bootstrap = None if smoke else paired_bootstrap(evidence, BOOTSTRAP_DRAWS)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float))
        and np.array_equal(bool_series(champion.high_conf).to_numpy(bool), bool_series(diagnostic.high_conf).to_numpy(bool))
    )
    gpu_audits = [audit["policy"]["gpu_model_audit"] for audit in audits] + [
        audit["outer_gpu_model_audit"] for audit in audits
    ]
    checks = {
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_ba_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_gt_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] > 0.0,
        "positive_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "positive_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net > 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_probability_gt_zero_ge_0_80": bool(not smoke and bootstrap["auc_delta"]["probability_gt_zero"] >= 0.80),
        "bootstrap_net_probability_gt_zero_ge_0_80": bool(not smoke and bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.80),
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "all_gpu_fits_completed": all(audit["actual_gpu_fit_completed"] and audit["task_type"] == "GPU" for audit in gpu_audits),
        "strict_nested_chronology": all(
            audit["inner_chronology"]["strict_35m_embargo"]
            and audit["outer_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "four_class_distribution_contract": all(
            set(audit["outer_gpu_model_audit"]["return_class_counts"]) == {"0", "1", "2", "3"}
            for audit in audits
        ),
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    selected_canonical = controller.canonical_research_gate(selected_summary)
    require(selected_summary["research_gate"] == selected_canonical, "selected canonical gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V75 fallback is not exact full V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": base,
        "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_all_trade_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_base, "candidate": canonical_candidate,
            "selected": selected_canonical, "reported_equals_controller_recomputed": True,
        },
        "gpu": {
            "fit_count": len(gpu_audits),
            "all_actual_gpu": checks["all_gpu_fits_completed"],
            "catboost_version": catboost.__version__,
            "device_ordinal": 0,
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
        "diagnostic_frame": diagnostic,
        "selected_frame": selected_frame,
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(not out.name.startswith("output_V75"), "runner must not write output_V75 directly")
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
            "version": "V75", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V75_RETURN_DISTRIBUTION_DIAGNOSTIC": evaluation["candidate_summary"],
                "V75_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority,
            "nested_fold_audits": audits, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V75", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V75", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V75_RETURN_DISTRIBUTION_GPU_REPORT.json": {
            "version": "V75", "hypothesis": HYPOTHESIS,
            "material_novelty": "fixed four-class return distribution with native source-conditional CatBoost GPU interactions",
            "return_boundary": RETURN_BOUNDARY, "blend_weights": BLEND_WEIGHTS,
            "gpu_contract": {"task_type": "GPU", "device": 0, "full_iterations": FULL_ITERATIONS},
            "authority_audit": authority, "nested_fold_audits": audits,
            "evaluation": compact,
        },
        "V75_DIRECTION_CONFIDENCE_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V75_RETURN_DISTRIBUTION_OUTER_OOF.csv.gz")
    selected_columns = list(evaluation["selected_frame"].columns)
    atomic_csv(evaluation["selected_frame"][selected_columns], out / "V75_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {"runner_files_written": sorted([*reports, "V75_RETURN_DISTRIBUTION_OUTER_OOF.csv.gz", "V75_FAIL_CLOSED_SELECTED_OOF.csv.gz"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="fit/evaluate but do not write outputs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.full_run,
                "explicit modes cannot be combined with controller output")
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
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "gpu_contract": {
                "library": "catboost", "version": catboost.__version__,
                "task_type": "GPU", "device_ordinal": 0,
            },
            "canonical_controller_gate_checks": 14,
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    dev, champion = load_authorized()
    diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, smoke=args.smoke_test)
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "folds_completed": len(audits),
        "gpu_fit_count": evaluation["gpu"]["fit_count"],
        "all_actual_gpu": evaluation["gpu"]["all_actual_gpu"],
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_all_trade_net_delta": evaluation["outer_all_trade_net_delta"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "output_written": False,
    }
    if not args.dry_run and args.full_run:
        summary.update(write_outputs(args.output, authority, evidence, audits, evaluation))
        summary["output_written"] = True
        summary["output"] = str(args.output)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
