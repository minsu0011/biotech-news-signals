"""V89 field-aware low-rank bilinear causal-state direction challenger.

The material hypothesis is explicit cross-field interaction: momentum may mean
something different under distinct risk/liquidity, price-position/time, and
benchmark states.  Each training cut robustly standardizes four causal numeric
fields (plus same-column missing flags).  One shallow factorization model adds
fixed-rank bilinear terms only across different fields to linear main effects.
It has no tree thresholds, random kernel, class density/covariance, trajectory
alignment, deep hidden stack, or time-varying coefficient filter.

Only immutable V36 DEV and atomic output_V69 are authoritative.  Folds 2--4
use nested expanding chronology, strict 35-minute embargo, event/duplicate
purge, inner-only selection among exact V69 and two fixed blends, and outer
evaluation-only outcomes.  V69 confidence/high-confidence membership remain
exact.  Material failure restores the exact entire V69 frame, and the
controller's canonical 14 checks are recomputed.  Preparation permits audit
and one bounded single-fold smoke only; no implicit full run occurs.
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
import torch
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

import autonomous_v37plus as controller
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V89_FIELD_AWARE_LOW_RANK_BILINEAR_V1"

VERSION = 89
HYPOTHESIS = "FIELD_AWARE_LOW_RANK_BILINEAR_DIRECTION_V1"
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
SEED = 8901
FACTOR_RANK = 6
FULL_EPOCHS = 220
SMOKE_EPOCHS = 100
LEARNING_RATE = 0.020
WEIGHT_DECAY = 0.004
BLEND_WEIGHTS = (0.25, 0.50)
BOOTSTRAP_DRAWS = 2000

# All fields are causal pre-decision market state.  Missing flags remain in the
# same field as their raw scalar.  No event/source/category/text/issuer/ticker,
# response prior, target statistic, or post-event variable is a model input.
FEATURE_FIELDS = {
    "MOMENTUM": (
        "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
        "pre_ret_90", "pre_ret_120", "entry_bar_ret", "intraday_ret_open",
        "return_autocorr_30", "up_fraction_10", "up_fraction_30",
        "trend_slope_30", "trend_slope_60",
    ),
    "RISK_LIQUIDITY": (
        "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
        "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
        "volume_ratio_10_60", "range_5m", "range_10m", "range_30m",
        "range_60m", "entry_bar_range",
    ),
    "PRICE_POSITION_TIME": (
        "close_position_10", "close_position_30", "close_position_60",
        "vwap_distance_30", "log_entry_price", "minutes_from_open",
        "minutes_to_close",
    ),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
}
NUMERIC_COLUMNS = tuple(column for columns in FEATURE_FIELDS.values() for column in columns)


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
    audit = chronology(inner_train, inner_valid, "V89 inner")
    audit["validation_source"] = "earlier committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


class FieldState:
    def __init__(self) -> None:
        self.median: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "FieldState":
        for field, columns in FEATURE_FIELDS.items():
            values = frame.loc[:, columns].to_numpy(float)
            median = np.nanmedian(values, axis=0)
            median = np.where(np.isfinite(median), median, 0.0)
            filled = np.where(np.isfinite(values), values, median)
            q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
            standard = np.std(filled, axis=0)
            scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
            self.median[field] = median
            self.scale[field] = scale
        require(len(self.median) == len(FEATURE_FIELDS), "field scaler fit failed")
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
        parts = []
        slices = {}
        start = 0
        for field, columns in FEATURE_FIELDS.items():
            values = frame.loc[:, columns].to_numpy(float)
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.median[field], values)
            standardized = np.clip((filled - self.median[field]) / self.scale[field], -8.0, 8.0)
            part = np.column_stack([standardized, missing.astype(float)]).astype(np.float32)
            parts.append(part)
            slices[field] = (start, start + part.shape[1])
            start += part.shape[1]
        output = np.column_stack(parts).astype(np.float32)
        require(np.isfinite(output).all(), "non-finite field state")
        return output, slices


class FieldAwareBilinear(nn.Module):
    def __init__(self, input_n: int, slices: dict[str, tuple[int, int]], rank: int) -> None:
        super().__init__()
        self.slices = slices
        self.fields = tuple(slices)
        self.linear = nn.Linear(input_n, 1, bias=True)
        self.left = nn.ParameterDict()
        self.right = nn.ParameterDict()
        for left_index, left_name in enumerate(self.fields):
            left_width = slices[left_name][1] - slices[left_name][0]
            for right_name in self.fields[left_index + 1:]:
                right_width = slices[right_name][1] - slices[right_name][0]
                key = f"{left_name}__{right_name}"
                self.left[key] = nn.Parameter(torch.empty(left_width, rank))
                self.right[key] = nn.Parameter(torch.empty(right_width, rank))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        for parameter in list(self.left.values()) + list(self.right.values()):
            nn.init.normal_(parameter, mean=0.0, std=0.025)

    def components(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear = self.linear(values).squeeze(1)
        interaction = torch.zeros_like(linear)
        for left_index, left_name in enumerate(self.fields):
            left_start, left_end = self.slices[left_name]
            for right_name in self.fields[left_index + 1:]:
                right_start, right_end = self.slices[right_name]
                key = f"{left_name}__{right_name}"
                left_factor = values[:, left_start:left_end] @ self.left[key]
                right_factor = values[:, right_start:right_end] @ self.right[key]
                interaction = interaction + torch.sum(left_factor * right_factor, dim=1)
        return linear, interaction

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        linear, interaction = self.components(values)
        return linear + interaction


def fit_bilinear(
    train: pd.DataFrame, target: pd.DataFrame, seed: int, smoke: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    processor = FieldState().fit(train)
    train_x, slices = processor.transform(train)
    target_x, target_slices = processor.transform(target)
    require(slices == target_slices, "field slices changed")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model = FieldAwareBilinear(train_x.shape[1], slices, FACTOR_RANK).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x_tensor = torch.from_numpy(train_x).to(device)
    y_tensor = torch.from_numpy(train.y.to_numpy(np.float32)).to(device)
    weight_tensor = torch.from_numpy(train.cluster_weight.to_numpy(np.float32)).to(device)
    weight_tensor = weight_tensor / weight_tensor.mean()
    epochs = SMOKE_EPOCHS if smoke else FULL_EPOCHS
    trace = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        row_loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none")
        objective = torch.sum(row_loss * weight_tensor) / torch.sum(weight_tensor)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            with torch.no_grad():
                _, interaction = model.components(x_tensor)
            trace.append({
                "epoch": epoch + 1,
                "weighted_bce": float(objective.detach()),
                "interaction_logit_std": float(interaction.std()),
            })
    with torch.no_grad():
        target_tensor = torch.from_numpy(target_x).to(device)
        linear_logit, interaction_logit = model.components(target_tensor)
        total_logit = linear_logit + interaction_logit
        probability = torch.sigmoid(total_logit).cpu().numpy()
        linear_values = linear_logit.cpu().numpy()
        interaction_values = interaction_logit.cpu().numpy()
    interaction_norms = {}
    for key in model.left:
        interaction_norms[key] = {
            "left_l2": float(torch.linalg.vector_norm(model.left[key]).detach()),
            "right_l2": float(torch.linalg.vector_norm(model.right[key]).detach()),
        }
    require(np.isfinite(probability).all(), "non-finite bilinear probability")
    require(float(np.std(interaction_values)) > 1e-8, "bilinear interaction path degenerate")
    predictions = {
        "probability": np.clip(np.asarray(probability, float), 1e-6, 1.0 - 1e-6),
        "linear_logit": np.asarray(linear_values, float),
        "interaction_logit": np.asarray(interaction_values, float),
        "total_logit": np.asarray(linear_values + interaction_values, float),
    }
    audit = {
        "architecture": "field-aware cross-field low-rank bilinear logistic",
        "objective": "duplicate-cluster-weighted binary cross entropy",
        "factor_rank": FACTOR_RANK,
        "fields": {name: list(columns) for name, columns in FEATURE_FIELDS.items()},
        "field_slices": slices,
        "raw_causal_numeric_feature_n": len(NUMERIC_COLUMNS),
        "state_feature_n": train_x.shape[1],
        "interaction_pair_n": len(interaction_norms),
        "interaction_norms": interaction_norms,
        "target_interaction_logit_quantiles": {
            str(q): float(np.quantile(interaction_values, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_interaction_logit_std": float(np.std(interaction_values)),
        "target_linear_logit_std": float(np.std(linear_values)),
        "training_n": len(train), "target_n": len(target),
        "epochs": epochs, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "trace": trace,
        "source_category_issuer_ticker_text_used": False,
        "target_rows_used_for_preprocessing_or_training": False,
        "tree_kernel_density_trajectory_filter_used": False,
        "device": "CPU", "gpu_used": False,
        "cpu_reason": "bounded shallow rank-six full-batch model while the main full runner retains resources",
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


def choose_inner(
    train: pd.DataFrame, valid: pd.DataFrame, champion_valid: pd.DataFrame,
    fold: int, smoke: bool,
) -> dict[str, Any]:
    predictions, model_audit = fit_bilinear(train, valid, SEED + fold * 10, smoke)
    baseline = champion_valid.prob.to_numpy(float)
    confidence = champion_valid.confidence_signal.to_numpy(float)
    high = bool_series(champion_valid.high_conf).to_numpy(bool)
    baseline_metric = metric(valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "weight": 0.0, "metrics": baseline_metric,
        "auc_delta": 0.0, "ba_delta": 0.0, "net_delta": 0.0,
        "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]),
    }]
    for weight in BLEND_WEIGHTS:
        probability = probability_blend(baseline, predictions["probability"], weight)
        values = metric(valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": f"FIELD_BILINEAR_W{weight:.2f}", "weight": float(weight),
            "metrics": values, "auc_delta": auc_delta, "ba_delta": ba_delta,
            "net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["ba_delta"], -trial["weight"]),
    )
    return {
        "selection_rule": "maximize 2*AUC+BA on inner-past OOF; BA delta>=-0.005 and net delta>=-0.0005; exact no-op and fixed 0.25/0.50 blends only",
        "selected": selected, "trials": trials,
        "bilinear_model_audit": model_audit,
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v89_model"] = "V69_NOOP"
    diagnostic["v89_blend_weight"] = 0.0
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
        outer_audit = chronology(outer_train, outer_valid, f"V89 outer fold {fold}")
        inner_train, inner_valid, inner_champion, inner_audit = inner_partition(
            dev, champion, pd.Timestamp(outer_valid.event_time_utc.min())
        )
        policy = choose_inner(inner_train, inner_valid, inner_champion, fold, smoke)
        selected = policy["selected"]
        predictions, outer_model_audit = fit_bilinear(
            outer_train, outer_valid, SEED + 1000 + fold, smoke
        )
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        weight = float(selected["weight"])
        candidate = baseline.copy() if weight == 0.0 else probability_blend(
            baseline, predictions["probability"], weight
        )
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        probability_map = dict(zip(outer_valid.event_id, candidate))
        positions = diagnostic.index[diagnostic.event_id.isin(probability_map)]
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v89_model"] = selected["name"]
        diagnostic.loc[positions, "v89_blend_weight"] = weight
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["field_bilinear_prob"] = predictions["probability"]
        evidence["linear_logit"] = predictions["linear_logit"]
        evidence["interaction_logit"] = predictions["interaction_logit"]
        evidence["total_logit"] = predictions["total_logit"]
        evidence["candidate_prob"] = candidate
        evidence["v89_model"] = selected["name"]
        evidence["v89_blend_weight"] = weight
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold, "inner_chronology": inner_audit, "outer_chronology": outer_audit,
            "policy": policy, "outer_bilinear_model_audit": outer_model_audit,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "outer_interaction_logit_std": float(np.std(predictions["interaction_logit"])),
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V89 FIELD-BILINEAR] fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f} "
            f"interaction_std={audits[-1]['outer_interaction_logit_std']:.4f}",
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
        "method": "paired fold/date block bootstrap over locked V89 field-bilinear policies",
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
    model_audits = [audit["policy"]["bilinear_model_audit"] for audit in audits] + [
        audit["outer_bilinear_model_audit"] for audit in audits
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
        "field_bilinear_architecture_all_fits": all(
            audit["factor_rank"] == FACTOR_RANK
            and audit["interaction_pair_n"] == 6
            and not audit["tree_kernel_density_trajectory_filter_used"] for audit in model_audits
        ),
        "causal_numeric_inputs_only": all(
            not audit["source_category_issuer_ticker_text_used"]
            and not audit["target_rows_used_for_preprocessing_or_training"] for audit in model_audits
        ),
        "interaction_path_nonconstant": bool(
            np.isfinite(evidence.interaction_logit.to_numpy(float)).all()
            and float(evidence.interaction_logit.std()) > 1e-8
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
        "interaction_diagnostics": {
            "linear_logit_quantiles": {
                str(q): float(np.quantile(evidence.linear_logit, q)) for q in (0.1, 0.5, 0.9)
            },
            "interaction_logit_quantiles": {
                str(q): float(np.quantile(evidence.interaction_logit, q)) for q in (0.1, 0.5, 0.9)
            },
            "interaction_logit_std": float(evidence.interaction_logit.std()),
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
    require(not out.name.startswith("output_V89"), "runner must not write output_V89 directly")
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
            "version": "V89", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V89_FIELD_BILINEAR_DIAGNOSTIC": evaluation["candidate_summary"],
                "V89_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V89", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V89", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V89_FIELD_AWARE_BILINEAR_REPORT.json": {
            "version": "V89", "hypothesis": HYPOTHESIS,
            "feature_fields": FEATURE_FIELDS, "factor_rank": FACTOR_RANK,
            "blend_weights": BLEND_WEIGHTS, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V89_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V89_FIELD_BILINEAR_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V89_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V89_FIELD_BILINEAR_OUTER_OOF.csv.gz", "V89_FAIL_CLOSED_SELECTED_OOF.csv.gz",
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
    if args.smoke_test:
        # Preparation must coexist with the active full runner.  Keep the
        # single-fold smoke to two numerical workers even though the eventual
        # controller full run may use the normal runtime policy.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    authority = verify_authority()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "model_contract": {
                "objective": "duplicate-weighted BCE with explicit cross-field rank-six bilinear interactions",
                "feature_fields": list(FEATURE_FIELDS),
                "model_inputs": "causal pre-event numeric state and same-column missing flags only",
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V88 root runners plus fixed V88 declaration",
                "factorization_machine_field_aware_bilinear_found": False,
                "disjoint_from_v84": "explicit finite cross-field rank-six terms, no RFF kernel or Laplace posterior",
                "disjoint_from_v85": "no entropy tilting or recent-regime moment matching",
                "disjoint_from_v86": "no copula marginal transform, class density, or covariance QDA",
                "disjoint_from_v87": "no path order, class prototype, distance, DTW, or Platt map",
                "disjoint_from_v88": "one static shallow interaction model, no sequential coefficient posterior or process noise",
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
        "bilinear_fit_count": 2 * len(audits),
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "interaction_diagnostics": evaluation["interaction_diagnostics"],
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
