"""V95 causal multi-resolution Haar-scattering direction challenger.

The material hypothesis is label-independent multi-resolution shape.  Causal
numeric measurements that share ordered lookback horizons are robustly scaled
inside each past training cut, edge-padded to a dyadic length, and transformed
by a fixed Haar cascade.  Signed details preserve orientation, first-order
moduli encode local energy, and a second Haar cascade over finest-scale moduli
provides a shallow second-order scattering summary.  One static L2 logistic
head is then fitted; no kernel, learned convolution, prototype, feature
selection, dependency tree, robust objective, or block-coefficient aggregate
is involved.

Only immutable V36 DEV and atomic output_V69 are authoritative.  Folds 2--4
use nested chronology, strict 35-minute embargo, event/duplicate purge,
inner-only selection among exact V69 and two fixed blends, and outer
evaluation-only outcomes.  V69 confidence/high-confidence membership remain
exact.  Material failure restores the entire V69 frame, and the controller's
canonical 14 checks are recomputed.  Preparation permits audit and one bounded
two-thread single-fold smoke only; no implicit full run occurs.
"""
from __future__ import annotations

import argparse
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
from threadpoolctl import threadpool_limits

import experiment_v89_field_aware_low_rank_bilinear_direction as scaffold


ROOT = scaffold.ROOT
DEV_PATH = scaffold.DEV_PATH
V69_DIR = scaffold.V69_DIR
V69_PATH = scaffold.V69_PATH
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V95_CAUSAL_MULTIRESOLUTION_HAAR_SCATTERING_V1"

VERSION = 95
HYPOTHESIS = "CAUSAL_MULTIRESOLUTION_HAAR_SCATTERING_DIRECTION_V1"
SCAFFOLD_PATH = ROOT / "experiment_v89_field_aware_low_rank_bilinear_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "98d0547f68e3c5959b1a5bb6521e823d4403b5fdb3dbb3211062b485c147a918"
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
LOGISTIC_C = 0.50
BLEND_WEIGHTS = (0.25, 0.50)
COST = scaffold.COST
SEED = 9501
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2

# Every channel is strictly pre-decision and ordered from shortest to longest
# horizon.  Source/category, issuer/ticker, text/keywords, historical response,
# entry-bar measurements, and post-event values are excluded.
HORIZON_CHANNELS = {
    "RETURN": (
        "pre_ret_1", "pre_ret_2", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_30", "pre_ret_60", "pre_ret_120",
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
    "UP_FRACTION": ("up_fraction_10", "up_fraction_30"),
}
FEATURES = tuple(column for columns in HORIZON_CHANNELS.values() for column in columns)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
bool_series = scaffold.bool_series
probability_blend = scaffold.probability_blend
chronology = scaffold.chronology
leakage_filtered_train = scaffold.leakage_filtered_train
aligned_dev = scaffold.aligned_dev
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44
runtime_limits = scaffold.runtime_limits


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V89 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic authority, chronology, metrics, and atomic IO only; no V89 output is read",
    }
    return audit


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev, champion = scaffold.load_authorized()
    missing = [column for column in FEATURES if column not in dev]
    require(not missing, f"V95 causal feature missing: {missing}")
    for column in FEATURES:
        dev[column] = pd.to_numeric(dev[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return dev, champion


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
    audit = chronology(inner_train, inner_valid, "V95 inner")
    audit["validation_source"] = "earlier committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def dyadic_length(length: int) -> int:
    require(length >= 2, "Haar channel too short")
    return 1 << int(math.ceil(math.log2(length)))


def haar_scattering_vector(path: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    path = np.asarray(path, float)
    padded_n = dyadic_length(len(path))
    if padded_n > len(path):
        path = np.pad(path, (0, padded_n - len(path)), mode="edge")
    values: list[float] = []
    names: list[str] = []
    current = path.copy()
    finest_modulus: np.ndarray | None = None
    level = 1
    while len(current) >= 2:
        average = (current[0::2] + current[1::2]) / math.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / math.sqrt(2.0)
        if finest_modulus is None:
            finest_modulus = np.abs(detail)
        for position, item in enumerate(detail):
            values.extend([float(item), float(abs(item))])
            names.extend([
                f"{prefix}::L{level}::D{position}",
                f"{prefix}::L{level}::ABS_D{position}",
            ])
        current = average
        level += 1
    values.append(float(current[0]))
    names.append(f"{prefix}::LOWPASS")

    # Second-order scattering: another Haar modulus cascade over the finest
    # first-order modulus.  Only modulus details and its terminal low-pass are
    # retained, so this branch is label-independent and sign-stable.
    require(finest_modulus is not None, "missing first-order Haar modulus")
    if len(finest_modulus) >= 2:
        second = finest_modulus.copy()
        second_level = 1
        while len(second) >= 2:
            second_average = (second[0::2] + second[1::2]) / math.sqrt(2.0)
            second_detail = (second[0::2] - second[1::2]) / math.sqrt(2.0)
            for position, item in enumerate(second_detail):
                values.append(float(abs(item)))
                names.append(f"{prefix}::S2_L{second_level}::ABS_D{position}")
            second = second_average
            second_level += 1
        values.append(float(second[0]))
        names.append(f"{prefix}::S2_LOWPASS")
    return np.asarray(values, dtype=float), names


class HaarState:
    def __init__(self) -> None:
        self.median: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}
        self.feature_names: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> "HaarState":
        for channel, columns in HORIZON_CHANNELS.items():
            values = frame.loc[:, columns].to_numpy(float)
            median = np.nanmedian(values, axis=0)
            median = np.where(np.isfinite(median), median, 0.0)
            filled = np.where(np.isfinite(values), values, median)
            q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
            standard = np.std(filled, axis=0)
            scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
            self.median[channel] = median
            self.scale[channel] = scale
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        rows = []
        names: list[str] | None = None
        channel_state = {}
        for channel, columns in HORIZON_CHANNELS.items():
            raw = frame.loc[:, columns].to_numpy(float)
            filled = np.where(np.isfinite(raw), raw, self.median[channel])
            channel_state[channel] = np.clip(
                (filled - self.median[channel]) / self.scale[channel], -7.0, 7.0
            )
        for row_index in range(len(frame)):
            parts = []
            row_names = []
            for channel in HORIZON_CHANNELS:
                part, part_names = haar_scattering_vector(channel_state[channel][row_index], channel)
                parts.append(part)
                row_names.extend(part_names)
            rows.append(np.concatenate(parts))
            if names is None:
                names = row_names
            else:
                require(names == row_names, "Haar feature order changed")
        require(names is not None, "empty Haar transform")
        if self.feature_names is None:
            self.feature_names = names
        else:
            require(self.feature_names == names, "Haar schema changed")
        output = np.asarray(rows, dtype=float)
        require(np.isfinite(output).all() and float(np.std(output)) > 1e-10, "degenerate Haar scattering")
        return output


def fit_haar_scattering(
    train: pd.DataFrame, target: pd.DataFrame, seed: int, smoke: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    transform = HaarState().fit(train)
    train_scattering = transform.transform(train)
    target_scattering = transform.transform(target)
    feature_mean = train_scattering.mean(axis=0)
    feature_scale = train_scattering.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    train_x = np.clip((train_scattering - feature_mean) / feature_scale, -8.0, 8.0)
    target_x = np.clip((target_scattering - feature_mean) / feature_scale, -8.0, 8.0)
    with threadpool_limits(limits=SMOKE_THREADS if smoke else None):
        model = LogisticRegression(
            C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=350,
            tol=1e-8, random_state=seed,
        )
        model.fit(
            train_x, train.y.to_numpy(int),
            sample_weight=train.cluster_weight.to_numpy(float),
        )
        probability = model.predict_proba(target_x)[:, 1]
        logit_values = model.decision_function(target_x)
    require(int(model.n_iter_[0]) < model.max_iter, "Haar scattering logistic did not converge")
    require(np.isfinite(probability).all(), "non-finite Haar probability")
    names = transform.feature_names or []
    modulus_mask = np.asarray(["ABS_" in name or "S2_" in name for name in names], bool)
    second_mask = np.asarray(["::S2_" in name for name in names], bool)
    require(modulus_mask.any() and second_mask.any(), "Haar nonlinear scattering branch missing")
    target_modulus_energy = np.mean(target_scattering[:, modulus_mask] ** 2, axis=1)
    target_second_energy = np.mean(target_scattering[:, second_mask] ** 2, axis=1)
    predictions = {
        "probability": np.clip(np.asarray(probability, float), 1e-6, 1.0 - 1e-6),
        "logit": np.asarray(logit_values, float),
        "modulus_energy": np.asarray(target_modulus_energy, float),
        "second_order_energy": np.asarray(target_second_energy, float),
    }
    audit = {
        "architecture": "fixed multiresolution Haar signed-detail/modulus/second-order scattering plus static L2 logistic",
        "channels": {name: list(columns) for name, columns in HORIZON_CHANNELS.items()},
        "raw_causal_numeric_feature_n": len(FEATURES),
        "scattering_feature_n": train_scattering.shape[1],
        "modulus_feature_n": int(modulus_mask.sum()),
        "second_order_feature_n": int(second_mask.sum()),
        "label_used_for_haar_or_scaling": False,
        "target_rows_used_for_scaling_or_training": False,
        "logistic_c": LOGISTIC_C,
        "logistic_iterations": int(model.n_iter_[0]),
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "target_logit_quantiles": {
            str(q): float(np.quantile(logit_values, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_modulus_energy_quantiles": {
            str(q): float(np.quantile(target_modulus_energy, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_second_order_energy_quantiles": {
            str(q): float(np.quantile(target_second_energy, q)) for q in (0.1, 0.5, 0.9)
        },
        "source_category_issuer_ticker_text_used": False,
        "kernel_tree_graph_density_prototype_knockoff_block_aggregate_used": False,
        "training_n": len(train), "target_n": len(target),
        "device": "CPU", "gpu_used": False,
        "cpu_reason": "deterministic short Haar cascades and one small two-thread logistic fit",
    }
    return predictions, audit


def choose_inner(
    train: pd.DataFrame, valid: pd.DataFrame, champion_valid: pd.DataFrame,
    fold: int, smoke: bool,
) -> dict[str, Any]:
    predictions, model_audit = fit_haar_scattering(train, valid, SEED + fold * 10, smoke)
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
            "name": f"HAAR_SCATTERING_W{weight:.2f}", "weight": float(weight),
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
        "haar_scattering_model_audit": model_audit,
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v95_model"] = "V69_NOOP"
    diagnostic["v95_blend_weight"] = 0.0
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
        outer_audit = chronology(outer_train, outer_valid, f"V95 outer fold {fold}")
        inner_train, inner_valid, inner_champion, inner_audit = inner_partition(
            dev, champion, pd.Timestamp(outer_valid.event_time_utc.min())
        )
        policy = choose_inner(inner_train, inner_valid, inner_champion, fold, smoke)
        selected = policy["selected"]
        predictions, outer_model_audit = fit_haar_scattering(
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
        diagnostic.loc[positions, "v95_model"] = selected["name"]
        diagnostic.loc[positions, "v95_blend_weight"] = weight
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["haar_scattering_prob"] = predictions["probability"]
        evidence["haar_logit"] = predictions["logit"]
        evidence["modulus_energy"] = predictions["modulus_energy"]
        evidence["second_order_energy"] = predictions["second_order_energy"]
        evidence["candidate_prob"] = candidate
        evidence["v95_model"] = selected["name"]
        evidence["v95_blend_weight"] = weight
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold, "inner_chronology": inner_audit, "outer_chronology": outer_audit,
            "policy": policy, "outer_haar_scattering_model_audit": outer_model_audit,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V95 HAAR] fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f} "
            f"second_energy={float(np.median(predictions['second_order_energy'])):.4f}",
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
        "method": "paired fold/date block bootstrap over locked V95 Haar-scattering policies",
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
    model_audits = [audit["policy"]["haar_scattering_model_audit"] for audit in audits] + [
        audit["outer_haar_scattering_model_audit"] for audit in audits
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
        "fixed_label_independent_haar_all_fits": all(
            not audit["label_used_for_haar_or_scaling"]
            and audit["modulus_feature_n"] > 0
            and audit["second_order_feature_n"] > 0 for audit in model_audits
        ),
        "causal_inputs_only": all(
            not audit["source_category_issuer_ticker_text_used"]
            and not audit["target_rows_used_for_scaling_or_training"] for audit in model_audits
        ),
        "no_prior_family_surrogate": all(
            not audit["kernel_tree_graph_density_prototype_knockoff_block_aggregate_used"]
            for audit in model_audits
        ),
        "second_order_energy_nonconstant": bool(
            np.isfinite(evidence.second_order_energy.to_numpy(float)).all()
            and float(evidence.second_order_energy.std()) > 1e-10
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
        "scattering_diagnostics": {
            "haar_logit_quantiles": {
                str(q): float(np.quantile(evidence.haar_logit, q)) for q in (0.1, 0.5, 0.9)
            },
            "modulus_energy_quantiles": {
                str(q): float(np.quantile(evidence.modulus_energy, q)) for q in (0.1, 0.5, 0.9)
            },
            "second_order_energy_quantiles": {
                str(q): float(np.quantile(evidence.second_order_energy, q)) for q in (0.1, 0.5, 0.9)
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
    require(not out.name.startswith("output_V95"), "runner must not write output_V95 directly")
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
            "version": "V95", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V95_HAAR_SCATTERING_DIAGNOSTIC": evaluation["candidate_summary"],
                "V95_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V95", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V95", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V95_HAAR_SCATTERING_REPORT.json": {
            "version": "V95", "hypothesis": HYPOTHESIS,
            "channels": HORIZON_CHANNELS, "logistic_c": LOGISTIC_C,
            "blend_weights": BLEND_WEIGHTS, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V95_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V95_HAAR_SCATTERING_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V95_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V95_HAAR_SCATTERING_OUTER_OOF.csv.gz", "V95_FAIL_CLOSED_SELECTED_OOF.csv.gz",
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
                "representation": "fixed signed-detail/modulus/second-order Haar scattering over causal horizon channels",
                "head": "one static L2 logistic",
                "model_inputs": "32 causal pre-event ordered numeric measurements only",
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V94 root runners plus fixed V94 declaration",
                "haar_wavelet_scattering_found": False,
                "disjoint_from_v84_v87": "no RFF kernel and no class prototype or DTW distance",
                "disjoint_from_v93_v94": "no knockoff selection and no chronological block coefficient aggregation",
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
        "haar_logistic_fit_count": 2 * len(audits),
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "scattering_diagnostics": evaluation["scattering_diagnostics"],
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
