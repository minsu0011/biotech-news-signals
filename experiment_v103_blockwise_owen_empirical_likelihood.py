"""V103 blockwise Owen empirical-likelihood causal-state challenger.

Within each embargoed same-market training cut, 37 causal pre-decision numeric
states are robustly scaled and reduced to eight fixed semantic-block medians.
For every direction class and block, event-group duplicate-balanced past values
define an Owen empirical-likelihood deviance curve.  The curve is evaluated on
a fixed 97-point train-only central grid by solving the one-moment EL dual
equation.  Query direction follows the composite DOWN-minus-UP deviance and a
train-only robust IQR temperature.

There is no fitted direction coefficient/head, ordered/fused-lasso path,
SVD/subspace reconstruction, neighbor/distance/prototype score, covariance or
probability density, copula, projection depth, tree, neural/kernel map, or
literal-pattern belief system.  Earlier committed V69 OOF labels alone choose
exact V69 or fixed 0.25/0.50 logit blends on the inner past.  Outer labels are
evaluation-only, all splits enforce a strict 35-minute embargo and event-group
purge, V69 confidence/high_conf stay exact, and material failure restores the
exact entire atomic V69 frame after recomputing canonical 14 gates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "2")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V103_BLOCKWISE_OWEN_EMPIRICAL_LIKELIHOOD_V1"
VERSION = 103
HYPOTHESIS = "BLOCKWISE_OWEN_EMPIRICAL_LIKELIHOOD_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

EL_GRID_POINTS = 97
EL_GRID_QUANTILES = (0.01, 0.99)
EL_BISECTION_STEPS = 48
EL_DENOMINATOR_FLOOR = 1e-12
TEMPERATURE_FLOOR = 0.02
SEED = 10301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
COST = scaffold.COST

BLOCKS = {
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
    "INTRADAY_STRUCTURE": (
        "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
        "up_fraction_30", "trend_slope_30", "trend_slope_60", "vwap_distance_30",
    ),
    "CLOCK": ("minutes_from_open", "minutes_to_close"),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
}

ARCHITECTURES = (
    {"name": "OWEN_EMPIRICAL_LIKELIHOOD_W0.25", "weight": 0.25},
    {"name": "OWEN_EMPIRICAL_LIKELIHOOD_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
chronology = scaffold.chronology
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metrics, blending, robust scaling, and IO only; no V85 output is read",
    }
    audit["v103_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def block_indices() -> dict[str, np.ndarray]:
    feature_index = {name: position for position, name in enumerate(FEATURES)}
    flattened = tuple(name for names in BLOCKS.values() for name in names)
    require(len(flattened) == len(FEATURES), "V103 block partition length changed")
    require(len(set(flattened)) == len(FEATURES), "V103 block partition repeats a feature")
    require(set(flattened) == set(FEATURES), "V103 block partition does not cover FEATURES")
    return {
        block: np.asarray([feature_index[name] for name in names], dtype=int)
        for block, names in BLOCKS.items()
    }


BLOCK_INDICES = block_indices()


def block_state(matrix: np.ndarray) -> np.ndarray:
    result = np.column_stack([
        np.median(matrix[:, indices], axis=1) for indices in BLOCK_INDICES.values()
    ])
    require(result.shape == (len(matrix), len(BLOCKS)), "V103 block-state shape changed")
    require(np.isfinite(result).all(), "V103 block state is non-finite")
    return result


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= float(weights.sum())
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "V103 duplicate weights invalid")
    return weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    require(len(values) == len(weights) and 0.0 <= probability <= 1.0, "V103 weighted quantile contract failed")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= float(cumulative[-1])
    position = int(np.searchsorted(cumulative, probability, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def owen_deviance(values: np.ndarray, weights: np.ndarray, candidate_mean: float) -> float:
    difference = values - candidate_mean
    moment = float(weights @ difference)
    if abs(moment) <= 1e-12:
        return 0.0
    positive = difference[difference > 1e-12]
    negative = difference[difference < -1e-12]
    if not len(positive) or not len(negative):
        return 1e6
    lower = (-1.0 / float(np.max(positive))) * (1.0 - 1e-10)
    upper = (-1.0 / float(np.min(negative))) * (1.0 - 1e-10)
    require(lower < 0.0 < upper, "V103 Owen dual interval invalid")
    for _ in range(EL_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        denominator = 1.0 + middle * difference
        require(np.all(denominator > 0.0), "V103 Owen dual left feasible domain")
        equation = float(np.sum(weights * difference / denominator))
        if equation > 0.0:
            lower = middle
        else:
            upper = middle
    multiplier = 0.5 * (lower + upper)
    denominator = np.maximum(1.0 + multiplier * difference, EL_DENOMINATOR_FLOOR)
    deviance = float(2.0 * np.sum(weights * np.log(denominator)))
    require(math.isfinite(deviance), "V103 Owen deviance is non-finite")
    return max(deviance, 0.0)


def empirical_likelihood_curve(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lower = weighted_quantile(values, weights, EL_GRID_QUANTILES[0])
    upper = weighted_quantile(values, weights, EL_GRID_QUANTILES[1])
    if upper - lower <= 1e-8:
        center = float(weights @ values)
        grid = np.linspace(center - 1e-6, center + 1e-6, EL_GRID_POINTS)
        deviance = np.zeros(EL_GRID_POINTS, dtype=float)
        return grid, deviance, {
            "grid_min": float(grid[0]), "grid_max": float(grid[-1]),
            "grid_points": len(grid), "weighted_mean": center,
            "minimum_deviance": 0.0, "grid_mean_deviance": 0.0,
            "maximum_deviance": 0.0, "degenerate_constant_block": True,
            "curve_sha256": array_sha256(np.column_stack([grid, deviance])),
        }
    grid = np.linspace(lower, upper, EL_GRID_POINTS)
    deviance = np.asarray([owen_deviance(values, weights, value) for value in grid], dtype=float)
    require(np.isfinite(deviance).all() and np.all(deviance >= 0.0), "V103 EL curve invalid")
    weighted_mean = float(weights @ values)
    mean_position = int(np.argmin(np.abs(grid - weighted_mean)))
    return grid, deviance, {
        "grid_min": float(grid[0]), "grid_max": float(grid[-1]),
        "grid_points": len(grid), "weighted_mean": weighted_mean,
        "minimum_deviance": float(np.min(deviance)),
        "grid_mean_deviance": float(deviance[mean_position]),
        "maximum_deviance": float(np.max(deviance)),
        "degenerate_constant_block": False,
        "curve_sha256": array_sha256(np.column_stack([grid, deviance])),
    }


def interpolate_curve(values: np.ndarray, grid: np.ndarray, deviance: np.ndarray) -> np.ndarray:
    result = np.interp(values, grid, deviance, left=deviance[0], right=deviance[-1])
    require(np.isfinite(result).all() and np.all(result >= 0.0), "V103 interpolated deviance invalid")
    return result


def robust_temperature(train_margin: np.ndarray) -> tuple[float, dict[str, Any]]:
    lower, median, upper = np.quantile(train_margin, (0.25, 0.50, 0.75))
    iqr_scale = float((upper - lower) / 1.349)
    mad_scale = float(1.4826 * np.median(np.abs(train_margin - median)))
    standard_scale = float(np.std(train_margin))
    scale = iqr_scale if math.isfinite(iqr_scale) and iqr_scale > 1e-10 else (
        mad_scale if math.isfinite(mad_scale) and mad_scale > 1e-10 else standard_scale
    )
    scale = max(float(scale), TEMPERATURE_FLOOR)
    return scale, {
        "method": "train-only composite-margin IQR/1.349 with deterministic MAD/std fallback",
        "temperature": scale, "temperature_floor": TEMPERATURE_FLOOR,
        "iqr_scale": iqr_scale, "mad_scale": mad_scale, "standard_scale": standard_scale,
        "train_margin_quantiles": {
            str(q): float(np.quantile(train_margin, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        },
        "target_batch_used": False,
    }


def blockwise_owen_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V103 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V103 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_block = block_state(processor.transform(train))
    target_block = block_state(processor.transform(target_frame))
    target = train.y.to_numpy(int)
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    curve_audit: dict[str, Any] = {}
    class_support: dict[str, int] = {}
    class_group_support: dict[str, int] = {}
    for label in (0, 1):
        index = target == label
        require(int(index.sum()) >= 100, f"V103 class {label} support too small")
        class_frame = train.loc[index].reset_index(drop=True)
        weight = duplicate_weights(class_frame)
        class_support[str(label)] = int(index.sum())
        class_group_support[str(label)] = int(class_frame.event_group_id.astype(str).nunique())
        for block_position, block_name in enumerate(BLOCKS):
            grid, deviance, audit = empirical_likelihood_curve(train_block[index, block_position], weight)
            curves[(label, block_position)] = (grid, deviance)
            curve_audit[f"{label}|{block_name}"] = audit

    def composite(matrix: np.ndarray, label: int) -> np.ndarray:
        parts = []
        for block_position in range(len(BLOCKS)):
            grid, deviance = curves[(label, block_position)]
            parts.append(interpolate_curve(matrix[:, block_position], grid, deviance))
        return np.mean(np.column_stack(parts), axis=1)

    train_down = composite(train_block, 0)
    train_up = composite(train_block, 1)
    train_margin = train_down - train_up
    temperature, temperature_audit = robust_temperature(train_margin)
    target_down = composite(target_block, 0)
    target_up = composite(target_block, 1)
    raw_margin = target_down - target_up
    probability = expit(np.clip(raw_margin / temperature, -30.0, 30.0))
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V103 probability invalid")
    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "feature_n": len(FEATURES), "block_n": len(BLOCKS),
        "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
        "source_feature": False, "ticker_feature": False,
        "architecture": "blockwise duplicate-balanced Owen empirical-likelihood composite deviance",
        "block_aggregation": "label-independent row median after train-cut robust scaling",
        "block_members": {name: list(members) for name, members in BLOCKS.items()},
        "class_support": class_support, "class_unique_event_groups": class_group_support,
        "el_grid_points": EL_GRID_POINTS, "el_grid_quantiles": EL_GRID_QUANTILES,
        "el_bisection_steps": EL_BISECTION_STEPS,
        "curve_audit": curve_audit, "temperature": temperature_audit,
        "event_group_duplicate_balanced": True,
        "query_specific_dual_refit": False,
        "curves_fit_on_train_only": True,
        "fitted_direction_coefficient_n": 0,
        "ordered_or_fused_lasso": False,
        "svd_or_subspace": False,
        "distance_neighbor_or_prototype": False,
        "covariance_copula_or_probability_density": False,
        "projection_or_depth": False,
        "tree_neural_or_kernel_map": False,
        "pattern_literal_or_belief_mass": False,
        "target_rows_used_for_scaling_curves_or_temperature": False,
        "target_rows_used_only_as_curve_queries": True,
        "target_labels_used": False,
        "down_composite_deviance_quantiles": {str(q): float(np.quantile(target_down, q)) for q in quantiles},
        "up_composite_deviance_quantiles": {str(q): float(np.quantile(target_up, q)) for q in quantiles},
        "raw_margin_quantiles": {str(q): float(np.quantile(raw_margin, q)) for q in quantiles},
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V103 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V103 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = blockwise_owen_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(
            2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]
            + 10.0 * baseline_metric["all_trade_mean_signed_net"]
        ),
    }]
    model_eligible = bool(
        model_audit["curves_fit_on_train_only"]
        and model_audit["event_group_duplicate_balanced"]
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["ordered_or_fused_lasso"]
        and not model_audit["svd_or_subspace"]
        and not model_audit["target_rows_used_for_scaling_curves_or_temperature"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture,
            "metrics": values, "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 Owen empirical-likelihood blend; maximize 2*AUC+BA+10*net with fixed model/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v103_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([("US", 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V103 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = blockwise_owen_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(
            baseline, challenger, selected["architecture"]["weight"],
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v103_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
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
        evidence["owen_empirical_likelihood_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v103_model"] = selected["name"]
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
            f"[V103 OWEN-EL] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V103 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V103 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V103 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V103 bootstrap blocks")
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
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V103 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-fold-time block bootstrap over inner-locked V103 Owen-EL policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta),
        "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V103/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    el_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["curves_fit_on_train_only"]
        and audit["outer_model_audit"]["event_group_duplicate_balanced"]
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["ordered_or_fused_lasso"]
        and not audit["outer_model_audit"]["svd_or_subspace"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_curves_or_temperature"]
        and audit["outer_model_audit"]["target_rows_used_only_as_curve_queries"]
        and not audit["outer_model_audit"]["target_labels_used"]
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
            and audit["policy_locked_before_outer_evaluation"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "blockwise_owen_empirical_likelihood_contract_verified": el_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V103 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap, "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
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
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def selected_contract(evaluation: dict[str, Any]) -> dict[str, Any]:
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    nested_bootstrap = json.loads(json.dumps(clean(evaluation["bootstrap"])))
    nested_bootstrap["contract_key"] = "selected.material_gate.nested_bootstrap"
    selected["material_gate"] = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
        "nested_bootstrap": nested_bootstrap,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 frame",
    }
    require(
        selected["material_gate"]["nested_bootstrap"]["contract_key"]
        == "selected.material_gate.nested_bootstrap",
        "V103 nested bootstrap contract key mismatch",
    )
    require("bootstrap" in selected["robustness"], "V103 selected native robustness bootstrap missing")
    return selected


def write_outputs(
    authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]],
    evaluation: dict[str, Any], out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = selected_contract(evaluation)
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V103", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V103_DIAGNOSTIC_BLOCKWISE_OWEN_EL": evaluation["candidate_summary"],
                "V103_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V103", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"], "bootstrap": evaluation["bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V103", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "bootstrap": evaluation["bootstrap"], "material_gate": selected["material_gate"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "BLOCKWISE_OWEN_EMPIRICAL_LIKELIHOOD_REPORT.json": {
            "version": "V103", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "blocks": BLOCKS, "architectures": ARCHITECTURES,
            "el_grid_points": EL_GRID_POINTS, "el_grid_quantiles": EL_GRID_QUANTILES,
            "el_bisection_steps": EL_BISECTION_STEPS,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": {
            "version": "V103", "hypothesis": HYPOTHESIS, "bootstrap": evaluation["bootstrap"],
            "nested_key_contract": {
                "auc_probability": "bootstrap.auc_delta.probability_gt_zero",
                "net_probability": "bootstrap.all_trade_net_delta.probability_gt_zero",
            },
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V103", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": selected["material_gate"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": scaffold.now(),
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
        "temperature": audit["outer_model_audit"]["temperature"]["temperature"],
        "el_grid_points": audit["outer_model_audit"]["el_grid_points"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V103_OWEN_EL_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V103_OWEN_EL_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V103_DIAGNOSTIC_OWEN_EL_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V103_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "feature_n": len(FEATURES), "blocks": BLOCKS,
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_feature": False,
            "architecture": "blockwise duplicate-balanced Owen empirical-likelihood composite deviance",
            "el_grid_points": EL_GRID_POINTS, "el_bisection_steps": EL_BISECTION_STEPS,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "selected_contains_current_material_gate": True,
                "bootstrap_nested_keys": [
                    "selected.material_gate.nested_bootstrap.auc_delta.probability_gt_zero",
                    "selected.material_gate.nested_bootstrap.all_trade_net_delta.probability_gt_zero",
                ],
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0,
                "reason": "bounded one-dimensional dual curves and two-thread smoke avoid full-run contention",
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "fold_keys": [f"{audit['market']}:{audit['fold']}" for audit in audits],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": selected_contract(evaluation)["material_gate"],
        "bootstrap": evaluation["bootstrap"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0, "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["owen_empirical_likelihood_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
