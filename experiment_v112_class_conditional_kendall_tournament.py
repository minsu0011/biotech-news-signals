"""V112 class-conditional Kendall tournament causal-state challenger.

Each same-market past training cut robustly scales the 37 causal numeric
coordinates and collapses duplicate rows to equal-weight event-group
centroids.  A fixed tournament records every pairwise coordinate ordering
inside each of eight semantic blocks, plus every pairwise ordering of the
eight within-row block medians.  Class-specific Jeffreys-smoothed ternary
preference tables model left-below, tie, or left-above outcomes.  Equal-weight
group pseudo-log-likelihoods yield a direct UP-minus-DOWN ordinal score with a
train-only robust temperature and no fitted discriminative head.

This is not target ranking across events, a symbol sequence/context code,
TAN dependency tree, graph/topology, conformal p-value, density/covariance,
hyperspherical vMF model, density-power objective, archetype/simplex,
neighbor, kernel, tree, or neural model.  Earlier committed V69 OOF alone
chooses exact V69 or fixed .25/.50 blends in the inner past.  Outer labels are
evaluation-only under strict 35-minute nested chronology; V69 confidence and
high_conf remain exact.  Material failure restores the exact entire atomic
V69 DataFrame after canonical 14-gate recomputation.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_name] = "2"

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V112_CLASS_CONDITIONAL_KENDALL_TOURNAMENT_V1"
VERSION = 112
HYPOTHESIS = "CLASS_CONDITIONAL_KENDALL_TOURNAMENT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

JEFFREYS_COUNT = 0.5
TIE_TOLERANCE = 1e-12
TEMPERATURE_FLOOR = 0.02
SEED = 11201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)

SEMANTIC_BLOCKS = {
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
    {"name": "KENDALL_TOURNAMENT_W0.25", "weight": 0.25},
    {"name": "KENDALL_TOURNAMENT_W0.50", "weight": 0.50},
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


FEATURE_INDEX = {name: position for position, name in enumerate(FEATURES)}
BLOCK_INDICES = {
    block: np.asarray([FEATURE_INDEX[name] for name in names], dtype=int)
    for block, names in SEMANTIC_BLOCKS.items()
}
BLOCK_NAMES = tuple(SEMANTIC_BLOCKS)
WITHIN_BLOCK_PAIRS = tuple(
    (block, FEATURE_INDEX[names[left]], FEATURE_INDEX[names[right]])
    for block, names in SEMANTIC_BLOCKS.items()
    for left in range(len(names))
    for right in range(left + 1, len(names))
)
BLOCK_MEDIAN_PAIRS = tuple(
    ("BLOCK_MEDIAN_ORDER", left, right)
    for left in range(len(BLOCK_NAMES))
    for right in range(left + 1, len(BLOCK_NAMES))
)
PAIR_GROUPS = tuple(SEMANTIC_BLOCKS) + ("BLOCK_MEDIAN_ORDER",)
PAIR_GROUP_POSITIONS = {
    group: np.asarray(
        [position for position, item in enumerate(WITHIN_BLOCK_PAIRS + BLOCK_MEDIAN_PAIRS) if item[0] == group],
        dtype=int,
    ) for group in PAIR_GROUPS
}


def verify_static_spec() -> dict[str, Any]:
    flattened = [name for names in SEMANTIC_BLOCKS.values() for name in names]
    require(len(flattened) == len(FEATURES), "V112 semantic blocks do not cover 37 coordinates")
    require(len(set(flattened)) == len(FEATURES), "V112 semantic blocks repeat a coordinate")
    require(set(flattened) == set(FEATURES), "V112 semantic blocks changed causal feature authority")
    require(len(WITHIN_BLOCK_PAIRS) == 81, "V112 within-block tournament pair count changed")
    require(len(BLOCK_MEDIAN_PAIRS) == 28, "V112 block-median tournament pair count changed")
    require(all(len(PAIR_GROUP_POSITIONS[group]) > 0 for group in PAIR_GROUPS), "V112 empty pair group")
    payload = "\n".join(
        f"{group}|{left}|{right}" for group, left, right in WITHIN_BLOCK_PAIRS + BLOCK_MEDIAN_PAIRS
    )
    block_payload = "\n".join(f"{block}|{','.join(names)}" for block, names in SEMANTIC_BLOCKS.items())
    return {
        "feature_n": len(FEATURES), "semantic_block_n": len(SEMANTIC_BLOCKS),
        "semantic_block_sizes": {name: len(items) for name, items in SEMANTIC_BLOCKS.items()},
        "block_assignment_sha256": hashlib.sha256(block_payload.encode("utf-8")).hexdigest(),
        "within_block_pair_n": len(WITHIN_BLOCK_PAIRS),
        "block_median_pair_n": len(BLOCK_MEDIAN_PAIRS),
        "total_pair_n": len(WITHIN_BLOCK_PAIRS) + len(BLOCK_MEDIAN_PAIRS),
        "pair_group_n": len(PAIR_GROUPS),
        "pair_group_sizes": {group: len(PAIR_GROUP_POSITIONS[group]) for group in PAIR_GROUPS},
        "pair_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_features_exactly_once": True, "pair_spec_label_independent": True,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, metrics, robust numeric transform, blending, canonical gate, and atomic IO only; no V85 output is read",
    }
    audit["v112_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    process = psutil.Process()
    process.cpu_affinity(list(PREPARATION_CPU_AFFINITY))
    actual = tuple(sorted(process.cpu_affinity()))
    require(actual == PREPARATION_CPU_AFFINITY, "V112 preparation CPU affinity is not exactly 30-31")
    return {
        "requested_logical_cpus": list(PREPARATION_CPU_AFFINITY),
        "actual_logical_cpus": list(actual), "exact_match": True,
        "process_id": process.pid,
    }


def group_centroids(
    frame: pd.DataFrame, matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(frame) == len(matrix), "V112 centroid input shape mismatch")
    state = pd.DataFrame(matrix, columns=FEATURES)
    state["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    state["y"] = frame.y.to_numpy(int)
    grouped = state.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V112 event group crosses direction labels")
    centroid = grouped[list(FEATURES)].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    require(np.isfinite(centroid).all(), "V112 group centroid is non-finite")
    return centroid, target, group_ids


def tournament_outcomes(matrix: np.ndarray) -> np.ndarray:
    require(matrix.ndim == 2 and matrix.shape[1] == len(FEATURES), "V112 tournament matrix invalid")
    block_median = np.column_stack([
        np.median(matrix[:, BLOCK_INDICES[block]], axis=1) for block in BLOCK_NAMES
    ])
    result = np.empty((len(matrix), STATIC_SPEC["total_pair_n"]), dtype=np.int8)
    for position, (_, left, right) in enumerate(WITHIN_BLOCK_PAIRS):
        difference = matrix[:, left] - matrix[:, right]
        result[:, position] = np.where(
            difference > TIE_TOLERANCE, 2,
            np.where(difference < -TIE_TOLERANCE, 0, 1),
        )
    offset = len(WITHIN_BLOCK_PAIRS)
    for relative, (_, left, right) in enumerate(BLOCK_MEDIAN_PAIRS):
        difference = block_median[:, left] - block_median[:, right]
        result[:, offset + relative] = np.where(
            difference > TIE_TOLERANCE, 2,
            np.where(difference < -TIE_TOLERANCE, 0, 1),
        )
    require(np.all((result >= 0) & (result <= 2)), "V112 tournament outcome outside ternary alphabet")
    return result


def fit_class_tournament(outcomes: np.ndarray, target: np.ndarray, label: int) -> dict[str, Any]:
    selected = outcomes[target == label]
    require(len(selected) >= 80, f"V112 class {label} event-group support too small")
    counts = np.full((outcomes.shape[1], 3), JEFFREYS_COUNT, dtype=float)
    for outcome in range(3):
        counts[:, outcome] += np.sum(selected == outcome, axis=0)
    probability = counts / counts.sum(axis=1, keepdims=True)
    require(np.isfinite(probability).all() and np.all(probability > 0.0), "V112 class table invalid")
    return {
        "label": label, "probability": probability, "log_probability": np.log(probability),
        "event_group_n": len(selected), "jeffreys_count": JEFFREYS_COUNT,
        "probability_min": float(probability.min()), "probability_max": float(probability.max()),
        "probability_sha256": array_sha256(probability),
        "counts_sha256": array_sha256(counts),
    }


def class_tournament_log_score(
    outcomes: np.ndarray, table: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pair = np.arange(outcomes.shape[1], dtype=int)[None, :]
    contribution = table["log_probability"][pair, outcomes]
    group_scores = {
        group: np.mean(contribution[:, positions], axis=1)
        for group, positions in PAIR_GROUP_POSITIONS.items()
    }
    score = np.mean(np.column_stack(list(group_scores.values())), axis=1)
    require(np.isfinite(score).all(), "V112 tournament log score invalid")
    return score, group_scores


def tournament_contrast(
    outcomes: np.ndarray, down_table: dict[str, Any], up_table: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    down_score, down_groups = class_tournament_log_score(outcomes, down_table)
    up_score, up_groups = class_tournament_log_score(outcomes, up_table)
    contrast = up_score - down_score
    return contrast, {
        "down_group_mean": {name: float(np.mean(value)) for name, value in down_groups.items()},
        "up_group_mean": {name: float(np.mean(value)) for name, value in up_groups.items()},
        "outcome_fractions": {
            str(outcome): float(np.mean(outcomes == outcome)) for outcome in range(3)
        },
        "outcomes_sha256": array_sha256(outcomes),
    }


def kendall_tournament_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V112 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V112 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_state, train_y, train_groups = group_centroids(train, processor.transform(train))
    target_state = processor.transform(target_frame)
    train_outcomes = tournament_outcomes(train_state)
    target_outcomes = tournament_outcomes(target_state)
    down_table = fit_class_tournament(train_outcomes, train_y, 0)
    up_table = fit_class_tournament(train_outcomes, train_y, 1)
    train_contrast, train_score_audit = tournament_contrast(train_outcomes, down_table, up_table)
    lower = float(np.quantile(train_contrast, 0.25))
    upper = float(np.quantile(train_contrast, 0.75))
    temperature = max((upper - lower) / 1.349, TEMPERATURE_FLOOR)
    target_contrast, target_score_audit = tournament_contrast(target_outcomes, down_table, up_table)
    probability = np.clip(expit(target_contrast / temperature), 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V112 probability invalid")
    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES), "causal_numeric_only": True,
        "categorical_feature_n": 0, "text_feature_n": 0,
        "source_or_ticker_identity": False,
        "architecture": "class-conditional Jeffreys-smoothed ternary Kendall tournament pseudo-likelihood",
        "static_spec": STATIC_SPEC, "train_scaling_only": True,
        "event_group_equal_weight": True, "train_event_group_n": len(train_groups),
        "down_table": {key: value for key, value in down_table.items() if key not in {"probability", "log_probability"}},
        "up_table": {key: value for key, value in up_table.items() if key not in {"probability", "log_probability"}},
        "train_score_audit": train_score_audit, "target_score_audit": target_score_audit,
        "temperature": temperature, "temperature_fit_on_train_only": True,
        "temperature_floor": TEMPERATURE_FLOOR,
        "ternary_outcomes": ["LEFT_BELOW_RIGHT", "TIE", "LEFT_ABOVE_RIGHT"],
        "tie_tolerance": TIE_TOLERANCE,
        "fitted_direction_coefficient_n": 0, "learned_discriminative_head": False,
        "target_ranking_across_events": False, "symbol_context_or_kt_code": False,
        "dependency_tree_or_graph": False, "conformal_p_value": False,
        "density_covariance_or_vmf": False, "density_power_objective": False,
        "archetype_simplex_neighbor_kernel_or_neural": False,
        "target_rows_used_for_scaling_table_or_temperature": False,
        "target_labels_used": False,
        "train_contrast_quantiles": {
            str(q): float(np.quantile(train_contrast, q)) for q in quantiles
        },
        "target_contrast_quantiles": {str(q): float(np.quantile(target_contrast, q)) for q in quantiles},
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V112 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V112 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = kendall_tournament_direction(inner_train, inner_valid)
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
    model_eligible = bool(
        model_audit["train_scaling_only"]
        and model_audit["event_group_equal_weight"]
        and model_audit["static_spec"]["pair_spec_label_independent"]
        and model_audit["static_spec"]["total_pair_n"] == 109
        and model_audit["down_table"]["event_group_n"] >= 80
        and model_audit["up_table"]["event_group_n"] >= 80
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["learned_discriminative_head"]
        and not model_audit["target_rows_used_for_scaling_table_or_temperature"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "model_eligible": model_eligible,
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 Kendall-tournament logit blend; fixed score and BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v112_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V112 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = kendall_tournament_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v112_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["kendall_tournament_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v112_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V112 KENDALL TOURNAMENT] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V112 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V112 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V112 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V112 nested-bootstrap blocks")
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
        ba_delta.append(float(
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V112 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V112 policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V112/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    tournament_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["train_scaling_only"]
        and audit["outer_model_audit"]["event_group_equal_weight"]
        and audit["outer_model_audit"]["static_spec"]["pair_spec_label_independent"]
        and audit["outer_model_audit"]["static_spec"]["total_pair_n"] == 109
        and audit["outer_model_audit"]["down_table"]["event_group_n"] >= 80
        and audit["outer_model_audit"]["up_table"]["event_group_n"] >= 80
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["learned_discriminative_head"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_table_or_temperature"]
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
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "class_conditional_kendall_tournament_contract_verified": tournament_valid,
    }
    material_pass = bool(all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V112 fallback is not exact entire V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
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
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V112 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V112 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V112 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V112 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V112", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V112_DIAGNOSTIC_KENDALL_TOURNAMENT": evaluation["candidate_summary"],
                "V112_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V112", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V112", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "CLASS_CONDITIONAL_KENDALL_TOURNAMENT_REPORT.json": {
            "version": "V112", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "semantic_blocks": SEMANTIC_BLOCKS,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V112", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
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
        "train_event_group_n": audit["outer_model_audit"]["train_event_group_n"],
        "down_event_group_n": audit["outer_model_audit"]["down_table"]["event_group_n"],
        "up_event_group_n": audit["outer_model_audit"]["up_table"]["event_group_n"],
        "ordinal_pair_n": audit["outer_model_audit"]["static_spec"]["total_pair_n"],
        "strict_outer_35m_embargo": audit["outer_chronology"]["strict_35m_embargo"],
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V112_KENDALL_TOURNAMENT_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V112_KENDALL_TOURNAMENT_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V112_DIAGNOSTIC_KENDALL_TOURNAMENT_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V112_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V112 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    affinity_audit = preparation_affinity() if (args.audit_only or args.smoke_test) else {
        "requested_logical_cpus": None, "actual_logical_cpus": psutil.Process().cpu_affinity(),
        "exact_match": None, "reason": "controller-authorized full run is not preparation",
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "class-conditional fixed ternary Kendall tournament pseudo-likelihood",
            "ordinal_pair_n": STATIC_SPEC["total_pair_n"],
            "pair_group_n": STATIC_SPEC["pair_group_n"],
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_identity": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "exact_entire_v69_fallback": True,
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "cpu_affinity": affinity_audit,
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
        "inner_baseline": audits[0]["policy"]["baseline"],
        "inner_selected": audits[0]["policy"]["selected"],
        "inner_best_non_noop": best_non_noop,
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_evaluation_only_architectures": audits[0]["outer_architecture_diagnostics_evaluation_only"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "model_audit": audits[0]["outer_model_audit"],
        "current_material_gate": current_material_gate(evaluation),
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "cpu_affinity": affinity_audit, "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    output = write_outputs(args.output, authority, evidence, audits, evaluation)
    summary["output_written"] = True
    summary["output"] = output
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
