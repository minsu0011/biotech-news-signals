"""V98 empirical-Bayes contrast-pattern belief direction challenger.

For every embargoed same-market training cut, 37 causal pre-decision numeric
states are discretized by duplicate-weighted train-only tertiles.  A fixed
symbolic dictionary contains every singleton literal and only predeclared
adjacent-within-channel conjunctions.  Duplicate-balanced pattern counts feed
an empirical-Bayes Beta posterior around the train-cut class prior.  Active
patterns assign bounded mass to UP, DOWN, or ignorance; deterministic
conflict-normalized Dempster--Shafer combination and the pignistic transform
produce a direction probability.

This is not a fitted linear/tree/neural head, generative TAN density, kernel,
projection depth, wavelet map, prototype, or nearest-event retrieval model.
Earlier same-market committed V69 OOF labels select exact V69 or one of two
fixed logit blends using only the inner past.  Outer labels are evaluation-only,
all cuts enforce a 35-minute embargo and event-group purge, V69 confidence and
high-confidence membership are frozen, and any failed material gate restores
the exact entire V69 frame.  The canonical controller recomputes all 14 gates.
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V98_EMPIRICAL_BAYES_CONTRAST_PATTERN_BELIEF_V1"
VERSION = 98
HYPOTHESIS = "EMPIRICAL_BAYES_CONTRAST_PATTERN_BELIEF_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

STATE_COUNT = 4  # LOW, MID, HIGH, MISSING
TERTILES = (1.0 / 3.0, 2.0 / 3.0)
BETA_PRIOR_STRENGTH = 8.0
MIN_EFFECTIVE_SUPPORT = 12.0
MIN_ABSOLUTE_CONTRAST = 0.015
PATTERN_MASS_CAP = 0.12
MAX_ACTIVE_EVIDENCE = 12
PRIOR_MASS_CAP = 0.20
CONFLICT_FLOOR = 1e-10
BLEND_WEIGHTS = (0.25, 0.50)
SEED = 9801
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = scaffold.COST

PATTERN_CHANNELS = {
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
    {"name": "CONTRAST_BELIEF_W0.25", "weight": 0.25},
    {"name": "CONTRAST_BELIEF_W0.50", "weight": 0.50},
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
inner_partition = scaffold.inner_partition
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def adjacent_feature_pairs() -> tuple[tuple[int, int], ...]:
    index = {name: position for position, name in enumerate(FEATURES)}
    flattened = tuple(name for names in PATTERN_CHANNELS.values() for name in names)
    require(len(flattened) == len(FEATURES), "V98 channel partition size changed")
    require(set(flattened) == set(FEATURES), "V98 channel partition does not match causal features")
    pairs: list[tuple[int, int]] = []
    for names in PATTERN_CHANNELS.values():
        pairs.extend((index[left], index[right]) for left, right in zip(names[:-1], names[1:]))
    require(len(pairs) == len(set(pairs)), "V98 fixed conjunction pair repeated")
    return tuple(pairs)


ADJACENT_PAIRS = adjacent_feature_pairs()
SINGLETON_DICTIONARY_SIZE = len(FEATURES) * STATE_COUNT
PAIR_DICTIONARY_SIZE = len(ADJACENT_PAIRS) * STATE_COUNT * STATE_COUNT
PATTERN_DICTIONARY_SIZE = SINGLETON_DICTIONARY_SIZE + PAIR_DICTIONARY_SIZE
PATTERNS_PER_ROW = len(FEATURES) + len(ADJACENT_PAIRS)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metric, blending and IO utilities only; no V85 output is read",
    }
    return audit


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "V98 duplicate weights invalid")
    return weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    require(int(finite.sum()) >= 20, "V98 finite feature support too small")
    value = values[finite]
    weight = weights[finite]
    order = np.argsort(value, kind="stable")
    value = value[order]
    cumulative = np.cumsum(weight[order])
    target = probability * float(cumulative[-1])
    position = int(np.searchsorted(cumulative, target, side="left"))
    return float(value[min(position, len(value) - 1)])


def fit_tertile_cutpoints(train: pd.DataFrame) -> np.ndarray:
    raw = train.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    weights = duplicate_unit_weights(train)
    cutpoints = np.empty((len(FEATURES), 2), dtype=float)
    for column in range(len(FEATURES)):
        cutpoints[column, 0] = weighted_quantile(raw[:, column], weights, TERTILES[0])
        cutpoints[column, 1] = weighted_quantile(raw[:, column], weights, TERTILES[1])
    require(np.isfinite(cutpoints).all(), "V98 tertile cutpoints are non-finite")
    require(np.all(cutpoints[:, 0] <= cutpoints[:, 1]), "V98 tertile cutpoint order failed")
    return cutpoints


def discretize(frame: pd.DataFrame, cutpoints: np.ndarray) -> np.ndarray:
    raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    states = np.full(raw.shape, 3, dtype=np.int16)
    for column in range(raw.shape[1]):
        finite = np.isfinite(raw[:, column])
        states[finite, column] = np.searchsorted(
            cutpoints[column], raw[finite, column], side="right",
        ).astype(np.int16)
    require(np.all((states >= 0) & (states < STATE_COUNT)), "V98 state outside dictionary")
    return states


def pattern_ids(states: np.ndarray) -> np.ndarray:
    rows = np.arange(len(states))[:, None]
    columns = np.arange(len(FEATURES))[None, :]
    singleton = columns * STATE_COUNT + states
    pair_columns: list[np.ndarray] = []
    for pair_index, (left, right) in enumerate(ADJACENT_PAIRS):
        identifier = (
            SINGLETON_DICTIONARY_SIZE
            + pair_index * STATE_COUNT * STATE_COUNT
            + states[:, left] * STATE_COUNT
            + states[:, right]
        )
        pair_columns.append(identifier[:, None])
    result = np.concatenate([singleton, *pair_columns], axis=1).astype(np.int32)
    require(result.shape == (len(states), PATTERNS_PER_ROW), "V98 active pattern shape changed")
    require(int(result.min()) >= 0 and int(result.max()) < PATTERN_DICTIONARY_SIZE, "V98 pattern ID outside dictionary")
    return result


def combine_binary_masses(
    up: float, down: float, ignorance: float,
    new_up: float, new_down: float, new_ignorance: float,
) -> tuple[float, float, float, float]:
    conflict = up * new_down + down * new_up
    denominator = max(1.0 - conflict, CONFLICT_FLOOR)
    combined_up = (up * new_up + up * new_ignorance + ignorance * new_up) / denominator
    combined_down = (down * new_down + down * new_ignorance + ignorance * new_down) / denominator
    combined_ignorance = ignorance * new_ignorance / denominator
    total = combined_up + combined_down + combined_ignorance
    require(math.isfinite(total) and total > 0.0, "V98 belief mass combination failed")
    return combined_up / total, combined_down / total, combined_ignorance / total, conflict


def contrast_pattern_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V98 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V98 market fit mismatch")
    cutpoints = fit_tertile_cutpoints(train)
    train_patterns = pattern_ids(discretize(train, cutpoints))
    target_patterns = pattern_ids(discretize(target_frame, cutpoints))
    weights = duplicate_unit_weights(train)
    labels = train.y.to_numpy(int)
    flat_pattern = train_patterns.reshape(-1)
    repeated_weight = np.repeat(weights, PATTERNS_PER_ROW)
    repeated_up_weight = np.repeat(weights * labels, PATTERNS_PER_ROW)
    support = np.bincount(flat_pattern, weights=repeated_weight, minlength=PATTERN_DICTIONARY_SIZE)
    up_support = np.bincount(flat_pattern, weights=repeated_up_weight, minlength=PATTERN_DICTIONARY_SIZE)
    total_weight = float(weights.sum())
    base_probability = float((np.sum(weights * labels) + 0.5) / (total_weight + 1.0))
    alpha = 0.5 + BETA_PRIOR_STRENGTH * base_probability
    beta = 0.5 + BETA_PRIOR_STRENGTH * (1.0 - base_probability)
    posterior = (up_support + alpha) / (support + alpha + beta)
    contrast = posterior - base_probability
    reliability = np.sqrt(support / np.maximum(support + BETA_PRIOR_STRENGTH, 1e-12))
    strength = np.minimum(PATTERN_MASS_CAP, 2.0 * np.abs(contrast) * reliability)
    eligible = (support >= MIN_EFFECTIVE_SUPPORT) & (np.abs(contrast) >= MIN_ABSOLUTE_CONTRAST)
    strength = np.where(eligible, strength, 0.0)

    probability = np.empty(len(target_frame), dtype=float)
    conflict_values = np.empty(len(target_frame), dtype=float)
    ignorance_values = np.empty(len(target_frame), dtype=float)
    active_values = np.empty(len(target_frame), dtype=int)
    used_values = np.empty(len(target_frame), dtype=int)
    base_strength = min(PRIOR_MASS_CAP, 2.0 * abs(base_probability - 0.5))
    for row, identifiers in enumerate(target_patterns):
        active = identifiers[strength[identifiers] > 0.0]
        active_values[row] = len(active)
        if len(active) > MAX_ACTIVE_EVIDENCE:
            order = np.lexsort((active, -strength[active]))
            active = active[order[:MAX_ACTIVE_EVIDENCE]]
        else:
            active = np.sort(active, kind="stable")
        used_values[row] = len(active)
        if base_probability >= 0.5:
            up, down, ignorance = base_strength, 0.0, 1.0 - base_strength
        else:
            up, down, ignorance = 0.0, base_strength, 1.0 - base_strength
        accumulated_conflict = 0.0
        for identifier in active:
            mass = float(strength[identifier])
            new_up = mass if contrast[identifier] > 0.0 else 0.0
            new_down = mass if contrast[identifier] < 0.0 else 0.0
            up, down, ignorance, conflict = combine_binary_masses(
                up, down, ignorance, new_up, new_down, 1.0 - mass,
            )
            accumulated_conflict = 1.0 - (1.0 - accumulated_conflict) * (1.0 - conflict)
        probability[row] = up + 0.5 * ignorance
        conflict_values[row] = accumulated_conflict
        ignorance_values[row] = ignorance
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V98 probability is non-finite")
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "feature_n": len(FEATURES), "state_count": STATE_COUNT,
        "singleton_dictionary_n": SINGLETON_DICTIONARY_SIZE,
        "adjacent_pair_n": len(ADJACENT_PAIRS), "pair_dictionary_n": PAIR_DICTIONARY_SIZE,
        "pattern_dictionary_n": PATTERN_DICTIONARY_SIZE, "patterns_per_row": PATTERNS_PER_ROW,
        "eligible_pattern_n": int(eligible.sum()),
        "effective_duplicate_weight_n": total_weight,
        "base_up_probability": base_probability,
        "tertiles": TERTILES, "cutpoints_sha256": array_sha256(cutpoints),
        "architecture": "empirical-Bayes symbolic contrast-pattern masses plus conflict-normalized Dempster-Shafer combination",
        "duplicate_balanced_counts": True, "beta_prior_strength": BETA_PRIOR_STRENGTH,
        "minimum_effective_support": MIN_EFFECTIVE_SUPPORT,
        "minimum_absolute_contrast": MIN_ABSOLUTE_CONTRAST,
        "pattern_mass_cap": PATTERN_MASS_CAP, "maximum_active_evidence": MAX_ACTIVE_EVIDENCE,
        "target_rows_used_for_cutpoints_or_pattern_counts": False,
        "target_labels_used": False, "outer_labels_used": False,
        "causal_numeric_only": True, "categorical_feature_n": 0,
        "text_feature_n": 0, "source_feature": False, "ticker_feature": False,
        "global_fitted_coefficient_n": 0, "learned_tree_edges": 0,
        "nearest_event_retrieval": False, "kernel_or_projection": False,
        "active_pattern_quantiles": {
            str(q): float(np.quantile(active_values, q)) for q in (0.10, 0.50, 0.90)
        },
        "used_pattern_quantiles": {
            str(q): float(np.quantile(used_values, q)) for q in (0.10, 0.50, 0.90)
        },
        "conflict_quantiles": {
            str(q): float(np.quantile(conflict_values, q)) for q in (0.10, 0.50, 0.90)
        },
        "ignorance_quantiles": {
            str(q): float(np.quantile(ignorance_values, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_quantiles": {
            str(q): float(np.quantile(probability, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = contrast_pattern_direction(inner_train, inner_valid)
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
    contract_eligible = bool(
        model_audit["eligible_pattern_n"] >= 20
        and model_audit["global_fitted_coefficient_n"] == 0
        and model_audit["learned_tree_edges"] == 0
        and not model_audit["target_rows_used_for_cutpoints_or_pattern_counts"]
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
            "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "model_contract_eligible": contract_eligible,
            "eligible": bool(contract_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 contrast-belief logit blend; maximize fixed 2*AUC+BA+10*net under contract/BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v98_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V98 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = contrast_pattern_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v98_model"] = selected["name"]
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
        evidence["contrast_belief_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v98_model"] = selected["name"]
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
            f"[V98 BELIEF] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V98 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V98 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V98 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V98 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, base)))
        ba_delta.append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, base >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(base >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V98 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market/fold/time-block bootstrap over inner-locked symbolic belief policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V98/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    belief_contract = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["global_fitted_coefficient_n"] == 0
        and audit["outer_model_audit"]["learned_tree_edges"] == 0
        and audit["outer_model_audit"]["eligible_pattern_n"] >= 20
        and not audit["outer_model_audit"]["target_rows_used_for_cutpoints_or_pattern_counts"]
        and not audit["outer_model_audit"]["target_labels_used"]
        and not audit["outer_model_audit"]["nearest_event_retrieval"]
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
        "contrast_pattern_belief_contract_verified": belief_contract,
    }
    material_pass = bool(all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V98 fallback is not exact entire V69")
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
        "direction_change": {
            "probability_changed_n": int(np.sum(champion.prob.to_numpy(float) != diagnostic_original.prob.to_numpy(float))),
            "direction_changed_n": int(np.sum((champion.prob.to_numpy(float) >= 0.5) != (diagnostic_original.prob.to_numpy(float) >= 0.5))),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic_original.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected_frame.prob.to_numpy(float)),
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 frame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(not out.name.startswith("output_V98"), "runner must not write output_V98 directly")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
    }
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V98", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V98_CONTRAST_PATTERN_BELIEF_DIAGNOSTIC": evaluation["candidate_summary"],
                "V98_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority,
            "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V98", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V98", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V98_CONTRAST_PATTERN_BELIEF_REPORT.json": {
            "version": "V98", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "channels": PATTERN_CHANNELS,
            "adjacent_pairs": [(FEATURES[left], FEATURES[right]) for left, right in ADJACENT_PAIRS],
            "architectures": ARCHITECTURES, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V98_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V98_CONTRAST_PATTERN_BELIEF_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V98_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V98_CONTRAST_PATTERN_BELIEF_OUTER_OOF.csv.gz",
            "V98_FAIL_CLOSED_SELECTED_OOF.csv.gz",
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
            "explicit mode cannot accompany MARKET_BIO_VERSION_OUTPUT",
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
            "model_contract": {
                "features": FEATURES, "feature_n": len(FEATURES),
                "representation": "train-only tertile literals plus fixed adjacent-within-channel conjunctions",
                "estimator": "empirical-Bayes Beta contrast masses, conflict-normalized Dempster-Shafer combination, pignistic probability",
                "pattern_dictionary_n": PATTERN_DICTIONARY_SIZE,
                "global_fitted_coefficient_n": 0, "learned_tree_edges": 0,
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V97 actual root runners/drafts plus V97 fixed declaration",
                "contrast_pattern_or_dempster_shafer_found": False,
                "V90": "TAN generative dependency density; V98 uses no learned edge or joint density",
                "V96": "fixed projection halfspace depth; V98 uses symbolic train-quantile patterns and belief masses",
                "V97": "exact L1/Gower nearest-event analog vote; V98 retrieves no events and stores aggregate pattern counts only",
            },
            "nested_contract": {
                "strict_embargo_minutes": 35, "outer_labels_used_for_selection": False,
                "bootstrap_key": "selected.material_gate.nested_bootstrap",
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_market_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "reason": "bounded deterministic symbolic CPU smoke",
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            250 if args.smoke_test else BOOTSTRAP_DRAWS,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(
        non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]),
    )
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["contrast_pattern_belief_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    if not args.dry_run and args.full_run:
        summary.update(write_outputs(args.output, authority, evidence, audits, evaluation))
        summary["output_written"] = True
        summary["output"] = str(args.output)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
