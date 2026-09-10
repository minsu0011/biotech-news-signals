"""V148 Mondrian uniform-Hoeffding tail correctness lower-bound selector.

Atomic V69 direction probabilities are immutable.  Strict-past equal-event-
group V69 OOF correctness is ordered by frozen absolute V69 logit margin in
each source-family by predicted-side Mondrian cell.  At every observed tail
prefix, a one-sided Hoeffding bound with a fixed union correction over all
nested thresholds gives a simultaneous distribution-free lower confidence
bound for correctness.  Queries use the local cell when supported and the
global predicted-side tail otherwise.  The bound becomes the candidate
confidence signal and the candidate high-confidence mask is exactly bound
>= 0.5; probabilities and predicted directions never change.

This is not V130 Beta-binomial reliability or V135 weighted-PAV calibration:
there is no prior, posterior odds correction, binning, isotonic projection,
hierarchical shrinkage, fitted coefficient, or correction-strength grid.
One fixed surface competes with exact V69 noop using only an earlier inner
past.  Outer labels are evaluation-only under a strict 35-minute embargo and
event purge.  Failure restores the exact entire atomic V69 DataFrame.
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
from scipy.special import logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v135_source_side_monotone_confidence_reliability_rerank as contract


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V148_MONDRIAN_UNIFORM_HOEFFDING_TAIL_CORRECTNESS_LOWER_BOUND_SELECTOR_V1"
VERSION = 148
HYPOTHESIS = "MONDRIAN_UNIFORM_HOEFFDING_TAIL_CORRECTNESS_LOWER_BOUND_SELECTOR_V1"
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v135_source_side_monotone_confidence_reliability_rerank.py"
EXPECTED_CONTRACT_SHA256 = "49e298699fea19292808b4b469d66135cc04e26c8ae0d339a829617849af40a6"

DELTA = 0.10
MIN_CELL_N = 30
MIN_TAIL_N = 12
HIGH_CONFIDENCE_LOWER_BOUND = 0.50
PROBABILITY_EPSILON = 1e-6
COST = 0.002
SEED = 14801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "MONDRIAN_UNIFORM_HOEFFDING_TAIL_LCB_FULL"},)

require = contract.require
sha256 = contract.sha256
array_sha256 = contract.array_sha256
clean = contract.clean
bool_series = contract.bool_series
atomic_json = contract.atomic_json
atomic_csv = contract.atomic_csv
load_authorized = contract.load_authorized
aligned_dev = contract.aligned_dev
chronology = contract.chronology
metric = contract.metric
controller = contract.controller
v44 = contract.v44
EMBARGO = contract.EMBARGO
INNER_VALID_FRACTION = contract.INNER_VALID_FRACTION


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "strict-past_equal-event-group_atomic-v69-correctness|"
        "source-family_by_frozen-predicted-side_mondrian|absolute-v69-logit-margin|"
        "nested-greater-margin-tail-prefix|one-sided-hoeffding-union-lower-bound|"
        "delta0.10|min-cell30|min-tail12|global-side-fallback|no-bins|no-prior|"
        "no-PAV|no-fit-head|confidence-only|high-conf-bound-ge0.5|single-full-surface"
    )
    require(DELTA == 0.10 and MIN_CELL_N == 30 and MIN_TAIL_N == 12, "V148 fixed support contract changed")
    require(HIGH_CONFIDENCE_LOWER_BOUND == 0.50 and len(ARCHITECTURES) == 1, "V148 fixed candidate contract changed")
    return {
        "representation": "strict-past equal-event-group atomic V69 correctness ordered by frozen absolute V69 logit margin",
        "mondrian_cells": "source_family by frozen V69 predicted side",
        "lower_bound": "empirical nested-tail correctness minus sqrt(log(K/delta)/(2n)); simultaneous one-sided Hoeffding union bound across K cell thresholds",
        "fallback": "global predicted-side tail when local cell or local query tail lacks fixed support; zero lower bound if even global tail has insufficient support",
        "delta": DELTA, "minimum_cell_n": MIN_CELL_N, "minimum_tail_n": MIN_TAIL_N,
        "high_confidence_rule": "lower bound >= 0.5", "confidence_bins": 0,
        "beta_binomial_or_prior": False, "weighted_pav_or_isotonic": False,
        "fitted_coefficient_n": 0, "candidate_count_excluding_noop": 1,
        "probability_and_direction_structurally_frozen": True,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V135 contract scaffold changed")
    audit = contract.verify_authority()
    audit["code_dependency"] = {
        "path": CONTRACT_PATH.name, "sha256": EXPECTED_CONTRACT_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, metrics, canonical gate, atomic IO and controller-native robustness utilities only; no V135 output is read",
    }
    audit["v148_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = contract.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V148 bounded affinity changed")
    return audit


def source_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.source_family.fillna("__MISSING_SOURCE__").astype(str).to_numpy()


def collapse_reference(reference: pd.DataFrame) -> pd.DataFrame:
    require(len(reference) > 0, "V148 empty correctness reference")
    work = reference[["event_group_id", "event_time_utc", "market", "source_family", "y", "prob"]].copy()
    work["source_family"] = work.source_family.fillna("__MISSING_SOURCE__").astype(str)
    work["predicted_side"] = (work.prob.to_numpy(float) >= 0.5).astype(int)
    work["margin"] = np.abs(logit(np.clip(work.prob.to_numpy(float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)))
    grouped = work.groupby("event_group_id", sort=True)
    for field in ("y", "predicted_side", "source_family"):
        require(int(grouped[field].nunique().max()) == 1, f"V148 event group crosses {field}")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "source_family": "first",
        "y": "first", "predicted_side": "first", "margin": "mean",
    }).reset_index()
    result["correct"] = (result.y.to_numpy(int) == result.predicted_side.to_numpy(int)).astype(float)
    require(np.isfinite(result.margin).all() and np.isfinite(result.correct).all(), "V148 collapsed values invalid")
    return result


def make_tail_table(margin: np.ndarray, correct: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    margin = np.asarray(margin, float)
    correct = np.asarray(correct, float)
    weight = np.asarray(weight, float)
    require(len(margin) > 0 and margin.shape == correct.shape == weight.shape, "V148 tail table shape invalid")
    require(np.isfinite(margin).all() and np.isfinite(correct).all() and np.isfinite(weight).all() and np.all(weight > 0), "V148 tail table input invalid")
    order = np.argsort(-margin, kind="stable")
    ordered_margin = margin[order]
    ordered_correct = correct[order]
    ordered_weight = weight[order]
    cumulative_weight = np.cumsum(ordered_weight)
    cumulative_success = np.cumsum(ordered_weight * ordered_correct)
    support = np.arange(1, len(order) + 1, dtype=float)
    radius = np.sqrt(np.log(max(len(order), 2) / DELTA) / (2.0 * support))
    lower = np.clip(cumulative_success / cumulative_weight - radius, 0.0, 1.0)
    return {
        "margin_descending": ordered_margin, "lower": lower,
        "support": support.astype(int), "weighted_support": cumulative_weight,
        "event_group_n": len(order), "threshold_n": len(order),
    }


def fit_tail_surfaces(groups: pd.DataFrame, weights: np.ndarray | None = None) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    if weights is None:
        weights = np.ones(len(groups), float)
    weights = np.asarray(weights, float)
    require(weights.shape == (len(groups),) and np.isfinite(weights).all() and np.all(weights > 0), "V148 fit weights invalid")
    side = groups.predicted_side.to_numpy(int)
    source = source_values(groups)
    margin = groups.margin.to_numpy(float)
    correct = groups.correct.to_numpy(float)
    global_tables: dict[int, dict[str, Any]] = {}
    local_tables: dict[tuple[str, int], dict[str, Any]] = {}
    for direction in (0, 1):
        mask = side == direction
        require(int(mask.sum()) >= MIN_CELL_N, f"V148 global side {direction} lacks support")
        global_tables[direction] = make_tail_table(margin[mask], correct[mask], weights[mask])
        for source_name in sorted(set(source[mask])):
            cell = mask & (source == source_name)
            if int(cell.sum()) >= MIN_CELL_N:
                local_tables[(source_name, direction)] = make_tail_table(margin[cell], correct[cell], weights[cell])
    return global_tables, local_tables, {
        "event_group_n": len(groups), "global_side_table_n": len(global_tables),
        "supported_local_cell_n": len(local_tables), "delta": DELTA,
        "minimum_cell_n": MIN_CELL_N, "minimum_tail_n": MIN_TAIL_N,
        "confidence_bins": 0, "uniform_hoeffding_union": True,
        "beta_binomial_or_prior": False, "weighted_pav_or_isotonic": False,
    }


def table_lower(table: dict[str, Any], query_margin: float) -> tuple[float, int]:
    ordered = np.asarray(table["margin_descending"], float)
    tail_n = int(np.searchsorted(-ordered, -float(query_margin), side="right"))
    if tail_n < MIN_TAIL_N:
        return 0.0, tail_n
    return float(np.asarray(table["lower"], float)[tail_n - 1]), tail_n


def predict_lower_bounds(target: pd.DataFrame, global_tables: dict[int, dict[str, Any]], local_tables: dict[tuple[str, int], dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    probability = target.prob.to_numpy(float)
    side = (probability >= 0.5).astype(int)
    margin = np.abs(logit(np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)))
    source = source_values(target)
    lower = np.zeros(len(target), float)
    route = np.empty(len(target), dtype=object)
    tail_support = np.zeros(len(target), int)
    for index, (source_name, direction, query_margin) in enumerate(zip(source, side, margin)):
        local = local_tables.get((source_name, int(direction)))
        if local is not None:
            value, support = table_lower(local, float(query_margin))
            if support >= MIN_TAIL_N:
                lower[index], tail_support[index], route[index] = value, support, "SOURCE_SIDE"
                continue
        value, support = table_lower(global_tables[int(direction)], float(query_margin))
        lower[index], tail_support[index], route[index] = value, support, "GLOBAL_SIDE" if support >= MIN_TAIL_N else "UNSUPPORTED_ZERO"
    require(np.isfinite(lower).all() and np.all((lower >= 0.0) & (lower <= 1.0)), "V148 lower bound invalid")
    counts = {str(key): int(np.sum(route == key)) for key in sorted(set(route))}
    return lower, {
        "route_counts": counts, "tail_support_min": int(tail_support.min()),
        "tail_support_median": float(np.median(tail_support)), "tail_support_max": int(tail_support.max()),
        "lower_bound_min": float(lower.min()), "lower_bound_median": float(np.median(lower)), "lower_bound_max": float(lower.max()),
        "lower_bound_sha256": array_sha256(lower),
    }


def native_tail_bootstrap(groups: pd.DataFrame, target: pd.DataFrame, nominal: np.ndarray, draws: int, seed: int) -> dict[str, Any] | None:
    if draws <= 0:
        return None
    generator = np.random.default_rng(seed)
    predictions = np.empty((len(target), draws), float)
    for draw in range(draws):
        weights = generator.exponential(1.0, len(groups))
        global_tables, local_tables, _ = fit_tail_surfaces(groups, weights)
        predictions[:, draw], _ = predict_lower_bounds(target, global_tables, local_tables)
    median = np.median(predictions, axis=1)
    nominal_high = np.asarray(nominal, float) >= HIGH_CONFIDENCE_LOWER_BOUND
    bootstrap_high = predictions >= HIGH_CONFIDENCE_LOWER_BOUND
    return {
        "method": "equal-event-group Bayesian exponential-weight refit of Mondrian correctness tails; fixed Hoeffding radius and routing",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "target_labels_used": False, "direction_probability_changed": False,
        "median_absolute_confidence_deviation": float(np.median(np.abs(predictions - np.asarray(nominal, float)[:, None]))),
        "median_confidence_sha256": array_sha256(median),
        "mean_high_confidence_agreement": float(np.mean(bootstrap_high == nominal_high[:, None])),
        "draw_inventory_sha256": array_sha256(predictions),
    }


def mondrian_lower_confidence(reference: pd.DataFrame, target: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = collapse_reference(reference)
    global_tables, local_tables, fit_audit = fit_tail_surfaces(groups)
    lower, prediction_audit = predict_lower_bounds(target, global_tables, local_tables)
    high = lower >= HIGH_CONFIDENCE_LOWER_BOUND
    probability = target.prob.to_numpy(float)
    native = native_tail_bootstrap(groups, target, lower, native_draws, native_seed)
    return lower, high, {
        "architecture": "source-side Mondrian nested-margin correctness tail with simultaneous one-sided Hoeffding-union lower bound",
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_only": True,
        "reference_event_group_n": len(groups), "fit_audit": fit_audit,
        "prediction_audit": prediction_audit, "native_tail_bootstrap": native,
        "probability_sha256": array_sha256(probability), "confidence_sha256": array_sha256(lower),
        "high_confidence_sha256": array_sha256(high), "predicted_direction_exact_v69": True,
        "probability_exact_v69": True, "fitted_causal_state_coefficient_n": 0,
        "target_rows_used_for_fit_or_selection": False, "target_labels_used": False,
        "confidence_bins": 0, "beta_binomial_or_prior": False,
        "weighted_pav_or_isotonic": False, "pair_list_auc_loss": False,
    }


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference = champion.loc[champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference) >= 120, f"V148 {label} reference too small")
    reference_valid = aligned_dev(dev, reference)
    audit = chronology(reference_valid, target_valid, f"V148 {label}")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF only"
    audit["target_labels_used_for_fit"] = False
    return reference, audit


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} V69 OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_reference, audit = strict_reference(dev, champion, market, inner_valid, f"inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_valid, inner_champion, inner_reference, audit


def confidence_score(values: dict[str, Any]) -> float:
    confidence_auc = values["confidence_correctness_auc"]
    high_accuracy = values["highconf_accuracy"]
    high_net = values["highconf_mean_signed_net"]
    if confidence_auc is None or high_accuracy is None or high_net is None:
        return -1e9
    return float(confidence_auc + 0.25 * high_accuracy + 5.0 * high_net)


def choose_inner(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    probability = inner_champion.prob.to_numpy(float)
    baseline_confidence = inner_champion.confidence_signal.to_numpy(float)
    baseline_high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, probability, baseline_confidence, baseline_high)
    lower, high, model_audit = mondrian_lower_confidence(inner_reference, inner_champion)
    values = metric(inner_valid, probability, lower, high)
    minimum_high_n = max(12, int(math.ceil(0.05 * len(inner_valid))))
    eligible = bool(
        values["highconf_n"] >= minimum_high_n
        and values["highconf_accuracy"] is not None
        and values["highconf_mean_signed_net"] is not None
        and (baseline_metric["highconf_accuracy"] is None or values["highconf_accuracy"] >= baseline_metric["highconf_accuracy"] - 0.01)
        and (baseline_metric["highconf_mean_signed_net"] is None or values["highconf_mean_signed_net"] >= baseline_metric["highconf_mean_signed_net"] - 0.0005)
    )
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "confidence_auc_delta": 0.0, "highconf_accuracy_delta": 0.0,
        "highconf_net_delta": 0.0, "eligible": True, "score": confidence_score(baseline_metric),
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": values,
        "confidence_auc_delta": None if values["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else values["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"],
        "highconf_accuracy_delta": None if values["highconf_accuracy"] is None or baseline_metric["highconf_accuracy"] is None else values["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
        "highconf_net_delta": None if values["highconf_mean_signed_net"] is None or baseline_metric["highconf_mean_signed_net"] is None else values["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
        "eligible": eligible, "minimum_highconf_n": minimum_high_n, "score": confidence_score(values),
    }]
    selected = max((trial for trial in trials if trial["eligible"]), key=lambda trial: (trial["score"], trial["confidence_auc_delta"] or 0.0, trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past OOF only; exact V69 confidence/high_conf versus one fixed Mondrian Hoeffding-tail lower-bound selector; maximize confidence-correctness AUC + 0.25*HC accuracy + 5*HC net after fixed HC coverage/accuracy/net safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False, "threshold_or_strength_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v148_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    folds = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([("US", 2)] if smoke else folds):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, f"outer {market} fold {fold}")
        inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_valid, inner_champion, inner_reference)
        lower, high, outer_model_audit = mondrian_lower_confidence(
            outer_reference, outer_champion,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        probability = outer_champion.prob.to_numpy(float)
        baseline_confidence = outer_champion.confidence_signal.to_numpy(float)
        baseline_high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected_lower = baseline_confidence if policy["selected"]["architecture"] is None else lower
        selected_high = baseline_high if policy["selected"]["architecture"] is None else high
        baseline_metric = metric(outer_valid, probability, baseline_confidence, baseline_high)
        surface_metric = metric(outer_valid, probability, lower, high)
        candidate_metric = metric(outer_valid, probability, selected_lower, selected_high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        confidence_lookup = dict(zip(outer_valid.event_id, selected_lower))
        high_lookup = dict(zip(outer_valid.event_id, selected_high))
        diagnostic.loc[positions, "confidence_signal"] = diagnostic.loc[positions, "event_id"].map(confidence_lookup)
        diagnostic.loc[positions, "high_conf"] = diagnostic.loc[positions, "event_id"].map(high_lookup).astype(bool)
        diagnostic.loc[positions, "v148_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m"]].copy()
        evidence["baseline_prob"] = probability
        evidence["candidate_prob"] = probability
        evidence["baseline_confidence"] = baseline_confidence
        evidence["candidate_confidence"] = selected_lower
        evidence["surface_confidence"] = lower
        evidence["baseline_high"] = baseline_high
        evidence["candidate_high"] = selected_high
        evidence["surface_high"] = high
        evidence["v148_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": {ARCHITECTURES[0]["name"]: surface_metric},
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_confidence_auc_delta": None if candidate_metric["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else candidate_metric["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"],
            "outer_highconf_accuracy_delta": None if candidate_metric["highconf_accuracy"] is None or baseline_metric["highconf_accuracy"] is None else candidate_metric["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
            "outer_highconf_net_delta": None if candidate_metric["highconf_mean_signed_net"] is None or baseline_metric["highconf_mean_signed_net"] is None else candidate_metric["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
            "outer_probability_exact_v69": True, "outer_direction_exact_v69": True,
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V148 MONDRIAN LCB] market={market} fold={fold} selected={policy['selected']['name']} inner_conf_score_delta={policy['selected']['score'] - confidence_score(policy['baseline']):+.6f} outer_conf_auc_delta={audits[-1]['outer_confidence_auc_delta'] or 0.0:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V148 evidence repeats event")
    require(np.array_equal(champion.prob.to_numpy(float), diagnostic.prob.to_numpy(float)), "V148 changed V69 probability")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V148 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    keys = ("auc_delta", "balanced_accuracy_delta", "all_trade_net_delta", "confidence_auc_delta", "highconf_accuracy_delta", "highconf_net_delta")
    values: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline_prob = sample.baseline_prob.to_numpy(float)
        candidate_prob = sample.candidate_prob.to_numpy(float)
        baseline_confidence = sample.baseline_confidence.to_numpy(float)
        candidate_confidence = sample.candidate_confidence.to_numpy(float)
        baseline_high = bool_series(sample.baseline_high).to_numpy(bool)
        candidate_high = bool_series(sample.candidate_high).to_numpy(bool)
        correct = (baseline_prob >= 0.5) == target
        signed = np.where(baseline_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float) - COST
        values["auc_delta"].append(float(roc_auc_score(target, candidate_prob) - roc_auc_score(target, baseline_prob)))
        values["balanced_accuracy_delta"].append(float(balanced_accuracy_score(target, candidate_prob >= 0.5) - balanced_accuracy_score(target, baseline_prob >= 0.5)))
        values["all_trade_net_delta"].append(float(np.mean(np.where(candidate_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float)) - np.mean(np.where(baseline_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float))))
        if np.unique(correct).size == 2 and baseline_high.any() and candidate_high.any():
            values["confidence_auc_delta"].append(float(roc_auc_score(correct, candidate_confidence) - roc_auc_score(correct, baseline_confidence)))
            values["highconf_accuracy_delta"].append(float(correct[candidate_high].mean() - correct[baseline_high].mean()))
            values["highconf_net_delta"].append(float(signed[candidate_high].mean() - signed[baseline_high].mean()))
    require(len(values["auc_delta"]) >= int(0.9 * draws), "V148 bootstrap lost standard draws")
    require(len(values["confidence_auc_delta"]) >= int(0.8 * draws), "V148 bootstrap lost confidence draws")
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V148 confidence policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V148 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V148 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.baseline_confidence, bool_series(evidence.baseline_high))
    candidate = metric(evidence, evidence.candidate_prob, evidence.candidate_confidence, bool_series(evidence.candidate_high))
    require(candidate["balanced_accuracy"] == base["balanced_accuracy"] and candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"], "V148 evaluated direction changed")
    nested = paired_nested_bootstrap(evidence, draws)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    probability_exact = bool(np.array_equal(champion.prob.to_numpy(float), diagnostic_original.prob.to_numpy(float)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    lcb_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_only"]
        and item["predicted_direction_exact_v69"] and item["probability_exact_v69"]
        and item["fit_audit"]["uniform_hoeffding_union"] and item["fit_audit"]["confidence_bins"] == 0
        and not item["fit_audit"]["beta_binomial_or_prior"] and not item["fit_audit"]["weighted_pav_or_isotonic"]
        and item["reference_event_group_n"] >= 120 and not item["target_rows_used_for_fit_or_selection"]
        and not item["target_labels_used"] and item["fitted_causal_state_coefficient_n"] == 0
        for item in full_models
    )
    model_native = [item["native_tail_bootstrap"] for item in full_models]
    model_native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and not item["target_labels_used"] and not item["direction_probability_changed"]
        for item in model_native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    confidence_auc_delta = None if candidate["confidence_correctness_auc"] is None or base["confidence_correctness_auc"] is None else candidate["confidence_correctness_auc"] - base["confidence_correctness_auc"]
    high_accuracy_delta = None if candidate["highconf_accuracy"] is None or base["highconf_accuracy"] is None else candidate["highconf_accuracy"] - base["highconf_accuracy"]
    high_net_delta = None if candidate["highconf_mean_signed_net"] is None or base["highconf_mean_signed_net"] is None else candidate["highconf_mean_signed_net"] - base["highconf_mean_signed_net"]
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_probability_and_direction_exact_v69": probability_exact and all(audit["outer_direction_exact_v69"] for audit in audits),
        "outer_balanced_accuracy_exact_v69": candidate["balanced_accuracy"] == base["balanced_accuracy"],
        "outer_all_trade_net_exact_v69": candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"],
        "outer_confidence_correctness_auc_delta_gt_0_01": confidence_auc_delta is not None and confidence_auc_delta > 0.01,
        "outer_highconf_accuracy_delta_ge_0": high_accuracy_delta is not None and high_accuracy_delta >= 0.0,
        "outer_highconf_net_delta_ge_0": high_net_delta is not None and high_net_delta >= 0.0,
        "candidate_highconf_coverage_ge_0_05": candidate["highconf_n"] >= int(math.ceil(0.05 * candidate["n"])),
        "nested_bootstrap_confidence_auc_probability_gt_zero_ge_0_75": nested["confidence_auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_highconf_accuracy_probability_gt_zero_ge_0_65": nested["highconf_accuracy_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_highconf_net_probability_gt_zero_ge_0_65": nested["highconf_net_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_ba_delta_exact_zero": all(nested["balanced_accuracy_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "nested_bootstrap_net_delta_exact_zero": all(nested["all_trade_net_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "full_overall_ba_exact_v69": candidate_metrics["balanced_accuracy"] == base_metrics["balanced_accuracy"],
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_tail_bootstrap_verified": model_native_valid,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "mondrian_uniform_hoeffding_tail_contract_verified": lcb_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V148 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": 0.0, "outer_ba_delta": 0.0, "outer_net_delta": 0.0,
        "outer_confidence_auc_delta": confidence_auc_delta,
        "outer_highconf_accuracy_delta": high_accuracy_delta, "outer_highconf_net_delta": high_net_delta,
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_tail_bootstrap": model_native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "immutability": {"probability_exact": probability_exact, "predicted_direction_exact": all(audit["outer_direction_exact_v69"] for audit in audits)},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_tail_bootstrap": evaluation["model_native_tail_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V148 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V148 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V148 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V148").resolve(), "V148 output outside controller staging")
    require(out.exists() and out.is_dir(), "V148 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V148 controller prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V148 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V148 selected native robustness missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V148", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V148_DIAGNOSTIC_MONDRIAN_TAIL_LCB": evaluation["candidate_summary"], "V148_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V148", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V148", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V148_MONDRIAN_TAIL_CORRECTNESS_LOWER_BOUND_REPORT.json": {"version": "V148", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V148", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V148_MONDRIAN_TAIL_LCB_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V148_DIAGNOSTIC_MONDRIAN_TAIL_LCB_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V148_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    inner_lower, inner_high, inner_audit = mondrian_lower_confidence(inner_reference, inner_champion)
    outer_lower, outer_high, outer_audit = mondrian_lower_confidence(outer_reference, outer_champion)
    probability_inner = inner_champion.prob.to_numpy(float)
    probability_outer = outer_champion.prob.to_numpy(float)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold,
        "inner_baseline": metric(inner_valid, probability_inner, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_surface": metric(inner_valid, probability_inner, inner_lower, inner_high),
        "outer_baseline": metric(outer_valid, probability_outer, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_surface_evaluation_only": metric(outer_valid, probability_outer, outer_lower, outer_high),
        "inner_model_audit": inner_audit, "outer_model_audit": outer_audit,
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used_for_fit_or_selection": False, "probability_exact_v69": True, "output_written": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe-kr2", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V148 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V148 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
    affinity = preparation_affinity() if bounded else {"reason": "controller-authorized full run is not bounded preparation"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC,
            "architecture": "strict-past source-side nested-margin correctness tails with simultaneous one-sided Hoeffding-union lower bound; exact V69 direction",
            "causal_pre_event_only": True, "threshold_or_strength_tuned": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run",
                "output_bound_to_environment": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True,
                "model_native_tail_bootstrap": True, "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            }, "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "output_written": False,
        }), indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe(dev, champion)
            result["preparation_affinity"] = affinity
            result["threadpools"] = threadpool_info()
            result["gpu_model_calls"] = 0
            print(json.dumps(clean(result), indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, args.smoke_test)
    if args.smoke_test:
        print(json.dumps(clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS, "preparation_affinity": affinity,
            "threadpools": threadpool_info(), "gpu_model_calls": 0, "folds": len(audits), "audits": audits,
            "outer_baseline": evaluation["outer_baseline"], "outer_candidate": evaluation["outer_candidate"],
            "outer_confidence_auc_delta": evaluation["outer_confidence_auc_delta"],
            "outer_highconf_accuracy_delta": evaluation["outer_highconf_accuracy_delta"],
            "outer_highconf_net_delta": evaluation["outer_highconf_net_delta"],
            "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"],
            "fallback": evaluation["fallback"], "output_written": False,
        }), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
