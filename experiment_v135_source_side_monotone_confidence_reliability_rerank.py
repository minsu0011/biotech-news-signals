"""V135 source-side monotone confidence-reliability decision-cell reranker.

Only strict-past atomic V69 OOF event groups are used.  Frozen V69 correctness is
modelled as a no-bin monotone function of frozen V69 confidence separately for
DOWN and UP.  Exact weighted pool-adjacent-violators (PAV) fits a global-side
curve and each observed source-family by side curve; the local curve is shrunk
pointwise to the global curve with one fixed support/(support+32) hierarchy.
The resulting correctness reliability orders scores only inside the original
V69 decision cell, so direction, balanced accuracy and all-trade net are exact.

This is richer than V130's one constant per source by side: it estimates an
entire shape-constrained confidence/correctness surface without bins or a
correction-strength grid.  It is not Platt probability calibration, conformal
typicality, return salience, a causal-state head, or a pairwise AUC objective.
Outer labels are evaluation-only under a 35-minute embargo/event purge.  V69
confidence/high-confidence stay exact and material failure restores entire V69.
Bounded preparation uses CPU30-31, two numeric threads, and zero GPU calls.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v130_source_family_decision_cell_beta_binomial_rerank as contract


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V135_SOURCE_SIDE_MONOTONE_CONFIDENCE_RELIABILITY_RERANK_V1"
VERSION = 135
HYPOTHESIS = "SOURCE_SIDE_MONOTONE_CONFIDENCE_RELIABILITY_RERANK_V1"
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v130_source_family_decision_cell_beta_binomial_rerank.py"
EXPECTED_CONTRACT_SHA256 = "ff32dc13bd6586f52b94a5a850ca7bfe0e716e0d00e515cd44186bfa28aaeb40"

HIERARCHY_STRENGTH = 32.0
CELL_SPAN = 0.499
RELIABILITY_EPSILON = 1e-4
SEED = 13501
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "MONOTONE_CONFIDENCE_RELIABILITY_SURFACE_FULL"},)

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
EMBARGO = contract.contract.scaffold.EMBARGO
INNER_VALID_FRACTION = contract.contract.scaffold.INNER_VALID_FRACTION


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "strict-past_equal-event-group|frozen-v69-side|frozen-confidence-signal|"
        "global-side_exact-weighted-PAV|source-side_exact-weighted-PAV|"
        "support-over-support-plus-32-hierarchical-shrink|no-bins|"
        "direct-correctness-order-inside-original-decision-cell|single-full-surface"
    )
    require(HIERARCHY_STRENGTH == 32.0 and CELL_SPAN == 0.499, "V135 static constants changed")
    require(len(ARCHITECTURES) == 1, "V135 correction-strength grid forbidden")
    return {
        "representation": "strict-past equal-event-group V69 correctness by frozen source family, predicted side and confidence_signal",
        "shape_constraint": "nondecreasing correctness reliability in frozen V69 confidence via exact weighted PAV",
        "hierarchy": "source-side curve shrunk pointwise to global-side curve by support/(support+32)",
        "confidence_bins": 0,
        "correction_strength_grid": False,
        "surface_candidates_excluding_noop": 1,
        "decision_cell_span": CELL_SPAN,
        "predicted_direction_structurally_frozen": True,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V130 contract scaffold changed")
    audit = contract.verify_authority()
    audit["code_dependency"] = {
        "path": CONTRACT_PATH.name, "sha256": EXPECTED_CONTRACT_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, metrics, canonical gate, bootstrap and atomic IO contract only; no V130 output is read",
    }
    audit["v135_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = contract.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V135 bounded affinity changed")
    return audit


def source_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.source_family.fillna("__MISSING_SOURCE__").astype(str).to_numpy()


def collapse_reference(reference: pd.DataFrame) -> pd.DataFrame:
    require(len(reference) > 0, "V135 empty reliability reference")
    work = reference[["event_group_id", "event_time_utc", "market", "source_family", "y", "prob", "confidence_signal"]].copy()
    work["source_family"] = work.source_family.fillna("__MISSING_SOURCE__").astype(str)
    work["predicted_side"] = (work.prob.to_numpy(float) >= 0.5).astype(int)
    grouped = work.groupby("event_group_id", sort=True)
    for field in ("y", "predicted_side", "source_family"):
        require(int(grouped[field].nunique().max()) == 1, f"V135 event group crosses {field}")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "source_family": "first",
        "y": "first", "predicted_side": "first", "confidence_signal": "mean",
    }).reset_index()
    result["correct"] = (result.y.to_numpy(int) == result.predicted_side.to_numpy(int)).astype(float)
    require(np.isfinite(result.confidence_signal).all() and np.isfinite(result.correct).all(), "V135 collapsed reliability values invalid")
    return result


def fit_pav(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    weight = np.asarray(weight, float)
    require(len(x) > 0 and x.shape == y.shape == weight.shape, "V135 PAV input shape invalid")
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(weight).all() and np.all(weight > 0), "V135 PAV input invalid")
    order = np.argsort(x, kind="stable")
    x, y, weight = x[order], y[order], weight[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    unique_weight = np.bincount(inverse, weights=weight)
    unique_success = np.bincount(inverse, weights=weight * y)
    blocks: list[list[float]] = []
    for value, support, success in zip(unique_x, unique_weight, unique_success):
        blocks.append([float(support), float(success), float(support * value)])
        while len(blocks) >= 2 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0] + 1e-15:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[index] + right[index] for index in range(3)])
    knot_x = np.asarray([block[2] / block[0] for block in blocks], float)
    knot_y = np.asarray([block[1] / block[0] for block in blocks], float)
    knot_weight = np.asarray([block[0] for block in blocks], float)
    knot_y = np.clip(knot_y, RELIABILITY_EPSILON, 1.0 - RELIABILITY_EPSILON)
    require(np.all(np.diff(knot_x) > 0) and np.all(np.diff(knot_y) >= -1e-15), "V135 PAV monotonicity failed")
    return {"x": knot_x, "y": knot_y, "weight": knot_weight}


def predict_curve(curve: dict[str, Any], confidence: np.ndarray) -> np.ndarray:
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["y"], float)
    query = np.asarray(confidence, float)
    if len(x) == 1:
        result = np.full(len(query), y[0], dtype=float)
    else:
        result = np.interp(query, x, y, left=y[0], right=y[-1])
    return np.clip(result, RELIABILITY_EPSILON, 1.0 - RELIABILITY_EPSILON)


def curve_audit(curve: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_n": len(curve["x"]), "effective_support": float(np.sum(curve["weight"])),
        "confidence_min": float(curve["x"][0]), "confidence_max": float(curve["x"][-1]),
        "reliability_min": float(np.min(curve["y"])), "reliability_max": float(np.max(curve["y"])),
        "monotone_non_decreasing": bool(np.all(np.diff(curve["y"]) >= -1e-15)),
        "knots_sha256": array_sha256(np.column_stack([curve["x"], curve["y"], curve["weight"]])),
    }


def fit_surface(groups: pd.DataFrame, weights: np.ndarray | None = None) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    require(len(groups) >= 120, "V135 surface reference support too small")
    weights = np.ones(len(groups), float) if weights is None else np.asarray(weights, float)
    require(weights.shape == (len(groups),) and np.all(weights > 0) and np.isfinite(weights).all(), "V135 surface weights invalid")
    side = groups.predicted_side.to_numpy(int)
    source = groups.source_family.to_numpy(str)
    confidence = groups.confidence_signal.to_numpy(float)
    correct = groups.correct.to_numpy(float)
    global_curves: dict[int, dict[str, Any]] = {}
    local_curves: dict[tuple[str, int], dict[str, Any]] = {}
    global_audits: dict[str, Any] = {}
    local_audits: list[dict[str, Any]] = []
    for direction in (0, 1):
        mask = side == direction
        require(int(mask.sum()) >= 40, f"V135 predicted-side {direction} support too small")
        global_curves[direction] = fit_pav(confidence[mask], correct[mask], weights[mask])
        global_audits[str(direction)] = curve_audit(global_curves[direction])
        for family in sorted(set(source[mask])):
            cell = mask & (source == family)
            curve = fit_pav(confidence[cell], correct[cell], weights[cell])
            local_curves[(str(family), direction)] = curve
            local_audits.append({"source_family": str(family), "predicted_side": direction, **curve_audit(curve)})
    require(all(item["monotone_non_decreasing"] for item in local_audits), "V135 local PAV monotonicity failed")
    canonical = [{key: item[key] for key in ("source_family", "predicted_side", "block_n", "effective_support", "knots_sha256")} for item in local_audits]
    return global_curves, local_curves, {
        "event_group_n": len(groups), "source_family_n": int(groups.source_family.nunique()),
        "source_side_curve_n": len(local_audits), "global_side_curves": global_audits,
        "source_side_curves": local_audits,
        "surface_table_sha256": hashlib.sha256(json.dumps(clean(canonical), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "confidence_bins": 0, "exact_weighted_pav": True,
    }


def surface_probability(target: pd.DataFrame, global_curves: dict[int, dict[str, Any]], local_curves: dict[tuple[str, int], dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = np.asarray(target.prob, float)
    side = (baseline >= 0.5).astype(int)
    source = source_values(target)
    confidence = target.confidence_signal.to_numpy(float)
    reliability = np.empty(len(target), float)
    local_weight = np.empty(len(target), float)
    unseen = np.zeros(len(target), bool)
    for direction in (0, 1):
        side_mask = side == direction
        global_prediction = predict_curve(global_curves[direction], confidence[side_mask])
        positions = np.flatnonzero(side_mask)
        for local_position, row_position in enumerate(positions):
            key = (str(source[row_position]), direction)
            curve = local_curves.get(key)
            if curve is None:
                reliability[row_position] = global_prediction[local_position]
                local_weight[row_position] = 0.0
                unseen[row_position] = True
            else:
                local_prediction = predict_curve(curve, confidence[row_position:row_position + 1])[0]
                support = float(np.sum(curve["weight"]))
                alpha = support / (support + HIERARCHY_STRENGTH)
                reliability[row_position] = alpha * local_prediction + (1.0 - alpha) * global_prediction[local_position]
                local_weight[row_position] = alpha
    reliability = np.clip(reliability, RELIABILITY_EPSILON, 1.0 - RELIABILITY_EPSILON)
    probability = np.where(side == 1, 0.5 + CELL_SPAN * reliability, 0.5 - CELL_SPAN * reliability)
    require(np.array_equal(probability >= 0.5, side.astype(bool)), "V135 decision-cell preservation failed")
    return probability, {
        "reliability_q10_q50_q90": [float(np.quantile(reliability, q)) for q in (0.1, 0.5, 0.9)],
        "local_hierarchy_weight_q10_q50_q90": [float(np.quantile(local_weight, q)) for q in (0.1, 0.5, 0.9)],
        "unseen_source_side_n": int(unseen.sum()), "confidence_min": float(confidence.min()), "confidence_max": float(confidence.max()),
        "predicted_direction_exact_v69": True, "probability_sha256": array_sha256(probability),
    }


def native_surface_bootstrap(groups: pd.DataFrame, target: pd.DataFrame, nominal: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    samples = np.empty((len(target), draws), float)
    for draw in range(draws):
        weight = generator.exponential(1.0, len(groups))
        global_curves, local_curves, _ = fit_surface(groups, weight)
        samples[:, draw], _ = surface_probability(target, global_curves, local_curves)
    median = np.median(samples, axis=1)
    std = np.std(samples, axis=1)
    return {
        "contract": "equal-event-group Bayesian bootstrap complete refit of global-side and source-side weighted PAV reliability surfaces",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "target_labels_used": False, "confidence_bins": 0,
        "direction_agreement_with_nominal": float(np.mean((samples >= 0.5) == (nominal[:, None] >= 0.5))),
        "probability_mean_absolute_deviation": float(np.mean(np.abs(samples - nominal[:, None]))),
        "row_probability_std_q10_q50_q90": [float(np.quantile(std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median),
    }


def monotone_reliability_direction(reference: pd.DataFrame, target: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, dict[str, Any]]:
    require(reference.market.nunique() == target.market.nunique() == 1 and str(reference.market.iloc[0]) == str(target.market.iloc[0]), "V135 market mismatch")
    groups = collapse_reference(reference)
    global_curves, local_curves, fit_audit = fit_surface(groups)
    probability, surface_audit = surface_probability(target, global_curves, local_curves)
    baseline = target.prob.to_numpy(float)
    native = None if native_draws <= 0 else native_surface_bootstrap(groups, target, probability, native_draws, native_seed)
    return probability, {
        "market": str(reference.market.iloc[0]), "reference_row_n": len(reference),
        "reference_event_group_n": len(groups), "target_n": len(target),
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_only": True,
        "source_family_feature_n": 1, "frozen_v69_side_feature_n": 1, "frozen_v69_confidence_feature_n": 1,
        "causal_numeric_text_ticker_issuer_or_return_feature_n": 0,
        "architecture": "no-bin exact weighted PAV global-side and hierarchical source-side confidence-correctness surface constrained to original V69 decision cells",
        "static_spec": STATIC_SPEC, "fit_audit": fit_audit, "surface_audit": surface_audit,
        "predicted_direction_exact_v69": bool(np.array_equal(probability >= 0.5, baseline >= 0.5)),
        "target_rows_used_for_surface_fit_or_selection": False, "target_labels_used": False,
        "fitted_causal_state_coefficient_n": 0, "pair_list_auc_loss": False,
        "platt_conformal_or_constant_beta_binomial": False,
        "native_surface_bootstrap": native,
        "probability_q10_q50_q90": [float(np.quantile(probability, q)) for q in (0.1, 0.5, 0.9)],
        "probability_sha256": array_sha256(probability),
    }


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference = champion.loc[champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference) >= 120, f"V135 {label} reliability reference too small")
    reference_valid = aligned_dev(dev, reference)
    audit = chronology(reference_valid, target_valid, f"V135 {label} reliability")
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


def choose_inner(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    probability, model_audit = monotone_reliability_direction(inner_reference, inner_champion)
    values = metric(inner_valid, probability, confidence, high)
    require(values["balanced_accuracy"] == baseline_metric["balanced_accuracy"] and values["all_trade_mean_signed_net"] == baseline_metric["all_trade_mean_signed_net"], "V135 inner decisions changed")
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True, "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]),
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": values,
        "auc_delta": values["auc"] - baseline_metric["auc"], "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": bool(model_audit["predicted_direction_exact_v69"]),
        "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
    }]
    selected = max(trials, key=lambda trial: (trial["score"], trial["auc_delta"], trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past OOF only; exact V69 versus one fully predeclared hierarchical no-bin monotone reliability surface; maximize 2*AUC+BA with structurally exact V69 decisions",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False, "correction_strength_or_bin_count_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v135_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    folds = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else folds):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, f"outer {market} fold {fold}")
        inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_valid, inner_champion, inner_reference)
        surface, outer_model_audit = monotone_reliability_direction(
            outer_reference, outer_champion,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        baseline = outer_champion.prob.to_numpy(float)
        candidate = baseline.copy() if policy["selected"]["architecture"] is None else surface
        require(np.array_equal(candidate >= 0.5, baseline >= 0.5), "V135 outer direction changed")
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        surface_metric = metric(outer_valid, surface, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        require(candidate_metric["balanced_accuracy"] == baseline_metric["balanced_accuracy"] and candidate_metric["all_trade_mean_signed_net"] == baseline_metric["all_trade_mean_signed_net"], "V135 outer decisions changed")
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        lookup = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(lookup)
        diagnostic.loc[positions, "v135_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf"]].copy()
        evidence["baseline_prob"] = baseline
        evidence["surface_prob"] = surface
        evidence["candidate_prob"] = candidate
        evidence["v135_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": {ARCHITECTURES[0]["name"]: surface_metric},
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": 0.0, "outer_net_delta": 0.0,
            "outer_direction_exact_v69": True, "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(f"[V135 MONOTONE SURFACE] market={market} fold={fold} selected={policy['selected']['name']} inner_auc_delta={policy['selected']['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V135 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V135 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V135 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V135 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        values["balanced_accuracy_delta"].append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)))
    require(len(values["auc_delta"]) >= int(0.9 * draws), "V135 bootstrap lost draws")
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V135 policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V135 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V135 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    require(candidate["balanced_accuracy"] == base["balanced_accuracy"] and candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"], "V135 evaluated decisions changed")
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    confidence_exact = bool(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    surface_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_only"]
        and item["predicted_direction_exact_v69"] and item["fit_audit"]["exact_weighted_pav"]
        and item["fit_audit"]["confidence_bins"] == 0 and item["reference_event_group_n"] >= 120
        and not item["target_rows_used_for_surface_fit_or_selection"] and not item["target_labels_used"]
        and item["fitted_causal_state_coefficient_n"] == 0 and not item["pair_list_auc_loss"]
        for item in full_models
    )
    model_native = [item["native_surface_bootstrap"] for item in full_models]
    model_native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and not item["target_labels_used"]
        and item["direction_agreement_with_nominal"] == 1.0 for item in model_native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_exact_v69": candidate["balanced_accuracy"] == base["balanced_accuracy"],
        "outer_all_trade_net_exact_v69": candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"],
        "outer_predicted_direction_exact_v69": all(audit["outer_direction_exact_v69"] for audit in audits),
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_ba_delta_exact_zero": all(nested["balanced_accuracy_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "nested_bootstrap_net_delta_exact_zero": all(nested["all_trade_net_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_exact_v69": candidate_metrics["balanced_accuracy"] == base_metrics["balanced_accuracy"],
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_monotone_surface_bootstrap_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "source_side_monotone_confidence_surface_contract_verified": surface_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V135 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"], "outer_ba_delta": 0.0, "outer_net_delta": 0.0,
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_monotone_surface_bootstrap": model_native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "immutability": {"confidence_exact": confidence_exact, "predicted_direction_exact": all(audit["outer_direction_exact_v69"] for audit in audits)},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_monotone_surface_bootstrap": evaluation["model_native_monotone_surface_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V135 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V135 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V135 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V135").resolve(), "V135 output outside controller staging")
    require(out.exists() and out.is_dir(), "V135 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V135 controller prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V135 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V135 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V135", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V135_DIAGNOSTIC_MONOTONE_SURFACE": evaluation["candidate_summary"], "V135_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V135", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V135", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V135_SOURCE_SIDE_MONOTONE_CONFIDENCE_SURFACE_REPORT.json": {"version": "V135", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V135", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V135_MONOTONE_SURFACE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V135_DIAGNOSTIC_MONOTONE_SURFACE_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V135_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    inner_probability, inner_audit = monotone_reliability_direction(inner_reference, inner_champion)
    outer_probability, outer_audit = monotone_reliability_direction(outer_reference, outer_champion)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold,
        "inner_baseline": metric(inner_valid, inner_champion.prob, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_surface": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "outer_baseline": metric(outer_valid, outer_champion.prob, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_surface_evaluation_only": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "inner_model_audit": inner_audit, "outer_model_audit": outer_audit,
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used_for_fit_or_selection": False, "output_written": False,
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V135 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V135 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
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
            "architecture": "strict-past source by side hierarchical no-bin monotone V69 confidence-correctness PAV surface within exact V69 decision cells",
            "causal_pre_event_only": True, "correction_strength_or_bin_count_tuned": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run",
                "output_bound_to_environment": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True,
                "model_native_surface_bootstrap": True, "exact_entire_v69_fallback": True,
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
            "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"],
            "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"],
            "fallback": evaluation["fallback"], "output_written": False,
        }), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
