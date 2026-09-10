"""V151 cross-fitted source-side Huber-expectile opportunity selector.

The atomic V69 probability and direction are immutable.  Every strict-past
reference is collapsed to equal event groups and split chronologically into an
early proper-fit segment and a later calibration segment with an internal
35-minute purge.  One fixed tau=.70 Huber-expectile linear head predicts the
realized signed net from following frozen V69 direction using 37 causal numeric
states, their missing flags, and source-family by frozen-side fixed effects.

Only out-of-fit later calibration predictions establish opportunity percentiles
and the high-confidence coverage threshold.  Frozen cross-fitted V69 confidence
percentiles supply the correctness-reliability component; their geometric mean
with opportunity percentile is the candidate confidence.  Candidate high-conf
matches calibration V69 coverage structurally and additionally requires positive
predicted signed net.  No correctness label is used to fit or calibrate this
selector.  One candidate competes with exact V69 noop on inner-past evidence;
outer labels are evaluation-only and failure restores the exact entire V69 frame.
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
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v148_mondrian_uniform_hoeffding_tail_correctness_lower_bound_selector as contract
import experiment_v85_recent_regime_entropy_balance as numeric


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V151_SOURCE_SIDE_CROSS_FITTED_HUBER_EXPECTILE_OPPORTUNITY_SELECTOR_V1"
VERSION = 151
HYPOTHESIS = "SOURCE_SIDE_CROSS_FITTED_HUBER_EXPECTILE_OPPORTUNITY_SELECTOR_V1"
FEATURES = numeric.FEATURES
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v148_mondrian_uniform_hoeffding_tail_correctness_lower_bound_selector.py"
EXPECTED_CONTRACT_SHA256 = "23974cd61b160e394dcb86711b6b26020d66d6da1b1096e494132a41a0fa8215"
NUMERIC_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_NUMERIC_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

PROPER_FRACTION = 0.70
EXPECTILE_TAU = 0.70
HUBER_DELTA = 1.50
RIDGE_L2 = 0.02
ROBUST_CLIP = 5.0
COST = 0.002
MIN_PROPER_N = 120
MIN_CALIBRATION_N = 50
MIN_HIGH_COVERAGE = 0.05
SEED = 15101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "CROSS_FITTED_HUBER_EXPECTILE_OPPORTUNITY_FULL"},)

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
        "strict-past_equal-event-groups|internal-proper70_calibration30_embargo35|"
        "37-causal-numeric+37-missing+source-by-v69-side-fixed-effects|"
        "single-global-linear-huber-expectile_tau0.70_delta1.50_l2.02|"
        "frozen-direction-signed-net-target|calibration-prediction-percentile|"
        "frozen-v69-confidence-percentile|fixed-geometric-mean|"
        "calibration-v69-highconf-coverage-match+predicted-net-positive|"
        "no-correctness-label-fit-or-calibration|one-candidate+noop"
    )
    require(PROPER_FRACTION == 0.70 and EXPECTILE_TAU == 0.70 and HUBER_DELTA == 1.50, "V151 fixed expectile spec changed")
    require(RIDGE_L2 == 0.02 and COST == 0.002 and len(ARCHITECTURES) == 1, "V151 fixed model spec changed")
    return {
        "input": "37 causal numeric plus 37 missing flags and source_family x frozen V69 predicted-side fixed effects",
        "cross_fit": "first 70% proper fit, later 30% calibration with internal strict 35m embargo",
        "target": "realized signed net from following exact frozen V69 direction",
        "head": "one global linear tau=.70 asymmetric Huber-expectile head, delta=1.5, L2=.02",
        "reliability": "frozen atomic V69 confidence percentile against later calibration",
        "opportunity": "target expectile prediction percentile against later calibration predictions",
        "combine": "fixed geometric mean of reliability and opportunity percentiles",
        "high_confidence": "calibration-matched frozen-V69 HC coverage threshold and predicted signed net > 0",
        "correctness_labels_used_for_fit_or_calibration": False,
        "candidate_count_excluding_noop": 1, "probability_and_direction_structurally_frozen": True,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V148 contract scaffold changed")
    require(sha256(NUMERIC_PATH) == EXPECTED_NUMERIC_SHA256, "pinned V85 numeric feature contract changed")
    audit = contract.verify_authority()
    audit["code_dependencies"] = [
        {"path": CONTRACT_PATH.name, "sha256": EXPECTED_CONTRACT_SHA256, "purpose": "authority, chronology, metrics, canonical gate, atomic IO and controller robustness only; no V148 output is read"},
        {"path": NUMERIC_PATH.name, "sha256": EXPECTED_NUMERIC_SHA256, "purpose": "immutable declaration of the 37 causal pre-decision numeric fields only; no V85 output is read"},
    ]
    audit["v151_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = contract.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V151 bounded affinity changed")
    return audit


def attach_atomic(dev: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    aligned = aligned_dev(dev, rows).copy()
    require(np.array_equal(aligned.event_id.astype(str), rows.event_id.astype(str)), "V151 atomic alignment order changed")
    for field in ("prob", "confidence_signal", "high_conf", "source_family"):
        aligned[field] = rows[field].to_numpy()
    aligned["high_conf"] = bool_series(aligned.high_conf)
    aligned["source_family"] = aligned.source_family.fillna("__MISSING_SOURCE__").astype(str)
    return aligned


def collapse_equal_event_groups(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["predicted_side"] = (work.prob.to_numpy(float) >= 0.5).astype(int)
    grouped = work.groupby("event_group_id", sort=True)
    for field in ("y", "predicted_side", "source_family"):
        require(int(grouped[field].nunique().max()) == 1, f"V151 event group crosses {field}")
    aggregate: dict[str, str] = {
        "event_id": "first", "event_time_utc": "min", "market": "first", "source_family": "first",
        "y": "first", "fwd_ret_30m": "mean", "prob": "mean", "confidence_signal": "mean",
        "high_conf": "first", "predicted_side": "first",
    }
    aggregate.update({feature: "mean" for feature in FEATURES})
    result = grouped.agg(aggregate).reset_index().sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    sign = np.where(result.predicted_side.to_numpy(int) == 1, 1.0, -1.0)
    result["signed_net_target"] = sign * result.fwd_ret_30m.to_numpy(float) - COST
    require(np.isfinite(result.signed_net_target).all(), "V151 signed-net target invalid")
    return result


def internal_cross_fit(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = collapse_equal_event_groups(reference)
    split = int(math.floor(PROPER_FRACTION * len(groups)))
    calibration = groups.iloc[split:].copy()
    require(len(calibration) >= MIN_CALIBRATION_N, "V151 calibration segment too small")
    calibration_start = pd.Timestamp(calibration.event_time_utc.min())
    proper = groups.loc[groups.event_time_utc < calibration_start - EMBARGO].copy()
    require(len(proper) >= MIN_PROPER_N, "V151 proper-fit segment too small")
    audit = chronology(proper, calibration, "V151 internal proper-to-calibration")
    audit.update({
        "reference_event_group_n": len(groups), "proper_event_group_n": len(proper),
        "calibration_event_group_n": len(calibration), "proper_fraction_pre_purge": PROPER_FRACTION,
        "correctness_labels_used_for_fit_or_calibration": False,
        "calibration_labels_or_returns_used": False,
    })
    return proper, calibration, audit


class OpportunityDesign:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)
        self.cells: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "OpportunityDesign":
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.median = np.nanmedian(raw, axis=0)
        lower = np.nanquantile(raw, 0.25, axis=0)
        upper = np.nanquantile(raw, 0.75, axis=0)
        spread = upper - lower
        fallback = np.nanstd(raw, axis=0)
        self.scale = np.where(np.isfinite(spread) & (spread > 1e-9), spread, np.where(fallback > 1e-9, fallback, 1.0))
        source = frame.source_family.fillna("__MISSING_SOURCE__").astype(str)
        side = (frame.prob.to_numpy(float) >= 0.5).astype(int)
        self.cells = tuple(sorted(set(source + "|" + pd.Series(side, index=frame.index).astype(str))))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all() and len(self.cells) >= 2, "V151 design fit invalid")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        missing = (~np.isfinite(raw)).astype(float)
        filled = np.where(np.isfinite(raw), raw, self.median)
        numeric_state = np.clip((filled - self.median) / self.scale, -ROBUST_CLIP, ROBUST_CLIP)
        source = frame.source_family.fillna("__MISSING_SOURCE__").astype(str)
        side = (frame.prob.to_numpy(float) >= 0.5).astype(int)
        keys = (source + "|" + pd.Series(side, index=frame.index).astype(str)).to_numpy()
        cell_state = np.column_stack([keys == cell for cell in self.cells]).astype(float)
        result = np.column_stack([np.ones(len(frame)), numeric_state, missing, cell_state])
        require(np.isfinite(result).all(), "V151 transformed design invalid")
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "causal_numeric_n": len(FEATURES), "missing_flag_n": len(FEATURES),
            "source_side_cell_n": len(self.cells), "source_side_cells": list(self.cells),
            "design_dimension": 1 + 2 * len(FEATURES) + len(self.cells),
            "median_sha256": array_sha256(self.median), "scale_sha256": array_sha256(self.scale),
        }


def fit_expectile(design: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(target, float)
    if sample_weight is None:
        sample_weight = np.ones(len(target), float)
    sample_weight = np.asarray(sample_weight, float)
    sample_weight = sample_weight / sample_weight.sum()
    center = float(np.median(target))
    scale = float(np.median(np.abs(target - center)) * 1.4826)
    if not math.isfinite(scale) or scale < 1e-4:
        scale = max(float(np.std(target)), 1e-4)
    standardized = (target - center) / scale
    penalty = np.ones(design.shape[1], float)
    penalty[0] = 0.0

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        residual = standardized - design @ coefficient
        asymmetric = np.where(residual >= 0.0, EXPECTILE_TAU, 1.0 - EXPECTILE_TAU)
        absolute = np.abs(residual)
        huber = np.where(absolute <= HUBER_DELTA, 0.5 * residual * residual, HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA))
        psi = np.clip(residual, -HUBER_DELTA, HUBER_DELTA)
        loss = float(np.sum(sample_weight * asymmetric * huber) + 0.5 * RIDGE_L2 * np.sum(penalty * coefficient * coefficient))
        gradient = -(design.T @ (sample_weight * asymmetric * psi)) + RIDGE_L2 * penalty * coefficient
        return loss, gradient

    initial = np.zeros(design.shape[1], float)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 220, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
    require(np.isfinite(result.x).all() and math.isfinite(float(result.fun)), "V151 expectile optimization failed")
    packed = np.concatenate([[center, scale], result.x])
    return packed, {
        "success": bool(result.success), "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun), "coefficient_n": len(result.x),
        "target_center": center, "target_scale": scale, "expectile_tau": EXPECTILE_TAU,
        "huber_delta": HUBER_DELTA, "ridge_l2": RIDGE_L2,
        "coefficient_sha256": array_sha256(result.x), "packed_head_sha256": array_sha256(packed),
    }


def head_predict(packed: np.ndarray, design: np.ndarray) -> np.ndarray:
    center, scale = float(packed[0]), float(packed[1])
    prediction = center + scale * (design @ packed[2:])
    require(np.isfinite(prediction).all(), "V151 expectile prediction invalid")
    return prediction


def percentile_against(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, float), kind="stable")
    query = np.asarray(query, float)
    left = np.searchsorted(ordered, query, side="left")
    right = np.searchsorted(ordered, query, side="right")
    result = (left + right + 1.0) / (2.0 * (len(ordered) + 1.0))
    return np.clip(result, 1e-6, 1.0 - 1e-6)


def calibrated_selector(calibration: pd.DataFrame, target: pd.DataFrame, calibration_opportunity: np.ndarray, target_opportunity: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    calibration_reliability = percentile_against(calibration.confidence_signal.to_numpy(float), calibration.confidence_signal.to_numpy(float))
    target_reliability = percentile_against(calibration.confidence_signal.to_numpy(float), target.confidence_signal.to_numpy(float))
    calibration_rank = percentile_against(calibration_opportunity, calibration_opportunity)
    target_rank = percentile_against(calibration_opportunity, target_opportunity)
    calibration_combined = np.sqrt(calibration_reliability * calibration_rank)
    target_combined = np.sqrt(target_reliability * target_rank)
    calibration_coverage = float(bool_series(calibration.high_conf).mean())
    if calibration_coverage <= 0.0:
        threshold = 1.0
    else:
        threshold = float(np.quantile(calibration_combined, 1.0 - calibration_coverage, method="higher"))
    high = (target_combined >= threshold) & (np.asarray(target_opportunity, float) > 0.0)
    return target_combined, high, {
        "calibration_v69_highconf_coverage": calibration_coverage,
        "calibration_matched_combined_threshold": threshold,
        "target_highconf_coverage": float(high.mean()), "target_positive_opportunity_fraction": float(np.mean(target_opportunity > 0.0)),
        "target_opportunity_min": float(np.min(target_opportunity)), "target_opportunity_median": float(np.median(target_opportunity)), "target_opportunity_max": float(np.max(target_opportunity)),
        "target_opportunity_sha256": array_sha256(target_opportunity),
        "target_confidence_sha256": array_sha256(target_combined), "target_highconf_sha256": array_sha256(high),
        "calibration_correctness_labels_used": False, "calibration_returns_used": False,
    }


def fit_selector(dev: pd.DataFrame, reference: pd.DataFrame, target: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_rows = attach_atomic(dev, reference)
    target_rows = attach_atomic(dev, target)
    proper, calibration, crossfit_audit = internal_cross_fit(reference_rows)
    design = OpportunityDesign().fit(proper)
    proper_x = design.transform(proper)
    calibration_x = design.transform(calibration)
    target_x = design.transform(target_rows)
    packed, head_audit = fit_expectile(proper_x, proper.signed_net_target.to_numpy(float))
    calibration_opportunity = head_predict(packed, calibration_x)
    target_opportunity = head_predict(packed, target_x)
    confidence, high, selector_audit = calibrated_selector(calibration, target_rows, calibration_opportunity, target_opportunity)
    native = None
    if native_draws > 0:
        generator = np.random.default_rng(native_seed)
        confidence_draws = np.empty((len(target_rows), native_draws), float)
        high_draws = np.empty((len(target_rows), native_draws), bool)
        opportunity_draws = np.empty((len(target_rows), native_draws), float)
        for draw in range(native_draws):
            weights = generator.exponential(1.0, len(proper))
            draw_head, _ = fit_expectile(proper_x, proper.signed_net_target.to_numpy(float), weights)
            draw_calibration = head_predict(draw_head, calibration_x)
            draw_target = head_predict(draw_head, target_x)
            confidence_draws[:, draw], high_draws[:, draw], _ = calibrated_selector(calibration, target_rows, draw_calibration, draw_target)
            opportunity_draws[:, draw] = draw_target
        native = {
            "method": "proper-fit equal-event-group Bayesian exponential-weight complete Huber-expectile refit with fixed later calibration mapping",
            "draws": native_draws, "seed": native_seed, "event_group_unit_weights": True,
            "target_or_calibration_labels_used": False, "direction_probability_changed": False,
            "median_absolute_confidence_deviation": float(np.median(np.abs(confidence_draws - confidence[:, None]))),
            "median_absolute_opportunity_deviation": float(np.median(np.abs(opportunity_draws - target_opportunity[:, None]))),
            "mean_high_confidence_agreement": float(np.mean(high_draws == high[:, None])),
            "confidence_draw_inventory_sha256": array_sha256(confidence_draws),
            "opportunity_draw_inventory_sha256": array_sha256(opportunity_draws),
        }
    return confidence, high, {
        "architecture": "internally cross-fitted source-side causal-state Huber-expectile frozen-direction signed-net opportunity selector",
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_only": True,
        "crossfit_audit": crossfit_audit, "design_audit": design.audit(), "head_audit": head_audit,
        "selector_audit": selector_audit, "native_opportunity_bootstrap": native,
        "proper_signed_net_target_sha256": array_sha256(proper.signed_net_target.to_numpy(float)),
        "calibration_opportunity_sha256": array_sha256(calibration_opportunity),
        "probability_sha256": array_sha256(target.prob.to_numpy(float)),
        "confidence_sha256": array_sha256(confidence), "high_confidence_sha256": array_sha256(high),
        "probability_exact_v69": True, "predicted_direction_exact_v69": True,
        "target_rows_used_for_fit_or_calibration": False, "target_labels_used": False,
        "correctness_labels_used_for_head_or_calibration": False,
        "direction_head_or_probability_blend": False,
    }


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference = champion.loc[champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference) >= 180, f"V151 {label} reference too small")
    reference_valid = aligned_dev(dev, reference)
    audit = chronology(reference_valid, target_valid, f"V151 {label}")
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
    if values["confidence_correctness_auc"] is None or values["highconf_accuracy"] is None or values["highconf_mean_signed_net"] is None:
        return -1e9
    return float(values["confidence_correctness_auc"] + 0.25 * values["highconf_accuracy"] + 10.0 * values["highconf_mean_signed_net"])


def choose_inner(dev: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    probability = inner_champion.prob.to_numpy(float)
    baseline_confidence = inner_champion.confidence_signal.to_numpy(float)
    baseline_high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, probability, baseline_confidence, baseline_high)
    confidence, high, model_audit = fit_selector(dev, inner_reference, inner_champion)
    values = metric(inner_valid, probability, confidence, high)
    minimum_high_n = max(12, int(math.ceil(MIN_HIGH_COVERAGE * len(inner_valid))))
    eligible = bool(
        values["highconf_n"] >= minimum_high_n
        and values["highconf_accuracy"] is not None and values["highconf_mean_signed_net"] is not None
        and (baseline_metric["highconf_accuracy"] is None or values["highconf_accuracy"] >= baseline_metric["highconf_accuracy"] - 0.01)
        and (baseline_metric["highconf_mean_signed_net"] is None or values["highconf_mean_signed_net"] >= baseline_metric["highconf_mean_signed_net"] - 0.0005)
    )
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "confidence_auc_delta": 0.0, "highconf_accuracy_delta": 0.0, "highconf_net_delta": 0.0,
        "eligible": True, "score": confidence_score(baseline_metric),
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": values,
        "confidence_auc_delta": None if values["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else values["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"],
        "highconf_accuracy_delta": None if values["highconf_accuracy"] is None or baseline_metric["highconf_accuracy"] is None else values["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
        "highconf_net_delta": None if values["highconf_mean_signed_net"] is None or baseline_metric["highconf_mean_signed_net"] is None else values["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
        "eligible": eligible, "minimum_highconf_n": minimum_high_n, "score": confidence_score(values),
    }]
    selected = max((trial for trial in trials if trial["eligible"]), key=lambda trial: (trial["score"], trial["highconf_net_delta"] or 0.0, trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past only; exact V69 confidence/high_conf versus one fixed cross-fitted Huber-expectile opportunity selector; maximize confidence-AUC + .25*HC accuracy + 10*HC net after fixed coverage/accuracy/net safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False, "threshold_or_model_hyperparameter_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v151_model"] = "V69_NOOP"
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
        policy, inner_model_audit = choose_inner(dev, inner_valid, inner_champion, inner_reference)
        opportunity_confidence, opportunity_high, outer_model_audit = fit_selector(
            dev, outer_reference, outer_champion,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        probability = outer_champion.prob.to_numpy(float)
        baseline_confidence = outer_champion.confidence_signal.to_numpy(float)
        baseline_high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected_confidence = baseline_confidence if policy["selected"]["architecture"] is None else opportunity_confidence
        selected_high = baseline_high if policy["selected"]["architecture"] is None else opportunity_high
        baseline_metric = metric(outer_valid, probability, baseline_confidence, baseline_high)
        raw_metric = metric(outer_valid, probability, opportunity_confidence, opportunity_high)
        candidate_metric = metric(outer_valid, probability, selected_confidence, selected_high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        confidence_lookup = dict(zip(outer_valid.event_id, selected_confidence))
        high_lookup = dict(zip(outer_valid.event_id, selected_high))
        diagnostic.loc[positions, "confidence_signal"] = diagnostic.loc[positions, "event_id"].map(confidence_lookup)
        diagnostic.loc[positions, "high_conf"] = diagnostic.loc[positions, "event_id"].map(high_lookup).astype(bool)
        diagnostic.loc[positions, "v151_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m"]].copy()
        evidence["baseline_prob"] = probability
        evidence["candidate_prob"] = probability
        evidence["baseline_confidence"] = baseline_confidence
        evidence["candidate_confidence"] = selected_confidence
        evidence["raw_opportunity_confidence"] = opportunity_confidence
        evidence["baseline_high"] = baseline_high
        evidence["candidate_high"] = selected_high
        evidence["raw_opportunity_high"] = opportunity_high
        evidence["v151_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": {ARCHITECTURES[0]["name"]: raw_metric},
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_confidence_auc_delta": None if candidate_metric["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else candidate_metric["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"],
            "outer_highconf_accuracy_delta": None if candidate_metric["highconf_accuracy"] is None or baseline_metric["highconf_accuracy"] is None else candidate_metric["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
            "outer_highconf_net_delta": None if candidate_metric["highconf_mean_signed_net"] is None or baseline_metric["highconf_mean_signed_net"] is None else candidate_metric["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
            "outer_probability_exact_v69": True, "outer_direction_exact_v69": True,
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V151 EXPECTILE OPPORTUNITY] market={market} fold={fold} selected={policy['selected']['name']} inner_score_delta={policy['selected']['score'] - confidence_score(policy['baseline']):+.6f} outer_hc_net_delta={audits[-1]['outer_highconf_net_delta'] or 0.0:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V151 evidence repeats event")
    require(np.array_equal(champion.prob.to_numpy(float), diagnostic.prob.to_numpy(float)), "V151 changed V69 probability")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V151 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    keys = ("auc_delta", "balanced_accuracy_delta", "all_trade_net_delta", "confidence_auc_delta", "highconf_accuracy_delta", "highconf_net_delta")
    values: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        probability = sample.baseline_prob.to_numpy(float)
        baseline_confidence = sample.baseline_confidence.to_numpy(float)
        candidate_confidence = sample.candidate_confidence.to_numpy(float)
        baseline_high = bool_series(sample.baseline_high).to_numpy(bool)
        candidate_high = bool_series(sample.candidate_high).to_numpy(bool)
        correct = (probability >= 0.5) == target
        signed = np.where(probability >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float) - COST
        values["auc_delta"].append(0.0)
        values["balanced_accuracy_delta"].append(0.0)
        values["all_trade_net_delta"].append(0.0)
        if np.unique(correct).size == 2 and baseline_high.any() and candidate_high.any():
            values["confidence_auc_delta"].append(float(roc_auc_score(correct, candidate_confidence) - roc_auc_score(correct, baseline_confidence)))
            values["highconf_accuracy_delta"].append(float(correct[candidate_high].mean() - correct[baseline_high].mean()))
            values["highconf_net_delta"].append(float(signed[candidate_high].mean() - signed[baseline_high].mean()))
    require(len(values["auc_delta"]) >= int(0.9 * draws), "V151 bootstrap lost standard draws")
    require(len(values["confidence_auc_delta"]) >= int(0.8 * draws), "V151 bootstrap lost confidence draws")
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V151 opportunity policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V151 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V151 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.baseline_confidence, bool_series(evidence.baseline_high))
    candidate = metric(evidence, evidence.candidate_prob, evidence.candidate_confidence, bool_series(evidence.candidate_high))
    require(candidate["balanced_accuracy"] == base["balanced_accuracy"] and candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"], "V151 evaluated direction changed")
    nested = paired_nested_bootstrap(evidence, draws)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    probability_exact = bool(np.array_equal(champion.prob.to_numpy(float), diagnostic_original.prob.to_numpy(float)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    opportunity_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_only"]
        and item["probability_exact_v69"] and item["predicted_direction_exact_v69"]
        and item["crossfit_audit"]["strict_35m_embargo"]
        and not item["crossfit_audit"]["correctness_labels_used_for_fit_or_calibration"]
        and not item["crossfit_audit"]["calibration_labels_or_returns_used"]
        and not item["correctness_labels_used_for_head_or_calibration"]
        and not item["target_rows_used_for_fit_or_calibration"] and not item["target_labels_used"]
        and not item["direction_head_or_probability_blend"] for item in full_models
    )
    model_native = [item["native_opportunity_bootstrap"] for item in full_models]
    model_native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and not item["target_or_calibration_labels_used"]
        and not item["direction_probability_changed"] for item in model_native
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
        "outer_confidence_correctness_auc_delta_ge_0": confidence_auc_delta is not None and confidence_auc_delta >= 0.0,
        "outer_highconf_accuracy_delta_ge_0": high_accuracy_delta is not None and high_accuracy_delta >= 0.0,
        "outer_highconf_net_delta_gt_0_0005": high_net_delta is not None and high_net_delta > 0.0005,
        "candidate_highconf_coverage_ge_0_05": candidate["highconf_n"] >= int(math.ceil(MIN_HIGH_COVERAGE * candidate["n"])),
        "nested_bootstrap_confidence_auc_probability_gt_zero_ge_0_65": nested["confidence_auc_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_highconf_accuracy_probability_gt_zero_ge_0_65": nested["highconf_accuracy_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_highconf_net_probability_gt_zero_ge_0_75": nested["highconf_net_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_ba_delta_exact_zero": all(nested["balanced_accuracy_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "nested_bootstrap_all_trade_net_delta_exact_zero": all(nested["all_trade_net_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "full_overall_ba_exact_v69": candidate_metrics["balanced_accuracy"] == base_metrics["balanced_accuracy"],
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_opportunity_bootstrap_verified": model_native_valid,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "cross_fitted_huber_expectile_opportunity_contract_verified": opportunity_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V151 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate, "outer_auc_delta": 0.0, "outer_ba_delta": 0.0, "outer_net_delta": 0.0,
        "outer_confidence_auc_delta": confidence_auc_delta, "outer_highconf_accuracy_delta": high_accuracy_delta, "outer_highconf_net_delta": high_net_delta,
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_opportunity_bootstrap": model_native,
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
        "model_native_opportunity_bootstrap": evaluation["model_native_opportunity_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V151 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V151 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V151 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V151").resolve(), "V151 output outside controller staging")
    require(out.exists() and out.is_dir(), "V151 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V151 controller prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V151 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V151 selected native robustness missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V151", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V151_DIAGNOSTIC_EXPECTILE_OPPORTUNITY": evaluation["candidate_summary"], "V151_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V151", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V151", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V151_CROSS_FITTED_EXPECTILE_OPPORTUNITY_REPORT.json": {"version": "V151", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V151", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V151_EXPECTILE_OPPORTUNITY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V151_DIAGNOSTIC_EXPECTILE_OPPORTUNITY_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V151_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    inner_confidence, inner_high, inner_audit = fit_selector(dev, inner_reference, inner_champion)
    outer_confidence, outer_high, outer_audit = fit_selector(dev, outer_reference, outer_champion)
    inner_probability = inner_champion.prob.to_numpy(float)
    outer_probability = outer_champion.prob.to_numpy(float)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold,
        "inner_baseline": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_opportunity": metric(inner_valid, inner_probability, inner_confidence, inner_high),
        "outer_baseline": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_opportunity_evaluation_only": metric(outer_valid, outer_probability, outer_confidence, outer_high),
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V151 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V151 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
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
            "architecture": "proper-fit/later-calibration source-side causal-state Huber-expectile frozen-direction signed-net opportunity selector",
            "causal_pre_event_only": True, "threshold_or_model_hyperparameter_tuned": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True,
                "model_native_opportunity_bootstrap": True, "exact_entire_v69_fallback": True,
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
            "outer_highconf_accuracy_delta": evaluation["outer_highconf_accuracy_delta"], "outer_highconf_net_delta": evaluation["outer_highconf_net_delta"],
            "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"],
            "fallback": evaluation["fallback"], "output_written": False,
        }), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
