"""V138 atomic-V69 offset-logistic causal-state residual correction.

Within every strict-past same-market cut, atomic V69 OOF rows collapse to equal
event-group observations.  The immutable V69 direction logit enters a Bernoulli
logistic objective as a coefficient-one offset.  A single no-intercept residual
head over train-robust 37 causal numeric fields and 37 missing flags is fitted
with one fixed L2 penalty; prediction is sigmoid(V69 logit + clipped correction).
Only exact V69 noop versus this one full correction is selected inner-only.

Unlike V91, no chronological anchor or residual-invariance projection is used
and the learned score is not standalone: it can only correct the frozen V69
offset.  Unlike V62/V130/V135, this is neither V69-logit calibration nor a
source/confidence correctness reliability model.  There is no blend, penalty
grid, pairwise AUC loss, threshold tuning or post-outer adjustment.  Outer labels
are evaluation-only under a strict 35-minute embargo/event purge; V69 confidence
and high_conf remain exact and material failure restores exact entire V69.
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
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v135_source_side_monotone_confidence_reliability_rerank as contract


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V138_ATOMIC_V69_OFFSET_LOGISTIC_CAUSAL_STATE_CORRECTION_V1"
VERSION = 138
HYPOTHESIS = "ATOMIC_V69_OFFSET_LOGISTIC_CAUSAL_STATE_CORRECTION_V1"
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v135_source_side_monotone_confidence_reliability_rerank.py"
EXPECTED_CONTRACT_SHA256 = "49e298699fea19292808b4b469d66135cc04e26c8ae0d339a829617849af40a6"

numeric = contract.contract.contract.scaffold
FEATURES = numeric.FEATURES
EMBARGO = contract.EMBARGO
INNER_VALID_FRACTION = contract.INNER_VALID_FRACTION
RIDGE = 0.10
CORRECTION_CLIP = 2.0
PROBABILITY_EPSILON = 1e-5
OPTIMIZER_MAX_ITER = 300
SEED = 13801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "ATOMIC_V69_OFFSET_LOGISTIC_FULL"},)

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


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "equal-event-group|37-causal-numeric-plus-37-missing|"
        "immutable-coefficient-one-logit-v69-offset|no-intercept-74d-residual|"
        "fixed-l2-0.10-Bernoulli-logistic|analytic-gradient-LBFGS|"
        "fixed-correction-clip-2|noop-versus-single-full-correction"
    )
    require(len(FEATURES) == 37 and RIDGE == 0.10 and CORRECTION_CLIP == 2.0, "V138 static spec changed")
    require(len(ARCHITECTURES) == 1, "V138 architecture grid forbidden")
    return {
        "causal_numeric_n": len(FEATURES), "missing_flag_n": len(FEATURES),
        "residual_design_n": 2 * len(FEATURES), "residual_intercept": False,
        "atomic_v69_logit_offset_coefficient": 1.0,
        "objective": "equal-event-group class-balanced Bernoulli logistic negative log likelihood plus fixed L2 residual penalty",
        "ridge": RIDGE, "correction_clip": [-CORRECTION_CLIP, CORRECTION_CLIP],
        "optimizer": "deterministic analytic-gradient L-BFGS-B",
        "architecture_candidates_excluding_noop": 1,
        "blend_or_regularization_grid": False,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V135 contract scaffold changed")
    audit = contract.verify_authority()
    audit["code_dependency"] = {
        "path": CONTRACT_PATH.name, "sha256": EXPECTED_CONTRACT_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust numeric state, metrics, canonical gate, bootstrap and atomic IO only; no V135 output is read",
    }
    audit["v138_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = contract.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V138 bounded affinity changed")
    return audit


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference_champion = champion.loc[champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference_champion) >= 120, f"V138 {label} reference too small")
    reference = aligned_dev(dev, reference_champion)
    reference["v69_prob"] = reference.event_id.map(dict(zip(reference_champion.event_id, reference_champion.prob))).to_numpy(float)
    audit = chronology(reference, target_valid, f"V138 {label} offset correction")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF joined to immutable V36 causal state"
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


def collapse_event_groups(reference: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_group_id", "event_time_utc", "market", "y", "v69_prob", *FEATURES]
    work = reference[columns].copy()
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V138 event group crosses labels")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "y": "first", "v69_prob": "mean",
        **{feature: "mean" for feature in FEATURES},
    }).reset_index()
    require(result.event_group_id.is_unique and np.isfinite(result.v69_prob).all(), "V138 group collapse invalid")
    return result


def class_balanced_group_weights(target: np.ndarray, bootstrap_weight: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(target, int)
    weight = np.ones(len(target), float) if bootstrap_weight is None else np.asarray(bootstrap_weight, float)
    require(weight.shape == (len(target),) and np.all(weight > 0) and np.isfinite(weight).all(), "V138 weight invalid")
    masses = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require(np.all(masses > 0), "V138 class mass missing")
    total = float(weight.sum())
    weight *= np.where(target == 1, total / (2.0 * masses[1]), total / (2.0 * masses[0]))
    return weight / float(np.mean(weight))


class RobustNumericMissingState:
    def __init__(self) -> None:
        self.median = np.empty(0, float)
        self.scale = np.empty(0, float)

    def fit(self, frame: pd.DataFrame) -> "RobustNumericMissingState":
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        self.median = np.nanmedian(raw, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        lower, upper = np.nanquantile(raw, [0.25, 0.75], axis=0)
        scale = upper - lower
        fallback = np.nanstd(raw, axis=0)
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.where(np.isfinite(fallback) & (fallback > 1e-9), fallback, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V138 robust state fit invalid")
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        scaled = np.clip((filled - self.median) / self.scale, -6.0, 6.0)
        result = np.column_stack([scaled, (~observed).astype(float)])
        require(result.shape[1] == 2 * len(FEATURES) and np.isfinite(result).all(), "V138 robust state transform invalid")
        return result, {
            "rows": len(frame), "numeric_feature_n": len(FEATURES),
            "missing_flag_n": len(FEATURES), "matrix_columns": result.shape[1],
            "missing_rate_mean": float((~observed).mean()),
            "missing_rate_max": float((~observed).mean(axis=0).max()),
        }


def fit_residual_head(design: np.ndarray, target: np.ndarray, offset: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    design = np.asarray(design, float)
    target = np.asarray(target, float)
    offset = np.asarray(offset, float)
    weight = np.asarray(weight, float)
    require(design.shape == (len(target), 2 * len(FEATURES)), "V138 design shape invalid")
    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = offset + design @ coefficient
        probability = expit(score)
        loss = float(np.sum(weight * (np.logaddexp(0.0, score) - target * score)) / np.sum(weight) + 0.5 * RIDGE * np.dot(coefficient, coefficient))
        gradient = design.T @ (weight * (probability - target)) / np.sum(weight) + RIDGE * coefficient
        return loss, gradient
    initial = np.zeros(design.shape[1], float)
    result = minimize(lambda beta: objective(beta), initial, method="L-BFGS-B", jac=True, options={"maxiter": OPTIMIZER_MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
    coefficient = np.asarray(result.x, float)
    loss, gradient = objective(coefficient)
    require(bool(result.success) and np.isfinite(coefficient).all() and np.isfinite(loss), f"V138 optimizer failed: {result.message}")
    return coefficient, {
        "optimizer_success": True, "optimizer_status": int(result.status), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "objective": loss, "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "coefficient_l2": float(np.linalg.norm(coefficient)), "coefficient_sha256": array_sha256(coefficient),
    }


def prepare_model(reference: pd.DataFrame) -> tuple[pd.DataFrame, Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    groups = collapse_event_groups(reference)
    scaler = RobustNumericMissingState().fit(groups)
    design, transform_audit = scaler.transform(groups)
    target = groups.y.to_numpy(int)
    offset = logit(np.clip(groups.v69_prob.to_numpy(float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))
    require(design.shape[1] == 2 * len(FEATURES), "V138 robust state dimension changed")
    return groups, scaler, design, target, offset, transform_audit


def predict_with_head(scaler: Any, coefficient: np.ndarray, target_frame: pd.DataFrame, baseline: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    target_design, target_transform = scaler.transform(target_frame)
    baseline = np.clip(np.asarray(baseline, float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    offset = logit(baseline)
    raw_correction = target_design @ coefficient
    correction = np.clip(raw_correction, -CORRECTION_CLIP, CORRECTION_CLIP)
    probability = np.clip(expit(offset + correction), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    require(np.isfinite(probability).all(), "V138 probability invalid")
    return probability, {
        "target_transform": target_transform,
        "raw_correction_q10_q50_q90": [float(np.quantile(raw_correction, q)) for q in (0.1, 0.5, 0.9)],
        "clipped_correction_q10_q50_q90": [float(np.quantile(correction, q)) for q in (0.1, 0.5, 0.9)],
        "correction_clipped_fraction": float(np.mean(np.abs(raw_correction) > CORRECTION_CLIP)),
        "baseline_probability_sha256": array_sha256(baseline), "probability_sha256": array_sha256(probability),
        "predicted_direction_change_fraction": float(np.mean((probability >= 0.5) != (baseline >= 0.5))),
    }


def native_head_bootstrap(groups: pd.DataFrame, scaler: Any, design: np.ndarray, target: np.ndarray, offset: np.ndarray, target_frame: pd.DataFrame, baseline: np.ndarray, nominal: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    samples = np.empty((len(target_frame), draws), float)
    coefficient_hashes: list[str] = []
    for draw in range(draws):
        exponential = generator.exponential(1.0, len(groups))
        coefficient, _ = fit_residual_head(design, target, offset, class_balanced_group_weights(target, exponential))
        samples[:, draw], _ = predict_with_head(scaler, coefficient, target_frame, baseline)
        coefficient_hashes.append(array_sha256(coefficient))
    std = np.std(samples, axis=1)
    median = np.median(samples, axis=1)
    return {
        "contract": "equal-event-group Bayesian bootstrap complete refit of fixed-offset residual logistic head",
        "draws": draws, "seed": seed, "event_group_unit_weights": True, "target_labels_used": False,
        "atomic_v69_offset_refit": False, "residual_head_complete_refit": True,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(samples - nominal[:, None]))),
        "row_probability_std_q10_q50_q90": [float(np.quantile(std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median),
        "coefficient_hash_inventory_sha256": hashlib.sha256("\n".join(coefficient_hashes).encode("utf-8")).hexdigest(),
    }


def offset_correction_direction(reference: pd.DataFrame, target_frame: pd.DataFrame, baseline: np.ndarray, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, dict[str, Any]]:
    require(reference.market.nunique() == target_frame.market.nunique() == 1 and str(reference.market.iloc[0]) == str(target_frame.market.iloc[0]), "V138 market mismatch")
    groups, scaler, design, target, offset, train_transform = prepare_model(reference)
    weight = class_balanced_group_weights(target)
    coefficient, fit_audit = fit_residual_head(design, target, offset, weight)
    probability, prediction_audit = predict_with_head(scaler, coefficient, target_frame, baseline)
    native = None if native_draws <= 0 else native_head_bootstrap(groups, scaler, design, target, offset, target_frame, baseline, probability, native_draws, native_seed)
    baseline_train_loss = float(np.mean(np.logaddexp(0.0, offset) - target * offset))
    corrected_train_score = offset + design @ coefficient
    corrected_train_loss = float(np.average(np.logaddexp(0.0, corrected_train_score) - target * corrected_train_score, weights=weight))
    return probability, {
        "market": str(reference.market.iloc[0]), "reference_row_n": len(reference),
        "reference_event_group_n": len(groups), "target_n": len(target_frame),
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_only": True,
        "equal_event_group_training": True, "causal_numeric_feature_n": len(FEATURES),
        "missing_flag_n": len(FEATURES), "source_text_ticker_issuer_or_return_feature_n": 0,
        "architecture": "immutable coefficient-one atomic V69 logit offset plus one fixed-L2 no-intercept causal-state residual logistic correction",
        "static_spec": STATIC_SPEC, "train_transform": train_transform,
        "fit_audit": fit_audit, "prediction_audit": prediction_audit,
        "baseline_unweighted_train_logloss": baseline_train_loss,
        "corrected_weighted_train_logloss": corrected_train_loss,
        "target_rows_used_for_scaler_or_head_fit": False, "target_labels_used": False,
        "atomic_v69_offset_coefficient_frozen_one": True, "residual_intercept_fitted": False,
        "chronological_anchor_residual_projection": False,
        "source_confidence_reliability_or_probability_calibration": False,
        "pair_list_auc_loss": False, "blend_or_regularization_grid": False,
        "native_residual_head_bootstrap": native,
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = inner_champion.prob.to_numpy(float)
    candidate, model_audit = offset_correction_direction(inner_reference, inner_valid, baseline)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    base_metric = metric(inner_valid, baseline, confidence, high)
    candidate_metric = metric(inner_valid, candidate, confidence, high)
    auc_delta = candidate_metric["auc"] - base_metric["auc"]
    ba_delta = candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"]
    net_delta = candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"]
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": base_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True, "score": 2.0 * base_metric["auc"] + base_metric["balanced_accuracy"],
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": candidate_metric,
        "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
        "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
        "score": 2.0 * candidate_metric["auc"] + candidate_metric["balanced_accuracy"],
    }]
    selected = max((trial for trial in trials if trial["eligible"]), key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past only; exact V69 noop versus one fixed atomic-offset residual correction; maximize 2*AUC+BA with fixed BA/net safety guard",
        "baseline": base_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False, "blend_penalty_clip_or_optimizer_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v138_model"] = "V69_NOOP"
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
        full_probability, outer_model_audit = offset_correction_direction(
            outer_reference, outer_valid, outer_champion.prob.to_numpy(float),
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        baseline = outer_champion.prob.to_numpy(float)
        candidate = baseline.copy() if policy["selected"]["architecture"] is None else full_probability
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        base_metric = metric(outer_valid, baseline, confidence, high)
        full_metric = metric(outer_valid, full_probability, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        lookup = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(lookup)
        diagnostic.loc[positions, "v138_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf"]].copy()
        evidence["baseline_prob"] = baseline
        evidence["full_correction_prob"] = full_probability
        evidence["candidate_prob"] = candidate
        evidence["v138_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": {ARCHITECTURES[0]["name"]: full_metric},
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V138 OFFSET] market={market} fold={fold} selected={policy['selected']['name']} inner_auc_delta={policy['selected']['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V138 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V138 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V138 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V138 bootstrap blocks")
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
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {"contract_key": "selected.material_gate.nested_bootstrap", "standardized": True, "method": "paired market-fold-time block bootstrap over inner-locked V138 atomic-offset policy", "seed": SEED, "requested_draws": draws, "blocks": len(blocks), **{key: interval(numbers) for key, numbers in values.items()}}


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V138 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V138 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    confidence_exact = bool(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    model_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_only"] and item["equal_event_group_training"]
        and item["atomic_v69_offset_coefficient_frozen_one"] and not item["residual_intercept_fitted"]
        and item["fit_audit"]["optimizer_success"] and not item["target_rows_used_for_scaler_or_head_fit"]
        and not item["target_labels_used"] and not item["chronological_anchor_residual_projection"]
        and not item["source_confidence_reliability_or_probability_calibration"]
        and not item["pair_list_auc_loss"] and not item["blend_or_regularization_grid"] for item in full_models
    )
    model_native = [item["native_residual_head_bootstrap"] for item in full_models]
    model_native_valid = all(item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS) and item["event_group_unit_weights"] and not item["target_labels_used"] and not item["atomic_v69_offset_refit"] and item["residual_head_complete_refit"] for item in model_native)
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_residual_head_bootstrap_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "atomic_v69_offset_logistic_residual_contract_verified": model_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V138 fallback not exact V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"], "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_residual_head_bootstrap": model_native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {"contract": HYPOTHESIS, "checks": evaluation["material_checks"], "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"], "model_native_residual_head_bootstrap": evaluation["model_native_residual_head_bootstrap"], "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V138 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V138 output not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V138").resolve(), "V138 output outside controller staging")
    require(out.exists() and out.is_dir(), "V138 controller stage missing")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing) and not (existing - allowed), "V138 controller prefile conflict")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    reports = {
        "MODEL_COMPARISON.json": {"version": "V138", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V138_DIAGNOSTIC_OFFSET_CORRECTION": evaluation["candidate_summary"], "V138_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V138", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V138", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V138_ATOMIC_V69_OFFSET_LOGISTIC_CORRECTION_REPORT.json": {"version": "V138", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V138", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V138_OFFSET_CORRECTION_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V138_DIAGNOSTIC_OFFSET_CORRECTION_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V138_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    inner_probability, inner_audit = offset_correction_direction(inner_reference, inner_valid, inner_champion.prob.to_numpy(float))
    outer_probability, outer_audit = offset_correction_direction(outer_reference, outer_valid, outer_champion.prob.to_numpy(float))
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold,
        "inner_baseline": metric(inner_valid, inner_champion.prob, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_full_correction": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "outer_baseline": metric(outer_valid, outer_champion.prob, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_full_correction_evaluation_only": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V138 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V138 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
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
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "coefficient-one atomic V69 logit offset plus one fixed-L2 no-intercept causal numeric/missing residual logistic head",
            "causal_pre_event_only": True, "blend_penalty_or_threshold_grid": False,
            "controller_no_arg_contract": {"environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run", "output_bound_to_environment": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True, "model_native_residual_head_bootstrap": True, "exact_entire_v69_fallback": True, "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"]},
            "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "output_written": False,
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
        print(json.dumps(clean({"status": "SMOKE_OK", "hypothesis": HYPOTHESIS, "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "folds": len(audits), "audits": audits, "outer_baseline": evaluation["outer_baseline"], "outer_candidate": evaluation["outer_candidate"], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"], "fallback": evaluation["fallback"], "output_written": False}), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
