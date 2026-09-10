"""V157 source-time density-ratio reweighted causal-state direction.

Each strict-past same-market fit collapses immutable V36 causal state to equal
event-group centroids.  Its last chronological quartile is a label-free recent
reference.  Deterministic pooled and sufficiently supported source-family
old-versus-recent logistic domain classifiers estimate train-only covariate
density odds.  Supported source log-ratios receive one fixed support shrinkage
toward the pooled log-ratio; ratios are clipped once to [0.25, 4].  These fixed
weights reweight a single class-balanced ridge-logistic direction head.

V85 instead solves an exponential entropy-tilting moment-matching problem and
has no density classifier or source hierarchy. V117 estimates the supervised
UP-to-DOWN class density ratio directly and has no chronological domain target
or reweighted direction head. V145 transports featurewise source-time marginal
quantiles rather than weighting rows. Inner labels select exact V69 or fixed
.25/.50 blends; outer labels are evaluation-only under 35-minute embargo/event
purge. Atomic V69 confidence/high_conf remain exact, and any material failure
restores the entire V69 frame. Prep uses CPU30-31, two threads, and no GPU.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "2"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import experiment_v154_nystrom_rbf_pairwise_auc_direction as utilities


ROOT = utilities.ROOT
V69_DIR = utilities.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V157" / "LOCAL_FORBIDDEN"
VERSION = 157
HYPOTHESIS = "SOURCE_TIME_DENSITY_RATIO_REWEIGHTED_DIRECTION_V1"
FEATURES = utilities.FEATURES
STATE_DIMENSION = utilities.STATE_DIMENSION
EXPECTED_V69_EXPERIMENT_ID = utilities.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v154_nystrom_rbf_pairwise_auc_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "df08de8ada279e613bba09c2d60299ee4b19f819586c2e49fcd208c785a5124c"

RECENT_REFERENCE_FRACTION = 0.25
DOMAIN_L2 = 0.20
DIRECTION_L2 = 0.05
MIN_SOURCE_OLD = 40
MIN_SOURCE_RECENT = 20
SOURCE_SHRINK_SUPPORT = 64.0
RATIO_FLOOR = 0.25
RATIO_CAP = 4.0
OPTIMIZER_MAX_ITERATIONS = 400
SCORE_CLIP = 24.0
SEED = 15701
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "SOURCE_TIME_DENSITY_RATIO_W0.25", "weight": 0.25},
    {"name": "SOURCE_TIME_DENSITY_RATIO_W0.50", "weight": 0.50},
)

require = utilities.require
sha256 = utilities.sha256
array_sha256 = utilities.array_sha256
clean = utilities.clean
bool_series = utilities.bool_series
atomic_json = utilities.atomic_json
atomic_csv = utilities.atomic_csv
load_authorized = utilities.load_authorized
metric = utilities.metric
blend_probability = utilities.blend_probability
controller = utilities.controller
v44 = utilities.v44
numeric = utilities.numeric
scaffold = utilities.scaffold
group_state_source = utilities.utilities.group_state_source

STATIC_SPEC = {
    "causal_numeric_feature_n": len(FEATURES),
    "deterministic_missing_flag_n": len(FEATURES),
    "state_dimension": STATE_DIMENSION,
    "event_unit": "equal event-group robust-state centroid",
    "recent_reference": "last chronological quartile inside the strict-past fit only",
    "recent_reference_fraction": RECENT_REFERENCE_FRACTION,
    "density_ratio": "balanced old-versus-recent logistic domain odds",
    "pooled_domain_l2": DOMAIN_L2,
    "source_domain_l2": DOMAIN_L2,
    "minimum_source_old": MIN_SOURCE_OLD,
    "minimum_source_recent": MIN_SOURCE_RECENT,
    "source_log_ratio_shrink": "min(old,recent)/(min(old,recent)+64)",
    "ratio_clip": [RATIO_FLOOR, RATIO_CAP],
    "direction": "one ratio-reweighted class-balanced ridge logistic head",
    "direction_l2": DIRECTION_L2,
    "reference_fraction_domain_l2_support_shrink_ratio_clip_direction_l2_or_blend_grid": False,
}
require(STATE_DIMENSION == 74 and 0.0 < RATIO_FLOOR < 1.0 < RATIO_CAP, "V157 static contract changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V154 utility scaffold changed")
    audit = utilities.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust causal state, equal-event grouping, metrics, canonical gate, fixed blending, bootstrap and atomic IO only; no V154 output is read",
    }
    audit["v157_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
        "prior_version_output_read": False,
    }
    return audit


def fit_balanced_logistic(
    design: np.ndarray,
    target: np.ndarray,
    l2: float,
    base_weight: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(target, int)
    require(design.ndim == 2 and len(design) == len(target), f"V157 {label} design invalid")
    require(np.unique(target).size == 2, f"V157 {label} lacks both classes")
    weight = np.ones(len(target), float) if base_weight is None else np.asarray(base_weight, float).copy()
    require(weight.shape == target.shape and np.all(weight > 0.0) and np.isfinite(weight).all(), f"V157 {label} base weights invalid")
    masses = np.asarray([weight[target == value].sum() for value in (0, 1)], float)
    weight *= np.where(target == 1, weight.sum() / (2.0 * masses[1]), weight.sum() / (2.0 * masses[0]))
    weight /= weight.sum()
    matrix = np.column_stack([np.ones(len(design)), design])

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = matrix @ coefficient
        value = float(np.sum(weight * (np.logaddexp(0.0, score) - target * score)))
        gradient = matrix.T @ (weight * (expit(score) - target))
        value += 0.5 * l2 * float(coefficient[1:] @ coefficient[1:])
        gradient[1:] += l2 * coefficient[1:]
        return value, gradient

    result = minimize(
        objective, np.zeros(matrix.shape[1], float), method="L-BFGS-B", jac=True,
        options={"maxiter": OPTIMIZER_MAX_ITERATIONS, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    final_value, final_gradient = objective(np.asarray(result.x, float))
    require(bool(result.success) and np.isfinite(result.x).all(), f"V157 {label} optimizer failed: {result.message}")
    return np.asarray(result.x, float), {
        "label": label, "objective": "class-balanced weighted Bernoulli logistic plus fixed feature L2",
        "l2": l2, "optimizer": "L-BFGS-B analytic gradient",
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "final_objective": final_value, "gradient_linf": float(np.max(np.abs(final_gradient))),
        "coefficient_l2": float(np.linalg.norm(result.x[1:])),
        "coefficient_sha256": array_sha256(np.asarray(result.x, float)),
        "class_counts": np.bincount(target, minlength=2).astype(int).tolist(),
    }


def logistic_score(state: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.clip(coefficient[0] + state @ coefficient[1:], -SCORE_CLIP, SCORE_CLIP)


def density_ratio_weights(
    state: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = len(state)
    recent_start = int(np.floor(rows * (1.0 - RECENT_REFERENCE_FRACTION)))
    recent = np.arange(rows) >= recent_start
    require(int(recent.sum()) >= 100 and int((~recent).sum()) >= 120, "V157 old/recent support too small")
    pooled_coefficient, pooled_fit = fit_balanced_logistic(state, recent.astype(int), DOMAIN_L2, None, "pooled old-versus-recent domain")
    pooled_log_ratio = logistic_score(state, pooled_coefficient)
    combined_log_ratio = pooled_log_ratio.copy()
    source_audits: list[dict[str, Any]] = []
    supported: list[str] = []
    for family in sorted(set(source.tolist())):
        mask = source == family
        old_n = int(np.sum(mask & ~recent))
        recent_n = int(np.sum(mask & recent))
        record: dict[str, Any] = {"source_family": str(family), "old_n": old_n, "recent_n": recent_n, "supported": False}
        if old_n >= MIN_SOURCE_OLD and recent_n >= MIN_SOURCE_RECENT:
            coefficient, fit_audit = fit_balanced_logistic(state[mask], recent[mask].astype(int), DOMAIN_L2, None, f"source={family} old-versus-recent domain")
            support = min(old_n, recent_n)
            alpha = float(support / (support + SOURCE_SHRINK_SUPPORT))
            source_log_ratio = logistic_score(state[mask], coefficient)
            combined_log_ratio[mask] = (1.0 - alpha) * pooled_log_ratio[mask] + alpha * source_log_ratio
            record.update({"supported": True, "shrink_alpha": alpha, "fit": fit_audit})
            supported.append(str(family))
        source_audits.append(record)
    raw_ratio = np.exp(np.clip(combined_log_ratio, np.log(1e-3), np.log(1e3)))
    clipped_ratio = np.clip(raw_ratio, RATIO_FLOOR, RATIO_CAP)
    normalized_weight = clipped_ratio / float(np.mean(clipped_ratio))
    require(np.isfinite(normalized_weight).all() and np.all(normalized_weight > 0.0), "V157 density ratio weights invalid")
    effective_n = float(np.square(normalized_weight.sum()) / np.square(normalized_weight).sum())
    return normalized_weight, {
        "estimator": "balanced discriminative old-versus-recent logistic density odds",
        "recent_definition": "last chronological quartile of strict-past event-group centroids",
        "rows": rows, "old_n": int((~recent).sum()), "recent_n": int(recent.sum()),
        "recent_start_position": recent_start,
        "pooled_fit": pooled_fit,
        "source_models": source_audits, "supported_sources": supported,
        "source_log_ratio_support_shrink": SOURCE_SHRINK_SUPPORT,
        "raw_ratio_quantiles": {str(q): float(np.quantile(raw_ratio, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
        "ratio_floor": RATIO_FLOOR, "ratio_cap": RATIO_CAP,
        "clipped_low_fraction": float(np.mean(raw_ratio < RATIO_FLOOR)),
        "clipped_high_fraction": float(np.mean(raw_ratio > RATIO_CAP)),
        "normalized_weight_quantiles": {str(q): float(np.quantile(normalized_weight, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
        "effective_sample_n": effective_n, "effective_sample_fraction": effective_n / rows,
        "weights_sha256": array_sha256(normalized_weight),
        "chronology_indicator_labels_only": True,
        "direction_or_return_labels_used": False,
        "validation_or_outer_features_used": False,
        "entropy_moment_matching": False,
        "class_density_ratio": False,
    }


def source_time_density_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V157 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    group_state, group_target, group_source, group_id, group_audit = group_state_source(ordered, train_state)
    ratio_weight, ratio_audit = density_ratio_weights(group_state, group_source)
    coefficient, direction_fit = fit_balanced_logistic(group_state, group_target, DIRECTION_L2, ratio_weight, "density-ratio reweighted direction")
    target_score = logistic_score(target_state, coefficient)
    probability = np.clip(expit(target_score), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V157 probability invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_id), "target_n": len(target_frame),
        "causal_numeric_and_missing_only": True,
        "categorical_text_ticker_or_issuer_direction_feature_n": 0,
        "source_family_role": "strict-past supported domain density-ratio hierarchy only; never a direction feature",
        "train_transform": train_transform, "target_transform": target_transform,
        "group_audit": group_audit, "density_ratio_audit": ratio_audit,
        "direction_fit": direction_fit,
        "target_rows_used_for_robust_scale_centroids_recent_reference_domain_fit_ratio_or_direction_fit": False,
        "target_labels_used": False,
        "train_only_source_time_density_ratio": True,
        "fixed_clipped_candidate_weights": True,
        "entropy_balance_or_moment_tilting": False,
        "up_down_class_density_ratio": False,
        "source_time_quantile_transport": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "target_score_sha256": array_sha256(target_score),
            "probability_sha256": array_sha256(probability),
        },
    }


def supported_source_ba_delta(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index].y.to_numpy(int)
        if len(index) < 40 or np.unique(target).size != 2:
            continue
        deltas[str(source)] = float(
            balanced_accuracy_score(target, candidate[index] >= 0.5)
            - balanced_accuracy_score(target, baseline[index] >= 0.5)
        )
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = source_time_density_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    ratio = model_audit["density_ratio_audit"]
    model_eligible = bool(
        model_audit["causal_numeric_and_missing_only"]
        and model_audit["train_only_source_time_density_ratio"]
        and model_audit["fixed_clipped_candidate_weights"]
        and ratio["pooled_fit"]["optimizer_success"]
        and ratio["direction_or_return_labels_used"] is False
        and ratio["validation_or_outer_features_used"] is False
        and ratio["entropy_moment_matching"] is False
        and ratio["class_density_ratio"] is False
        and model_audit["direction_fit"]["optimizer_success"]
        and not model_audit["target_rows_used_for_robust_scale_centroids_recent_reference_domain_fit_ratio_or_direction_fit"]
        and not model_audit["target_labels_used"]
        and not model_audit["entropy_balance_or_moment_tilting"]
        and not model_audit["up_down_class_density_ratio"]
        and not model_audit["source_time_quantile_transport"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 source-time density-ratio reweighted direction blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "reference_fraction_domain_l2_support_shrink_ratio_clip_direction_l2_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.SEED = SEED
    scaffold.choose_inner = choose_inner
    scaffold.ordinal_direction = source_time_density_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v144_model": "v157_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "density_ratio_prob", "v144_model": "v157_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V157 SOURCE-TIME DENSITY RATIO] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V157 train-only source-time density-ratio reweighted policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def model_contract(audits: list[dict[str, Any]]) -> bool:
    return all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["train_only_source_time_density_ratio"]
        and audit[side]["fixed_clipped_candidate_weights"]
        and audit[side]["density_ratio_audit"]["pooled_fit"]["optimizer_success"]
        and audit[side]["density_ratio_audit"]["direction_or_return_labels_used"] is False
        and audit[side]["density_ratio_audit"]["validation_or_outer_features_used"] is False
        and audit[side]["density_ratio_audit"]["entropy_moment_matching"] is False
        and audit[side]["density_ratio_audit"]["class_density_ratio"] is False
        and audit[side]["direction_fit"]["optimizer_success"]
        and not audit[side]["target_rows_used_for_robust_scale_centroids_recent_reference_domain_fit_ratio_or_direction_fit"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["entropy_balance_or_moment_tilting"]
        and not audit[side]["up_down_class_density_ratio"]
        and not audit[side]["source_time_quantile_transport"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V157 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    native = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "source_time_density_ratio_reweighted_contract_verified": model_contract(audits),
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V157 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "candidate_native_robustness_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V157 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V157 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V157 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V157").resolve(), "V157 output is outside controller research/staging/V157")
    require(out.exists() and out.is_dir(), "V157 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V157 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V157 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V157 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V157 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V157", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V157_DIAGNOSTIC_SOURCE_TIME_DENSITY_RATIO": evaluation["candidate_summary"],
                "V157_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V157", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V157", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V157_SOURCE_TIME_DENSITY_RATIO_REWEIGHTED_REPORT.json": {
            "version": "V157", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V157", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V157", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V157", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V157_SOURCE_TIME_DENSITY_RATIO_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V157_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V157 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V157 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V157 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "train-only pooled and supported-source old-versus-recent density-odds weights followed by one reweighted balanced direction head",
            "v85_entropy_balance_collision": False,
            "v117_class_density_ratio_collision": False,
            "v145_source_time_quantile_transport_collision": False,
            "v158_event_arrival_hazard_collision": False,
            "causal_numeric_only": True, "target_row_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V157 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS,
                "runtime": runtime, "market": "KR", "fold": 2,
                "strict_inner_chronology": audit["inner_chronology"],
                "strict_outer_chronology": audit["outer_chronology"],
                "raw_inner_selected": audit["policy"]["selected"],
                "raw_inner_best_non_noop": best_non_noop,
                "raw_outer_baseline": audit["outer_baseline"],
                "raw_outer_selected": audit["outer_candidate"],
                "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
                "inner_model_audit": audit["inner_model_audit"],
                "outer_model_audit": audit["outer_model_audit"],
                "selection_locked_before_outer_evaluation": True,
                "nested_bootstrap_executed": False,
                "exact_entire_v69_fallback_contract": True,
                "output_written": False,
            }), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "runtime": runtime,
        "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None,
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": material_gate(evaluation),
        "output_written": False,
    }
    if args.smoke_test:
        summary.update({
            "raw_inner_selected": audits[0]["policy"]["selected"],
            "raw_inner_best_non_noop": best_non_noop,
            "raw_outer_baseline": audits[0]["outer_baseline"],
            "raw_outer_selected": audits[0]["outer_candidate"],
            "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
            "inner_model_audit": audits[0]["inner_model_audit"],
            "outer_model_audit": audits[0]["outer_model_audit"],
            "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        })
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
