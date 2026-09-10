"""V152 source-deviation hierarchical group-lasso direction challenger.

Each strict-past same-market cut robustly scales the immutable 37 causal
numeric fields plus deterministic missing flags and collapses duplicate rows
to equal event-group state centroids.  A single convex logistic model jointly
fits one global direction and deviations for strictly supported source
families.  Exact proximal groups couple each feature's global and source
deviation coefficients, while a fixed L2 penalty shrinks source deviations.
Targets use a deviation only when that source had fixed minimum past support;
otherwise prediction is the global direction.

This differs from independently fit/time-decayed source experts, source-local
additive heads, and random-effects meta-analysis of environment heads.  Inner
labels choose exact V69 or fixed .25/.50 blends; outer labels are evaluation
only under a strict 35-minute embargo/event purge.  V69 confidence/high_conf
remain exact and every material or execution failure restores the entire
atomic V69 frame.  Prep is CPU30-31, two-thread, no-GPU and no-write.
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
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import experiment_v149_multichannel_causal_increment_coskewness_tensor_direction as utilities


ROOT = utilities.ROOT
V69_DIR = utilities.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V152" / "LOCAL_FORBIDDEN"
VERSION = 152
HYPOTHESIS = "SOURCE_DEVIATION_HIERARCHICAL_GROUP_LASSO_DIRECTION_V1"
FEATURES = utilities.FEATURES
STATE_DIMENSION = 2 * len(FEATURES)
EXPECTED_V69_EXPERIMENT_ID = utilities.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v149_multichannel_causal_increment_coskewness_tensor_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "4b9a364db95f598ebb75ca3406333412f1263ac1bb0722f68be81f7e17ab2fe0"

GROUP_LASSO_LAMBDA = 0.010
SOURCE_DEVIATION_L2 = 0.10
MIN_SOURCE_EVENT_GROUPS = 60
MIN_SOURCE_CLASS_EVENT_GROUPS = 15
OPTIMIZER_ITERATIONS = 600
POWER_ITERATIONS = 64
SCORE_CLIP = 20.0
SEED = 15201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "SOURCE_DEVIATION_GROUP_LASSO_W0.25", "weight": 0.25},
    {"name": "SOURCE_DEVIATION_GROUP_LASSO_W0.50", "weight": 0.50},
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
blend_probability = utilities.base.blend_probability
controller = utilities.controller
v44 = utilities.v44
numeric = utilities.numeric
scaffold = utilities.base.scaffold

STATIC_SPEC = {
    "causal_numeric_feature_n": len(FEATURES),
    "deterministic_missing_flag_n": len(FEATURES),
    "state_dimension": STATE_DIMENSION,
    "event_unit": "equal event-group robust-state centroid",
    "model": "joint global plus supported-source deviation affine logistic",
    "structured_penalty": "per-feature exact proximal group lasso across global and all supported-source deviations",
    "group_lasso_lambda": GROUP_LASSO_LAMBDA,
    "source_deviation_l2": SOURCE_DEVIATION_L2,
    "optimizer": "deterministic fixed-step proximal gradient with exact group proximal map",
    "optimizer_iterations": OPTIMIZER_ITERATIONS,
    "source_support": {
        "minimum_event_groups": MIN_SOURCE_EVENT_GROUPS,
        "minimum_each_class_event_groups": MIN_SOURCE_CLASS_EVENT_GROUPS,
    },
    "target_routing": "supported strict-past source deviation else global direction",
    "head_or_penalty_grid": False,
}
require(STATE_DIMENSION == 74, "V152 state dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V149 utility scaffold changed")
    audit = utilities.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust causal state, metrics, canonical gate, fixed blending, bootstrap and atomic IO only; no V149 output is read",
    }
    audit["v152_access_contract"] = {
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


def group_state_source(
    frame: pd.DataFrame,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape == (len(frame), STATE_DIMENSION), "V152 robust state shape changed")
    work = pd.DataFrame(state)
    work["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    work["y"] = frame.y.to_numpy(int)
    work["source_family"] = frame.source_family.fillna("UNKNOWN").astype(str).to_numpy()
    work["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True).to_numpy()
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V152 event group crosses direction labels")
    require(int(grouped.source_family.nunique().max()) == 1, "V152 event group crosses sources")
    state_group = grouped[list(range(STATE_DIMENSION))].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    source = grouped.source_family.first().to_numpy(str)
    group_id = grouped.size().index.to_numpy(str)
    event_time = pd.to_datetime(grouped.event_time_utc.first(), utc=True)
    order = np.lexsort((group_id, event_time.astype("int64").to_numpy()))
    state_group = state_group[order]
    target = target[order]
    source = source[order]
    group_id = group_id[order]
    require(np.isfinite(state_group).all(), "V152 event-group state invalid")
    counts = pd.crosstab(pd.Series(source, name="source"), pd.Series(target, name="y"))
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    return state_group, target, source, group_id, {
        "event_group_n": len(group_id),
        "source_family_n": int(pd.Series(source).nunique()),
        "source_class_event_group_counts": {
            str(name): [int(row[0]), int(row[1])] for name, row in counts.iterrows()
        },
        "group_id_sha256": array_sha256(np.asarray(group_id, dtype="U")),
    }


def balanced_event_weights(target: np.ndarray) -> np.ndarray:
    weight = np.zeros(len(target), float)
    for direction in (0, 1):
        mask = target == direction
        require(int(mask.sum()) >= MIN_SOURCE_CLASS_EVENT_GROUPS, "V152 market class support too small")
        weight[mask] = 0.5 / float(mask.sum())
    weight *= len(target)
    return weight


def supported_sources(source: np.ndarray, target: np.ndarray) -> tuple[str, ...]:
    supported: list[str] = []
    for name in sorted(pd.unique(source)):
        mask = source == name
        counts = [int(np.sum(target[mask] == direction)) for direction in (0, 1)]
        if int(mask.sum()) >= MIN_SOURCE_EVENT_GROUPS and min(counts) >= MIN_SOURCE_CLASS_EVENT_GROUPS:
            supported.append(str(name))
    return tuple(supported)


def expanded_design(state: np.ndarray, source: np.ndarray, supported: tuple[str, ...]) -> np.ndarray:
    block_dimension = STATE_DIMENSION + 1
    design = np.zeros((len(state), (1 + len(supported)) * block_dimension), float)
    design[:, :STATE_DIMENSION] = state
    design[:, STATE_DIMENSION] = 1.0
    for index, name in enumerate(supported, start=1):
        mask = source == name
        start = index * block_dimension
        design[mask, start:start + STATE_DIMENSION] = state[mask]
        design[mask, start + STATE_DIMENSION] = 1.0
    require(np.isfinite(design).all(), "V152 expanded design invalid")
    return design


def spectral_lipschitz(design: np.ndarray, weight: np.ndarray) -> float:
    weighted = design * np.sqrt(weight / float(weight.sum()))[:, None]
    vector = np.linspace(1.0, 2.0, weighted.shape[1], dtype=float)
    vector /= np.linalg.norm(vector)
    eigenvalue = 0.0
    for _ in range(POWER_ITERATIONS):
        candidate = weighted.T @ (weighted @ vector)
        eigenvalue = float(np.linalg.norm(candidate))
        require(np.isfinite(eigenvalue), "V152 power iteration invalid")
        if eigenvalue <= 1e-14:
            return SOURCE_DEVIATION_L2 + 1.0
        vector = candidate / eigenvalue
    rayleigh = float(np.sum(np.square(weighted @ vector)))
    return 1.05 * 0.25 * rayleigh + SOURCE_DEVIATION_L2 + 1e-8


def group_proximal(parameters: np.ndarray, step: float, block_n: int) -> np.ndarray:
    block_dimension = STATE_DIMENSION + 1
    matrix = parameters.reshape(block_n, block_dimension).copy()
    coefficient = matrix[:, :STATE_DIMENSION]
    norms = np.linalg.norm(coefficient, axis=0)
    shrink = np.maximum(0.0, 1.0 - step * GROUP_LASSO_LAMBDA / np.maximum(norms, 1e-15))
    matrix[:, :STATE_DIMENSION] = coefficient * shrink[None, :]
    return matrix.ravel()


def fit_hierarchical_group_lasso(
    state: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    supported = supported_sources(source, target)
    design = expanded_design(state, source, supported)
    weight = balanced_event_weights(target)
    total_weight = float(weight.sum())
    block_n = 1 + len(supported)
    block_dimension = STATE_DIMENSION + 1
    step = 1.0 / spectral_lipschitz(design, weight)
    parameters = np.zeros(design.shape[1], float)
    accelerated = parameters.copy()
    acceleration = 1.0
    for _ in range(OPTIMIZER_ITERATIONS):
        score = np.clip(design @ accelerated, -SCORE_CLIP, SCORE_CLIP)
        residual = weight * (expit(score) - target) / total_weight
        gradient = design.T @ residual
        matrix = accelerated.reshape(block_n, block_dimension)
        gradient_matrix = gradient.reshape(block_n, block_dimension)
        if block_n > 1:
            gradient_matrix[1:, :] += SOURCE_DEVIATION_L2 * matrix[1:, :]
        candidate = group_proximal(accelerated - step * gradient, step, block_n)
        next_acceleration = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration * acceleration))
        accelerated = candidate + ((acceleration - 1.0) / next_acceleration) * (candidate - parameters)
        parameters = candidate
        acceleration = next_acceleration
    require(np.isfinite(parameters).all(), "V152 proximal optimizer returned invalid parameters")
    final_score = np.clip(design @ parameters, -SCORE_CLIP, SCORE_CLIP)
    logistic_loss = float(np.sum(weight * (np.logaddexp(0.0, final_score) - target * final_score)) / total_weight)
    matrix = parameters.reshape(block_n, block_dimension)
    group_norms = np.linalg.norm(matrix[:, :STATE_DIMENSION], axis=0)
    group_penalty = float(GROUP_LASSO_LAMBDA * group_norms.sum())
    deviation_penalty = float(0.5 * SOURCE_DEVIATION_L2 * np.sum(np.square(matrix[1:, :])))
    gradient = design.T @ (weight * (expit(final_score) - target) / total_weight)
    if block_n > 1:
        gradient.reshape(block_n, block_dimension)[1:, :] += SOURCE_DEVIATION_L2 * matrix[1:, :]
    proximal_residual = float(np.linalg.norm(parameters - group_proximal(parameters - step * gradient, step, block_n)) / step)
    return parameters, supported, {
        "supported_source_families": list(supported),
        "supported_source_n": len(supported),
        "parameter_n": len(parameters),
        "block_n": block_n,
        "optimizer": "fixed-step FISTA exact feature-group proximal map",
        "iterations": OPTIMIZER_ITERATIONS,
        "step": step,
        "logistic_loss": logistic_loss,
        "group_penalty": group_penalty,
        "source_deviation_penalty": deviation_penalty,
        "objective": logistic_loss + group_penalty + deviation_penalty,
        "active_feature_group_n": int(np.sum(group_norms > 1e-10)),
        "active_feature_group_fraction": float(np.mean(group_norms > 1e-10)),
        "global_coefficient_l2": float(np.linalg.norm(matrix[0, :STATE_DIMENSION])),
        "source_deviation_l2": {name: float(np.linalg.norm(matrix[index + 1])) for index, name in enumerate(supported)},
        "proximal_residual_l2": proximal_residual,
        "parameter_sha256": array_sha256(parameters),
        "joint_single_objective": True,
        "exact_group_proximal_map": True,
    }


def routed_probability(
    state: np.ndarray,
    source: np.ndarray,
    parameters: np.ndarray,
    supported: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    design = expanded_design(state, source, supported)
    probability = np.clip(expit(np.clip(design @ parameters, -SCORE_CLIP, SCORE_CLIP)), 1e-6, 1.0 - 1e-6)
    supported_mask = np.isin(source, supported)
    return probability, {
        "rows": len(state),
        "supported_source_routed_n": int(supported_mask.sum()),
        "global_fallback_n": int((~supported_mask).sum()),
        "supported_source_routed_fraction": float(np.mean(supported_mask)),
        "source_route_counts": pd.Series(source[supported_mask]).value_counts().sort_index().astype(int).to_dict(),
        "global_fallback_sources": sorted(pd.unique(source[~supported_mask]).tolist()),
        "probability_sha256": array_sha256(probability),
    }


def source_deviation_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V152 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    group_state, group_target, group_source, group_ids, group_audit = group_state_source(ordered, train_state)
    parameters, supported, fit_audit = fit_hierarchical_group_lasso(group_state, group_target, group_source)
    target_source = target_frame.source_family.fillna("UNKNOWN").astype(str).to_numpy()
    probability, route_audit = routed_probability(target_state, target_source, parameters, supported)
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_ids), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_ticker_or_issuer_feature_n": 0,
        "source_family_role": "supported strict-past deviation routing only",
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC, "group_audit": group_audit,
        "fit_audit": fit_audit, "route_audit": route_audit,
        "target_rows_used_for_robust_scale_support_fit_penalty_or_selection": False,
        "target_labels_used": False,
        "joint_global_source_deviation_objective": True,
        "feature_group_penalty_verified": True,
        "independent_source_experts": False,
        "source_time_random_effects_meta": False,
        "source_additive_shape_heads": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def supported_source_ba_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, dict[str, float]]:
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
    challenger, model_audit = source_deviation_direction(inner_train, inner_valid)
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
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["static_spec"]["state_dimension"] == STATE_DIMENSION
        and model_audit["joint_global_source_deviation_objective"]
        and model_audit["feature_group_penalty_verified"]
        and model_audit["fit_audit"]["joint_single_objective"]
        and model_audit["fit_audit"]["exact_group_proximal_map"]
        and np.isfinite(model_audit["fit_audit"]["proximal_residual_l2"])
        and not model_audit["target_rows_used_for_robust_scale_support_fit_penalty_or_selection"]
        and not model_audit["target_labels_used"]
        and not model_audit["independent_source_experts"]
        and not model_audit["source_time_random_effects_meta"]
        and not model_audit["source_additive_shape_heads"]
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
        "selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 hierarchical source-deviation group-lasso blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "support_penalty_optimizer_blend_or_source_floor_micro_tuning": False,
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
    scaffold.ordinal_direction = source_deviation_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v144_model": "v152_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "source_deviation_prob", "v144_model": "v152_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V152 HIERARCHICAL GROUP LASSO] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V152 hierarchical source-deviation group-lasso policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


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
    require(candidate_summary["research_gate"] == canonical_candidate, "V152 canonical mismatch")
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
    model_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["state_dimension"] == STATE_DIMENSION
        and audit[side]["joint_global_source_deviation_objective"]
        and audit[side]["feature_group_penalty_verified"]
        and audit[side]["fit_audit"]["joint_single_objective"]
        and audit[side]["fit_audit"]["exact_group_proximal_map"]
        and audit[side]["route_audit"]["rows"] == audit[side]["target_n"]
        and not audit[side]["target_rows_used_for_robust_scale_support_fit_penalty_or_selection"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["independent_source_experts"]
        and not audit[side]["source_time_random_effects_meta"]
        and not audit[side]["source_additive_shape_heads"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
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
        "source_deviation_hierarchical_group_lasso_contract_verified": model_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V152 fallback not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V152 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V152 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V152 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V152").resolve(), "V152 output is outside controller research/staging/V152")
    require(out.exists() and out.is_dir(), "V152 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V152 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V152 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
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
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V152 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V152 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V152", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V152_DIAGNOSTIC_SOURCE_DEVIATION_GROUP_LASSO": evaluation["candidate_summary"],
                "V152_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V152", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V152", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V152_SOURCE_DEVIATION_HIERARCHICAL_GROUP_LASSO_REPORT.json": {
            "version": "V152", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V152", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V152", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V152", "status": "MATCH", "canonical_check_count": 14,
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
    atomic_csv(evidence, out / "V152_SOURCE_DEVIATION_GROUP_LASSO_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V152_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V152 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V152 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V152 controller full-run output mismatch")
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
            "architecture": "one joint convex global plus supported-source deviation logistic with exact per-feature group proximal map and fixed deviation L2",
            "prior_source_expert_group_lasso_collision_found": False,
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
            require(args.smoke_market == "KR", "V152 support probe is reserved for KR:2")
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
