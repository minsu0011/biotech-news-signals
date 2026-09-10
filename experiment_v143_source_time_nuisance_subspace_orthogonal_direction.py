"""V143 source-time nuisance-subspace orthogonal direction challenger.

For every strict-past same-market cut, 37 immutable causal numeric fields plus
missing flags are robustly scaled and collapsed to equal event-group centroids.
Equal supported source-family by chronological-tercile means define a between-
environment state matrix.  Its deterministic top singular subspace (fixed rank
cap six) is treated as nuisance covariate shift and removed by exact orthogonal
projection before fitting one duplicate/class-balanced fixed ridge-logit head.
No target row is source-routed and no V69 probability, confidence, return,
worst-environment loss, coefficient ensemble, or sign gate enters the model.

Inner-past evidence selects exact V69 or fixed .25/.50 logit blends.  Outer
labels remain evaluation-only, confidence/high_conf stay byte-exact, and any
material failure restores the entire atomic V69 frame.  Audit/support/smoke use
CPU30-31, two threads, no GPU and no writes.  Full requires an exact controller
MARKET_BIO_VERSION_OUTPUT binding.
"""
from __future__ import annotations

import os

_thread_default = "32" if os.environ.get("MARKET_BIO_VERSION_OUTPUT") else "2"
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = _thread_default

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v140_source_market_time_random_effects_meta_logit_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V143" / "LOCAL_FORBIDDEN"

VERSION = 143
HYPOTHESIS = "SOURCE_TIME_NUISANCE_SUBSPACE_ORTHOGONAL_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v140_source_market_time_random_effects_meta_logit_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "b1f74121a34896c3a6523560c8fba653c3729c438ed03081337cc3142d3b4bac"

STATE_DIMENSION = 74
NUISANCE_RANK_CAP = 6
MIN_ENVIRONMENT_ROWS = 60
MIN_ENVIRONMENT_CLASS_ROWS = 15
RIDGE_L2 = 0.05
SCORE_CLIP = 20.0
SEED = 14301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 96
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 24
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "NUISANCE_ORTHOGONAL_LOGIT_W0.25", "weight": 0.25},
    {"name": "NUISANCE_ORTHOGONAL_LOGIT_W0.50", "weight": 0.50},
)
STATIC_SPEC = {
    "numeric_feature_n": 37,
    "missing_flag_n": 37,
    "state_dimension": STATE_DIMENSION,
    "duplicate_unit": "equal event-group robust-state centroid",
    "environment": "same-market source_family x chronological training tercile",
    "environment_measure": "equal-environment between-mean state matrix",
    "nuisance_subspace": "deterministic SVD top singular vectors",
    "fixed_rank_cap": NUISANCE_RANK_CAP,
    "representation": "exact orthogonal complement of nuisance subspace",
    "head": "one duplicate/class-balanced fixed ridge logistic",
    "head_L2": RIDGE_L2,
    "target_score": "one global source-agnostic affine direction",
    "probability": "direct sigmoid with no calibration or threshold tuning",
}

require = scaffold.require
sha256 = scaffold.sha256
clean = scaffold.clean
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
bool_series = scaffold.bool_series
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
chronology = scaffold.chronology
inner_partition = scaffold.inner_partition
metric = scaffold.metric
blend_probability = scaffold.blend_probability
array_sha256 = scaffold.array_sha256
numeric = scaffold.numeric
v44 = scaffold.v44
controller = scaffold.controller


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V140 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust state, environment metadata, fixed logistic optimizer, metrics, gates, bootstrap and atomic IO only; no V140 output is read",
    }
    audit["v143_access_contract"] = {
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


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    return scaffold.load_authorized()


def supported_environment_masks(
    target: np.ndarray,
    environment: np.ndarray,
) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    names: list[str] = []
    masks: list[np.ndarray] = []
    counts: dict[str, Any] = {}
    for name in sorted(pd.unique(environment)):
        mask = environment == name
        class_counts = [int(np.sum(target[mask] == direction)) for direction in (0, 1)]
        counts[str(name)] = {"rows": int(mask.sum()), "down_up_support": class_counts}
        if int(mask.sum()) >= MIN_ENVIRONMENT_ROWS and min(class_counts) >= MIN_ENVIRONMENT_CLASS_ROWS:
            names.append(str(name))
            masks.append(mask)
    require(len(names) >= 3, "V143 fewer than three supported source-time environments")
    return names, masks, {
        "raw_environment_n": int(pd.Series(environment).nunique()),
        "supported_environment_n": len(names),
        "supported_environments": names,
        "supported_source_family_n": len({name.split("|")[1] for name in names}),
        "environment_support": counts,
        "covered_event_group_fraction": float(np.mean(np.isin(environment, names))),
    }


def fit_nuisance_subspace(
    state: np.ndarray,
    target: np.ndarray,
    environment: np.ndarray,
    event_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(state.shape[1] == STATE_DIMENSION, "V143 state dimension changed")
    if event_weights is None:
        event_weights = np.ones(len(state), float)
    event_weights = np.asarray(event_weights, float)
    require(event_weights.shape == (len(state),) and np.isfinite(event_weights).all() and np.all(event_weights > 0.0), "V143 subspace weights invalid")
    names, masks, support = supported_environment_masks(target, environment)
    means = []
    for mask in masks:
        weight = event_weights[mask] / float(event_weights[mask].sum())
        means.append(weight @ state[mask])
    mean_matrix = np.vstack(means)
    equal_environment_center = mean_matrix.mean(axis=0)
    centered = mean_matrix - equal_environment_center
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    numerical_rank = int(np.sum(singular_values > max(float(singular_values[0]), 1.0) * 1e-10))
    rank = min(NUISANCE_RANK_CAP, len(names) - 1, numerical_rank)
    require(rank >= 1, "V143 nuisance subspace has zero rank")
    basis = right[:rank].T.copy()
    require(np.allclose(basis.T @ basis, np.eye(rank), atol=1e-8), "V143 nuisance basis not orthonormal")
    projected_environment_means = centered - (centered @ basis) @ basis.T
    total_energy = float(np.square(centered).sum())
    residual_energy = float(np.square(projected_environment_means).sum())
    return basis, {
        **support,
        "rank": rank,
        "rank_cap": NUISANCE_RANK_CAP,
        "singular_values": [float(value) for value in singular_values],
        "captured_between_environment_energy_fraction": float(1.0 - residual_energy / max(total_energy, 1e-12)),
        "residual_between_environment_energy_fraction": float(residual_energy / max(total_energy, 1e-12)),
        "basis_sha256": array_sha256(basis),
    }


def project_orthogonal(state: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projected = state - (state @ basis) @ basis.T
    require(projected.shape == state.shape and np.isfinite(projected).all(), "V143 projected state invalid")
    require(float(np.max(np.abs(projected @ basis))) <= 1e-7, "V143 projection not orthogonal")
    return projected


def fit_orthogonal_parameters(
    state: np.ndarray,
    target: np.ndarray,
    environment: np.ndarray,
    event_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    basis, subspace_audit = fit_nuisance_subspace(state, target, environment, event_weights)
    projected = project_orthogonal(state, basis)
    parameters, _, head_audit = scaffold.fit_environment_logit(projected, target, event_weights)
    coefficient = parameters[:STATE_DIMENSION]
    orthogonal_component = basis.T @ coefficient
    require(float(np.max(np.abs(orthogonal_component))) <= 1e-6, "V143 head escaped nuisance orthogonal complement")
    return parameters, basis, {
        "subspace": subspace_audit,
        "head": head_audit,
        "coefficient_l2": float(np.linalg.norm(coefficient)),
        "coefficient_nuisance_component_max_abs": float(np.max(np.abs(orthogonal_component))),
        "coefficient_sha256": array_sha256(coefficient),
        "intercept": float(parameters[-1]),
    }


def native_orthogonal_bootstrap(
    state: np.ndarray,
    target: np.ndarray,
    environment: np.ndarray,
    target_state: np.ndarray,
    nominal_probability: np.ndarray,
    nominal_parameters: np.ndarray,
    nominal_basis: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    probabilities = np.empty((len(target_state), draws), float)
    cosines = np.empty(draws, float)
    overlaps = np.empty(draws, float)
    nominal_coefficient = nominal_parameters[:STATE_DIMENSION]
    nominal_norm = max(float(np.linalg.norm(nominal_coefficient)), 1e-12)
    for draw in range(draws):
        weights = generator.exponential(size=len(target))
        parameters, basis, _ = fit_orthogonal_parameters(state, target, environment, weights)
        projected_target = project_orthogonal(target_state, basis)
        probabilities[:, draw] = 1.0 / (1.0 + np.exp(-np.clip(projected_target @ parameters[:STATE_DIMENSION] + parameters[-1], -SCORE_CLIP, SCORE_CLIP)))
        coefficient = parameters[:STATE_DIMENSION]
        current_norm = max(float(np.linalg.norm(coefficient)), 1e-12)
        cosines[draw] = float(coefficient @ nominal_coefficient / (current_norm * nominal_norm))
        common_rank = min(nominal_basis.shape[1], basis.shape[1])
        overlaps[draw] = float(np.square(nominal_basis.T @ basis).sum() / max(common_rank, 1))
    median_probability = np.median(probabilities, axis=1)
    row_std = probabilities.std(axis=1)
    return {
        "contract": "event-group Bayesian complete refit of source-time nuisance SVD subspace and orthogonal balanced direction head",
        "draws": draws,
        "seed": seed,
        "event_group_unit_weights": True,
        "nuisance_subspace_completely_refit": True,
        "direction_head_completely_refit": True,
        "target_labels_used": False,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probabilities - nominal_probability[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal_probability))),
        "direction_agreement_with_nominal": float(np.mean((median_probability >= 0.5) == (nominal_probability >= 0.5))),
        "coefficient_cosine_q10_q50_q90": [float(np.quantile(cosines, q)) for q in (0.1, 0.5, 0.9)],
        "subspace_overlap_q10_q50_q90": [float(np.quantile(overlaps, q)) for q in (0.1, 0.5, 0.9)],
        "row_probability_std_q10_q50_q90": [float(np.quantile(row_std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median_probability),
    }


def orthogonal_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V143 expects one market")
    require(str(ordered.market.iloc[0]) == str(target_frame.market.iloc[0]), "V143 market mismatch")
    processor = numeric.RobustNumericState().fit(ordered)
    train_rows, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_state, train_y, group_ids, environment, environment_audit = scaffold.group_state_and_environment(ordered, train_rows)
    parameters, basis, fit_audit = fit_orthogonal_parameters(train_state, train_y, environment)
    projected_target = project_orthogonal(target_state, basis)
    score = np.clip(projected_target @ parameters[:STATE_DIMENSION] + parameters[-1], -SCORE_CLIP, SCORE_CLIP)
    probability = np.clip(1.0 / (1.0 + np.exp(-score)), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V143 probability invalid")
    native = None if native_draws <= 0 else native_orthogonal_bootstrap(
        train_state, train_y, environment, target_state, probability, parameters, basis, native_draws, native_seed,
    )
    return probability, {
        "market": str(ordered.market.iloc[0]),
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "train_event_group_n": len(group_ids),
        "causal_numeric_and_missing_only": True,
        "numeric_feature_n": len(FEATURES),
        "missing_flag_n": len(FEATURES),
        "source_used_only_for_training_nuisance_environment": True,
        "source_market_time_target_routing": False,
        "categorical_text_ticker_issuer_return_v69_probability_or_confidence_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "environment_audit": environment_audit,
        "static_spec": STATIC_SPEC,
        "fit_audit": fit_audit,
        "equal_environment_between_mean_svd": True,
        "exact_nuisance_orthogonal_projection": True,
        "single_global_balanced_ridge_head": True,
        "environment_heads_meta_group_dro_maximin_or_cvar": False,
        "delete_block_sign_intersection_or_coefficient_ensemble": False,
        "v69_offset_residual_or_confidence_surface": False,
        "spatial_median_tyler_scatter_or_class_density": False,
        "target_rows_used_for_scale_subspace_head_or_selection": False,
        "target_labels_used": False,
        "rank_l2_threshold_calibration_or_blend_grid_tuned": False,
        "native_orthogonal_bootstrap": native,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "predicted_up_rate": float(np.mean(probability >= 0.5)),
            "score_q10_q50_q90": [float(np.quantile(score, q)) for q in (0.1, 0.5, 0.9)],
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = orthogonal_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline_metric,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"],
    }]
    fit = model_audit["fit_audit"]
    model_eligible = bool(
        model_audit["causal_numeric_and_missing_only"]
        and model_audit["source_used_only_for_training_nuisance_environment"]
        and not model_audit["source_market_time_target_routing"]
        and model_audit["equal_environment_between_mean_svd"]
        and model_audit["exact_nuisance_orthogonal_projection"]
        and model_audit["single_global_balanced_ridge_head"]
        and fit["subspace"]["supported_environment_n"] >= 3
        and fit["subspace"]["covered_event_group_fraction"] >= 0.80
        and fit["subspace"]["rank"] >= 1
        and fit["head"]["optimizer_success"]
        and fit["coefficient_nuisance_component_max_abs"] <= 1e-6
        and not model_audit["environment_heads_meta_group_dro_maximin_or_cvar"]
        and not model_audit["delete_block_sign_intersection_or_coefficient_ensemble"]
        and not model_audit["v69_offset_residual_or_confidence_surface"]
        and not model_audit["target_rows_used_for_scale_subspace_head_or_selection"]
        and not model_audit["target_labels_used"]
        and not model_audit["rank_l2_threshold_calibration_or_blend_grid_tuned"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"],
            "architecture": architecture,
            "metrics": current,
            "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"] + 10.0 * current["all_trade_mean_signed_net"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 source-time nuisance-orthogonal direction blend; fixed 2*AUC+BA+10*net with BA/net guards",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "environment_rank_projection_l2_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.SMOKE_NATIVE_BOOTSTRAP_DRAWS = SMOKE_NATIVE_BOOTSTRAP_DRAWS
    scaffold.NATIVE_BOOTSTRAP_DRAWS = NATIVE_BOOTSTRAP_DRAWS
    scaffold.SEED = SEED
    scaffold.choose_inner = choose_inner
    scaffold.meta_direction = orthogonal_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke)
    diagnostic = diagnostic.rename(columns={"v140_model": "v143_model"})
    evidence = evidence.rename(columns={"meta_logit_prob": "nuisance_orthogonal_prob", "v140_model": "v143_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V143 NUISANCE ORTHOGONAL] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V143 nuisance-orthogonal policy"
    result["seed"] = SEED
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V143 canonical mismatch")
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
    orthogonal_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["source_used_only_for_training_nuisance_environment"]
        and not audit[side]["source_market_time_target_routing"]
        and audit[side]["equal_environment_between_mean_svd"]
        and audit[side]["exact_nuisance_orthogonal_projection"]
        and audit[side]["single_global_balanced_ridge_head"]
        and audit[side]["fit_audit"]["subspace"]["supported_environment_n"] >= 3
        and audit[side]["fit_audit"]["subspace"]["covered_event_group_fraction"] >= 0.80
        and audit[side]["fit_audit"]["head"]["optimizer_success"]
        and audit[side]["fit_audit"]["coefficient_nuisance_component_max_abs"] <= 1e-6
        and not audit[side]["environment_heads_meta_group_dro_maximin_or_cvar"]
        and not audit[side]["delete_block_sign_intersection_or_coefficient_ensemble"]
        and not audit[side]["v69_offset_residual_or_confidence_surface"]
        and not audit[side]["target_rows_used_for_scale_subspace_head_or_selection"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["rank_l2_threshold_calibration_or_blend_grid_tuned"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    model_native = [audit["outer_model_audit"]["native_orthogonal_bootstrap"] for audit in audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"]
        and item["nuisance_subspace_completely_refit"]
        and item["direction_head_completely_refit"]
        and not item["target_labels_used"]
        and 0.0 <= item["direction_agreement_with_nominal"] <= 1.0
        for item in model_native
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
        "orthogonal_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "source_time_nuisance_orthogonal_contract_verified": orthogonal_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V143 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": baseline,
        "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested,
        "candidate_native_robustness_bootstrap": native,
        "model_native_orthogonal_bootstrap": model_native,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "model_native_orthogonal_bootstrap": evaluation["model_native_orthogonal_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V143 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V143 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V143 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V143").resolve(), "V143 output is outside controller research/staging/V143")
    require(out.exists() and out.is_dir(), "V143 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V143 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V143 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True,
        "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V143 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V143 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V143", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V143_DIAGNOSTIC_NUISANCE_ORTHOGONAL": evaluation["candidate_summary"],
                "V143_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V143", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V143", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "nuisance_subspace_audits": [
                {"market": audit["market"], "fold": audit["fold"], "inner": audit["inner_model_audit"]["fit_audit"]["subspace"], "outer": audit["outer_model_audit"]["fit_audit"]["subspace"]}
                for audit in audits
            ],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "V143_NUISANCE_SUBSPACE_ORTHOGONAL_REPORT.json": {
            "version": "V143", "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V143", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V143", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
            "model_native_orthogonal_bootstrap": evaluation["model_native_orthogonal_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V143", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
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
    atomic_csv(evidence, out / "V143_NUISANCE_ORTHOGONAL_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V143_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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


def support_probe_kr2(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    outer_champion = champion.loc[
        champion.market.eq("KR") & champion.fold.eq(2)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS["KR"], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_train = prior_train(dev, "KR", start, outer_valid)
    outer_chronology = chronology(outer_train, outer_valid, "V143 KR2 support outer")
    inner_train, inner_valid, _, inner_chronology = inner_partition(dev, champion, "KR", start)
    inner_probability, inner_audit = orthogonal_direction(inner_train, inner_valid)
    outer_probability, outer_audit = orthogonal_direction(outer_train, outer_valid)
    require(np.isfinite(inner_probability).all() and np.isfinite(outer_probability).all(), "V143 KR2 support probability invalid")
    return {
        "status": "KR2_SUPPORT_OK_NO_OUTER_EVALUATION",
        "inner_target_n": len(inner_valid), "outer_target_n": len(outer_valid),
        "inner_train_event_group_n": inner_audit["train_event_group_n"],
        "outer_train_event_group_n": outer_audit["train_event_group_n"],
        "inner_supported_environment_n": inner_audit["fit_audit"]["subspace"]["supported_environment_n"],
        "outer_supported_environment_n": outer_audit["fit_audit"]["subspace"]["supported_environment_n"],
        "inner_subspace_rank": inner_audit["fit_audit"]["subspace"]["rank"],
        "outer_subspace_rank": outer_audit["fit_audit"]["subspace"]["rank"],
        "inner_captured_environment_energy": inner_audit["fit_audit"]["subspace"]["captured_between_environment_energy_fraction"],
        "outer_captured_environment_energy": outer_audit["fit_audit"]["subspace"]["captured_between_environment_energy_fraction"],
        "inner_coverage": inner_audit["fit_audit"]["subspace"]["covered_event_group_fraction"],
        "outer_coverage": outer_audit["fit_audit"]["subspace"]["covered_event_group_fraction"],
        "inner_probability_sha256": array_sha256(inner_probability),
        "outer_probability_sha256": array_sha256(outer_probability),
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used": False,
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
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe_kr2
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V143 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V143 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
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
            "architecture": "source-time between-environment mean SVD nuisance removal followed by one global orthogonal balanced direction head",
            "causal_numeric_and_missing_only": True,
            "source_used_only_for_training_nuisance_environment": True,
            "target_rows_or_labels_used": False,
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
                "controller_native_robustness_bootstrap": True,
                "model_native_orthogonal_bootstrap": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe_kr2(dev, champion)
            result["runtime"] = runtime
            result["gpu_model_calls"] = 0
            result["output_written"] = False
            print(json.dumps(clean(result), ensure_ascii=False, indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
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
        "folds_executed": len(audits),
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
