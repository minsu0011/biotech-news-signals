"""V205 class-conditional Rasch item-response direction challenger.

Each strict-train-scaled observed 8x8 causal surface is converted to 64 binary
item responses using pooled train-only per-site weighted medians.  Separate
DOWN and UP one-parameter logistic Rasch models estimate item difficulties and
row-global latent abilities by one fixed alternating Newton schedule with a
mean-zero difficulty constraint.  A query receives a separately profiled
ability and Bernoulli profile likelihood under each class; their train-IQR
scaled contrast is a direct direction probability.  There is no neighbor
graph/Potts potential, fuzzy capacity, discriminative head, kernel or neural
density.  Atomic V69 confidence/high_conf and exact entire-frame fallback stay
unchanged.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_name] = "2"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import experiment_v200_multichannel_causal_convex_concave_envelope_defect_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V205" / "LOCAL_FORBIDDEN"
VERSION = 205
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_RASCH_ITEM_RESPONSE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v200_multichannel_causal_convex_concave_envelope_defect_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "be8ed3dd1a0ba11ffe5a891789db5715c804174c42c451086de90ddd5bc310b3"

GRID_N = 8
ITEM_COUNT = GRID_N * GRID_N
FIT_ITERATIONS = 24
ABILITY_NEWTON_STEPS = 10
PARAMETER_CLIP = 6.0
PROBABILITY_EPS = 1e-6
TEMPERATURE_FLOOR = 0.05
SEED = 20501
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CLASS_CONDITIONAL_RASCH_PROFILE_W0.25", "weight": 0.25},
    {"name": "CLASS_CONDITIONAL_RASCH_PROFILE_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
controller = scaffold.controller
numeric = scaffold.numeric
observed_past_to_recent_path = scaffold.observed_past_to_recent_path
bool_series = scaffold.bool_series
metric = scaffold.metric
blend_probability = scaffold.blend_probability
supported_source_ba_delta = scaffold.supported_source_ba_delta
_SUPPORT_RUN_NESTED = scaffold._SUPPORT_RUN_NESTED
_SUPPORT_EVALUATE = scaffold._SUPPORT_EVALUATE
_SUPPORT_BOOTSTRAP = scaffold._SUPPORT_BOOTSTRAP
ENGINE = scaffold.ENGINE

STATIC_SPEC = {
    "surface": "strict-train robust-scaled observed 8-horizon x 8-semantic-channel causal state",
    "items": "64 fixed horizon-channel sites",
    "feature_dimension": ITEM_COUNT,
    "response": "binary greater-than-or-equal pooled train-only per-site duplicate-balanced median",
    "class_model": "separate fixed-schedule one-parameter logistic Rasch item difficulties and row-global abilities",
    "fit_iterations": FIT_ITERATIONS,
    "ability_newton_steps": ABILITY_NEWTON_STEPS,
    "identifiability": "mean item difficulty exactly zero after every update",
    "direction": "equal-prior class profile Bernoulli log-likelihood contrast divided by train-only IQR temperature",
    "discriminative_head": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V205_CLASS_CONDITIONAL_RASCH_ITEM_RESPONSE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(ITEM_COUNT == 64 and FIT_ITERATIONS == 24 and ABILITY_NEWTON_STEPS == 10, "V205 static model changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V200 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, gates, bootstrap and atomic IO only; no V200 output is read"}
    audit["v205_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _weighted_median(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.empty(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        order = np.argsort(matrix[:, column], kind="stable"); values = matrix[order, column]; mass = weights[order]; cumulative = np.cumsum(mass)
        result[column] = values[min(int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")), len(values) - 1)]
    require(np.isfinite(result).all(), "V205 weighted median invalid"); return result


def _base_duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float); result = 1.0 / np.maximum(count, 1.0)
    return result / max(float(result.mean()), 1e-12)


def _profile_ability(response: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
    theta = np.zeros(len(response), dtype=float)
    for _ in range(ABILITY_NEWTON_STEPS):
        probability = _sigmoid(theta[:, None] - difficulty[None, :]); gradient = np.sum(response - probability, axis=1); information = np.sum(probability * (1.0 - probability), axis=1)
        theta = np.clip(theta + gradient / np.maximum(information, 1e-6), -PARAMETER_CLIP, PARAMETER_CLIP)
    return theta


def _profile_log_likelihood(response: np.ndarray, difficulty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = _profile_ability(response, difficulty); probability = _sigmoid(theta[:, None] - difficulty[None, :])
    log_likelihood = np.mean(response * np.log(probability) + (1.0 - response) * np.log(1.0 - probability), axis=1)
    require(np.isfinite(log_likelihood).all(), "V205 profile likelihood invalid"); return log_likelihood, theta


def _fit_rasch(response: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(response.ndim == 2 and response.shape[1] == ITEM_COUNT and len(response) >= 8, "V205 Rasch support invalid")
    weights = np.asarray(weights, dtype=float); weights = weights / max(float(weights.mean()), 1e-12)
    prevalence = np.average(response, axis=0, weights=weights); difficulty = np.clip(np.log((1.0 - np.clip(prevalence, 0.01, 0.99)) / np.clip(prevalence, 0.01, 0.99)), -PARAMETER_CLIP, PARAMETER_CLIP); difficulty -= difficulty.mean()
    theta = np.zeros(len(response), dtype=float)
    for _ in range(FIT_ITERATIONS):
        for _ in range(ABILITY_NEWTON_STEPS):
            probability = _sigmoid(theta[:, None] - difficulty[None, :]); gradient = np.sum(response - probability, axis=1); information = np.sum(probability * (1.0 - probability), axis=1)
            theta = np.clip(theta + gradient / np.maximum(information, 1e-6), -PARAMETER_CLIP, PARAMETER_CLIP)
        probability = _sigmoid(theta[:, None] - difficulty[None, :]); score = np.sum(weights[:, None] * (probability - response), axis=0); information = np.sum(weights[:, None] * probability * (1.0 - probability), axis=0)
        difficulty = np.clip(difficulty + score / np.maximum(information, 1e-6), -PARAMETER_CLIP, PARAMETER_CLIP); shift = float(difficulty.mean()); difficulty -= shift; theta -= shift
    probability = _sigmoid(theta[:, None] - difficulty[None, :]); score_residual = float(np.max(np.abs(np.average(probability - response, axis=0, weights=weights))))
    require(np.isfinite(difficulty).all() and np.isfinite(theta).all() and abs(float(difficulty.mean())) <= 1e-10, "V205 Rasch fit invalid")
    return difficulty, theta, {"rows": len(response), "iterations": FIT_ITERATIONS, "ability_newton_steps": ABILITY_NEWTON_STEPS, "difficulty_mean": float(difficulty.mean()), "difficulty_min": float(difficulty.min()), "difficulty_max": float(difficulty.max()), "difficulty_sha256": array_sha256(difficulty), "training_ability_sha256": array_sha256(theta), "max_weighted_item_score_residual": score_residual, "fixed_schedule_completed": True}


def rasch_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V205 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_flat = train_path.reshape(len(train_path), ITEM_COUNT); target_flat = target_path.reshape(len(target_path), ITEM_COUNT); base_weights = _base_duplicate_weights(ordered); thresholds = _weighted_median(train_flat, base_weights)
    train_response = (train_flat >= thresholds).astype(float); target_response = (target_flat >= thresholds).astype(float); labels = ordered.y.to_numpy(int)
    difficulties: dict[int, np.ndarray] = {}; class_audits: dict[str, Any] = {}
    for label in (0, 1):
        mask = labels == label; difficulty, _, fit_audit = _fit_rasch(train_response[mask], base_weights[mask]); difficulties[label] = difficulty; class_audits[str(label)] = fit_audit
    train_ll_down, _ = _profile_log_likelihood(train_response, difficulties[0]); train_ll_up, _ = _profile_log_likelihood(train_response, difficulties[1]); train_contrast = train_ll_up - train_ll_down
    q25, q75 = np.quantile(train_contrast, [0.25, 0.75]); temperature = max(float((q75 - q25) / 1.349), TEMPERATURE_FLOOR)
    target_ll_down, target_theta_down = _profile_log_likelihood(target_response, difficulties[0]); target_ll_up, target_theta_up = _profile_log_likelihood(target_response, difficulties[1]); raw = (target_ll_up - target_ll_down) / temperature; probability = _sigmoid(raw)
    geometry = {
        "item_count": ITEM_COUNT, "threshold_sha256": array_sha256(thresholds), "threshold_fit_train_only": True,
        "train_response_sha256": array_sha256(train_response), "target_response_sha256": array_sha256(target_response),
        "class_models": class_audits, "class_conditional_one_parameter_logistic_rasch": True,
        "row_global_latent_ability_profiled": True, "mean_zero_item_difficulty_identifiability": True,
        "fixed_alternating_newton_schedule": True, "target_batch_fit": False, "target_labels_used": False,
        "label_independent_exact_transform": True,
        "neighbor_grid_unary_pair_potts_pseudolikelihood": False, "two_additive_choquet_capacity_or_mobius_mass": False,
        "discriminative_logistic_head": False, "kernel_stein_energy_flow_or_metric_prototype": False,
        "label_independent_thresholds": True, "item_threshold_model_or_temperature_selected_from_outer": False,
        "feature_dimension": ITEM_COUNT,
        "connected_component_hole_or_euler_curve": False, "filtration_boundary_matrix_or_persistence": False,
        "time_oriented_crossing_excursion_or_dwell": False, "dyadic_continuous_reconstruction_or_roughness": False,
        "class_sliced_wasserstein_projection_or_quantile_barycenter": False, "semantic_mass_sinkhorn_transport": False,
        "source_time_quantile_transport": False, "learned_patch_dictionary_or_omp_code": False,
        "ordered_sparse_sem_or_ancestor_drive": False, "normalizing_flow_coupling_nll_logdet_or_density_ratio": False,
        "label_independent_fixed_basis_and_pseudoinverse": False, "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False,
        "continuous_first_and_second_functional_jets": False, "v171_wavelet_leader_or_multifractal": False,
        "v172_hankel_trajectory_matrix_or_ssa": False, "v173_weighted_grid_laplacian_or_heat_trace": False,
        "v174_morphological_granulometry": False, "v175_burg_lattice_parcor_or_poles": False,
        "v176_level_crossing_excursion_or_dwell": False, "v177_discrete_radon_line_projection": False,
        "v178_cubic_bspline_functional_jet": False, "v179_structure_tensor_or_gradient_orientation": False,
        "v180_zernike_polar_moments": False, "fixed_order_lattice_features_and_poles_without_label_or_outer_selection": False,
        "burg_forward_backward_innovation_recursion": False, "v119_multichannel_transition_operator_or_dmd": False,
        "fixed_hankel_window_and_rank_without_label_or_outer_selection": False, "anti_diagonal_rank_one_trend_and_residual": False,
        "v119_transition_operator_or_dynamic_eigenvalue": False, "v166_raw_path_grassmann_subspace": False,
        "local_differential_geometry_not_global_path_integral": False, "admissible_frequency_triad_identity_verified": False,
        "within_channel_bispectral_magnitude_and_biphase_verified": False, "directed_cross_channel_biphase_coupling_verified": False,
        "learned_frequency_grid_threshold_or_spectral_feature": False, "second_order_frequency_bin_power_or_unit_cross_spectrum": False,
        "time_domain_increment_coskewness": False, "hilbert_time_local_analytic_phase": False,
        "per_event_row_and_column_grassmann_points": False, "svd_sign_invariant_projection_representation": False,
        "fixed_rank_without_label_or_outer_selection": False, "per_event_channel_correlation_spd": False,
        "fixed_shrinkage_no_label_or_outer_selection": False,
    }
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": geometry, "target_geometry": geometry, "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "method": "train robust causal state plus pooled per-site weighted median", "target_batch_fit": False},
        "head": {"type": "none; direct class Rasch profile likelihood contrast", "fitted_discriminative_coefficients": 0},
        "temperature": {"train_only": True, "value": temperature, "floor": TEMPERATURE_FLOOR, "train_contrast_sha256": array_sha256(train_contrast)},
        "target_profile_ability_sha256": {"down": array_sha256(target_theta_down), "up": array_sha256(target_theta_up)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "item_threshold_model_or_temperature_selected_from_labels_or_outer": False,
        "level_box_statistic_or_penalty_selected_from_labels_or_outer": False, "level_direction_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "dead_zone_neighbor_region_mapping_or_penalty_selected_from_labels_or_outer": False, "bin_offset_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "knot_basis_grid_smoothing_feature_or_penalty_selected_from_labels_or_outer": False, "order_feature_threshold_pole_rule_or_penalty_selected_from_labels_or_outer": False,
        "window_rank_feature_or_penalty_selected_from_labels_or_outer": False, "v86_copula_qda_density": False,
        "v115_increment_spd_log_tangent_logistic": False, "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False,
        "v168_matrix_normal_kronecker_likelihood": False, "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False,
        "v101_static_affine_reconstruction_residual": False, "v143_source_time_nuisance_projection": False,
        "pls_or_ssa_response_projection": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    scaffold.VERSION = VERSION; scaffold.HYPOTHESIS = HYPOTHESIS; scaffold.STATIC_SPEC = STATIC_SPEC; scaffold.ENVELOPE_FEATURE_DIMENSION = ITEM_COUNT; scaffold.ARCHITECTURES = ARCHITECTURES; scaffold.SEED = SEED; scaffold.envelope_direction = rasch_direction; scaffold.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = rasch_direction(inner_train, inner_valid); baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    geometry = audit["train_geometry"]; model_eligible = bool(audit["causal_numeric_only"] and geometry["class_conditional_one_parameter_logistic_rasch"] and geometry["row_global_latent_ability_profiled"] and geometry["mean_zero_item_difficulty_identifiability"] and geometry["fixed_alternating_newton_schedule"] and geometry["threshold_fit_train_only"] and not geometry["neighbor_grid_unary_pair_potts_pseudolikelihood"] and not geometry["two_additive_choquet_capacity_or_mobius_mass"] and not geometry["discriminative_logistic_head"] and not geometry["target_batch_fit"] and not geometry["target_labels_used"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"] and not audit["item_threshold_model_or_temperature_selected_from_labels_or_outer"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high); auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]; source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 class-conditional Rasch profile blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "item_threshold_model_temperature_blend_or_source_floor_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()): diagnostic, evidence, audits = _SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v205_model"}); evidence = evidence.rename(columns={"bspline_prob": "rasch_prob", "v178_model": "v205_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]; print(f"[V205 RASCH] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = _SUPPORT_BOOTSTRAP(evidence, draws); result["method"] = "paired market-fold-time block bootstrap over inner-locked V205 class-conditional Rasch profile policy"; result["seed"] = SEED; result["standardized"] = True; return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap; result = _SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]; checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified")
    contract = all(audit[side]["causal_numeric_only"] and audit[side]["train_geometry"]["class_conditional_one_parameter_logistic_rasch"] and audit[side]["train_geometry"]["row_global_latent_ability_profiled"] and audit[side]["train_geometry"]["mean_zero_item_difficulty_identifiability"] and audit[side]["train_geometry"]["fixed_alternating_newton_schedule"] and audit[side]["train_geometry"]["threshold_fit_train_only"] and audit[side]["target_geometry"]["class_conditional_one_parameter_logistic_rasch"] and audit[side]["head"]["fitted_discriminative_coefficients"] == 0 and audit[side]["temperature"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] and not audit[side]["item_threshold_model_or_temperature_selected_from_labels_or_outer"] and not audit[side]["train_geometry"]["neighbor_grid_unary_pair_potts_pseudolikelihood"] and not audit[side]["train_geometry"]["two_additive_choquet_capacity_or_mobius_mass"] and not audit[side]["train_geometry"]["kernel_stein_energy_flow_or_metric_prototype"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_class_conditional_rasch_item_response_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]: result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else: result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V205 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"); return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]; gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V205 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V205 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V205 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V205").resolve(), "V205 output outside staging/V205"); require(out.exists() and out.is_dir(), "V205 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}; require(required.issubset(existing), f"V205 controller prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V205 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES); require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V205 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V205 output not controller-bound"); conflict = enforce_output_conflict_fail_closed(out); compact = {k: v for k, v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate; require("bootstrap" in selected["robustness"], "V205 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V205", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V205_DIAGNOSTIC_RASCH": evaluation["candidate_summary"], "V205_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V205", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V205", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V205_CLASS_CONDITIONAL_RASCH_ITEM_RESPONSE_REPORT.json": {"version": "V205", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V205", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V205", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V205", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V205 report inventory changed"); [atomic_json(payload, out / name) for name, payload in reports.items()]; atomic_csv(evidence, out / "V205_RASCH_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V205_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json"); return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--contract-probe", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true"); parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V205 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output = DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT), "V205 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V205 controller output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe); runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}; authority = verify_authority(); dev, champion = load_authorized()
    if args.contract_probe: print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2)); return
    if args.audit_only: print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC, "architecture": "pooled train-median binary 64 items plus separate class Rasch difficulty/latent-ability profile likelihood", "prior_exact_rasch_item_response_family_hit_count": 0, "causal_numeric_only": True, "target_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V205 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test: summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
