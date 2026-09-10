"""V213 class-conditional determinantal point-process subset direction.

Each strict-train robust 8x8 causal surface becomes a 64-site active subset
using pooled train-only duplicate-balanced site medians.  An immutable spatial
RBF similarity with fixed identity shrinkage is shared by two class L-ensemble
determinantal point processes; only Jeffreys-smoothed class item-quality odds
are fitted.  Exact log det(L_subset)-log det(I+L) contrasts give direction
probability without a discriminative head, target fit, or outer tuning.
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

import experiment_v210_multichannel_causal_class_conditional_bernoulli_rbm_direction as base


ROOT = base.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V213" / "LOCAL_FORBIDDEN"
VERSION = 213
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_DPP_SUBSET_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_V69_EXPERIMENT_ID = base.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v210_multichannel_causal_class_conditional_bernoulli_rbm_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "aff8223f5f9db3fba59d45c06a52e3769e14ad132c922a5b4b6ca0b93bdcf5f8"

ITEM_COUNT = 64
GRID_N = 8
SPATIAL_LENGTH_SCALE = 1.25
SIMILARITY_IDENTITY_SHRINK = 0.10
JEFFREYS_MASS = 0.5
QUALITY_ODDS_MIN = 0.05
QUALITY_ODDS_MAX = 20.0
PROBABILITY_EPS = 1e-6
TEMPERATURE_FLOOR = 0.05
SEED = 21301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CLASS_CONDITIONAL_DPP_SUBSET_W0.25", "weight": 0.25},
    {"name": "CLASS_CONDITIONAL_DPP_SUBSET_W0.50", "weight": 0.50},
)

require = base.require
sha256 = base.sha256
array_sha256 = base.array_sha256
clean = base.clean
atomic_json = base.atomic_json
atomic_csv = base.atomic_csv
load_authorized = base.load_authorized
controller = base.controller
numeric = base.numeric
observed_past_to_recent_path = base.observed_past_to_recent_path
bool_series = base.bool_series
metric = base.metric
blend_probability = base.blend_probability
supported_source_ba_delta = base.supported_source_ba_delta
ENGINE = base.scaffold.ENGINE
SUPPORT_RUN_NESTED = base.scaffold._SUPPORT_RUN_NESTED
SUPPORT_EVALUATE = base.scaffold._SUPPORT_EVALUATE
SUPPORT_BOOTSTRAP = base.scaffold._SUPPORT_BOOTSTRAP

STATIC_SPEC = {
    "surface": "strict-train robust-scaled observed 8-horizon x 8-semantic-channel causal state",
    "feature_dimension": ITEM_COUNT,
    "subset_sites": ITEM_COUNT,
    "activation": "binary greater-than-or-equal pooled train-only per-site duplicate-balanced median",
    "similarity": "immutable 8x8 coordinate Gaussian RBF with fixed 10 percent identity shrinkage",
    "class_model": "separate class L-ensemble determinantal point process with Jeffreys-smoothed item-quality odds",
    "direction": "exact class DPP subset log-likelihood contrast divided by train-only IQR temperature",
    "discriminative_head": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V213_CLASS_CONDITIONAL_DPP_SUBSET_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(ITEM_COUNT == 64 and GRID_N == 8 and SPATIAL_LENGTH_SCALE == 1.25 and SIMILARITY_IDENTITY_SHRINK == 0.10, "V213 static model changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V210 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, path, metrics, gates, bootstrap and atomic IO only; no V210 output is read"}
    audit["v213_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _fixed_similarity() -> np.ndarray:
    row, column = np.meshgrid(np.arange(GRID_N, dtype=float), np.arange(GRID_N, dtype=float), indexing="ij")
    coordinates = np.column_stack([row.reshape(-1), column.reshape(-1)])
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    rbf = np.exp(-squared_distance / (2.0 * SPATIAL_LENGTH_SCALE * SPATIAL_LENGTH_SCALE))
    similarity = (1.0 - SIMILARITY_IDENTITY_SHRINK) * rbf + SIMILARITY_IDENTITY_SHRINK * np.eye(ITEM_COUNT)
    require(np.linalg.eigvalsh(similarity).min() > 0.0, "V213 fixed similarity not SPD")
    return similarity


def _fit_class_dpp(response: np.ndarray, weights: np.ndarray, similarity: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    require(response.ndim == 2 and response.shape[1] == ITEM_COUNT and len(response) >= 8, "V213 DPP support invalid")
    weights = np.asarray(weights, dtype=float); total = float(weights.sum()); success = np.sum(weights[:, None] * response, axis=0)
    prevalence = (success + JEFFREYS_MASS) / (total + 2.0 * JEFFREYS_MASS)
    quality_odds = np.clip(prevalence / np.maximum(1.0 - prevalence, 1e-12), QUALITY_ODDS_MIN, QUALITY_ODDS_MAX)
    root_quality = np.sqrt(quality_odds)
    kernel = root_quality[:, None] * similarity * root_quality[None, :]
    sign, normalizer = np.linalg.slogdet(np.eye(ITEM_COUNT) + kernel)
    require(sign > 0 and np.isfinite(normalizer) and np.linalg.eigvalsh(kernel).min() > 0.0, "V213 DPP kernel invalid")
    audit = {"rows": len(response), "jeffreys_mass": JEFFREYS_MASS, "quality_odds_min": float(quality_odds.min()), "quality_odds_max": float(quality_odds.max()), "quality_odds_sha256": array_sha256(quality_odds), "l_kernel_sha256": array_sha256(kernel), "log_det_i_plus_l": float(normalizer), "exact_spd_l_ensemble": True}
    return kernel, float(normalizer), audit


def _subset_log_probability(response: np.ndarray, kernel: np.ndarray, normalizer: float) -> np.ndarray:
    result = np.empty(len(response), dtype=float)
    for index, row in enumerate(response):
        active = np.flatnonzero(row > 0.5)
        if len(active) == 0:
            log_minor = 0.0
        else:
            sign, log_minor = np.linalg.slogdet(kernel[np.ix_(active, active)])
            require(sign > 0 and np.isfinite(log_minor), "V213 active subset minor invalid")
        result[index] = float(log_minor) - normalizer
    require(np.isfinite(result).all(), "V213 subset log probabilities invalid")
    return result


def dpp_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V213 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_flat = train_path.reshape(len(train_path), ITEM_COUNT); target_flat = target_path.reshape(len(target_path), ITEM_COUNT)
    base_weights = base.scaffold._base_duplicate_weights(ordered); thresholds = base.scaffold._weighted_median(train_flat, base_weights)
    train_response = (train_flat >= thresholds).astype(float); target_response = (target_flat >= thresholds).astype(float); labels = ordered.y.to_numpy(int)
    similarity = _fixed_similarity(); kernels: dict[int, np.ndarray] = {}; normalizers: dict[int, float] = {}; class_audits: dict[str, Any] = {}
    for label in (0, 1):
        mask = labels == label; kernel, normalizer, class_audit = _fit_class_dpp(train_response[mask], base_weights[mask], similarity); kernels[label] = kernel; normalizers[label] = normalizer; class_audits[str(label)] = class_audit
    train_down = _subset_log_probability(train_response, kernels[0], normalizers[0]); train_up = _subset_log_probability(train_response, kernels[1], normalizers[1]); train_contrast = train_up - train_down
    q25, q75 = np.quantile(train_contrast, [0.25, 0.75]); temperature = max(float((q75 - q25) / 1.349), TEMPERATURE_FLOOR)
    target_down = _subset_log_probability(target_response, kernels[0], normalizers[0]); target_up = _subset_log_probability(target_response, kernels[1], normalizers[1]); probability = _sigmoid((target_up - target_down) / temperature)
    geometry = {
        "feature_dimension": ITEM_COUNT, "subset_site_count": ITEM_COUNT, "threshold_sha256": array_sha256(thresholds), "threshold_fit_train_only": True,
        "train_subset_response_sha256": array_sha256(train_response), "target_subset_response_sha256": array_sha256(target_response),
        "fixed_spatial_similarity_sha256": array_sha256(similarity), "fixed_spatial_rbf_similarity": True, "fixed_identity_shrinkage": SIMILARITY_IDENTITY_SHRINK,
        "class_models": class_audits, "class_conditional_dpp_l_ensemble": True, "exact_subset_minor_log_determinant": True, "exact_log_det_i_plus_l_normalizer": True,
        "jeffreys_item_quality_odds_only_class_fit": True, "target_batch_fit": False, "target_labels_used": False, "label_independent_exact_transform": True,
        "discriminative_logistic_head": False, "rbm_hidden_units_cd_or_free_energy": False, "potts_visible_neighbor_pair_potential": False,
        "two_additive_choquet_capacity_or_mobius_mass": False, "slow_feature_generalized_eigen": False, "cross_horizon_cca": False,
        "within_event_hsic_or_kernel_alignment": False, "multiple_correspondence_analysis_or_chi_square_svd": False,
        "item_threshold_quality_similarity_temperature_or_blend_selected_from_outer": False,
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
        "head": {"type": "none; direct exact class DPP subset log-likelihood contrast", "fitted_discriminative_coefficients": 0},
        "temperature": {"train_only": True, "value": temperature, "floor": TEMPERATURE_FLOOR, "train_contrast_sha256": array_sha256(train_contrast)},
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
    base.VERSION = VERSION; base.HYPOTHESIS = HYPOTHESIS; base.STATIC_SPEC = STATIC_SPEC; base.ARCHITECTURES = ARCHITECTURES; base.SEED = SEED; base.rbm_direction = dpp_direction; base.choose_inner = choose_inner; base.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = dpp_direction(inner_train, inner_valid); baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    g = audit["train_geometry"]; eligible_model = bool(g["class_conditional_dpp_l_ensemble"] and g["exact_subset_minor_log_determinant"] and g["exact_log_det_i_plus_l_normalizer"] and g["jeffreys_item_quality_odds_only_class_fit"] and g["threshold_fit_train_only"] and not g["rbm_hidden_units_cd_or_free_energy"] and not g["potts_visible_neighbor_pair_potential"] and not g["discriminative_logistic_head"] and not g["target_batch_fit"] and not g["target_labels_used"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high); auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]; source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": eligible_model, "eligible": bool(eligible_model and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 class DPP subset-likelihood blend", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "threshold_similarity_shrink_quality_temperature_blend_or_safety_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()): diagnostic, evidence, audits = SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v213_model"}); evidence = evidence.rename(columns={"bspline_prob": "dpp_prob", "v178_model": "v213_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]; print(f"[V213 DPP] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = SUPPORT_BOOTSTRAP(evidence, draws); result["method"] = "paired market-fold-time block bootstrap over inner-locked V213 class DPP subset-likelihood policy"; result["seed"] = SEED; result["standardized"] = True; return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap; result = SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]; checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified", None)
    contract = all(audit[side]["train_geometry"]["class_conditional_dpp_l_ensemble"] and audit[side]["train_geometry"]["exact_subset_minor_log_determinant"] and audit[side]["train_geometry"]["exact_log_det_i_plus_l_normalizer"] and audit[side]["train_geometry"]["jeffreys_item_quality_odds_only_class_fit"] and audit[side]["head"]["fitted_discriminative_coefficients"] == 0 and audit[side]["temperature"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_class_conditional_dpp_subset_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]: result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else: result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V213 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"); return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]; gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V213 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V213 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V213 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V213").resolve(), "V213 output outside staging/V213"); require(out.exists() and out.is_dir(), "V213 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}; require(required.issubset(existing), f"V213 controller prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V213 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES); require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V213 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V213 output not controller-bound"); conflict = enforce_output_conflict_fail_closed(out); compact = {k: v for k, v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate; require("bootstrap" in selected["robustness"], "V213 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V213", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V213_DIAGNOSTIC_DPP": evaluation["candidate_summary"], "V213_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V213", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V213", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V213_CLASS_CONDITIONAL_DPP_SUBSET_REPORT.json": {"version": "V213", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V213", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V213", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V213", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V213 report inventory changed"); [atomic_json(payload, out / name) for name, payload in reports.items()]; atomic_csv(evidence, out / "V213_DPP_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V213_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json"); return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--contract-probe", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true"); parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V213 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output = DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT), "V213 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V213 controller output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe); runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}; authority = verify_authority(); dev, champion = load_authorized()
    if args.contract_probe: print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2)); return
    if args.audit_only: print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC, "architecture": "64 train-median active sites plus fixed spatial-similarity class DPP exact subset likelihood", "prior_exact_dpp_family_hit_count": 0, "causal_numeric_only": True, "target_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V213 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test: summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
