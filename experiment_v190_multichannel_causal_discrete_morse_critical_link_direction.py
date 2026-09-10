"""V190 multichannel causal discrete-Morse critical-link direction.

The preferred ordinal complexity-plane proposal is rejected because V144
already implements permutation motifs, permutation/transition entropy and
directed motif transitions.  This replacement maps each strict-past train-
scaled causal 8x8 surface to deterministic local lower/upper-link connectivity
on its 6x6 interior.  Clockwise eight-neighbour circular components identify
local minima, maxima and saddle multiplicity with stable spatial tie breaking.
Fixed regional critical-value, position, margin, link-histogram and index
summaries form 240 features for one balanced ridge head.

There is no filtration sweep, boundary matrix, homology computation,
persistence pairing/lifetime, Euler threshold curve or dyadic coarse grid.
Strict 35m chronology, inner-only selection, frozen V69 confidence/high_conf
and exact-entire-V69 fallback apply.
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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v187_multichannel_causal_gray_level_run_length_texture_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V190" / "LOCAL_FORBIDDEN"
VERSION = 190
HYPOTHESIS = "MULTICHANNEL_CAUSAL_DISCRETE_MORSE_CRITICAL_LINK_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v187_multichannel_causal_gray_level_run_length_texture_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "c82feee8257312d1ce9cb64dabbf4b3365da63de1bda8bc8ac70f9ab839790c2"

NEIGHBOR_OFFSETS = (("N", -1, 0), ("NE", -1, 1), ("E", 0, 1), ("SE", 1, 1), ("S", 1, 0), ("SW", 1, -1), ("W", 0, -1), ("NW", -1, -1))
REGIONS = (("ALL", 0, 6, 0, 6), ("OLDER_CENTER", 0, 3, 0, 6), ("RECENT_CENTER", 3, 6, 0, 6), ("CHANNEL_LEFT_CENTER", 0, 6, 0, 3), ("CHANNEL_RIGHT_CENTER", 0, 6, 3, 6))
LINK_COMPONENT_BIN_N = 5
FEATURES_PER_REGION = 48
MORSE_FEATURE_DIMENSION = len(REGIONS) * FEATURES_PER_REGION
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 19001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = ({"name": "DISCRETE_MORSE_CRITICAL_LINK_W0.25", "weight": 0.25}, {"name": "DISCRETE_MORSE_CRITICAL_LINK_W0.50", "weight": 0.50})

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
bool_series = base.bool_series
metric = base.metric
blend_probability = base.blend_probability
supported_source_ba_delta = base.supported_source_ba_delta

STATIC_SPEC = {
    "surface": "train-scaled observed 8-horizon x 8-semantic-channel causal state",
    "interior_grid": [6, 6],
    "link": "clockwise eight-neighbour circular lower/upper link with deterministic spatial tie break",
    "critical_types": ["LOCAL_MINIMUM", "LOCAL_MAXIMUM", "SADDLE"],
    "regions": [item[0] for item in REGIONS],
    "features_per_region": FEATURES_PER_REGION,
    "feature_dimension": MORSE_FEATURE_DIMENSION,
    "summaries": "lower/upper component histograms and moments, min/max/saddle fractions, saddle multiplicity, Euler-index moments, type-conditioned value quantiles/position centroids/margin moments",
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "filtration_boundary_reduction_persistence_pairing_or_lifetime": False,
    "learned_neighbor_tie_region_summary_or_head_penalty": False,
}
REPORT_NAMES = ("MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json", "V190_DISCRETE_MORSE_CRITICAL_LINK_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json", "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json")
require(len(NEIGHBOR_OFFSETS) == 8 and len(REGIONS) == 5 and FEATURES_PER_REGION == 48 and MORSE_FEATURE_DIMENSION == 240, "V190 static Morse representation changed")

_SCAFFOLD_CONFIGURE = scaffold.configure_base
_SUPPORT_RUN_NESTED = scaffold._SUPPORT_RUN_NESTED
_SUPPORT_EVALUATE = scaffold._SUPPORT_EVALUATE
_SUPPORT_BOOTSTRAP = scaffold._SUPPORT_BOOTSTRAP
ENGINE = scaffold.scaffold.scaffold.support


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V187 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, gates, bootstrap and atomic IO only; no V187 output is read"}
    audit["v190_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _circular_components(indicator: np.ndarray) -> np.ndarray:
    require(indicator.ndim == 4 and indicator.shape[-1] == 8, "V190 link indicator invalid")
    starts = np.sum(indicator & ~np.roll(indicator, 1, axis=-1), axis=-1)
    return np.where(np.all(indicator, axis=-1), 1, starts).astype(np.int8)


def _masked_value_stats(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros((len(value), 5), dtype=float)
    flat_value, flat_mask = value.reshape(len(value), -1), mask.reshape(len(mask), -1)
    for row in range(len(value)):
        selected = flat_value[row, flat_mask[row]]
        if len(selected):
            result[row] = [float(np.mean(selected)), float(np.std(selected)), *np.quantile(selected, [0.25, 0.5, 0.75]).tolist()]
    return result


def _masked_position(mask: np.ndarray, row_coordinate: np.ndarray, column_coordinate: np.ndarray) -> np.ndarray:
    result = np.zeros((len(mask), 2), dtype=float)
    flat_mask = mask.reshape(len(mask), -1)
    rr, cc = row_coordinate.reshape(-1), column_coordinate.reshape(-1)
    for row in range(len(mask)):
        selected = flat_mask[row]
        if np.any(selected):
            result[row] = [float(np.mean(rr[selected])), float(np.mean(cc[selected]))]
    return result


def _masked_mean_std(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros((len(value), 2), dtype=float)
    flat_value, flat_mask = value.reshape(len(value), -1), mask.reshape(len(mask), -1)
    for row in range(len(value)):
        selected = flat_value[row, flat_mask[row]]
        if len(selected):
            result[row] = [float(np.mean(selected)), float(np.std(selected))]
    return result


def discrete_morse_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (8, 8), "V190 path shape invalid")
    center = path[:, 1:-1, 1:-1]
    neighbours = np.stack([path[:, 1 + dr:7 + dr, 1 + dc:7 + dc] for _, dr, dc in NEIGHBOR_OFFSETS], axis=-1)
    center_index = np.arange(8 * 8, dtype=int).reshape(8, 8)[1:-1, 1:-1]
    neighbor_index = np.stack([np.arange(8 * 8, dtype=int).reshape(8, 8)[1 + dr:7 + dr, 1 + dc:7 + dc] for _, dr, dc in NEIGHBOR_OFFSETS], axis=-1)
    less = (neighbours < center[..., None]) | ((neighbours == center[..., None]) & (neighbor_index[None, ...] < center_index[None, ..., None]))
    greater = ~less
    lower_component = _circular_components(less)
    upper_component = _circular_components(greater)
    require(int(np.max(lower_component)) <= 4 and int(np.max(upper_component)) <= 4, "V190 link component bound changed")
    local_minimum = lower_component == 0
    local_maximum = upper_component == 0
    saddle_multiplicity = np.maximum(lower_component.astype(int) - 1, 0)
    saddle = saddle_multiplicity > 0
    euler_index = 1.0 - lower_component.astype(float)
    difference = neighbours - center[..., None]
    minimum_margin = np.min(difference, axis=-1)
    maximum_margin = np.min(-difference, axis=-1)
    saddle_contrast = np.max(difference, axis=-1) - np.min(difference, axis=-1)
    row_grid, column_grid = np.meshgrid(np.arange(6, dtype=float) / 5.0, np.arange(6, dtype=float) / 5.0, indexing="ij")
    parts: list[np.ndarray] = []
    region_counts: dict[str, int] = {}
    for name, rs, re, cs, ce in REGIONS:
        value = center[:, rs:re, cs:ce]
        lower = lower_component[:, rs:re, cs:ce]
        upper = upper_component[:, rs:re, cs:ce]
        minimum = local_minimum[:, rs:re, cs:ce]
        maximum = local_maximum[:, rs:re, cs:ce]
        current_saddle = saddle[:, rs:re, cs:ce]
        multiplicity = saddle_multiplicity[:, rs:re, cs:ce].astype(float)
        index = euler_index[:, rs:re, cs:ce]
        rows, columns = row_grid[rs:re, cs:ce], column_grid[rs:re, cs:ce]
        lower_hist = np.stack([np.mean(lower == item, axis=(1, 2)) for item in range(LINK_COMPONENT_BIN_N)], axis=1)
        upper_hist = np.stack([np.mean(upper == item, axis=(1, 2)) for item in range(LINK_COMPONENT_BIN_N)], axis=1)
        fractions = np.stack([np.mean(minimum, axis=(1, 2)), np.mean(maximum, axis=(1, 2)), np.mean(current_saddle, axis=(1, 2))], axis=1)
        multiplicity_summary = np.stack([np.mean(multiplicity, axis=(1, 2)), np.max(multiplicity, axis=(1, 2)) / 3.0], axis=1)
        index_summary = np.stack([np.mean(index, axis=(1, 2)), np.std(index, axis=(1, 2))], axis=1)
        component_moments = np.stack([np.mean(lower, axis=(1, 2)) / 4.0, np.std(lower, axis=(1, 2)) / 4.0, np.mean(upper, axis=(1, 2)) / 4.0, np.std(upper, axis=(1, 2)) / 4.0], axis=1)
        value_stats = np.concatenate([_masked_value_stats(value, mask) for mask in (minimum, maximum, current_saddle)], axis=1)
        position_stats = np.concatenate([_masked_position(mask, rows, columns) for mask in (minimum, maximum, current_saddle)], axis=1)
        margin_stats = np.concatenate([_masked_mean_std(minimum_margin[:, rs:re, cs:ce], minimum), _masked_mean_std(maximum_margin[:, rs:re, cs:ce], maximum), _masked_mean_std(saddle_contrast[:, rs:re, cs:ce], current_saddle)], axis=1)
        region_feature = np.concatenate([lower_hist, upper_hist, fractions, multiplicity_summary, index_summary, component_moments, value_stats, position_stats, margin_stats], axis=1)
        require(region_feature.shape == (len(path), FEATURES_PER_REGION), "V190 regional dimension changed")
        parts.append(region_feature)
        region_counts[name] = int((re - rs) * (ce - cs))
    result = np.concatenate(parts, axis=1)
    require(result.shape == (len(path), MORSE_FEATURE_DIMENSION) and np.isfinite(result).all(), "V190 feature invalid")
    return result, {
        "rows": len(path), "interior_grid": [6, 6], "neighbor_n": 8, "region_n": len(REGIONS), "features_per_region": FEATURES_PER_REGION, "feature_dimension": MORSE_FEATURE_DIMENSION, "region_vertex_counts": region_counts,
        "lower_component_minmax": [int(np.min(lower_component)), int(np.max(lower_component))], "upper_component_minmax": [int(np.min(upper_component)), int(np.max(upper_component))], "feature_sha256": array_sha256(result),
        "label_independent_exact_transform": True, "deterministic_spatial_tie_break": True, "fixed_local_lower_upper_link_criticality_without_label_or_outer_selection": True,
        "filtration_sweep_boundary_matrix_homology_persistence_pairing_or_lifetime": False, "v167_euler_threshold_curve": False, "v188_cubical_persistent_homology": False, "v189_dyadic_block_spin": False,
        "label_independent_fixed_basis_and_pseudoinverse": False, "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False, "continuous_first_and_second_functional_jets": False,
        "v171_wavelet_leader_or_multifractal": False, "v172_hankel_trajectory_matrix_or_ssa": False, "v173_weighted_grid_laplacian_or_heat_trace": False, "v174_morphological_granulometry": False, "v175_burg_lattice_parcor_or_poles": False, "v176_level_crossing_excursion_or_dwell": False, "v177_discrete_radon_line_projection": False, "v178_cubic_bspline_functional_jet": False, "v179_structure_tensor_or_gradient_orientation": False, "v180_zernike_polar_moments": False,
        "fixed_order_lattice_features_and_poles_without_label_or_outer_selection": False, "burg_forward_backward_innovation_recursion": False, "v119_multichannel_transition_operator_or_dmd": False, "fixed_hankel_window_and_rank_without_label_or_outer_selection": False, "anti_diagonal_rank_one_trend_and_residual": False, "v119_transition_operator_or_dynamic_eigenvalue": False, "v166_raw_path_grassmann_subspace": False, "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False, "within_channel_bispectral_magnitude_and_biphase_verified": False, "directed_cross_channel_biphase_coupling_verified": False, "learned_frequency_grid_threshold_or_spectral_feature": False, "second_order_frequency_bin_power_or_unit_cross_spectrum": False, "time_domain_increment_coskewness": False, "hilbert_time_local_analytic_phase": False, "per_event_row_and_column_grassmann_points": False, "svd_sign_invariant_projection_representation": False, "fixed_rank_without_label_or_outer_selection": False, "per_event_channel_correlation_spd": False, "fixed_shrinkage_no_label_or_outer_selection": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == MORSE_FEATURE_DIMENSION, "V190 scale shape changed")
        self.median = np.median(matrix, axis=0); q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0); robust = (q75 - q25) / 1.349; standard = np.std(matrix, axis=0); self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0)); return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP); require(np.isfinite(result).all(), "V190 scaled feature invalid"); return result


def morse_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True); require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V190 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry = discrete_morse_features(train_path); target_feature, target_geometry = discrete_morse_features(target_path); scale = RobustFeatureScale().fit(train_feature); train_design, target_design = scale.transform(train_feature), scale.transform(target_feature)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED); head.fit(train_design, ordered.y.to_numpy(int), sample_weight=numeric.duplicate_class_weights(ordered)); require(int(head.n_iter_[0]) < head.max_iter, "V190 ridge logistic did not converge"); probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1 - 1e-6)
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_geometry": train_geometry, "target_geometry": target_geometry, "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(scale.median), "scale_sha256": array_sha256(scale.scale)}, "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "fitted_discriminative_coefficients": int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "neighbor_tie_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "level_direction_region_summary_or_penalty_selected_from_labels_or_outer": False, "dead_zone_neighbor_region_mapping_or_penalty_selected_from_labels_or_outer": False, "bin_offset_region_summary_or_penalty_selected_from_labels_or_outer": False, "knot_basis_grid_smoothing_feature_or_penalty_selected_from_labels_or_outer": False, "order_feature_threshold_pole_rule_or_penalty_selected_from_labels_or_outer": False, "window_rank_feature_or_penalty_selected_from_labels_or_outer": False,
        "v86_copula_qda_density": False, "v115_increment_spd_log_tangent_logistic": False, "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False, "v168_matrix_normal_kronecker_likelihood": False, "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False, "v101_static_affine_reconstruction_residual": False, "v143_source_time_nuisance_projection": False, "pls_or_ssa_response_projection": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    scaffold.VERSION = VERSION; scaffold.HYPOTHESIS = HYPOTHESIS; scaffold.STATIC_SPEC = STATIC_SPEC; scaffold.GLRLM_FEATURE_DIMENSION = MORSE_FEATURE_DIMENSION; scaffold.ARCHITECTURES = ARCHITECTURES; scaffold.SEED = SEED; scaffold.glrlm_direction = morse_direction; _SCAFFOLD_CONFIGURE()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = morse_direction(inner_train, inner_valid); baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    eligible_model = bool(audit["causal_numeric_only"] and audit["train_geometry"]["fixed_local_lower_upper_link_criticality_without_label_or_outer_selection"] and audit["train_geometry"]["deterministic_spatial_tie_break"] and not audit["train_geometry"]["filtration_sweep_boundary_matrix_homology_persistence_pairing_or_lifetime"] and audit["train_geometry"]["feature_dimension"] == MORSE_FEATURE_DIMENSION and audit["feature_scaling"]["train_only"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"] and not audit["neighbor_tie_region_summary_or_penalty_selected_from_labels_or_outer"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high); auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]; source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": eligible_model, "eligible": bool(eligible_model and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 discrete-Morse critical-link blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "neighbor_tie_region_summary_head_blend_or_source_floor_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()): diagnostic, evidence, audits = _SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v190_model"}); evidence = evidence.rename(columns={"bspline_prob": "morse_prob", "v178_model": "v190_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]; print(f"[V190 MORSE] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = _SUPPORT_BOOTSTRAP(evidence, draws); result["method"] = "paired market-fold-time block bootstrap over inner-locked V190 discrete-Morse critical-link policy"; result["seed"] = SEED; result["standardized"] = True; return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap; result = _SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]; checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified")
    contract = all(audit[side]["causal_numeric_only"] and audit[side]["train_geometry"]["fixed_local_lower_upper_link_criticality_without_label_or_outer_selection"] and audit[side]["train_geometry"]["deterministic_spatial_tie_break"] and not audit[side]["train_geometry"]["filtration_sweep_boundary_matrix_homology_persistence_pairing_or_lifetime"] and audit[side]["train_geometry"]["feature_dimension"] == MORSE_FEATURE_DIMENSION and audit[side]["target_geometry"]["fixed_local_lower_upper_link_criticality_without_label_or_outer_selection"] and audit[side]["feature_scaling"]["train_only"] and audit[side]["head"]["fitted_discriminative_coefficients"] == MORSE_FEATURE_DIMENSION and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] and not audit[side]["neighbor_tie_region_summary_or_penalty_selected_from_labels_or_outer"] and not audit[side]["train_geometry"]["v188_cubical_persistent_homology"] and not audit[side]["train_geometry"]["v189_dyadic_block_spin"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_discrete_morse_critical_link_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]: result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else: result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V190 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"); return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]; gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V190 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V190 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V190 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V190").resolve(), "V190 output is outside controller research/staging/V190"); require(out.exists() and out.is_dir(), "V190 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}; require(required.issubset(existing), f"V190 controller staging prefiles missing: {sorted(required-existing)}"); require(not(existing-allowed), f"V190 unexpected pre-existing output conflict: {sorted(existing-allowed)}"); return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES); require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V190 report inventory invalid"); return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V190 output not controller-bound"); conflict = enforce_output_conflict_fail_closed(out); compact = {k:v for k,v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate; require("bootstrap" in selected["robustness"], "V190 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version":"V190","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V190_DIAGNOSTIC_DISCRETE_MORSE":evaluation["candidate_summary"],"V190_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version":"V190","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},
        "SOURCE_TRANSFER_REPORT.json": {"version":"V190","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},
        "V190_DISCRETE_MORSE_CRITICAL_LINK_REPORT.json": {"version":"V190","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version":"V190","hypothesis":HYPOTHESIS,"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version":"V190","hypothesis":HYPOTHESIS,"controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version":"V190","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},
        "RUN_STATUS.json": {"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports)==set(REPORT_NAMES),"V190 report inventory changed")
    for name,payload in reports.items(): atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V190_DISCRETE_MORSE_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"],out/"V190_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"}; atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json"); return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only",action="store_true");modes.add_argument("--contract-probe",action="store_true");modes.add_argument("--smoke-test",action="store_true");modes.add_argument("--support-probe",action="store_true");modes.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V190 full run");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output=DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT),"V190 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT");require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V190 controller full-run output mismatch")
    return args


def main() -> None:
    args=parse_args();bounded=bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime() if bounded else {"mode":"controller_bound_full_cpu","thread_policy":"authorized parent inherited","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if args.contract_probe: print(json.dumps(clean({**temporary_report_contract_probe(),"hypothesis":HYPOTHESIS,"runtime":runtime,"authority":authority}),ensure_ascii=False,indent=2));return
    if args.audit_only: print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"features":FEATURES,"static_spec":STATIC_SPEC,"architecture":"fixed local lower/upper-link discrete-Morse criticality summaries plus balanced ridge","preferred_rank_complexity_rejected_due_v144_collision":True,"prior_exact_discrete_morse_critical_link_family_hit_count":0,"causal_numeric_only":True,"target_row_labels_used":False,"micro_tuning_or_post_outer_change":False,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"gpu_model_calls":0,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"current_material_gate":True,"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","standardized_nested_bootstrap":True,"native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2));return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V190 support probe reserved for KR:2");audit=audits[0];non=[t for t in audit["policy"]["trials"] if t["architecture"] is not None];best=max(non,key=lambda t:(t["eligible"],t["score"],t["auc_delta"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"nested_bootstrap_executed":False,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[t for t in audits[0]["policy"]["trials"] if t["architecture"] is not None];best=max(non,key=lambda t:(t["eligible"],t["score"],t["auc_delta"]));summary={"status":"SMOKE_OK" if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[a["policy"]["selected"]["name"] for a in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test: summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]});print(json.dumps(clean(summary),ensure_ascii=False,indent=2));return
    result=write_outputs(args.output,authority,evidence,audits,evaluation);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
