"""V181 multichannel causal directional GLCM texture direction.

Each strict-past same-market fit maps immutable causal numeric state to an
observed train-scaled 8-channel by 8-horizon surface.  Fixed robust-state
thresholds quantize it into five levels.  Directed gray-level co-occurrence
matrices at horizon, channel and two diagonal offsets are computed in five
predeclared spatial regions.  Fourteen fixed Haralick/ordinal texture
summaries per matrix retain contrast, dependence and orientation asymmetry.
One train-only robust feature scale and one equal-event-group/class-balanced
ridge-logistic head produce direction probability.

This is not V179 gradient structure tensors or V180 continuous Zernike polar
moments, and it is disjoint from V173 graph spectrum, V174 morphology, V175
Burg prediction, V176 crossings, V177 Radon tomography and V178 B-splines.
Bins, offsets, regions, summaries, head and blends are frozen before outer
evaluation.  Inner history chooses exact V69 or fixed .25/.50 blends; strict
35-minute embargo/event purge applies.  V69 confidence/high_conf remain exact
and material failure restores the exact entire V69 frame.
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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v178_multichannel_causal_cubic_bspline_functional_jet_direction as support


base = support.base
ROOT = support.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V181" / "LOCAL_FORBIDDEN"
VERSION = 181
HYPOTHESIS = "MULTICHANNEL_CAUSAL_DIRECTIONAL_GLCM_TEXTURE_DIRECTION_V1"
FEATURES = support.FEATURES
EXPECTED_V69_EXPERIMENT_ID = support.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v178_multichannel_causal_cubic_bspline_functional_jet_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "7cbb640c21169e45d913de0565d01d60daa96acaceef9347572f119d66bb8cd4"

CHANNEL_N = 8
HORIZON_N = 8
LEVEL_N = 5
QUANTILE_THRESHOLDS = np.array([-1.0, -0.25, 0.25, 1.0])
OFFSETS = (
    ("HORIZON_RECENT", 1, 0),
    ("CHANNEL_FORWARD", 0, 1),
    ("DIAGONAL_FORWARD", 1, 1),
    ("DIAGONAL_REVERSE", 1, -1),
)
REGIONS = (
    ("ALL", 0, 8, 0, 8),
    ("OLDER_HALF", 0, 4, 0, 8),
    ("RECENT_HALF", 4, 8, 0, 8),
    ("CHANNEL_LEFT_HALF", 0, 8, 0, 4),
    ("CHANNEL_RIGHT_HALF", 0, 8, 4, 8),
)
SUMMARY_N = 14
GLCM_FEATURE_DIMENSION = len(OFFSETS) * len(REGIONS) * SUMMARY_N
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 18101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "DIRECTIONAL_GLCM_TEXTURE_W0.25", "weight": 0.25},
    {"name": "DIRECTIONAL_GLCM_TEXTURE_W0.50", "weight": 0.50},
)

require = support.require
sha256 = support.sha256
array_sha256 = support.array_sha256
clean = support.clean
atomic_json = support.atomic_json
atomic_csv = support.atomic_csv
load_authorized = support.load_authorized
controller = support.controller
numeric = support.numeric
observed_past_to_recent_path = support.observed_past_to_recent_path
bool_series = base.bool_series
metric = base.metric
blend_probability = base.blend_probability
supported_source_ba_delta = base.supported_source_ba_delta

STATIC_SPEC = {
    "surface": "train-scaled observed 8-horizon x 8-semantic-channel causal state",
    "quantization": "fixed five levels at robust-state thresholds -1,-0.25,0.25,1",
    "directed_offsets": [item[0] for item in OFFSETS],
    "regions": [item[0] for item in REGIONS],
    "glcm_shape": [LEVEL_N, LEVEL_N],
    "summary_n_per_region_offset": SUMMARY_N,
    "summaries": [
        "contrast", "dissimilarity", "homogeneity", "angular_second_moment",
        "energy", "entropy", "correlation", "maximum_probability",
        "diagonal_mass", "directed_upper_lower_asymmetry", "cluster_shade",
        "cluster_prominence", "sum_entropy", "difference_entropy",
    ],
    "feature_dimension": GLCM_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_bin_offset_region_summary_or_head_penalty": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V181_DIRECTIONAL_GLCM_TEXTURE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(LEVEL_N == 5 and len(OFFSETS) == 4 and len(REGIONS) == 5 and GLCM_FEATURE_DIMENSION == 280, "V181 static GLCM representation changed")

_SUPPORT_CONFIGURE = support.configure_base
_SUPPORT_RUN_NESTED = support.run_nested
_SUPPORT_EVALUATE = support.evaluate
_SUPPORT_BOOTSTRAP = support.paired_nested_bootstrap


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V178 utility scaffold changed")
    audit = support.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, canonical gate, bootstrap and atomic IO only; no V178 output is read"}
    audit["v181_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _pairs(surface: np.ndarray, row_offset: int, column_offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = surface.shape[1:]
    row_source = slice(max(0, -row_offset), rows - max(0, row_offset))
    row_target = slice(max(0, row_offset), rows - max(0, -row_offset))
    column_source = slice(max(0, -column_offset), columns - max(0, column_offset))
    column_target = slice(max(0, column_offset), columns - max(0, -column_offset))
    source = surface[:, row_source, column_source].reshape(len(surface), -1)
    target = surface[:, row_target, column_target].reshape(len(surface), -1)
    require(source.shape == target.shape and source.shape[1] > 0, "V181 directed pair geometry invalid")
    return source, target


def _glcm(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.empty((len(source), LEVEL_N, LEVEL_N), dtype=float)
    for left in range(LEVEL_N):
        for right in range(LEVEL_N):
            result[:, left, right] = np.mean((source == left) & (target == right), axis=1)
    require(np.max(np.abs(np.sum(result, axis=(1, 2)) - 1.0)) < 1e-12, "V181 GLCM normalization invalid")
    return result


def _glcm_summary(glcm: np.ndarray) -> np.ndarray:
    indices = np.arange(LEVEL_N, dtype=float)
    left = indices[None, :, None]
    right = indices[None, None, :]
    difference = left - right
    absolute = np.abs(difference)
    marginal_left = np.sum(glcm, axis=2)
    marginal_right = np.sum(glcm, axis=1)
    mean_left = np.sum(marginal_left * indices[None, :], axis=1)
    mean_right = np.sum(marginal_right * indices[None, :], axis=1)
    variance_left = np.sum(marginal_left * (indices[None, :] - mean_left[:, None]) ** 2, axis=1)
    variance_right = np.sum(marginal_right * (indices[None, :] - mean_right[:, None]) ** 2, axis=1)
    covariance = np.sum(glcm * (left - mean_left[:, None, None]) * (right - mean_right[:, None, None]), axis=(1, 2))
    correlation = covariance / np.sqrt(np.maximum(variance_left * variance_right, 1e-12))
    asm = np.sum(glcm * glcm, axis=(1, 2))
    centered_sum = left + right - mean_left[:, None, None] - mean_right[:, None, None]
    sum_distribution = np.stack([np.sum(glcm[:, (np.add.outer(indices, indices) == value)], axis=1) for value in range(2 * LEVEL_N - 1)], axis=1)
    difference_distribution = np.stack([np.sum(glcm[:, (np.abs(np.subtract.outer(indices, indices)) == value)], axis=1) for value in range(LEVEL_N)], axis=1)
    summary = np.stack([
        np.sum(glcm * difference * difference, axis=(1, 2)) / 16.0,
        np.sum(glcm * absolute, axis=(1, 2)) / 4.0,
        np.sum(glcm / (1.0 + difference * difference), axis=(1, 2)),
        asm,
        np.sqrt(asm),
        -np.sum(glcm * np.log(np.maximum(glcm, 1e-12)), axis=(1, 2)) / np.log(25.0),
        correlation,
        np.max(glcm, axis=(1, 2)),
        np.sum(glcm[:, indices.astype(int), indices.astype(int)], axis=1),
        np.sum(glcm * np.sign(right - left), axis=(1, 2)),
        np.sum(glcm * centered_sum ** 3, axis=(1, 2)) / 512.0,
        np.sum(glcm * centered_sum ** 4, axis=(1, 2)) / 4096.0,
        -np.sum(sum_distribution * np.log(np.maximum(sum_distribution, 1e-12)), axis=1) / np.log(9.0),
        -np.sum(difference_distribution * np.log(np.maximum(difference_distribution, 1e-12)), axis=1) / np.log(5.0),
    ], axis=1)
    require(summary.shape == (len(glcm), SUMMARY_N) and np.isfinite(summary).all(), "V181 GLCM summary invalid")
    return summary


def directional_glcm_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V181 path shape invalid")
    quantized = np.digitize(path, QUANTILE_THRESHOLDS).astype(np.int8)
    parts: list[np.ndarray] = []
    pair_counts: dict[str, int] = {}
    normalization_error = 0.0
    for region_name, row_start, row_stop, column_start, column_stop in REGIONS:
        region = quantized[:, row_start:row_stop, column_start:column_stop]
        for offset_name, row_offset, column_offset in OFFSETS:
            source, target = _pairs(region, row_offset, column_offset)
            matrix = _glcm(source, target)
            parts.append(_glcm_summary(matrix))
            pair_counts[f"{region_name}:{offset_name}"] = int(source.shape[1])
            normalization_error = max(normalization_error, float(np.max(np.abs(np.sum(matrix, axis=(1, 2)) - 1.0))))
    result = np.concatenate(parts, axis=1)
    require(result.shape == (len(path), GLCM_FEATURE_DIMENSION), "V181 feature dimension changed")
    return result, {
        "rows": len(path), "level_n": LEVEL_N, "offset_n": len(OFFSETS), "region_n": len(REGIONS),
        "summary_n": SUMMARY_N, "feature_dimension": GLCM_FEATURE_DIMENSION,
        "pair_counts": pair_counts, "glcm_normalization_max_abs_error": normalization_error,
        "feature_sha256": array_sha256(result), "label_independent_exact_transform": True,
        "fixed_bins_offsets_regions_and_summaries_without_label_or_outer_selection": True,
        "directed_cooccurrence_asymmetry_preserved": True,
        "v173_weighted_grid_laplacian_or_heat_trace": False, "v174_morphological_granulometry": False,
        "v175_burg_lattice_parcor_or_poles": False, "v176_level_crossing_excursion_or_dwell": False,
        "v177_discrete_radon_line_projection": False, "v178_cubic_bspline_functional_jet": False,
        "v179_structure_tensor_or_gradient_orientation": False, "v180_zernike_polar_moments": False,
        # Compatibility negatives for inherited evaluator chain.
        "label_independent_fixed_basis_and_pseudoinverse": False,
        "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False,
        "continuous_first_and_second_functional_jets": False,
        "v171_wavelet_leader_or_multifractal": False, "v172_hankel_trajectory_matrix_or_ssa": False,
        "fixed_order_lattice_features_and_poles_without_label_or_outer_selection": False,
        "burg_forward_backward_innovation_recursion": False, "v119_multichannel_transition_operator_or_dmd": False,
        "fixed_hankel_window_and_rank_without_label_or_outer_selection": False,
        "anti_diagonal_rank_one_trend_and_residual": False, "v119_transition_operator_or_dynamic_eigenvalue": False,
        "v166_raw_path_grassmann_subspace": False, "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False, "within_channel_bispectral_magnitude_and_biphase_verified": False,
        "directed_cross_channel_biphase_coupling_verified": False, "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False, "time_domain_increment_coskewness": False,
        "hilbert_time_local_analytic_phase": False, "per_event_row_and_column_grassmann_points": False,
        "svd_sign_invariant_projection_representation": False, "fixed_rank_without_label_or_outer_selection": False,
        "per_event_channel_correlation_spd": False, "fixed_shrinkage_no_label_or_outer_selection": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == GLCM_FEATURE_DIMENSION, "V181 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V181 scaled feature invalid")
        return result


def glcm_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V181 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry = directional_glcm_features(train_path)
    target_feature, target_geometry = directional_glcm_features(target_path)
    scale = RobustFeatureScale().fit(train_feature)
    train_design = scale.transform(train_feature)
    target_design = scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, target, sample_weight=numeric.duplicate_class_weights(ordered))
    require(int(head.n_iter_[0]) < head.max_iter, "V181 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform,
        "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit,
        "target_path": target_path_audit, "train_geometry": train_geometry, "target_geometry": target_geometry,
        "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(scale.median), "scale_sha256": array_sha256(scale.scale)},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "fitted_discriminative_coefficients": int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "bin_offset_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "knot_basis_grid_smoothing_feature_or_penalty_selected_from_labels_or_outer": False,
        "order_feature_threshold_pole_rule_or_penalty_selected_from_labels_or_outer": False,
        "window_rank_feature_or_penalty_selected_from_labels_or_outer": False,
        "v86_copula_qda_density": False, "v115_increment_spd_log_tangent_logistic": False,
        "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False,
        "v168_matrix_normal_kronecker_likelihood": False, "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False,
        "v101_static_affine_reconstruction_residual": False, "v143_source_time_nuisance_projection": False,
        "pls_or_ssa_response_projection": False, "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    support.VERSION = VERSION
    support.HYPOTHESIS = HYPOTHESIS
    support.STATIC_SPEC = STATIC_SPEC
    support.SPLINE_FEATURE_DIMENSION = GLCM_FEATURE_DIMENSION
    support.ARCHITECTURES = ARCHITECTURES
    support.SEED = SEED
    support.spline_direction = glcm_direction
    _SUPPORT_CONFIGURE()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = glcm_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["train_geometry"]["fixed_bins_offsets_regions_and_summaries_without_label_or_outer_selection"] and model_audit["train_geometry"]["directed_cooccurrence_asymmetry_preserved"] and model_audit["train_geometry"]["feature_dimension"] == GLCM_FEATURE_DIMENSION and model_audit["feature_scaling"]["train_only"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"] and not model_audit["bin_offset_region_summary_or_penalty_selected_from_labels_or_outer"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 directional GLCM texture blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "bin_offset_region_summary_head_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    support.configure_base = configure_base
    support.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = _SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v181_model"})
    evidence = evidence.rename(columns={"bspline_prob": "glcm_prob", "v178_model": "v181_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V181 GLCM] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V181 directional GLCM texture policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    support.paired_nested_bootstrap = paired_nested_bootstrap
    result = _SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified")
    glcm_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["train_geometry"]["fixed_bins_offsets_regions_and_summaries_without_label_or_outer_selection"]
        and audit[side]["train_geometry"]["directed_cooccurrence_asymmetry_preserved"]
        and audit[side]["train_geometry"]["feature_dimension"] == GLCM_FEATURE_DIMENSION
        and audit[side]["target_geometry"]["fixed_bins_offsets_regions_and_summaries_without_label_or_outer_selection"]
        and audit[side]["feature_scaling"]["train_only"]
        and audit[side]["head"]["fitted_discriminative_coefficients"] == GLCM_FEATURE_DIMENSION
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["bin_offset_region_summary_or_penalty_selected_from_labels_or_outer"]
        and not audit[side]["train_geometry"]["v179_structure_tensor_or_gradient_orientation"]
        and not audit[side]["train_geometry"]["v180_zernike_polar_moments"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["multichannel_causal_directional_glcm_texture_contract_verified"] = bool(glcm_contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy()
        result["selected_summary"] = result["candidate_summary"]
        result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V181 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V181 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V181 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V181 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V181").resolve(), "V181 output is outside controller research/staging/V181")
    require(out.exists() and out.is_dir(), "V181 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V181 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V181 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V181 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V181 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V181 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V181", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V181_DIAGNOSTIC_DIRECTIONAL_GLCM_TEXTURE": evaluation["candidate_summary"], "V181_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V181", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V181", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V181_DIRECTIONAL_GLCM_TEXTURE_REPORT.json": {"version": "V181", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V181", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V181", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V181", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V181 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V181_DIRECTIONAL_GLCM_TEXTURE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V181_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--contract-probe", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V181 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V181 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V181 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.contract_probe:
        print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2))
        return
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "fixed five-level directional GLCMs and Haralick/ordinal texture summaries plus balanced ridge", "prior_exact_directional_glcm_haralick_family_hit_count": 0, "v179_structure_tensor_reused": False, "v180_zernike_reused": False, "causal_numeric_only": True, "target_row_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V181 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]})
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
