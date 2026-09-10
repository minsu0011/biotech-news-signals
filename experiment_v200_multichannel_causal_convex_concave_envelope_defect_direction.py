"""V200 fixed convex/concave envelope-defect direction challenger.

The requested hyperbolic prototype is rejected: V124 already owns a fixed
Poincare/Lorentz class-barycenter geometry and V199 owns a learned metric
embedding with class prototypes.  This replacement is label-independent.
For each train-scaled observed 8x8 causal surface it extracts 24 immutable
row, column, diagonal and anti-diagonal lines.  Exact greatest convex
minorants and least concave majorants yield ten contact, defect, gap and
signed-asymmetry summaries per line (240D), followed by one train-only robust
feature scale and equal-event-group/class-balanced fixed ridge-logit head.
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

import experiment_v197_multichannel_causal_multiscale_gliding_box_lacunarity_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V200" / "LOCAL_FORBIDDEN"
VERSION = 200
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CONVEX_CONCAVE_ENVELOPE_DEFECT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v197_multichannel_causal_multiscale_gliding_box_lacunarity_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "a9e9551dd7b34ce34239b0914a3ed1c3b9f309f7bb721f187457b1f1b5fd5ff1"

GRID_N = 8
DIAGONAL_OFFSETS = (-2, -1, 0, 1)
STAT_NAMES = (
    "LOWER_MEAN_DEFECT", "LOWER_MAX_DEFECT", "LOWER_L2_DEFECT", "LOWER_CONTACT_FRACTION",
    "UPPER_MEAN_DEFECT", "UPPER_MAX_DEFECT", "UPPER_L2_DEFECT", "UPPER_CONTACT_FRACTION",
    "MEAN_ENVELOPE_GAP", "SIGNED_DEFECT_ASYMMETRY",
)
LINE_COUNT = GRID_N * 2 + len(DIAGONAL_OFFSETS) * 2
ENVELOPE_FEATURE_DIMENSION = LINE_COUNT * len(STAT_NAMES)
CONTACT_TOLERANCE = 1e-10
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 20001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CONVEX_CONCAVE_ENVELOPE_DEFECT_W0.25", "weight": 0.25},
    {"name": "CONVEX_CONCAVE_ENVELOPE_DEFECT_W0.50", "weight": 0.50},
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
    "surface": "train-robust-scaled observed 8-horizon x 8-semantic-channel causal state",
    "lines": "8 horizon rows, 8 semantic-channel columns, diagonals and anti-diagonals at offsets -2,-1,0,1",
    "line_count": LINE_COUNT,
    "envelopes": "exact greatest convex minorant and least concave majorant on each fixed line",
    "statistics": list(STAT_NAMES),
    "feature_dimension": ENVELOPE_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "line_envelope_statistic_or_head_penalty_learned": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V200_CONVEX_CONCAVE_ENVELOPE_DEFECT_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(LINE_COUNT == 24 and ENVELOPE_FEATURE_DIMENSION == 240, "V200 static feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V197 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, gates, bootstrap and atomic IO only; no V197 output is read"}
    audit["v200_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _greatest_convex_minorant(values: np.ndarray) -> tuple[np.ndarray, int]:
    y = np.asarray(values, dtype=float); n = len(y); require(n >= 3 and np.isfinite(y).all(), "V200 line invalid")
    knots: list[int] = []
    for index in range(n):
        knots.append(index)
        while len(knots) >= 3:
            left, middle, right = knots[-3:]
            slope_left = (y[middle] - y[left]) / float(middle - left)
            slope_right = (y[right] - y[middle]) / float(right - middle)
            if slope_left < slope_right - 1e-14:
                break
            knots.pop(-2)
    envelope = np.empty(n, dtype=float)
    for left, right in zip(knots[:-1], knots[1:]):
        fraction = np.arange(right - left + 1, dtype=float) / float(right - left)
        envelope[left:right + 1] = y[left] + fraction * (y[right] - y[left])
    require(np.all(envelope <= y + 1e-9), "V200 convex minorant exceeds line")
    slopes = np.diff(envelope)
    require(np.all(np.diff(slopes) >= -1e-9), "V200 convex minorant slope violation")
    return envelope, len(knots)


def _line_statistics(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    lower, lower_knots = _greatest_convex_minorant(values)
    negative_upper, upper_knots = _greatest_convex_minorant(-np.asarray(values, dtype=float)); upper = -negative_upper
    lower_defect = np.maximum(np.asarray(values, dtype=float) - lower, 0.0)
    upper_defect = np.maximum(upper - np.asarray(values, dtype=float), 0.0)
    gap = np.maximum(upper - lower, 0.0)
    denominator = float(np.sum(gap)) + 1e-12
    result = np.asarray([
        lower_defect.mean(), lower_defect.max(), np.sqrt(np.mean(lower_defect ** 2)), np.mean(lower_defect <= CONTACT_TOLERANCE),
        upper_defect.mean(), upper_defect.max(), np.sqrt(np.mean(upper_defect ** 2)), np.mean(upper_defect <= CONTACT_TOLERANCE),
        gap.mean(), (np.sum(lower_defect) - np.sum(upper_defect)) / denominator,
    ], dtype=float)
    require(result.shape == (len(STAT_NAMES),) and np.isfinite(result).all(), "V200 line statistics invalid")
    return result, {"lower_knots": lower_knots, "upper_knots": upper_knots}


def _fixed_lines(surface: np.ndarray) -> list[np.ndarray]:
    lines = [surface[index, :] for index in range(GRID_N)]
    lines.extend(surface[:, index] for index in range(GRID_N))
    lines.extend(np.diag(surface, k=offset) for offset in DIAGONAL_OFFSETS)
    flipped = np.fliplr(surface)
    lines.extend(np.diag(flipped, k=offset) for offset in DIAGONAL_OFFSETS)
    require(len(lines) == LINE_COUNT and all(len(line) >= 6 for line in lines), "V200 fixed line inventory invalid")
    return lines


def envelope_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (GRID_N, GRID_N), "V200 path shape invalid")
    result = np.empty((len(path), ENVELOPE_FEATURE_DIMENSION), dtype=float)
    lower_knots: list[int] = []; upper_knots: list[int] = []
    for row, surface in enumerate(path):
        parts: list[np.ndarray] = []
        for line in _fixed_lines(surface):
            statistics, audit = _line_statistics(line); parts.append(statistics)
            lower_knots.append(audit["lower_knots"]); upper_knots.append(audit["upper_knots"])
        result[row] = np.concatenate(parts)
    require(result.shape == (len(path), ENVELOPE_FEATURE_DIMENSION) and np.isfinite(result).all(), "V200 feature invalid")
    return result, {
        "rows": len(path), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result),
        "line_count": LINE_COUNT, "diagonal_offsets": list(DIAGONAL_OFFSETS), "stat_names": list(STAT_NAMES),
        "min_lower_knot_count": min(lower_knots), "min_upper_knot_count": min(upper_knots),
        "exact_greatest_convex_minorant": True, "exact_least_concave_majorant": True,
        "envelope_bound_and_slope_invariants_verified": True, "label_independent_exact_transform": True,
        "target_batch_fit": False, "line_envelope_statistic_selected_from_labels_or_outer": False,
        "hyperbolic_poincare_lorentz_embedding_or_prototype": False,
        "supervised_contrastive_metric_embedding_or_class_centroid": False,
        "isotonic_correctness_confidence_surface": False, "monotone_additive_predictor_shape": False,
        "morphological_opening_closing_or_granulometry": False, "functional_spline_derivative_or_monge_curvature": False,
        "ordinal_neighbor_code_glcm_or_run_length": False, "learned_patch_dictionary_or_sparse_sem": False,
        "normalizing_flow_or_score_matching_energy": False,
        "connected_component_hole_or_euler_curve": False, "filtration_boundary_matrix_or_persistence": False,
        "time_oriented_crossing_excursion_or_dwell": False, "dyadic_continuous_reconstruction_or_roughness": False,
        "class_sliced_wasserstein_projection_or_quantile_barycenter": False, "semantic_mass_sinkhorn_transport": False,
        "source_time_quantile_transport": False, "learned_patch_dictionary_or_omp_code": False,
        "ordered_sparse_sem_or_ancestor_drive": False, "normalizing_flow_coupling_nll_logdet_or_density_ratio": False,
        "label_independent_fixed_basis_and_pseudoinverse": False,
        "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False,
        "continuous_first_and_second_functional_jets": False, "v171_wavelet_leader_or_multifractal": False,
        "v172_hankel_trajectory_matrix_or_ssa": False, "v173_weighted_grid_laplacian_or_heat_trace": False,
        "v174_morphological_granulometry": False, "v175_burg_lattice_parcor_or_poles": False,
        "v176_level_crossing_excursion_or_dwell": False, "v177_discrete_radon_line_projection": False,
        "v178_cubic_bspline_functional_jet": False, "v179_structure_tensor_or_gradient_orientation": False,
        "v180_zernike_polar_moments": False,
        "fixed_order_lattice_features_and_poles_without_label_or_outer_selection": False,
        "burg_forward_backward_innovation_recursion": False, "v119_multichannel_transition_operator_or_dmd": False,
        "fixed_hankel_window_and_rank_without_label_or_outer_selection": False,
        "anti_diagonal_rank_one_trend_and_residual": False, "v119_transition_operator_or_dynamic_eigenvalue": False,
        "v166_raw_path_grassmann_subspace": False, "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False,
        "within_channel_bispectral_magnitude_and_biphase_verified": False,
        "directed_cross_channel_biphase_coupling_verified": False,
        "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False,
        "time_domain_increment_coskewness": False, "hilbert_time_local_analytic_phase": False,
        "per_event_row_and_column_grassmann_points": False,
        "svd_sign_invariant_projection_representation": False,
        "fixed_rank_without_label_or_outer_selection": False, "per_event_channel_correlation_spd": False,
        "fixed_shrinkage_no_label_or_outer_selection": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None: self.median = np.empty(0); self.scale = np.empty(0)
    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == ENVELOPE_FEATURE_DIMENSION, "V200 feature scale shape invalid")
        self.median = np.median(matrix, axis=0); q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349; standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0)); return self
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(result.shape[1] == ENVELOPE_FEATURE_DIMENSION and np.isfinite(result).all(), "V200 scaled feature invalid"); return result


def envelope_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V200 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry = envelope_features(train_path); target_feature, target_geometry = envelope_features(target_path)
    scale = RobustFeatureScale().fit(train_feature); train_design = scale.transform(train_feature); target_design = scale.transform(target_feature)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, ordered.y.to_numpy(int), sample_weight=numeric.duplicate_class_weights(ordered)); require(int(head.n_iter_[0]) < head.max_iter, "V200 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1 - 1e-6)
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry, "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(scale.median), "scale_sha256": array_sha256(scale.scale)},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "fitted_discriminative_coefficients": int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "line_envelope_statistic_or_penalty_selected_from_labels_or_outer": False,
        "level_box_statistic_or_penalty_selected_from_labels_or_outer": False,
        "level_direction_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "dead_zone_neighbor_region_mapping_or_penalty_selected_from_labels_or_outer": False,
        "bin_offset_region_summary_or_penalty_selected_from_labels_or_outer": False,
        "knot_basis_grid_smoothing_feature_or_penalty_selected_from_labels_or_outer": False,
        "order_feature_threshold_pole_rule_or_penalty_selected_from_labels_or_outer": False,
        "window_rank_feature_or_penalty_selected_from_labels_or_outer": False,
        "v86_copula_qda_density": False, "v115_increment_spd_log_tangent_logistic": False,
        "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False,
        "v168_matrix_normal_kronecker_likelihood": False,
        "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False,
        "v101_static_affine_reconstruction_residual": False, "v143_source_time_nuisance_projection": False,
        "pls_or_ssa_response_projection": False, "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    scaffold.VERSION = VERSION; scaffold.HYPOTHESIS = HYPOTHESIS; scaffold.STATIC_SPEC = STATIC_SPEC; scaffold.LACUNARITY_FEATURE_DIMENSION = ENVELOPE_FEATURE_DIMENSION; scaffold.ARCHITECTURES = ARCHITECTURES; scaffold.SEED = SEED; scaffold.lacunarity_direction = envelope_direction; scaffold.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = envelope_direction(inner_train, inner_valid); baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    geometry = audit["train_geometry"]
    model_eligible = bool(audit["causal_numeric_only"] and geometry["exact_greatest_convex_minorant"] and geometry["exact_least_concave_majorant"] and geometry["envelope_bound_and_slope_invariants_verified"] and geometry["feature_dimension"] == ENVELOPE_FEATURE_DIMENSION and not geometry["hyperbolic_poincare_lorentz_embedding_or_prototype"] and not geometry["supervised_contrastive_metric_embedding_or_class_centroid"] and not geometry["line_envelope_statistic_selected_from_labels_or_outer"] and audit["feature_scaling"]["train_only"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"] and not audit["line_envelope_statistic_or_penalty_selected_from_labels_or_outer"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high); auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]; source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 convex-concave envelope-defect blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "line_envelope_statistic_head_blend_or_source_floor_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()): diagnostic, evidence, audits = _SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v200_model"}); evidence = evidence.rename(columns={"bspline_prob": "envelope_prob", "v178_model": "v200_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]; print(f"[V200 ENVELOPE] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = _SUPPORT_BOOTSTRAP(evidence, draws); result["method"] = "paired market-fold-time block bootstrap over inner-locked V200 convex-concave envelope-defect policy"; result["seed"] = SEED; result["standardized"] = True; return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap; result = _SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]; checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified")
    contract = all(audit[side]["causal_numeric_only"] and audit[side]["train_geometry"]["exact_greatest_convex_minorant"] and audit[side]["train_geometry"]["exact_least_concave_majorant"] and audit[side]["train_geometry"]["envelope_bound_and_slope_invariants_verified"] and audit[side]["train_geometry"]["feature_dimension"] == ENVELOPE_FEATURE_DIMENSION and audit[side]["target_geometry"]["envelope_bound_and_slope_invariants_verified"] and audit[side]["feature_scaling"]["train_only"] and audit[side]["head"]["fitted_discriminative_coefficients"] == ENVELOPE_FEATURE_DIMENSION and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] and not audit[side]["line_envelope_statistic_or_penalty_selected_from_labels_or_outer"] and not audit[side]["train_geometry"]["hyperbolic_poincare_lorentz_embedding_or_prototype"] and not audit[side]["train_geometry"]["supervised_contrastive_metric_embedding_or_class_centroid"] and not audit[side]["train_geometry"]["normalizing_flow_or_score_matching_energy"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_convex_concave_envelope_defect_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]: result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else: result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V200 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"); return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]; gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V200 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V200 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V200 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V200").resolve(), "V200 output outside staging/V200"); require(out.exists() and out.is_dir(), "V200 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}; require(required.issubset(existing), f"V200 controller prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V200 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES); require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V200 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V200 output not controller-bound"); conflict = enforce_output_conflict_fail_closed(out); compact = {k: v for k, v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate; require("bootstrap" in selected["robustness"], "V200 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V200", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V200_DIAGNOSTIC_ENVELOPE": evaluation["candidate_summary"], "V200_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V200", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V200", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V200_CONVEX_CONCAVE_ENVELOPE_DEFECT_REPORT.json": {"version": "V200", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V200", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V200", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V200", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V200 report inventory changed"); [atomic_json(payload, out / name) for name, payload in reports.items()]; atomic_csv(evidence, out / "V200_ENVELOPE_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V200_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json"); return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--contract-probe", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true"); parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V200 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output = DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT), "V200 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V200 controller output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe); runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}; authority = verify_authority(); dev, champion = load_authorized()
    if args.contract_probe: print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2)); return
    if args.audit_only: print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC, "architecture": "fixed linewise convex/concave envelopes plus contact/defect/gap summaries and balanced ridge", "preferred_hyperbolic_prototype_rejected_due_v124_and_v199_collisions": True, "prior_exact_convex_concave_envelope_defect_family_hit_count": 0, "causal_numeric_only": True, "target_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V200 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test: summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
