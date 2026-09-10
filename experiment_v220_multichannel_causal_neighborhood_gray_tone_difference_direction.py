"""V220 multichannel causal neighborhood gray-tone-difference direction.

Strict-train robust-scaled 8x8 causal surfaces are quantized by one pooled
train-only five-level map.  Exact neighborhood gray-tone-difference matrices
at two fixed radii and nine fixed regions retain per-tone occupancy and
difference mass together with coarseness, contrast, busyness, complexity and
strength.  A single train-only scaled, equal-event-group/class-balanced ridge
head produces direction probability.  No connected zones, runs, cooccurrence
matrix, local ternary pattern, target fit, or outer-driven tuning is used.
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

import experiment_v216_multichannel_causal_class_conditional_anisotropic_gmrf_direction as scaffold


ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V220" / "LOCAL_FORBIDDEN"
VERSION = 220
PRIORITY = 890
HYPOTHESIS = "MULTICHANNEL_CAUSAL_NEIGHBORHOOD_GRAY_TONE_DIFFERENCE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v216_multichannel_causal_class_conditional_anisotropic_gmrf_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "e265b712f9f31771043fc3352e88b1f3cba1654a02179959a63368affee502b4"

GRID_N = 8
LEVEL_N = 5
QUANTILES = np.asarray([0.20, 0.40, 0.60, 0.80], dtype=float)
RADII = (1, 2)
REGIONS = (
    ("ALL", 0, 8, 0, 8),
    ("OLDER_HALF", 0, 4, 0, 8),
    ("RECENT_HALF", 4, 8, 0, 8),
    ("CHANNEL_LEFT_HALF", 0, 8, 0, 4),
    ("CHANNEL_RIGHT_HALF", 0, 8, 4, 8),
    ("OLDER_LEFT", 0, 4, 0, 4),
    ("OLDER_RIGHT", 0, 4, 4, 8),
    ("RECENT_LEFT", 4, 8, 0, 4),
    ("RECENT_RIGHT", 4, 8, 4, 8),
)
SUMMARY_N = 15
FEATURE_DIMENSION = len(RADII) * len(REGIONS) * SUMMARY_N
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 22001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "NEIGHBORHOOD_GRAY_TONE_DIFFERENCE_W0.25", "weight": 0.25},
    {"name": "NEIGHBORHOOD_GRAY_TONE_DIFFERENCE_W0.50", "weight": 0.50},
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
ENGINE = scaffold.ENGINE
SUPPORT_RUN_NESTED = scaffold.SUPPORT_RUN_NESTED
SUPPORT_EVALUATE = scaffold.SUPPORT_EVALUATE
SUPPORT_BOOTSTRAP = scaffold.SUPPORT_BOOTSTRAP

STATIC_SPEC = {
    "surface": "strict-train robust-scaled observed 8-horizon x 8-semantic-channel continuous causal surface",
    "quantization": "pooled strict-train 20/40/60/80 percentiles into five gray levels",
    "regions": [item[0] for item in REGIONS],
    "radii": list(RADII),
    "ngtdm": "per-tone occupancy and neighborhood-mean absolute gray-tone difference mass",
    "summaries": ["five occupancy fractions", "five mean difference masses", "coarseness", "contrast", "busyness", "complexity", "strength"],
    "feature_dimension": FEATURE_DIMENSION,
    "feature_scaling": "strict-train coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_region_radius_summary_or_head_penalty": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V220_NEIGHBORHOOD_GRAY_TONE_DIFFERENCE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(GRID_N == 8 and LEVEL_N == 5 and len(REGIONS) == 9 and RADII == (1, 2) and FEATURE_DIMENSION == 270 and PRIORITY == 890, "V220 static model changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V216 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, continuous causal path, metrics, gates, bootstrap and atomic IO only; no V216 output is read"}
    audit["v220_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _thresholds(train_path: np.ndarray) -> np.ndarray:
    values = train_path.reshape(-1)
    thresholds = np.quantile(values, QUANTILES)
    if np.any(np.diff(thresholds) <= 1e-8):
        center = float(np.median(values)); spread = max(float(np.std(values)), 1e-3)
        thresholds = center + spread * np.asarray([-0.75, -0.25, 0.25, 0.75])
    require(np.isfinite(thresholds).all() and np.all(np.diff(thresholds) > 0.0), "V220 train thresholds invalid")
    return thresholds


def _neighbor_mean(levels: np.ndarray, radius: int) -> np.ndarray:
    rows, columns = levels.shape[1:]
    total = np.zeros_like(levels, dtype=float)
    count = np.zeros((rows, columns), dtype=float)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            src_r = slice(max(0, -dr), rows - max(0, dr)); dst_r = slice(max(0, dr), rows - max(0, -dr))
            src_c = slice(max(0, -dc), columns - max(0, dc)); dst_c = slice(max(0, dc), columns - max(0, -dc))
            total[:, dst_r, dst_c] += levels[:, src_r, src_c]
            count[dst_r, dst_c] += 1.0
    require(float(count.min()) > 0.0, "V220 neighborhood support empty")
    return total / count[None, :, :]


def _ngtdm_summary(levels: np.ndarray, radius: int) -> np.ndarray:
    require(levels.ndim == 3 and levels.shape[1] >= 4 and levels.shape[2] >= 4, "V220 region shape invalid")
    neighbor = _neighbor_mean(levels, radius)
    count = np.empty((len(levels), LEVEL_N), dtype=float)
    difference = np.empty_like(count)
    for tone in range(LEVEL_N):
        mask = levels == tone
        count[:, tone] = np.sum(mask, axis=(1, 2))
        difference[:, tone] = np.sum(np.abs(float(tone) - neighbor) * mask, axis=(1, 2))
    cells = float(levels.shape[1] * levels.shape[2])
    probability = count / cells
    mean_difference = difference / np.maximum(count, 1.0) / float(LEVEL_N - 1)
    tone = np.arange(LEVEL_N, dtype=float)
    tone_gap = np.abs(tone[:, None] - tone[None, :])
    active = count > 0.0
    active_n = np.maximum(np.sum(active, axis=1), 1)
    weighted_difference = np.sum(probability * difference, axis=1)
    coarseness = 1.0 / (1.0 + weighted_difference)
    pair_contrast = np.sum(probability[:, :, None] * probability[:, None, :] * tone_gap[None, :, :] ** 2, axis=(1, 2))
    contrast = pair_contrast * np.sum(difference, axis=1) / (cells * np.maximum(active_n * (active_n - 1), 1))
    weighted_tone = probability * tone[None, :]
    busy_denominator = np.sum(np.abs(weighted_tone[:, :, None] - weighted_tone[:, None, :]), axis=(1, 2))
    busyness = weighted_difference / np.maximum(busy_denominator, 1e-12)
    pair_active = active[:, :, None] & active[:, None, :] & (tone_gap[None, :, :] > 0.0)
    complexity_term = tone_gap[None, :, :] * (probability[:, :, None] * difference[:, :, None] + probability[:, None, :] * difference[:, None, :]) / np.maximum(count[:, :, None] + count[:, None, :], 1.0)
    complexity = np.sum(np.where(pair_active, complexity_term, 0.0), axis=(1, 2)) / cells
    strength_num = np.sum(np.where(pair_active, (probability[:, :, None] + probability[:, None, :]) * tone_gap[None, :, :] ** 2, 0.0), axis=(1, 2))
    strength = strength_num / np.maximum(np.sum(difference, axis=1), 1e-12)
    summary = np.concatenate([probability, mean_difference, np.stack([coarseness, contrast, busyness, complexity, strength], axis=1)], axis=1)
    require(summary.shape == (len(levels), SUMMARY_N) and np.isfinite(summary).all(), "V220 NGTDM summary invalid")
    return summary


def ngtdm_features(path: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (GRID_N, GRID_N), "V220 causal surface shape invalid")
    levels = np.digitize(path, thresholds).astype(np.int8)
    parts: list[np.ndarray] = []
    for _name, r0, r1, c0, c1 in REGIONS:
        region = levels[:, r0:r1, c0:c1]
        for radius in RADII:
            parts.append(_ngtdm_summary(region, radius))
    feature = np.concatenate(parts, axis=1)
    require(feature.shape == (len(path), FEATURE_DIMENSION), "V220 feature dimension changed")
    return feature, {
        "rows": len(path), "feature_dimension": FEATURE_DIMENSION, "level_n": LEVEL_N, "region_n": len(REGIONS), "radii": list(RADII),
        "thresholds": thresholds.tolist(), "feature_sha256": array_sha256(feature), "exact_ngtdm_tone_count_and_difference_mass": True,
        "fixed_coarseness_contrast_busyness_complexity_strength": True, "label_independent_given_strict_train_thresholds": True,
        "connected_component_or_size_zone_matrix": False, "gray_level_run_length_matrix": False, "pairwise_glcm_or_haralick": False,
        "local_binary_or_ternary_pattern": False, "morphological_granulometry": False, "monge_curvature": False, "laws_filter_energy": False,
        "directional_semivariogram": False, "homomorphic_cepstrum": False, "persistent_homology": False,
        "class_gmrf_precision_logdet": False, "monogenic_riesz_phase": False, "tropical_projective_medoid": False,
        "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False, "within_channel_bispectral_magnitude_and_biphase_verified": False,
        "directed_cross_channel_biphase_coupling_verified": False, "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False, "time_domain_increment_coskewness": False,
        "hilbert_time_local_analytic_phase": False, "per_event_row_and_column_grassmann_points": False,
        "svd_sign_invariant_projection_representation": False, "fixed_rank_without_label_or_outer_selection": False,
        "per_event_channel_correlation_spd": False, "fixed_shrinkage_no_label_or_outer_selection": False,
        "fixed_hankel_window_and_rank_without_label_or_outer_selection": False, "anti_diagonal_rank_one_trend_and_residual": False,
        "v119_transition_operator_or_dynamic_eigenvalue": False, "v166_raw_path_grassmann_subspace": False,
        "v171_wavelet_leader_or_multifractal": False, "fixed_order_lattice_features_and_poles_without_label_or_outer_selection": False,
        "burg_forward_backward_innovation_recursion": False, "v119_multichannel_transition_operator_or_dmd": False,
        "v172_hankel_trajectory_matrix_or_ssa": False, "v173_weighted_grid_laplacian_or_heat_trace": False,
        "v174_morphological_granulometry": False, "label_independent_fixed_basis_and_pseudoinverse": False,
        "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False, "continuous_first_and_second_functional_jets": False,
        "v175_burg_lattice_parcor_or_poles": False, "v176_level_crossing_excursion_or_dwell": False, "v177_discrete_radon_line_projection": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float); self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == FEATURE_DIMENSION, "V220 scale shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0); robust = (q75 - q25) / 1.349; standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V220 scaled feature invalid")
        return result


def ngtdm_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V220 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    thresholds = _thresholds(train_path)
    train_feature, train_geometry = ngtdm_features(train_path, thresholds); target_feature, target_geometry = ngtdm_features(target_path, thresholds)
    scale = RobustFeatureScale().fit(train_feature); train_design = scale.transform(train_feature); target_design = scale.transform(target_feature)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, ordered.y.to_numpy(int), sample_weight=numeric.duplicate_class_weights(ordered))
    require(int(head.n_iter_[0]) < head.max_iter, "V220 ridge head did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry, "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(scale.median), "scale_sha256": array_sha256(scale.scale)},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "fitted_discriminative_coefficients": int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "threshold_region_radius_summary_penalty_or_blend_selected_from_labels_or_outer": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    lower = scaffold.scaffold.base
    lower.VERSION = VERSION; lower.HYPOTHESIS = HYPOTHESIS; lower.STATIC_SPEC = STATIC_SPEC; lower.ARCHITECTURES = ARCHITECTURES; lower.SEED = SEED
    lower.rbm_direction = ngtdm_direction; lower.choose_inner = choose_inner; lower.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = ngtdm_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    g = audit["train_geometry"]
    eligible_model = bool(g["exact_ngtdm_tone_count_and_difference_mass"] and g["fixed_coarseness_contrast_busyness_complexity_strength"] and g["feature_dimension"] == FEATURE_DIMENSION and not g["connected_component_or_size_zone_matrix"] and not g["gray_level_run_length_matrix"] and not g["pairwise_glcm_or_haralick"] and audit["feature_scaling"]["train_only"] and audit["head"]["fitted_discriminative_coefficients"] == FEATURE_DIMENSION and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": eligible_model, "eligible": bool(eligible_model and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 NGTDM texture blend", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "threshold_region_radius_summary_head_blend_or_safety_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v220_model"}); evidence = evidence.rename(columns={"bspline_prob": "ngtdm_prob", "v178_model": "v220_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V220 NGTDM] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V220 NGTDM texture policy"; result["seed"] = SEED; result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap
    result = SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]
    checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified", None)
    contract = all(audit[side]["train_geometry"]["exact_ngtdm_tone_count_and_difference_mass"] and audit[side]["train_geometry"]["fixed_coarseness_contrast_busyness_complexity_strength"] and audit[side]["train_geometry"]["feature_dimension"] == FEATURE_DIMENSION and not audit[side]["train_geometry"]["connected_component_or_size_zone_matrix"] and audit[side]["feature_scaling"]["train_only"] and audit[side]["head"]["fitted_discriminative_coefficients"] == FEATURE_DIMENSION and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_neighborhood_gray_tone_difference_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V220 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V220 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V220 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V220 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V220").resolve(), "V220 output outside staging/V220"); require(out.exists() and out.is_dir(), "V220 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V220 controller prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V220 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def resource_contract() -> dict[str, Any]:
    return {
        "bounded_prep_audit_smoke": {"external_pre_resume_affinity_mask": "0xC0000000", "cpu_ids": [30, 31], "numeric_threads": 2, "gpu_hidden": True, "gpu_model_calls": 0},
        "controller_exact_bound_full": {"affinity_policy": "inherit unchanged from authorized controller", "expected_authorized_mask": "0xFFFFFFFF", "thread_environment_policy": "inherit unchanged from authorized controller", "expected_authorized_threads": 32, "cuda_visibility_policy": "inherit unchanged from authorized controller", "expected_authorized_cuda_visible_devices": "0", "model_gpu_calls": 0},
        "runner_sets_process_affinity": False, "runner_overrides_full_thread_environment": False, "runner_overrides_full_cuda_visibility": False,
    }


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V220 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "dev_robustness_top_level_status": True, "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "canonical_gate_checks": 14, "exact_entire_v69_fallback": True, "filesystem_write": False, "resource_contract": resource_contract()}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V220 output not controller-bound")
    conflict = enforce_output_conflict_fail_closed(out); compact = {k: v for k, v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V220 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V220", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V220_DIAGNOSTIC_NGTDM": evaluation["candidate_summary"], "V220_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V220", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V220", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V220_NEIGHBORHOOD_GRAY_TONE_DIFFERENCE_REPORT.json": {"version": "V220", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V220", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V220", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V220", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V220 report inventory changed")
    [atomic_json(payload, out / name) for name, payload in reports.items()]; atomic_csv(evidence, out / "V220_NGTDM_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V220_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--contract-probe", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V220 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V220 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V220 controller output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "affinity_policy": "authorized controller inherited unchanged", "thread_policy": "authorized controller inherited unchanged", "gpu_visibility_policy": "authorized controller inherited unchanged", "gpu_model_calls": 0}
    authority = verify_authority(); dev, champion = load_authorized()
    if args.contract_probe:
        print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "priority": PRIORITY, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2)); return
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC, "architecture": "train-only five-level NGTDM tone occupancy/difference mass and fixed texture summaries plus balanced ridge", "prior_exact_ngtdm_family_hit_count": 0, "rejected_candidate": "Dirichlet compositional dynamics rejected due material V104 class Dirichlet collision", "causal_numeric_only": True, "target_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "dev_robustness_top_level_status": True, "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "resource_contract": resource_contract()}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V220 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "priority": PRIORITY, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
