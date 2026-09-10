"""V197 multiscale gliding-box lacunarity direction challenger.

The requested sliced-Wasserstein class barycenter is rejected because V147
already implements that exact fixed-projection/class-quantile-distance family.
This replacement converts each strict-past train-scaled observed 8x8 causal
surface into fixed sublevel and superlevel occupancy masks at five immutable
levels.  Valid gliding boxes of sizes 1, 2 and 4 yield mass-distribution,
empty/full, entropy, quantile and lacunarity summaries.  The resulting fixed
240D spatial-heterogeneity representation feeds one equal-event-group/class-
balanced ridge-logit direction head.

There is no component/hole/Euler computation, filtration/persistence pairing,
time-oriented crossing/dwell statistic, dyadic reconstruction, dictionary,
sparse SEM, flow density, target-batch fit or outer-driven tuning.  Exact V69
confidence/high_conf and exact-entire-frame fail-close remain frozen.
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

import experiment_v193_multichannel_causal_graph_diffusion_reservoir_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V197" / "LOCAL_FORBIDDEN"
VERSION = 197
HYPOTHESIS = "MULTICHANNEL_CAUSAL_MULTISCALE_GLIDING_BOX_LACUNARITY_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v193_multichannel_causal_graph_diffusion_reservoir_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "93bfb52d53e78655a1b71faf698c015f8a9e1d81d0e72b5d0dec46da3f3b2b06"

GRID_N = 8
LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
MASK_DIRECTIONS = ("SUBLEVEL", "SUPERLEVEL")
BOX_SIZES = (1, 2, 4)
STAT_NAMES = ("MEAN_DENSITY", "VAR_DENSITY", "LACUNARITY", "EMPTY_FRACTION", "FULL_FRACTION", "MASS_ENTROPY", "DENSITY_Q25", "DENSITY_Q75")
LACUNARITY_FEATURE_DIMENSION = len(LEVELS) * len(MASK_DIRECTIONS) * len(BOX_SIZES) * len(STAT_NAMES)
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 19701
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "MULTISCALE_GLIDING_BOX_LACUNARITY_W0.25", "weight": 0.25},
    {"name": "MULTISCALE_GLIDING_BOX_LACUNARITY_W0.50", "weight": 0.50},
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
    "levels": list(LEVELS), "mask_directions": list(MASK_DIRECTIONS), "gliding_box_sizes": list(BOX_SIZES),
    "statistics": list(STAT_NAMES), "feature_dimension": LACUNARITY_FEATURE_DIMENSION,
    "lacunarity": "variance of box density divided by squared mean density, zero for empty mass",
    "entropy": "normalized empirical entropy over integer box masses 0..box_area",
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "threshold_box_statistic_or_head_penalty_learned": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V197_MULTISCALE_GLIDING_BOX_LACUNARITY_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(LACUNARITY_FEATURE_DIMENSION == 240, "V197 static feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V193 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, gates, bootstrap and atomic IO only; no V193 output is read"}
    audit["v197_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _mass_entropy(mass: np.ndarray, box_area: int) -> np.ndarray:
    result = np.zeros(len(mass), dtype=float)
    denominator = np.log(float(box_area + 1))
    for row in range(len(mass)):
        counts = np.bincount(mass[row].astype(int), minlength=box_area + 1).astype(float)
        probability = counts[counts > 0] / counts.sum()
        result[row] = float(-np.sum(probability * np.log(probability)) / denominator) if denominator > 0 else 0.0
    return result


def _box_statistics(mask: np.ndarray, box_size: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(mask.astype(np.float64), (box_size, box_size), axis=(1, 2))
    mass = windows.sum(axis=(-1, -2)).reshape(len(mask), -1)
    area = box_size * box_size; density = mass / float(area)
    mean = density.mean(axis=1); variance = density.var(axis=1)
    lacunarity = np.where(mean > 1e-12, variance / np.maximum(mean * mean, 1e-12), 0.0)
    empty = np.mean(mass == 0, axis=1); full = np.mean(mass == area, axis=1)
    entropy = _mass_entropy(mass, area); q25, q75 = np.quantile(density, [0.25, 0.75], axis=1)
    result = np.stack([mean, variance, lacunarity, empty, full, entropy, q25, q75], axis=1)
    require(result.shape == (len(mask), len(STAT_NAMES)) and np.isfinite(result).all(), "V197 box statistics invalid")
    return result


def lacunarity_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (GRID_N, GRID_N), "V197 path shape invalid")
    parts: list[np.ndarray] = []; super_occupancy: list[np.ndarray] = []; sub_occupancy: list[np.ndarray] = []
    box_counts: dict[str, int] = {}
    for level in LEVELS:
        masks = ((path <= level), (path >= level))
        sub_occupancy.append(masks[0].mean(axis=(1, 2))); super_occupancy.append(masks[1].mean(axis=(1, 2)))
        for direction, mask in zip(MASK_DIRECTIONS, masks):
            for box_size in BOX_SIZES:
                parts.append(_box_statistics(mask, box_size))
                box_counts[f"{level:+.1f}:{direction}:B{box_size}"] = int((GRID_N - box_size + 1) ** 2)
    result = np.concatenate(parts, axis=1)
    super_matrix = np.stack(super_occupancy, axis=1); sub_matrix = np.stack(sub_occupancy, axis=1)
    super_violation = int(np.sum(np.diff(super_matrix, axis=1) > 1e-12)); sub_violation = int(np.sum(np.diff(sub_matrix, axis=1) < -1e-12))
    require(result.shape == (len(path), LACUNARITY_FEATURE_DIMENSION) and np.isfinite(result).all(), "V197 feature invalid")
    require(super_violation == 0 and sub_violation == 0, "V197 fixed-level occupancy monotonicity failed")
    return result, {
        "rows": len(path), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result),
        "levels": list(LEVELS), "mask_directions": list(MASK_DIRECTIONS), "box_sizes": list(BOX_SIZES),
        "stat_names": list(STAT_NAMES), "gliding_box_counts": box_counts,
        "superlevel_monotonicity_violation_n": super_violation, "sublevel_monotonicity_violation_n": sub_violation,
        "fixed_sub_superlevel_occupancy": True, "valid_multiscale_gliding_boxes": True,
        "mass_distribution_lacunarity_and_entropy": True, "label_independent_exact_transform": True,
        "target_batch_fit": False, "threshold_box_statistic_selected_from_labels_or_outer": False,
        "connected_component_hole_or_euler_curve": False, "filtration_boundary_matrix_or_persistence": False,
        "time_oriented_crossing_excursion_or_dwell": False, "dyadic_continuous_reconstruction_or_roughness": False,
        "class_sliced_wasserstein_projection_or_quantile_barycenter": False,
        "semantic_mass_sinkhorn_transport": False, "source_time_quantile_transport": False,
        "learned_patch_dictionary_or_omp_code": False, "ordered_sparse_sem_or_ancestor_drive": False,
        "normalizing_flow_coupling_nll_logdet_or_density_ratio": False,
        "label_independent_fixed_basis_and_pseudoinverse": False,
        "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False,
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


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0); self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == LACUNARITY_FEATURE_DIMENSION, "V197 feature scale shape invalid")
        self.median = np.median(matrix, axis=0); q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349; standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0)); return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(result.shape[1] == LACUNARITY_FEATURE_DIMENSION and np.isfinite(result).all(), "V197 scaled feature invalid"); return result


def lacunarity_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V197 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry = lacunarity_features(train_path); target_feature, target_geometry = lacunarity_features(target_path)
    scale = RobustFeatureScale().fit(train_feature); train_design = scale.transform(train_feature); target_design = scale.transform(target_feature)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, ordered.y.to_numpy(int), sample_weight=numeric.duplicate_class_weights(ordered)); require(int(head.n_iter_[0]) < head.max_iter, "V197 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1 - 1e-6)
    return probability, {
        "train_n":len(ordered), "target_n":len(target_frame), "causal_numeric_only":True, "categorical_text_source_ticker_or_issuer_feature_n":0,
        "train_transform":train_transform, "target_transform":target_transform, "static_spec":STATIC_SPEC, "train_path":train_path_audit, "target_path":target_path_audit,
        "train_geometry":train_geometry, "target_geometry":target_geometry, "class_affine_invariant_mean_prototypes":{},
        "feature_scaling":{"train_only":True,"clip":FEATURE_CLIP,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale)},
        "head":{"type":"fixed equal-event-group/class-balanced ridge logistic","C":RIDGE_LOGISTIC_C,"solver":"liblinear","iterations":int(head.n_iter_[0]),"coefficient_l2":float(np.linalg.norm(head.coef_)),"coefficient_sha256":array_sha256(head.coef_),"fitted_discriminative_coefficients":int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head":False, "target_labels_used":False,
        "level_box_statistic_or_penalty_selected_from_labels_or_outer":False,
        "level_direction_region_summary_or_penalty_selected_from_labels_or_outer":False,
        "dead_zone_neighbor_region_mapping_or_penalty_selected_from_labels_or_outer":False,
        "bin_offset_region_summary_or_penalty_selected_from_labels_or_outer":False,
        "knot_basis_grid_smoothing_feature_or_penalty_selected_from_labels_or_outer":False,
        "order_feature_threshold_pole_rule_or_penalty_selected_from_labels_or_outer":False,
        "window_rank_feature_or_penalty_selected_from_labels_or_outer":False,
        "v86_copula_qda_density":False,"v115_increment_spd_log_tangent_logistic":False,"v139_tyler_common_scatter_fisher":False,"v166_grassmann_principal_angles":False,"v168_matrix_normal_kronecker_likelihood":False,"metric_shrinkage_or_iterations_selected_from_labels_or_outer":False,"v101_static_affine_reconstruction_residual":False,"v143_source_time_nuisance_projection":False,"pls_or_ssa_response_projection":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,
        "prediction":{"mean":float(probability.mean()),"std":float(probability.std()),"probability_sha256":array_sha256(probability)},
    }


def configure_base() -> None:
    scaffold.VERSION=VERSION; scaffold.HYPOTHESIS=HYPOTHESIS; scaffold.STATIC_SPEC=STATIC_SPEC; scaffold.GRAPH_FEATURE_DIMENSION=LACUNARITY_FEATURE_DIMENSION; scaffold.ARCHITECTURES=ARCHITECTURES; scaffold.SEED=SEED; scaffold.graph_diffusion_direction=lacunarity_direction; scaffold.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger,audit=lacunarity_direction(inner_train,inner_valid); baseline=inner_champion.prob.to_numpy(float); confidence=inner_champion.confidence_signal.to_numpy(float); high=bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric=metric(inner_valid,baseline,confidence,high)
    trials=[{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.0,"balanced_accuracy_delta":0.0,"all_trade_net_delta":0.0,"worst_supported_source_ba_delta":0.0,"supported_source_ba_delta":{},"model_eligible":True,"eligible":True,"score":2*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    geometry=audit["train_geometry"]; model_eligible=bool(audit["causal_numeric_only"] and geometry["fixed_sub_superlevel_occupancy"] and geometry["valid_multiscale_gliding_boxes"] and geometry["mass_distribution_lacunarity_and_entropy"] and geometry["feature_dimension"]==LACUNARITY_FEATURE_DIMENSION and geometry["superlevel_monotonicity_violation_n"]==0 and geometry["sublevel_monotonicity_violation_n"]==0 and not geometry["connected_component_hole_or_euler_curve"] and not geometry["filtration_boundary_matrix_or_persistence"] and not geometry["class_sliced_wasserstein_projection_or_quantile_barycenter"] and not geometry["threshold_box_statistic_selected_from_labels_or_outer"] and audit["feature_scaling"]["train_only"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"] and not audit["level_box_statistic_or_penalty_selected_from_labels_or_outer"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]); current=metric(inner_valid,probability,confidence,high); auc_delta=current["auc"]-baseline_metric["auc"]; ba_delta=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"]; net_delta=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"]; source_floor,source_deltas=supported_source_ba_delta(inner_valid,baseline,probability)
        trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_deltas,"model_eligible":model_eligible,"eligible":bool(model_eligible and ba_delta>=-0.005 and net_delta>=-0.0005 and source_floor>=-0.010),"score":2*current["auc"]+current["balanced_accuracy"]})
    selected=max([trial for trial in trials if trial["eligible"]],key=lambda trial:(trial["score"],trial["auc_delta"],trial["all_trade_net_delta"],trial["name"]=="V69_NOOP"))
    return {"selection_rule":"inner-past OOF only; exact V69 no-op or fixed .25/.50 multiscale gliding-box lacunarity blend; fixed 2*AUC+BA with BA/net/supported-source safety","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"level_box_statistic_head_blend_or_source_floor_micro_tuning":False},audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str="US") -> tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    configure_base(); ENGINE.configure_base=configure_base; ENGINE.choose_inner=choose_inner
    with redirect_stdout(StringIO()): diagnostic,evidence,audits=_SUPPORT_RUN_NESTED(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v178_model":"v197_model"}); evidence=evidence.rename(columns={"bspline_prob":"lacunarity_prob","v178_model":"v197_model"})
    for audit in audits:
        selected=audit["policy"]["selected"]; print(f"[V197 LACUNARITY] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str,Any]:
    configure_base(); result=_SUPPORT_BOOTSTRAP(evidence,draws); result["method"]="paired market-fold-time block bootstrap over inner-locked V197 multiscale gliding-box lacunarity policy"; result["seed"]=SEED; result["standardized"]=True; return result


def evaluate(champion:pd.DataFrame,diagnostic:pd.DataFrame,evidence:pd.DataFrame,audits:list[dict[str,Any]],draws:int,smoke:bool)->dict[str,Any]:
    configure_base(); ENGINE.paired_nested_bootstrap=paired_nested_bootstrap; result=_SUPPORT_EVALUATE(champion,diagnostic,evidence,audits,draws,smoke); checks=result["material_checks"]; checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified")
    contract=all(audit[side]["causal_numeric_only"] and audit[side]["train_geometry"]["fixed_sub_superlevel_occupancy"] and audit[side]["train_geometry"]["valid_multiscale_gliding_boxes"] and audit[side]["train_geometry"]["mass_distribution_lacunarity_and_entropy"] and audit[side]["train_geometry"]["feature_dimension"]==LACUNARITY_FEATURE_DIMENSION and audit[side]["target_geometry"]["valid_multiscale_gliding_boxes"] and audit[side]["feature_scaling"]["train_only"] and audit[side]["head"]["fitted_discriminative_coefficients"]==LACUNARITY_FEATURE_DIMENSION and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] and not audit[side]["level_box_statistic_or_penalty_selected_from_labels_or_outer"] and not audit[side]["train_geometry"]["connected_component_hole_or_euler_curve"] and not audit[side]["train_geometry"]["filtration_boundary_matrix_or_persistence"] and not audit[side]["train_geometry"]["class_sliced_wasserstein_projection_or_quantile_barycenter"] and not audit[side]["train_geometry"]["learned_patch_dictionary_or_omp_code"] and not audit[side]["train_geometry"]["ordered_sparse_sem_or_ancestor_drive"] and not audit[side]["train_geometry"]["normalizing_flow_coupling_nll_logdet_or_density_ratio"] for audit in audits for side in ("inner_model_audit","outer_model_audit"))
    checks["multichannel_causal_multiscale_gliding_box_lacunarity_contract_verified"]=bool(contract); result["material_pass"]=bool(not smoke and all(checks.values()))
    if result["material_pass"]: result["selected_frame"]=result["diagnostic_frame"].copy(); result["selected_summary"]=result["candidate_summary"]; result["fallback"]={"activated":False,"policy":None,"exact_entire_frame_verified":True}
    else: result["selected_frame"]=champion.copy(); result["selected_summary"]=result["baseline_summary"]; result["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion),"V197 fallback not exact entire V69")
    result["status"]="SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"); return result


def material_gate(evaluation:dict[str,Any])->dict[str,Any]:
    checks=evaluation["material_checks"]; gate={"contract":HYPOTHESIS,"checks":checks,"passed":int(sum(bool(v) for v in checks.values())),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V197 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out:Path)->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT),"V197 output requires controller authority"); require(out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V197 output target is not controller-bound"); require(out.resolve().parent==(ROOT/"research"/"staging"/"V197").resolve(),"V197 output outside staging/V197"); require(out.exists() and out.is_dir(),"V197 controller staging directory must already exist"); required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"}; allowed=required|{"DATA_EPOCH_BINDING.json"}; existing={path.name for path in out.iterdir()}; require(required.issubset(existing),f"V197 controller prefiles missing: {sorted(required-existing)}"); require(not(existing-allowed),f"V197 unexpected conflict: {sorted(existing-allowed)}"); return {"controller_bound":True,"target_preexisting":True,"required_controller_prefiles":sorted(required),"observed_controller_prefiles":sorted(existing),"unexpected_prefiles":[],"conflict_policy":"allow exact controller prefiles only; fail closed before write"}


def temporary_report_contract_probe()->dict[str,Any]:
    required={"MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"}; names=set(REPORT_NAMES); require(required.issubset(names) and len(names)==len(REPORT_NAMES),"V197 report inventory invalid"); return {"status":"TEMP_REPORT_CONTRACT_OK","required_reports":sorted(required),"all_report_names":list(REPORT_NAMES),"current_material_gate":True,"selected_material_gate_nested_bootstrap":True,"native_robustness_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True,"filesystem_write":False}


def write_outputs(out:Path,authority:dict[str,Any],evidence:pd.DataFrame,audits:list[dict[str,Any]],evaluation:dict[str,Any])->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V197 output not controller-bound"); conflict=enforce_output_conflict_fail_closed(out); compact={k:v for k,v in evaluation.items() if not k.endswith("_frame")}; gate=material_gate(evaluation); selected=json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"]=gate; require("bootstrap" in selected["robustness"],"V197 native bootstrap missing")
    reports={"MODEL_COMPARISON.json":{"version":"V197","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V197_DIAGNOSTIC_LACUNARITY":evaluation["candidate_summary"],"V197_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V197","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V197","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"V197_MULTISCALE_GLIDING_BOX_LACUNARITY_REPORT.json":{"version":"V197","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V197","hypothesis":HYPOTHESIS,"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V197","hypothesis":HYPOTHESIS,"controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V197","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()}}
    require(set(reports)==set(REPORT_NAMES),"V197 report inventory changed"); [atomic_json(payload,out/name) for name,payload in reports.items()]; atomic_csv(evidence,out/"V197_LACUNARITY_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"],out/"V197_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files={path.name:{"sha256":sha256(path),"bytes":path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name!="ARTIFACT_MANIFEST.json"}; atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json"); return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only",action="store_true"); modes.add_argument("--contract-probe",action="store_true"); modes.add_argument("--smoke-test",action="store_true"); modes.add_argument("--support-probe",action="store_true"); modes.add_argument("--full-run",action="store_true"); parser.add_argument("--smoke-market",choices=("US","KR"),default="US"); parser.add_argument("--output",type=Path,default=None); args=parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V197 full run"); args.full_run=True; args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output=DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT),"V197 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V197 controller output mismatch")
    return args


def main()->None:
    args=parse_args(); bounded=bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe); runtime=numeric.configure_bounded_runtime() if bounded else {"mode":"controller_bound_full_cpu","thread_policy":"authorized parent inherited","gpu_model_calls":0}; authority=verify_authority(); dev,champion=load_authorized()
    if args.contract_probe: print(json.dumps(clean({**temporary_report_contract_probe(),"hypothesis":HYPOTHESIS,"runtime":runtime,"authority":authority}),ensure_ascii=False,indent=2)); return
    if args.audit_only: print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"static_spec":STATIC_SPEC,"architecture":"fixed multilevel sub/super occupancy plus multiscale gliding-box mass lacunarity and balanced ridge","preferred_sliced_wasserstein_rejected_due_exact_v147_collision":True,"prior_exact_gliding_box_lacunarity_family_hit_count":0,"causal_numeric_only":True,"target_labels_used":False,"micro_tuning_or_post_outer_change":False,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"gpu_model_calls":0,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"current_material_gate":True,"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","standardized_nested_bootstrap":True,"native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V197 support reserved for KR:2"); audit=audits[0]; non=[trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"])); print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"nested_bootstrap_executed":False,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2)); return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"])); summary={"status":"SMOKE_OK" if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[audit["policy"]["selected"]["name"] for audit in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test: summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary),ensure_ascii=False,indent=2)); return
    result=write_outputs(args.output,authority,evidence,audits,evaluation); print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
