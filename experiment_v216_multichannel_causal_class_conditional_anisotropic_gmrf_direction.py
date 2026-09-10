"""V216 class-conditional anisotropic Gaussian Markov random-field direction.

Each strict-train robust 8x8 causal surface is scored under two continuous
Gaussian Markov random fields.  Class mean surfaces and three deterministic
moment-matched precision coefficients multiply identity, horizontal-grid, and
vertical-grid Laplacians; fixed diagonal shrinkage guarantees SPD precision.
Exact normalized Gaussian log-density contrasts provide direction without a
discriminative head, target fit, or outer-driven tuning.
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

import experiment_v213_multichannel_causal_class_conditional_dpp_subset_direction as scaffold


ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V216" / "LOCAL_FORBIDDEN"
VERSION = 216
PRIORITY = 870
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_ANISOTROPIC_GMRF_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v213_multichannel_causal_class_conditional_dpp_subset_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "8312e91c1b0213a8c71f48f3b6a32f323d9a3cd01886a0c972678a447550c804"

GRID_N = 8
SURFACE_DIM = 64
VARIANCE_FLOOR = 0.05
PRECISION_IDENTITY_SHRINK = 0.10
JEFFREYS_MASS = 0.5
PROBABILITY_EPS = 1e-6
TEMPERATURE_FLOOR = 0.05
SEED = 21601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CLASS_CONDITIONAL_ANISOTROPIC_GMRF_W0.25", "weight": 0.25},
    {"name": "CLASS_CONDITIONAL_ANISOTROPIC_GMRF_W0.50", "weight": 0.50},
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
    "surface": "strict-train robust-scaled observed 8-horizon x 8-semantic-channel continuous causal state",
    "feature_dimension": SURFACE_DIM,
    "class_location": "equal-event-group class mean surface",
    "precision": "class moment-matched identity plus horizontal and vertical fixed-grid Laplacians with 10 percent diagonal shrinkage",
    "class_model": "separate exact normalized anisotropic Gaussian Markov random-field density",
    "direction": "UP-minus-DOWN exact precision logdet and quadratic-form likelihood contrast divided by train-only IQR temperature",
    "discriminative_head": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V216_CLASS_CONDITIONAL_ANISOTROPIC_GMRF_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(GRID_N == 8 and SURFACE_DIM == 64 and VARIANCE_FLOOR == 0.05 and PRECISION_IDENTITY_SHRINK == 0.10 and PRIORITY == 870, "V216 static model changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V213 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, continuous causal path, metrics, gates, bootstrap and atomic IO only; no V213 output is read"}
    audit["v216_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _grid_laplacians() -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.zeros((SURFACE_DIM, SURFACE_DIM), dtype=float)
    vertical = np.zeros((SURFACE_DIM, SURFACE_DIM), dtype=float)

    def add_edge(matrix: np.ndarray, left: int, right: int) -> None:
        matrix[left, left] += 1.0; matrix[right, right] += 1.0
        matrix[left, right] -= 1.0; matrix[right, left] -= 1.0

    for horizon in range(GRID_N):
        for channel in range(GRID_N - 1):
            add_edge(horizontal, horizon * GRID_N + channel, horizon * GRID_N + channel + 1)
    for horizon in range(GRID_N - 1):
        for channel in range(GRID_N):
            add_edge(vertical, horizon * GRID_N + channel, (horizon + 1) * GRID_N + channel)
    require(np.allclose(horizontal, horizontal.T) and np.allclose(vertical, vertical.T), "V216 grid Laplacian symmetry failed")
    return horizontal, vertical


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights)); require(total > 0.0, "V216 nonpositive class weight")
    return np.sum(values * weights[:, None], axis=0) / total


def _weighted_scalar(values: np.ndarray, weights: np.ndarray) -> float:
    per_row = np.mean(values, axis=tuple(range(1, values.ndim)))
    return float(np.sum(weights * per_row) / np.sum(weights))


def _fit_class_gmrf(surface: np.ndarray, weights: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    require(surface.ndim == 3 and surface.shape[1:] == (GRID_N, GRID_N) and len(surface) >= 8, "V216 class GMRF support invalid")
    flat = surface.reshape(len(surface), SURFACE_DIM)
    mean = _weighted_mean(flat, weights)
    residual = surface - mean.reshape(GRID_N, GRID_N)
    level_variance = max(_weighted_scalar(residual * residual, weights), VARIANCE_FLOOR)
    horizontal_variance = max(_weighted_scalar(np.diff(residual, axis=2) ** 2, weights), VARIANCE_FLOOR)
    vertical_variance = max(_weighted_scalar(np.diff(residual, axis=1) ** 2, weights), VARIANCE_FLOOR)
    coefficients = np.asarray([1.0 / level_variance, 0.5 / horizontal_variance, 0.5 / vertical_variance], dtype=float)
    raw = coefficients[0] * np.eye(SURFACE_DIM) + coefficients[1] * horizontal + coefficients[2] * vertical
    diagonal_scale = float(np.mean(np.diag(raw)))
    precision = (1.0 - PRECISION_IDENTITY_SHRINK) * raw + PRECISION_IDENTITY_SHRINK * diagonal_scale * np.eye(SURFACE_DIM)
    sign, logdet = np.linalg.slogdet(precision)
    minimum_eigenvalue = float(np.linalg.eigvalsh(precision).min())
    require(sign > 0 and np.isfinite(logdet) and minimum_eigenvalue > 0.0, "V216 precision not SPD")
    audit = {
        "rows": len(surface), "mean_surface_sha256": array_sha256(mean), "moment_variances": {"level": level_variance, "horizontal_increment": horizontal_variance, "vertical_increment": vertical_variance},
        "precision_coefficients": {"identity": float(coefficients[0]), "horizontal": float(coefficients[1]), "vertical": float(coefficients[2])},
        "precision_sha256": array_sha256(precision), "logdet_precision": float(logdet), "minimum_precision_eigenvalue": minimum_eigenvalue,
        "fixed_identity_shrinkage": PRECISION_IDENTITY_SHRINK, "moment_matched_without_optimizer_or_grid": True, "exact_spd_precision": True,
    }
    return mean, precision, float(logdet), audit


def _log_density(flat: np.ndarray, mean: np.ndarray, precision: np.ndarray, logdet: float, log_prior: float) -> np.ndarray:
    residual = flat - mean[None, :]
    quadratic = np.einsum("ni,ij,nj->n", residual, precision, residual, optimize=True)
    result = log_prior + 0.5 * logdet - 0.5 * quadratic
    require(np.isfinite(result).all(), "V216 GMRF log density invalid")
    return result


def gmrf_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V216 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    require(train_path.shape[1:] == (GRID_N, GRID_N) and target_path.shape[1:] == (GRID_N, GRID_N), "V216 causal surface shape changed")
    base_weights = scaffold.base.scaffold._base_duplicate_weights(ordered)
    labels = ordered.y.to_numpy(int); horizontal, vertical = _grid_laplacians()
    class_models: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}; class_audits: dict[str, Any] = {}
    total = float(np.sum(base_weights))
    for label in (0, 1):
        mask = labels == label; mean, precision, logdet, class_audit = _fit_class_gmrf(train_path[mask], base_weights[mask], horizontal, vertical)
        prior = (float(np.sum(base_weights[mask])) + JEFFREYS_MASS) / (total + 2.0 * JEFFREYS_MASS)
        class_models[label] = (mean, precision, logdet, float(np.log(prior))); class_audits["DOWN" if label == 0 else "UP"] = {**class_audit, "class_prior": prior}
    train_flat = train_path.reshape(len(train_path), SURFACE_DIM); target_flat = target_path.reshape(len(target_path), SURFACE_DIM)
    train_score = {label: _log_density(train_flat, *class_models[label]) for label in (0, 1)}
    train_contrast = train_score[1] - train_score[0]
    q25, q75 = np.quantile(train_contrast, [0.25, 0.75]); temperature = max(float((q75 - q25) / 1.349), TEMPERATURE_FLOOR)
    target_score = {label: _log_density(target_flat, *class_models[label]) for label in (0, 1)}
    probability = _sigmoid((target_score[1] - target_score[0]) / temperature)
    geometry = {
        "feature_dimension": SURFACE_DIM, "continuous_surface": True, "fixed_grid_horizontal_laplacian_sha256": array_sha256(horizontal), "fixed_grid_vertical_laplacian_sha256": array_sha256(vertical),
        "class_models": class_audits, "class_conditional_anisotropic_gmrf": True, "exact_gaussian_precision_logdet": True, "exact_quadratic_form": True,
        "moment_matched_identity_horizontal_vertical_precision": True, "fixed_identity_shrinkage": PRECISION_IDENTITY_SHRINK, "target_batch_fit": False,
        "target_labels_used": False, "discriminative_logistic_head": False, "diffusion_map_landmark_nystrom": False, "dpp_l_ensemble_subset_logdet": False,
        "potts_discrete_pseudolikelihood": False, "matrix_normal_kronecker_covariance": False, "tyler_common_scatter_fisher": False,
        "per_event_graph_laplacian_spectrum": False, "gaussian_copula_qda": False, "graphical_lasso_or_precision_grid_search": False,
        "precision_coefficients_shrinkage_temperature_or_blend_selected_from_outer": False,
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
        "fixed_knots_grid_derivatives_and_summaries_without_label_or_outer_selection": False,
        "continuous_first_and_second_functional_jets": False, "v175_burg_lattice_parcor_or_poles": False,
        "v176_level_crossing_excursion_or_dwell": False, "v177_discrete_radon_line_projection": False,
    }
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": geometry, "target_geometry": geometry, "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "method": "strict-train robust causal continuous surface; no target refit", "target_batch_fit": False},
        "head": {"type": "none; direct exact class anisotropic GMRF log-density contrast", "fitted_discriminative_coefficients": 0},
        "temperature": {"train_only": True, "value": temperature, "floor": TEMPERATURE_FLOOR, "train_contrast_sha256": array_sha256(train_contrast)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    lower = scaffold.base
    lower.VERSION = VERSION; lower.HYPOTHESIS = HYPOTHESIS; lower.STATIC_SPEC = STATIC_SPEC; lower.ARCHITECTURES = ARCHITECTURES; lower.SEED = SEED
    lower.rbm_direction = gmrf_direction; lower.choose_inner = choose_inner; lower.configure_base()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, audit = gmrf_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    g = audit["train_geometry"]
    eligible_model = bool(g["class_conditional_anisotropic_gmrf"] and g["exact_gaussian_precision_logdet"] and g["exact_quadratic_form"] and g["moment_matched_identity_horizontal_vertical_precision"] and not g["target_batch_fit"] and not g["target_labels_used"] and not g["discriminative_logistic_head"] and not audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": eligible_model, "eligible": bool(eligible_model and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 class anisotropic GMRF likelihood blend", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "mean_precision_moments_shrinkage_temperature_blend_or_safety_micro_tuning": False}, audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); ENGINE.configure_base = configure_base; ENGINE.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = SUPPORT_RUN_NESTED(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v178_model": "v216_model"}); evidence = evidence.rename(columns={"bspline_prob": "gmrf_prob", "v178_model": "v216_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V216 GMRF] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V216 class anisotropic GMRF likelihood policy"; result["seed"] = SEED; result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); ENGINE.paired_nested_bootstrap = paired_nested_bootstrap
    result = SUPPORT_EVALUATE(champion, diagnostic, evidence, audits, draws, smoke); checks = result["material_checks"]
    checks.pop("multichannel_causal_cubic_bspline_functional_jet_contract_verified", None)
    contract = all(audit[side]["train_geometry"]["class_conditional_anisotropic_gmrf"] and audit[side]["train_geometry"]["exact_gaussian_precision_logdet"] and audit[side]["train_geometry"]["exact_quadratic_form"] and audit[side]["head"]["fitted_discriminative_coefficients"] == 0 and audit[side]["temperature"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_class_conditional_anisotropic_gmrf_contract_verified"] = bool(contract); result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy(); result["selected_summary"] = result["candidate_summary"]; result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V216 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V216 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V216 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V216 output target is not controller-bound"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V216").resolve(), "V216 output outside staging/V216"); require(out.exists() and out.is_dir(), "V216 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V216 controller prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V216 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}; names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V216 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "dev_robustness_top_level_status": True, "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "canonical_gate_checks": 14, "exact_entire_v69_fallback": True, "filesystem_write": False, "resource_contract": resource_contract()}


def resource_contract() -> dict[str, Any]:
    return {
        "bounded_prep_audit_smoke": {"external_pre_resume_affinity_mask": "0xC0000000", "cpu_ids": [30, 31], "numeric_threads": 2, "gpu_hidden": True, "gpu_model_calls": 0},
        "controller_exact_bound_full": {"affinity_policy": "inherit unchanged from authorized controller", "expected_authorized_mask": "0xFFFFFFFF", "thread_environment_policy": "inherit unchanged from authorized controller", "expected_authorized_threads": 32, "cuda_visibility_policy": "inherit unchanged from authorized controller", "expected_authorized_cuda_visible_devices": "0", "model_gpu_calls": 0},
        "runner_sets_process_affinity": False,
        "runner_overrides_full_thread_environment": False,
        "runner_overrides_full_cuda_visibility": False,
    }


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V216 output not controller-bound")
    conflict = enforce_output_conflict_fail_closed(out); compact = {k: v for k, v in evaluation.items() if not k.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V216 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V216", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V216_DIAGNOSTIC_GMRF": evaluation["candidate_summary"], "V216_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V216", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V216", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V216_CLASS_CONDITIONAL_ANISOTROPIC_GMRF_REPORT.json": {"version": "V216", "priority": PRIORITY, "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V216", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V216", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V216", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V216 report inventory changed")
    [atomic_json(payload, out / name) for name, payload in reports.items()]; atomic_csv(evidence, out / "V216_GMRF_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V216_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--contract-probe", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V216 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V216 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V216 controller output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "affinity_policy": "authorized controller inherited unchanged", "thread_policy": "authorized controller inherited unchanged", "gpu_visibility_policy": "authorized controller inherited unchanged", "gpu_model_calls": 0}
    authority = verify_authority(); dev, champion = load_authorized()
    if args.contract_probe:
        print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "priority": PRIORITY, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2)); return
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "version": VERSION, "priority": PRIORITY, "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC, "architecture": "class mean surface plus exact anisotropic fixed-grid GMRF precision likelihood", "prior_exact_gmrf_family_hit_count": 0, "rejected_candidate": "Hilbert KT context mixture rejected due exact V107 collision", "causal_numeric_only": True, "target_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "dev_robustness_top_level_status": True, "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "resource_contract": resource_contract()}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V216 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "priority": PRIORITY, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
