"""V169 multichannel causal affine-invariant SPD covariance direction.

Within every strict-past same-market fit, immutable causal fields are robustly
scaled and mapped to the observed eight-channel by eight-horizon path.  Each
event becomes one fixed-shrinkage channel-correlation SPD matrix.  Equal-event
group class prototypes are fixed eight-step affine-invariant Karcher means.
The direct direction score is the DOWN-minus-UP affine-invariant Riemannian
geodesic distance contrast with a train-score-only robust temperature.

This is not V115's per-event increment second moment, log-Euclidean tangent
vector and logistic head; not V86 state-space copula QDA; not V139 pooled Tyler
scatter Fisher direction; not V166 Grassmann principal angles; and not V168
matrix-normal Kronecker likelihood.  No tangent coordinate, covariance grid,
shrinkage, iteration count, metric or rank is selected from labels or outer
rows.  Inner history chooses exact V69 or fixed .25/.50 blends.  Outer labels
are evaluation-only under strict 35-minute embargo/event purge.  Atomic V69
confidence/high_conf remain exact, and any material failure restores the exact
entire V69 frame.
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
from threadpoolctl import threadpool_limits

import experiment_v166_multichannel_causal_grassmann_principal_angle_direction as support


base = support.base
ROOT = support.ROOT
V69_DIR = support.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V169" / "LOCAL_FORBIDDEN"
VERSION = 169
HYPOTHESIS = "MULTICHANNEL_CAUSAL_SPD_RIEMANNIAN_COVARIANCE_DIRECTION_V1"
FEATURES = support.FEATURES
EXPECTED_MARKET_FOLD_ROWS = support.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = support.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v166_multichannel_causal_grassmann_principal_angle_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "c32edeb3bf95c06c4fbd8483b9ccd94c7574ab8d01be8c4c2ab0a48ec9e06bfd"

CHANNEL_N = 8
HORIZON_N = 8
SPD_COORDINATE_DIMENSION = CHANNEL_N * (CHANNEL_N + 1) // 2
CORRELATION_SHRINKAGE = 0.25
SPD_EIGEN_FLOOR = 1e-8
CHANNEL_SCALE_FLOOR = 1e-6
KARCHER_ITERATIONS = 8
TEMPERATURE_FLOOR = 0.05
SEED = 16901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "SPD_AIRM_COVARIANCE_W0.25", "weight": 0.25},
    {"name": "SPD_AIRM_COVARIANCE_W0.50", "weight": 0.50},
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
    "channel_n": CHANNEL_N,
    "channel_names": list(base.scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "per_event_spd": "horizon-centered and per-channel-RMS standardized 8x8 channel correlation with fixed 0.25 identity shrinkage",
    "class_prototype": "equal-event-group weighted affine-invariant Karcher mean, exactly 8 iterations",
    "distance": "affine-invariant Riemannian metric ||log(G^-1/2 A G^-1/2)||F",
    "score": "DOWN prototype distance minus UP prototype distance",
    "temperature": "train-score IQR/1.349, standard-deviation fallback and fixed 0.05 floor",
    "feature_dimension": SPD_COORDINATE_DIMENSION,
    "learned_metric_shrinkage_iteration_or_tangent_head": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V169_SPD_RIEMANNIAN_COVARIANCE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(
    CHANNEL_N == HORIZON_N == 8 and SPD_COORDINATE_DIMENSION == 36
    and CORRELATION_SHRINKAGE == 0.25 and KARCHER_ITERATIONS == 8,
    "V169 static SPD Riemannian representation changed",
)

_SUPPORT_CONFIGURE = support.configure_base
_SUPPORT_BOOTSTRAP = support.paired_nested_bootstrap


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V166 utility scaffold changed")
    audit = support.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, canonical gate, bootstrap and atomic IO only; no V166 output is read",
    }
    audit["v169_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def _spd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = (matrix + np.swapaxes(matrix, -1, -2)) / 2.0
    value, vector = np.linalg.eigh(symmetric)
    require(np.isfinite(value).all() and float(np.min(value)) > 0.0, "V169 SPD eigendecomposition invalid")
    return value, vector


def _spd_function(matrix: np.ndarray, function: str) -> np.ndarray:
    value, vector = _spd_eigh(matrix)
    value = np.maximum(value, SPD_EIGEN_FLOOR)
    if function == "sqrt":
        transformed = np.sqrt(value)
    elif function == "invsqrt":
        transformed = 1.0 / np.sqrt(value)
    elif function == "log":
        transformed = np.log(value)
    elif function == "exp":
        transformed = np.exp(np.clip(value, -30.0, 30.0))
    else:
        raise RuntimeError(f"V169 unknown SPD function {function}")
    result = np.einsum("...ik,...k,...jk->...ij", vector, transformed, vector)
    return (result + np.swapaxes(result, -1, -2)) / 2.0


def _symmetric_exp(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + np.swapaxes(matrix, -1, -2)) / 2.0
    value, vector = np.linalg.eigh(symmetric)
    transformed = np.exp(np.clip(value, -30.0, 30.0))
    result = np.einsum("...ik,...k,...jk->...ij", vector, transformed, vector)
    return (result + np.swapaxes(result, -1, -2)) / 2.0


def channel_correlation_spd(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V169 path shape invalid")
    centered = path - np.mean(path, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    standardized = centered / np.maximum(rms[:, None, :], CHANNEL_SCALE_FLOOR)
    correlation = np.einsum("nhi,nhj->nij", standardized, standardized) / float(HORIZON_N)
    identity = np.eye(CHANNEL_N)[None, :, :]
    spd = (1.0 - CORRELATION_SHRINKAGE) * correlation + CORRELATION_SHRINKAGE * identity
    eigenvalue, _ = _spd_eigh(spd)
    require(float(np.min(eigenvalue)) >= CORRELATION_SHRINKAGE - 1e-8, "V169 shrinkage eigenvalue lower bound failed")
    return spd, {
        "rows": len(path), "matrix_dimension": CHANNEL_N,
        "feature_dimension": SPD_COORDINATE_DIMENSION,
        "correlation_shrinkage": CORRELATION_SHRINKAGE,
        "minimum_eigenvalue": float(np.min(eigenvalue)),
        "maximum_condition_number": float(np.max(eigenvalue[:, -1] / eigenvalue[:, 0])),
        "spd_sha256": array_sha256(spd),
        "label_independent_exact_transform": True,
        "per_event_channel_correlation_spd": True,
        "fixed_shrinkage_no_label_or_outer_selection": True,
        "v115_increment_second_moment_tangent": False,
        "v168_matrix_normal_kronecker_likelihood": False,
        # Compatibility negatives for inherited generic evaluators.
        "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False,
        "within_channel_bispectral_magnitude_and_biphase_verified": False,
        "directed_cross_channel_biphase_coupling_verified": False,
        "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False,
        "time_domain_increment_coskewness": False,
        "hilbert_time_local_analytic_phase": False,
        "per_event_row_and_column_grassmann_points": False,
        "svd_sign_invariant_projection_representation": False,
        "fixed_rank_without_label_or_outer_selection": False,
    }


def affine_invariant_mean(matrices: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(len(matrices) == len(weight) and float(np.sum(weight)) > 0.0, "V169 Karcher support invalid")
    normalized = weight / np.sum(weight)
    mean = np.einsum("n,nij->ij", normalized, matrices)
    residuals: list[float] = []
    for _ in range(KARCHER_ITERATIONS):
        sqrt = _spd_function(mean, "sqrt")
        invsqrt = _spd_function(mean, "invsqrt")
        relative = np.einsum("ij,njk,kl->nil", invsqrt, matrices, invsqrt)
        log_relative = _spd_function(relative, "log")
        tangent = np.einsum("n,nij->ij", normalized, log_relative)
        residuals.append(float(np.linalg.norm(tangent, ord="fro")))
        mean = sqrt @ _symmetric_exp(tangent) @ sqrt
        mean = (mean + mean.T) / 2.0
        _spd_eigh(mean)
    return mean, {
        "support_n": len(matrices), "effective_weight": float(np.sum(weight)),
        "iterations_fixed": KARCHER_ITERATIONS,
        "tangent_residual_by_iteration": residuals,
        "final_tangent_residual": residuals[-1],
        "prototype_sha256": array_sha256(mean),
        "affine_invariant_karcher_update": True,
    }


def airm_distance(matrices: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    invsqrt = _spd_function(prototype, "invsqrt")
    relative = np.einsum("ij,njk,kl->nil", invsqrt, matrices, invsqrt)
    value, _ = _spd_eigh(relative)
    distance = np.sqrt(np.sum(np.log(np.maximum(value, SPD_EIGEN_FLOOR)) ** 2, axis=1))
    require(np.isfinite(distance).all(), "V169 AIRM distance invalid")
    return distance


def _distance_contrast(matrices: np.ndarray, prototypes: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    down = airm_distance(matrices, prototypes[0])
    up = airm_distance(matrices, prototypes[1])
    raw = down - up
    return raw, {
        "down_distance_mean": float(np.mean(down)), "up_distance_mean": float(np.mean(up)),
        "raw_contrast_mean": float(np.mean(raw)), "raw_contrast_std": float(np.std(raw)),
        "raw_contrast_sha256": array_sha256(raw),
    }


def spd_riemannian_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V169 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_spd, train_geometry = channel_correlation_spd(train_path)
    target_spd, target_geometry = channel_correlation_spd(target_path)
    target = ordered.y.to_numpy(int)
    sample_weight = numeric.duplicate_class_weights(ordered)
    prototypes: dict[int, np.ndarray] = {}
    prototype_audit: dict[str, Any] = {}
    for label in (0, 1):
        mask = target == label
        require(int(mask.sum()) >= 8, f"V169 class {label} support too small")
        prototypes[label], prototype_audit[str(label)] = affine_invariant_mean(train_spd[mask], sample_weight[mask])
    train_raw, train_score_audit = _distance_contrast(train_spd, prototypes)
    target_raw, target_score_audit = _distance_contrast(target_spd, prototypes)
    q25, q75 = np.quantile(train_raw, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    standard = float(np.std(train_raw))
    temperature = max(robust if robust > SPD_EIGEN_FLOOR else standard, TEMPERATURE_FLOOR)
    probability = np.clip(1.0 / (1.0 + np.exp(-np.clip(target_raw / temperature, -30.0, 30.0))), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V169 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry,
        "class_affine_invariant_mean_prototypes": prototype_audit,
        "train_score": train_score_audit, "target_score": target_score_audit,
        "feature_scaling": {"train_only": True, "temperature": temperature, "temperature_floor": TEMPERATURE_FLOOR},
        "head": {"type": "direct affine-invariant geodesic distance contrast", "fitted_discriminative_coefficients": 0},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False, "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False,
        "v86_copula_qda_density": False, "v115_increment_spd_log_tangent_logistic": False,
        "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False,
        "v168_matrix_normal_kronecker_likelihood": False,
        # Compatibility negatives for V166's evaluator.
        "v101_static_affine_reconstruction_residual": False,
        "v143_source_time_nuisance_projection": False, "pls_or_ssa_response_projection": False,
        "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": True, "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    support.VERSION = VERSION
    support.HYPOTHESIS = HYPOTHESIS
    support.STATIC_SPEC = STATIC_SPEC
    support.GRASSMANN_COORDINATE_DIMENSION = SPD_COORDINATE_DIMENSION
    support.ARCHITECTURES = ARCHITECTURES
    support.SEED = SEED
    support.grassmann_direction = spd_riemannian_direction
    _SUPPORT_CONFIGURE()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = spd_riemannian_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["train_geometry"]["per_event_channel_correlation_spd"]
        and model_audit["train_geometry"]["fixed_shrinkage_no_label_or_outer_selection"]
        and all(item["affine_invariant_karcher_update"] and item["iterations_fixed"] == KARCHER_ITERATIONS for item in model_audit["class_affine_invariant_mean_prototypes"].values())
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["metric_shrinkage_or_iterations_selected_from_labels_or_outer"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 SPD AIRM covariance blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "shrinkage_metric_iteration_temperature_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    support.configure_base = configure_base
    support.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = support.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v166_model": "v169_model"})
    evidence = evidence.rename(columns={"grassmann_prob": "spd_airm_prob", "v166_model": "v169_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V169 SPD AIRM] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V169 SPD AIRM covariance policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    support.paired_nested_bootstrap = paired_nested_bootstrap
    result = support.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_causal_grassmann_principal_angle_contract_verified")
    spd_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["train_geometry"]["per_event_channel_correlation_spd"]
        and audit[side]["train_geometry"]["fixed_shrinkage_no_label_or_outer_selection"]
        and audit[side]["target_geometry"]["per_event_channel_correlation_spd"]
        and all(item["affine_invariant_karcher_update"] and item["iterations_fixed"] == KARCHER_ITERATIONS for item in audit[side]["class_affine_invariant_mean_prototypes"].values())
        and audit[side]["feature_scaling"]["train_only"]
        and audit[side]["head"]["fitted_discriminative_coefficients"] == 0
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v86_copula_qda_density"]
        and not audit[side]["v115_increment_spd_log_tangent_logistic"]
        and not audit[side]["v139_tyler_common_scatter_fisher"]
        and not audit[side]["v166_grassmann_principal_angles"]
        and not audit[side]["v168_matrix_normal_kronecker_likelihood"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["multichannel_causal_spd_airm_covariance_contract_verified"] = bool(spd_contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy()
        result["selected_summary"] = result["candidate_summary"]
        result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V169 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V169 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V169 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V169 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V169").resolve(), "V169 output is outside controller research/staging/V169")
    require(out.exists() and out.is_dir(), "V169 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V169 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V169 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V169 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V169 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V169 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V169", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V169_DIAGNOSTIC_SPD_AIRM_COVARIANCE": evaluation["candidate_summary"], "V169_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V169", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V169", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V169_SPD_RIEMANNIAN_COVARIANCE_REPORT.json": {"version": "V169", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V169", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V169", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V169", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V169 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V169_SPD_RIEMANNIAN_COVARIANCE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V169_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V169 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V169 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V169 controller full-run output mismatch")
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
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "per-event fixed-shrinkage channel correlation SPD, class affine-invariant Karcher mean prototypes, direct AIRM distance contrast", "v115_log_euclidean_increment_spd_tangent_reused": False, "v86_copula_qda_reused": False, "v139_tyler_fisher_reused": False, "v166_grassmann_reused": False, "v168_matrix_normal_reused": False, "causal_numeric_only": True, "target_row_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V169 support probe is reserved for KR:2")
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
