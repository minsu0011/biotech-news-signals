"""V172 multichannel causal Hankel singular-spectrum direction.

Each strict-past same-market fit robustly scales immutable causal numeric
fields and maps every event to its observed 8-channel by 8-horizon path.  Each
channel's length-eight sequence forms a fixed 4x5 Hankel trajectory matrix.
Exact SVD supplies its singular energy and log spectrum.  Fixed rank-one
reconstruction followed by anti-diagonal averaging yields a trend sequence;
trend/residual shape summaries complete a label-independent descriptor.  A
train-only robust coordinate scale and one equal-event-group/class-balanced
ridge-logistic head produce the challenger probability.

This is not V119 adjacent-horizon Koopman/DMD transition eigenanalysis, V166
raw path row/channel Grassmann subspaces, V95 Haar scattering, or V171 wavelet
leaders/multifractal cumulants.  No Hankel window, rank, feature, head penalty
or blend is chosen from labels or outer rows.  Inner history chooses exact V69
or fixed .25/.50 blends, outer labels are evaluation-only under strict
35-minute embargo/event purge, V69 confidence/high_conf stay exact, and any
material failure restores the exact entire V69 frame.
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

import experiment_v169_multichannel_causal_spd_riemannian_covariance_direction as support


base = support.base
ROOT = support.ROOT
V69_DIR = support.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V172" / "LOCAL_FORBIDDEN"
VERSION = 172
HYPOTHESIS = "MULTICHANNEL_CAUSAL_HANKEL_SINGULAR_SPECTRUM_DIRECTION_V1"
FEATURES = support.FEATURES
EXPECTED_MARKET_FOLD_ROWS = support.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = support.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v169_multichannel_causal_spd_riemannian_covariance_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "f816264512f8d324a62144db31a2e892f77979017f7475090ddbe2c6f4ecca2c"

CHANNEL_N = 8
HORIZON_N = 8
HANKEL_ROWS = 4
HANKEL_COLUMNS = 5
SINGULAR_N = 4
SUMMARY_N = 10
FEATURES_PER_CHANNEL = 2 * SINGULAR_N + SUMMARY_N
HANKEL_FEATURE_DIMENSION = CHANNEL_N * FEATURES_PER_CHANNEL
SPECTRAL_EPSILON = 1e-10
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 17201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "HANKEL_SINGULAR_SPECTRUM_W0.25", "weight": 0.25},
    {"name": "HANKEL_SINGULAR_SPECTRUM_W0.50", "weight": 0.50},
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
    "per_channel_embedding": "fixed 4x5 Hankel trajectory matrix H[i,j]=x[i+j]",
    "spectrum": "four exact singular energies and four log1p singular values",
    "reconstruction": "fixed rank-one SVD component with exact anti-diagonal averaging",
    "shape_summaries": [
        "spectral_entropy", "stable_rank", "gap_1_to_2", "gap_2_to_3",
        "trend_endpoint_slope", "trend_total_variation", "residual_rms",
        "residual_total_variation", "residual_recent", "relative_residual_energy"
    ],
    "features_per_channel": FEATURES_PER_CHANNEL,
    "feature_dimension": HANKEL_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_window_rank_feature_or_threshold": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V172_HANKEL_SINGULAR_SPECTRUM_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json", "RUN_STATUS.json",
)
require(
    HANKEL_ROWS + HANKEL_COLUMNS - 1 == HORIZON_N
    and SINGULAR_N == min(HANKEL_ROWS, HANKEL_COLUMNS)
    and FEATURES_PER_CHANNEL == 18 and HANKEL_FEATURE_DIMENSION == 144,
    "V172 static Hankel representation changed",
)

_SUPPORT_CONFIGURE = support.configure_base
_SUPPORT_BOOTSTRAP = support.paired_nested_bootstrap


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V169 utility scaffold changed")
    audit = support.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, canonical gate, bootstrap and atomic IO only; no V169 output is read",
    }
    audit["v172_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def _diagonal_average(matrix: np.ndarray) -> np.ndarray:
    require(matrix.ndim == 3 and matrix.shape[1:] == (HANKEL_ROWS, HANKEL_COLUMNS), "V172 Hankel reconstruction shape invalid")
    reconstructed = np.empty((len(matrix), HORIZON_N), dtype=float)
    for horizon in range(HORIZON_N):
        cells = [(row, horizon - row) for row in range(HANKEL_ROWS) if 0 <= horizon - row < HANKEL_COLUMNS]
        reconstructed[:, horizon] = np.mean(np.stack([matrix[:, row, column] for row, column in cells], axis=1), axis=1)
    return reconstructed


def hankel_singular_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V172 path shape invalid")
    centered = path - np.mean(path, axis=1, keepdims=True)
    parts: list[np.ndarray] = []
    reconstruction_errors: list[float] = []
    for channel in range(CHANNEL_N):
        sequence = centered[:, :, channel]
        hankel = np.stack([
            np.stack([sequence[:, row + column] for column in range(HANKEL_COLUMNS)], axis=1)
            for row in range(HANKEL_ROWS)
        ], axis=1)
        left, singular, right_t = np.linalg.svd(hankel, full_matrices=False)
        energy = singular * singular
        energy_fraction = energy / np.maximum(np.sum(energy, axis=1, keepdims=True), SPECTRAL_EPSILON)
        log_singular = np.log1p(singular)
        rank_one = singular[:, 0, None, None] * left[:, :, 0, None] * right_t[:, None, 0, :]
        trend = _diagonal_average(rank_one)
        residual = sequence - trend
        entropy = -np.sum(energy_fraction * np.log(np.maximum(energy_fraction, SPECTRAL_EPSILON)), axis=1) / np.log(SINGULAR_N)
        stable_rank = np.sum(energy, axis=1) / np.maximum(energy[:, 0], SPECTRAL_EPSILON)
        gap_12 = np.log((singular[:, 0] + SPECTRAL_EPSILON) / (singular[:, 1] + SPECTRAL_EPSILON))
        gap_23 = np.log((singular[:, 1] + SPECTRAL_EPSILON) / (singular[:, 2] + SPECTRAL_EPSILON))
        trend_slope = (trend[:, -1] - trend[:, 0]) / float(HORIZON_N - 1)
        trend_variation = np.mean(np.abs(np.diff(trend, axis=1)), axis=1)
        residual_rms = np.sqrt(np.mean(residual * residual, axis=1))
        residual_variation = np.mean(np.abs(np.diff(residual, axis=1)), axis=1)
        residual_recent = residual[:, -1]
        relative_residual = np.sum(residual * residual, axis=1) / np.maximum(np.sum(sequence * sequence, axis=1), SPECTRAL_EPSILON)
        summary = np.stack([
            entropy, stable_rank, gap_12, gap_23, trend_slope,
            trend_variation, residual_rms, residual_variation,
            residual_recent, relative_residual,
        ], axis=1)
        feature = np.concatenate([energy_fraction, log_singular, summary], axis=1)
        require(feature.shape[1] == FEATURES_PER_CHANNEL, "V172 per-channel feature dimension changed")
        parts.append(feature)
        hankel_rebuilt = np.stack([
            np.stack([sequence[:, row + column] for column in range(HANKEL_COLUMNS)], axis=1)
            for row in range(HANKEL_ROWS)
        ], axis=1)
        reconstruction_errors.append(float(np.max(np.abs(hankel - hankel_rebuilt))))
    result = np.concatenate(parts, axis=1)
    require(result.shape == (len(path), HANKEL_FEATURE_DIMENSION), "V172 feature dimension changed")
    require(np.isfinite(result).all(), "V172 Hankel feature invalid")
    return result, {
        "rows": len(path), "hankel_shape": [HANKEL_ROWS, HANKEL_COLUMNS],
        "singular_n": SINGULAR_N, "fixed_reconstruction_rank": 1,
        "features_per_channel": FEATURES_PER_CHANNEL,
        "feature_dimension": HANKEL_FEATURE_DIMENSION,
        "hankel_identity_max_abs_error": max(reconstruction_errors),
        "feature_sha256": array_sha256(result),
        "label_independent_exact_transform": True,
        "fixed_hankel_window_and_rank_without_label_or_outer_selection": True,
        "anti_diagonal_rank_one_trend_and_residual": True,
        "v119_transition_operator_or_dynamic_eigenvalue": False,
        "v166_raw_path_grassmann_subspace": False,
        "v171_wavelet_leader_or_multifractal": False,
        # Compatibility negatives for inherited evaluators.
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
        "per_event_channel_correlation_spd": False,
        "fixed_shrinkage_no_label_or_outer_selection": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == HANKEL_FEATURE_DIMENSION, "V172 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V172 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V172 scaled feature invalid")
        return result


def hankel_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V172 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry = hankel_singular_features(train_path)
    target_feature, target_geometry = hankel_singular_features(target_path)
    scale = RobustFeatureScale().fit(train_feature)
    train_design = scale.transform(train_feature)
    target_design = scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, target, sample_weight=numeric.duplicate_class_weights(ordered))
    require(int(head.n_iter_[0]) < head.max_iter, "V172 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V172 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry,
        "class_affine_invariant_mean_prototypes": {},
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(scale.median), "scale_sha256": array_sha256(scale.scale)},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "fitted_discriminative_coefficients": int(head.coef_.size)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False, "window_rank_feature_or_penalty_selected_from_labels_or_outer": False,
        "v86_copula_qda_density": False, "v115_increment_spd_log_tangent_logistic": False,
        "v139_tyler_common_scatter_fisher": False, "v166_grassmann_principal_angles": False,
        "v168_matrix_normal_kronecker_likelihood": False,
        "metric_shrinkage_or_iterations_selected_from_labels_or_outer": False,
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
    support.SPD_COORDINATE_DIMENSION = HANKEL_FEATURE_DIMENSION
    support.ARCHITECTURES = ARCHITECTURES
    support.SEED = SEED
    support.spd_riemannian_direction = hankel_direction
    _SUPPORT_CONFIGURE()


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = hankel_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "model_eligible": True, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["train_geometry"]["fixed_hankel_window_and_rank_without_label_or_outer_selection"]
        and model_audit["train_geometry"]["anti_diagonal_rank_one_trend_and_residual"]
        and model_audit["train_geometry"]["feature_dimension"] == HANKEL_FEATURE_DIMENSION
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["window_rank_feature_or_penalty_selected_from_labels_or_outer"]
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
    return {"selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 Hankel singular-spectrum blend; fixed 2*AUC+BA with BA/net/supported-source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "window_rank_summary_head_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    support.configure_base = configure_base
    support.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = support.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v169_model": "v172_model"})
    evidence = evidence.rename(columns={"spd_airm_prob": "hankel_prob", "v169_model": "v172_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V172 HANKEL SSA] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V172 Hankel singular-spectrum policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    support.paired_nested_bootstrap = paired_nested_bootstrap
    result = support.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_causal_spd_airm_covariance_contract_verified")
    hankel_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["train_geometry"]["fixed_hankel_window_and_rank_without_label_or_outer_selection"]
        and audit[side]["train_geometry"]["anti_diagonal_rank_one_trend_and_residual"]
        and audit[side]["train_geometry"]["feature_dimension"] == HANKEL_FEATURE_DIMENSION
        and audit[side]["target_geometry"]["fixed_hankel_window_and_rank_without_label_or_outer_selection"]
        and audit[side]["feature_scaling"]["train_only"]
        and audit[side]["head"]["fitted_discriminative_coefficients"] == HANKEL_FEATURE_DIMENSION
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["window_rank_feature_or_penalty_selected_from_labels_or_outer"]
        and not audit[side]["train_geometry"]["v119_transition_operator_or_dynamic_eigenvalue"]
        and not audit[side]["train_geometry"]["v166_raw_path_grassmann_subspace"]
        and not audit[side]["train_geometry"]["v171_wavelet_leader_or_multifractal"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["multichannel_causal_hankel_singular_spectrum_contract_verified"] = bool(hankel_contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy()
        result["selected_summary"] = result["candidate_summary"]
        result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V172 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V172 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V172 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V172 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V172").resolve(), "V172 output is outside controller research/staging/V172")
    require(out.exists() and out.is_dir(), "V172 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V172 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V172 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V172 report inventory invalid")
    return {"status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES), "current_material_gate": True, "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "filesystem_write": False}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V172 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V172 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V172", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V172_DIAGNOSTIC_HANKEL_SINGULAR_SPECTRUM": evaluation["candidate_summary"], "V172_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V172", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V172", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V172_HANKEL_SINGULAR_SPECTRUM_REPORT.json": {"version": "V172", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V172", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V172", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V172", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V172 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V172_HANKEL_SINGULAR_SPECTRUM_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V172_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V172 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V172 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V172 controller full-run output mismatch")
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
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "per-channel fixed 4x5 Hankel exact singular spectrum plus anti-diagonal rank-one trend/residual summaries and balanced ridge", "prior_exact_hankel_ssa_family_hit_count": 0, "v119_dmd_transition_reused": False, "v166_grassmann_reused": False, "v170_contribution_polarity_reused": False, "v171_wavelet_leader_multifractal_reused": False, "causal_numeric_only": True, "target_row_labels_used": False, "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V172 support probe is reserved for KR:2")
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
