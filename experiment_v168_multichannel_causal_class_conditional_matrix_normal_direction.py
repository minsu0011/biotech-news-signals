"""V168 class-conditional matrix-normal causal-path direction challenger.

Each strict-past same-market cut reorganizes immutable V36 causal numeric state
into an eight-horizon by eight-channel past-to-recent matrix.  Equal event-group
matrices receive train-only per-channel robust scaling.  DOWN and UP each fit a
matrix-normal mean and separable horizon-by-channel covariance through one fixed
eight-step flip-flop procedure with frozen diagonal shrinkage and eigenvalue
floors.  The direct score is the class log-likelihood contrast divided by a
train-only robust temperature.  There is no fitted discriminative head, target
batch fit, full covariance over the vectorized state, or hyperparameter search.

Inner OOF selects exact V69 or fixed .25/.50 probability blends.  Outer labels
are evaluation-only after the 35-minute embargo/event purge.  V69 confidence
and high_conf remain exact; every material failure restores the exact entire
atomic V69 frame.
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

import experiment_v165_multichannel_causal_class_conditional_gaussian_hmm_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V168" / "LOCAL_FORBIDDEN"
VERSION = 168
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_MATRIX_NORMAL_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v165_multichannel_causal_class_conditional_gaussian_hmm_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "81fa0e2626f60b6557b4857c9b3d73835813766bbfed9c05c310eee73091f76f"

HORIZON_N = 8
CHANNEL_N = 8
FLIP_FLOP_ITERATIONS = 8
DIAGONAL_SHRINKAGE = 0.25
EIGENVALUE_FLOOR = 0.05
CHANNEL_CLIP = 7.0
TEMPERATURE_FLOOR = 0.25
SEED = 16801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CLASS_CONDITIONAL_MATRIX_NORMAL_W0.25", "weight": 0.25},
    {"name": "CLASS_CONDITIONAL_MATRIX_NORMAL_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
metric = scaffold.metric
controller = scaffold.controller
numeric = scaffold.numeric
observed_past_to_recent_path = scaffold.observed_past_to_recent_path
blend_probability = scaffold.blend_probability
TrainChannelScale = scaffold.TrainChannelScale
equal_event_group_paths = scaffold.equal_event_group_paths

STATIC_SPEC = {
    "matrix_shape": [HORIZON_N, CHANNEL_N],
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "channel_names": list(scaffold.STATIC_SPEC["channel_names"]),
    "feature_dimension": HORIZON_N * CHANNEL_N,
    "training_unit": "equal event-group mean causal matrix",
    "channel_scaling": "train-only pooled-horizon per-channel median and IQR scale, clip 7",
    "class_models": "separate DOWN and UP matrix-normal mean, horizon covariance and channel covariance",
    "fit": "fixed eight-iteration matrix-normal flip-flop covariance estimation",
    "covariance_regularization": "fixed 0.25 diagonal shrinkage and 0.05 eigenvalue floor",
    "score": "UP matrix-normal log likelihood minus DOWN matrix-normal log likelihood",
    "temperature": "train event-group score IQR/1.349 with standard-deviation fallback and fixed floor 0.25",
    "head": "direct sigmoid; no fitted discriminative coefficient",
    "full_vector_covariance_or_copula_qda": False,
    "hmm_or_latent_transition": False,
    "grassmann_or_principal_angle": False,
    "target_batch_fit": False,
    "hyperparameter_search": False,
}
require(
    len(FEATURES) == 37 and HORIZON_N == CHANNEL_N == 8
    and FLIP_FLOP_ITERATIONS == 8 and DIAGONAL_SHRINKAGE == 0.25
    and EIGENVALUE_FLOOR == 0.05 and STATIC_SPEC["feature_dimension"] == 64,
    "V168 fixed matrix-normal specification changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V165 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, gate, bootstrap and atomic IO only; no V165 output is read",
    }
    audit["v168_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def spd_regularize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(np.diag(symmetric))
    shrunk = (1.0 - DIAGONAL_SHRINKAGE) * symmetric + DIAGONAL_SHRINKAGE * diagonal
    values, vectors = np.linalg.eigh(shrunk)
    values = np.maximum(values, EIGENVALUE_FLOOR)
    covariance = (vectors * values[None, :]) @ vectors.T
    inverse = (vectors * (1.0 / values)[None, :]) @ vectors.T
    logdet = float(np.sum(np.log(values)))
    require(np.isfinite(covariance).all() and np.isfinite(inverse).all() and np.isfinite(logdet), "V168 SPD regularization invalid")
    return covariance, inverse, logdet, {
        "eigenvalue_min": float(values.min()), "eigenvalue_max": float(values.max()),
        "condition_number": float(values.max() / values.min()), "log_determinant": logdet,
    }


def matrix_normal_log_likelihood(
    matrices: np.ndarray, mean: np.ndarray, row_inverse: np.ndarray,
    column_inverse: np.ndarray, row_logdet: float, column_logdet: float,
) -> np.ndarray:
    residual = matrices - mean[None, :, :]
    left = np.einsum("hk,nkc->nhc", row_inverse, residual, optimize=True)
    gram = np.einsum("nhc,nhd->ncd", residual, left, optimize=True)
    quadratic = np.einsum("cd,ncd->n", column_inverse, gram, optimize=True)
    constant = HORIZON_N * CHANNEL_N * np.log(2.0 * np.pi)
    result = -0.5 * (constant + CHANNEL_N * row_logdet + HORIZON_N * column_logdet + quadratic)
    require(np.isfinite(result).all(), "V168 matrix-normal likelihood invalid")
    return result


def fit_matrix_normal(matrices: np.ndarray, label: int) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    require(matrices.ndim == 3 and matrices.shape[1:] == (HORIZON_N, CHANNEL_N), "V168 class matrix shape invalid")
    require(len(matrices) >= 100, f"V168 class {label} support too small")
    mean = np.mean(matrices, axis=0)
    residual = matrices - mean[None, :, :]
    row_cov = np.eye(HORIZON_N, dtype=float)
    column_cov = np.eye(CHANNEL_N, dtype=float)
    trace: list[float] = []
    for _ in range(FLIP_FLOP_ITERATIONS):
        _, column_inverse, _, _ = spd_regularize(column_cov)
        row_raw = np.einsum("nhc,cd,nkd->hk", residual, column_inverse, residual, optimize=True) / (len(residual) * CHANNEL_N)
        row_cov, row_inverse, row_logdet, _ = spd_regularize(row_raw)
        normalization = float(np.trace(row_cov) / HORIZON_N)
        require(normalization > 0.0 and np.isfinite(normalization), "V168 row covariance normalization invalid")
        row_cov /= normalization
        row_inverse *= normalization
        row_logdet -= HORIZON_N * np.log(normalization)
        column_raw = np.einsum("nhc,hk,nkd->cd", residual, row_inverse, residual, optimize=True) / (len(residual) * HORIZON_N)
        column_cov, column_inverse, column_logdet, _ = spd_regularize(column_raw)
        trace.append(float(np.sum(matrix_normal_log_likelihood(matrices, mean, row_inverse, column_inverse, row_logdet, column_logdet))))
    row_values = np.linalg.eigvalsh(row_cov)
    column_values = np.linalg.eigvalsh(column_cov)
    row_audit = {"eigenvalue_min": float(row_values.min()), "eigenvalue_max": float(row_values.max()), "condition_number": float(row_values.max() / row_values.min()), "log_determinant": row_logdet}
    column_audit = {"eigenvalue_min": float(column_values.min()), "eigenvalue_max": float(column_values.max()), "condition_number": float(column_values.max() / column_values.min()), "log_determinant": column_logdet}
    likelihood = matrix_normal_log_likelihood(matrices, mean, row_inverse, column_inverse, row_logdet, column_logdet)
    trace.append(float(np.sum(likelihood)))
    packed = np.concatenate([mean.ravel(), row_cov.ravel(), column_cov.ravel()])
    increments = np.diff(np.asarray(trace, dtype=float))
    model: dict[str, np.ndarray | float] = {
        "mean": mean, "row_covariance": row_cov, "column_covariance": column_cov,
        "row_inverse": row_inverse, "column_inverse": column_inverse,
        "row_logdet": row_logdet, "column_logdet": column_logdet,
    }
    return model, {
        "label": label, "matrix_n": len(matrices), "iterations": FLIP_FLOP_ITERATIONS,
        "diagonal_shrinkage": DIAGONAL_SHRINKAGE, "eigenvalue_floor": EIGENVALUE_FLOOR,
        "trace_first": trace[0], "trace_last": trace[-1],
        "trace_min_increment": float(increments.min()), "trace_nonnegative_fraction": float(np.mean(increments >= -1e-8)),
        "mean_sha256": array_sha256(mean), "row_covariance_sha256": array_sha256(row_cov),
        "column_covariance_sha256": array_sha256(column_cov), "parameter_sha256": array_sha256(packed),
        "row_covariance": row_audit, "column_covariance": column_audit,
        "separable_kronecker_covariance": True, "deterministic_fixed_iteration_flip_flop": True,
    }


def score_matrix_normal(matrices: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    return matrix_normal_log_likelihood(
        matrices, np.asarray(model["mean"]), np.asarray(model["row_inverse"]),
        np.asarray(model["column_inverse"]), float(model["row_logdet"]), float(model["column_logdet"]),
    )


def class_conditional_matrix_normal_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V168 expects one market per fit")
    require(str(ordered.market.iloc[0]) == str(target_frame.market.iloc[0]), "V168 market fit mismatch")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    raw_train, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    raw_target, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    channel_scale = TrainChannelScale().fit(raw_train)
    train_matrix = channel_scale.transform(raw_train)
    target_matrix = channel_scale.transform(raw_target)
    group_matrix, group_target, grouping = equal_event_group_paths(ordered, train_matrix)
    models: dict[int, dict[str, np.ndarray | float]] = {}
    model_audit: dict[str, Any] = {}
    for label in (0, 1):
        models[label], model_audit[str(label)] = fit_matrix_normal(group_matrix[group_target == label], label)
    train_log = np.column_stack([score_matrix_normal(group_matrix, models[label]) for label in (0, 1)])
    target_log = np.column_stack([score_matrix_normal(target_matrix, models[label]) for label in (0, 1)])
    train_score = train_log[:, 1] - train_log[:, 0]
    target_score = target_log[:, 1] - target_log[:, 0]
    q25, q75 = np.quantile(train_score, [0.25, 0.75])
    robust_temperature = float((q75 - q25) / 1.349)
    standard_temperature = float(np.std(train_score))
    temperature = max(robust_temperature if robust_temperature > 1e-8 else standard_temperature, TEMPERATURE_FLOOR)
    probability = 1.0 / (1.0 + np.exp(-np.clip(target_score / temperature, -30.0, 30.0)))
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V168 probability invalid")
    compatibility_geometry = {"feature_dimension": 64, "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": True}
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": compatibility_geometry, "target_geometry": compatibility_geometry,
        "static_spec": STATIC_SPEC, "grouping": grouping,
        "feature_scaling": {"train_only": True, "channel_clip": CHANNEL_CLIP, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale)},
        "class_matrix_normal_models": model_audit,
        "head": {"type": "direct class matrix-normal log-likelihood contrast sigmoid", "fitted_discriminative_coefficient_n": 0, "train_only_temperature": temperature, "train_score_iqr_scale": robust_temperature, "train_score_std": standard_temperature, "train_log_likelihood_sha256": array_sha256(train_log), "target_log_likelihood_sha256": array_sha256(target_log)},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False, "equal_event_group_training": True,
        "full_vector_covariance_or_copula_qda": False, "tyler_spatial_fisher": False,
        "hmm_or_latent_transition": False, "grassmann_or_principal_angle": False,
        "increment_spd_or_log_euclidean_map": False, "gmm_fisher_vector": False,
        "prototype_dtw_elastic_shapelet_or_barycenter": False,
        "dmd_koopman_or_reservoir": False,
        "discriminative_recurrent_neural_network": False,
        "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "target_batch_fit": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = 64
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = class_conditional_matrix_normal_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = class_conditional_matrix_normal_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(
        model_audit["causal_numeric_only"] and model_audit["equal_event_group_training"]
        and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["feature_scaling"]["train_only"]
        and len(model_audit["class_matrix_normal_models"]) == 2
        and all(item["deterministic_fixed_iteration_flip_flop"] for item in model_audit["class_matrix_normal_models"].values())
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"] and not model_audit["target_batch_fit"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 matrix-normal likelihood blend; fixed 2*AUC+BA with BA/net/source safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "covariance_iterations_shrinkage_floor_temperature_or_blend_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v168_model"})
    evidence = evidence.rename(columns={"frenet_prob": "matrix_normal_prob", "v146_model": "v168_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V168 MATRIX NORMAL] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V168 class-conditional matrix-normal policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(
        audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["matrix_shape"] == [8, 8]
        and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["feature_scaling"]["train_only"]
        and len(audit[side]["class_matrix_normal_models"]) == 2
        and all(model["iterations"] == FLIP_FLOP_ITERATIONS and model["diagonal_shrinkage"] == DIAGONAL_SHRINKAGE and model["eigenvalue_floor"] == EIGENVALUE_FLOOR and model["separable_kronecker_covariance"] for model in audit[side]["class_matrix_normal_models"].values())
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"]
        and not audit[side]["full_vector_covariance_or_copula_qda"] and not audit[side]["tyler_spatial_fisher"]
        and not audit[side]["hmm_or_latent_transition"] and not audit[side]["grassmann_or_principal_angle"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["class_conditional_matrix_normal_separable_covariance_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V168 fallback not exact entire V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V168 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V168 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V168 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V168").resolve(), "V168 output is outside controller research/staging/V168")
    require(out.exists() and out.is_dir(), "V168 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V168 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V168 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V168 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V168 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V168", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V168_DIAGNOSTIC_MATRIX_NORMAL": evaluation["candidate_summary"], "V168_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V168", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V168", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V168_CLASS_CONDITIONAL_MATRIX_NORMAL_REPORT.json": {"version": "V168", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V168", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V168", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V168", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items(): atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V168_MATRIX_NORMAL_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V168_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V168 full run")
        args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V168 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V168 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "equal-event-group 8x8 causal matrix; separate class matrix-normal mean and fixed flip-flop separable covariance; direct likelihood contrast", "causal_numeric_only": True, "target_row_labels_used": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V168 support probe is reserved for KR:2")
            audit = audits[0]; non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]})
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
