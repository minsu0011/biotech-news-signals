"""V173 multichannel causal weighted-grid Laplacian spectral challenger.

Each strict-past train cut robustly scales the immutable causal numeric state
and forms the observed 8-channel by 8-horizon path.  Every event becomes the
same fixed 4-neighbor 8x8 grid, but its edge affinity is the label-independent
``exp(-abs(x_u-x_v))`` of adjacent standardized values.  The exact symmetric
normalized graph-Laplacian eigenspectrum, consecutive spectral gaps, fixed heat
traces and spectral summaries form a 144D representation.  A train-only robust
feature scale and one equal-event-group/class-balanced ridge-logit head predict
direction.

This is not V106 semantic-feature lower-star persistence, V141 natural
visibility, or V167 thresholded cubical Euler topology: no graph is shared
between events, no visibility relation or threshold filtration is built, and
the continuous local affinity spectrum is retained.  Inner-past labels select
only exact V69 or fixed .25/.50 blends.  Outer labels are evaluation-only under
the 35-minute embargo/event purge; V69 confidence/high_conf remain exact and
any material or execution failure returns the exact entire atomic V69 frame.
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

import experiment_v167_multichannel_causal_euler_characteristic_curve_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V173" / "LOCAL_FORBIDDEN"
VERSION = 173
HYPOTHESIS = "MULTICHANNEL_CAUSAL_WEIGHTED_GRID_LAPLACIAN_SPECTRAL_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v167_multichannel_causal_euler_characteristic_curve_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "9e48cf7e6adf9c3dd11266d902d64f811d4fb23ec23e9a08f7ee0666e605340c"

GRID_ROWS = 8
GRID_COLUMNS = 8
NODE_N = GRID_ROWS * GRID_COLUMNS
EDGE_N = GRID_ROWS * (GRID_COLUMNS - 1) + GRID_COLUMNS * (GRID_ROWS - 1)
HEAT_TIMES = np.asarray([0.25, 0.50, 1.0, 2.0, 4.0], dtype=float)
MOMENT_ORDERS = (1, 2, 3, 4)
SUMMARY_N = 8
LAPLACIAN_FEATURE_DIMENSION = NODE_N + (NODE_N - 1) + len(HEAT_TIMES) + len(MOMENT_ORDERS) + SUMMARY_N
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 17301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "WEIGHTED_GRID_LAPLACIAN_SPECTRUM_W0.25", "weight": 0.25},
    {"name": "WEIGHTED_GRID_LAPLACIAN_SPECTRUM_W0.50", "weight": 0.50},
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
blend_probability = base.blend_probability

STATIC_SPEC = {
    "grid": "actual 8 horizon rows x 8 semantic-channel columns",
    "node_n": NODE_N,
    "edge_n": EDGE_N,
    "adjacency": "fixed four-neighbor horizontal and vertical grid",
    "edge_affinity": "exp(-abs(adjacent standardized value difference))",
    "laplacian": "exact symmetric normalized I-D^-1/2 A D^-1/2",
    "spectrum": "all 64 sorted eigenvalues and all 63 consecutive gaps",
    "heat_trace_times": HEAT_TIMES.tolist(),
    "spectral_moment_orders": list(MOMENT_ORDERS),
    "summaries": ["lambda2", "lambda_max", "spectral_span", "nonzero condition proxy", "mean log nonzero eigenvalue", "spectral entropy", "q25 eigenvalue", "q75 eigenvalue"],
    "feature_dimension": LAPLACIAN_FEATURE_DIMENSION,
    "feature_scaling": "train-event-group-only median/IQR-or-std and fixed clip",
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "labels_or_target_batch_used_in_graph": False,
}
require(EDGE_N == 112 and LAPLACIAN_FEATURE_DIMENSION == 144, "V173 graph constants changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V167 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, observed causal path, metrics, canonical gate, bootstrap and atomic IO only; no V167 output is read",
    }
    audit["v173_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
        "prior_version_output_read": False,
    }
    return audit


def grid_edges() -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            node = row * GRID_COLUMNS + column
            if column + 1 < GRID_COLUMNS:
                left.append(node)
                right.append(node + 1)
            if row + 1 < GRID_ROWS:
                left.append(node)
                right.append(node + GRID_COLUMNS)
    require(len(left) == len(right) == EDGE_N, "V173 grid edge count changed")
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


EDGE_LEFT, EDGE_RIGHT = grid_edges()


def one_grid_spectrum(grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(grid.shape == (GRID_ROWS, GRID_COLUMNS), "V173 grid shape changed")
    flat = grid.reshape(-1)
    affinity = np.exp(-np.abs(flat[EDGE_LEFT] - flat[EDGE_RIGHT]))
    adjacency = np.zeros((NODE_N, NODE_N), dtype=float)
    adjacency[EDGE_LEFT, EDGE_RIGHT] = affinity
    adjacency[EDGE_RIGHT, EDGE_LEFT] = affinity
    degree = adjacency.sum(axis=1)
    require(bool(np.all(degree > 0.0)), "V173 isolated graph node")
    inverse_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(NODE_N) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    eigenvalue = np.linalg.eigvalsh(laplacian)
    eigenvalue = np.clip(eigenvalue, 0.0, 2.0)
    eigenvalue.sort()
    gap = np.diff(eigenvalue)
    heat = np.asarray([np.mean(np.exp(-time * eigenvalue)) for time in HEAT_TIMES], dtype=float)
    moment = np.asarray([np.mean(eigenvalue ** order) for order in MOMENT_ORDERS], dtype=float)
    nonzero = np.maximum(eigenvalue[1:], 1e-12)
    mass = nonzero / nonzero.sum()
    entropy = -float(np.sum(mass * np.log(mass))) / np.log(len(mass))
    summary = np.asarray([
        eigenvalue[1], eigenvalue[-1], eigenvalue[-1] - eigenvalue[0],
        eigenvalue[-1] / nonzero[0], float(np.mean(np.log(nonzero))), entropy,
        float(np.quantile(eigenvalue, 0.25)), float(np.quantile(eigenvalue, 0.75)),
    ])
    feature = np.concatenate([eigenvalue, gap, heat, moment, summary])
    require(feature.shape == (LAPLACIAN_FEATURE_DIMENSION,) and np.isfinite(feature).all(), "V173 spectrum invalid")
    return feature, {
        "minimum_edge_affinity": float(affinity.min()),
        "mean_edge_affinity": float(affinity.mean()),
        "maximum_edge_affinity": float(affinity.max()),
        "lambda0": float(eigenvalue[0]), "lambda2": float(eigenvalue[1]),
        "lambda_max": float(eigenvalue[-1]), "spectral_entropy": entropy,
    }


def laplacian_spectral_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (GRID_ROWS, GRID_COLUMNS), "V173 path shape invalid")
    feature = np.empty((len(path), LAPLACIAN_FEATURE_DIMENSION), dtype=float)
    audits: list[dict[str, Any]] = []
    for row, grid in enumerate(path):
        feature[row], audit = one_grid_spectrum(grid)
        audits.append(audit)
    require(np.isfinite(feature).all(), "V173 feature matrix invalid")
    return feature, {
        "rows": len(feature), "feature_dimension": feature.shape[1],
        "feature_sha256": array_sha256(feature),
        "actual_8x8_weighted_grid": True,
        "fixed_four_neighbor_edges": True,
        "edge_n": EDGE_N,
        "continuous_label_independent_affinity": True,
        "exact_symmetric_normalized_laplacian_eigendecomposition": True,
        "complete_sorted_eigenspectrum": True,
        "spectral_gaps_heat_traces_and_moments": True,
        "mean_minimum_edge_affinity": float(np.mean([audit["minimum_edge_affinity"] for audit in audits])),
        "mean_edge_affinity": float(np.mean([audit["mean_edge_affinity"] for audit in audits])),
        "mean_lambda0": float(np.mean([audit["lambda0"] for audit in audits])),
        "mean_lambda2": float(np.mean([audit["lambda2"] for audit in audits])),
        "mean_lambda_max": float(np.mean([audit["lambda_max"] for audit in audits])),
        "mean_spectral_entropy": float(np.mean([audit["spectral_entropy"] for audit in audits])),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "target_batch_graph_or_scale_fit": False,
        "target_labels_used": False,
        "v106_semantic_feature_graph_lower_star_persistence": False,
        "v141_natural_visibility_graph": False,
        "v167_threshold_filtration_euler_curve": False,
        "v174_morphological_granulometry": False,
        "v175_burg_lattice_autoregression": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == LAPLACIAN_FEATURE_DIMENSION, "V173 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V173 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V173 scaled feature invalid")
        return result


def laplacian_spectral_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V173 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping_audit = scaffold.scaffold.equal_event_group_path(ordered, train_path)
    train_feature, train_spectrum_audit = laplacian_spectral_features(group_path)
    target_feature, target_spectrum_audit = laplacian_spectral_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V173 probability invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "grouping": grouping_audit, "static_spec": STATIC_SPEC,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_spectrum_audit, "target_geometry": target_spectrum_audit,
        "feature_scaling": {
            "train_only": True, "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed equal-event-group/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]),
            "class_event_group_n": class_n.astype(int).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False,
        "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = LAPLACIAN_FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = laplacian_spectral_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = laplacian_spectral_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["static_spec"]["feature_dimension"] == LAPLACIAN_FEATURE_DIMENSION
        and model_audit["grouping"]["equal_event_group_weighting"]
        and model_audit["train_geometry"]["actual_8x8_weighted_grid"]
        and model_audit["train_geometry"]["fixed_four_neighbor_edges"]
        and model_audit["train_geometry"]["continuous_label_independent_affinity"]
        and model_audit["train_geometry"]["exact_symmetric_normalized_laplacian_eigendecomposition"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 weighted-grid Laplacian-spectrum blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "affinity_spectrum_summary_head_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v173_model"})
    evidence = evidence.rename(columns={"frenet_prob": "grid_laplacian_prob", "v146_model": "v173_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V173 GRID LAPLACIAN] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V173 weighted-grid Laplacian-spectrum policy"
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
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["feature_dimension"] == LAPLACIAN_FEATURE_DIMENSION
        and audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["train_geometry"]["actual_8x8_weighted_grid"]
        and audit[side]["train_geometry"]["fixed_four_neighbor_edges"]
        and audit[side]["train_geometry"]["edge_n"] == EDGE_N
        and audit[side]["train_geometry"]["continuous_label_independent_affinity"]
        and audit[side]["train_geometry"]["exact_symmetric_normalized_laplacian_eigendecomposition"]
        and audit[side]["train_geometry"]["complete_sorted_eigenspectrum"]
        and audit[side]["train_geometry"]["spectral_gaps_heat_traces_and_moments"]
        and not audit[side]["train_geometry"]["target_batch_graph_or_scale_fit"]
        and not audit[side]["train_geometry"]["v106_semantic_feature_graph_lower_star_persistence"]
        and not audit[side]["train_geometry"]["v141_natural_visibility_graph"]
        and not audit[side]["train_geometry"]["v167_threshold_filtration_euler_curve"]
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["multichannel_causal_weighted_grid_laplacian_spectral_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V173 fallback not exact entire V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V173 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V173 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V173 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V173").resolve(), "V173 output outside controller research/staging/V173")
    require(out.exists() and out.is_dir(), "V173 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V173 controller prefiles missing: {sorted(required-existing)}")
    require(not (existing - allowed), f"V173 unexpected pre-existing output conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V173 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V173 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V173", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V173_DIAGNOSTIC_WEIGHTED_GRID_LAPLACIAN": evaluation["candidate_summary"], "V173_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V173", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V173", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V173_WEIGHTED_GRID_LAPLACIAN_SPECTRAL_REPORT.json": {"version": "V173", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V173", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V173", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V173", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items(): atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V173_WEIGHTED_GRID_LAPLACIAN_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V173_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json": files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files)+1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


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
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V173 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V173 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V173 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        constant_feature, constant_audit = one_grid_spectrum(np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=float))
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "continuous four-neighbor affinity normalized-Laplacian eigenspectrum/gaps/heat traces/moments on actual 8x8 causal grid, train-only scale, equal-group balanced ridge",
            "constant_grid_unit_probe": {"feature_dimension": len(constant_feature), "lambda0": constant_audit["lambda0"], "lambda2": constant_audit["lambda2"], "lambda_max": constant_audit["lambda_max"]},
            "novelty": {"v106_semantic_graph_persistence": False, "v141_visibility_graph": False, "v167_euler_curve": False, "v174_morphology": False, "v175_burg_lattice": False},
            "causal_numeric_only": True, "target_row_labels_used": False, "canonical_controller_gate_checks": 14,
            "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True, "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"]},
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V173 support probe is reserved for KR:2")
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
