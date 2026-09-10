"""V154 deterministic RBF Nystrom pairwise-AUC direction challenger.

Each strict-past same-market cut robustly scales the immutable 37 causal
numeric fields plus missing flags and collapses duplicates to equal event-group
centroids.  Label-independent deterministic farthest-first landmarks and a
train-only median squared-distance RBF scale define an exact regularized
Nystrom map.  A single no-intercept ridge head then minimizes a globally
class-balanced deterministic pairwise logistic AUC loss in that nonlinear map.

V84 instead uses label-independent random Fourier frequencies followed by
pointwise Bayesian logistic/Laplace inference.  V131 uses a raw-state linear
score and smooth-max source-time environment pairwise losses.  V154 has neither
RFF/Laplace uncertainty nor environment maximin aggregation.  Inner labels
select exact V69 or fixed .25/.50 blends; outer labels are evaluation-only
under 35-minute embargo/event purge.  V69 confidence/high_conf remain exact,
and failure restores the entire atomic V69 frame.  Prep uses CPU30-31, two
threads, no GPU and no workspace output.
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import experiment_v152_source_deviation_hierarchical_group_lasso_direction as utilities


ROOT = utilities.ROOT
V69_DIR = utilities.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V154" / "LOCAL_FORBIDDEN"
VERSION = 154
HYPOTHESIS = "NYSTROM_RBF_PAIRWISE_AUC_DIRECTION_V1"
FEATURES = utilities.FEATURES
STATE_DIMENSION = utilities.STATE_DIMENSION
EXPECTED_V69_EXPERIMENT_ID = utilities.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v152_source_deviation_hierarchical_group_lasso_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "2f3666d8676a9ca65e054761922096afe881625395298cbe7c9a9338b91123a7"

LANDMARK_N = 64
NYSTROM_REGULARIZATION = 1e-4
MAX_CLASS_QUANTILES = 512
PAIR_OFFSETS = (0.0, 0.25, 0.50, 0.75)
PAIRWISE_L2 = 0.05
OPTIMIZER_MAX_ITERATIONS = 400
SCORE_CLIP = 24.0
SEED = 15401
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "NYSTROM_RBF_PAIRWISE_AUC_W0.25", "weight": 0.25},
    {"name": "NYSTROM_RBF_PAIRWISE_AUC_W0.50", "weight": 0.50},
)

require = utilities.require
sha256 = utilities.sha256
array_sha256 = utilities.array_sha256
clean = utilities.clean
bool_series = utilities.bool_series
atomic_json = utilities.atomic_json
atomic_csv = utilities.atomic_csv
load_authorized = utilities.load_authorized
metric = utilities.metric
blend_probability = utilities.blend_probability
controller = utilities.controller
v44 = utilities.v44
numeric = utilities.numeric
scaffold = utilities.scaffold

STATIC_SPEC = {
    "causal_numeric_feature_n": len(FEATURES),
    "deterministic_missing_flag_n": len(FEATURES),
    "state_dimension": STATE_DIMENSION,
    "event_unit": "equal event-group robust-state centroid",
    "landmarks": "label-independent deterministic train-only farthest-first",
    "landmark_n": LANDMARK_N,
    "kernel": "RBF with train-only median train-to-landmark squared-distance scale",
    "nystrom": "K_nm times regularized K_mm inverse square root",
    "nystrom_regularization": NYSTROM_REGULARIZATION,
    "pair_design": "global class-quantile UP-minus-DOWN pairs at four fixed cyclic offsets",
    "maximum_class_quantiles": MAX_CLASS_QUANTILES,
    "pair_offsets": list(PAIR_OFFSETS),
    "objective": "mean pairwise logistic AUC loss plus fixed L2",
    "pairwise_l2": PAIRWISE_L2,
    "head": "single no-intercept nonlinear Nystrom direction",
    "landmark_kernel_pair_l2_or_blend_grid": False,
}
require(STATE_DIMENSION == 74 and LANDMARK_N == 64 and len(PAIR_OFFSETS) == 4, "V154 static contract changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V152 utility scaffold changed")
    audit = utilities.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust causal state, equal-event grouping, metrics, canonical gate, fixed blending, bootstrap and atomic IO only; no V152 output is read",
    }
    audit["v154_access_contract"] = {
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


def squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distance, 0.0)


def deterministic_landmarks(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.ndim == 2 and state.shape[1] == STATE_DIMENSION, "V154 landmark state shape invalid")
    require(len(state) >= LANDMARK_N, "V154 landmark support too small")
    coordinate_median = np.median(state, axis=0)
    first_distance = np.sum(np.square(state - coordinate_median), axis=1)
    selected = [int(np.argmin(first_distance))]
    minimum_distance = np.sum(np.square(state - state[selected[0]]), axis=1)
    for _ in range(1, LANDMARK_N):
        minimum_distance[selected] = -1.0
        next_index = int(np.argmax(minimum_distance))
        require(next_index not in selected and minimum_distance[next_index] >= 0.0, "V154 landmark selection degenerate")
        selected.append(next_index)
        current = np.sum(np.square(state - state[next_index]), axis=1)
        minimum_distance = np.minimum(np.maximum(minimum_distance, 0.0), current)
    index = np.asarray(selected, int)
    landmarks = state[index].copy()
    require(np.unique(index).size == LANDMARK_N and np.isfinite(landmarks).all(), "V154 landmarks invalid")
    return landmarks, index, {
        "method": "label-independent deterministic farthest-first from point nearest coordinatewise median",
        "landmark_n": LANDMARK_N,
        "landmark_indices_sha256": array_sha256(index),
        "landmark_state_sha256": array_sha256(landmarks),
        "labels_used": False,
        "target_rows_used": False,
    }


class NystromRBF:
    def __init__(self) -> None:
        self.landmarks = np.empty((0, STATE_DIMENSION), float)
        self.projection = np.empty((0, 0), float)
        self.length_squared = 0.0
        self.mean = np.empty(0, float)
        self.scale = np.empty(0, float)
        self.audit: dict[str, Any] = {}

    def fit(self, state: np.ndarray) -> "NystromRBF":
        self.landmarks, landmark_index, landmark_audit = deterministic_landmarks(state)
        train_landmark_squared = squared_distance(state, self.landmarks)
        positive = train_landmark_squared[train_landmark_squared > 1e-12]
        require(len(positive) >= len(state), "V154 positive distance support too small")
        self.length_squared = float(np.median(positive))
        require(np.isfinite(self.length_squared) and self.length_squared > 1e-10, "V154 kernel scale invalid")
        landmark_squared = squared_distance(self.landmarks, self.landmarks)
        kernel_mm = np.exp(-landmark_squared / (2.0 * self.length_squared))
        eigenvalue, eigenvector = np.linalg.eigh(kernel_mm + NYSTROM_REGULARIZATION * np.eye(LANDMARK_N))
        require(float(eigenvalue.min()) > 0.0 and np.isfinite(eigenvalue).all(), "V154 regularized kernel eigensystem invalid")
        self.projection = (eigenvector / np.sqrt(eigenvalue)[None, :]) @ eigenvector.T
        kernel_nm = np.exp(-train_landmark_squared / (2.0 * self.length_squared))
        raw = kernel_nm @ self.projection
        self.mean = raw.mean(axis=0)
        standard = raw.std(axis=0)
        self.scale = np.where(standard > 1e-8, standard, 1.0)
        transformed = np.clip((raw - self.mean) / self.scale, -8.0, 8.0)
        require(np.isfinite(transformed).all(), "V154 train Nystrom map invalid")
        self.audit = {
            **landmark_audit,
            "length_squared": self.length_squared,
            "kernel_definition": "exp(-squared_distance/(2*train_median_positive_squared_distance))",
            "regularization": NYSTROM_REGULARIZATION,
            "kernel_mm_eigenvalue_min_median_max": [float(eigenvalue.min()), float(np.median(eigenvalue)), float(eigenvalue.max())],
            "projection_sha256": array_sha256(self.projection),
            "feature_mean_sha256": array_sha256(self.mean),
            "feature_scale_sha256": array_sha256(self.scale),
            "train_feature_sha256": array_sha256(transformed),
            "train_only": True,
        }
        return self

    def transform(self, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        kernel = np.exp(-squared_distance(state, self.landmarks) / (2.0 * self.length_squared))
        raw = kernel @ self.projection
        transformed = np.clip((raw - self.mean) / self.scale, -8.0, 8.0)
        require(transformed.shape == (len(state), LANDMARK_N), "V154 target Nystrom shape invalid")
        require(np.isfinite(transformed).all(), "V154 target Nystrom map invalid")
        return transformed, {
            "rows": len(state), "feature_n": LANDMARK_N,
            "feature_sha256": array_sha256(transformed),
            "target_rows_used_for_landmarks_kernel_scale_projection_or_feature_scale": False,
        }


def quantile_positions(length: int, count: int) -> np.ndarray:
    require(1 <= count <= length, "V154 quantile position count invalid")
    return np.floor((np.arange(count, dtype=float) + 0.5) * length / count).astype(int)


def deterministic_pair_design(feature: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    down = np.flatnonzero(target == 0)
    up = np.flatnonzero(target == 1)
    pair_quantiles = min(MAX_CLASS_QUANTILES, len(down), len(up))
    require(pair_quantiles >= 100, "V154 pair support too small")
    down_feature = feature[down[quantile_positions(len(down), pair_quantiles)]]
    up_feature = feature[up[quantile_positions(len(up), pair_quantiles)]]
    differences = []
    offsets = []
    for fraction in PAIR_OFFSETS:
        offset = int(np.floor(fraction * pair_quantiles))
        offsets.append(offset)
        differences.append(up_feature - np.roll(down_feature, offset, axis=0))
    pair = np.vstack(differences)
    require(pair.shape == (len(PAIR_OFFSETS) * pair_quantiles, LANDMARK_N), "V154 pair design shape changed")
    require(np.isfinite(pair).all(), "V154 pair design invalid")
    return pair, {
        "down_event_group_n": len(down), "up_event_group_n": len(up),
        "class_quantile_n": pair_quantiles,
        "offset_fractions": list(PAIR_OFFSETS), "offset_positions": offsets,
        "pair_n": len(pair),
        "pairing": "UP-minus-DOWN class-quantile Nystrom features at four fixed cyclic DOWN offsets",
        "pair_sha256": array_sha256(pair),
        "source_time_environment_used": False,
        "return_magnitude_or_order_used": False,
    }


def fit_pairwise_auc(pair: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        margin = np.clip(pair @ coefficient, -SCORE_CLIP, SCORE_CLIP)
        value = float(np.mean(np.logaddexp(0.0, -margin)) + 0.5 * PAIRWISE_L2 * np.dot(coefficient, coefficient))
        gradient = -(pair.T @ expit(-margin)) / len(pair) + PAIRWISE_L2 * coefficient
        return value, gradient

    result = minimize(
        objective, np.zeros(LANDMARK_N, float), method="L-BFGS-B", jac=True,
        options={"maxiter": OPTIMIZER_MAX_ITERATIONS, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    require(bool(result.success) and np.isfinite(result.x).all(), f"V154 pairwise optimizer failed: {result.message}")
    final_value, final_gradient = objective(np.asarray(result.x, float))
    return np.asarray(result.x, float), {
        "objective": "mean global class-balanced pairwise logistic AUC loss plus fixed L2",
        "optimizer": "L-BFGS-B analytic gradient",
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "l2": PAIRWISE_L2, "final_objective": final_value,
        "gradient_linf": float(np.max(np.abs(final_gradient))),
        "coefficient_l2": float(np.linalg.norm(result.x)),
        "coefficient_sha256": array_sha256(np.asarray(result.x, float)),
        "intercept": False,
        "environment_maximin_or_adversarial_weights": False,
    }


def nystrom_pairwise_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V154 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    group_state, group_target, group_source, group_id, group_audit = utilities.group_state_source(ordered, train_state)
    mapper = NystromRBF().fit(group_state)
    train_feature, train_feature_audit = mapper.transform(group_state)
    target_feature, target_feature_audit = mapper.transform(target_state)
    pair, pair_audit = deterministic_pair_design(train_feature, group_target)
    coefficient, fit_audit = fit_pairwise_auc(pair)
    target_score = np.clip(target_feature @ coefficient, -SCORE_CLIP, SCORE_CLIP)
    probability = np.clip(expit(target_score), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V154 probability invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_id), "target_n": len(target_frame),
        "causal_numeric_and_missing_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "group_audit": group_audit,
        "landmark_and_map_audit": mapper.audit,
        "train_feature_audit": train_feature_audit,
        "target_feature_audit": target_feature_audit,
        "pair_audit": pair_audit, "fit_audit": fit_audit,
        "target_rows_used_for_robust_scale_centroids_landmarks_kernel_pairs_or_fit": False,
        "target_labels_used": False,
        "label_independent_landmarks": True,
        "exact_regularized_nystrom_map": True,
        "pairwise_auc_objective": True,
        "random_fourier_or_laplace_bayes": False,
        "source_time_maximin_pairwise": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "target_score_sha256": array_sha256(target_score),
            "probability_sha256": array_sha256(probability),
        },
    }


def supported_source_ba_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index].y.to_numpy(int)
        if len(index) < 40 or np.unique(target).size != 2:
            continue
        deltas[str(source)] = float(
            balanced_accuracy_score(target, candidate[index] >= 0.5)
            - balanced_accuracy_score(target, baseline[index] >= 0.5)
        )
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = nystrom_pairwise_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_and_missing_only"]
        and model_audit["landmark_and_map_audit"]["landmark_n"] == LANDMARK_N
        and not model_audit["landmark_and_map_audit"]["labels_used"]
        and model_audit["landmark_and_map_audit"]["train_only"]
        and model_audit["label_independent_landmarks"]
        and model_audit["exact_regularized_nystrom_map"]
        and model_audit["pairwise_auc_objective"]
        and model_audit["fit_audit"]["optimizer_success"]
        and not model_audit["fit_audit"]["environment_maximin_or_adversarial_weights"]
        and not model_audit["target_rows_used_for_robust_scale_centroids_landmarks_kernel_pairs_or_fit"]
        and not model_audit["target_labels_used"]
        and not model_audit["random_fourier_or_laplace_bayes"]
        and not model_audit["source_time_maximin_pairwise"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 deterministic RBF Nystrom pairwise-AUC blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "landmark_kernel_pair_l2_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.SEED = SEED
    scaffold.choose_inner = choose_inner
    scaffold.ordinal_direction = nystrom_pairwise_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v144_model": "v154_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "nystrom_prob", "v144_model": "v154_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V154 NYSTROM PAIRWISE AUC] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V154 deterministic RBF Nystrom pairwise-AUC policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V154 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    model_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["landmark_and_map_audit"]["landmark_n"] == LANDMARK_N
        and not audit[side]["landmark_and_map_audit"]["labels_used"]
        and audit[side]["landmark_and_map_audit"]["train_only"]
        and audit[side]["label_independent_landmarks"]
        and audit[side]["exact_regularized_nystrom_map"]
        and audit[side]["pairwise_auc_objective"]
        and audit[side]["fit_audit"]["optimizer_success"]
        and not audit[side]["fit_audit"]["environment_maximin_or_adversarial_weights"]
        and not audit[side]["target_rows_used_for_robust_scale_centroids_landmarks_kernel_pairs_or_fit"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["random_fourier_or_laplace_bayes"]
        and not audit[side]["source_time_maximin_pairwise"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "nystrom_rbf_pairwise_auc_contract_verified": model_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V154 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "candidate_native_robustness_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V154 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V154 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V154 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V154").resolve(), "V154 output is outside controller research/staging/V154")
    require(out.exists() and out.is_dir(), "V154 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V154 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V154 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V154 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V154 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V154", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V154_DIAGNOSTIC_NYSTROM_RBF_PAIRWISE_AUC": evaluation["candidate_summary"],
                "V154_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V154", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V154", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V154_NYSTROM_RBF_PAIRWISE_AUC_REPORT.json": {
            "version": "V154", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V154", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V154", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V154", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V154_NYSTROM_RBF_PAIRWISE_AUC_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V154_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out), "artifact_count": len(files) + 1,
        "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json"),
    }


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
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V154 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V154 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V154 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "label-independent farthest-first RBF Nystrom map plus one global no-intercept pairwise logistic AUC head",
            "prior_nystrom_collision_found": False,
            "v84_random_fourier_laplace_collision": False,
            "v131_linear_source_time_maximin_collision": False,
            "v155_random_maclaurin_polynomial_collision": False,
            "causal_numeric_only": True, "target_row_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V154 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS,
                "runtime": runtime, "market": "KR", "fold": 2,
                "strict_inner_chronology": audit["inner_chronology"],
                "strict_outer_chronology": audit["outer_chronology"],
                "raw_inner_selected": audit["policy"]["selected"],
                "raw_inner_best_non_noop": best_non_noop,
                "raw_outer_baseline": audit["outer_baseline"],
                "raw_outer_selected": audit["outer_candidate"],
                "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
                "inner_model_audit": audit["inner_model_audit"],
                "outer_model_audit": audit["outer_model_audit"],
                "selection_locked_before_outer_evaluation": True,
                "nested_bootstrap_executed": False,
                "exact_entire_v69_fallback_contract": True,
                "output_written": False,
            }), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "runtime": runtime,
        "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None,
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": material_gate(evaluation),
        "output_written": False,
    }
    if args.smoke_test:
        summary.update({
            "raw_inner_selected": audits[0]["policy"]["selected"],
            "raw_inner_best_non_noop": best_non_noop,
            "raw_outer_baseline": audits[0]["outer_baseline"],
            "raw_outer_selected": audits[0]["outer_candidate"],
            "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
            "inner_model_audit": audits[0]["inner_model_audit"],
            "outer_model_audit": audits[0]["outer_model_audit"],
            "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        })
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
