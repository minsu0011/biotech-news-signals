"""V133 class-balanced Cantelli minimax-moment direction challenger.

Each strict embargoed same-market train cut robust-scales the immutable 37
causal numeric coordinates plus deterministic missing flags and collapses
duplicates to equal event-group centroids.  Separate class means and fixed
shrinkage covariances define a distribution-free one-sided moment problem.
A deterministic unit-sphere optimizer maximizes projected class-mean gap
divided by the sum of class projected standard deviations.  For the resulting
direction, the affine threshold analytically equalizes DOWN and UP standardized
margins; the associated worst-class correctness guarantee is the Cantelli
lower bound k^2/(1+k^2).  The standardized affine margin maps directly through
a sigmoid.

This is a moment-robust chance-bound objective, not a Gaussian/QDA density,
source reliability reranker, pair/list/AUC loss, twin hyperplane, path/reservoir
representation, view-head ensemble, geometry, tree, kernel, or neural model.
Atomic V69 inner-past evidence chooses exact V69 or fixed .25/.50 blends.
Outer labels are evaluation-only behind the strict 35-minute embargo; V69
confidence/high-confidence remain exact.  Any material failure restores the
exact entire V69 DataFrame.  Preparation uses CPU30-31, two numeric threads,
zero GPU model calls, and performs no output write.
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
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v129_economic_salience_twin_hyperplane_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V133_CLASS_BALANCED_CANTELLI_MINIMAX_MOMENT_DIRECTION_V1"
VERSION = 133
HYPOTHESIS = "CLASS_BALANCED_CANTELLI_MINIMAX_MOMENT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.scaffold.scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v129_economic_salience_twin_hyperplane_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "aee75dc19c6e803a0c51fc9e4e833fc85f6d0e2fe0d393e3e099b938d6ab2473"

STATE_DIMENSION = 2 * len(FEATURES)
COVARIANCE_DIAGONAL_SHRINKAGE = 0.25
COVARIANCE_FLOOR = 1e-3
OPTIMIZER_MAX_ITERATIONS = 240
SCORE_CLIP = 20.0
SEED = 13301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CANTELLI_MINIMAX_MOMENT_W0.25", "weight": 0.25},
    {"name": "CANTELLI_MINIMAX_MOMENT_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
chronology = scaffold.chronology
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
inner_partition = scaffold.inner_partition
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44
numeric = scaffold.numeric


STATIC_SPEC = {
    "numeric_feature_n": len(FEATURES),
    "missing_flag_n": len(FEATURES),
    "state_dimension": STATE_DIMENSION,
    "duplicate_unit": "equal event-group robust-state centroid",
    "moments": "separate duplicate-balanced class mean and fixed-shrinkage covariance",
    "covariance_diagonal_shrinkage": COVARIANCE_DIAGONAL_SHRINKAGE,
    "covariance_floor": COVARIANCE_FLOOR,
    "objective": "maximize projected class mean gap divided by sum of projected class standard deviations on the unit sphere",
    "threshold": "analytic equalization of DOWN and UP standardized one-sided margins",
    "guarantee": "Cantelli lower bound k^2/(1+k^2) for the worst class",
    "probability": "sigmoid of affine margin divided by mean class projected standard deviation",
}
require(STATE_DIMENSION == 74, "V133 robust-state dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V129 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust state, metrics, canonical gate, blending and atomic IO only; no V129 output is read",
    }
    audit["v133_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def group_state(frame: pd.DataFrame, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(state.shape == (len(frame), STATE_DIMENSION), "V133 robust state shape changed")
    work = pd.DataFrame(state)
    work["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    work["y"] = frame.y.to_numpy(int)
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V133 event group crosses direction labels")
    centroids = grouped[list(range(STATE_DIMENSION))].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    require(np.isfinite(centroids).all(), "V133 event-group centroids invalid")
    return centroids, target, group_ids


def weighted_class_moments(
    state: np.ndarray,
    target: np.ndarray,
    event_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if event_weights is None:
        event_weights = np.ones(len(state), float)
    event_weights = np.asarray(event_weights, float)
    require(event_weights.shape == (len(state),) and np.isfinite(event_weights).all() and np.all(event_weights > 0.0), "V133 event weights invalid")
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    supports: list[int] = []
    effective_masses: list[float] = []
    for direction in (0, 1):
        mask = target == direction
        supports.append(int(mask.sum()))
        require(supports[-1] >= 80, f"V133 class {direction} event-group support too small")
        weights = event_weights[mask]
        weights = weights / float(weights.sum())
        values = state[mask]
        mean = np.sum(weights[:, None] * values, axis=0)
        centered = values - mean
        covariance = centered.T @ (weights[:, None] * centered)
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - COVARIANCE_DIAGONAL_SHRINKAGE) * covariance
            + COVARIANCE_DIAGONAL_SHRINKAGE * diagonal
            + COVARIANCE_FLOOR * np.eye(STATE_DIMENSION)
        )
        require(np.isfinite(mean).all() and np.isfinite(covariance).all(), "V133 class moments invalid")
        means.append(mean)
        covariances.append(covariance)
        effective_masses.append(float(event_weights[mask].sum()))
    return means[0], means[1], covariances[0], covariances[1], {
        "down_event_group_n": supports[0], "up_event_group_n": supports[1],
        "down_effective_mass": effective_masses[0], "up_effective_mass": effective_masses[1],
        "down_mean_sha256": array_sha256(means[0]), "up_mean_sha256": array_sha256(means[1]),
        "down_covariance_sha256": array_sha256(covariances[0]),
        "up_covariance_sha256": array_sha256(covariances[1]),
        "down_covariance_condition": float(np.linalg.cond(covariances[0])),
        "up_covariance_condition": float(np.linalg.cond(covariances[1])),
    }


def fit_cantelli_parameters(
    state: np.ndarray,
    target: np.ndarray,
    event_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    mean_down, mean_up, covariance_down, covariance_up, moment_audit = weighted_class_moments(
        state, target, event_weights,
    )
    gap_vector = mean_up - mean_down
    pooled = covariance_down + covariance_up
    initial = np.linalg.solve(pooled, gap_vector)
    initial_norm = float(np.linalg.norm(initial))
    require(initial_norm > 1e-10, "V133 initial moment direction degenerate")
    initial = initial / initial_norm

    def components(direction: np.ndarray) -> tuple[float, float, float, float]:
        projected_gap = float(direction @ gap_vector)
        down_sd = float(np.sqrt(max(direction @ covariance_down @ direction, 1e-12)))
        up_sd = float(np.sqrt(max(direction @ covariance_up @ direction, 1e-12)))
        return projected_gap, down_sd, up_sd, down_sd + up_sd

    def objective(direction: np.ndarray) -> float:
        projected_gap, _, _, denominator = components(direction)
        return -projected_gap / denominator

    def gradient(direction: np.ndarray) -> np.ndarray:
        projected_gap, down_sd, up_sd, denominator = components(direction)
        denominator_gradient = covariance_down @ direction / down_sd + covariance_up @ direction / up_sd
        return -(gap_vector * denominator - projected_gap * denominator_gradient) / (denominator * denominator)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda direction: float(direction @ direction - 1.0), "jac": lambda direction: 2.0 * direction},
            {"type": "ineq", "fun": lambda direction: float(direction @ gap_vector - 1e-8), "jac": lambda direction: gap_vector},
        ],
        options={"maxiter": OPTIMIZER_MAX_ITERATIONS, "ftol": 1e-11, "disp": False},
    )
    direction = np.asarray(result.x, float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    if float(direction @ gap_vector) < 0.0:
        direction = -direction
    projected_gap, down_sd, up_sd, denominator = components(direction)
    require(np.isfinite(direction).all() and projected_gap > 0.0, "V133 optimized moment direction invalid")
    require(bool(result.success) or abs(float(result.fun) - objective(direction)) <= 1e-7, "V133 moment optimizer failed")
    projected_down_mean = float(direction @ mean_down)
    projected_up_mean = float(direction @ mean_up)
    threshold = (down_sd * projected_up_mean + up_sd * projected_down_mean) / denominator
    standardized_margin = projected_gap / denominator
    cantelli_lower_bound = standardized_margin * standardized_margin / (1.0 + standardized_margin * standardized_margin)
    score_scale = 0.5 * denominator
    require(score_scale > 0.0 and np.isfinite(threshold), "V133 score threshold invalid")
    audit = {
        "moments": moment_audit,
        "optimizer": {
            "method": "deterministic SLSQP on unit sphere with positive projected-gap constraint",
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev),
            "objective": float(result.fun),
            "direction_norm": float(np.linalg.norm(direction)),
            "direction_sha256": array_sha256(direction),
        },
        "projected_gap": projected_gap,
        "down_projected_sd": down_sd, "up_projected_sd": up_sd,
        "standardized_minimax_margin": standardized_margin,
        "cantelli_worst_class_correctness_lower_bound": cantelli_lower_bound,
        "threshold": threshold, "score_scale": score_scale,
        "standardized_down_margin": (threshold - projected_down_mean) / down_sd,
        "standardized_up_margin": (projected_up_mean - threshold) / up_sd,
    }
    require(
        abs(audit["standardized_down_margin"] - audit["standardized_up_margin"]) <= 1e-8,
        "V133 standardized class margins are not equalized",
    )
    return direction, float(threshold), float(score_scale), audit


def native_moment_bootstrap(
    train_state: np.ndarray,
    train_y: np.ndarray,
    target_state: np.ndarray,
    nominal_probability: np.ndarray,
    nominal_direction: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    probabilities = np.empty((len(target_state), draws), float)
    directions = np.empty((draws, STATE_DIMENSION), float)
    for draw in range(draws):
        weights = np.empty(len(train_state), float)
        for direction_class in (0, 1):
            mask = train_y == direction_class
            weights[mask] = generator.exponential(1.0, int(mask.sum()))
        direction, threshold, scale, _ = fit_cantelli_parameters(train_state, train_y, weights)
        if float(direction @ nominal_direction) < 0.0:
            direction = -direction
            threshold = -threshold
        score = np.clip((target_state @ direction - threshold) / scale, -SCORE_CLIP, SCORE_CLIP)
        probabilities[:, draw] = np.clip(expit(score), 1e-6, 1.0 - 1e-6)
        directions[draw] = direction
    median_probability = np.median(probabilities, axis=1)
    row_std = np.std(probabilities, axis=1)
    cosine = directions @ nominal_direction
    return {
        "contract": "class-stratified event-group Bayesian bootstrap of class moments and complete Cantelli direction refit",
        "draws": draws, "seed": seed, "class_stratified": True,
        "event_group_unit_weights": True, "target_labels_used": False,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probabilities - nominal_probability[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal_probability))),
        "direction_agreement_with_nominal": float(np.mean((probabilities >= 0.5) == (nominal_probability[:, None] >= 0.5))),
        "direction_cosine_q10_q50_q90": [float(np.quantile(cosine, q)) for q in (0.1, 0.5, 0.9)],
        "row_probability_std_q10_q50_q90": [float(np.quantile(row_std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median_probability),
    }


def cantelli_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V133 expects one market per fit")
    require(str(ordered.market.iloc[0]) == str(target_frame.market.iloc[0]), "V133 market mismatch")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state_rows, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_state, train_y, group_ids = group_state(ordered, train_state_rows)
    direction, threshold, score_scale, parameter_audit = fit_cantelli_parameters(train_state, train_y)
    score = np.clip((target_state @ direction - threshold) / score_scale, -SCORE_CLIP, SCORE_CLIP)
    probability = np.clip(expit(score), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V133 probability invalid")
    native = None if native_draws <= 0 else native_moment_bootstrap(
        train_state, train_y, target_state, probability, direction, native_draws, native_seed,
    )
    return probability, {
        "market": str(ordered.market.iloc[0]),
        "train_n": len(ordered), "target_n": len(target_frame),
        "train_event_group_n": len(group_ids),
        "causal_numeric_and_missing_only": True,
        "numeric_feature_n": len(FEATURES), "missing_flag_n": len(FEATURES),
        "categorical_text_source_ticker_issuer_or_return_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC, "parameter_audit": parameter_audit,
        "distribution_free_moment_bound": True,
        "gaussian_density_loglikelihood_or_determinant": False,
        "pair_list_auc_or_ranking_loss": False,
        "source_reliability_or_decision_cell_rerank": False,
        "twin_hyperplane_or_return_salience": False,
        "path_reservoir_fourier_or_view_head": False,
        "geometry_tree_kernel_or_neural": False,
        "target_rows_used_for_scale_moments_direction_threshold_or_selection": False,
        "target_labels_used": False,
        "threshold_or_calibration_tuned": False,
        "native_moment_bootstrap": native,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "predicted_up_rate": float(np.mean(probability >= 0.5)),
            "score_q10_q50_q90": [float(np.quantile(score, q)) for q in (0.1, 0.5, 0.9)],
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = cantelli_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_and_missing_only"]
        and model_audit["distribution_free_moment_bound"]
        and model_audit["parameter_audit"]["optimizer"]["success"]
        and model_audit["parameter_audit"]["moments"]["down_event_group_n"] >= 80
        and model_audit["parameter_audit"]["moments"]["up_event_group_n"] >= 80
        and not model_audit["gaussian_density_loglikelihood_or_determinant"]
        and not model_audit["target_rows_used_for_scale_moments_direction_threshold_or_selection"]
        and not model_audit["target_labels_used"]
        and not model_audit["threshold_or_calibration_tuned"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"] + 10.0 * current["all_trade_mean_signed_net"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 Cantelli minimax-moment logit blend; fixed 2*AUC+BA+10*net with BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "covariance_shrinkage_optimizer_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v133_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[
            champion.market.eq(market) & champion.fold.eq(fold)
        ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V133 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = cantelli_direction(
            outer_train, outer_valid,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(
            baseline, challenger, selected["architecture"]["weight"],
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v133_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["cantelli_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v133_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(
            f"[V133 CANTELLI] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V133 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V133 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V133 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V133 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        values["balanced_accuracy_delta"].append(float(
            balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "paired market-fold-time block bootstrap over inner-locked V133 Cantelli policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{name: interval(items) for name, items in values.items()},
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int, smoke: bool,
) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V133 canonical mismatch")
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
    cantelli_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["distribution_free_moment_bound"]
        and audit[side]["parameter_audit"]["optimizer"]["success"]
        and audit[side]["parameter_audit"]["moments"]["down_event_group_n"] >= 80
        and audit[side]["parameter_audit"]["moments"]["up_event_group_n"] >= 80
        and abs(audit[side]["parameter_audit"]["standardized_down_margin"] - audit[side]["parameter_audit"]["standardized_up_margin"]) <= 1e-8
        and not audit[side]["gaussian_density_loglikelihood_or_determinant"]
        and not audit[side]["pair_list_auc_or_ranking_loss"]
        and not audit[side]["source_reliability_or_decision_cell_rerank"]
        and not audit[side]["target_rows_used_for_scale_moments_direction_threshold_or_selection"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["threshold_or_calibration_tuned"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    model_native = [audit["outer_model_audit"]["native_moment_bootstrap"] for audit in audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["class_stratified"] and item["event_group_unit_weights"]
        and not item["target_labels_used"]
        and 0.0 <= item["direction_agreement_with_nominal"] <= 1.0
        for item in model_native
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
        "cantelli_moment_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "class_balanced_cantelli_minimax_moment_contract_verified": cantelli_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V133 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested,
        "candidate_native_robustness_bootstrap": native,
        "model_native_cantelli_moment_bootstrap": model_native,
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
        "model_native_cantelli_moment_bootstrap": evaluation["model_native_cantelli_moment_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V133 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V133 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V133 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V133").resolve(), "V133 output is outside controller research/staging/V133")
    require(out.exists() and out.is_dir(), "V133 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V133 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V133 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V133 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V133 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V133", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V133_DIAGNOSTIC_CANTELLI_MINIMAX_MOMENT": evaluation["candidate_summary"],
                "V133_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V133", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V133", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "V133_CANTELLI_MINIMAX_MOMENT_REPORT.json": {
            "version": "V133", "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V133", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V133", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
            "model_native_cantelli_moment_bootstrap": evaluation["model_native_cantelli_moment_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V133", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V133_CANTELLI_MINIMAX_MOMENT_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V133_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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


def support_probe_kr2(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    outer_champion = champion.loc[
        champion.market.eq("KR") & champion.fold.eq(2)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS["KR"], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_train = prior_train(dev, "KR", start, outer_valid)
    outer_chronology = chronology(outer_train, outer_valid, "V133 KR2 support outer")
    inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, "KR", start)
    inner_probability, inner_audit = cantelli_direction(inner_train, inner_valid)
    outer_probability, outer_audit = cantelli_direction(outer_train, outer_valid)
    require(np.isfinite(inner_probability).all() and np.isfinite(outer_probability).all(), "V133 KR2 support probability invalid")
    return {
        "status": "KR2_SUPPORT_OK_NO_OUTER_EVALUATION",
        "inner_target_n": len(inner_valid), "outer_target_n": len(outer_valid),
        "inner_train_event_group_n": inner_audit["train_event_group_n"],
        "outer_train_event_group_n": outer_audit["train_event_group_n"],
        "inner_class_support": inner_audit["parameter_audit"]["moments"],
        "outer_class_support": outer_audit["parameter_audit"]["moments"],
        "inner_probability_sha256": array_sha256(inner_probability),
        "outer_probability_sha256": array_sha256(outer_probability),
        "inner_optimizer_success": inner_audit["parameter_audit"]["optimizer"]["success"],
        "outer_optimizer_success": outer_audit["parameter_audit"]["optimizer"]["success"],
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe-kr2", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe_kr2
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V133 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V133 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
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
            "architecture": "class-balanced distribution-free Cantelli minimax one-sided moment direction with analytic equal-standardized-margin threshold",
            "causal_numeric_and_missing_only": True,
            "target_rows_or_labels_used": False,
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
                "controller_native_robustness_bootstrap": True,
                "model_native_cantelli_moment_bootstrap": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe_kr2(dev, champion)
            result["runtime"] = runtime
            result["gpu_model_calls"] = 0
            result["output_written"] = False
            print(json.dumps(clean(result), ensure_ascii=False, indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
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
        "folds_executed": len(audits),
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
