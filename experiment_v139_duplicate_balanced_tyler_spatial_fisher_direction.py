"""V139 duplicate-balanced Tyler spatial-Fisher direction challenger.

Every embargoed same-market train cut robust-scales the immutable 37 causal
numeric fields plus deterministic missing flags, then collapses duplicate rows
to equal event-group centroids.  Deterministic spatial medians estimate pooled
and class locations.  A fixed-shrinkage Tyler fixed-point M-estimator supplies
one robust common elliptical scatter shape, whose radial scale is fixed from the
training median squared radius.  The equal-prior Fisher/LDA closed-form log odds
from class spatial-center separation and common robust scatter directly predict
direction, without a supervised optimizer or fitted probability calibrator.

There is no V69 offset or residual correction, class-tail CVaR/superquantile,
QDA class covariance/determinant, coefficient median ensemble, source/confidence
surface, threshold grid, return weighting, text, ticker or external data.
Inner-past evidence selects exact V69 or fixed .25/.50 blends; outer labels stay
evaluation-only under a strict 35-minute embargo. V69 confidence/high-confidence
remain exact and material failure restores the exact entire V69 DataFrame.

Audit, US:2 smoke and KR:2 support use CPU30-31/two threads/no GPU and write
nothing. Full execution is accepted only by no-argument
MARKET_BIO_VERSION_OUTPUT binding.
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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V139_DUPLICATE_BALANCED_TYLER_SPATIAL_FISHER_V1"
VERSION = 139
HYPOTHESIS = "DUPLICATE_BALANCED_TYLER_SPATIAL_FISHER_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

STATE_DIMENSION = 2 * len(FEATURES)
SPATIAL_MEDIAN_MAX_ITERATIONS = 500
SPATIAL_MEDIAN_TOLERANCE = 1e-7
TYLER_MAX_ITERATIONS = 300
TYLER_TOLERANCE = 1e-8
TYLER_IDENTITY_SHRINKAGE = 0.25
RADIAL_SCALE_FLOOR = 1e-4
SCORE_CLIP = 20.0
SEED = 13901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "TYLER_SPATIAL_FISHER_W0.25", "weight": 0.25},
    {"name": "TYLER_SPATIAL_FISHER_W0.50", "weight": 0.50},
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


def verify_static_spec() -> dict[str, Any]:
    require(STATE_DIMENSION == 74, "V139 state dimension changed")
    require(0.0 < TYLER_IDENTITY_SHRINKAGE < 1.0, "V139 Tyler shrinkage invalid")
    return {
        "causal_numeric_feature_n": len(FEATURES),
        "deterministic_missing_flag_n": len(FEATURES),
        "state_dimension": STATE_DIMENSION,
        "event_group_representation": "equal centroid per event_group after train-cut robust state transform",
        "pooled_location": "deterministic spatial median",
        "class_locations": "deterministic spatial medians",
        "scatter": "pooled fixed-point Tyler M-shape with fixed identity shrinkage and trace normalization",
        "tyler_identity_shrinkage": TYLER_IDENTITY_SHRINKAGE,
        "radial_scale": "training median Tyler squared radius divided by fixed chi-square median approximation",
        "direction": "equal-prior common-scatter Fisher/LDA closed-form log odds",
        "supervised_head_optimizer": False,
        "learned_threshold_calibrator_or_regularization_grid": False,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, robust causal state, metrics, canonical gate, blending and atomic IO only; no V119 output is read",
    }
    audit["v139_access_contract"] = {
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


def event_group_centroids(
    ordered: pd.DataFrame,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape == (len(ordered), STATE_DIMENSION), "V139 state shape changed")
    group_ids = ordered.event_group_id.astype(str).to_numpy()
    labels = ordered.y.to_numpy(int)
    unique_groups = np.unique(group_ids)
    centroids: list[np.ndarray] = []
    group_labels: list[int] = []
    group_sizes: list[int] = []
    for group_id in unique_groups:
        positions = np.flatnonzero(group_ids == group_id)
        values = np.unique(labels[positions])
        require(len(values) == 1, "V139 event group label conflict")
        centroids.append(np.mean(state[positions], axis=0))
        group_labels.append(int(values[0]))
        group_sizes.append(len(positions))
    matrix = np.asarray(centroids, dtype=float)
    target = np.asarray(group_labels, dtype=int)
    require(
        matrix.ndim == 2 and matrix.shape[1] == STATE_DIMENSION
        and np.isfinite(matrix).all() and set(np.unique(target)) == {0, 1},
        "V139 event-group centroids invalid",
    )
    return matrix, target, {
        "row_n": len(ordered),
        "event_group_n": len(matrix),
        "duplicate_rows_removed": len(ordered) - len(matrix),
        "class_event_group_n": np.bincount(target, minlength=2).tolist(),
        "maximum_group_size": int(max(group_sizes)),
        "centroid_sha256": array_sha256(matrix),
        "equal_event_group_weight": True,
    }


def spatial_median(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(points.ndim == 2 and len(points) >= 2, "V139 spatial median support invalid")
    current = np.median(points, axis=0)
    converged = False
    exact_observation = False
    movement = float("inf")
    iteration = 0
    for iteration in range(1, SPATIAL_MEDIAN_MAX_ITERATIONS + 1):
        delta = points - current
        distance = np.linalg.norm(delta, axis=1)
        nearest = int(np.argmin(distance))
        if float(distance[nearest]) <= SPATIAL_MEDIAN_TOLERANCE:
            current = points[nearest].copy()
            exact_observation = True
            converged = True
            movement = 0.0
            break
        inverse = 1.0 / np.maximum(distance, SPATIAL_MEDIAN_TOLERANCE)
        updated = np.sum(points * inverse[:, None], axis=0) / np.sum(inverse)
        movement = float(np.linalg.norm(updated - current))
        current = updated
        if movement <= SPATIAL_MEDIAN_TOLERANCE:
            converged = True
            break
    require(converged and np.isfinite(current).all(), "V139 spatial median failed to converge")
    return current, {
        "algorithm": "deterministic Weiszfeld spatial median of event-group state",
        "iterations": iteration,
        "converged": converged,
        "exact_observation": exact_observation,
        "final_movement_l2": movement,
        "location_sha256": array_sha256(current),
    }


def tyler_common_scatter(
    points: np.ndarray,
    pooled_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residual = points - pooled_center
    dimension = residual.shape[1]
    require(dimension == STATE_DIMENSION and len(points) > dimension, "V139 Tyler support invalid")
    shape = np.eye(dimension, dtype=float)
    converged = False
    relative_change = float("inf")
    iteration = 0
    for iteration in range(1, TYLER_MAX_ITERATIONS + 1):
        inverse = np.linalg.inv(shape)
        radius = np.einsum("ni,ij,nj->n", residual, inverse, residual)
        radius = np.maximum(radius, 1e-10)
        raw = dimension * np.einsum("ni,nj,n->ij", residual, residual, 1.0 / radius) / len(points)
        updated = (1.0 - TYLER_IDENTITY_SHRINKAGE) * raw + TYLER_IDENTITY_SHRINKAGE * np.eye(dimension)
        updated = 0.5 * (updated + updated.T)
        updated *= dimension / float(np.trace(updated))
        relative_change = float(np.linalg.norm(updated - shape, ord="fro") / dimension)
        shape = updated
        if relative_change <= TYLER_TOLERANCE:
            converged = True
            break
    require(converged and np.isfinite(shape).all(), "V139 Tyler fixed point failed to converge")
    eigenvalues = np.linalg.eigvalsh(shape)
    require(float(eigenvalues[0]) > 0.0, "V139 Tyler shape not positive definite")
    shape_inverse = np.linalg.inv(shape)
    squared_radius = np.einsum("ni,ij,nj->n", residual, shape_inverse, residual)
    chi_square_median_approx = dimension * (1.0 - 2.0 / (9.0 * dimension)) ** 3
    radial_scale = max(float(np.median(squared_radius) / chi_square_median_approx), RADIAL_SCALE_FLOOR)
    scatter = radial_scale * shape
    scatter_inverse = np.linalg.inv(scatter)
    require(np.isfinite(scatter).all() and np.isfinite(scatter_inverse).all(), "V139 scatter invalid")
    return scatter, scatter_inverse, {
        "iterations": iteration,
        "converged": converged,
        "final_relative_frobenius_change": relative_change,
        "identity_shrinkage": TYLER_IDENTITY_SHRINKAGE,
        "trace_normalized_dimension": float(np.trace(shape)),
        "shape_min_eigenvalue": float(eigenvalues[0]),
        "shape_max_eigenvalue": float(eigenvalues[-1]),
        "shape_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "radial_scale": radial_scale,
        "median_squared_radius": float(np.median(squared_radius)),
        "scatter_sha256": array_sha256(scatter),
        "scatter_inverse_sha256": array_sha256(scatter_inverse),
    }


def tyler_spatial_fisher_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V139 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    centroids, group_target, group_audit = event_group_centroids(ordered, train_state)
    pooled_center, pooled_center_audit = spatial_median(centroids)
    down_center, down_center_audit = spatial_median(centroids[group_target == 0])
    up_center, up_center_audit = spatial_median(centroids[group_target == 1])
    scatter, scatter_inverse, scatter_audit = tyler_common_scatter(centroids, pooled_center)
    difference = up_center - down_center
    coefficient = scatter_inverse @ difference
    midpoint = 0.5 * (up_center + down_center)
    intercept = -float(midpoint @ coefficient)
    raw_score = target_state @ coefficient + intercept
    clipped_score = np.clip(raw_score, -SCORE_CLIP, SCORE_CLIP)
    probability = 1.0 / (1.0 + np.exp(-clipped_score))
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all() and float(np.linalg.norm(difference)) > 0.0, "V139 probability invalid")
    train_group_score = centroids @ coefficient + intercept
    center_margin = float((up_center - down_center) @ coefficient)
    require(center_margin > 0.0, "V139 Fisher orientation invalid")
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_and_missing_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "event_groups": group_audit,
        "pooled_spatial_center": pooled_center_audit,
        "down_spatial_center": down_center_audit,
        "up_spatial_center": up_center_audit,
        "tyler_scatter": scatter_audit,
        "fisher_direction": {
            "equal_class_prior": True,
            "coefficient_l2": float(np.linalg.norm(coefficient)),
            "coefficient_sha256": array_sha256(coefficient),
            "intercept": intercept,
            "class_center_mahalanobis_squared": center_margin,
            "train_group_score_median": float(np.median(train_group_score)),
            "train_group_score_iqr": float(np.quantile(train_group_score, 0.75) - np.quantile(train_group_score, 0.25)),
            "supervised_optimizer": False,
            "fitted_probability_calibrator": False,
        },
        "target_rows_used_for_robust_scale_centroids_centers_scatter_or_direction": False,
        "target_labels_used": False,
        "v69_offset_or_residual_correction": False,
        "worst_class_superquantile_cvar_or_tail_objective": False,
        "class_specific_covariance_qda_or_density_determinant": False,
        "chronological_coefficient_median_ensemble": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "raw_score_min": float(raw_score.min()),
            "raw_score_max": float(raw_score.max()),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = tyler_spatial_fisher_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline_metric,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"],
            "architecture": architecture,
            "metrics": current,
            "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 causal Tyler spatial-Fisher blend; fixed 2*AUC+BA with BA/net eligibility",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "scatter_shrinkage_direction_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v139_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([(smoke_market, 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V139 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = tyler_spatial_fisher_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v139_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                blend_probability(baseline, challenger, architecture["weight"]),
                confidence,
                high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["tyler_fisher_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v139_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market,
            "fold": fold,
            "inner_chronology": inner_chronology,
            "outer_chronology": outer_chronology,
            "policy": policy,
            "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V139 TYLER_FISHER] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V139 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V139 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V139 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V139 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V139 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V139 duplicate-balanced Tyler spatial-Fisher policy",
        "seed": SEED,
        "requested_draws": draws,
        "blocks": len(blocks),
        **{name: interval(items) for name, items in values.items()},
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V139 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    tyler_fisher_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["static_spec"]["state_dimension"] == STATE_DIMENSION
        and audit[side]["event_groups"]["equal_event_group_weight"]
        and audit[side]["pooled_spatial_center"]["converged"]
        and audit[side]["down_spatial_center"]["converged"]
        and audit[side]["up_spatial_center"]["converged"]
        and audit[side]["tyler_scatter"]["converged"]
        and audit[side]["tyler_scatter"]["identity_shrinkage"] == TYLER_IDENTITY_SHRINKAGE
        and audit[side]["tyler_scatter"]["shape_min_eigenvalue"] > 0.0
        and audit[side]["fisher_direction"]["equal_class_prior"]
        and not audit[side]["fisher_direction"]["supervised_optimizer"]
        and not audit[side]["fisher_direction"]["fitted_probability_calibrator"]
        and not audit[side]["target_rows_used_for_robust_scale_centroids_centers_scatter_or_direction"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v69_offset_or_residual_correction"]
        and not audit[side]["worst_class_superquantile_cvar_or_tail_objective"]
        and not audit[side]["class_specific_covariance_qda_or_density_determinant"]
        and not audit[side]["chronological_coefficient_median_ensemble"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
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
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "duplicate_balanced_tyler_spatial_fisher_contract_verified": tyler_fisher_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V139 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": baseline,
        "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "candidate_native_robustness_bootstrap": native_bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V139 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V139 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V139 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V139", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V139_DIAGNOSTIC_TYLER_SPATIAL_FISHER": evaluation["candidate_summary"],
                "V139_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V139", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V139", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V139_DUPLICATE_BALANCED_TYLER_SPATIAL_FISHER_REPORT.json": {
            "version": "V139", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V139", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V139_DUPLICATE_BALANCED_TYLER_SPATIAL_FISHER_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V139_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {"output": str(out), "required_controller_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"]}


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
            "explicit mode/output conflicts with controller-bound no-argument V139 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V139 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V139 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test or args.support_probe) else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "equal event-group 74D state; pooled/class spatial medians; fixed-shrinkage Tyler common scatter; equal-prior closed-form Fisher/LDA direction",
            "causal_numeric_and_missing_only": True, "target_row_labels_used": False,
            "threshold_calibration_regularization_or_post_outer_tuning": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v139_compatible": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    bounded = bool(args.smoke_test or args.support_probe)
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V139 support probe is reserved for KR:2")
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
                "nested_bootstrap_reason": "KR:2 support probe validates model support only; standard material bootstrap remains on US smoke/full evaluation",
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
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["inner_model_audit"] = audits[0]["inner_model_audit"]
        summary["outer_model_audit"] = audits[0]["outer_model_audit"]
        summary["candidate_native_robustness_bootstrap"] = evaluation["candidate_native_robustness_bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
