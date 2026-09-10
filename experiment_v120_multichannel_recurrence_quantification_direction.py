"""V120 multichannel recurrence-quantification direction challenger.

Each embargoed same-market train cut robust-scales only the immutable 37 causal
numeric fields.  The fixed V119 interpolation maps every row to eight semantic
channels on eight increasing pre-event horizons.  Duplicate-balanced train-only
pooled path distances fix two recurrence radii (q20/q40).  Each row then yields
two symmetric temporal recurrence plots whose recurrence rate, diagonal-line
determinism, vertical-line laminarity, line-length entropies, and recurrence-time
summaries form a fixed 24-dimensional label-free representation.  A train-only
robust representation scale and one fixed duplicate/class-balanced ridge
logistic head produce direction probability.

This is not DTW/prototype matching, Haar scattering, a rough-path signature,
increment SPD covariance, Koopman/DMD transition estimation, graph persistence,
Sinkhorn transport, Bernstein uLSIF, or quantum density fidelity.  Atomic V69
inner-past labels alone select exact V69 or a fixed .25/.50 logit blend.  Outer
labels remain evaluation-only under a strict 35-minute embargo.  V69 confidence
and high-confidence are frozen.  Material failure restores the exact entire V69
DataFrame after canonical 14-gate recomputation.

Audit and US:2 smoke are CPU30-31/two-thread/no-write/no-GPU.  Only a future
authorized no-argument MARKET_BIO_VERSION_OUTPUT invocation may run all folds;
its output is bound exactly to the controller path (including
research/staging/V120) and writes the required reports/current material gate.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V120_MULTICHANNEL_RQA_V1"
VERSION = 120
HYPOTHESIS = "MULTICHANNEL_RECURRENCE_QUANTIFICATION_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

RECURRENCE_QUANTILES = (0.20, 0.40)
MIN_LINE_LENGTH = 2
RQA_FEATURES_PER_RADIUS = 12
RQA_FEATURE_DIMENSION = len(RECURRENCE_QUANTILES) * RQA_FEATURES_PER_RADIUS
RQA_FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 12001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "MULTICHANNEL_RQA_W0.25", "weight": 0.25},
    {"name": "MULTICHANNEL_RQA_W0.50", "weight": 0.50},
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
joint_path = scaffold.joint_path


def verify_static_spec() -> dict[str, Any]:
    require(len(scaffold.CHANNEL_SPEC) == 8 and len(scaffold.HORIZON_GRID) == 8, "V120 path geometry changed")
    require(RQA_FEATURE_DIMENSION == 24 and RECURRENCE_QUANTILES == (0.20, 0.40), "V120 RQA specification changed")
    return {
        "channel_n": 8,
        "horizon_n": 8,
        "horizon_grid_minutes": scaffold.HORIZON_GRID.tolist(),
        "channel_names": list(scaffold.CHANNEL_NAMES),
        "recurrence_quantiles": list(RECURRENCE_QUANTILES),
        "minimum_line_length": MIN_LINE_LENGTH,
        "features_per_radius": RQA_FEATURES_PER_RADIUS,
        "feature_dimension": RQA_FEATURE_DIMENSION,
        "threshold_fit": "duplicate-balanced pooled train-only off-diagonal path distances",
        "recurrence_plot": "symmetric temporal 8x8 with diagonal excluded",
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending, and atomic IO only; no V119 output is read",
    }
    audit["v120_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    group_size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(group_size, 1.0)
    require(np.isfinite(weight).all() and np.all(weight > 0.0), "V120 duplicate weights invalid")
    return weight


def pairwise_path_distances(path: np.ndarray) -> np.ndarray:
    require(path.ndim == 3 and path.shape[1:] == (8, 8), "V120 path shape changed")
    difference = path[:, :, None, :] - path[:, None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=3))
    require(np.isfinite(distance).all(), "V120 path distance invalid")
    return distance


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    require(values.ndim == weights.ndim == 1 and len(values) == len(weights), "V120 weighted quantile shape invalid")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    require(float(cumulative[-1]) > 0.0, "V120 weighted quantile has zero mass")
    index = int(np.searchsorted(cumulative, quantile * float(cumulative[-1]), side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def fit_recurrence_thresholds(path: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    distances = pairwise_path_distances(path)
    upper = np.triu_indices(path.shape[1], k=1)
    pooled = distances[:, upper[0], upper[1]]
    row_weight = duplicate_unit_weights(frame)
    repeated_weight = np.repeat(row_weight, pooled.shape[1])
    flat = pooled.reshape(-1)
    thresholds = np.asarray([
        weighted_quantile(flat, repeated_weight, quantile) for quantile in RECURRENCE_QUANTILES
    ], dtype=float)
    require(np.isfinite(thresholds).all() and np.all(thresholds > 0.0), "V120 recurrence threshold invalid")
    require(np.all(np.diff(thresholds) >= 0.0), "V120 recurrence thresholds not ordered")
    return thresholds, {
        "train_only": True,
        "duplicate_balanced": True,
        "pooled_distance_n": int(flat.size),
        "unique_event_group_n": int(frame.event_group_id.nunique()),
        "quantiles": list(RECURRENCE_QUANTILES),
        "thresholds": thresholds.tolist(),
        "threshold_sha256": array_sha256(thresholds),
        "target_rows_used": False,
        "labels_used": False,
    }


def run_lengths(sequence: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in sequence.astype(bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def length_entropy(lengths: list[int]) -> float:
    retained = np.asarray([length for length in lengths if length >= MIN_LINE_LENGTH], dtype=int)
    if retained.size == 0:
        return 0.0
    _, counts = np.unique(retained, return_counts=True)
    probability = counts.astype(float) / float(counts.sum())
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-15))))
    return entropy / float(np.log(8.0))


def plot_statistics(recurrence: np.ndarray) -> np.ndarray:
    require(recurrence.shape == (8, 8), "V120 recurrence plot shape changed")
    recurrent_points = float(recurrence.sum())
    recurrence_rate = recurrent_points / 56.0
    diagonal_lengths: list[int] = []
    for offset in range(-6, 7):
        if offset == 0:
            continue
        diagonal_lengths.extend(run_lengths(np.diagonal(recurrence, offset=offset)))
    vertical_lengths: list[int] = []
    for column in range(8):
        vertical_lengths.extend(run_lengths(recurrence[:, column]))
    diagonal = [length for length in diagonal_lengths if length >= MIN_LINE_LENGTH]
    vertical = [length for length in vertical_lengths if length >= MIN_LINE_LENGTH]
    diagonal_points = float(sum(diagonal))
    vertical_points = float(sum(vertical))
    upper_i, upper_j = np.triu_indices(8, k=1)
    gap = (upper_j - upper_i)[recurrence[upper_i, upper_j]]
    gap_array = np.asarray(gap, float) / 7.0 if len(gap) else np.zeros(0, dtype=float)
    return np.asarray([
        recurrence_rate,
        diagonal_points / max(recurrent_points, 1.0),
        float(np.mean(diagonal)) / 7.0 if diagonal else 0.0,
        float(max(diagonal)) / 7.0 if diagonal else 0.0,
        length_entropy(diagonal_lengths),
        vertical_points / max(recurrent_points, 1.0),
        float(np.mean(vertical)) / 7.0 if vertical else 0.0,
        float(max(vertical)) / 7.0 if vertical else 0.0,
        length_entropy(vertical_lengths),
        float(np.mean(gap_array)) if gap_array.size else 0.0,
        float(np.std(gap_array)) if gap_array.size else 0.0,
        float(np.max(gap_array)) if gap_array.size else 0.0,
    ], dtype=float)


def recurrence_features(path: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    distances = pairwise_path_distances(path)
    result = np.zeros((len(path), RQA_FEATURE_DIMENSION), dtype=float)
    recurrence_rates: list[list[float]] = []
    for row in range(len(path)):
        row_rates: list[float] = []
        parts: list[np.ndarray] = []
        for threshold in thresholds:
            recurrence = distances[row] <= float(threshold)
            np.fill_diagonal(recurrence, False)
            statistic = plot_statistics(recurrence)
            parts.append(statistic)
            row_rates.append(float(statistic[0]))
        result[row] = np.concatenate(parts)
        recurrence_rates.append(row_rates)
    require(result.shape[1] == RQA_FEATURE_DIMENSION and np.isfinite(result).all(), "V120 RQA feature invalid")
    rate = np.asarray(recurrence_rates, float)
    return result, {
        "rows": len(result),
        "feature_dimension": result.shape[1],
        "thresholds": thresholds.tolist(),
        "recurrence_rate_mean_by_radius": rate.mean(axis=0).tolist(),
        "recurrence_rate_std_by_radius": rate.std(axis=0).tolist(),
        "feature_sha256": array_sha256(result),
        "symmetric_temporal_recurrence_plot": True,
        "diagonal_excluded": True,
        "line_statistics_order_invariant_within_each_line_family": True,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == RQA_FEATURE_DIMENSION, "V120 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        self.scale = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V120 scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -RQA_FEATURE_CLIP, RQA_FEATURE_CLIP)
        require(np.isfinite(result).all(), "V120 scaled RQA feature invalid")
        return result


def rqa_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V120 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_joint_path, train_path_audit = joint_path(train_state[:, :len(FEATURES)])
    target_joint_path, target_path_audit = joint_path(target_state[:, :len(FEATURES)])
    # V119 prepends an artificial origin and a normalized-time coordinate for
    # DMD.  RQA is predeclared on the eight observed causal horizons and eight
    # semantic state channels only, so both artificial axes are deterministically
    # removed before any train threshold or target representation is computed.
    train_path = train_joint_path[:, 1:, 1:]
    target_path = target_joint_path[:, 1:, 1:]
    require(train_path.shape[1:] == target_path.shape[1:] == (8, 8), "V120 observed path slice changed")
    thresholds, threshold_audit = fit_recurrence_thresholds(train_path, ordered)
    train_feature, train_rqa_audit = recurrence_features(train_path, thresholds)
    target_feature, target_rqa_audit = recurrence_features(target_path, thresholds)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=500,
        random_state=SEED, n_jobs=1,
    )
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V120 probability invalid")
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "static_path_spec": STATIC_SPEC,
        "train_path": train_path_audit,
        "target_path": target_path_audit,
        "threshold_fit": threshold_audit,
        "train_rqa": train_rqa_audit,
        "target_rqa": target_rqa_audit,
        "rqa_scaling": {
            "train_only": True,
            "clip": RQA_FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C,
            "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_scaling_thresholds_rqa_scale_or_head": False,
        "target_labels_used": False,
        "dtw_or_trajectory_prototype": False,
        "haar_modulus_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_matrix_log": False,
        "koopman_dmd_transition_or_spectrum": False,
        "feature_graph_or_persistence": False,
        "sinkhorn_or_optimal_transport": False,
        "bernstein_ulsif_density_ratio": False,
        "quantum_density_or_born_fidelity": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = rqa_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 RQA blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "recurrence_quantiles_line_statistics_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v120_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V120 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = rqa_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v120_model"] = selected["name"]
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
        evidence["rqa_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v120_model"] = selected["name"]
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
            f"[V120 RQA] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V120 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V120 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V120 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V120 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V120 bootstrap lost too many draws")

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
        "method": "paired market-fold-time block bootstrap over inner-locked V120 multichannel RQA policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V120 canonical mismatch")
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
    rqa_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_path_spec"]["channel_n"] == 8
        and audit[side]["static_path_spec"]["horizon_n"] == 8
        and audit[side]["threshold_fit"]["train_only"]
        and audit[side]["threshold_fit"]["duplicate_balanced"]
        and not audit[side]["threshold_fit"]["labels_used"]
        and audit[side]["train_rqa"]["feature_dimension"] == RQA_FEATURE_DIMENSION
        and audit[side]["target_rqa"]["symmetric_temporal_recurrence_plot"]
        and audit[side]["rqa_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_scaling_thresholds_rqa_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["dtw_or_trajectory_prototype"]
        and not audit[side]["haar_modulus_or_scattering"]
        and not audit[side]["rough_path_signature_or_levy_area"]
        and not audit[side]["increment_spd_or_matrix_log"]
        and not audit[side]["koopman_dmd_transition_or_spectrum"]
        and not audit[side]["feature_graph_or_persistence"]
        and not audit[side]["sinkhorn_or_optimal_transport"]
        and not audit[side]["bernstein_ulsif_density_ratio"]
        and not audit[side]["quantum_density_or_born_fidelity"]
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
        "multichannel_recurrence_quantification_contract_verified": rqa_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V120 fallback is not exact entire V69")
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
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V120 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V120 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V120 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V120",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V120_DIAGNOSTIC_MULTICHANNEL_RQA": evaluation["candidate_summary"],
                "V120_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact,
            "authority_audit": authority,
            "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V120",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "opened_dev_only": True,
            "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V120",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate,
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V120_MULTICHANNEL_RECURRENCE_QUANTIFICATION_REPORT.json": {
            "version": "V120",
            "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES,
            "nested_fold_audits": audits,
            "evaluation": compact,
            "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V120",
            "status": "MATCH",
            "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V120_MULTICHANNEL_RQA_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V120_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "output": str(out),
        "required_controller_reports": [
            "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V120 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V120 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_bound_full",
        "thread_policy": "authorized parent inherited",
        "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK",
            "hypothesis": HYPOTHESIS,
            "authority": authority,
            "runtime": runtime,
            "dev_rows": len(dev),
            "v69_rows": len(champion),
            "features": FEATURES,
            "static_spec": STATIC_SPEC,
            "architecture": "two train-only radii multichannel temporal recurrence plots, fixed RQA line statistics, and one fixed balanced ridge-logistic head",
            "causal_numeric_only": True,
            "target_row_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v120_compatible": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if args.smoke_test else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion,
            diagnostic,
            evidence,
            audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "runtime": runtime,
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
