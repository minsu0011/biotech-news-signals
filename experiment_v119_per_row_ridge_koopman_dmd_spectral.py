"""V119 per-row ridge Koopman/DMD spectral direction challenger.

Each embargoed same-market train cut robust-scales only the 37 causal numeric
state fields.  The immutable label-free V113 horizon interpolation maps each
row to eight semantic channels observed on eight increasing lookback horizons.
For every row independently, chronological adjacent-horizon snapshots define
a fixed-rank ridge dynamic-mode-decomposition operator.  Its invariant
singular spectrum, complex transition spectrum, reconstruction error and
nonnormality form a 36-dimensional dynamic representation.  A train-only
robust representation scale and one fixed duplicate/class-balanced ridge
logistic head produce direction probability.

This is not a rough-path signature, increment SPD covariance, DTW prototype,
Haar scattering, dynamic cross-event Bayesian coefficient filter, Sinkhorn
transport, Bernstein uLSIF density ratio, or quantum density/Born-fidelity
model.  Atomic V69 inner-past labels alone select exact V69 or a fixed .25/.50
logit blend.  Outer labels remain evaluation-only under a strict 35-minute
embargo.  V69 confidence/high-confidence are frozen, and material failure
restores the exact entire V69 DataFrame.

Audit and US:2 smoke are CPU30-31/two-thread/no-write/no-GPU.  A future
authorized no-argument MARKET_BIO_VERSION_OUTPUT invocation performs all six
folds and writes the three required controller reports plus the current
material-gate/nested-bootstrap contract.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v102_ordered_horizon_fused_lasso_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V119_PER_ROW_RIDGE_KOOPMAN_DMD_SPECTRAL_V1"
VERSION = 119
HYPOTHESIS = "PER_ROW_RIDGE_KOOPMAN_DMD_SPECTRAL_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v102_ordered_horizon_fused_lasso_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "d398c651102ff2e47e747d277aca5ee24718f0ae29013b3fb02f0bfc07fba3b2"

HORIZON_GRID = np.asarray([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0], dtype=float)
CHANNEL_SPEC: dict[str, tuple[tuple[float, str], ...]] = {
    "RETURN": tuple(zip(HORIZON_GRID.tolist(), (
        "pre_ret_1", "pre_ret_2", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_30", "pre_ret_60", "pre_ret_120",
    ))),
    "VOLATILITY": ((5.0, "pre_vol_5"), (10.0, "pre_vol_10"), (30.0, "pre_vol_30"), (60.0, "pre_vol_60")),
    "VOLUME": ((1.0, "volume_ratio_1_30"), (2.0, "volume_ratio_2_30"), (5.0, "volume_ratio_5_30"), (10.0, "volume_ratio_10_60")),
    "RANGE": ((5.0, "range_5m"), (10.0, "range_10m"), (30.0, "range_30m"), (60.0, "range_60m")),
    "CLOSE_POSITION": ((10.0, "close_position_10"), (30.0, "close_position_30"), (60.0, "close_position_60")),
    "BENCHMARK": ((2.0, "benchmark_ret_2"), (5.0, "benchmark_ret_5"), (15.0, "benchmark_ret_15"), (30.0, "benchmark_ret_30"), (60.0, "benchmark_ret_60")),
    "TREND": ((30.0, "trend_slope_30"), (60.0, "trend_slope_60")),
    "UP_FRACTION": ((10.0, "up_fraction_10"), (30.0, "up_fraction_30")),
}
CHANNEL_NAMES = tuple(CHANNEL_SPEC)

DMD_RANK = 4
DMD_RIDGE = 0.10
DMD_FEATURE_DIMENSION = 36
DMD_FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 11901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "RIDGE_KOOPMAN_DMD_SPECTRAL_W0.25", "weight": 0.25},
    {"name": "RIDGE_KOOPMAN_DMD_SPECTRAL_W0.50", "weight": 0.50},
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
numeric = scaffold.scaffold


def interpolation_weights(source_horizon: np.ndarray) -> np.ndarray:
    source = np.log(np.asarray(source_horizon, float))
    target = np.log(HORIZON_GRID)
    weights = np.zeros((len(HORIZON_GRID), len(source)), dtype=float)
    for row, value in enumerate(target):
        if value <= source[0]:
            weights[row, 0] = 1.0
        elif value >= source[-1]:
            weights[row, -1] = 1.0
        else:
            right = int(np.searchsorted(source, value, side="right"))
            left = right - 1
            fraction = (value - source[left]) / (source[right] - source[left])
            weights[row, left] = 1.0 - fraction
            weights[row, right] = fraction
    require(float(np.max(np.abs(weights.sum(axis=1) - 1.0))) <= 1e-12, "V119 interpolation weights invalid")
    return weights


INTERPOLATION = {
    channel: interpolation_weights(np.asarray([horizon for horizon, _ in items], float))
    for channel, items in CHANNEL_SPEC.items()
}
FEATURE_INDEX = {name: position for position, name in enumerate(FEATURES)}


def verify_static_spec() -> dict[str, Any]:
    all_columns = [column for items in CHANNEL_SPEC.values() for _, column in items]
    require(set(all_columns).issubset(set(FEATURES)), "V119 horizon feature outside causal authority")
    require(len(HORIZON_GRID) == 8 and len(CHANNEL_NAMES) == 8, "V119 horizon/channel grid changed")
    payload = "\n".join(
        f"{channel}|" + ",".join(f"{horizon:g}:{column}" for horizon, column in items)
        for channel, items in CHANNEL_SPEC.items()
    )
    return {
        "causal_authority_feature_n": len(FEATURES),
        "path_input_unique_feature_n": len(set(all_columns)),
        "channel_n": len(CHANNEL_NAMES), "horizon_n": len(HORIZON_GRID),
        "path_dimension_with_time": 9,
        "channel_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "label_independent_path_spec": True,
    }


STATIC_SPEC = verify_static_spec()


def joint_path(robust_numeric: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(robust_numeric.ndim == 2 and robust_numeric.shape[1] == len(FEATURES), "V119 robust state shape invalid")
    channels: list[np.ndarray] = []
    for channel, items in CHANNEL_SPEC.items():
        positions = np.asarray([FEATURE_INDEX[column] for _, column in items], dtype=int)
        channels.append(robust_numeric[:, positions] @ INTERPOLATION[channel].T)
    state = np.stack(channels, axis=2)
    normalized_time = (np.log(HORIZON_GRID) + 1.0) / (np.log(HORIZON_GRID[-1]) + 1.0)
    time = np.broadcast_to(normalized_time[None, :, None], (len(state), len(HORIZON_GRID), 1))
    points = np.concatenate([time, state], axis=2)
    path = np.concatenate([np.zeros((len(state), 1, 9), dtype=float), points], axis=1)
    require(path.shape[1:] == (9, 9) and np.isfinite(path).all(), "V119 joint path invalid")
    return path, {
        "rows": len(path), "points_including_origin": path.shape[1],
        "path_dimension": path.shape[2], "normalized_time": normalized_time.tolist(),
        "path_sha256": array_sha256(path),
    }


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V102 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust state, metrics, canonical gate, blending, and atomic IO only; no V102 output is read",
    }
    audit["v119_access_contract"] = {
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


class RobustDmdScale:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustDmdScale":
        require(matrix.ndim == 2 and matrix.shape[1] == DMD_FEATURE_DIMENSION, "V119 DMD design dimension changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        standard = np.std(matrix, axis=0)
        self.scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V119 DMD scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -DMD_FEATURE_CLIP, DMD_FEATURE_CLIP)
        require(np.isfinite(result).all(), "V119 scaled DMD design invalid")
        return result


def ridge_dmd_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Batched fixed-rank DMD features, invariant to eigenvalue ordering."""
    require(path.ndim == 3 and path.shape[1:] == (9, 9), "V119 joint path shape changed")
    state = path[:, 1:, 1:]
    left = np.transpose(state[:, :-1, :], (0, 2, 1))
    right = np.transpose(state[:, 1:, :], (0, 2, 1))
    u, singular, vh = np.linalg.svd(left, full_matrices=False)
    u_rank = u[:, :, :DMD_RANK]
    vh_rank = vh[:, :DMD_RANK, :]
    v_rank = np.transpose(vh_rank, (0, 2, 1))
    gain = singular[:, :DMD_RANK] / (singular[:, :DMD_RANK] ** 2 + DMD_RIDGE)

    reduced = np.matmul(np.matmul(np.transpose(u_rank, (0, 2, 1)), right), v_rank)
    reduced = reduced * gain[:, None, :]
    lifted = np.matmul(right, v_rank) * gain[:, None, :]
    operator = np.matmul(lifted, np.transpose(u_rank, (0, 2, 1)))
    reconstruction = np.matmul(operator, left)
    eigenvalue = np.linalg.eigvals(reduced)

    radius = np.abs(eigenvalue)
    angle = np.abs(np.angle(eigenvalue)) / np.pi
    real = np.real(eigenvalue)
    imaginary = np.abs(np.imag(eigenvalue))
    singular_top = singular[:, :DMD_RANK]
    singular_energy = singular_top ** 2 / np.maximum(np.sum(singular ** 2, axis=1, keepdims=True), 1e-12)

    residual = np.linalg.norm(right - reconstruction, axis=(1, 2)) / np.maximum(np.linalg.norm(right, axis=(1, 2)), 1e-12)
    transpose = np.transpose(operator, (0, 2, 1))
    gram_left = np.matmul(transpose, operator)
    gram_right = np.matmul(operator, transpose)
    nonnormality = np.linalg.norm(gram_left - gram_right, axis=(1, 2)) / np.maximum(
        np.linalg.norm(operator, axis=(1, 2)) ** 2, 1e-12,
    )
    _, logabsdet = np.linalg.slogdet(reduced)
    first_norm = np.linalg.norm(state[:, 0, :], axis=1)
    last_norm = np.linalg.norm(state[:, -1, :], axis=1)
    summary = np.column_stack([
        residual,
        np.max(radius, axis=1),
        np.mean(radius, axis=1),
        np.mean(radius < 1.0, axis=1),
        np.mean(imaginary > 1e-8, axis=1),
        nonnormality,
        np.log1p(np.linalg.norm(operator, axis=(1, 2))),
        np.real(np.trace(reduced, axis1=1, axis2=2)) / DMD_RANK,
        np.clip(logabsdet / DMD_RANK, -20.0, 20.0),
        np.sum(singular_energy, axis=1),
        np.linalg.norm(state[:, -1, :] - state[:, 0, :], axis=1) / np.maximum(first_norm + last_norm, 1e-12),
        np.log(np.maximum(last_norm, 1e-8) / np.maximum(first_norm, 1e-8)),
    ])
    feature = np.column_stack([
        np.log1p(singular_top),
        singular_energy,
        np.sort(radius, axis=1)[:, ::-1],
        np.sort(angle, axis=1)[:, ::-1],
        np.sort(real, axis=1)[:, ::-1],
        np.sort(imaginary, axis=1)[:, ::-1],
        summary,
    ])
    require(feature.shape == (len(path), DMD_FEATURE_DIMENSION), "V119 DMD feature count changed")
    require(np.isfinite(feature).all(), "V119 DMD feature non-finite")
    return feature, {
        "rows": len(feature),
        "rank": DMD_RANK,
        "ridge": DMD_RIDGE,
        "feature_dimension": feature.shape[1],
        "chronological_adjacent_horizon_transition": True,
        "per_row_operator_only": True,
        "eigenvalue_order_invariant_sorting": True,
        "reconstruction_error_mean": float(np.mean(residual)),
        "spectral_radius_mean": float(np.mean(np.max(radius, axis=1))),
        "stable_eigenvalue_fraction_mean": float(np.mean(radius < 1.0)),
        "oscillatory_eigenvalue_fraction_mean": float(np.mean(imaginary > 1e-8)),
        "nonnormality_mean": float(np.mean(nonnormality)),
        "feature_sha256": array_sha256(feature),
    }


def koopman_dmd_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V119 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = joint_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = joint_path(target_state[:, :len(FEATURES)])
    train_feature, train_dmd_audit = ridge_dmd_features(train_path)
    target_feature, target_dmd_audit = ridge_dmd_features(target_path)
    feature_scale = RobustDmdScale().fit(train_feature)
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
    require(np.isfinite(probability).all(), "V119 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_path_spec": STATIC_SPEC,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_dmd": train_dmd_audit, "target_dmd": target_dmd_audit,
        "dmd_scaling": {
            "train_only": True, "clip": DMD_FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C, "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_scaling_path_spec_dmd_scale_or_head": False,
        "target_labels_used": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_matrix_log": False,
        "haar_modulus_or_scattering": False,
        "dtw_or_trajectory_prototype": False,
        "cross_event_dynamic_bayes_filter": False,
        "sinkhorn_or_optimal_transport": False,
        "bernstein_ulsif_density_ratio": False,
        "quantum_density_or_born_fidelity": False,
        "graph_kernel_tree_or_neural": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = koopman_dmd_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 per-row ridge-Koopman-DMD spectral blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "dmd_rank_ridge_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v119_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V119 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = koopman_dmd_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v119_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["dmd_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v119_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V119 KOOPMAN-DMD] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V119 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V119 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V119 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V119 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V119 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V119 per-row ridge-Koopman-DMD spectral policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V119 canonical mismatch")
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
    dmd_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_path_spec"]["channel_n"] == 8
        and audit[side]["static_path_spec"]["horizon_n"] == 8
        and audit[side]["train_dmd"]["rank"] == DMD_RANK
        and audit[side]["train_dmd"]["feature_dimension"] == DMD_FEATURE_DIMENSION
        and audit[side]["train_dmd"]["chronological_adjacent_horizon_transition"]
        and audit[side]["target_dmd"]["per_row_operator_only"]
        and audit[side]["dmd_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_scaling_path_spec_dmd_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["rough_path_signature_or_levy_area"]
        and not audit[side]["increment_spd_or_matrix_log"]
        and not audit[side]["haar_modulus_or_scattering"]
        and not audit[side]["dtw_or_trajectory_prototype"]
        and not audit[side]["cross_event_dynamic_bayes_filter"]
        and not audit[side]["sinkhorn_or_optimal_transport"]
        and not audit[side]["bernstein_ulsif_density_ratio"]
        and not audit[side]["quantum_density_or_born_fidelity"]
        and not audit[side]["graph_kernel_tree_or_neural"]
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
        "per_row_ridge_koopman_dmd_spectral_contract_verified": dmd_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V119 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "candidate_native_robustness_bootstrap": native_bootstrap,
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
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V119 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V119 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V119 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V119", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V119_DIAGNOSTIC_RIDGE_KOOPMAN_DMD": evaluation["candidate_summary"],
                "V119_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V119", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V119", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V119_PER_ROW_RIDGE_KOOPMAN_DMD_SPECTRAL_REPORT.json": {
            "version": "V119", "hypothesis": HYPOTHESIS,
            "static_path_spec": STATIC_SPEC,
            "dmd_rank": DMD_RANK, "dmd_ridge": DMD_RIDGE,
            "dmd_feature_dimension": DMD_FEATURE_DIMENSION,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V119", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V119_KOOPMAN_DMD_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V119_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V119 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V119 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_or_explicit_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_path_spec": STATIC_SPEC,
            "architecture": "per-row fixed-rank ridge Koopman/DMD of adjacent causal horizons; invariant transition spectrum/residual/nonnormality; train-only robust spectral scale; fixed ridge head",
            "dmd_rank": DMD_RANK, "dmd_ridge": DMD_RIDGE,
            "dmd_feature_dimension": DMD_FEATURE_DIMENSION,
            "causal_numeric_only": True, "target_labels_used": False,
            "rough_path_signature_or_levy_area": False,
            "increment_spd_or_matrix_log": False,
            "cross_event_dynamic_bayes_filter": False,
            "sinkhorn_or_optimal_transport": False,
            "bernstein_ulsif_density_ratio": False,
            "quantum_density_or_born_fidelity": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "exact_output_binding": True, "explicit_conflict_fail_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness": True, "source_transfer": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if args.smoke_test else None):
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
