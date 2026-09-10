"""V141 multichannel signed natural-visibility-graph direction challenger.

Every embargoed same-market train cut robust-scales the immutable 37 causal
numeric fields and interpolates eight semantic channels over eight pre-event
horizons in proper 120-minute-to-1-minute order.  For every row/channel, exact
threshold-free natural visibility is computed for the path and its negation on
the fixed real lookback coordinates.  Upper/lower nonadjacent edge indicators,
their original signed slopes, node degrees, global transitivity and endpoint
displacement form a fixed 824-dimensional peak/valley visibility representation.
Train-only robust feature scaling and one fixed duplicate/class-balanced
ridge-logistic head predict direction.

There is no environment meta-logit/random-effects aggregation, V69 offset or
residual correction, Tyler scatter, CVaR, learned graph edge/radius/threshold,
recurrence plot, outer-label tuning, text, ticker or external data. Inner-past
evidence selects exact V69 or fixed .25/.50 blends; outer labels stay
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V141_MULTICHANNEL_SIGNED_NATURAL_VISIBILITY_GRAPH_V1"
VERSION = 141
HYPOTHESIS = "MULTICHANNEL_SIGNED_NATURAL_VISIBILITY_GRAPH_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

CHANNEL_N = 8
HORIZON_N = 8
NONADJACENT_PAIR_N = (HORIZON_N - 1) * (HORIZON_N - 2) // 2
EDGE_BLOCKS_PER_CHANNEL = 4
DEGREE_FEATURE_N_PER_CHANNEL = 2 * HORIZON_N
GRAPH_SUMMARY_N_PER_CHANNEL = 3
FEATURES_PER_CHANNEL = (
    EDGE_BLOCKS_PER_CHANNEL * NONADJACENT_PAIR_N
    + DEGREE_FEATURE_N_PER_CHANNEL
    + GRAPH_SUMMARY_N_PER_CHANNEL
)
VISIBILITY_FEATURE_DIMENSION = CHANNEL_N * FEATURES_PER_CHANNEL
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 14101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
PAST_TO_RECENT_COORDINATES = np.asarray([-120.0, -60.0, -30.0, -15.0, -10.0, -5.0, -2.0, -1.0])
ARCHITECTURES = (
    {"name": "SIGNED_NATURAL_VISIBILITY_W0.25", "weight": 0.25},
    {"name": "SIGNED_NATURAL_VISIBILITY_W0.50", "weight": 0.50},
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


NONADJACENT_LEFT, NONADJACENT_RIGHT = np.asarray([
    (left, right)
    for left in range(HORIZON_N)
    for right in range(left + 2, HORIZON_N)
], dtype=int).T


def verify_static_spec() -> dict[str, Any]:
    require(len(scaffold.CHANNEL_SPEC) == CHANNEL_N and len(scaffold.HORIZON_GRID) == HORIZON_N, "V141 path geometry changed")
    require(NONADJACENT_PAIR_N == 21 and len(NONADJACENT_LEFT) == 21, "V141 visibility pair count changed")
    require(FEATURES_PER_CHANNEL == 103 and VISIBILITY_FEATURE_DIMENSION == 824, "V141 visibility dimension changed")
    require(np.all(np.diff(PAST_TO_RECENT_COORDINATES) > 0.0), "V141 time coordinates invalid")
    return {
        "channel_n": CHANNEL_N,
        "channel_names": list(scaffold.CHANNEL_NAMES),
        "horizon_n": HORIZON_N,
        "past_to_recent_coordinates_minutes": PAST_TO_RECENT_COORDINATES.tolist(),
        "upper_graph": "exact natural visibility of the robust channel path",
        "lower_graph": "exact natural visibility of the negated robust channel path",
        "visibility_rule": "all intermediate points strictly below endpoint line on fixed real lookback coordinates",
        "nonadjacent_pair_n": NONADJACENT_PAIR_N,
        "edge_blocks": ["upper_edge", "lower_edge", "upper_visible_original_slope", "lower_visible_original_slope"],
        "degree_features_per_channel": DEGREE_FEATURE_N_PER_CHANNEL,
        "graph_summaries": ["upper_transitivity", "lower_transitivity", "endpoint_displacement"],
        "features_per_channel": FEATURES_PER_CHANNEL,
        "feature_dimension": VISIBILITY_FEATURE_DIMENSION,
        "learned_edge_radius_threshold_or_graph_topology": False,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending and atomic IO only; no V119 output is read",
    }
    audit["v141_access_contract"] = {
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


def observed_past_to_recent_path(robust_numeric: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    joint, inherited_audit = joint_path(robust_numeric)
    path = joint[:, 1:, 1:][:, ::-1, :]
    require(path.shape[1:] == (HORIZON_N, CHANNEL_N), "V141 observed path slice changed")
    return path, {
        "rows": len(path),
        "horizons": path.shape[1],
        "channels": path.shape[2],
        "path_order": "120m,60m,30m,15m,10m,5m,2m,1m",
        "past_to_recent": True,
        "observed_path_sha256": array_sha256(path),
        "artificial_origin_removed": True,
        "normalized_time_coordinate_removed": True,
        "inherited_interpolation_audit": inherited_audit,
    }


def natural_visibility_adjacency(channel_path: np.ndarray) -> np.ndarray:
    require(channel_path.ndim == 3 and channel_path.shape[1:] == (CHANNEL_N, HORIZON_N), "V141 channel path shape invalid")
    adjacency = np.zeros((len(channel_path), CHANNEL_N, HORIZON_N, HORIZON_N), dtype=bool)
    for left in range(HORIZON_N):
        for right in range(left + 1, HORIZON_N):
            visible = np.ones((len(channel_path), CHANNEL_N), dtype=bool)
            if right > left + 1:
                fraction = (
                    PAST_TO_RECENT_COORDINATES[left + 1:right] - PAST_TO_RECENT_COORDINATES[left]
                ) / (
                    PAST_TO_RECENT_COORDINATES[right] - PAST_TO_RECENT_COORDINATES[left]
                )
                endpoint_line = (
                    channel_path[:, :, left, None] * (1.0 - fraction[None, None, :])
                    + channel_path[:, :, right, None] * fraction[None, None, :]
                )
                visible = np.all(channel_path[:, :, left + 1:right] < endpoint_line, axis=2)
            adjacency[:, :, left, right] = visible
            adjacency[:, :, right, left] = visible
    require(
        np.array_equal(adjacency, np.swapaxes(adjacency, 2, 3))
        and not np.any(np.diagonal(adjacency, axis1=2, axis2=3)),
        "V141 visibility adjacency algebra failed",
    )
    return adjacency


def graph_transitivity(adjacency: np.ndarray) -> np.ndarray:
    matrix = adjacency.astype(float)
    squared = matrix @ matrix
    triangles = np.sum(squared * matrix, axis=(2, 3)) / 6.0
    degree = np.sum(matrix, axis=3)
    wedges = np.sum(degree * (degree - 1.0), axis=2) / 2.0
    return np.divide(3.0 * triangles, wedges, out=np.zeros_like(triangles), where=wedges > 0.0)


def signed_visibility_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V141 visibility path shape invalid")
    channel_path = np.transpose(path, (0, 2, 1))
    upper = natural_visibility_adjacency(channel_path)
    lower = natural_visibility_adjacency(-channel_path)
    upper_edge = upper[:, :, NONADJACENT_LEFT, NONADJACENT_RIGHT].astype(float)
    lower_edge = lower[:, :, NONADJACENT_LEFT, NONADJACENT_RIGHT].astype(float)
    time_delta = PAST_TO_RECENT_COORDINATES[NONADJACENT_RIGHT] - PAST_TO_RECENT_COORDINATES[NONADJACENT_LEFT]
    slope = (
        channel_path[:, :, NONADJACENT_RIGHT] - channel_path[:, :, NONADJACENT_LEFT]
    ) / time_delta[None, None, :]
    upper_degree = np.sum(upper, axis=3).astype(float)
    lower_degree = np.sum(lower, axis=3).astype(float)
    upper_transitivity = graph_transitivity(upper)[:, :, None]
    lower_transitivity = graph_transitivity(lower)[:, :, None]
    endpoint_displacement = (channel_path[:, :, -1] - channel_path[:, :, 0])[:, :, None]
    per_channel = np.concatenate([
        upper_edge,
        lower_edge,
        upper_edge * slope,
        lower_edge * slope,
        upper_degree,
        lower_degree,
        upper_transitivity,
        lower_transitivity,
        endpoint_displacement,
    ], axis=2)
    feature = per_channel.reshape(len(path), -1)
    adjacent = np.arange(HORIZON_N - 1)
    adjacent_contract = bool(
        np.all(upper[:, :, adjacent, adjacent + 1])
        and np.all(lower[:, :, adjacent, adjacent + 1])
    )
    require(
        feature.shape[1] == VISIBILITY_FEATURE_DIMENSION
        and np.isfinite(feature).all() and adjacent_contract,
        "V141 visibility feature contract failed",
    )
    return feature, {
        "rows": len(feature),
        "feature_dimension": feature.shape[1],
        "features_per_channel": FEATURES_PER_CHANNEL,
        "upper_nonadjacent_edge_rate": float(np.mean(upper_edge)),
        "lower_nonadjacent_edge_rate": float(np.mean(lower_edge)),
        "upper_transitivity_mean": float(np.mean(upper_transitivity)),
        "lower_transitivity_mean": float(np.mean(lower_transitivity)),
        "adjacent_edges_always_visible": adjacent_contract,
        "upper_symmetry_exact": bool(np.array_equal(upper, np.swapaxes(upper, 2, 3))),
        "lower_symmetry_exact": bool(np.array_equal(lower, np.swapaxes(lower, 2, 3))),
        "feature_sha256": array_sha256(feature),
        "label_independent_exact_transform": True,
        "learned_edge_radius_threshold_or_graph_topology": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == VISIBILITY_FEATURE_DIMENSION, "V141 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        self.scale = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V141 scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V141 scaled visibility features invalid")
        return result


def visibility_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V141 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_visibility_audit = signed_visibility_features(train_path)
    target_feature, target_visibility_audit = signed_visibility_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V141 probability invalid")
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit,
        "target_path": target_path_audit,
        "train_visibility": train_visibility_audit,
        "target_visibility": target_visibility_audit,
        "visibility_scaling": {
            "train_only": True,
            "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C,
            "solver": "liblinear",
            "iterations": int(head.n_iter_[0]),
            "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_visibility_scale_or_head": False,
        "target_labels_used": False,
        "learned_edge_radius_threshold_or_graph_topology": False,
        "source_market_time_random_effects_meta_logit": False,
        "v69_offset_or_residual_correction": False,
        "tyler_spatial_scatter_or_fisher_direction": False,
        "recurrence_radius_plot_or_line_statistics": False,
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
    challenger, model_audit = visibility_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 causal signed natural visibility graph blend; fixed 2*AUC+BA with BA/net eligibility",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "visibility_terms_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v141_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V141 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = visibility_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v141_model"] = selected["name"]
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
        evidence["visibility_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v141_model"] = selected["name"]
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
            f"[V141 VISIBILITY] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V141 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V141 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V141 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V141 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V141 bootstrap lost too many draws")

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
        "method": "paired market-fold-time block bootstrap over inner-locked V141 signed natural-visibility-graph policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V141 canonical mismatch")
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
    visibility_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["feature_dimension"] == VISIBILITY_FEATURE_DIMENSION
        and not audit[side]["static_spec"]["learned_edge_radius_threshold_or_graph_topology"]
        and audit[side]["train_visibility"]["feature_dimension"] == VISIBILITY_FEATURE_DIMENSION
        and audit[side]["train_visibility"]["label_independent_exact_transform"]
        and audit[side]["train_visibility"]["adjacent_edges_always_visible"]
        and audit[side]["target_visibility"]["adjacent_edges_always_visible"]
        and audit[side]["target_visibility"]["upper_symmetry_exact"]
        and audit[side]["target_visibility"]["lower_symmetry_exact"]
        and audit[side]["visibility_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_visibility_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["learned_edge_radius_threshold_or_graph_topology"]
        and not audit[side]["source_market_time_random_effects_meta_logit"]
        and not audit[side]["v69_offset_or_residual_correction"]
        and not audit[side]["tyler_spatial_scatter_or_fisher_direction"]
        and not audit[side]["recurrence_radius_plot_or_line_statistics"]
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
        "multichannel_signed_natural_visibility_graph_contract_verified": visibility_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V141 fallback is not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V141 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V141 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V141 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V141", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V141_DIAGNOSTIC_SIGNED_NATURAL_VISIBILITY": evaluation["candidate_summary"],
                "V141_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V141", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V141", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V141_MULTICHANNEL_SIGNED_NATURAL_VISIBILITY_GRAPH_REPORT.json": {
            "version": "V141", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V141", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V141_MULTICHANNEL_SIGNED_NATURAL_VISIBILITY_GRAPH_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V141_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V141 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V141 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V141 controller full-run output mismatch")
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
            "architecture": "exact upper/lower natural-visibility graphs on eight causal channel paths; fixed edge/slope/degree/transitivity representation plus balanced ridge logistic",
            "causal_numeric_only": True, "target_row_labels_used": False,
            "learned_graph_threshold_radius_regularization_or_post_outer_tuning": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v141_compatible": True,
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
            require(args.smoke_market == "KR", "V141 support probe is reserved for KR:2")
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

