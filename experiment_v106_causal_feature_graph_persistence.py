"""V106 causal-feature-graph lower-star persistence direction challenger.

A fixed label-independent graph connects the 37 causal numeric coordinates by
adjacent lookback horizons, same-horizon cross-channel relationships, and a
small set of predeclared standalone links.  After train-cut robust scaling,
each event induces a scalar field on that graph.  Deterministic union-find
computes exact zero-dimensional lower-star persistence for the field and its
negation.  Fixed top-lifetime and persistence-distribution summaries form a
topological signature, which is train-cut scaled and passed to one fixed-L2,
duplicate/class-balanced logistic head.

This is not the historical-event graph/message passing of V80, an ordered raw
coefficient or fused-lasso path, a wavelet, class subspace/SVD, FastICA latent
component, simplex/Dirichlet likelihood, covariance density, depth, neighbor,
prototype, pairwise energy score, or empirical-likelihood dual.  Earlier
same-market committed V69 OOF alone chooses exact V69 or fixed 0.25/0.50 logit
blends.  Outer labels are evaluation-only under a strict 35-minute embargo;
V69 confidence/high_conf are frozen; material failure returns the exact entire
atomic V69 frame after controller recomputation of all canonical 14 gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "2")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V106_CAUSAL_FEATURE_GRAPH_PERSISTENCE_V1"
VERSION = 106
HYPOTHESIS = "CAUSAL_FEATURE_GRAPH_LOWER_STAR_PERSISTENCE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

TOP_LIFETIMES = 12
PERSISTENCE_EPSILON = 1e-12
SIGNATURE_CLIP = 6.0
SEED = 10601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
COST = scaffold.COST

ORDERED_CHANNELS = {
    "RETURN": (
        "pre_ret_1", "pre_ret_2", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_30", "pre_ret_60", "pre_ret_120",
    ),
    "VOLATILITY": ("pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60"),
    "VOLUME": (
        "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
        "volume_ratio_10_60",
    ),
    "RANGE": ("range_5m", "range_10m", "range_30m", "range_60m"),
    "CLOSE_POSITION": ("close_position_10", "close_position_30", "close_position_60"),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
    "TREND": ("trend_slope_30", "trend_slope_60"),
    "UP_FRACTION": ("up_fraction_10", "up_fraction_30"),
}

STANDALONE_EDGES = (
    ("intraday_ret_open", "pre_ret_30"),
    ("return_autocorr_30", "pre_ret_30"),
    ("range_30m", "pre_ret_30"),
    ("vwap_distance_30", "close_position_30"),
    ("minutes_from_open", "minutes_to_close"),
    ("minutes_from_open", "intraday_ret_open"),
)

ARCHITECTURES = (
    {"name": "FEATURE_GRAPH_PERSISTENCE_W0.25", "weight": 0.25},
    {"name": "FEATURE_GRAPH_PERSISTENCE_W0.50", "weight": 0.50},
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
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metrics, blending, robust scaling, logistic fitting, and IO only; no V85 output is read",
    }
    audit["v106_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def trailing_horizon(name: str) -> int | None:
    token = name.rsplit("_", 1)[-1]
    return int(token) if token.isdigit() else None


def feature_graph() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...], dict[str, Any]]:
    index = {name: position for position, name in enumerate(FEATURES)}
    edges: set[tuple[int, int]] = set()

    def add(left: str, right: str) -> None:
        require(left in index and right in index and left != right, "V106 graph edge feature invalid")
        edge = tuple(sorted((index[left], index[right])))
        edges.add(edge)

    for names in ORDERED_CHANNELS.values():
        for left, right in zip(names[:-1], names[1:]):
            add(left, right)
    by_horizon: dict[int, list[str]] = {}
    for name in FEATURES:
        horizon = trailing_horizon(name)
        if horizon is not None and not name.startswith("minutes_"):
            by_horizon.setdefault(horizon, []).append(name)
    for names in by_horizon.values():
        for left_position, left in enumerate(names):
            for right in names[left_position + 1:]:
                add(left, right)
    for left, right in STANDALONE_EDGES:
        add(left, right)

    ordered_edges = tuple(sorted(edges))
    adjacency: list[list[int]] = [[] for _ in FEATURES]
    for left, right in ordered_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    require(len(seen) == len(FEATURES), "V106 causal feature graph is disconnected")
    adjacency_tuple = tuple(tuple(sorted(neighbors)) for neighbors in adjacency)
    graph_array = np.asarray(ordered_edges, dtype=np.int16)
    return ordered_edges, adjacency_tuple, {
        "vertex_n": len(FEATURES), "edge_n": len(ordered_edges),
        "connected_components": 1,
        "construction": "adjacent horizon edges + same-horizon cross-channel cliques + fixed standalone links",
        "label_independent": True,
        "edge_sha256": array_sha256(graph_array),
        "degree_min": int(min(map(len, adjacency_tuple))),
        "degree_median": float(np.median([len(item) for item in adjacency_tuple])),
        "degree_max": int(max(map(len, adjacency_tuple))),
    }


GRAPH_EDGES, GRAPH_ADJACENCY, GRAPH_AUDIT = feature_graph()


def zero_dimensional_lifetimes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    require(values.shape == (len(FEATURES),) and np.isfinite(values).all(), "V106 scalar field invalid")
    parent = np.full(len(values), -1, dtype=int)
    birth = np.zeros(len(values), dtype=float)
    active = np.zeros(len(values), dtype=bool)
    lifetimes: list[float] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    order = np.lexsort((np.arange(len(values)), values))
    for vertex in order:
        vertex = int(vertex)
        active[vertex] = True
        parent[vertex] = vertex
        birth[vertex] = values[vertex]
        for neighbor in GRAPH_ADJACENCY[vertex]:
            if not active[neighbor]:
                continue
            left_root = find(vertex)
            right_root = find(neighbor)
            if left_root == right_root:
                continue
            left_key = (birth[left_root], left_root)
            right_key = (birth[right_root], right_root)
            elder, younger = (left_root, right_root) if left_key <= right_key else (right_root, left_root)
            death = max(values[vertex], values[neighbor])
            lifetimes.append(max(float(death - birth[younger]), 0.0))
            parent[younger] = elder
    require(len(lifetimes) == len(FEATURES) - 1, "V106 persistence pairing count changed")
    result = np.asarray(lifetimes, dtype=float)
    require(np.isfinite(result).all() and np.all(result >= 0.0), "V106 persistence lifetime invalid")
    return result


def lifetime_summary(lifetimes: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(lifetimes, float))[::-1]
    top = np.zeros(TOP_LIFETIMES, dtype=float)
    top[:min(TOP_LIFETIMES, len(ordered))] = ordered[:TOP_LIFETIMES]
    positive = ordered[ordered > PERSISTENCE_EPSILON]
    total = float(positive.sum())
    if len(positive) > 1 and total > 0.0:
        mass = positive / total
        entropy = float(-np.sum(mass * np.log(mass)) / math.log(len(positive)))
    else:
        entropy = 0.0
    statistics = np.asarray([
        total,
        float(ordered[0]) if len(ordered) else 0.0,
        float(ordered.mean()) if len(ordered) else 0.0,
        float(ordered.std()) if len(ordered) else 0.0,
        float(len(positive) / max(len(ordered), 1)),
        entropy,
    ], dtype=float)
    result = np.concatenate([top, statistics])
    require(np.isfinite(result).all(), "V106 persistence summary invalid")
    return result


def persistence_signature(matrix: np.ndarray) -> np.ndarray:
    signatures = []
    for row in np.asarray(matrix, float):
        lower = lifetime_summary(zero_dimensional_lifetimes(row))
        upper = lifetime_summary(zero_dimensional_lifetimes(-row))
        signatures.append(np.concatenate([lower, upper]))
    result = np.asarray(signatures, dtype=float)
    expected_columns = 2 * (TOP_LIFETIMES + 6)
    require(result.shape == (len(matrix), expected_columns), "V106 signature shape changed")
    require(np.isfinite(result).all(), "V106 signature non-finite")
    return result


class RobustSignatureState:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustSignatureState":
        self.median = np.median(matrix, axis=0)
        lower = np.quantile(matrix, 0.25, axis=0)
        upper = np.quantile(matrix, 0.75, axis=0)
        spread = upper - lower
        fallback = np.std(matrix, axis=0)
        self.scale = np.where(spread > 1e-9, spread, np.where(fallback > 1e-9, fallback, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V106 signature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -SIGNATURE_CLIP, SIGNATURE_CLIP)
        require(np.isfinite(result).all(), "V106 scaled signature invalid")
        return result


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= float(weights.sum())
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "V106 duplicate weights invalid")
    return weights


def feature_graph_persistence_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V106 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V106 market fit mismatch")
    state = scaffold.RobustNumericState().fit(train)
    train_state = state.transform(train)
    target_state = state.transform(target_frame)
    train_raw_signature = persistence_signature(train_state)
    target_raw_signature = persistence_signature(target_state)
    signature_state = RobustSignatureState().fit(train_raw_signature)
    train_signature = signature_state.transform(train_raw_signature)
    target_signature = signature_state.transform(target_raw_signature)
    coefficient, logistic_audit = scaffold.fit_weighted_logistic(
        train_signature, train.y.to_numpy(int), duplicate_weights(train),
    )
    probability = expit(coefficient[0] + target_signature @ coefficient[1:])
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V106 persistence probability invalid")
    quantiles = (0.10, 0.50, 0.90)
    positive_counts = np.count_nonzero(
        target_raw_signature[:, :TOP_LIFETIMES] > PERSISTENCE_EPSILON, axis=1,
    )
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES),
        "persistence_signature_n": train_signature.shape[1],
        "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
        "source_feature": False, "ticker_feature": False,
        "architecture": "fixed causal-feature graph exact 0D lower-star persistence plus one L2 logistic head",
        "feature_graph": GRAPH_AUDIT,
        "filtrations": ["lower_star_state", "lower_star_negated_state"],
        "homology_dimension": 0,
        "finite_pair_count_per_filtration": len(FEATURES) - 1,
        "top_lifetimes_per_filtration": TOP_LIFETIMES,
        "summary_statistics_per_filtration": [
            "total_persistence", "max_lifetime", "mean_lifetime", "std_lifetime",
            "positive_lifetime_fraction", "persistence_entropy",
        ],
        "union_find_exact": True, "graph_fit_with_labels": False,
        "topology_transform_fit_with_labels": False,
        "signature_scaling_fit_on_train_only": True,
        "event_group_duplicate_balanced_head": True,
        "logistic_head": logistic_audit,
        "raw_causal_direction_coefficient_n": 0,
        "signature_head_coefficient_n": len(coefficient),
        "event_graph_or_message_passing": False,
        "ordered_or_fused_lasso": False,
        "wavelet_or_scattering": False,
        "svd_subspace_or_fastica": False,
        "simplex_or_dirichlet": False,
        "covariance_density_or_copula": False,
        "projection_depth_neighbor_or_prototype": False,
        "empirical_likelihood_or_pairwise_energy": False,
        "target_rows_used_for_state_or_signature_scaling_or_head_fit": False,
        "target_rows_used_only_for_deterministic_topology_transform": True,
        "target_labels_used": False,
        "lower_top_positive_count_quantiles": {
            str(q): float(np.quantile(positive_counts, q)) for q in quantiles
        },
        "raw_signature_l2_quantiles": {
            str(q): float(np.quantile(np.linalg.norm(target_raw_signature, axis=1), q)) for q in quantiles
        },
        "head_logit_quantiles": {
            str(q): float(np.quantile(coefficient[0] + target_signature @ coefficient[1:], q)) for q in quantiles
        },
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V106 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V106 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = feature_graph_persistence_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"]),
    }]
    model_eligible = bool(
        model_audit["feature_graph"]["label_independent"]
        and model_audit["union_find_exact"]
        and not model_audit["topology_transform_fit_with_labels"]
        and model_audit["signature_scaling_fit_on_train_only"]
        and model_audit["logistic_head"]["success"]
        and not model_audit["svd_subspace_or_fastica"]
        and not model_audit["target_rows_used_for_state_or_signature_scaling_or_head_fit"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture,
            "metrics": values, "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 feature-graph-persistence blend; maximize 2*AUC+BA+10*net with fixed topology/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v106_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([("US", 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V106 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = feature_graph_persistence_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v106_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["feature_graph_persistence_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v106_model"] = selected["name"]
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
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(
            f"[V106 PERSISTENCE] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V106 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V106 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V106 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V106 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        ba_delta.append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V106 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-fold-time block bootstrap over inner-locked V106 persistence policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V106/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    topology_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["feature_graph"]["label_independent"]
        and audit["outer_model_audit"]["union_find_exact"]
        and not audit["outer_model_audit"]["topology_transform_fit_with_labels"]
        and audit["outer_model_audit"]["signature_scaling_fit_on_train_only"]
        and audit["outer_model_audit"]["logistic_head"]["success"]
        and not audit["outer_model_audit"]["svd_subspace_or_fastica"]
        and not audit["outer_model_audit"]["target_rows_used_for_state_or_signature_scaling_or_head_fit"]
        and not audit["outer_model_audit"]["target_labels_used"]
        for audit in audits
    )
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "bootstrap_net_delta_probability_gt_zero_ge_0_65": bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and audit["policy_locked_before_outer_evaluation"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "causal_feature_graph_persistence_contract_verified": topology_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V106 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap, "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "immutability": {
            "confidence_exact": confidence_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {
            "activated": not material_pass, "policy": "exact V69 entire frame" if not material_pass else None,
            "exact_frame_verified": bool(material_pass or selected.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def selected_contract(evaluation: dict[str, Any]) -> dict[str, Any]:
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    nested_bootstrap = json.loads(json.dumps(clean(evaluation["bootstrap"])))
    nested_bootstrap["contract_key"] = "selected.material_gate.nested_bootstrap"
    selected["material_gate"] = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": nested_bootstrap,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 frame",
    }
    require(
        selected["material_gate"]["nested_bootstrap"]["contract_key"]
        == "selected.material_gate.nested_bootstrap",
        "V106 nested bootstrap contract key mismatch",
    )
    require("bootstrap" in selected["robustness"], "V106 selected native robustness bootstrap missing")
    return selected


def write_outputs(
    authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]],
    evaluation: dict[str, Any], out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = selected_contract(evaluation)
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V106", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V106_DIAGNOSTIC_FEATURE_GRAPH_PERSISTENCE": evaluation["candidate_summary"],
                "V106_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V106", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"], "bootstrap": evaluation["bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V106", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "bootstrap": evaluation["bootstrap"], "material_gate": selected["material_gate"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "CAUSAL_FEATURE_GRAPH_PERSISTENCE_REPORT.json": {
            "version": "V106", "hypothesis": HYPOTHESIS, "features": FEATURES,
            "ordered_channels": ORDERED_CHANNELS, "standalone_edges": STANDALONE_EDGES,
            "graph_audit": GRAPH_AUDIT, "top_lifetimes": TOP_LIFETIMES,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": {
            "version": "V106", "hypothesis": HYPOTHESIS, "bootstrap": evaluation["bootstrap"],
            "nested_key_contract": {
                "auc_probability": "selected.material_gate.nested_bootstrap.auc_delta.probability_gt_zero",
                "net_probability": "selected.material_gate.nested_bootstrap.all_trade_net_delta.probability_gt_zero",
            },
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V106", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": selected["material_gate"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": scaffold.now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"], "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "signature_n": audit["outer_model_audit"]["persistence_signature_n"],
        "graph_edge_n": audit["outer_model_audit"]["feature_graph"]["edge_n"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V106_PERSISTENCE_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V106_PERSISTENCE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V106_DIAGNOSTIC_PERSISTENCE_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V106_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "feature_n": len(FEATURES),
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_feature": False,
            "architecture": "fixed causal-feature graph exact 0D lower-star persistence plus one L2 logistic head",
            "graph_audit": GRAPH_AUDIT,
            "filtrations": ["state", "negated_state"],
            "top_lifetimes_per_filtration": TOP_LIFETIMES,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run",
                "output_bound_to_environment": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "selected_contains_current_material_gate": True,
                "bootstrap_nested_keys": [
                    "selected.material_gate.nested_bootstrap.auc_delta.probability_gt_zero",
                    "selected.material_gate.nested_bootstrap.all_trade_net_delta.probability_gt_zero",
                ],
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0,
                "reason": "small exact union-find signatures and a two-thread shallow head avoid full-run contention",
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "fold_keys": [f"{audit['market']}:{audit['fold']}" for audit in audits],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": selected_contract(evaluation)["material_gate"],
        "bootstrap": evaluation["bootstrap"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0, "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["feature_graph_persistence_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
