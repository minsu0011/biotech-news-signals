"""V127 duplicate-balanced Mann-Whitney Borda-rank challenger.

Each same-market past train cut robustly scales the immutable 37 causal
numeric coordinates and collapses duplicate rows to equal-weight event-group
centroids.  For each coordinate, the exact DOWN-vs-UP Mann-Whitney win
probability fixes its direction orientation and rank-biserial effect strength.
Every query coordinate is then mapped to its equal-class-mixture empirical-CDF
percentile, reversed when the training rank effect is negative, and aggregated
by a normalized effect-weighted Borda average.  The result is the direction
probability directly: there is no fitted head, pair/list loss, return ordering,
temperature, geometry, density likelihood, or event retrieval.

This is not V78 PairLogit, V125 ListNet, GAF time-image, Hodges-Lehmann view
aggregation, GMM/Fisher-vector, geometry, density, graph, tree, kernel, or
neural training.  Earlier
committed V69 OOF alone chooses exact V69 or fixed .25/.50 blends in the inner
past.  Outer labels are evaluation-only under strict 35-minute nested
chronology; V69 confidence/high_conf remain exact.  Material failure restores
the exact entire atomic V69 frame.  A class-stratified event-group Bayesian
bootstrap separately audits weighted empirical-CDF and rank-effect stability.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_name] = "2"

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V127_DUPLICATE_BALANCED_MANN_WHITNEY_BORDA_RANK_V1"
VERSION = 127
HYPOTHESIS = "DUPLICATE_BALANCED_MANN_WHITNEY_BORDA_RANK_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

ROBUST_STATE_CLIP = 5.0
RANK_TIE_WEIGHT = 0.50
WEIGHT_FLOOR = 1e-12
SEED = 12701
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)

ARCHITECTURES = (
    {"name": "MANN_WHITNEY_BORDA_RANK_W0.25", "weight": 0.25},
    {"name": "MANN_WHITNEY_BORDA_RANK_W0.50", "weight": 0.50},
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


def verify_static_spec() -> dict[str, Any]:
    require(len(FEATURES) == 37, "V127 feature count changed")
    require(
        abs(float(scaffold.ROBUST_CLIP) - ROBUST_STATE_CLIP) <= 1e-15,
        "V127 robust state clip mismatch",
    )
    payload = (
        "numeric:37|robust_clip:5|equal_event_group_centroid|mann_whitney_tie_half|"
        "rank_biserial_absolute_weight|sign_orientation|equal_class_empirical_cdf|"
        "effect_weighted_borda_direct_probability|no_head_no_calibration"
    )
    return {
        "causal_numeric_n": len(FEATURES), "model_feature_n": len(FEATURES),
        "robust_state_clip": ROBUST_STATE_CLIP,
        "rank_effect": "exact duplicate-balanced Mann-Whitney UP-over-DOWN win probability with half ties",
        "orientation": "sign of rank-biserial effect",
        "feature_weight": "absolute rank-biserial effect normalized across 37 coordinates",
        "query_rank": "equal-class-mixture empirical CDF with half ties",
        "aggregation": "oriented effect-weighted Borda average",
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "target_representation_label_independent": True, "equal_class_prior": True,
        "fitted_head": False, "fitted_temperature": False,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, metrics, robust numeric transform, blending, canonical gate, and atomic IO only; no V85 output is read",
    }
    audit["v127_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    process = psutil.Process()
    process.cpu_affinity(list(PREPARATION_CPU_AFFINITY))
    actual = tuple(sorted(process.cpu_affinity()))
    require(actual == PREPARATION_CPU_AFFINITY, "V127 preparation CPU affinity is not exactly 30-31")
    return {
        "requested_logical_cpus": list(PREPARATION_CPU_AFFINITY),
        "actual_logical_cpus": list(actual), "exact_match": True,
        "process_id": process.pid,
    }


def group_numeric_state(
    frame: pd.DataFrame, numeric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(frame) == len(numeric), "V127 group-state shape mismatch")
    state = pd.DataFrame(numeric)
    state["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    state["y"] = frame.y.to_numpy(int)
    grouped = state.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V127 event group crosses direction labels")
    result = grouped[list(range(len(FEATURES)))].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    require(result.shape[1] == len(FEATURES) and np.isfinite(result).all(), "V127 group numeric state invalid")
    return result, target, group_ids


def unweighted_midrank_cdf(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    left = np.searchsorted(sorted_values, query, side="left")
    right = np.searchsorted(sorted_values, query, side="right")
    return (left + RANK_TIE_WEIGHT * (right - left)) / float(len(sorted_values))


def rank_effect(down: np.ndarray, up: np.ndarray) -> float:
    ordered_down = np.sort(down, kind="stable")
    return float(np.mean(unweighted_midrank_cdf(ordered_down, up)))


def nominal_rank_model(
    train_state: np.ndarray, target: np.ndarray, query_state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    down = train_state[target == 0]
    up = train_state[target == 1]
    require(len(down) >= 80 and len(up) >= 80, "V127 class event-group support too small")
    effect = np.asarray([rank_effect(down[:, j], up[:, j]) for j in range(len(FEATURES))], float)
    orientation = np.where(effect >= 0.5, 1.0, -1.0)
    strength = np.abs(2.0 * effect - 1.0)
    require(float(strength.sum()) > WEIGHT_FLOOR, "V127 rank effects have zero total strength")
    feature_weight = strength / strength.sum()
    percentile = np.empty((len(query_state), len(FEATURES)), float)
    for j in range(len(FEATURES)):
        down_cdf = unweighted_midrank_cdf(np.sort(down[:, j], kind="stable"), query_state[:, j])
        up_cdf = unweighted_midrank_cdf(np.sort(up[:, j], kind="stable"), query_state[:, j])
        balanced_cdf = 0.5 * (down_cdf + up_cdf)
        percentile[:, j] = balanced_cdf if orientation[j] > 0.0 else 1.0 - balanced_cdf
    probability = np.clip(percentile @ feature_weight, 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V127 Borda probability invalid")
    ordered = np.argsort(-feature_weight, kind="stable")
    return probability, {
        "down_event_group_n": len(down), "up_event_group_n": len(up),
        "mann_whitney_effect": effect.tolist(),
        "orientation_positive_n": int(np.sum(orientation > 0.0)),
        "orientation_negative_n": int(np.sum(orientation < 0.0)),
        "rank_biserial_strength_sum": float(strength.sum()),
        "feature_weight_sum": float(feature_weight.sum()),
        "feature_weight_max": float(feature_weight.max()),
        "top_weighted_features": [
            {"feature": FEATURES[j], "weight": float(feature_weight[j]), "up_win_probability": float(effect[j]), "orientation": int(orientation[j])}
            for j in ordered[:8]
        ],
        "feature_effect_sha256": array_sha256(effect),
        "feature_weight_sha256": array_sha256(feature_weight),
        "oriented_percentile_sha256": array_sha256(percentile),
    }


def weighted_prefix_cdf(sorted_values: np.ndarray, sorted_weight: np.ndarray, query: np.ndarray) -> np.ndarray:
    prefix = np.vstack([np.zeros((1, sorted_weight.shape[1]), float), np.cumsum(sorted_weight, axis=0)])
    left = np.searchsorted(sorted_values, query, side="left")
    right = np.searchsorted(sorted_values, query, side="right")
    return prefix[left] + RANK_TIE_WEIGHT * (prefix[right] - prefix[left])


def native_rank_bootstrap(
    query: np.ndarray,
    down_state: np.ndarray,
    up_state: np.ndarray,
    nominal_probability: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    require(draws >= 16, "V127 native rank bootstrap draw count too small")
    generator = np.random.default_rng(seed)
    down_weight = generator.exponential(size=(len(down_state), draws))
    up_weight = generator.exponential(size=(len(up_state), draws))
    down_weight /= down_weight.sum(axis=0, keepdims=True)
    up_weight /= up_weight.sum(axis=0, keepdims=True)
    effect = np.empty((len(FEATURES), draws), float)
    percentile = np.empty((len(query), len(FEATURES), draws), float)
    for j in range(len(FEATURES)):
        down_order = np.argsort(down_state[:, j], kind="stable")
        up_order = np.argsort(up_state[:, j], kind="stable")
        down_values = down_state[down_order, j]
        up_values = up_state[up_order, j]
        down_sorted_weight = down_weight[down_order]
        up_sorted_weight = up_weight[up_order]
        up_vs_down = weighted_prefix_cdf(down_values, down_sorted_weight, up_values)
        effect[j] = np.sum(up_sorted_weight * up_vs_down, axis=0)
        down_cdf = weighted_prefix_cdf(down_values, down_sorted_weight, query[:, j])
        up_cdf = weighted_prefix_cdf(up_values, up_sorted_weight, query[:, j])
        balanced_cdf = 0.5 * (down_cdf + up_cdf)
        percentile[:, j, :] = np.where(effect[j][None, :] >= 0.5, balanced_cdf, 1.0 - balanced_cdf)
    strength = np.abs(2.0 * effect - 1.0)
    require(np.all(strength.sum(axis=0) > WEIGHT_FLOOR), "V127 bootstrap rank effect collapsed")
    feature_weight = strength / strength.sum(axis=0, keepdims=True)
    probability = np.einsum("nfd,fd->nd", percentile, feature_weight, optimize=True)
    require(probability.shape == (len(query), draws) and np.isfinite(probability).all(), "V127 native bootstrap invalid")
    median_probability = np.median(probability, axis=1)
    row_std = np.std(probability, axis=1)
    direction_agreement = np.mean((probability >= 0.5) == (nominal_probability[:, None] >= 0.5))
    quantiles = (0.10, 0.50, 0.90)
    return {
        "contract": "class-stratified event-group Bayesian bootstrap of weighted empirical CDFs and Mann-Whitney rank effects",
        "draws": draws, "seed": seed, "class_stratified": True,
        "event_group_unit_weights": True, "target_labels_used": False,
        "feature_n": len(FEATURES), "rank_tie_weight": RANK_TIE_WEIGHT,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probability - nominal_probability[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal_probability))),
        "direction_agreement_with_nominal": float(direction_agreement),
        "row_probability_std_quantiles": {str(q): float(np.quantile(row_std, q)) for q in quantiles},
        "median_probability_sha256": array_sha256(median_probability),
    }


def mann_whitney_borda_rank_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V127 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V127 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_numeric = processor.transform(train)
    train_state, train_y, train_groups = group_numeric_state(train, train_numeric)
    target_state = processor.transform(target_frame)
    probability, rank_audit = nominal_rank_model(train_state, train_y, target_state)
    down_state = train_state[train_y == 0]
    up_state = train_state[train_y == 1]
    native = None if native_draws <= 0 else native_rank_bootstrap(
        target_state, down_state, up_state, probability, native_draws, native_seed,
    )
    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES), "model_feature_n": len(FEATURES),
        "causal_pre_event_only": True,
        "categorical_feature_n": 0, "text_feature_n": 0,
        "source_or_ticker_identity": False,
        "architecture": "duplicate-balanced featurewise Mann-Whitney orientation and rank-biserial weighted Borda empirical-CDF probability",
        "static_spec": STATIC_SPEC, "train_scaling_only": True,
        "event_group_equal_weight": True, "train_event_group_n": len(train_groups),
        "rank_model": rank_audit, "equal_class_prior": True,
        "mann_whitney_exact_half_ties": True, "rank_biserial_effect_weight": True,
        "equal_class_empirical_cdf": True, "effect_weighted_borda": True,
        "inverse_or_determinant": False, "fitted_temperature": False,
        "fitted_direction_coefficient_n": 0, "learned_discriminative_head": False,
        "pairlogit_listnet_or_return_ranking_loss": False,
        "gaf_time_image_or_hodges_lehmann_view_ensemble": False,
        "geometry_or_density_likelihood": False,
        "path_signature_spd_or_dynamic_operator": False,
        "graph_tree_kernel_objective_or_neural": False,
        "nearest_analog_or_retrieval": False,
        "target_rows_used_for_scaling_effect_weight_ecdf_or_selection": False,
        "target_labels_used": False,
        "native_rank_bootstrap": native,
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V127 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V127 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = mann_whitney_borda_rank_direction(inner_train, inner_valid)
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
        model_audit["train_scaling_only"]
        and model_audit["event_group_equal_weight"]
        and model_audit["static_spec"]["target_representation_label_independent"]
        and model_audit["static_spec"]["model_feature_n"] == len(FEATURES)
        and model_audit["rank_model"]["down_event_group_n"] >= 80
        and model_audit["rank_model"]["up_event_group_n"] >= 80
        and model_audit["mann_whitney_exact_half_ties"]
        and model_audit["rank_biserial_effect_weight"]
        and model_audit["equal_class_empirical_cdf"]
        and model_audit["effect_weighted_borda"]
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["learned_discriminative_head"]
        and not model_audit["target_rows_used_for_scaling_effect_weight_ecdf_or_selection"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "model_eligible": model_eligible,
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 mann-whitney-borda-rank logit blend; fixed score and BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v127_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V127 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = mann_whitney_borda_rank_direction(
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
        diagnostic.loc[positions, "v127_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["mann_whitney_borda_rank_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v127_model"] = selected["name"]
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
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V127 MANN-WHITNEY BORDA] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V127 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V127 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V127 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V127 nested-bootstrap blocks")
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
        ba_delta.append(float(
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V127 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V127 policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int, smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V127/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    rank_valid = all(
        audit["outer_model_audit"]["causal_pre_event_only"]
        and audit["outer_model_audit"]["train_scaling_only"]
        and audit["outer_model_audit"]["event_group_equal_weight"]
        and audit["outer_model_audit"]["static_spec"]["target_representation_label_independent"]
        and audit["outer_model_audit"]["model_feature_n"] == len(FEATURES)
        and audit["outer_model_audit"]["rank_model"]["down_event_group_n"] >= 80
        and audit["outer_model_audit"]["rank_model"]["up_event_group_n"] >= 80
        and audit["outer_model_audit"]["mann_whitney_exact_half_ties"]
        and audit["outer_model_audit"]["rank_biserial_effect_weight"]
        and audit["outer_model_audit"]["equal_class_empirical_cdf"]
        and audit["outer_model_audit"]["effect_weighted_borda"]
        and audit["outer_model_audit"]["equal_class_prior"]
        and not audit["outer_model_audit"]["inverse_or_determinant"]
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["learned_discriminative_head"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_effect_weight_ecdf_or_selection"]
        and not audit["outer_model_audit"]["target_labels_used"]
        for audit in audits
    )
    controller_native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_controller_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    model_native_bootstrap = [audit["outer_model_audit"]["native_rank_bootstrap"] for audit in audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["class_stratified"]
        and item["event_group_unit_weights"]
        and not item["target_labels_used"]
        and 0.0 <= item["direction_agreement_with_nominal"] <= 1.0
        for item in model_native_bootstrap
    )
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_controller_native.issubset(controller_native_bootstrap),
        "mann_whitney_borda_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "duplicate_balanced_mann_whitney_borda_rank_contract_verified": rank_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V127 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "controller_native_robustness_bootstrap": controller_native_bootstrap,
        "model_native_rank_bootstrap": model_native_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
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
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_rank_bootstrap": evaluation["model_native_rank_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V127 nested bootstrap contract key mismatch",
    )
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V127 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V127 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / f"V{VERSION}").resolve(), "V127 output is outside controller research/staging/V127")
    require(out.exists() and out.is_dir(), "V127 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V127 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V127 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V127 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V127 output is not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V127 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V127", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V127_DIAGNOSTIC_MANN_WHITNEY_BORDA_RANK": evaluation["candidate_summary"],
                "V127_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V127", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V127", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "MANN_WHITNEY_BORDA_RANK_REPORT.json": {
            "version": "V127", "hypothesis": HYPOTHESIS,
            "features": FEATURES,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V127", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V127", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
            "model_native_mann_whitney_borda_bootstrap": evaluation["model_native_rank_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V127", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit, "completed_at": scaffold.now(),
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
        "train_event_group_n": audit["outer_model_audit"]["train_event_group_n"],
        "down_event_group_n": audit["outer_model_audit"]["rank_model"]["down_event_group_n"],
        "up_event_group_n": audit["outer_model_audit"]["rank_model"]["up_event_group_n"],
        "rank_feature_n": audit["outer_model_audit"]["model_feature_n"],
        "native_bootstrap_draws": audit["outer_model_audit"]["native_rank_bootstrap"]["draws"],
        "strict_outer_35m_embargo": audit["outer_chronology"]["strict_35m_embargo"],
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V127_MANN_WHITNEY_BORDA_RANK_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V127_MANN_WHITNEY_BORDA_RANK_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V127_DIAGNOSTIC_MANN_WHITNEY_BORDA_RANK_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V127_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V127 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    affinity_audit = preparation_affinity() if (args.audit_only or args.smoke_test) else {
        "requested_logical_cpus": None, "actual_logical_cpus": psutil.Process().cpu_affinity(),
        "exact_match": None, "reason": "controller-authorized full run is not preparation",
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "duplicate-balanced Mann-Whitney feature orientation and rank-biserial weighted Borda empirical-CDF probability",
            "rank_feature_n": STATIC_SPEC["model_feature_n"],
            "rank_tie_weight": RANK_TIE_WEIGHT,
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_identity": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "controller_native_robustness_bootstrap": True,
                "model_native_mann_whitney_borda_bootstrap": True,
                "exact_entire_v69_fallback": True,
                "unexpected_output_conflict_fail_closed": True,
                "authorized_controller_prefiles_required": True,
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "cpu_affinity": affinity_audit,
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
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "fold_keys": [f"{audit['market']}:{audit['fold']}" for audit in audits],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "inner_baseline": audits[0]["policy"]["baseline"],
        "inner_selected": audits[0]["policy"]["selected"],
        "inner_best_non_noop": best_non_noop,
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_evaluation_only_architectures": audits[0]["outer_architecture_diagnostics_evaluation_only"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "model_audit": audits[0]["outer_model_audit"],
        "current_material_gate": current_material_gate(evaluation),
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "cpu_affinity": affinity_audit, "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    output = write_outputs(args.output, authority, evidence, audits, evaluation)
    summary["output_written"] = True
    summary["output"] = output
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
