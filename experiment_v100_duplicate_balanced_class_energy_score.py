"""V100 duplicate-balanced class energy-score direction challenger.

Every embargoed same-market training cut is represented by 37 robustly scaled
causal pre-decision numeric states.  For each direction class, the estimator
uses the global empirical energy proper score

    E ||x - X_c||_2 - 0.5 E ||X_c - X'_c||_2,

where all expectations use event-group duplicate-balanced past-only weights.
The difference between DOWN and UP energy scores is transformed with a
train-only robust IQR temperature.  This is a global distributional scoring
rule, not local neighbor retrieval, a class prototype, covariance/density,
projection depth, fitted coefficient/head, tree, neural representation,
kernel map, pattern-rule system, or supervised PLS projection.

Earlier same-market committed V69 OOF labels alone select exact V69 or fixed
0.25/0.50 logit blends.  Outer labels are evaluation-only, all cuts enforce a
strict 35-minute embargo and event-group purge, V69 confidence/high_conf are
frozen, the controller recomputes canonical 14 gates, and material failure
restores the exact entire atomic V69 frame.
"""
from __future__ import annotations

import argparse
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
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.special import expit
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "research" / "staging" / "V100_DUPLICATE_BALANCED_CLASS_ENERGY_SCORE_V1"
)
VERSION = 100
HYPOTHESIS = "DUPLICATE_BALANCED_CLASS_ENERGY_SCORE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

DISTANCE_CHUNK_ROWS = 192
TEMPERATURE_FLOOR = 0.02
SEED = 10001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
COST = scaffold.COST

ARCHITECTURES = (
    {"name": "CLASS_ENERGY_SCORE_W0.25", "weight": 0.25},
    {"name": "CLASS_ENERGY_SCORE_W0.50", "weight": 0.50},
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
        "purpose": "generic atomic authority, chronology, metrics, blending, robust scaling, and IO only; no V85 output is read",
    }
    audit["v100_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def class_duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    total = float(weights.sum())
    require(total > 0.0 and np.isfinite(weights).all(), "V100 duplicate weights invalid")
    return weights / total


def weighted_mean_euclidean_distance(
    query: np.ndarray, reference: np.ndarray, reference_weight: np.ndarray,
) -> np.ndarray:
    require(query.ndim == reference.ndim == 2, "V100 distance matrices must be two-dimensional")
    require(query.shape[1] == reference.shape[1], "V100 distance feature mismatch")
    require(len(reference) == len(reference_weight), "V100 reference weight mismatch")
    require(np.isclose(float(reference_weight.sum()), 1.0, rtol=0.0, atol=1e-10), "V100 weights do not sum to one")
    result = np.empty(len(query), dtype=float)
    normalizer = math.sqrt(float(query.shape[1]))
    for start in range(0, len(query), DISTANCE_CHUNK_ROWS):
        stop = min(start + DISTANCE_CHUNK_ROWS, len(query))
        distances = cdist(query[start:stop], reference, metric="euclidean") / normalizer
        require(np.isfinite(distances).all(), "V100 pairwise distance is non-finite")
        result[start:stop] = distances @ reference_weight
    require(np.isfinite(result).all() and np.all(result >= 0.0), "V100 weighted distance invalid")
    return result


def robust_temperature(train_margin: np.ndarray) -> tuple[float, dict[str, Any]]:
    lower, median, upper = np.quantile(train_margin, (0.25, 0.50, 0.75))
    iqr_scale = float((upper - lower) / 1.349)
    mad_scale = float(1.4826 * np.median(np.abs(train_margin - median)))
    standard_scale = float(np.std(train_margin))
    candidates = [value for value in (iqr_scale, mad_scale, standard_scale) if math.isfinite(value) and value > 1e-10]
    scale = max(candidates[0] if candidates else TEMPERATURE_FLOOR, TEMPERATURE_FLOOR)
    return scale, {
        "method": "train-only IQR/1.349 with deterministic MAD/std degeneracy diagnostics and fixed floor",
        "temperature": scale,
        "temperature_floor": TEMPERATURE_FLOOR,
        "iqr_scale": iqr_scale,
        "mad_scale": mad_scale,
        "standard_scale": standard_scale,
        "train_margin_quantiles": {
            "0.1": float(np.quantile(train_margin, 0.10)),
            "0.25": float(lower), "0.5": float(median), "0.75": float(upper),
            "0.9": float(np.quantile(train_margin, 0.90)),
        },
        "target_batch_used": False,
    }


def class_energy_score_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V100 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V100 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_matrix = processor.transform(train)
    target_matrix = processor.transform(target_frame)
    target = train.y.to_numpy(int)
    class_state: dict[int, dict[str, Any]] = {}
    for label in (0, 1):
        index = target == label
        require(int(index.sum()) >= 100, f"V100 class {label} support too small")
        frame = train.loc[index].reset_index(drop=True)
        matrix = train_matrix[index]
        weight = class_duplicate_weights(frame)
        within_mean = weighted_mean_euclidean_distance(matrix, matrix, weight)
        dispersion = float(weight @ within_mean)
        require(math.isfinite(dispersion) and dispersion > 0.0, "V100 class dispersion invalid")
        class_state[label] = {
            "matrix": matrix, "weight": weight, "dispersion": dispersion,
            "support": int(index.sum()), "effective_groups": int(frame.event_group_id.astype(str).nunique()),
            "effective_sample_size": float(1.0 / np.sum(np.square(weight))),
        }

    def score(matrix: np.ndarray, label: int) -> np.ndarray:
        state = class_state[label]
        mean_distance = weighted_mean_euclidean_distance(matrix, state["matrix"], state["weight"])
        return mean_distance - 0.5 * state["dispersion"]

    train_down_score = score(train_matrix, 0)
    train_up_score = score(train_matrix, 1)
    train_margin = train_down_score - train_up_score
    temperature, temperature_audit = robust_temperature(train_margin)
    target_down_score = score(target_matrix, 0)
    target_up_score = score(target_matrix, 1)
    raw_margin = target_down_score - target_up_score
    probability = expit(np.clip(raw_margin / temperature, -30.0, 30.0))
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V100 energy-score probability invalid")

    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]),
        "train_n": len(train), "target_n": len(target_frame),
        "feature_n": len(FEATURES), "causal_numeric_only": True,
        "categorical_feature_n": 0, "text_feature_n": 0,
        "source_feature": False, "ticker_feature": False,
        "architecture": "duplicate-balanced empirical class energy proper-score difference",
        "energy_score_equation": "E||x-X_c||2 - 0.5*E||X_c-X'_c||2",
        "distance": "Euclidean divided by sqrt(feature_n) after train-cut robust scaling",
        "all_class_samples_used": True,
        "neighbor_selection_or_index": False,
        "event_group_duplicate_balanced": True,
        "class_support": {str(label): class_state[label]["support"] for label in (0, 1)},
        "class_unique_event_groups": {str(label): class_state[label]["effective_groups"] for label in (0, 1)},
        "class_effective_sample_size": {str(label): class_state[label]["effective_sample_size"] for label in (0, 1)},
        "class_pairwise_dispersion": {str(label): class_state[label]["dispersion"] for label in (0, 1)},
        "temperature": temperature_audit,
        "fitted_direction_coefficient_n": 0,
        "tree_or_neural_model": False,
        "kernel_or_rff_map": False,
        "covariance_or_probability_density": False,
        "projection_or_depth": False,
        "class_prototype": False,
        "pattern_rule_or_belief_mass": False,
        "pls_or_latent_deflation": False,
        "target_rows_used_for_scaling_reference_or_temperature": False,
        "target_rows_used_only_as_score_queries": True,
        "target_labels_used": False,
        "down_energy_quantiles": {str(q): float(np.quantile(target_down_score, q)) for q in quantiles},
        "up_energy_quantiles": {str(q): float(np.quantile(target_up_score, q)) for q in quantiles},
        "raw_margin_quantiles": {str(q): float(np.quantile(raw_margin, q)) for q in quantiles},
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
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V100 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V100 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = class_energy_score_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(
            2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]
            + 10.0 * baseline_metric["all_trade_mean_signed_net"]
        ),
    }]
    model_eligible = bool(
        model_audit["all_class_samples_used"]
        and not model_audit["neighbor_selection_or_index"]
        and model_audit["event_group_duplicate_balanced"]
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["target_rows_used_for_scaling_reference_or_temperature"]
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
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 class-energy-score logit blend; maximize 2*AUC+BA+10*net with fixed model/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v100_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V100 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = class_energy_score_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v100_model"] = selected["name"]
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
        evidence["class_energy_score_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v100_model"] = selected["name"]
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
            f"[V100 ENERGY] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V100 evidence repeats an event")
    require(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)),
        "V100 changed V69 confidence",
    )
    require(
        np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)),
        "V100 changed V69 high-confidence",
    )
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V100 bootstrap blocks")
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
    require(len(auc_delta) >= int(0.90 * draws), "V100 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-fold-time block bootstrap over inner-locked V100 class-energy policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta),
        "balanced_accuracy_delta": interval(ba_delta),
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V100/controller gate mismatch")
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
    energy_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["all_class_samples_used"]
        and not audit["outer_model_audit"]["neighbor_selection_or_index"]
        and audit["outer_model_audit"]["event_group_duplicate_balanced"]
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_reference_or_temperature"]
        and audit["outer_model_audit"]["target_rows_used_only_as_score_queries"]
        and not audit["outer_model_audit"]["target_labels_used"]
        and not audit["outer_model_audit"]["pls_or_latent_deflation"]
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
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "duplicate_balanced_class_energy_contract_verified": energy_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(
        canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected,
        "selected/controller gate mismatch",
    )
    if not material_pass:
        require(selected.equals(champion), "V100 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
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
            "policy": "exact V69 entire frame" if not material_pass else None,
            "exact_frame_verified": bool(material_pass or selected.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected,
    }


def selected_contract(evaluation: dict[str, Any]) -> dict[str, Any]:
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
        "bootstrap": evaluation["bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 frame",
    }
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
            "version": "V100", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V100_DIAGNOSTIC_CLASS_ENERGY_SCORE": evaluation["candidate_summary"],
                "V100_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V100", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "bootstrap": evaluation["bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V100", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "bootstrap": evaluation["bootstrap"],
            "material_gate": selected["material_gate"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "DUPLICATE_BALANCED_CLASS_ENERGY_SCORE_REPORT.json": {
            "version": "V100", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "architectures": ARCHITECTURES,
            "distance_chunk_rows": DISTANCE_CHUNK_ROWS,
            "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": {
            "version": "V100", "hypothesis": HYPOTHESIS,
            "bootstrap": evaluation["bootstrap"],
            "nested_key_contract": {
                "auc_probability": "bootstrap.auc_delta.probability_gt_zero",
                "net_probability": "bootstrap.all_trade_net_delta.probability_gt_zero",
            },
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V100", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": selected["material_gate"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False,
            "completed_at": scaffold.now(),
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
        "down_dispersion": audit["outer_model_audit"]["class_pairwise_dispersion"]["0"],
        "up_dispersion": audit["outer_model_audit"]["class_pairwise_dispersion"]["1"],
        "temperature": audit["outer_model_audit"]["temperature"]["temperature"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V100_CLASS_ENERGY_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V100_CLASS_ENERGY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V100_DIAGNOSTIC_CLASS_ENERGY_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V100_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT and not (args.audit_only or args.smoke_test or args.full_run):
        args.full_run = True
    if CONTROLLER_OUTPUT and args.full_run:
        require(
            args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(),
            "controller full run must write MARKET_BIO_VERSION_OUTPUT",
        )
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
            "causal_numeric_only": True, "categorical_feature_n": 0,
            "text_feature_n": 0, "source_or_ticker_feature": False,
            "architecture": "duplicate-balanced empirical class energy proper-score difference",
            "energy_score_equation": "E||x-X_c||2 - 0.5*E||X_c-X'_c||2",
            "all_class_samples_used": True, "neighbor_selection_or_index": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "selected_contains_current_material_gate": True,
                "bootstrap_nested_keys": [
                    "bootstrap.auc_delta.probability_gt_zero",
                    "bootstrap.all_trade_net_delta.probability_gt_zero",
                ],
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0,
                "reason": "bounded chunked CPU energy-score smoke avoids contention with active full runners",
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
    best_non_noop = max(
        non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]),
    )
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
        "bootstrap": evaluation["bootstrap"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = (
            audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        )
        summary["class_energy_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
