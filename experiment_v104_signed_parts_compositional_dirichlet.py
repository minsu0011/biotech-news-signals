"""V104 signed-parts compositional Dirichlet direction challenger.

Each embargoed same-market training cut robustly scales 37 causal numeric
states.  Positive and negative magnitudes become separate parts and a fixed
neutral part closes them onto a continuous 75-dimensional simplex.  For each
direction class, event-group duplicate weights estimate a Dirichlet mean and
fixed method-of-moments concentration.  The averaged class log-likelihood
contrast directly produces direction probability.

There is no fitted discriminative coefficient/head, ordered fused-lasso path,
Owen empirical-likelihood sample dual, pairwise class energy score, neighbor,
SVD subspace, covariance density, tree, kernel, or neural representation.
Earlier committed V69 OOF rows alone choose exact V69 or fixed .25/.50 logit
blends on the inner past.  Outer labels are evaluation-only under a strict
35-minute embargo; V69 confidence/high_conf remain exact.  Any failed material
gate restores the exact entire V69 frame and all 14 canonical gates are
controller-recomputed.
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
from scipy.special import expit, gammaln
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v101_class_conditional_robust_affine_subspace as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V104_SIGNED_PARTS_COMPOSITIONAL_DIRICHLET_V1"
VERSION = 104
HYPOTHESIS = "SIGNED_PARTS_COMPOSITIONAL_DIRICHLET_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_MARKET_FOLD_ROWS = base.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = base.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v101_class_conditional_robust_affine_subspace.py"
EXPECTED_SCAFFOLD_SHA256 = "34b62edbdf7d1292569c78266e108cf24a1700e1920a52076799671f0772b6db"

PART_FLOOR = 1e-3
NEUTRAL_PART = 1.0
CONCENTRATION_FLOOR = 2.0
CONCENTRATION_CAP = 200.0
ALPHA_FLOOR = 0.05
MOMENT_VARIANCE_FLOOR = 1e-10
LOG_LIKELIHOOD_TEMPERATURE = 1.0
COMPOSITION_DIMENSION = 2 * len(FEATURES) + 1
BLEND_WEIGHTS = (0.25, 0.50)
SEED = 10401
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = base.COST

ARCHITECTURES = (
    {"name": "COMPOSITIONAL_DIRICHLET_W0.25", "weight": 0.25},
    {"name": "COMPOSITIONAL_DIRICHLET_W0.50", "weight": 0.50},
)

require = base.require
sha256 = base.sha256
array_sha256 = base.array_sha256
clean = base.clean
bool_series = base.bool_series
atomic_json = base.atomic_json
atomic_csv = base.atomic_csv
load_authorized = base.load_authorized
chronology = base.chronology
aligned_dev = base.aligned_dev
prior_train = base.prior_train
inner_partition = base.inner_partition
blend_probability = base.blend_probability
metric = base.metric
controller = base.controller
v44 = base.v44


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V101 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metrics, blending and IO utilities only; no V101 output is read",
        "transitive_v98_utility_sha256": base.EXPECTED_SCAFFOLD_SHA256,
    }
    return audit


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "V104 duplicate weights invalid")
    return weights


def signed_parts_composition(matrix: np.ndarray) -> np.ndarray:
    positive = np.maximum(matrix, 0.0) + PART_FLOOR
    negative = np.maximum(-matrix, 0.0) + PART_FLOOR
    neutral = np.full((len(matrix), 1), NEUTRAL_PART, dtype=float)
    raw = np.concatenate([positive, negative, neutral], axis=1)
    composition = raw / raw.sum(axis=1, keepdims=True)
    require(composition.shape[1] == COMPOSITION_DIMENSION, "V104 composition dimension changed")
    require(np.isfinite(composition).all() and np.all(composition > 0.0), "V104 composition invalid")
    require(np.allclose(composition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "V104 simplex closure failed")
    return composition


def fit_dirichlet(composition: np.ndarray, weights: np.ndarray, label: int) -> dict[str, Any]:
    require(len(composition) >= 100 and composition.shape[1] == COMPOSITION_DIMENSION, "V104 class support changed")
    normalized = weights / weights.sum()
    mean = np.sum(normalized[:, None] * composition, axis=0)
    variance = np.sum(normalized[:, None] * np.square(composition - mean), axis=0)
    eligible = (
        (variance > MOMENT_VARIANCE_FLOOR)
        & (mean > PART_FLOOR / 100.0)
        & (mean < 1.0 - PART_FLOOR / 100.0)
    )
    candidates = mean[eligible] * (1.0 - mean[eligible]) / variance[eligible] - 1.0
    candidates = candidates[np.isfinite(candidates) & (candidates > 0.0)]
    require(len(candidates) >= COMPOSITION_DIMENSION // 2, "V104 concentration moment support too small")
    raw_concentration = float(np.median(candidates))
    concentration = float(np.clip(raw_concentration, CONCENTRATION_FLOOR, CONCENTRATION_CAP))
    alpha = mean * concentration + ALPHA_FLOOR
    require(np.isfinite(alpha).all() and np.all(alpha > 0.0), "V104 Dirichlet alpha invalid")
    return {
        "label": label, "alpha": alpha,
        "log_normalizer": float(gammaln(alpha.sum()) - np.sum(gammaln(alpha))),
        "audit": {
            "label": label, "train_n": len(composition),
            "effective_duplicate_weight_n": float(weights.sum()),
            "composition_dimension": COMPOSITION_DIMENSION,
            "moment_component_n": int(len(candidates)),
            "raw_concentration_median": raw_concentration,
            "concentration": concentration, "alpha_sum": float(alpha.sum()),
            "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
            "mean_composition_sha256": array_sha256(mean),
            "alpha_sha256": array_sha256(alpha),
        },
    }


def dirichlet_log_likelihood(composition: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    values = float(model["log_normalizer"]) + np.log(composition) @ (model["alpha"] - 1.0)
    require(np.isfinite(values).all(), "V104 Dirichlet log likelihood non-finite")
    return values


def compositional_dirichlet_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V104 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V104 market fit mismatch")
    processor = base.base.scaffold.RobustNumericState().fit(train)
    train_matrix = processor.transform(train)
    target_matrix = processor.transform(target_frame)
    train_composition = signed_parts_composition(train_matrix)
    target_composition = signed_parts_composition(target_matrix)
    labels = train.y.to_numpy(int)
    duplicate_weights = duplicate_unit_weights(train)
    models: dict[int, dict[str, Any]] = {}
    for label in (0, 1):
        index = labels == label
        require(int(index.sum()) >= 100, "V104 class support too small")
        models[label] = fit_dirichlet(train_composition[index], duplicate_weights[index], label)
    down_score = dirichlet_log_likelihood(target_composition, models[0])
    up_score = dirichlet_log_likelihood(target_composition, models[1])
    log_likelihood_contrast = np.clip(
        (up_score - down_score) / (COMPOSITION_DIMENSION * LOG_LIKELIHOOD_TEMPERATURE),
        -30.0, 30.0,
    )
    probability = np.clip(expit(log_likelihood_contrast), 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V104 probability non-finite")
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "feature_n": len(FEATURES), "composition_dimension": COMPOSITION_DIMENSION,
        "architecture": "signed positive/negative causal-state parts closed with neutral mass, then duplicate-weighted class Dirichlet moment likelihood contrast",
        "class_dirichlet": {str(label): models[label]["audit"] for label in (0, 1)},
        "part_floor": PART_FLOOR, "neutral_part": NEUTRAL_PART,
        "log_likelihood_temperature": LOG_LIKELIHOOD_TEMPERATURE,
        "target_rows_used_for_scaling_composition_or_dirichlet_fit": False,
        "target_labels_used": False, "outer_labels_used": False,
        "causal_numeric_only": True, "categorical_feature_n": 0,
        "text_feature_n": 0, "source_feature": False, "ticker_feature": False,
        "global_fitted_discriminative_coefficient_n": 0,
        "ordered_fused_lasso_or_total_variation": False,
        "owen_empirical_likelihood_dual_or_sample_reweighting": False,
        "pairwise_sample_distance_or_energy_score": False,
        "nearest_event_retrieval": False, "covariance_density": False,
        "down_log_likelihood_quantiles": {
            str(q): float(np.quantile(down_score, q)) for q in (0.10, 0.50, 0.90)
        },
        "up_log_likelihood_quantiles": {
            str(q): float(np.quantile(up_score, q)) for q in (0.10, 0.50, 0.90)
        },
        "contrast_quantiles": {
            str(q): float(np.quantile(log_likelihood_contrast, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_quantiles": {
            str(q): float(np.quantile(probability, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = compositional_dirichlet_direction(inner_train, inner_valid)
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
    contract_eligible = bool(
        model_audit["composition_dimension"] == COMPOSITION_DIMENSION
        and model_audit["global_fitted_discriminative_coefficient_n"] == 0
        and not model_audit["ordered_fused_lasso_or_total_variation"]
        and not model_audit["owen_empirical_likelihood_dual_or_sample_reweighting"]
        and not model_audit["pairwise_sample_distance_or_energy_score"]
        and not model_audit["target_rows_used_for_scaling_composition_or_dirichlet_fit"]
        and not model_audit["target_labels_used"]
        and all(
            side["composition_dimension"] == COMPOSITION_DIMENSION
            and side["alpha_min"] > 0.0
            for side in model_audit["class_dirichlet"].values()
        )
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
            "model_contract_eligible": contract_eligible,
            "eligible": bool(contract_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 compositional-Dirichlet logit blend; maximize fixed 2*AUC+BA+10*net under contract/BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v104_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V104 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = compositional_dirichlet_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v104_model"] = selected["name"]
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
        evidence["compositional_dirichlet_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v104_model"] = selected["name"]
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
            f"[V104 DIRICHLET] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V104 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V104 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V104 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V104 nested-bootstrap blocks")
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
    require(len(auc_delta) >= int(0.90 * draws), "V104 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market/fold/time-block bootstrap over inner-locked compositional-Dirichlet policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V104/controller gate mismatch")
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
    dirichlet_contract = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["composition_dimension"] == COMPOSITION_DIMENSION
        and audit["outer_model_audit"]["global_fitted_discriminative_coefficient_n"] == 0
        and not audit["outer_model_audit"]["ordered_fused_lasso_or_total_variation"]
        and not audit["outer_model_audit"]["owen_empirical_likelihood_dual_or_sample_reweighting"]
        and not audit["outer_model_audit"]["pairwise_sample_distance_or_energy_score"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_composition_or_dirichlet_fit"]
        and not audit["outer_model_audit"]["target_labels_used"]
        for audit in audits
    )
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
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "signed_parts_compositional_dirichlet_contract_verified": dirichlet_contract,
    }
    material_pass = bool(all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V104 fallback is not exact entire V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "direction_change": {
            "probability_changed_n": int(np.sum(champion.prob.to_numpy(float) != diagnostic_original.prob.to_numpy(float))),
            "direction_changed_n": int(np.sum((champion.prob.to_numpy(float) >= 0.5) != (diagnostic_original.prob.to_numpy(float) >= 0.5))),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic_original.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected_frame.prob.to_numpy(float)),
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 frame" if not material_pass else None,
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
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V104 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(not out.name.startswith("output_V104"), "runner must not write output_V104 directly")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = current_material_gate(evaluation)
    require("bootstrap" in selected["robustness"], "V104 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V104", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V104_COMPOSITIONAL_DIRICHLET_DIAGNOSTIC": evaluation["candidate_summary"],
                "V104_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V104", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V104", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": selected["material_gate"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V104_COMPOSITIONAL_DIRICHLET_REPORT.json": {
            "version": "V104", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "composition_dimension": COMPOSITION_DIMENSION,
            "architectures": ARCHITECTURES, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V104_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V104_COMPOSITIONAL_DIRICHLET_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V104_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V104_COMPOSITIONAL_DIRICHLET_OUTER_OOF.csv.gz",
            "V104_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        ])
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run
            and not args.dry_run and args.output is None,
            "explicit mode cannot accompany MARKET_BIO_VERSION_OUTPUT",
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
            "model_contract": {
                "features": FEATURES, "feature_n": len(FEATURES),
                "representation": "positive/negative robust-state parts plus fixed neutral mass closed to a continuous simplex",
                "estimator": "duplicate-weighted class Dirichlet method-of-moments log-likelihood contrast",
                "composition_dimension": COMPOSITION_DIMENSION,
                "global_fitted_discriminative_coefficient_n": 0,
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V103 actual root runners/drafts",
                "signed_parts_compositional_dirichlet_found": False,
                "V90": "discrete quantile TAN with dependency edges and Dirichlet table smoothing; V104 is one continuous simplex likelihood with no edge/table",
                "V102": "ordered coefficient-jump L1 logistic; V104 has no coefficient or optimization path",
                "V103": "blockwise Owen sample empirical-likelihood dual/deviance grid; V104 fits finite-dimensional class Dirichlet moments and never reweights query constraints",
            },
            "nested_contract": {
                "strict_embargo_minutes": 35, "outer_labels_used_for_selection": False,
                "bootstrap_key": "selected.material_gate.nested_bootstrap",
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_market_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "reason": "bounded two-thread closed-form compositional moment smoke",
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            250 if args.smoke_test else BOOTSTRAP_DRAWS,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(
        non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]),
    )
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": current_material_gate(evaluation),
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
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["compositional_dirichlet_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    if not args.dry_run and args.full_run:
        summary.update(write_outputs(args.output, authority, evidence, audits, evaluation))
        summary["output_written"] = True
        summary["output"] = str(args.output)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
