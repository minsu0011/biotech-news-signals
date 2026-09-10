"""V101 class-conditional robust affine-subspace direction challenger.

Each embargoed same-market training cut is robustly standardized from 37
causal pre-decision numeric states.  Direction classes are then treated
separately: duplicate-balanced class centers and fixed-rank weighted SVD bases
define two affine subspaces.  A target receives one normalized orthogonal
reconstruction residual per class, and their contrast directly produces a
direction probability.

There is no fitted discriminative head, response covariance, PLS/NIPALS
deflation, class density determinant, pairwise sample energy score, neighbor
retrieval, kernel, tree, neural representation, or dynamic coefficient state.
Earlier committed V69 OOF rows alone select exact V69 or fixed .25/.50 logit
blends in the inner past.  Outer outcomes are evaluation-only under a strict
35-minute embargo; V69 confidence/high_conf are frozen.  Any failed material
gate returns the exact entire V69 frame and the controller recomputes all 14
canonical gates.
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
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v98_empirical_bayes_contrast_pattern_belief as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V101_CLASS_CONDITIONAL_AFFINE_SUBSPACE_V1"
VERSION = 101
HYPOTHESIS = "CLASS_CONDITIONAL_ROBUST_AFFINE_SUBSPACE_RESIDUAL_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_MARKET_FOLD_ROWS = base.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = base.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v98_empirical_bayes_contrast_pattern_belief.py"
EXPECTED_SCAFFOLD_SHA256 = "aef1939e9401c2411ccd224792e333688ddcf4601b13e0e5c363abf840f83bdb"

SUBSPACE_RANK = 6
RESIDUAL_SCALE_FLOOR = 1e-4
RESIDUAL_TEMPERATURE = 1.0
BLEND_WEIGHTS = (0.25, 0.50)
SEED = 10101
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = base.COST

ARCHITECTURES = (
    {"name": "AFFINE_SUBSPACE_W0.25", "weight": 0.25},
    {"name": "AFFINE_SUBSPACE_W0.50", "weight": 0.50},
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
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V98 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metric, bootstrap, blending and IO utilities only; no V98 output is read",
        "transitive_v85_utility_sha256": base.EXPECTED_SCAFFOLD_SHA256,
    }
    return audit


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "V101 duplicate weights invalid")
    return weights


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    require(int(finite.sum()) >= 20, "V101 weighted median support too small")
    value = values[finite]
    weight = weights[finite]
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    position = int(np.searchsorted(cumulative, 0.5 * float(cumulative[-1]), side="left"))
    return float(value[order[min(position, len(value) - 1)]])


def fit_class_subspace(
    matrix: np.ndarray, weights: np.ndarray, label: int,
) -> dict[str, Any]:
    require(len(matrix) >= 100 and matrix.shape[1] == len(FEATURES), "V101 class subspace support changed")
    normalized = weights / weights.sum()
    center = np.sum(normalized[:, None] * matrix, axis=0)
    centered = matrix - center
    weighted = centered * np.sqrt(normalized[:, None])
    _, singular, right = np.linalg.svd(weighted, full_matrices=False)
    require(len(singular) >= SUBSPACE_RANK, "V101 SVD rank support too small")
    basis = right[:SUBSPACE_RANK].T
    gram = basis.T @ basis
    orthogonality_error = float(np.max(np.abs(gram - np.eye(SUBSPACE_RANK))))
    require(orthogonality_error <= 1e-10, "V101 class basis lost orthogonality")
    projection = centered @ basis @ basis.T
    residual = np.mean(np.square(centered - projection), axis=1)
    residual_scale = max(weighted_median(residual, weights), RESIDUAL_SCALE_FLOOR)
    total_energy = float(np.sum(np.square(singular)))
    retained_energy = float(np.sum(np.square(singular[:SUBSPACE_RANK])))
    require(total_energy > 0.0 and math.isfinite(residual_scale), "V101 class subspace degenerate")
    return {
        "label": label, "center": center, "basis": basis,
        "residual_scale": residual_scale,
        "audit": {
            "label": label, "train_n": len(matrix), "effective_duplicate_weight_n": float(weights.sum()),
            "rank": SUBSPACE_RANK, "center_sha256": array_sha256(center),
            "basis_sha256": array_sha256(basis),
            "singular_values": singular[:SUBSPACE_RANK].tolist(),
            "retained_energy_fraction": retained_energy / total_energy,
            "orthogonality_max_abs_error": orthogonality_error,
            "training_residual_scale": residual_scale,
            "training_residual_quantiles": {
                str(q): float(np.quantile(residual, q)) for q in (0.10, 0.50, 0.90)
            },
        },
    }


def normalized_residual(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    centered = matrix - model["center"]
    projection = centered @ model["basis"] @ model["basis"].T
    residual = np.mean(np.square(centered - projection), axis=1)
    result = residual / float(model["residual_scale"])
    require(np.isfinite(result).all() and np.all(result >= -1e-12), "V101 normalized residual invalid")
    return np.maximum(result, 0.0)


def affine_subspace_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V101 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V101 market fit mismatch")
    processor = base.scaffold.RobustNumericState().fit(train)
    train_matrix = processor.transform(train)
    target_matrix = processor.transform(target_frame)
    labels = train.y.to_numpy(int)
    duplicate_weights = duplicate_unit_weights(train)
    models: dict[int, dict[str, Any]] = {}
    for label in (0, 1):
        index = labels == label
        require(int(index.sum()) >= 100, "V101 class support too small")
        models[label] = fit_class_subspace(train_matrix[index], duplicate_weights[index], label)
    down_residual = normalized_residual(target_matrix, models[0])
    up_residual = normalized_residual(target_matrix, models[1])
    residual_contrast = np.clip(
        (down_residual - up_residual) / RESIDUAL_TEMPERATURE, -30.0, 30.0,
    )
    probability = np.clip(expit(residual_contrast), 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V101 probability is non-finite")
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "feature_n": len(FEATURES), "subspace_rank": SUBSPACE_RANK,
        "architecture": "two duplicate-weighted class-conditional robust affine SVD subspaces with normalized orthogonal reconstruction-residual contrast",
        "class_subspaces": {str(label): models[label]["audit"] for label in (0, 1)},
        "residual_temperature": RESIDUAL_TEMPERATURE,
        "target_rows_used_for_scaling_centers_bases_or_residual_scales": False,
        "target_labels_used": False, "outer_labels_used": False,
        "causal_numeric_only": True, "categorical_feature_n": 0,
        "text_feature_n": 0, "source_feature": False, "ticker_feature": False,
        "global_fitted_discriminative_coefficient_n": 0,
        "pls_response_covariance_or_nipals": False,
        "pairwise_sample_distance_or_energy_score": False,
        "nearest_event_retrieval": False, "density_log_determinant": False,
        "down_residual_quantiles": {
            str(q): float(np.quantile(down_residual, q)) for q in (0.10, 0.50, 0.90)
        },
        "up_residual_quantiles": {
            str(q): float(np.quantile(up_residual, q)) for q in (0.10, 0.50, 0.90)
        },
        "residual_contrast_quantiles": {
            str(q): float(np.quantile(residual_contrast, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_quantiles": {
            str(q): float(np.quantile(probability, q)) for q in (0.10, 0.50, 0.90)
        },
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = affine_subspace_direction(inner_train, inner_valid)
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
        model_audit["subspace_rank"] == SUBSPACE_RANK
        and model_audit["global_fitted_discriminative_coefficient_n"] == 0
        and not model_audit["pls_response_covariance_or_nipals"]
        and not model_audit["pairwise_sample_distance_or_energy_score"]
        and not model_audit["target_rows_used_for_scaling_centers_bases_or_residual_scales"]
        and not model_audit["target_labels_used"]
        and all(
            side["rank"] == SUBSPACE_RANK and side["orthogonality_max_abs_error"] <= 1e-10
            for side in model_audit["class_subspaces"].values()
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 affine-subspace logit blend; maximize fixed 2*AUC+BA+10*net under contract/BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v101_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V101 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = affine_subspace_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v101_model"] = selected["name"]
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
        evidence["affine_subspace_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v101_model"] = selected["name"]
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
            f"[V101 SUBSPACE] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V101 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V101 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V101 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V101 nested-bootstrap blocks")
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
    require(len(auc_delta) >= int(0.90 * draws), "V101 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market/fold/time-block bootstrap over inner-locked affine-subspace policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V101/controller gate mismatch")
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
    subspace_contract = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["subspace_rank"] == SUBSPACE_RANK
        and audit["outer_model_audit"]["global_fitted_discriminative_coefficient_n"] == 0
        and not audit["outer_model_audit"]["pls_response_covariance_or_nipals"]
        and not audit["outer_model_audit"]["pairwise_sample_distance_or_energy_score"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_centers_bases_or_residual_scales"]
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
        "class_affine_subspace_contract_verified": subspace_contract,
    }
    material_pass = bool(all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V101 fallback is not exact entire V69")
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
    return {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(not out.name.startswith("output_V101"), "runner must not write output_V101 directly")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = current_material_gate(evaluation)
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V101", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V101_AFFINE_SUBSPACE_DIAGNOSTIC": evaluation["candidate_summary"],
                "V101_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V101", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V101", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V101_AFFINE_SUBSPACE_REPORT.json": {
            "version": "V101", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "subspace_rank": SUBSPACE_RANK,
            "architectures": ARCHITECTURES, "authority_audit": authority,
            "nested_fold_audits": audits, "evaluation": compact,
        },
        "V101_IMMUTABILITY_AUDIT.json": {
            "direction_change": evaluation["direction_change"],
            "confidence_and_highconf_exact": evaluation["material_checks"]["v69_confidence_and_highconf_exact"],
            "fallback": evaluation["fallback"],
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V101_AFFINE_SUBSPACE_OUTER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V101_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "runner_files_written": sorted([
            *reports, "V101_AFFINE_SUBSPACE_OUTER_OOF.csv.gz",
            "V101_FAIL_CLOSED_SELECTED_OOF.csv.gz",
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
            not args.audit_only and not args.smoke_test and not args.full_run,
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
                "representation": "train-cut robust causal state with two class-conditional weighted affine SVD subspaces",
                "estimator": "normalized orthogonal reconstruction-residual contrast",
                "subspace_rank": SUBSPACE_RANK,
                "global_fitted_discriminative_coefficient_n": 0,
                "pairwise_sample_distance_or_energy_score": False,
                "device": "CPU", "gpu_used": False,
            },
            "prior_art_audit": {
                "versions": "V37-V100 actual root runners/drafts plus V100 fixed declaration",
                "class_conditional_affine_subspace_residual_found": False,
                "V99": "global supervised PLS1 X-y covariance/NIPALS deflation plus ridge head; V101 fits no response covariance or head",
                "V100": "class empirical energy score over pairwise sample distances; V101 retains only two centers/bases/scales and computes no pairwise sample distance",
            },
            "nested_contract": {
                "strict_embargo_minutes": 35, "outer_labels_used_for_selection": False,
                "bootstrap_key": "selected.material_gate.nested_bootstrap",
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_market_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "reason": "bounded two-thread dense SVD smoke",
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
        summary["affine_subspace_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    if not args.dry_run and args.full_run:
        summary.update(write_outputs(args.output, authority, evidence, audits, evaluation))
        summary["output_written"] = True
        summary["output"] = str(args.output)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
