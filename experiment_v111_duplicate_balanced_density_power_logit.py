"""V111 duplicate-balanced density-power-divergence direction challenger.

Every embargoed same-market train cut is represented only by robustly scaled
causal numeric state and deterministic missingness flags.  A fixed linear
direction model minimizes the exact Bernoulli density-power-divergence scoring
rule with beta 0.30, duplicate/class weights, and one fixed L2 penalty.  Unlike
ordinary log loss, the resulting estimating equation redescends for observations
assigned vanishing observed-class probability, targeting label/outlier robustness
without environment weights, trimming, adversarial feature perturbations, or
post-hoc calibration.  The analytic objective and gradient are solved once by
deterministic L-BFGS.  This is not a class density, graph, conformal typicality,
hyperspherical vMF model, archetype, retrieval, tree, kernel, or neural model.

Earlier committed V69 OOF labels alone choose exact V69 or fixed .25/.50 logit
blends.  Outer labels are evaluation-only under strict 35-minute nested
chronology.  V69 confidence/high-confidence remain exact.  Material failure
restores the exact entire V69 DataFrame.

Audit and US:2 two-thread smoke write nothing.  With a future authorized
MARKET_BIO_VERSION_OUTPUT, no-argument invocation performs the full six folds
and writes the required controller reports/current material-gate contract.
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
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

import experiment_v102_ordered_horizon_fused_lasso_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V111_DUPLICATE_BALANCED_DENSITY_POWER_LOGIT_V1"
VERSION = 111
HYPOTHESIS = "DUPLICATE_BALANCED_DENSITY_POWER_DIVERGENCE_LOGIT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v102_ordered_horizon_fused_lasso_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "d398c651102ff2e47e747d277aca5ee24718f0ae29013b3fb02f0bfc07fba3b2"

DPD_BETA = 0.30
L2_PENALTY = 0.02
MAX_ITERATIONS = 300
GRADIENT_TOLERANCE = 1e-7
SEED = 11101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "DENSITY_POWER_DIVERGENCE_LOGIT_W0.25", "weight": 0.25},
    {"name": "DENSITY_POWER_DIVERGENCE_LOGIT_W0.50", "weight": 0.50},
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


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V102 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust-state, metrics, canonical gate, blending, and atomic IO only; no V102 output is read",
    }
    return audit


def density_power_objective(
    parameter: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Exact Bernoulli density-power-divergence objective and analytic gradient."""
    coefficient = parameter[:-1]
    intercept = float(parameter[-1])
    probability = np.clip(expit(matrix @ coefficient + intercept), 1e-12, 1.0 - 1e-12)
    complement = 1.0 - probability
    observed_power = target * probability ** DPD_BETA + (1.0 - target) * complement ** DPD_BETA
    model_power = probability ** (1.0 + DPD_BETA) + complement ** (1.0 + DPD_BETA)
    row_loss = model_power / (1.0 + DPD_BETA) - observed_power / DPD_BETA
    objective = float(np.sum(weight * row_loss) + 0.5 * L2_PENALTY * np.dot(coefficient, coefficient))

    dp_dz = probability * complement
    score = dp_dz * (
        probability ** DPD_BETA - complement ** DPD_BETA
        - target * probability ** (DPD_BETA - 1.0)
        + (1.0 - target) * complement ** (DPD_BETA - 1.0)
    )
    weighted_score = weight * score
    gradient = np.r_[matrix.T @ weighted_score + L2_PENALTY * coefficient, np.sum(weighted_score)]
    require(np.isfinite(objective) and np.isfinite(gradient).all(), "V111 density-power objective is non-finite")
    return objective, gradient


def density_power_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = scaffold.scaffold.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    target = ordered.y.to_numpy(float)
    head_weight = np.asarray(scaffold.scaffold.duplicate_class_weights(ordered), float)
    head_weight /= float(head_weight.sum())
    initial = np.zeros(train_state.shape[1] + 1, dtype=float)
    initial_objective, initial_gradient = density_power_objective(initial, train_state, target, head_weight)
    fit = minimize(
        density_power_objective,
        initial,
        args=(train_state, target, head_weight),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_ITERATIONS, "gtol": GRADIENT_TOLERANCE, "ftol": 1e-12, "maxls": 40},
    )
    require(np.isfinite(fit.x).all() and np.isfinite(fit.fun), "V111 density-power optimizer is non-finite")
    require(int(fit.nit) < MAX_ITERATIONS, "V111 density-power optimizer reached iteration cap")
    final_objective, final_gradient = density_power_objective(fit.x, train_state, target, head_weight)
    require(final_objective <= initial_objective + 1e-10, "V111 density-power objective did not improve")
    coefficient = fit.x[:-1]
    intercept = float(fit.x[-1])
    train_probability = np.clip(expit(train_state @ coefficient + intercept), 1e-12, 1.0 - 1e-12)
    probability = np.clip(expit(target_state @ coefficient + intercept), 1e-6, 1.0 - 1e-6)
    observed_probability = np.where(target > 0.5, train_probability, 1.0 - train_probability)
    redescending_weight = observed_probability ** DPD_BETA
    target_integer = target.astype(int)
    audit = {
        "train_n": len(ordered), "target_n": len(target_frame),
        "raw_feature_n": len(FEATURES), "state_dimension": train_state.shape[1],
        "causal_numeric_only": True, "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "density_power_head": {
            "architecture": "linear Bernoulli density-power-divergence M-estimator",
            "beta": DPD_BETA,
            "l2_penalty": L2_PENALTY,
            "optimizer": "deterministic analytic-gradient L-BFGS-B",
            "optimizer_success": bool(fit.success),
            "optimizer_status": int(fit.status),
            "optimizer_message": str(fit.message),
            "iterations": int(fit.nit),
            "function_evaluations": int(fit.nfev),
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "initial_gradient_inf_norm": float(np.max(np.abs(initial_gradient))),
            "final_gradient_inf_norm": float(np.max(np.abs(final_gradient))),
            "coefficient_l1": float(np.sum(np.abs(coefficient))),
            "coefficient_l2": float(np.linalg.norm(coefficient)),
            "coefficient_sha256": array_sha256(np.r_[intercept, coefficient]),
            "observed_class_probability_quantiles": {
                str(q): float(np.quantile(observed_probability, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
            "redescending_weight_quantiles": {
                str(q): float(np.quantile(redescending_weight, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
            "down_class_n": int(np.sum(target_integer == 0)),
            "up_class_n": int(np.sum(target_integer == 1)),
        },
        "target_rows_used_for_scaling_objective_or_head": False,
        "target_labels_used": False,
        "environment_or_recent_regime_weighting": False,
        "adversarial_feature_perturbation": False,
        "trimming_or_median_of_means": False,
        "class_conditional_density": False,
        "conformal_calibration": False,
        "hyperspherical_vmf": False,
        "archetype_or_retrieval": False,
        "feature_graph_or_persistence": False,
        "symbol_context_or_kt_code": False,
        "prediction": {
            "probability_mean": float(probability.mean()),
            "probability_std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }
    return probability, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = density_power_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 density-power-divergence logit blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "component_or_head_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v111_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V111 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = density_power_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v111_model"] = selected["name"]
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
        evidence["density_power_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v111_model"] = selected["name"]
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
            f"[V111 DPD] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V111 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V111 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V111 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V111 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(scaffold.scaffold.scaffold.roc_auc_score(target, candidate) - scaffold.scaffold.scaffold.roc_auc_score(target, base)))
        values["balanced_accuracy_delta"].append(float(
            scaffold.scaffold.scaffold.balanced_accuracy_score(target, candidate >= 0.5)
            - scaffold.scaffold.scaffold.balanced_accuracy_score(target, base >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(base >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V111 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V111 density-power-divergence logit policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V111 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
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
    density_power_valid = all(
        audit[side]["density_power_head"]["beta"] == DPD_BETA
        and audit[side]["density_power_head"]["optimizer_success"] is True
        and audit[side]["density_power_head"]["iterations"] < MAX_ITERATIONS
        and audit[side]["density_power_head"]["final_objective"] <= audit[side]["density_power_head"]["initial_objective"]
        and audit[side]["target_rows_used_for_scaling_objective_or_head"] is False
        and audit[side]["target_labels_used"] is False
        and audit[side]["environment_or_recent_regime_weighting"] is False
        and audit[side]["adversarial_feature_perturbation"] is False
        and audit[side]["class_conditional_density"] is False
        and audit[side]["conformal_calibration"] is False
        and audit[side]["hyperspherical_vmf"] is False
        and audit[side]["archetype_or_retrieval"] is False
        and audit[side]["feature_graph_or_persistence"] is False
        and audit[side]["symbol_context_or_kt_code"] is False
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
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
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "duplicate_balanced_density_power_divergence_contract_verified": density_power_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V111 fallback is not exact entire V69")
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
    return {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V111", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V111_DIAGNOSTIC_DENSITY_POWER_DIVERGENCE_LOGIT": evaluation["candidate_summary"],
                "V111_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V111", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V111", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V111_DENSITY_POWER_DIVERGENCE_LOGIT_REPORT.json": {
            "version": "V111", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "density_power_beta": DPD_BETA,
            "l2_penalty": L2_PENALTY, "maximum_iterations": MAX_ITERATIONS,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V111", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V111_DENSITY_POWER_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V111_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.smoke_test and not args.full_run and args.output is None, "explicit mode cannot accompany controller output")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    runtime = scaffold.scaffold.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_or_explicit_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "state_dimension": 2 * len(FEATURES),
            "architecture": "duplicate/class-balanced fixed-beta Bernoulli density-power-divergence linear direction M-estimator with analytic gradient and fixed L2 penalty",
            "density_power_beta": DPD_BETA,
            "l2_penalty": L2_PENALTY,
            "maximum_iterations": MAX_ITERATIONS,
            "causal_numeric_only": True,
            "environment_or_recent_regime_weighting": False,
            "adversarial_feature_perturbation": False,
            "class_conditional_density": False,
            "conformal_calibration": False,
            "hyperspherical_vmf": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "source_transfer": True, "exact_entire_v69_fallback": True,
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
