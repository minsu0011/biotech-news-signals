"""V114 duplicate-weighted nonnegative parts-factor direction challenger.

Every embargoed same-market train cut is represented only by robustly scaled
causal numeric state and deterministic missingness flags.  The numeric channels
are mapped by one fixed affine transform from the robust clipping interval to
the unit interval.  A label-free, duplicate-weighted Euclidean nonnegative
matrix factorization learns twelve additive parts by deterministic multiplicative
updates.  Target rows only solve nonnegative activations against the frozen past
basis, and one fixed ridge-logistic direction head uses those activations.  This
is not a neural autoencoder, ICA independence model, convex archetype simplex,
signed-parts class density, graph, conformal typicality, Kendall tournament,
rough-path signature, retrieval, tree, or kernel model.

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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v102_ordered_horizon_fused_lasso_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V114_DUPLICATE_WEIGHTED_NONNEGATIVE_PARTS_FACTOR_V1"
VERSION = 114
HYPOTHESIS = "DUPLICATE_WEIGHTED_NONNEGATIVE_PARTS_FACTOR_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v102_ordered_horizon_fused_lasso_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "d398c651102ff2e47e747d277aca5ee24718f0ae29013b3fb02f0bfc07fba3b2"

FACTOR_RANK = 12
NMF_TRAIN_ITERATIONS = 96
NMF_TRANSFORM_ITERATIONS = 64
NMF_L2 = 0.01
NMF_EPSILON = 1e-10
RIDGE_LOGISTIC_C = 0.50
SEED = 11401
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "NONNEGATIVE_PARTS_FACTOR_LOGIT_W0.25", "weight": 0.25},
    {"name": "NONNEGATIVE_PARTS_FACTOR_LOGIT_W0.50", "weight": 0.50},
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


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(size, 1.0)
    require(np.isfinite(weight).all() and float(weight.sum()) > 0.0, "V114 duplicate weights invalid")
    return weight / float(weight.sum())


def nonnegative_state(matrix: np.ndarray) -> np.ndarray:
    """Fixed affine map; no target-fitted minimum or maximum is permitted."""
    require(matrix.ndim == 2 and matrix.shape[1] == 2 * len(FEATURES), "V114 robust-state dimension changed")
    numeric = np.clip((matrix[:, :len(FEATURES)] + 7.0) / 14.0, 0.0, 1.0)
    missing = np.clip(matrix[:, len(FEATURES):], 0.0, 1.0)
    output = np.c_[numeric, missing]
    require(np.isfinite(output).all() and float(output.min()) >= 0.0, "V114 nonnegative state invalid")
    return output


def weighted_nmf_objective(
    matrix: np.ndarray,
    activation: np.ndarray,
    basis: np.ndarray,
    weight: np.ndarray,
) -> float:
    residual = matrix - activation @ basis
    reconstruction = np.sum(weight[:, None] * residual * residual)
    regularization = NMF_L2 * (
        np.sum(weight[:, None] * activation * activation) + np.mean(weight) * np.sum(basis * basis)
    )
    return float(reconstruction + regularization)


def fit_weighted_nmf(
    matrix: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train-only duplicate-weighted Euclidean NMF with fixed multiplicative updates."""
    require(matrix.ndim == 2 and len(matrix) == len(weight), "V114 weighted NMF shape mismatch")
    generator = np.random.default_rng(SEED)
    activation = generator.uniform(0.25, 0.75, size=(len(matrix), FACTOR_RANK))
    basis = generator.uniform(0.25, 0.75, size=(FACTOR_RANK, matrix.shape[1]))
    trace = [weighted_nmf_objective(matrix, activation, basis, weight)]
    weighted_matrix = weight[:, None] * matrix
    for iteration in range(1, NMF_TRAIN_ITERATIONS + 1):
        numerator_basis = activation.T @ weighted_matrix
        denominator_basis = activation.T @ (weight[:, None] * (activation @ basis))
        denominator_basis += NMF_L2 * np.mean(weight) * basis + NMF_EPSILON
        basis *= numerator_basis / denominator_basis
        basis = np.maximum(basis, NMF_EPSILON)

        numerator_activation = matrix @ basis.T
        denominator_activation = activation @ (basis @ basis.T) + NMF_L2 * activation + NMF_EPSILON
        activation *= numerator_activation / denominator_activation
        activation = np.maximum(activation, NMF_EPSILON)
        if iteration in {NMF_TRAIN_ITERATIONS // 4, NMF_TRAIN_ITERATIONS // 2, 3 * NMF_TRAIN_ITERATIONS // 4, NMF_TRAIN_ITERATIONS}:
            trace.append(weighted_nmf_objective(matrix, activation, basis, weight))
    require(np.isfinite(activation).all() and np.isfinite(basis).all(), "V114 weighted NMF is non-finite")
    require(trace[-1] <= trace[0] + 1e-10, "V114 weighted NMF objective did not improve")
    reconstruction = activation @ basis
    audit = {
        "rank": FACTOR_RANK,
        "train_iterations": NMF_TRAIN_ITERATIONS,
        "l2": NMF_L2,
        "objective_trace": trace,
        "objective_relative_reduction": float((trace[0] - trace[-1]) / max(abs(trace[0]), NMF_EPSILON)),
        "weighted_reconstruction_rmse": float(np.sqrt(np.sum(weight[:, None] * (matrix - reconstruction) ** 2) / matrix.shape[1])),
        "basis_zero_fraction_le_1e_6": float(np.mean(basis <= 1e-6)),
        "activation_zero_fraction_le_1e_6": float(np.mean(activation <= 1e-6)),
        "basis_sha256": array_sha256(basis),
        "train_activation_sha256": array_sha256(activation),
        "representation_labels_used": False,
    }
    return activation, basis, audit


def transform_nmf(matrix: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve nonnegative activations with the past-only basis frozen."""
    scale = np.maximum(np.mean(basis, axis=1), NMF_EPSILON)
    activation = np.tile(np.mean(matrix, axis=1, keepdims=True), (1, FACTOR_RANK))
    activation /= (FACTOR_RANK * scale[None, :])
    activation = np.maximum(activation, NMF_EPSILON)
    initial_residual = matrix - activation @ basis
    initial_objective = float(np.mean(initial_residual * initial_residual) + NMF_L2 * np.mean(activation * activation))
    gram = basis @ basis.T
    for _ in range(NMF_TRANSFORM_ITERATIONS):
        numerator = matrix @ basis.T
        denominator = activation @ gram + NMF_L2 * activation + NMF_EPSILON
        activation *= numerator / denominator
        activation = np.maximum(activation, NMF_EPSILON)
    residual = matrix - activation @ basis
    final_objective = float(np.mean(residual * residual) + NMF_L2 * np.mean(activation * activation))
    require(np.isfinite(activation).all() and final_objective <= initial_objective + 1e-10, "V114 frozen-basis transform invalid")
    return activation, {
        "iterations": NMF_TRANSFORM_ITERATIONS,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "reconstruction_rmse": float(np.sqrt(np.mean(residual * residual))),
        "activation_zero_fraction_le_1e_6": float(np.mean(activation <= 1e-6)),
        "activation_sha256": array_sha256(activation),
        "basis_refit": False,
    }


def nonnegative_parts_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = scaffold.scaffold.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_nonnegative = nonnegative_state(train_state)
    target_nonnegative = nonnegative_state(target_state)
    representation_weight = duplicate_weights(ordered)
    train_activation, basis, nmf_audit = fit_weighted_nmf(train_nonnegative, representation_weight)
    target_activation, target_activation_audit = transform_nmf(target_nonnegative, basis)
    target = ordered.y.to_numpy(int)
    head_weight = scaffold.scaffold.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="lbfgs",
        max_iter=500, random_state=SEED, n_jobs=1,
    )
    head.fit(train_activation, target, sample_weight=head_weight)
    require(int(head.n_iter_[0]) < 500, "V114 ridge-logistic head did not converge")
    probability = np.clip(head.predict_proba(target_activation)[:, 1], 1e-6, 1.0 - 1e-6)
    audit = {
        "train_n": len(ordered), "target_n": len(target_frame),
        "raw_feature_n": len(FEATURES), "state_dimension": train_state.shape[1],
        "causal_numeric_only": True, "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "nonnegative_parts_factor": {
            **nmf_audit,
            "target_activation": target_activation_audit,
            "fixed_affine_numeric_map": "clip((robust_numeric+7)/14,0,1)",
            "missing_flags_unchanged_0_1": True,
            "target_rows_used_for_basis_fit": False,
        },
        "head": {
            "architecture": "fixed ridge logistic on 12 label-free nonnegative parts activations",
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "iterations": int(head.n_iter_[0]),
            "coefficient_sha256": array_sha256(np.r_[head.intercept_, head.coef_.ravel()]),
        },
        "target_rows_used_for_scaling_basis_or_head": False,
        "target_labels_used": False,
        "class_conditional_density": False,
        "neural_autoencoder": False,
        "ica_or_independence_objective": False,
        "archetype_simplex": False,
        "signed_parts_composition": False,
        "feature_graph_or_persistence": False,
        "symbol_context_or_kt_code": False,
        "kendall_tournament": False,
        "rough_path_signature": False,
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
    challenger, model_audit = nonnegative_parts_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 nonnegative-parts factor logit blend; fixed 2*AUC+BA score with BA/net eligibility",
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
    diagnostic["v114_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V114 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = nonnegative_parts_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v114_model"] = selected["name"]
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
        evidence["nonnegative_parts_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v114_model"] = selected["name"]
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
            f"[V114 NMF] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V114 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V114 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V114 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V114 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V114 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V114 nonnegative-parts factor logit policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V114 canonical mismatch")
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
    nonnegative_parts_valid = all(
        audit[side]["nonnegative_parts_factor"]["rank"] == FACTOR_RANK
        and audit[side]["nonnegative_parts_factor"]["train_iterations"] == NMF_TRAIN_ITERATIONS
        and audit[side]["nonnegative_parts_factor"]["objective_trace"][-1] <= audit[side]["nonnegative_parts_factor"]["objective_trace"][0]
        and audit[side]["nonnegative_parts_factor"]["representation_labels_used"] is False
        and audit[side]["nonnegative_parts_factor"]["target_rows_used_for_basis_fit"] is False
        and audit[side]["nonnegative_parts_factor"]["target_activation"]["basis_refit"] is False
        and audit[side]["target_rows_used_for_scaling_basis_or_head"] is False
        and audit[side]["target_labels_used"] is False
        and audit[side]["class_conditional_density"] is False
        and audit[side]["neural_autoencoder"] is False
        and audit[side]["ica_or_independence_objective"] is False
        and audit[side]["archetype_simplex"] is False
        and audit[side]["signed_parts_composition"] is False
        and audit[side]["feature_graph_or_persistence"] is False
        and audit[side]["symbol_context_or_kt_code"] is False
        and audit[side]["kendall_tournament"] is False
        and audit[side]["rough_path_signature"] is False
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
        "duplicate_weighted_nonnegative_parts_factor_contract_verified": nonnegative_parts_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V114 fallback is not exact entire V69")
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
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V114 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V114 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V114 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V114", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V114_DIAGNOSTIC_NONNEGATIVE_PARTS_FACTOR": evaluation["candidate_summary"],
                "V114_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V114", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V114", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V114_NONNEGATIVE_PARTS_FACTOR_REPORT.json": {
            "version": "V114", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "factor_rank": FACTOR_RANK,
            "train_iterations": NMF_TRAIN_ITERATIONS,
            "transform_iterations": NMF_TRANSFORM_ITERATIONS,
            "nmf_l2": NMF_L2, "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V114", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V114_NONNEGATIVE_PARTS_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V114_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V114 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
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
            "architecture": "duplicate-weighted label-free fixed-rank Euclidean nonnegative parts factorization with frozen-basis target activations and fixed ridge-logistic head",
            "factor_rank": FACTOR_RANK,
            "train_iterations": NMF_TRAIN_ITERATIONS,
            "transform_iterations": NMF_TRANSFORM_ITERATIONS,
            "nmf_l2": NMF_L2,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "causal_numeric_only": True,
            "class_conditional_density": False,
            "neural_autoencoder": False,
            "ica_or_independence_objective": False,
            "archetype_simplex": False,
            "kendall_tournament": False,
            "rough_path_signature": False,
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
