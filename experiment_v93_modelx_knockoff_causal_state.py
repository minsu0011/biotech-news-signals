"""V93 second-order model-X knockoff causal-state direction challenger.

Within each embargoed same-market training cut, 37 causal pre-decision numeric
states are robustly standardized.  Deterministic-seed second-order Gaussian
model-X knockoffs reproduce the unlabeled training covariance while serving as
negative-control features.  One fixed L1 logistic fit on originals plus their
knockoffs yields antisymmetric |beta|-|beta_tilde| statistics.  A predeclared
knockoff+ FDR threshold selects original variables; only those originals enter
a fixed ridge-logistic direction refit.  No validation/outer row participates
in scaling, covariance, knockoff generation, selection, or fitting.

This negative-control variable-selection architecture is distinct from stable
correlation ranking, adversarial margin training, residual invariance, RFF,
entropy weighting, copula QDA, DTW prototypes, dynamic Bayesian filtering,
bilinear factors, and quantile TAN.  Inner-past committed V69 OOF labels alone
select exact V69 or fixed 0.25/0.50 logit blends.  Outer labels are evaluation-
only; V69 confidence/high_conf are frozen; the controller recomputes all 14
canonical gates; material failure restores the exact entire V69 frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

_thread_default = "32" if os.environ.get("MARKET_BIO_VERSION_OUTPUT") else "2"
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = _thread_default

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "staging" / "V93_MODELX_KNOCKOFF_CAUSAL_STATE_V1"
)
VERSION = 93
HYPOTHESIS = "SECOND_ORDER_MODELX_KNOCKOFF_CAUSAL_STATE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

COVARIANCE_SHRINKAGE = 0.20
EQUI_S_FRACTION = 0.90
KNOCKOFF_L1_C = 0.25
KNOCKOFF_FDR_Q = 0.50
KNOCKOFF_OFFSET = 1
REFIT_RIDGE_C = 0.50
SEED = 9301
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = scaffold.COST

ARCHITECTURES = (
    {"name": "MODELX_KNOCKOFF_W0.25", "weight": 0.25},
    {"name": "MODELX_KNOCKOFF_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
safe_auc = scaffold.safe_auc
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
chronology = scaffold.chronology
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
inner_partition = scaffold.inner_partition
blend_probability = scaffold.blend_probability
controller = scaffold.controller
v44 = scaffold.v44


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic-authority, chronology, blending and IO utilities only; no V85 output is read",
    }
    return audit


def duplicate_class_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    target = frame.y.to_numpy(int)
    mass = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require((mass > 0).all(), "knockoff fit has one class")
    total = float(weight.sum())
    factor = np.where(target == 1, total / (2.0 * mass[1]), total / (2.0 * mass[0]))
    result = weight * factor
    return result / result.mean()


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    return weight / weight.sum()


def second_order_knockoffs(
    matrix: np.ndarray, unlabeled_weight: np.ndarray, random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    weight = np.asarray(unlabeled_weight, float)
    weight /= weight.sum()
    mean = np.sum(weight[:, None] * matrix, axis=0)
    centered = matrix - mean
    covariance = centered.T @ (weight[:, None] * centered)
    diagonal = np.diag(np.maximum(np.diag(covariance), 1e-6))
    covariance = (1.0 - COVARIANCE_SHRINKAGE) * covariance + COVARIANCE_SHRINKAGE * diagonal
    covariance += np.eye(matrix.shape[1]) * 1e-6
    eigenvalues = np.linalg.eigvalsh(covariance)
    require(float(eigenvalues.min()) > 0.0, "knockoff covariance is not positive definite")
    equi_s = EQUI_S_FRACTION * min(1.0, 2.0 * float(eigenvalues.min()))
    require(equi_s > 1e-8, "knockoff equi-s is degenerate")
    s_matrix = np.eye(matrix.shape[1]) * equi_s
    covariance_inverse = np.linalg.inv(covariance)
    conditional_covariance = 2.0 * s_matrix - s_matrix @ covariance_inverse @ s_matrix
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    conditional_eigen = np.linalg.eigvalsh(conditional_covariance)
    if float(conditional_eigen.min()) < 1e-10:
        conditional_covariance += np.eye(matrix.shape[1]) * (1e-10 - float(conditional_eigen.min()))
    factor = np.linalg.cholesky(conditional_covariance)
    generator = np.random.default_rng(random_seed)
    noise = generator.standard_normal(matrix.shape)
    knockoff = mean + centered @ (np.eye(matrix.shape[1]) - covariance_inverse @ s_matrix) + noise @ factor.T
    require(np.isfinite(knockoff).all(), "generated knockoff is non-finite")
    original_cov = np.cov(matrix, rowvar=False)
    knockoff_cov = np.cov(knockoff, rowvar=False)
    covariance_relative_error = float(
        np.linalg.norm(original_cov - knockoff_cov, ord="fro")
        / max(np.linalg.norm(original_cov, ord="fro"), 1e-12)
    )
    return knockoff, {
        "generator": "second-order Gaussian model-X equi-correlated knockoff",
        "random_seed": random_seed,
        "covariance_shrinkage": COVARIANCE_SHRINKAGE,
        "equi_s": equi_s,
        "original_covariance_min_eigenvalue": float(eigenvalues.min()),
        "conditional_covariance_min_eigenvalue": float(np.linalg.eigvalsh(conditional_covariance).min()),
        "empirical_covariance_relative_error": covariance_relative_error,
        "uses_labels": False, "uses_target_rows": False,
        "knockoff_sha256": array_sha256(knockoff),
    }


def knockoff_threshold(statistic: np.ndarray) -> float | None:
    candidates = np.unique(np.abs(statistic[np.abs(statistic) > 1e-12]))
    for threshold in np.sort(candidates):
        false_count = KNOCKOFF_OFFSET + int(np.sum(statistic <= -threshold))
        discovery_count = max(1, int(np.sum(statistic >= threshold)))
        if false_count / discovery_count <= KNOCKOFF_FDR_Q:
            return float(threshold)
    return None


def knockoff_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V93 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V93 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_matrix = processor.transform(train)
    target_matrix = processor.transform(target_frame)
    seed = SEED + len(train) * 17 + (0 if str(train.market.iloc[0]) == "US" else 1)
    knockoff, generator_audit = second_order_knockoffs(
        train_matrix, duplicate_weights(train), seed,
    )
    augmented = np.column_stack([train_matrix, knockoff])
    target = train.y.to_numpy(int)
    sample_weight = duplicate_class_weights(train)
    selector = LogisticRegression(
        penalty="l1", C=KNOCKOFF_L1_C, solver="liblinear", fit_intercept=True,
        max_iter=2000, random_state=SEED,
    )
    selector.fit(augmented, target, sample_weight=sample_weight)
    coefficient = selector.coef_.reshape(-1)
    feature_n = len(FEATURES)
    original_coefficient = coefficient[:feature_n]
    knockoff_coefficient = coefficient[feature_n:]
    statistic = np.abs(original_coefficient) - np.abs(knockoff_coefficient)
    threshold = knockoff_threshold(statistic)
    selected = np.flatnonzero(statistic >= threshold) if threshold is not None else np.empty(0, dtype=int)
    available = len(selected) > 0
    if available:
        refit = LogisticRegression(
            penalty="l2", C=REFIT_RIDGE_C, solver="liblinear", fit_intercept=True,
            max_iter=2000, random_state=SEED,
        )
        refit.fit(train_matrix[:, selected], target, sample_weight=sample_weight)
        probability = refit.predict_proba(target_matrix[:, selected])[:, 1]
        refit_iterations = int(refit.n_iter_[0])
        refit_coefficient_n = len(selected) + 1
    else:
        probability = np.full(len(target_frame), 0.5, dtype=float)
        refit_iterations = 0
        refit_coefficient_n = 0
    probability = np.clip(np.asarray(probability, float), 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V93 probability is non-finite")
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train),
        "target_n": len(target_frame), "feature_n": feature_n,
        "causal_numeric_only": True, "categorical_feature_n": 0,
        "text_feature_n": 0, "source_feature": False, "ticker_feature": False,
        "architecture": "second-order model-X knockoff filter plus selected-original ridge refit",
        "generator_audit": generator_audit,
        "selection_uses_training_labels_only": True,
        "target_labels_used": False, "target_rows_used_for_selection": False,
        "l1_selector_c": KNOCKOFF_L1_C,
        "selector_iterations": int(selector.n_iter_[0]),
        "knockoff_fdr_q": KNOCKOFF_FDR_Q, "knockoff_offset": KNOCKOFF_OFFSET,
        "threshold": threshold, "model_available": available,
        "selected_feature_n": len(selected),
        "selected_features": [FEATURES[index] for index in selected],
        "selected_statistics": {FEATURES[index]: float(statistic[index]) for index in selected},
        "original_nonzero_n": int(np.sum(np.abs(original_coefficient) > 1e-12)),
        "knockoff_nonzero_n": int(np.sum(np.abs(knockoff_coefficient) > 1e-12)),
        "original_win_n": int(np.sum(statistic > 0.0)),
        "knockoff_win_n": int(np.sum(statistic < 0.0)),
        "statistic_min": float(statistic.min()), "statistic_max": float(statistic.max()),
        "statistic_sha256": array_sha256(statistic),
        "refit_ridge_c": REFIT_RIDGE_C,
        "refit_iterations": refit_iterations,
        "refit_coefficient_n": refit_coefficient_n,
        "probability_sha256": array_sha256(probability),
    }


def metric(
    frame: pd.DataFrame, probability: np.ndarray,
    confidence: np.ndarray | pd.Series, high: np.ndarray | pd.Series,
) -> dict[str, Any]:
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    target = frame.y.to_numpy(int)
    confidence_array = np.asarray(confidence, float)
    high_array = np.asarray(high, bool)
    correct = (prediction == target).astype(int)
    signed_net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return {
        "n": len(frame), "auc": float(roc_auc_score(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "accuracy": float(correct.mean()), "pred_up": float(prediction.mean()),
        "all_trade_mean_signed_net": float(signed_net.mean()),
        "confidence_correctness_auc": safe_auc(correct, confidence_array),
        "highconf_n": int(high_array.sum()),
        "highconf_accuracy": float(correct[high_array].mean()) if high_array.any() else None,
        "highconf_mean_signed_net": float(signed_net[high_array].mean()) if high_array.any() else None,
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = knockoff_direction(inner_train, inner_valid)
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
        model_audit["model_available"]
        and model_audit["selected_feature_n"] > 0
        and not model_audit["generator_audit"]["uses_labels"]
        and not model_audit["generator_audit"]["uses_target_rows"]
        and not model_audit["target_labels_used"]
        and not model_audit["target_rows_used_for_selection"]
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
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 model-X knockoff logit blend; maximize 2*AUC+BA+10*net with fixed availability/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v93_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V93 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = knockoff_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        deployment_fallback = bool(selected["architecture"] is not None and not outer_model_audit["model_available"])
        candidate = baseline.copy() if selected["architecture"] is None or deployment_fallback else blend_probability(
            baseline, challenger, selected["architecture"]["weight"],
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v93_model"] = "V69_NOOP_OUTER_MODEL_UNAVAILABLE" if deployment_fallback else selected["name"]
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
        evidence["knockoff_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v93_model"] = "V69_NOOP_OUTER_MODEL_UNAVAILABLE" if deployment_fallback else selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_model_unavailable_fallback": deployment_fallback,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V93 KNOCKOFF] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V93 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V93 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V93 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V93 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, base)))
        ba_delta.append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, base >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(base >= 0.5, 1.0, -1.0) * returns)))
    require(len(auc_delta) >= int(0.90 * draws), "V93 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked model-X knockoff policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V93/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    knockoff_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and not audit["outer_model_audit"]["generator_audit"]["uses_labels"]
        and not audit["outer_model_audit"]["generator_audit"]["uses_target_rows"]
        and audit["outer_model_audit"]["selection_uses_training_labels_only"]
        and not audit["outer_model_audit"]["target_labels_used"]
        and not audit["outer_model_audit"]["target_rows_used_for_selection"]
        and audit["outer_model_audit"]["generator_audit"]["conditional_covariance_min_eigenvalue"] > 0.0
        and not audit["outer_model_unavailable_fallback"]
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
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "modelx_knockoff_contract_verified": knockoff_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V93 fallback is not exact complete V69")
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
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V69 entire frame" if not material_pass else None,
            "exact_frame_verified": bool(material_pass or selected.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def write_outputs(
    authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]],
    evaluation: dict[str, Any], out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected_contract = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
    }
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V93", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V93_DIAGNOSTIC_MODELX_KNOCKOFF": evaluation["candidate_summary"],
                "V93_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V93", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V93", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected_contract["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "net_delta": evaluation["outer_net_delta"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "MODELX_KNOCKOFF_CAUSAL_STATE_REPORT.json": {
            "version": "V93", "hypothesis": HYPOTHESIS, "features": FEATURES,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V93", "status": "MATCH", "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
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
        "selected_feature_n": audit["outer_model_audit"]["selected_feature_n"],
        "outer_model_unavailable_fallback": audit["outer_model_unavailable_fallback"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V93_MODELX_KNOCKOFF_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V93_MODELX_KNOCKOFF_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V93_DIAGNOSTIC_MODELX_KNOCKOFF_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V93_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


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
            "architecture": "second-order model-X knockoff negative controls plus FDR filter and selected-original ridge refit",
            "knockoff_fdr_q": KNOCKOFF_FDR_Q, "knockoff_offset": KNOCKOFF_OFFSET,
            "target_rows_used_for_selection": False,
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_cpu_thread_cap": SMOKE_THREADS, "gpu_model_calls": 0,
                "reason": "bounded deterministic CPU smoke avoids contention with other full runners",
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
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
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
        summary["inner_knockoff_audit"] = audits[0]["inner_model_audit"]
        summary["outer_knockoff_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
