"""V94 chronological geometric-median block-logit direction challenger.

Every embargoed same-market past cut robustly scales causal pre-decision numeric
state, divides the ordered past into five fixed chronological blocks, and fits
one fixed ridge-logistic coefficient vector per block.  A deterministic
Weiszfeld geometric median aggregates the five estimators into one static
direction head.  The material hypothesis is robustness to a minority of
temporally contaminated coefficient estimates, not worst-environment loss,
residual-anchor invariance, a dynamic posterior, local adversarial margins,
knockoff selection, kernels, class densities, DTW, bilinear terms, or TAN.

Earlier same-market committed V69 OOF labels alone select exact V69 or fixed
0.25/0.50 logit blends.  Outer labels are evaluation-only, every split has a
strict 35-minute embargo plus event-group purge, and V69 confidence/high_conf
remain frozen.  Material failure restores the entire atomic V69 frame.  Audit
and one bounded two-thread CPU smoke write no output.
"""
from __future__ import annotations

import os

_thread_default = "32" if os.environ.get("MARKET_BIO_VERSION_OUTPUT") else "2"
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = _thread_default

import argparse
import ctypes
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
DEV_PATH = scaffold.DEV_PATH
V69_DIR = scaffold.V69_DIR
V69_PATH = scaffold.V69_PATH
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "research" / "staging" / "V94_CHRONOLOGICAL_GEOMETRIC_MEDIAN_BLOCK_LOGIT_V1"
)

VERSION = 94
HYPOTHESIS = "CHRONOLOGICAL_GEOMETRIC_MEDIAN_BLOCK_LOGIT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

CHRONOLOGICAL_BLOCKS = 5
RIDGE_LOGISTIC_C = 0.50
ROBUST_CLIP = 6.0
MIN_BLOCK_ROWS = 80
MIN_BLOCK_CLASS_ROWS = 15
WEISZFELD_MAX_ITERATIONS = 250
WEISZFELD_TOLERANCE = 1e-10
SEED = 9401
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = scaffold.COST

ARCHITECTURES = (
    {"name": "GEOMETRIC_MEDIAN_BLOCK_LOGIT_W0.25", "weight": 0.25},
    {"name": "GEOMETRIC_MEDIAN_BLOCK_LOGIT_W0.50", "weight": 0.50},
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


def configure_bounded_runtime() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    controller_full = bool(os.environ.get("MARKET_BIO_VERSION_OUTPUT"))
    cpu_ids = list(range(logical)) if controller_full else list(range(max(0, logical - 2), logical))
    if os.name == "nt":
        mask = sum(1 << cpu_id for cpu_id in cpu_ids)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        require(bool(kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(mask))), "V94 bounded CPU affinity failed")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
    return {
        "logical_cpu_count": logical,
        "affinity_cpu_ids": cpu_ids,
        "numeric_thread_cap": logical if controller_full else SMOKE_THREADS,
        "gpu_used": False,
        "concurrent_full_run_interference_guard": not controller_full,
    }


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic atomic authority, chronology, metric, and atomic IO utilities only; no V85 output is read",
    }
    return audit


class BlockNumericState:
    def fit(self, frame: pd.DataFrame) -> "BlockNumericState":
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        self.median = np.nanmedian(raw, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        lower = np.nanquantile(raw, 0.25, axis=0)
        upper = np.nanquantile(raw, 0.75, axis=0)
        scale = upper - lower
        fallback = np.nanstd(raw, axis=0)
        self.scale = np.where(
            np.isfinite(scale) & (scale > 1e-9), scale,
            np.where(np.isfinite(fallback) & (fallback > 1e-9), fallback, 1.0),
        )
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V94 scaler fit failed")
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        scaled = np.clip((filled - self.median) / self.scale, -ROBUST_CLIP, ROBUST_CLIP)
        matrix = np.column_stack([scaled, (~observed).astype(float)]).astype(np.float64)
        require(np.isfinite(matrix).all(), "V94 numeric transform failed")
        return matrix, {
            "rows": len(frame), "matrix_columns": matrix.shape[1],
            "missing_rate_mean": float((~observed).mean()),
            "missing_rate_max": float((~observed).mean(axis=0).max()),
        }


def duplicate_class_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(count, 1.0)
    target = frame.y.to_numpy(int)
    mass = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require((mass > 0).all(), "V94 block has one class")
    total = float(weight.sum())
    factor = np.where(target == 1, total / (2.0 * mass[1]), total / (2.0 * mass[0]))
    result = weight * factor
    return result / max(float(result.mean()), 1e-12)


def chronological_block_indices(rows: int) -> list[np.ndarray]:
    require(rows >= CHRONOLOGICAL_BLOCKS * MIN_BLOCK_ROWS, "V94 chronological block support insufficient")
    blocks = [np.asarray(index, int) for index in np.array_split(np.arange(rows), CHRONOLOGICAL_BLOCKS)]
    require(all(len(index) >= MIN_BLOCK_ROWS for index in blocks), "V94 block too small")
    return blocks


def fit_block_logit(matrix: np.ndarray, frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    target = frame.y.to_numpy(int)
    counts = np.bincount(target, minlength=2)
    require(int(counts.min()) >= MIN_BLOCK_CLASS_ROWS, "V94 block class support insufficient")
    model = LogisticRegression(
        penalty="l2", C=RIDGE_LOGISTIC_C, solver="lbfgs", max_iter=500,
        fit_intercept=True, random_state=seed, tol=1e-8,
    )
    model.fit(matrix, target, sample_weight=duplicate_class_weights(frame))
    coefficient = np.concatenate([model.intercept_.astype(float), model.coef_.ravel().astype(float)])
    require(np.isfinite(coefficient).all(), "V94 block coefficient non-finite")
    return coefficient, {
        "rows": len(frame),
        "class_0_n": int(counts[0]), "class_1_n": int(counts[1]),
        "iterations": int(model.n_iter_[0]),
        "coefficient_l2": float(np.linalg.norm(coefficient)),
        "coefficient_sha256": array_sha256(coefficient),
    }


def geometric_median(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(points.ndim == 2 and len(points) == CHRONOLOGICAL_BLOCKS, "V94 coefficient points invalid")
    arithmetic = points.mean(axis=0)
    current = arithmetic.copy()
    converged = False
    iterations = 0
    for iteration in range(1, WEISZFELD_MAX_ITERATIONS + 1):
        distance = np.linalg.norm(points - current[None, :], axis=1)
        nearest = int(np.argmin(distance))
        if float(distance[nearest]) <= WEISZFELD_TOLERANCE:
            current = points[nearest].copy()
            converged = True
            iterations = iteration
            break
        inverse = 1.0 / np.maximum(distance, WEISZFELD_TOLERANCE)
        updated = np.sum(points * inverse[:, None], axis=0) / inverse.sum()
        iterations = iteration
        if float(np.linalg.norm(updated - current)) <= WEISZFELD_TOLERANCE:
            current = updated
            converged = True
            break
        current = updated
    geometric_objective = float(np.linalg.norm(points - current[None, :], axis=1).sum())
    arithmetic_objective = float(np.linalg.norm(points - arithmetic[None, :], axis=1).sum())
    require(converged, "V94 Weiszfeld did not converge")
    require(geometric_objective <= arithmetic_objective + 1e-9, "V94 geometric median objective invalid")
    return current, {
        "algorithm": "deterministic Weiszfeld geometric median",
        "iterations": iterations, "converged": converged,
        "geometric_objective": geometric_objective,
        "arithmetic_mean_objective": arithmetic_objective,
        "objective_ratio_vs_arithmetic_mean": float(geometric_objective / max(arithmetic_objective, 1e-12)),
        "distance_to_block_coefficients": np.linalg.norm(points - current[None, :], axis=1).tolist(),
        "coefficient_dispersion_max": float(np.linalg.norm(points - current[None, :], axis=1).max()),
        "geometric_coefficient_sha256": array_sha256(current),
    }


class ChronologicalGeometricMedianLogit:
    def fit(self, frame: pd.DataFrame) -> "ChronologicalGeometricMedianLogit":
        train = frame.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
        require(len(train) >= CHRONOLOGICAL_BLOCKS * MIN_BLOCK_ROWS, "V94 past support insufficient")
        self.processor = BlockNumericState().fit(train)
        matrix, transform_audit = self.processor.transform(train)
        block_indices = chronological_block_indices(len(train))
        coefficients: list[np.ndarray] = []
        block_audits: list[dict[str, Any]] = []
        for block_number, index in enumerate(block_indices):
            coefficient, audit = fit_block_logit(
                matrix[index], train.iloc[index].copy(), SEED + block_number,
            )
            coefficients.append(coefficient)
            block_audits.append({
                **audit, "block": block_number,
                "start_time": pd.Timestamp(train.event_time_utc.iloc[index[0]]),
                "end_time": pd.Timestamp(train.event_time_utc.iloc[index[-1]]),
            })
        coefficient_matrix = np.vstack(coefficients)
        self.coefficient, median_audit = geometric_median(coefficient_matrix)
        self.audit = {
            "train_n": len(train), "feature_n": len(FEATURES),
            "state_dimension": matrix.shape[1],
            "causal_numeric_prediction_inputs_only": True,
            "categorical_text_source_ticker_features": 0,
            "chronological_block_count": CHRONOLOGICAL_BLOCKS,
            "block_boundaries_fit_on_training_cut_only": True,
            "target_rows_used_in_scaler_blocks_or_fit": False,
            "block_estimators_are_fixed_ridge_logistic": True,
            "geometric_median_static_aggregate": True,
            "worst_environment_optimization": False,
            "dynamic_coefficient_updates": False,
            "train_transform": transform_audit,
            "block_audits": block_audits,
            "geometric_median": median_audit,
        }
        return self

    def predict_probability(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        matrix, transform_audit = self.processor.transform(frame)
        score = self.coefficient[0] + matrix @ self.coefficient[1:]
        probability = np.clip(expit(np.clip(score, -12.0, 12.0)), 1e-5, 1.0 - 1e-5)
        require(np.isfinite(probability).all(), "V94 probability failed")
        return probability, {
            "target_n": len(frame), "target_transform": transform_audit,
            "target_rows_used_in_fit": False,
            "target_rows_used_to_form_blocks": False,
            "score_mean": float(score.mean()), "score_std": float(score.std()),
            "probability_mean": float(probability.mean()),
            "probability_sha256": array_sha256(probability),
        }


def geometric_block_direction(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model = ChronologicalGeometricMedianLogit().fit(train)
    probability, prediction = model.predict_probability(target)
    return probability, {**model.audit, "prediction": prediction}


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = geometric_block_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]),
    }]
    aggregate_valid = bool(
        model_audit["geometric_median"]["converged"]
        and model_audit["geometric_median"]["geometric_objective"]
        <= model_audit["geometric_median"]["arithmetic_mean_objective"] + 1e-9
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
            "geometric_aggregate_valid": aggregate_valid,
            "eligible": bool(aggregate_valid and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 geometric-block blend; maximize 2*AUC+BA with fixed aggregate/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "block_count_ridge_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v94_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V94 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = geometric_block_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v94_model"] = selected["name"]
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
        evidence["geometric_block_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v94_model"] = selected["name"]
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
            f"[V94 GEOMED] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V94 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V94 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V94 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V94 bootstrap blocks")
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
        auc_delta.append(float(scaffold.roc_auc_score(target, candidate) - scaffold.roc_auc_score(target, baseline)))
        ba_delta.append(float(
            scaffold.balanced_accuracy_score(target, candidate >= 0.5)
            - scaffold.balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V94 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked geometric-block blend",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V94/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    geometric_valid = all(
        audit["inner_model_audit"]["geometric_median"]["converged"]
        and audit["outer_model_audit"]["geometric_median"]["converged"]
        and audit["inner_model_audit"]["geometric_median"]["geometric_objective"] <= audit["inner_model_audit"]["geometric_median"]["arithmetic_mean_objective"] + 1e-9
        and audit["outer_model_audit"]["geometric_median"]["geometric_objective"] <= audit["outer_model_audit"]["geometric_median"]["arithmetic_mean_objective"] + 1e-9
        and audit["inner_model_audit"]["prediction"]["target_rows_used_to_form_blocks"] is False
        and audit["outer_model_audit"]["prediction"]["target_rows_used_to_form_blocks"] is False
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
        "chronological_geometric_median_aggregate_verified": geometric_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V94 fallback is not exact complete V69")
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
            "activated": not material_pass,
            "policy": "exact V69 entire frame" if not material_pass else None,
            "exact_frame_verified": bool(material_pass or selected.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def write_outputs(
    authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any], out: Path,
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
            "version": "V94", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V94_DIAGNOSTIC_GEOMETRIC_BLOCK": evaluation["candidate_summary"],
                "V94_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact, "authority_audit": authority, "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V94", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V94", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
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
        "CHRONOLOGICAL_GEOMETRIC_MEDIAN_BLOCK_LOGIT_REPORT.json": {
            "version": "V94", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "architectures": ARCHITECTURES,
            "chronological_blocks": CHRONOLOGICAL_BLOCKS,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V94", "status": "MATCH", "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False,
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
        "geometric_objective_ratio": audit["outer_model_audit"]["geometric_median"]["objective_ratio_vs_arithmetic_mean"],
        "coefficient_dispersion_max": audit["outer_model_audit"]["geometric_median"]["coefficient_dispersion_max"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V94_GEOMETRIC_BLOCK_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V94_GEOMETRIC_BLOCK_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V94_DIAGNOSTIC_GEOMETRIC_BLOCK_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V94_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    runtime = configure_bounded_runtime()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "feature_n": len(FEATURES),
            "causal_numeric_prediction_inputs_only": True,
            "chronological_block_count": CHRONOLOGICAL_BLOCKS,
            "fixed_ridge_logistic_c": RIDGE_LOGISTIC_C,
            "aggregate": "deterministic Weiszfeld geometric median of coefficients",
            "architectures": ARCHITECTURES,
            "canonical_controller_gate_checks": 14,
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
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
        summary["bootstrap"] = evaluation["bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
