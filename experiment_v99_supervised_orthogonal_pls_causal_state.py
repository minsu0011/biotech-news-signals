"""V99 supervised orthogonal-PLS causal-state direction challenger.

Each strictly embargoed same-market training cut is represented only by causal
pre-decision numeric state and deterministic missingness flags.  A fixed
four-component weighted PLS1/NIPALS deflation extracts supervised latent scores
that maximize residual state/label covariance in succession.  A single fixed
ridge-logistic head maps those four train-only scores to direction probability.
This is neither unsupervised PCA nor a kernel, tree, density, retrieval, depth,
wavelet, belief-rule, dynamic, adversarial, or robust-location estimator.

Earlier committed V69 OOF rows alone choose exact V69 or one of two fixed logit
blends.  Outer labels are evaluation-only under nested chronology and a strict
35-minute embargo.  V69 confidence/high-confidence columns remain byte-value
exact.  Any material-gate failure returns the exact entire V69 frame.

Audit and one bounded two-thread US:2 smoke write no files.  A future authorized
controller may invoke this runner with no arguments and MARKET_BIO_VERSION_OUTPUT;
that path performs the full six-fold run and writes the controller contract.
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
import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
DEV_PATH = scaffold.DEV_PATH
V69_DIR = scaffold.V69_DIR
V69_PATH = scaffold.V69_PATH
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V99_SUPERVISED_ORTHOGONAL_PLS_V1"

VERSION = 99
HYPOTHESIS = "SUPERVISED_ORTHOGONAL_PLS_CAUSAL_STATE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

PLS_COMPONENTS = 4
PLS_NORM_FLOOR = 1e-12
RIDGE_LOGISTIC_C = 0.50
ROBUST_CLIP = 6.0
SEED = 9901
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = scaffold.COST

ARCHITECTURES = (
    {"name": "ORTHOGONAL_PLS_LOGIT_W0.25", "weight": 0.25},
    {"name": "ORTHOGONAL_PLS_LOGIT_W0.50", "weight": 0.50},
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
    cpu_ids = list(range(max(0, logical - 2), logical))
    if os.name == "nt":
        mask = sum(1 << cpu_id for cpu_id in cpu_ids)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        require(
            bool(kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(mask))),
            "V99 bounded CPU affinity failed",
        )
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
    return {
        "logical_cpu_count": logical,
        "affinity_cpu_ids": cpu_ids,
        "numeric_thread_cap": SMOKE_THREADS,
        "gpu_model_calls": 0,
        "concurrent_full_run_interference_guard": True,
    }


def verify_authority() -> dict[str, Any]:
    require(
        sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256,
        "pinned V85 authority/chronology scaffold changed",
    )
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, metrics, canonical gate, blending, and atomic IO only; no V85 output is read",
    }
    return audit


def duplicate_class_weights(frame: pd.DataFrame) -> np.ndarray:
    group_size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(group_size, 1.0)
    target = frame.y.to_numpy(int)
    for label in (0, 1):
        index = target == label
        require(int(index.sum()) >= 50, f"V99 class {label} support too small")
        total = float(weight[index].sum())
        require(total > 0.0, "V99 invalid class weight total")
        weight[index] *= 0.5 / total
    weight *= len(weight) / float(weight.sum())
    return weight


class RobustNumericState:
    """Train-cut-only robust numeric state with deterministic missing flags."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def raw(frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    def fit(self, frame: pd.DataFrame) -> "RobustNumericState":
        values = self.raw(frame)
        self.median = np.nanmedian(values, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        q25 = np.nanpercentile(values, 25.0, axis=0)
        q75 = np.nanpercentile(values, 75.0, axis=0)
        scale = q75 - q25
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        require(self.median is not None and self.scale is not None, "V99 state is not fit")
        values = self.raw(frame)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        numeric = np.clip((filled - self.median) / self.scale, -ROBUST_CLIP, ROBUST_CLIP)
        matrix = np.concatenate([numeric, missing.astype(float)], axis=1)
        require(np.isfinite(matrix).all(), "V99 state transform is non-finite")
        return matrix, {
            "rows": len(frame),
            "numeric_feature_n": len(FEATURES),
            "missing_flag_n": len(FEATURES),
            "matrix_columns": matrix.shape[1],
            "missing_rate_mean": float(missing.mean()),
            "missing_rate_max": float(missing.mean(axis=0).max()),
        }


def weighted_mean(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.sum(matrix * weight[:, None], axis=0) / float(np.sum(weight))


def fit_weighted_pls1(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Fixed-component weighted PLS1 with sequential NIPALS-style deflation."""
    require(matrix.ndim == 2 and len(matrix) == len(target) == len(sample_weight), "V99 PLS shape mismatch")
    weight = np.asarray(sample_weight, float)
    weight = weight / float(weight.sum())
    x_mean = weighted_mean(matrix, weight)
    y_mean = float(np.dot(weight, target))
    x_residual = matrix - x_mean
    y_residual = target.astype(float) - y_mean
    original_y_energy = float(np.dot(weight, y_residual * y_residual))
    require(original_y_energy > PLS_NORM_FLOOR, "V99 PLS response is degenerate")

    directions: list[np.ndarray] = []
    loadings: list[np.ndarray] = []
    response_loadings: list[float] = []
    scores: list[np.ndarray] = []
    component_audits: list[dict[str, Any]] = []
    for component in range(PLS_COMPONENTS):
        covariance = x_residual.T @ (weight * y_residual)
        covariance_norm = float(np.linalg.norm(covariance))
        require(covariance_norm > PLS_NORM_FLOOR, f"V99 PLS component {component} covariance degenerate")
        direction = covariance / covariance_norm
        score = x_residual @ direction
        score_energy = float(np.dot(weight, score * score))
        require(score_energy > PLS_NORM_FLOOR, f"V99 PLS component {component} score degenerate")
        loading = (x_residual.T @ (weight * score)) / score_energy
        response_loading = float(np.dot(weight, score * y_residual) / score_energy)
        before = float(np.dot(weight, y_residual * y_residual))
        x_residual = x_residual - np.outer(score, loading)
        y_residual = y_residual - response_loading * score
        after = float(np.dot(weight, y_residual * y_residual))
        require(after <= before + 1e-10, "V99 PLS response residual energy increased")
        directions.append(direction)
        loadings.append(loading)
        response_loadings.append(response_loading)
        scores.append(score)
        component_audits.append({
            "component": component,
            "residual_state_response_covariance_norm": covariance_norm,
            "weighted_score_energy": score_energy,
            "response_loading": response_loading,
            "response_residual_energy_before": before,
            "response_residual_energy_after": after,
            "direction_sha256": array_sha256(direction),
            "loading_sha256": array_sha256(loading),
        })

    score_matrix = np.column_stack(scores)
    score_scale = np.sqrt(np.sum(weight[:, None] * score_matrix * score_matrix, axis=0))
    require(np.all(score_scale > PLS_NORM_FLOOR), "V99 PLS score scale degenerate")
    standardized_scores = score_matrix / score_scale
    gram = standardized_scores.T @ (weight[:, None] * standardized_scores)
    off_diagonal = gram - np.diag(np.diag(gram))
    orthogonality_error = float(np.max(np.abs(off_diagonal)))
    require(orthogonality_error <= 1e-7, "V99 PLS scores are not weighted-orthogonal")
    model = {
        "x_mean": x_mean,
        "directions": np.stack(directions),
        "loadings": np.stack(loadings),
        "response_loadings": np.asarray(response_loadings),
        "score_scale": score_scale,
    }
    audit = {
        "algorithm": "fixed four-component weighted PLS1/NIPALS residual covariance deflation",
        "component_n": PLS_COMPONENTS,
        "component_audits": component_audits,
        "weighted_score_gram": gram.tolist(),
        "weighted_score_orthogonality_max_abs_off_diagonal": orthogonality_error,
        "response_residual_energy_initial": original_y_energy,
        "response_residual_energy_final": float(np.dot(weight, y_residual * y_residual)),
        "response_energy_explained_fraction": float(
            1.0 - np.dot(weight, y_residual * y_residual) / original_y_energy
        ),
        "directions_sha256": array_sha256(model["directions"]),
        "loadings_sha256": array_sha256(model["loadings"]),
        "target_rows_used_for_components": False,
        "target_labels_used_for_components": False,
    }
    return model, standardized_scores, audit


def transform_pls(matrix: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    residual = matrix - model["x_mean"]
    scores: list[np.ndarray] = []
    for direction, loading in zip(model["directions"], model["loadings"]):
        score = residual @ direction
        scores.append(score)
        residual = residual - np.outer(score, loading)
    transformed = np.column_stack(scores) / model["score_scale"]
    require(np.isfinite(transformed).all(), "V99 target PLS scores are non-finite")
    return transformed


def supervised_pls_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    state = RobustNumericState().fit(ordered)
    train_matrix, train_transform = state.transform(ordered)
    target_matrix, target_transform = state.transform(target_frame)
    target = ordered.y.to_numpy(int)
    sample_weight = duplicate_class_weights(ordered)
    pls_model, train_scores, pls_audit = fit_weighted_pls1(train_matrix, target, sample_weight)
    target_scores = transform_pls(target_matrix, pls_model)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
        n_jobs=1,
    )
    head.fit(train_scores, target, sample_weight=sample_weight)
    require(int(head.n_iter_[0]) < 500, "V99 PLS logistic head did not converge")
    probability = np.clip(head.predict_proba(target_scores)[:, 1], 1e-6, 1.0 - 1e-6)
    audit = {
        "train_n": len(ordered),
        "feature_n": len(FEATURES),
        "state_dimension": train_matrix.shape[1],
        "causal_numeric_prediction_inputs_only": True,
        "categorical_text_source_ticker_features": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "pls": pls_audit,
        "head": {
            "architecture": "fixed ridge logistic on four supervised orthogonal PLS scores",
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "iterations": int(head.n_iter_[0]),
            "coefficient": head.coef_.ravel().tolist(),
            "intercept": float(head.intercept_[0]),
            "coefficient_sha256": array_sha256(np.r_[head.intercept_, head.coef_.ravel()]),
        },
        "prediction": {
            "target_n": len(target_frame),
            "target_rows_used_for_scaling_components_or_head": False,
            "target_labels_used": False,
            "probability_mean": float(probability.mean()),
            "probability_std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
        "unsupervised_pca": False,
        "kernel_or_random_features": False,
        "tree_density_retrieval_depth_wavelet_belief_model": False,
    }
    return probability, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = supervised_pls_direction(inner_train, inner_valid)
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
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = max(
        eligible,
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 PLS blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "component_count_or_head_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v99_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V99 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = supervised_pls_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v99_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                blend_probability(baseline, challenger, architecture["weight"]),
                confidence,
                high,
            )
            for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["pls_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v99_model"] = selected["name"]
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
            f"[V99 PLS] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V99 evidence repeats an event")
    require(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)),
        "V99 changed V69 confidence",
    )
    require(
        np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)),
        "V99 changed V69 high-confidence membership",
    )
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"),
        timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V99 paired-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    deltas: dict[str, list[float]] = {
        "auc_delta": [],
        "balanced_accuracy_delta": [],
        "all_trade_net_delta": [],
    }
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        deltas["auc_delta"].append(float(
            scaffold.roc_auc_score(target, candidate) - scaffold.roc_auc_score(target, baseline)
        ))
        deltas["balanced_accuracy_delta"].append(float(
            scaffold.balanced_accuracy_score(target, candidate >= 0.5)
            - scaffold.balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        deltas["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(deltas["auc_delta"]) >= int(0.90 * draws), "V99 bootstrap lost too many draws")

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
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-time block bootstrap over inner-locked V99 PLS policy",
        "seed": SEED,
        "requested_draws": draws,
        "blocks": len(blocks),
        **{name: interval(values) for name, values in deltas.items()},
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
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V99/controller gate mismatch")

    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    candidate_robust_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_nested_bootstrap_keys = {
        "balanced_accuracy_lower95",
        "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95",
        "highconf_strategy_net_upper95",
    }
    nested_bootstrap_contract = (
        required_nested_bootstrap_keys.issubset(candidate_robust_bootstrap)
        and all(np.isfinite(float(candidate_robust_bootstrap[key])) for key in required_nested_bootstrap_keys)
    )
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    pls_valid = all(
        audit[side]["pls"]["component_n"] == PLS_COMPONENTS
        and audit[side]["pls"]["weighted_score_orthogonality_max_abs_off_diagonal"] <= 1e-7
        and audit[side]["prediction"]["target_rows_used_for_scaling_components_or_head"] is False
        and audit[side]["prediction"]["target_labels_used"] is False
        and audit[side]["causal_numeric_prediction_inputs_only"]
        for audit in audits
        for side in ("inner_model_audit", "outer_model_audit")
    )
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "paired_bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "paired_bootstrap_all_trade_net_delta_probability_gt_zero_ge_0_65": bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_summary_correct_nested_robustness_bootstrap_keys": bool(nested_bootstrap_contract),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "fixed_supervised_orthogonal_pls_contract_verified": pls_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14, "selected canonical gate count changed")
    require(selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V99 fallback is not exact entire V69")
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
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "candidate_nested_robustness_bootstrap": candidate_robust_bootstrap,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
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
            "exact_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    return {
        "contract": HYPOTHESIS,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }


def write_outputs(
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    material_gate = current_material_gate(evaluation)
    selected_contract = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected_contract["material_gate"] = material_gate
    require(
        selected_contract["material_gate"]["nested_bootstrap"]["contract_key"]
        == "selected.material_gate.nested_bootstrap",
        "V99 current material-gate nested-bootstrap contract key is invalid",
    )
    require(
        "bootstrap" in selected_contract["robustness"]
        and "balanced_accuracy_lower95" in selected_contract["robustness"]["bootstrap"]
        and "highconf_strategy_net_lower95" in selected_contract["robustness"]["bootstrap"],
        "V99 selected robustness bootstrap nesting is invalid",
    )
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V99",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V99_DIAGNOSTIC_ORTHOGONAL_PLS": evaluation["candidate_summary"],
                "V99_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact,
            "authority_audit": authority,
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V99",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "opened_dev_only": True,
            "selected": selected_contract,
            "candidate": evaluation["candidate_summary"],
            "material_gate": material_gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V99",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected_contract["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": material_gate,
            "nested_bootstrap": evaluation["bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V99_SUPERVISED_ORTHOGONAL_PLS_REPORT.json": {
            "version": "V99",
            "hypothesis": HYPOTHESIS,
            "features": FEATURES,
            "pls_components": PLS_COMPONENTS,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "architectures": ARCHITECTURES,
            "nested_fold_audits": audits,
            "evaluation": compact,
            "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V99",
            "status": "MATCH",
            "canonical_check_count": 14,
            "reported": selected_contract["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": material_gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION,
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected_contract["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED",
            "seal_authorized": False,
            "completed_at": scaffold.now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"],
        "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "outer_pls_response_energy_explained": audit["outer_model_audit"]["pls"]["response_energy_explained_fraction"],
        "outer_pls_orthogonality_error": audit["outer_model_audit"]["pls"]["weighted_score_orthogonality_max_abs_off_diagonal"],
        "strict_35m_embargo": True,
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V99_ORTHOGONAL_PLS_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V99_ORTHOGONAL_PLS_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V99_DIAGNOSTIC_ORTHOGONAL_PLS_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V99_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out),
        "artifact_count": len(files) + 1,
        "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json"),
        "required_controller_reports": [
            "MODEL_COMPARISON.json",
            "DEV_ROBUSTNESS_REPORT.json",
            "SOURCE_TRANSFER_REPORT.json",
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
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode/output cannot accompany MARKET_BIO_VERSION_OUTPUT",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test)
    runtime = configure_bounded_runtime() if bounded else {
        "mode": "controller_or_explicit_full",
        "numeric_thread_cap": "inherited_from_authorized_parent",
        "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK",
            "hypothesis": HYPOTHESIS,
            "authority": authority,
            "runtime": runtime,
            "dev_rows": len(dev),
            "v69_rows": len(champion),
            "features": FEATURES,
            "feature_n": len(FEATURES),
            "state_dimension": 2 * len(FEATURES),
            "causal_numeric_prediction_inputs_only": True,
            "architecture": "fixed four-component duplicate/class-balanced supervised weighted PLS1/NIPALS deflation plus fixed ridge-logistic head",
            "pls_components": PLS_COMPONENTS,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "architectures": ARCHITECTURES,
            "strict_embargo_minutes": 35,
            "outer_labels_used_for_selection": False,
            "canonical_controller_gate_checks": 14,
            "controller_no_argument_contract": {
                "environment": "MARKET_BIO_VERSION_OUTPUT",
                "full_run": True,
                "required_reports": [
                    "MODEL_COMPARISON.json",
                    "DEV_ROBUSTNESS_REPORT.json",
                    "SOURCE_TRANSFER_REPORT.json",
                ],
                "selected_current_material_gate": True,
                "selected_material_gate_nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "selected_robustness_bootstrap_nested": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return

    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion,
            diagnostic,
            evidence,
            audits,
            250 if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "runtime": runtime,
        "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": current_material_gate(evaluation),
        "candidate_nested_robustness_bootstrap": evaluation["candidate_nested_robustness_bootstrap"],
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
        summary["paired_delta_bootstrap"] = evaluation["bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
