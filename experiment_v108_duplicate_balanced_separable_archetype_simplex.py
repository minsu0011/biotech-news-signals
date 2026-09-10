"""V108 duplicate-balanced separable-archetype simplex challenger.

Every embargoed same-market train cut is represented only by robustly scaled
causal numeric state and deterministic missingness flags.  Training rows are
collapsed to label-free event-group centroids.  A deterministic farthest-first
traversal fixes twelve global separable archetype anchors, and each row is
encoded as a ridge-regularized convex simplex reconstruction of those anchors.
One fixed ridge-logistic direction head operates on the barycentric code.  This
is not nearest-neighbor retrieval, class prototype scoring, affine-subspace
residual classification, neural reconstruction, ICA, graph persistence,
symbolic context coding, tree, kernel, or class density.

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
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V108_DUPLICATE_BALANCED_SEPARABLE_ARCHETYPE_SIMPLEX_V1"
VERSION = 108
HYPOTHESIS = "DUPLICATE_BALANCED_SEPARABLE_ARCHETYPE_SIMPLEX_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v102_ordered_horizon_fused_lasso_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "d398c651102ff2e47e747d277aca5ee24718f0ae29013b3fb02f0bfc07fba3b2"

ARCHETYPE_COUNT = 12
SIMPLEX_ITERATIONS = 64
SIMPLEX_RIDGE = 0.05
RIDGE_LOGISTIC_C = 0.50
SEED = 10801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "SEPARABLE_ARCHETYPE_SIMPLEX_LOGIT_W0.25", "weight": 0.25},
    {"name": "SEPARABLE_ARCHETYPE_SIMPLEX_LOGIT_W0.50", "weight": 0.50},
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


def event_group_centroids(frame: pd.DataFrame, matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Collapse repeated event groups without labels so every group has one vote."""
    group_id = frame.event_group_id.astype(str).to_numpy()
    code, unique = pd.factorize(group_id, sort=False)
    require(len(unique) >= ARCHETYPE_COUNT, "V108 has too few event groups for archetypes")
    total = np.zeros((len(unique), matrix.shape[1]), dtype=float)
    count = np.zeros(len(unique), dtype=float)
    np.add.at(total, code, matrix)
    np.add.at(count, code, 1.0)
    centroid = total / count[:, None]
    require(np.isfinite(centroid).all(), "V108 event-group centroid is non-finite")
    return centroid, [str(value) for value in unique]


def select_separable_archetypes(
    centroid: np.ndarray,
    group_ids: list[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Deterministic farthest-first anchors, each an observed group centroid."""
    center = np.mean(centroid, axis=0)
    center_distance = np.sum((centroid - center) ** 2, axis=1)
    selected = [int(np.argmax(center_distance))]
    nearest_distance = np.sum((centroid - centroid[selected[0]]) ** 2, axis=1)
    for _ in range(1, ARCHETYPE_COUNT):
        nearest_distance[np.asarray(selected, dtype=int)] = -np.inf
        candidate = int(np.argmax(nearest_distance))
        require(np.isfinite(nearest_distance[candidate]) and nearest_distance[candidate] > 1e-12, "V108 archetype geometry collapsed")
        selected.append(candidate)
        distance = np.sum((centroid - centroid[candidate]) ** 2, axis=1)
        nearest_distance = np.minimum(nearest_distance, distance)
    anchors = centroid[np.asarray(selected, dtype=int)]
    pairwise = np.sum((anchors[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    off_diagonal = pairwise[~np.eye(ARCHETYPE_COUNT, dtype=bool)]
    require(float(off_diagonal.min()) > 1e-12, "V108 repeated archetype anchor")
    return anchors, [group_ids[index] for index in selected], {
        "archetype_count": ARCHETYPE_COUNT,
        "selection": "label-free deterministic farthest-first over duplicate-balanced event-group centroids",
        "minimum_pairwise_squared_distance": float(off_diagonal.min()),
        "maximum_pairwise_squared_distance": float(off_diagonal.max()),
        "anchor_group_ids": [group_ids[index] for index in selected],
        "anchor_sha256": array_sha256(anchors),
        "labels_used": False,
    }


def project_rows_to_simplex(matrix: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row onto the probability simplex."""
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    divisor = np.arange(1, matrix.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / divisor[None, :] > 0.0
    rho = np.maximum(np.sum(positive, axis=1) - 1, 0)
    threshold = cumulative[np.arange(len(matrix)), rho] / (rho + 1.0)
    projected = np.maximum(matrix - threshold[:, None], 0.0)
    require(float(np.max(np.abs(projected.sum(axis=1) - 1.0))) <= 1e-10, "V108 simplex projection failed")
    return projected


def simplex_codes(matrix: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed projected-gradient convex reconstruction codes; rows never refit anchors."""
    require(matrix.ndim == 2 and anchors.shape == (ARCHETYPE_COUNT, matrix.shape[1]), "V108 simplex-code shape mismatch")
    gram = anchors @ anchors.T
    lipschitz = 2.0 * (float(np.linalg.eigvalsh(gram).max()) + SIMPLEX_RIDGE)
    require(np.isfinite(lipschitz) and lipschitz > 0.0, "V108 simplex-code Lipschitz constant invalid")
    step = 1.0 / lipschitz
    code = np.full((len(matrix), ARCHETYPE_COUNT), 1.0 / ARCHETYPE_COUNT, dtype=float)

    def objective(current: np.ndarray) -> np.ndarray:
        residual = current @ anchors - matrix
        return np.sum(residual * residual, axis=1) + SIMPLEX_RIDGE * np.sum(current * current, axis=1)

    initial = objective(code)
    for _ in range(SIMPLEX_ITERATIONS):
        gradient = 2.0 * ((code @ anchors - matrix) @ anchors.T + SIMPLEX_RIDGE * code)
        code = project_rows_to_simplex(code - step * gradient)
    final = objective(code)
    gradient = 2.0 * ((code @ anchors - matrix) @ anchors.T + SIMPLEX_RIDGE * code)
    next_code = project_rows_to_simplex(code - step * gradient)
    fixed_point_error = float(np.max(np.abs(next_code - code)))
    require(np.isfinite(code).all() and np.isfinite(final).all(), "V108 simplex code is non-finite")
    require(float(np.max(final - initial)) <= 1e-8, "V108 simplex objective increased")
    active = np.sum(code > 1e-6, axis=1)
    audit = {
        "iterations": SIMPLEX_ITERATIONS,
        "ridge": SIMPLEX_RIDGE,
        "step": step,
        "fixed_point_max_abs_error": fixed_point_error,
        "initial_objective_mean": float(np.mean(initial)),
        "final_objective_mean": float(np.mean(final)),
        "final_reconstruction_rmse": float(np.sqrt(np.mean((code @ anchors - matrix) ** 2))),
        "active_archetypes_mean": float(np.mean(active)),
        "active_archetypes_min": int(active.min()),
        "active_archetypes_max": int(active.max()),
        "simplex_sum_max_abs_error": float(np.max(np.abs(code.sum(axis=1) - 1.0))),
        "code_sha256": array_sha256(code),
    }
    return code, audit


def archetype_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = scaffold.scaffold.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    group_state, group_ids = event_group_centroids(ordered, train_state)
    anchors, anchor_ids, archetype_audit = select_separable_archetypes(group_state, group_ids)
    train_code, train_code_audit = simplex_codes(train_state, anchors)
    target_code, target_code_audit = simplex_codes(target_state, anchors)
    target = ordered.y.to_numpy(int)
    head_weight = scaffold.scaffold.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="lbfgs",
        max_iter=500, random_state=SEED, n_jobs=1,
    )
    head.fit(train_code, target, sample_weight=head_weight)
    require(int(head.n_iter_[0]) < 500, "V108 ridge-logistic head did not converge")
    probability = np.clip(head.predict_proba(target_code)[:, 1], 1e-6, 1.0 - 1e-6)
    audit = {
        "train_n": len(ordered), "target_n": len(target_frame),
        "raw_feature_n": len(FEATURES), "state_dimension": train_state.shape[1],
        "causal_numeric_only": True, "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "archetype": {
            **archetype_audit,
            "event_group_centroid_n": len(group_state),
            "anchor_group_ids": anchor_ids,
            "train_codes": train_code_audit,
            "target_codes": target_code_audit,
            "representation_labels_used": False,
            "target_rows_used_for_anchor_selection": False,
        },
        "head": {
            "architecture": "fixed ridge logistic on 12 label-free convex barycentric codes",
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "iterations": int(head.n_iter_[0]),
            "coefficient": head.coef_.ravel().tolist(),
            "coefficient_sha256": array_sha256(np.r_[head.intercept_, head.coef_.ravel()]),
        },
        "target_rows_used_for_scaling_anchor_selection_or_head": False,
        "target_labels_used": False,
        "supervised_pls_xy_deflation": False,
        "class_conditional_svd_subspace": False,
        "owen_empirical_likelihood": False,
        "compositional_dirichlet_density": False,
        "nearest_neighbor_or_class_prototype_vote": False,
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
    challenger, model_audit = archetype_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 separable-archetype simplex blend; fixed 2*AUC+BA score with BA/net eligibility",
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
    diagnostic["v108_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V108 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = archetype_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v108_model"] = selected["name"]
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
        evidence["archetype_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v108_model"] = selected["name"]
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
            f"[V108 ARCHETYPE] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V108 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V108 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V108 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V108 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V108 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V108 separable-archetype simplex policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V108 canonical mismatch")
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
    archetype_valid = all(
        audit[side]["archetype"]["archetype_count"] == ARCHETYPE_COUNT
        and audit[side]["archetype"]["representation_labels_used"] is False
        and audit[side]["archetype"]["target_rows_used_for_anchor_selection"] is False
        and audit[side]["target_rows_used_for_scaling_anchor_selection_or_head"] is False
        and audit[side]["target_labels_used"] is False
        and audit[side]["owen_empirical_likelihood"] is False
        and audit[side]["compositional_dirichlet_density"] is False
        and audit[side]["nearest_neighbor_or_class_prototype_vote"] is False
        and audit[side]["feature_graph_or_persistence"] is False
        and audit[side]["symbol_context_or_kt_code"] is False
        and audit[side]["archetype"]["train_codes"]["simplex_sum_max_abs_error"] <= 1e-10
        and audit[side]["archetype"]["target_codes"]["simplex_sum_max_abs_error"] <= 1e-10
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
        "duplicate_balanced_separable_archetype_simplex_contract_verified": archetype_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V108 fallback is not exact entire V69")
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
            "version": "V108", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V108_DIAGNOSTIC_SEPARABLE_ARCHETYPE_SIMPLEX": evaluation["candidate_summary"],
                "V108_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V108", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V108", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V108_SEPARABLE_ARCHETYPE_SIMPLEX_REPORT.json": {
            "version": "V108", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "archetype_count": ARCHETYPE_COUNT,
            "simplex_iterations": SIMPLEX_ITERATIONS, "simplex_ridge": SIMPLEX_RIDGE,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V108", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V108_ARCHETYPE_SIMPLEX_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V108_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "architecture": "duplicate-balanced label-free 12-anchor separable archetype geometry plus fixed projected-gradient simplex codes and ridge-logistic head",
            "archetype_count": ARCHETYPE_COUNT,
            "simplex_iterations": SIMPLEX_ITERATIONS,
            "simplex_ridge": SIMPLEX_RIDGE,
            "ridge_logistic_c": RIDGE_LOGISTIC_C,
            "representation_labels_used": False, "causal_numeric_only": True,
            "owen_empirical_likelihood": False, "compositional_dirichlet_density": False,
            "feature_graph_or_persistence": False, "symbol_context_or_kt_code": False,
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
