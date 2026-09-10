"""V131 source-time maximin pairwise-AUC direction challenger.

Every embargoed same-market past cut uses only 37 causal numeric states and
their deterministic missing flags under a train-only robust transform.  Rows
are collapsed to equal event-group centroids.  Within each observed source
family, proper-past centroids are split into three fixed chronological
pseudo-folds.  Deterministic class-quantile UP-minus-DOWN state differences
form a bounded pairwise direction-rank design in every valid environment.
One no-intercept linear score minimizes a smooth maximum of environment-wise
pairwise logistic AUC losses plus fixed L2.  Source and pseudo-fold identity
define robustness environments but are never model features.

This directly targets source/fold-stable direction ranking.  It is not
calibration-only, threshold tuning, V128 semantic-view aggregation, V81 BCE
Group-DRO, V78 signed-return PairLogit, or V125 listwise slates.  It uses no
return magnitude/order, text, category, ticker, tree, neural model, or GPU.

Atomic V69 inner-past evidence selects exact V69 or a fixed .25/.50 logit
blend.  Outer labels are evaluation-only behind a strict 35-minute embargo.
V69 confidence/high-confidence remain exact.  Material failure restores the
exact entire V69 DataFrame.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "2"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

import experiment_v122_semantic_block_legendre_polynomial_chaos as scaffold


ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V131"
VERSION = 131
HYPOTHESIS = "SOURCE_TIME_MAXIMIN_PAIRWISE_AUC_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v122_semantic_block_legendre_polynomial_chaos.py"
EXPECTED_SCAFFOLD_SHA256 = "ab9b06c0a697d6d3fc1a0e8a3ac5ad74a85f5fb1b86effebf320ece7249d9ec4"

STATE_DIMENSION = 2 * len(FEATURES)
CHRONOLOGICAL_PSEUDO_FOLDS = 3
MAX_CLASS_QUANTILE_PAIRS = 96
MIN_CLASS_GROUPS_PER_ENVIRONMENT = 24
SMOOTH_MAX_TEMPERATURE = 0.05
PAIRWISE_L2 = 0.10
MAX_ITERATIONS = 300
SEED = 13101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "SOURCE_TIME_MAXIMIN_PAIRWISE_AUC_W0.25", "weight": 0.25},
    {"name": "SOURCE_TIME_MAXIMIN_PAIRWISE_AUC_W0.50", "weight": 0.50},
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
numeric = scaffold.numeric


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V122 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust state, metrics, canonical gate, blending, bootstrap and atomic IO only; no V122 output is read",
    }
    audit["v131_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
        "source_family_used_only_as_training_environment": True,
        "source_text_ticker_or_categorical_model_input": False,
    }
    return audit


def event_group_centroids(
    frame: pd.DataFrame,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require(len(frame) == len(design) and design.shape[1] == STATE_DIMENSION, "V131 centroid input mismatch")
    ordered = frame.reset_index(drop=True).copy()
    ordered["_row_position"] = np.arange(len(ordered), dtype=int)
    group_design: list[np.ndarray] = []
    group_target: list[int] = []
    group_time: list[np.datetime64] = []
    group_source: list[str] = []
    group_size: list[int] = []
    for _, group in ordered.groupby("event_group_id", sort=False):
        positions = group._row_position.to_numpy(int)
        target = group.y.to_numpy(int)
        source = group.source_family.astype(str).to_numpy()
        require(np.unique(target).size == 1, "V131 event-group direction label changed")
        require(np.unique(source).size == 1, "V131 event-group source family changed")
        group_design.append(np.mean(design[positions], axis=0))
        group_target.append(int(target[0]))
        group_time.append(pd.Timestamp(group.event_time_utc.min()).to_datetime64())
        group_source.append(str(source[0]))
        group_size.append(len(group))
    matrix = np.asarray(group_design, float)
    target = np.asarray(group_target, int)
    timestamp = np.asarray(group_time, dtype="datetime64[ns]")
    source = np.asarray(group_source, dtype=str)
    size = np.asarray(group_size, int)
    order = np.argsort(timestamp, kind="stable")
    matrix, target, timestamp, source, size = matrix[order], target[order], timestamp[order], source[order], size[order]
    # KR:2's strict inner proper-past cut contains 490 rows.  The operative
    # support contract is enforced below per environment (three valid
    # pseudo-folds, >=24 groups per class, >=400 pair differences), so this
    # global guard only rejects genuinely undersized or one-class cuts.
    require(len(matrix) >= 300 and np.unique(target).size == 2, "V131 centroid support insufficient")
    require(np.isfinite(matrix).all() and np.all(size >= 1), "V131 centroid invalid")
    return matrix, target, timestamp, source, {
        "source_row_n": len(frame), "event_group_centroid_n": len(matrix),
        "duplicate_balance": "one equal row per event_group centroid",
        "source_rows_per_group_mean": float(np.mean(size)),
        "source_rows_per_group_max": int(np.max(size)),
        "class_n": np.bincount(target, minlength=2).tolist(),
        "source_family_n": int(np.unique(source).size),
        "source_families": sorted(np.unique(source).tolist()),
        "centroid_design_sha256": array_sha256(matrix),
        "chronologically_sorted": True,
    }


def quantile_positions(length: int, count: int) -> np.ndarray:
    require(length >= count >= 1, "V131 quantile position support invalid")
    position = np.floor((np.arange(count, dtype=float) + 0.5) * length / count).astype(int)
    require(np.all(np.diff(position) > 0) and position[0] >= 0 and position[-1] < length, "V131 quantile positions invalid")
    return position


def environment_pair_designs(
    matrix: np.ndarray,
    target: np.ndarray,
    timestamp: np.ndarray,
    source: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], list[dict[str, Any]]]:
    environments: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for source_family in sorted(np.unique(source).tolist()):
        source_index = np.flatnonzero(source == source_family)
        source_index = source_index[np.argsort(timestamp[source_index], kind="stable")]
        pseudo_fold = np.minimum(
            CHRONOLOGICAL_PSEUDO_FOLDS - 1,
            np.floor(np.arange(len(source_index)) * CHRONOLOGICAL_PSEUDO_FOLDS / len(source_index)).astype(int),
        )
        for fold in range(CHRONOLOGICAL_PSEUDO_FOLDS):
            index = source_index[pseudo_fold == fold]
            down_index = index[target[index] == 0]
            up_index = index[target[index] == 1]
            if min(len(down_index), len(up_index)) < MIN_CLASS_GROUPS_PER_ENVIRONMENT:
                continue
            pair_count = min(MAX_CLASS_QUANTILE_PAIRS, len(down_index), len(up_index))
            down_position = quantile_positions(len(down_index), pair_count)
            up_position = quantile_positions(len(up_index), pair_count)
            down = matrix[down_index[down_position]]
            up = matrix[up_index[up_position]]
            first = up - down[::-1]
            second = up - np.roll(down, pair_count // 2, axis=0)
            difference = np.vstack([first, second])
            require(difference.shape == (2 * pair_count, STATE_DIMENSION), "V131 pair design shape changed")
            require(np.isfinite(difference).all(), "V131 pair design invalid")
            environments.append(difference)
            audits.append({
                "source_family": source_family, "chronological_pseudo_fold": fold,
                "centroid_n": len(index), "down_n": len(down_index), "up_n": len(up_index),
                "class_quantile_count": pair_count, "pair_difference_n": len(difference),
                "pairing": "UP-minus-DOWN aligned reverse quantiles plus fixed half-cycle DOWN shift",
                "difference_sha256": array_sha256(difference),
                "source_and_pseudo_fold_used_as_model_feature": False,
            })
    require(len(environments) >= CHRONOLOGICAL_PSEUDO_FOLDS, "V131 too few valid source-time environments")
    require(sum(len(item) for item in environments) >= 400, "V131 pairwise rank support insufficient")
    return tuple(environments), audits


def fit_smooth_max_pairwise_auc(
    environments: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = environments[0].shape[1]

    def loss_and_gradient(coefficient: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        losses: list[float] = []
        gradients: list[np.ndarray] = []
        for difference in environments:
            margin = difference @ coefficient
            losses.append(float(np.mean(np.logaddexp(0.0, -margin))))
            gradients.append(-(difference.T @ expit(-margin)) / len(difference))
        environment_loss = np.asarray(losses, float)
        environment_weight = np.exp(
            environment_loss / SMOOTH_MAX_TEMPERATURE
            - logsumexp(environment_loss / SMOOTH_MAX_TEMPERATURE)
        )
        gradient_matrix = np.vstack(gradients)
        objective = float(
            SMOOTH_MAX_TEMPERATURE * logsumexp(environment_loss / SMOOTH_MAX_TEMPERATURE)
            + 0.5 * PAIRWISE_L2 * np.dot(coefficient, coefficient)
        )
        gradient = environment_weight @ gradient_matrix + PAIRWISE_L2 * coefficient
        return objective, gradient, environment_loss, environment_weight

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _, _ = loss_and_gradient(coefficient)
        return value, gradient

    result = minimize(
        objective, np.zeros(dimension, dtype=float), method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITERATIONS, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    require(result.success and np.isfinite(result.x).all(), f"V131 optimizer failed: {result.message}")
    final_objective, final_gradient, environment_loss, environment_weight = loss_and_gradient(result.x)
    require(np.isfinite(final_objective) and np.isfinite(final_gradient).all(), "V131 final objective invalid")
    return result.x, {
        "objective": "smooth maximum of source-time environment pairwise logistic AUC losses plus fixed L2",
        "optimizer": "L-BFGS-B analytic gradient",
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "smooth_max_temperature": SMOOTH_MAX_TEMPERATURE, "l2": PAIRWISE_L2,
        "final_objective": float(final_objective),
        "gradient_linf": float(np.max(np.abs(final_gradient))),
        "coefficient_l2": float(np.linalg.norm(result.x)),
        "coefficient_sha256": array_sha256(result.x),
        "environment_losses": environment_loss.tolist(),
        "environment_adversarial_weights": environment_weight.tolist(),
        "worst_environment_loss": float(np.max(environment_loss)),
        "best_environment_loss": float(np.min(environment_loss)),
        "environment_loss_range": float(np.ptp(environment_loss)),
        "intercept": False,
    }


def source_time_maximin_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V131 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    centroid, centroid_target, centroid_time, centroid_source, centroid_audit = event_group_centroids(ordered, train_state)
    environments, environment_audit = environment_pair_designs(
        centroid, centroid_target, centroid_time, centroid_source,
    )
    coefficient, fit_audit = fit_smooth_max_pairwise_auc(environments)
    target_score = target_state @ coefficient
    probability = np.clip(expit(np.clip(target_score, -30.0, 30.0)), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V131 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_and_missing_only": True,
        "numeric_feature_n": len(FEATURES), "missing_flag_n": len(FEATURES),
        "categorical_text_ticker_feature_n": 0,
        "return_magnitude_or_order_used": False,
        "opportunity_or_confidence_target_used": False,
        "train_transform": train_transform, "target_transform": target_transform,
        "centroids": centroid_audit,
        "environment_definition": "observed source_family x within-source chronological proper-past tercile",
        "environment_n": len(environments), "environments": environment_audit,
        "fit": fit_audit,
        "target_rows_used_for_transform_centroids_environments_pairs_or_fit": False,
        "target_labels_used": False,
        "source_or_pseudo_fold_identity_used_as_model_feature": False,
        "calibration_or_threshold_tuning": False,
        "semantic_view_hodges_lehmann": False,
        "environment_binary_cross_entropy_group_dro": False,
        "signed_return_pairlogit_or_listnet": False,
        "twin_hyperplane_or_beta_binomial_rerank": False,
        "tree_kernel_graph_density_geometry_path_neural_or_gpu": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "target_score_sha256": array_sha256(target_score),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = source_time_maximin_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 source-time maximin pairwise-AUC blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "environment_pair_temperature_l2_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v131_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    selected_specs = [(smoke_market, 2)] if smoke else fold_specs
    for market, fold in selected_specs:
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V131 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = source_time_maximin_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v131_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["source_time_maximin_pairwise_auc_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v131_model"] = selected["name"]
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
            f"[V131 MAXIMIN-AUC] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V131 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V131 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V131 changed high_conf")
    return diagnostic, evidence, audits


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    compatibility_audits = copy.deepcopy(audits)
    for audit in compatibility_audits:
        for side in ("inner_model_audit", "outer_model_audit"):
            model = audit[side]
            model.update({
                "causal_numeric_only": True,
                "train_block": {"block_n": scaffold.CHAOS_VARIABLES},
                "train_chaos": {
                    "dimension": scaffold.CHAOS_DIMENSION,
                    "maximum_total_degree": scaffold.CHAOS_MAX_DEGREE,
                    "degree_counts": {"1": 8, "2": 36, "3": 120},
                    "multi_index_sha256": scaffold.MULTI_INDEX_SHA256,
                },
                "target_chaos": {"complete_total_degree_basis": True},
                "chaos_scaling": {"train_only": True},
                "target_rows_used_for_block_basis_scale_or_head": False,
                "source_shape_additive_ramps": False,
                "field_aware_low_rank_bilinear": False,
                "bernstein_partition_or_ulsif": False,
                "recurrence_plot_or_rqa": False,
                "hyperdimensional_binding_or_bundle": False,
                "path_dynamics_signature_spd_or_dmd": False,
                "tree_kernel_graph_or_neural": False,
            })
    evaluation = scaffold.evaluate(champion, diagnostic, evidence, compatibility_audits, draws, smoke)
    evaluation["nested_bootstrap"]["method"] = "paired market-fold-time block bootstrap over inner-locked V131 source-time maximin pairwise-AUC policy"
    checks = evaluation["material_checks"]
    checks.pop("semantic_block_legendre_polynomial_chaos_contract_verified")
    maximin_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["centroids"]["duplicate_balance"] == "one equal row per event_group centroid"
        and audit[side]["environment_n"] >= CHRONOLOGICAL_PSEUDO_FOLDS
        and all(not environment["source_and_pseudo_fold_used_as_model_feature"] for environment in audit[side]["environments"])
        and audit[side]["fit"]["optimizer_success"]
        and audit[side]["fit"]["smooth_max_temperature"] == SMOOTH_MAX_TEMPERATURE
        and audit[side]["fit"]["l2"] == PAIRWISE_L2
        and not audit[side]["target_rows_used_for_transform_centroids_environments_pairs_or_fit"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["source_or_pseudo_fold_identity_used_as_model_feature"]
        and not audit[side]["calibration_or_threshold_tuning"]
        and not audit[side]["semantic_view_hodges_lehmann"]
        and not audit[side]["environment_binary_cross_entropy_group_dro"]
        and not audit[side]["signed_return_pairlogit_or_listnet"]
        and not audit[side]["twin_hyperplane_or_beta_binomial_rerank"]
        and not audit[side]["tree_kernel_graph_density_geometry_path_neural_or_gpu"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["source_time_maximin_pairwise_auc_direction_contract_verified"] = maximin_contract
    material_pass = bool(not smoke and all(checks.values()))
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = evaluation["candidate_summary"] if material_pass else evaluation["baseline_summary"]
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "V131 selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V131 fallback is not exact entire V69")
    evaluation.update({
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "selected_summary": selected_summary,
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
        "canonical_gate_audit": {
            **evaluation["canonical_gate_audit"], "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
    })
    return evaluation


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V131 nested bootstrap contract key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V131 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V131 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V131", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V131_DIAGNOSTIC_SOURCE_TIME_MAXIMIN_PAIRWISE_AUC": evaluation["candidate_summary"],
                "V131_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V131", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V131", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V131_SOURCE_TIME_MAXIMIN_PAIRWISE_AUC_REPORT.json": {
            "version": "V131", "hypothesis": HYPOTHESIS,
            "state_dimension": STATE_DIMENSION,
            "chronological_pseudo_folds": CHRONOLOGICAL_PSEUDO_FOLDS,
            "maximum_class_quantile_pairs": MAX_CLASS_QUANTILE_PAIRS,
            "smooth_max_temperature": SMOOTH_MAX_TEMPERATURE,
            "l2": PAIRWISE_L2,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V131", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V131_SOURCE_TIME_MAXIMIN_PAIRWISE_AUC_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V131_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "output": str(out),
        "required_controller_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V131 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V131 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V131 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test or args.support_probe) else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "state_dimension": STATE_DIMENSION,
            "architecture": "equal event-group centroids; observed source-family by chronological-tercile deterministic class-quantile direction pairs; smooth max environment pairwise logistic AUC linear score",
            "chronological_pseudo_folds": CHRONOLOGICAL_PSEUDO_FOLDS,
            "maximum_class_quantile_pairs": MAX_CLASS_QUANTILE_PAIRS,
            "smooth_max_temperature": SMOOTH_MAX_TEMPERATURE,
            "l2": PAIRWISE_L2,
            "calibration_or_threshold_tuning": False,
            "return_magnitude_or_order_used": False,
            "source_text_ticker_or_categorical_model_input": False,
            "semantic_view_hodges_lehmann": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "exact_output_binding": True,
                "research_staging_v131_compatible": True,
                "explicit_conflict_fail_closed": True,
                "direct_full_without_env_fail_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness": True, "source_transfer": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    bounded = bool(args.smoke_test or args.support_probe)
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V131 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS,
                "runtime": runtime, "market": "KR", "fold": 2,
                "strict_inner_chronology": audit["inner_chronology"],
                "strict_outer_chronology": audit["outer_chronology"],
                "raw_inner_selected": audit["policy"]["selected"],
                "raw_inner_best_non_noop": best_non_noop,
                "raw_outer_baseline": audit["outer_baseline"],
                "raw_outer_selected": audit["outer_candidate"],
                "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
                "inner_model_audit": audit["inner_model_audit"],
                "outer_model_audit": audit["outer_model_audit"],
                "selection_locked_before_outer_evaluation": True,
                "nested_bootstrap_executed": False,
                "nested_bootstrap_reason": "KR:2 support probe validates model support only; standard material bootstrap remains on US smoke/full evaluation",
                "exact_entire_v69_fallback_contract": True,
                "output_written": False,
            }), ensure_ascii=False, indent=2))
            return
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
        "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None,
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
