"""V102 ordered-horizon fused-lasso causal-state direction challenger.

Within every embargoed same-market training cut, causal numeric state is
robustly scaled.  For each predeclared lookback-horizon channel, the ordered
coordinates are converted to a cumulative-increment design: an L1 coefficient
on the first transformed coordinate controls the first horizon coefficient,
while later L1 coefficients are exactly successive horizon-coefficient jumps.
A fixed L1 logistic fit therefore favors piecewise-constant response profiles
across adjacent horizons.  Standalone causal states and deterministic missing
flags remain ordinary sparse terms.  There is no pairwise class energy score,
class-conditional SVD/subspace residual, PLS, density, retrieval, tree, kernel,
wavelet, graph, or pattern-belief system.

Only immutable V36 DEV and atomic output_V69 are authoritative.  Strict nested
chronology uses a 35-minute embargo; inner-past labels choose exact V69 or fixed
.25/.50 blends, while outer labels are evaluation-only.  V69 confidence and
high-confidence membership are frozen.  Material failure returns the exact
entire V69 DataFrame.

Audit and US:2 smoke write nothing.  A future authorized controller may invoke
the runner without arguments using MARKET_BIO_VERSION_OUTPUT for a six-fold
full run and the required reports/current material-gate contract.
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
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v99_supervised_orthogonal_pls_causal_state as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V102_ORDERED_HORIZON_FUSED_LASSO_V1"
VERSION = 102
HYPOTHESIS = "ORDERED_HORIZON_FUSED_LASSO_CAUSAL_STATE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v99_supervised_orthogonal_pls_causal_state.py"
EXPECTED_SCAFFOLD_SHA256 = "7b4b595595a67177dc16d175260dfcf02e05b9c6d7449090824cb2057e55b9f1"

FUSED_L1_C = 0.20
SEED = 10201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
COEFFICIENT_ZERO_TOLERANCE = 1e-9

HORIZON_CHANNELS: dict[str, tuple[str, ...]] = {
    "RETURN": (
        "pre_ret_1", "pre_ret_2", "pre_ret_5", "pre_ret_10",
        "pre_ret_15", "pre_ret_30", "pre_ret_60", "pre_ret_120",
    ),
    "VOLATILITY": ("pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60"),
    "VOLUME": (
        "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
        "volume_ratio_10_60",
    ),
    "RANGE": ("range_5m", "range_10m", "range_30m", "range_60m"),
    "CLOSE_POSITION": ("close_position_10", "close_position_30", "close_position_60"),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
    "TREND": ("trend_slope_30", "trend_slope_60"),
    "UP_FRACTION": ("up_fraction_10", "up_fraction_30"),
}
GROUPED_FEATURES = tuple(column for columns in HORIZON_CHANNELS.values() for column in columns)
STANDALONE_FEATURES = tuple(column for column in FEATURES if column not in set(GROUPED_FEATURES))
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

ARCHITECTURES = (
    {"name": "ORDERED_HORIZON_FUSED_LOGIT_W0.25", "weight": 0.25},
    {"name": "ORDERED_HORIZON_FUSED_LOGIT_W0.50", "weight": 0.50},
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V99 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust state, metrics, canonical gate, blending, and atomic IO only; no V99 output is read",
    }
    return audit


def cumulative_increment_design(
    robust_state: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Map original horizon coefficients to a base plus adjacent jumps."""
    require(robust_state.ndim == 2 and robust_state.shape[1] == 2 * len(FEATURES), "V102 state shape changed")
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    numeric = robust_state[:, :len(FEATURES)]
    missing = robust_state[:, len(FEATURES):]
    columns: list[np.ndarray] = []
    names: list[str] = []
    channel_positions: dict[str, list[int]] = {}
    used: set[str] = set()
    for channel, channel_features in HORIZON_CHANNELS.items():
        indices = [feature_index[name] for name in channel_features]
        block = numeric[:, indices]
        positions: list[int] = []
        for offset, name in enumerate(channel_features):
            # Z_k=sum_{j>=k} X_j gives beta_j=sum_{k<=j} gamma_k, so
            # gamma_0 is the base and gamma_k=beta_k-beta_{k-1} thereafter.
            columns.append(block[:, offset:].sum(axis=1))
            names.append(f"{channel}__BASE" if offset == 0 else f"{channel}__JUMP_TO__{name}")
            positions.append(len(columns) - 1)
        channel_positions[channel] = positions
        used.update(channel_features)
    for name in STANDALONE_FEATURES:
        columns.append(numeric[:, feature_index[name]])
        names.append(f"STANDALONE__{name}")
        used.add(name)
    require(used == set(FEATURES), "V102 feature partition is not exhaustive")
    for name in FEATURES:
        columns.append(missing[:, feature_index[name]])
        names.append(f"MISSING__{name}")
    design = np.column_stack(columns)
    require(design.shape == robust_state.shape, "V102 cumulative design dimension changed")
    require(np.isfinite(design).all() and len(names) == design.shape[1], "V102 cumulative design invalid")
    return design, names, channel_positions


def reconstruct_horizon_coefficients(
    transformed_coefficient: np.ndarray,
    names: list[str],
    channel_positions: dict[str, list[int]],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    original: dict[str, list[float]] = {}
    audits: dict[str, Any] = {}
    for channel, positions in channel_positions.items():
        increment = transformed_coefficient[positions]
        coefficient = np.cumsum(increment)
        channel_features = HORIZON_CHANNELS[channel]
        require(len(coefficient) == len(channel_features), "V102 reconstructed channel length mismatch")
        original[channel] = coefficient.tolist()
        audits[channel] = {
            "features": list(channel_features),
            "increment_names": [names[index] for index in positions],
            "increments": increment.tolist(),
            "original_horizon_coefficients": coefficient.tolist(),
            "nonzero_increment_n": int(np.sum(np.abs(increment) > COEFFICIENT_ZERO_TOLERANCE)),
            "nonzero_jump_n": int(np.sum(np.abs(increment[1:]) > COEFFICIENT_ZERO_TOLERANCE)),
            "total_variation": float(np.sum(np.abs(np.diff(coefficient)))),
            "increment_l1": float(np.sum(np.abs(increment))),
        }
    return original, audits


def fused_horizon_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = scaffold.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_design, design_names, channel_positions = cumulative_increment_design(train_state)
    target_design, target_names, target_positions = cumulative_increment_design(target_state)
    require(design_names == target_names and channel_positions == target_positions, "V102 design contract changed on target")
    target = ordered.y.to_numpy(int)
    sample_weight = scaffold.duplicate_class_weights(ordered)
    model = LogisticRegression(
        C=FUSED_L1_C,
        penalty="l1",
        solver="liblinear",
        max_iter=3000,
        tol=1e-7,
        random_state=SEED,
        fit_intercept=True,
    )
    model.fit(train_design, target, sample_weight=sample_weight)
    require(int(model.n_iter_[0]) < 3000, "V102 fused logistic did not converge")
    probability = np.clip(model.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    coefficient = model.coef_.ravel()
    original, channel_audits = reconstruct_horizon_coefficients(coefficient, design_names, channel_positions)
    total_jump_n = int(sum(audit["nonzero_jump_n"] for audit in channel_audits.values()))
    possible_jump_n = int(sum(len(features) - 1 for features in HORIZON_CHANNELS.values()))
    audit = {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "raw_feature_n": len(FEATURES),
        "design_feature_n": train_design.shape[1],
        "causal_numeric_only": True,
        "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "architecture": "ordered-horizon cumulative-increment L1 logistic fused coefficient profile",
        "penalty_equivalence": "gamma_0=beta_0 and gamma_k=beta_k-beta_(k-1); L1(gamma) penalizes base plus exact adjacent coefficient jumps",
        "fused_l1_c": FUSED_L1_C,
        "iterations": int(model.n_iter_[0]),
        "intercept": float(model.intercept_[0]),
        "transformed_nonzero_n": int(np.sum(np.abs(coefficient) > COEFFICIENT_ZERO_TOLERANCE)),
        "transformed_coefficient_l1": float(np.sum(np.abs(coefficient))),
        "transformed_coefficient_sha256": array_sha256(coefficient),
        "design_names_sha256": hashlib.sha256("\n".join(design_names).encode()).hexdigest(),
        "channel_audits": channel_audits,
        "original_horizon_coefficients": original,
        "nonzero_horizon_jump_n": total_jump_n,
        "possible_horizon_jump_n": possible_jump_n,
        "piecewise_constant_fraction": float(1.0 - total_jump_n / possible_jump_n),
        "target_rows_used_for_scaling_design_or_fit": False,
        "target_labels_used": False,
        "class_energy_pairwise_distance": False,
        "class_conditional_svd_subspace": False,
        "pls_or_latent_deflation": False,
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
    challenger, model_audit = fused_horizon_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 fused-logit blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "penalty_or_architecture_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v102_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V102 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = fused_horizon_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v102_model"] = selected["name"]
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
        evidence["fused_logit_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v102_model"] = selected["name"]
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
            f"[V102 FUSED] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V102 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V102 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V102 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V102 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(scaffold.scaffold.roc_auc_score(target, candidate) - scaffold.scaffold.roc_auc_score(target, base)))
        values["balanced_accuracy_delta"].append(float(
            scaffold.scaffold.balanced_accuracy_score(target, candidate >= 0.5)
            - scaffold.scaffold.balanced_accuracy_score(target, base >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(base >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V102 nested bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V102 fused-horizon policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V102 canonical mismatch")
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
    fused_valid = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["target_rows_used_for_scaling_design_or_fit"] is False
        and audit[side]["target_labels_used"] is False
        and audit[side]["class_energy_pairwise_distance"] is False
        and audit[side]["class_conditional_svd_subspace"] is False
        and audit[side]["pls_or_latent_deflation"] is False
        and audit[side]["possible_horizon_jump_n"] > 0
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    candidate_native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    nested_native_keys = {
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
        "candidate_native_robustness_bootstrap_keys": nested_native_keys.issubset(candidate_native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "ordered_horizon_fused_lasso_contract_verified": fused_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V102 fallback is not exact entire V69")
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
        "candidate_native_robustness_bootstrap": candidate_native_bootstrap,
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
            "version": "V102", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V102_DIAGNOSTIC_FUSED_HORIZON": evaluation["candidate_summary"],
                "V102_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V102", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V102", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V102_ORDERED_HORIZON_FUSED_LASSO_REPORT.json": {
            "version": "V102", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "horizon_channels": HORIZON_CHANNELS,
            "standalone_features": STANDALONE_FEATURES, "fused_l1_c": FUSED_L1_C,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V102", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V102_FUSED_HORIZON_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V102_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    runtime = scaffold.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_or_explicit_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "horizon_channels": HORIZON_CHANNELS,
            "standalone_features": STANDALONE_FEATURES,
            "architecture": "ordered-horizon cumulative-increment fixed L1 logistic; L1 coordinates equal base plus exact adjacent horizon coefficient jumps",
            "fused_l1_c": FUSED_L1_C, "causal_numeric_only": True,
            "class_energy_pairwise_distance": False,
            "class_conditional_svd_subspace": False,
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
