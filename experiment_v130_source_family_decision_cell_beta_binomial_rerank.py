"""V130 source-family decision-cell Beta-binomial odds reranker.

Only strict-past atomic V69 OOF rows estimate how reliably each source family
was correct separately inside V69's predicted-DOWN and predicted-UP decision
cells.  Equal event-group observations feed a Jeffreys-smoothed global side
rate and a fixed-strength hierarchical Beta-binomial source posterior.  The
source-versus-global reliability odds ratio corrects each V69 logit, after
which the result is projected back into its original V69 decision cell.

Consequently every predicted direction, balanced accuracy, all-trade signed
net return, V69 confidence, and high-confidence mask are structurally frozen;
only within-decision-cell ordering can change, directly targeting AUC without
trading away BA or net.  This is not a Hodges-Lehmann/view-head ensemble,
PairLogit/ListNet/pairwise AUC loss, twin hyperplane, return-weighted learner,
causal-state coefficient model, density, geometry, tree, kernel, or neural
model.  Inner-past labels choose exact V69, HALF, or FULL fixed reliability
correction.  Outer labels remain evaluation-only under a strict 35-minute
nested chronology, and any material failure restores the exact entire V69
frame.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_name] = "2"

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v127_mann_whitney_borda_rank_direction as contract


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V130_SOURCE_FAMILY_DECISION_CELL_BETA_BINOMIAL_ODDS_RERANK_V1"
VERSION = 130
HYPOTHESIS = "SOURCE_FAMILY_DECISION_CELL_BETA_BINOMIAL_ODDS_RERANK_V1"
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v127_mann_whitney_borda_rank_direction.py"
EXPECTED_CONTRACT_SHA256 = "1e1ad1e2c40a41cb3eba8c9ebb605d75f1faecdaa04b533d089f4e263dbe040f"

PRIOR_STRENGTH = 16.0
JEFFREYS_ALPHA = 0.5
CELL_EPSILON = 1e-6
PROBABILITY_EPSILON = 1e-6
SEED = 13001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = (
    {"name": "SOURCE_RELIABILITY_ODDS_HALF", "strength": 0.50},
    {"name": "SOURCE_RELIABILITY_ODDS_FULL", "strength": 1.00},
)

require = contract.require
sha256 = contract.sha256
array_sha256 = contract.array_sha256
clean = contract.clean
bool_series = contract.bool_series
atomic_json = contract.atomic_json
atomic_csv = contract.atomic_csv
load_authorized = contract.load_authorized
aligned_dev = contract.aligned_dev
chronology = contract.chronology
metric = contract.metric
controller = contract.controller
v44 = contract.v44


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "source_family|v69_predicted_side|equal_event_group_correctness|"
        "global_jeffreys_half|hierarchical_beta_binomial_strength16|"
        "source_vs_global_reliability_log_odds|fixed_half_full|"
        "original_v69_decision_cell_projection|no_state_head_no_threshold_tuning"
    )
    require(PRIOR_STRENGTH == 16.0, "V130 prior strength changed")
    require(CELL_EPSILON == 1e-6, "V130 decision-cell epsilon changed")
    return {
        "representation": "strict-past V69 source-family by predicted-side event-group correctness",
        "global_prior": "Jeffreys Beta(0.5,0.5) separately by predicted side",
        "source_shrinkage_strength": PRIOR_STRENGTH,
        "correction": "source posterior reliability odds minus global reliability odds",
        "fixed_strengths": [item["strength"] for item in ARCHITECTURES],
        "decision_cell_projection_epsilon": CELL_EPSILON,
        "predicted_direction_structurally_frozen": True,
        "fitted_causal_state_coefficient_n": 0,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V127 contract scaffold changed")
    audit = contract.verify_authority()
    audit["code_dependency"] = {
        "path": CONTRACT_PATH.name,
        "sha256": EXPECTED_CONTRACT_SHA256,
        "purpose": "generic immutable V36/V69 authority, chronology, metrics, canonical gate, bootstrap and atomic IO contract only; no V127 output is read",
    }
    audit["v130_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
        "prior_version_output_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    process = psutil.Process()
    process.cpu_affinity(list(PREPARATION_CPU_AFFINITY))
    actual = tuple(sorted(process.cpu_affinity()))
    require(actual == PREPARATION_CPU_AFFINITY, "V130 preparation CPU affinity is not exactly 30-31")
    return {
        "requested_logical_cpus": list(PREPARATION_CPU_AFFINITY),
        "actual_logical_cpus": list(actual), "exact_match": True,
        "process_id": process.pid,
    }


def source_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.source_family.fillna("__MISSING_SOURCE__").astype(str).to_numpy()


def collapse_reference(reference: pd.DataFrame) -> pd.DataFrame:
    require(len(reference) > 0, "V130 empty reliability reference")
    work = reference[["event_group_id", "event_time_utc", "market", "source_family", "y", "prob"]].copy()
    work["source_family"] = work.source_family.fillna("__MISSING_SOURCE__").astype(str)
    work["predicted_side"] = (work.prob.to_numpy(float) >= 0.5).astype(int)
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V130 event group crosses direction labels")
    require(int(grouped.predicted_side.nunique().max()) == 1, "V130 event group crosses V69 decision sides")
    require(int(grouped.source_family.nunique().max()) == 1, "V130 event group crosses source families")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "source_family": "first",
        "y": "first", "predicted_side": "first",
    }).reset_index()
    result["correct"] = (result.y.to_numpy(int) == result.predicted_side.to_numpy(int)).astype(float)
    require(np.isfinite(result.correct).all(), "V130 reference correctness invalid")
    return result


def fit_reliability(
    groups: pd.DataFrame, weights: np.ndarray | None = None,
) -> tuple[dict[int, float], dict[tuple[str, int], float], dict[str, Any]]:
    require(len(groups) >= 120, "V130 reliability reference support too small")
    if weights is None:
        weights = np.ones(len(groups), float)
    weights = np.asarray(weights, float)
    require(weights.shape == (len(groups),) and np.isfinite(weights).all() and np.all(weights > 0.0), "V130 reliability weights invalid")
    side = groups.predicted_side.to_numpy(int)
    correct = groups.correct.to_numpy(float)
    source = groups.source_family.to_numpy(str)
    global_rate: dict[int, float] = {}
    source_rate: dict[tuple[str, int], float] = {}
    cell_rows: list[dict[str, Any]] = []
    for direction in (0, 1):
        mask = side == direction
        support = float(weights[mask].sum())
        require(int(mask.sum()) >= 40 and support > 0.0, f"V130 predicted-side {direction} support too small")
        correct_weight = float(np.sum(weights[mask] * correct[mask]))
        global_value = (correct_weight + JEFFREYS_ALPHA) / (support + 2.0 * JEFFREYS_ALPHA)
        global_value = float(np.clip(global_value, 1e-4, 1.0 - 1e-4))
        global_rate[direction] = global_value
        for family in sorted(set(source[mask])):
            cell = mask & (source == family)
            cell_support = float(weights[cell].sum())
            cell_correct = float(np.sum(weights[cell] * correct[cell]))
            posterior = (cell_correct + PRIOR_STRENGTH * global_value) / (cell_support + PRIOR_STRENGTH)
            posterior = float(np.clip(posterior, 1e-4, 1.0 - 1e-4))
            source_rate[(str(family), direction)] = posterior
            cell_rows.append({
                "source_family": str(family), "predicted_side": direction,
                "event_group_n": int(cell.sum()), "effective_weight": cell_support,
                "correct_weight": cell_correct, "posterior_reliability": posterior,
                "global_side_reliability": global_value,
            })
    rates = np.asarray([row["posterior_reliability"] for row in cell_rows], float)
    return global_rate, source_rate, {
        "event_group_n": len(groups),
        "source_family_n": int(groups.source_family.nunique()),
        "source_side_cell_n": len(cell_rows),
        "global_down_reliability": global_rate[0],
        "global_up_reliability": global_rate[1],
        "source_posterior_min": float(rates.min()),
        "source_posterior_median": float(np.median(rates)),
        "source_posterior_max": float(rates.max()),
        "cell_table": cell_rows,
        "cell_table_sha256": hashlib.sha256(
            json.dumps(clean(cell_rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def corrected_probability(
    baseline: np.ndarray,
    target_source: np.ndarray,
    global_rate: dict[int, float],
    source_rate: dict[tuple[str, int], float],
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = np.clip(np.asarray(baseline, float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    target_source = np.asarray(target_source, str)
    require(len(baseline) == len(target_source), "V130 target source shape mismatch")
    side = (baseline >= 0.5).astype(int)
    reliability = np.asarray([
        source_rate.get((str(family), int(direction)), global_rate[int(direction)])
        for family, direction in zip(target_source, side)
    ], float)
    global_values = np.asarray([global_rate[int(direction)] for direction in side], float)
    reliability_offset = logit(reliability) - logit(global_values)
    up_logit_offset = np.where(side == 1, reliability_offset, -reliability_offset)
    raw = expit(logit(baseline) + float(strength) * up_logit_offset)
    corrected = np.where(
        side == 1,
        np.maximum(raw, 0.5 + CELL_EPSILON),
        np.minimum(raw, 0.5 - CELL_EPSILON),
    )
    corrected = np.clip(corrected, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    require(np.array_equal(corrected >= 0.5, side.astype(bool)), "V130 decision-cell projection failed")
    unseen = np.asarray([
        (str(family), int(direction)) not in source_rate
        for family, direction in zip(target_source, side)
    ], bool)
    return corrected, {
        "strength": float(strength),
        "direction_exact": True,
        "unseen_source_side_rows": int(unseen.sum()),
        "unseen_source_side_fraction": float(unseen.mean()),
        "reliability_q10_q50_q90": [float(np.quantile(reliability, q)) for q in (0.1, 0.5, 0.9)],
        "up_logit_offset_q10_q50_q90": [float(np.quantile(up_logit_offset, q)) for q in (0.1, 0.5, 0.9)],
        "baseline_probability_sha256": array_sha256(baseline),
        "corrected_probability_sha256": array_sha256(corrected),
    }


def native_reliability_bootstrap(
    groups: pd.DataFrame,
    target_prob: np.ndarray,
    target_source: np.ndarray,
    nominal: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    probabilities = np.empty((len(target_prob), draws), float)
    for draw in range(draws):
        weights = generator.exponential(1.0, len(groups))
        global_rate, source_rate, _ = fit_reliability(groups, weights)
        probabilities[:, draw], _ = corrected_probability(
            target_prob, target_source, global_rate, source_rate, 1.0,
        )
    median_probability = np.median(probabilities, axis=1)
    row_std = np.std(probabilities, axis=1)
    return {
        "contract": "event-group Bayesian bootstrap of source-family by V69 decision-side hierarchical Beta-binomial reliability",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "target_labels_used": False, "prior_strength": PRIOR_STRENGTH,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probabilities - nominal[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal))),
        "direction_agreement_with_nominal": float(np.mean((probabilities >= 0.5) == (nominal[:, None] >= 0.5))),
        "row_probability_std_q10_q50_q90": [float(np.quantile(row_std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median_probability),
    }


def source_reliability_direction(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    strength: float,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(reference.market.nunique() == 1 and target.market.nunique() == 1, "V130 expects one market")
    require(str(reference.market.iloc[0]) == str(target.market.iloc[0]), "V130 market mismatch")
    groups = collapse_reference(reference)
    global_rate, source_rate, fit_audit = fit_reliability(groups)
    baseline = target.prob.to_numpy(float)
    target_source = source_values(target)
    probability, correction_audit = corrected_probability(
        baseline, target_source, global_rate, source_rate, strength,
    )
    full_probability, _ = corrected_probability(
        baseline, target_source, global_rate, source_rate, 1.0,
    )
    native = None if native_draws <= 0 else native_reliability_bootstrap(
        groups, baseline, target_source, full_probability, native_draws, native_seed,
    )
    return probability, {
        "market": str(reference.market.iloc[0]),
        "reference_row_n": len(reference), "reference_event_group_n": len(groups),
        "target_n": len(target), "causal_pre_event_only": True,
        "strict_past_atomic_v69_oof_only": True,
        "source_family_feature_n": 1, "causal_numeric_feature_n": 0,
        "text_ticker_issuer_or_return_feature_n": 0,
        "architecture": "strict-past source-family by V69 decision-side hierarchical Beta-binomial reliability log-odds reranking constrained to original V69 decision cells",
        "static_spec": STATIC_SPEC, "fit_audit": fit_audit,
        "correction_audit": correction_audit,
        "predicted_direction_exact_v69": bool(np.array_equal(probability >= 0.5, baseline >= 0.5)),
        "target_rows_used_for_reliability_fit_or_selection": False,
        "target_labels_used": False,
        "fitted_causal_state_coefficient_n": 0,
        "pair_list_auc_loss": False,
        "hodges_lehmann_or_view_head": False,
        "twin_hyperplane_or_return_salience": False,
        "density_geometry_tree_kernel_neural": False,
        "native_reliability_bootstrap": native,
        "probability_q10_q50_q90": [float(np.quantile(probability, q)) for q in (0.1, 0.5, 0.9)],
        "probability_sha256": array_sha256(probability),
    }


def strict_reference(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    market: str,
    target_valid: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference = champion.loc[
        champion.market.eq(market)
        & (champion.event_time_utc < target_start - contract.scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    # The model-level support contract below is 120 total groups and 40 per
    # predicted side.  Keep the chronology guard equal to that hypothesis-
    # implied floor so the smaller immutable KR inner cut (192 groups) is
    # valid without weakening either side-specific reliability requirement.
    require(len(reference) >= 120, f"V130 {label} reliability reference too small")
    reference_valid = aligned_dev(dev, reference)
    audit = chronology(reference_valid, target_valid, f"V130 {label} reliability")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF only"
    audit["target_labels_used_for_fit"] = False
    return reference, audit


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - contract.scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} V69 OOF")
    split = int(math.floor(len(prior) * (1.0 - contract.scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_reference, audit = strict_reference(dev, champion, market, inner_valid, f"inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_valid, inner_champion, inner_reference, audit


def choose_inner(
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
    inner_reference: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    model_audits: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        probability, model_audit = source_reliability_direction(
            inner_reference, inner_champion, architecture["strength"],
        )
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        decision_exact = bool(np.array_equal(probability >= 0.5, baseline >= 0.5))
        require(abs(ba_delta) <= 1e-15 and abs(net_delta) <= 1e-15, "V130 inner BA/net changed despite decision-cell contract")
        eligible = bool(
            decision_exact
            and model_audit["strict_past_atomic_v69_oof_only"]
            and not model_audit["target_rows_used_for_reliability_fit_or_selection"]
            and not model_audit["target_labels_used"]
        )
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "model_eligible": eligible,
            "eligible": eligible,
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
        model_audits[architecture["name"]] = model_audit
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed HALF/FULL source reliability odds correction; maximize fixed 2*AUC+BA+10*net with structurally exact V69 decisions",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audits


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v130_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([("US", 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_reference, outer_chronology = strict_reference(
            dev, champion, market, outer_valid, f"outer {market} fold {fold}",
        )
        inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audits = choose_inner(inner_valid, inner_champion, inner_reference)
        selected = policy["selected"]
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        outer_probabilities: dict[str, np.ndarray] = {}
        outer_model_audits: dict[str, Any] = {}
        for architecture in ARCHITECTURES:
            probability, model_audit = source_reliability_direction(
                outer_reference, outer_champion, architecture["strength"],
                native_draws=(SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
                if architecture["strength"] == 1.0 else 0,
                native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
            )
            outer_probabilities[architecture["name"]] = probability
            outer_model_audits[architecture["name"]] = model_audit
        candidate = baseline.copy() if selected["architecture"] is None else outer_probabilities[selected["name"]]
        require(np.array_equal(candidate >= 0.5, baseline >= 0.5), "V130 outer decision direction changed")
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v130_model"] = selected["name"]
        outer_diagnostics = {
            name: metric(outer_valid, probability, confidence, high)
            for name, probability in outer_probabilities.items()
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        require(candidate_metric["balanced_accuracy"] == baseline_metric["balanced_accuracy"], "V130 outer BA changed")
        require(candidate_metric["all_trade_mean_signed_net"] == baseline_metric["all_trade_mean_signed_net"], "V130 outer net changed")
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["candidate_prob"] = candidate
        evidence["v130_model"] = selected["name"]
        for name, probability in outer_probabilities.items():
            evidence[name.lower() + "_prob"] = probability
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audits": inner_model_audits,
            "outer_model_audits": outer_model_audits,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "outer_direction_exact_v69": True,
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V130 SOURCE RELIABILITY] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V130 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V130 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V130 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V130 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values: dict[str, list[float]] = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        values["balanced_accuracy_delta"].append(float(
            balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V130 nested bootstrap lost too many draws")

    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V130 policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int, smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V130/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    direction_exact = all(audit["outer_direction_exact_v69"] for audit in audits)
    require(candidate["balanced_accuracy"] == base["balanced_accuracy"], "V130 evaluated BA changed")
    require(candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"], "V130 evaluated net changed")
    full_model_audits = [audit["outer_model_audits"]["SOURCE_RELIABILITY_ODDS_FULL"] for audit in audits]
    reliability_valid = all(
        item["causal_pre_event_only"]
        and item["strict_past_atomic_v69_oof_only"]
        and item["predicted_direction_exact_v69"]
        and item["fit_audit"]["event_group_n"] >= 120
        and item["fitted_causal_state_coefficient_n"] == 0
        and not item["target_rows_used_for_reliability_fit_or_selection"]
        and not item["target_labels_used"]
        for item in full_model_audits
    )
    model_native = [item["native_reliability_bootstrap"] for item in full_model_audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"]
        and not item["target_labels_used"]
        and item["direction_agreement_with_nominal"] == 1.0
        for item in model_native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_exact_v69": candidate["balanced_accuracy"] == base["balanced_accuracy"],
        "outer_all_trade_net_exact_v69": candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"],
        "outer_predicted_direction_exact_v69": direction_exact,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_ba_delta_exact_zero": all(
            nested_bootstrap["balanced_accuracy_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")
        ),
        "nested_bootstrap_net_delta_exact_zero": all(
            nested_bootstrap["all_trade_net_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")
        ),
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_exact_v69": candidate_metrics["balanced_accuracy"] == base_metrics["balanced_accuracy"],
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "source_reliability_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "source_family_decision_cell_beta_binomial_contract_verified": reliability_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V130 fallback is not exact entire V69")
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
        "controller_native_robustness_bootstrap": controller_native,
        "model_native_source_reliability_bootstrap": model_native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "immutability": {
            "confidence_exact": confidence_exact, "predicted_direction_exact": direction_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
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
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_source_reliability_bootstrap": evaluation["model_native_source_reliability_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V130 nested bootstrap key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V130 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V130 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V130").resolve(), "V130 output is outside controller research/staging/V130")
    require(out.exists() and out.is_dir(), "V130 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V130 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V130 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V130 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V130 output is not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V130 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V130", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V130_DIAGNOSTIC_SOURCE_RELIABILITY_RERANK": evaluation["candidate_summary"],
                "V130_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V130", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V130", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
                "predicted_direction_exact_v69": evaluation["immutability"]["predicted_direction_exact"],
            },
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_FAMILY_DECISION_CELL_RELIABILITY_REPORT.json": {
            "version": "V130", "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V130", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V130", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
            "model_native_source_reliability_bootstrap": evaluation["model_native_source_reliability_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V130", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit, "completed_at": contract.scaffold.now(),
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
        "outer_direction_exact_v69": audit["outer_direction_exact_v69"],
        "reference_event_group_n": audit["outer_model_audits"]["SOURCE_RELIABILITY_ODDS_FULL"]["reference_event_group_n"],
        "source_family_n": audit["outer_model_audits"]["SOURCE_RELIABILITY_ODDS_FULL"]["fit_audit"]["source_family_n"],
        "strict_outer_35m_embargo": audit["outer_chronology"]["strict_35m_embargo"],
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V130_SOURCE_RELIABILITY_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V130_SOURCE_RELIABILITY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V130_DIAGNOSTIC_SOURCE_RELIABILITY_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V130_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
    outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    inner_probability, inner_audit = source_reliability_direction(inner_reference, inner_champion, 1.0)
    outer_probability, outer_audit = source_reliability_direction(outer_reference, outer_champion, 1.0)
    require(np.isfinite(inner_probability).all() and np.isfinite(outer_probability).all(), "V130 KR support probability invalid")
    return {
        "status": "KR2_SUPPORT_OK_NO_OUTER_EVALUATION",
        "market": market, "fold": fold,
        "inner_target_n": len(inner_valid), "outer_target_n": len(outer_valid),
        "inner_reference_event_group_n": inner_audit["reference_event_group_n"],
        "outer_reference_event_group_n": outer_audit["reference_event_group_n"],
        "inner_source_family_n": inner_audit["fit_audit"]["source_family_n"],
        "outer_source_family_n": outer_audit["fit_audit"]["source_family_n"],
        "inner_probability_sha256": array_sha256(inner_probability),
        "outer_probability_sha256": array_sha256(outer_probability),
        "inner_direction_exact_v69": inner_audit["predicted_direction_exact_v69"],
        "outer_direction_exact_v69": outer_audit["predicted_direction_exact_v69"],
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V130 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    affinity_audit = preparation_affinity() if (args.audit_only or args.smoke_test or args.support_probe) else {
        "requested_logical_cpus": None, "actual_logical_cpus": psutil.Process().cpu_affinity(),
        "exact_match": None, "reason": "controller-authorized full run is not preparation",
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "static_spec": STATIC_SPEC,
            "architecture": "source-family decision-cell hierarchical Beta-binomial reliability odds reranking constrained to exact V69 directions",
            "causal_pre_event_only": True, "source_family_feature_n": 1,
            "numeric_text_ticker_issuer_or_return_feature_n": 0,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "controller_native_robustness_bootstrap": True,
                "model_native_source_reliability_bootstrap": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "preparation_affinity": affinity_audit,
            "threadpools": threadpool_info(), "gpu_model_calls": 0,
        }), indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if (args.smoke_test or args.support_probe) else None):
        if args.support_probe:
            result = support_probe(dev, champion)
            result["preparation_affinity"] = affinity_audit
            result["threadpools"] = threadpool_info()
            result["gpu_model_calls"] = 0
            print(json.dumps(clean(result), indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            args.smoke_test,
        )
    if args.smoke_test:
        print(json.dumps(clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS,
            "preparation_affinity": affinity_audit,
            "threadpools": threadpool_info(), "gpu_model_calls": 0,
            "folds": len(audits), "audits": audits,
            "outer_baseline": evaluation["outer_baseline"],
            "outer_candidate": evaluation["outer_candidate"],
            "outer_auc_delta": evaluation["outer_auc_delta"],
            "outer_ba_delta": evaluation["outer_ba_delta"],
            "outer_net_delta": evaluation["outer_net_delta"],
            "material_gate": current_material_gate(evaluation),
            "canonical_gate": evaluation["canonical_gate_audit"]["selected"],
            "fallback": evaluation["fallback"], "output_written": False,
        }), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
