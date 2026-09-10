"""V90 quantile Tree-Augmented Naive Bayes causal-state challenger.

Every embargoed same-market training cut discretizes only causal pre-decision
numeric state with training-cut quartiles and a dedicated missing state.  A
deterministic maximum-spanning dependency tree is learned from duplicate- and
class-balanced conditional mutual information.  Dirichlet-smoothed root and
parent-child class-conditional tables then produce a generative direction
likelihood ratio.  This is a discrete conditional-dependency architecture, not
continuous Gaussian-copula QDA, RFF/kernel Bayes, bilinear BCE, DTW prototypes,
dynamic coefficient filtering, event graphs, or text/categorical modeling.

Earlier same-market committed V69 OOF labels alone select exact V69 or fixed
0.25/0.50 logit blends.  Outer labels are evaluation-only, every split has a
strict 35-minute embargo, and V69 confidence/high_conf remain frozen.  The
controller recomputes all canonical 14 gates.  Material failure restores the
exact entire atomic V69 frame.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
DEV_PATH = scaffold.DEV_PATH
V69_DIR = scaffold.V69_DIR
V69_PATH = scaffold.V69_PATH
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "staging" / "V90_QUANTILE_TREE_AUGMENTED_NAIVE_BAYES_V1"
)

VERSION = 90
HYPOTHESIS = "QUANTILE_TREE_AUGMENTED_NAIVE_BAYES_CAUSAL_STATE_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"
QUANTILE_LEVELS = (0.25, 0.50, 0.75)
OBSERVED_STATES = 4
MISSING_STATE = 4
STATE_COUNT = 5
STRUCTURE_ALPHA = 0.05
TABLE_ALPHA = 0.50
SEED = 9001
BOOTSTRAP_DRAWS = 2000
SMOKE_THREADS = 2
COST = scaffold.COST

ARCHITECTURES = (
    {"name": "QUANTILE_TAN_W0.25", "weight": 0.25},
    {"name": "QUANTILE_TAN_W0.50", "weight": 0.50},
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
        "purpose": "generic atomic-authority, chronology, metric, and IO utilities only; no V85 outputs are read",
    }
    return audit


def training_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_group_id").event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    target = frame.y.to_numpy(int)
    class_mass = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require((class_mass > 0).all(), "TAN training cut has one class")
    total = float(weight.sum())
    factor = np.where(target == 1, total / (2.0 * class_mass[1]), total / (2.0 * class_mass[0]))
    balanced = weight * factor
    require(np.isfinite(balanced).all() and (balanced > 0).all(), "invalid TAN weights")
    return balanced


class QuantileStateBinner:
    def __init__(self) -> None:
        self.edges: list[np.ndarray] = []

    def fit(self, frame: pd.DataFrame) -> "QuantileStateBinner":
        self.edges = []
        for feature in FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            require(len(finite) >= 200, f"{feature} finite support is too small")
            edges = np.unique(np.quantile(finite, QUANTILE_LEVELS)).astype(float)
            require(len(edges) >= 1 and np.isfinite(edges).all(), f"{feature} quartiles failed")
            self.edges.append(edges)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        state = np.empty((len(frame), len(FEATURES)), dtype=np.int8)
        for position, (feature, edges) in enumerate(zip(FEATURES, self.edges)):
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
            missing = ~np.isfinite(values)
            observed = np.searchsorted(edges, np.where(missing, 0.0, values), side="right")
            observed = np.minimum(observed, OBSERVED_STATES - 1)
            state[:, position] = np.where(missing, MISSING_STATE, observed).astype(np.int8)
        require(np.all((state >= 0) & (state < STATE_COUNT)), "invalid quantile state")
        return state

    def audit(self) -> dict[str, Any]:
        payload = json.dumps([edge.tolist() for edge in self.edges], separators=(",", ":"))
        return {
            "quantile_levels": QUANTILE_LEVELS, "observed_state_count": OBSERVED_STATES,
            "missing_state": MISSING_STATE,
            "edge_counts": [len(edge) for edge in self.edges],
            "edges_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "edges_fit_on_training_cut_only": True,
        }


def conditional_mutual_information(
    state: np.ndarray, target: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    feature_n = state.shape[1]
    result = np.zeros((feature_n, feature_n), dtype=float)
    for left in range(feature_n):
        for right in range(left + 1, feature_n):
            information = 0.0
            for label in (0, 1):
                index = target == label
                table = np.full((STATE_COUNT, STATE_COUNT), STRUCTURE_ALPHA, dtype=float)
                np.add.at(table, (state[index, left], state[index, right]), weights[index])
                probability = table / table.sum()
                marginal_left = probability.sum(axis=1, keepdims=True)
                marginal_right = probability.sum(axis=0, keepdims=True)
                ratio = probability / np.maximum(marginal_left * marginal_right, 1e-15)
                information += 0.5 * float(np.sum(probability * np.log(np.maximum(ratio, 1e-15))))
            result[left, right] = result[right, left] = information
    require(np.isfinite(result).all() and np.all(result >= -1e-12), "conditional mutual information failed")
    return result


def maximum_spanning_tree(information: np.ndarray) -> tuple[int, np.ndarray, list[dict[str, Any]]]:
    feature_n = information.shape[0]
    root = int(np.argmax(information.sum(axis=1)))
    selected = {root}
    parent = np.full(feature_n, -1, dtype=int)
    edges: list[dict[str, Any]] = []
    while len(selected) < feature_n:
        choices = [
            (float(information[left, right]), -left, -right, left, right)
            for left in sorted(selected) for right in range(feature_n) if right not in selected
        ]
        require(bool(choices), "TAN tree construction stalled")
        _, _, _, left, right = max(choices)
        parent[right] = left
        selected.add(right)
        edges.append({
            "parent_index": left, "child_index": right,
            "parent": FEATURES[left], "child": FEATURES[right],
            "conditional_mutual_information": float(information[left, right]),
        })
    require(parent[root] == -1 and np.sum(parent >= 0) == feature_n - 1, "TAN tree is invalid")
    return root, parent, edges


def fit_tables(
    state: np.ndarray, target: np.ndarray, weights: np.ndarray,
    root: int, parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    class_count = np.full(2, TABLE_ALPHA, dtype=float)
    for label in (0, 1):
        class_count[label] += weights[target == label].sum()
    class_probability = class_count / class_count.sum()
    root_probability = np.empty((2, STATE_COUNT), dtype=float)
    for label in (0, 1):
        count = np.full(STATE_COUNT, TABLE_ALPHA, dtype=float)
        index = target == label
        np.add.at(count, state[index, root], weights[index])
        root_probability[label] = count / count.sum()
    conditional: dict[int, np.ndarray] = {}
    for child in range(state.shape[1]):
        if child == root:
            continue
        table = np.empty((2, STATE_COUNT, STATE_COUNT), dtype=float)
        for label in (0, 1):
            count = np.full((STATE_COUNT, STATE_COUNT), TABLE_ALPHA, dtype=float)
            index = target == label
            np.add.at(count, (state[index, parent[child]], state[index, child]), weights[index])
            table[label] = count / count.sum(axis=1, keepdims=True)
        conditional[child] = table
    require(np.isfinite(root_probability).all(), "TAN root table failed")
    require(all(np.isfinite(table).all() for table in conditional.values()), "TAN child table failed")
    return class_probability, root_probability, conditional


def predict_tables(
    state: np.ndarray, root: int, parent: np.ndarray, class_probability: np.ndarray,
    root_probability: np.ndarray, conditional: dict[int, np.ndarray],
) -> np.ndarray:
    log_likelihood = np.tile(np.log(class_probability), (len(state), 1))
    for label in (0, 1):
        log_likelihood[:, label] += np.log(root_probability[label, state[:, root]])
        for child, table in conditional.items():
            log_likelihood[:, label] += np.log(table[label, state[:, parent[child]], state[:, child]])
    probability = expit(log_likelihood[:, 1] - log_likelihood[:, 0])
    require(np.isfinite(probability).all(), "TAN probability is non-finite")
    return np.clip(probability, 1e-5, 1.0 - 1e-5)


def tan_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V90 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V90 market fit mismatch")
    binner = QuantileStateBinner().fit(train)
    train_state = binner.transform(train)
    target_state = binner.transform(target_frame)
    target = train.y.to_numpy(int)
    weights = training_weights(train)
    information = conditional_mutual_information(train_state, target, weights)
    root, parent, edges = maximum_spanning_tree(information)
    class_probability, root_probability, conditional = fit_tables(
        train_state, target, weights, root, parent,
    )
    probability = predict_tables(
        target_state, root, parent, class_probability, root_probability, conditional,
    )
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train),
        "target_n": len(target_frame), "feature_n": len(FEATURES),
        "causal_numeric_only": True, "categorical_feature_n": 0,
        "text_feature_n": 0, "source_feature": False, "ticker_feature": False,
        "continuous_gaussian_density": False, "bilinear_terms": 0,
        "kernel_or_rff": False, "trajectory_or_dtw": False,
        "architecture": "discrete quantile Tree-Augmented Naive Bayes",
        "binner": binner.audit(), "structure_uses_training_labels_only": True,
        "target_labels_used": False, "target_rows_used_for_binning": False,
        "root_index": root, "root_feature": FEATURES[root],
        "edge_n": len(edges), "tree_connected": len(edges) == len(FEATURES) - 1,
        "tree_acyclic": len(edges) == len(FEATURES) - 1,
        "edges": edges,
        "conditional_mutual_information_max": float(np.max(information)),
        "conditional_mutual_information_mean_upper": float(information[np.triu_indices(len(FEATURES), 1)].mean()),
        "class_probability": class_probability.tolist(),
        "table_alpha": TABLE_ALPHA,
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
    challenger, model_audit = tan_direction(inner_train, inner_valid)
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
        model_audit["tree_connected"] and model_audit["tree_acyclic"]
        and model_audit["edge_n"] == len(FEATURES) - 1
        and model_audit["binner"]["edges_fit_on_training_cut_only"]
        and not model_audit["target_labels_used"]
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
        "selection_rule": "inner-past OOF only; exact no-op or fixed .25/.50 quantile-TAN logit blend; maximize 2*AUC+BA+10*net with fixed tree/BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v90_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V90 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = tan_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v90_model"] = selected["name"]
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
        evidence["tan_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v90_model"] = selected["name"]
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
            f"[V90 QUANTILE-TAN] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V90 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V90 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V90 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V90 bootstrap blocks")
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
    require(len(auc_delta) >= int(0.90 * draws), "V90 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked quantile-TAN policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V90/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    tan_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["tree_connected"]
        and audit["outer_model_audit"]["tree_acyclic"]
        and audit["outer_model_audit"]["edge_n"] == len(FEATURES) - 1
        and audit["outer_model_audit"]["binner"]["edges_fit_on_training_cut_only"]
        and audit["outer_model_audit"]["structure_uses_training_labels_only"]
        and not audit["outer_model_audit"]["target_labels_used"]
        and not audit["outer_model_audit"]["target_rows_used_for_binning"]
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
        "quantile_tan_contract_verified": tan_valid,
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V90 fallback is not exact complete V69")
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
            "version": "V90", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V90_DIAGNOSTIC_QUANTILE_TAN": evaluation["candidate_summary"],
                "V90_FAIL_CLOSED_SELECTED": selected_contract,
            },
            "evaluation": compact, "authority_audit": authority,
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V90", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected_contract,
            "candidate": evaluation["candidate_summary"],
            "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V90", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
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
        "QUANTILE_TREE_AUGMENTED_NAIVE_BAYES_REPORT.json": {
            "version": "V90", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact,
            "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V90", "status": "MATCH", "canonical_check_count": 14,
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
        "root_feature": audit["outer_model_audit"]["root_feature"],
        "edge_n": audit["outer_model_audit"]["edge_n"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V90_QUANTILE_TAN_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V90_QUANTILE_TAN_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V90_DIAGNOSTIC_QUANTILE_TAN_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V90_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "quantile_levels": QUANTILE_LEVELS, "state_count": STATE_COUNT,
            "architecture": "discrete class-conditional maximum-spanning dependency tree",
            "target_rows_used_for_binning": False,
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
        summary["tan_audit"] = audits[0]["outer_model_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
