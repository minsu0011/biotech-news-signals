"""V159 multichannel causal pointwise transfer-entropy direction challenger.

The proposed recurrence/RQA family was rejected because V120 already implements
the same train-radius recurrence plots and recurrence/diagonal/vertical line
statistics. V159 instead converts the immutable train-scaled observed 8-channel
by 8-horizon path into train-only channel-tertile symbols. Duplicate-balanced,
label-free transition tables estimate every ordered channel pair's lagged
pointwise conditional information contribution. Four fixed summaries for each
of 56 directed pairs form a 224-dimensional representation. One train-only
robust scale and duplicate/class-balanced ridge-logistic direction head follows.

V90 uses supervised class-conditional mutual information to select a static TAN
feature tree. V107 codes a fixed semantic field order with class KT contexts.
V144 counts within-channel ordinal motifs. V159 instead models label-free lagged
cross-channel conditional transition information along the causal horizon axis.
Inner labels select exact V69 or fixed .25/.50 blends; outer labels are evaluation
only under 35-minute embargo/event purge. Atomic V69 confidence/high_conf remain
exact, and material failure restores the entire V69 frame.
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
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import experiment_v157_source_time_density_ratio_reweighted_direction as utilities
import experiment_v144_multichannel_weak_ordinal_motif_transition_direction as path_util


ROOT = utilities.ROOT
V69_DIR = utilities.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V159" / "LOCAL_FORBIDDEN"
VERSION = 159
HYPOTHESIS = "MULTICHANNEL_CAUSAL_TRANSFER_ENTROPY_DIRECTION_V1"
FEATURES = utilities.FEATURES
EXPECTED_V69_EXPERIMENT_ID = utilities.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v157_source_time_density_ratio_reweighted_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "2d6ef425ef9a927de90bdca91dabea63565ca731fd5649f5a0e016e15540a372"
PATH_UTILITY_PATH = ROOT / "experiment_v144_multichannel_weak_ordinal_motif_transition_direction.py"
EXPECTED_PATH_UTILITY_SHA256 = "19817afa20b5243fea7831fd1d26a17cc2159085c5a774a2d9524a82d0f4064c"

CHANNEL_N = 8
HORIZON_N = 8
SYMBOL_N = 3
ORDERED_PAIR_N = CHANNEL_N * (CHANNEL_N - 1)
SUMMARIES_PER_PAIR = 4
TRANSFER_FEATURE_DIMENSION = ORDERED_PAIR_N * SUMMARIES_PER_PAIR
DIRICHLET_ALPHA = 0.5
RECENT_TRANSITIONS = 3
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 15901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "MULTICHANNEL_TRANSFER_ENTROPY_W0.25", "weight": 0.25},
    {"name": "MULTICHANNEL_TRANSFER_ENTROPY_W0.50", "weight": 0.50},
)

require = utilities.require
sha256 = utilities.sha256
array_sha256 = utilities.array_sha256
clean = utilities.clean
bool_series = utilities.bool_series
atomic_json = utilities.atomic_json
atomic_csv = utilities.atomic_csv
load_authorized = utilities.load_authorized
metric = utilities.metric
blend_probability = utilities.blend_probability
controller = utilities.controller
v44 = utilities.v44
numeric = utilities.numeric
scaffold = utilities.scaffold

PAIR_SOURCE = np.asarray([source for source in range(CHANNEL_N) for target in range(CHANNEL_N) if source != target], int)
PAIR_TARGET = np.asarray([target for source in range(CHANNEL_N) for target in range(CHANNEL_N) if source != target], int)

STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(path_util.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "symbol_n": SYMBOL_N,
    "quantization": "train-only duplicate-balanced pooled-horizon channel tertiles",
    "ordered_channel_pair_n": ORDERED_PAIR_N,
    "transition_model": "Dirichlet-smoothed target-next conditional on target-prev versus target-prev plus source-prev",
    "dirichlet_alpha": DIRICHLET_ALPHA,
    "pointwise_contribution": "log P(target_next|target_prev,source_prev) minus log P(target_next|target_prev)",
    "summaries_per_pair": ["mean", "standard_deviation", "positive_fraction", "recent_three_mean"],
    "feature_dimension": TRANSFER_FEATURE_DIMENSION,
    "head": "train-only robust feature scale plus duplicate/class-balanced ridge logistic C0.50",
    "quantile_alpha_summary_head_or_blend_grid": False,
}
require(
    len(PAIR_SOURCE) == len(PAIR_TARGET) == ORDERED_PAIR_N == 56
    and TRANSFER_FEATURE_DIMENSION == 224,
    "V159 static transfer-entropy dimension changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V157 utility scaffold changed")
    require(sha256(PATH_UTILITY_PATH) == EXPECTED_PATH_UTILITY_SHA256, "pinned V144 path utility changed")
    audit = utilities.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust state, metrics, canonical gate, fixed blending, bootstrap and atomic IO only; no V157 output is read",
    }
    audit["auxiliary_code_dependency"] = {
        "path": PATH_UTILITY_PATH.name,
        "sha256": EXPECTED_PATH_UTILITY_SHA256,
        "purpose": "fixed label-free 8-channel by 8 observed-horizon path construction only; no V144 output is read",
    }
    audit["v159_access_contract"] = {
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


def duplicate_weights(frame: pd.DataFrame) -> np.ndarray:
    size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(size, 1.0)
    return weight / float(np.mean(weight))


class TransferEntropyMap:
    def __init__(self) -> None:
        self.thresholds = np.empty((CHANNEL_N, SYMBOL_N - 1), float)
        self.base_probability = np.empty((ORDERED_PAIR_N, SYMBOL_N, SYMBOL_N), float)
        self.joint_probability = np.empty((ORDERED_PAIR_N, SYMBOL_N, SYMBOL_N, SYMBOL_N), float)
        self.fit_audit: dict[str, Any] = {}

    def symbols(self, path: np.ndarray) -> np.ndarray:
        require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V159 path shape invalid")
        result = np.zeros(path.shape, int)
        for channel in range(CHANNEL_N):
            result[:, :, channel] = np.searchsorted(self.thresholds[channel], path[:, :, channel], side="right")
        require(np.all((result >= 0) & (result < SYMBOL_N)), "V159 symbol range invalid")
        return result

    def fit(self, path: np.ndarray, event_weight: np.ndarray) -> "TransferEntropyMap":
        require(len(path) == len(event_weight) and np.all(event_weight > 0.0), "V159 fit weights invalid")
        repeated_weight = np.repeat(event_weight, HORIZON_N)
        for channel in range(CHANNEL_N):
            values = path[:, :, channel].reshape(-1)
            order = np.argsort(values, kind="stable")
            ordered_values = values[order]
            ordered_weight = repeated_weight[order]
            cumulative = np.cumsum(ordered_weight)
            total = float(cumulative[-1])
            self.thresholds[channel] = [
                ordered_values[min(int(np.searchsorted(cumulative, fraction * total, side="left")), len(values) - 1)]
                for fraction in (1.0 / 3.0, 2.0 / 3.0)
            ]
        symbol = self.symbols(path)
        base_count = np.full((ORDERED_PAIR_N, SYMBOL_N, SYMBOL_N), DIRICHLET_ALPHA, float)
        joint_count = np.full((ORDERED_PAIR_N, SYMBOL_N, SYMBOL_N, SYMBOL_N), DIRICHLET_ALPHA, float)
        for pair_index, (source, target) in enumerate(zip(PAIR_SOURCE, PAIR_TARGET)):
            source_previous = symbol[:, :-1, source]
            target_previous = symbol[:, :-1, target]
            target_next = symbol[:, 1:, target]
            for row in range(len(path)):
                weight = float(event_weight[row])
                for step in range(HORIZON_N - 1):
                    sp = int(source_previous[row, step]); tp = int(target_previous[row, step]); tn = int(target_next[row, step])
                    base_count[pair_index, tp, tn] += weight
                    joint_count[pair_index, sp, tp, tn] += weight
        self.base_probability = base_count / base_count.sum(axis=2, keepdims=True)
        self.joint_probability = joint_count / joint_count.sum(axis=3, keepdims=True)
        require(
            np.isfinite(self.base_probability).all() and np.isfinite(self.joint_probability).all()
            and np.allclose(self.base_probability.sum(axis=2), 1.0)
            and np.allclose(self.joint_probability.sum(axis=3), 1.0),
            "V159 transition probability contract failed",
        )
        self.fit_audit = {
            "rows": len(path), "thresholds_sha256": array_sha256(self.thresholds),
            "thresholds": self.thresholds.tolist(),
            "base_probability_sha256": array_sha256(self.base_probability),
            "joint_probability_sha256": array_sha256(self.joint_probability),
            "label_independent": True, "duplicate_balanced": True,
            "target_rows_used": False, "dirichlet_alpha": DIRICHLET_ALPHA,
        }
        return self

    def transform(self, path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        symbol = self.symbols(path)
        contributions = np.empty((len(path), ORDERED_PAIR_N, HORIZON_N - 1), float)
        for pair_index, (source, target) in enumerate(zip(PAIR_SOURCE, PAIR_TARGET)):
            sp = symbol[:, :-1, source]
            tp = symbol[:, :-1, target]
            tn = symbol[:, 1:, target]
            joint = self.joint_probability[pair_index, sp, tp, tn]
            base = self.base_probability[pair_index, tp, tn]
            contributions[:, pair_index, :] = np.log(joint) - np.log(base)
        feature = np.stack([
            np.mean(contributions, axis=2),
            np.std(contributions, axis=2),
            np.mean(contributions > 0.0, axis=2),
            np.mean(contributions[:, :, -RECENT_TRANSITIONS:], axis=2),
        ], axis=2).reshape(len(path), -1)
        require(feature.shape == (len(path), TRANSFER_FEATURE_DIMENSION) and np.isfinite(feature).all(), "V159 transfer feature invalid")
        return feature, {
            "rows": len(path), "feature_dimension": feature.shape[1],
            "symbol_frequency": np.bincount(symbol.ravel(), minlength=SYMBOL_N).astype(float).tolist(),
            "pointwise_contribution_mean": float(np.mean(contributions)),
            "pointwise_contribution_positive_fraction": float(np.mean(contributions > 0.0)),
            "feature_sha256": array_sha256(feature),
            "train_transition_tables_only": True,
        }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, float)
        self.scale = np.empty(0, float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.shape[1] == TRANSFER_FEATURE_DIMENSION, "V159 scale shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        scale = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V159 scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V159 scaled feature invalid")
        return result


def transfer_entropy_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V159 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = path_util.scaffold.observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = path_util.scaffold.observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    event_weight = duplicate_weights(ordered)
    transfer_map = TransferEntropyMap().fit(train_path, event_weight)
    train_feature, train_feature_audit = transfer_map.transform(train_path)
    target_feature, target_feature_audit = transfer_map.transform(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V159 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "static_spec": STATIC_SPEC, "transfer_map_fit": transfer_map.fit_audit,
        "train_transfer_feature": train_feature_audit, "target_transfer_feature": target_feature_audit,
        "feature_scaling": {"train_only": True, "clip": FEATURE_CLIP, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale)},
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C,
            "solver": "liblinear", "iterations": int(head.n_iter_[0]),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_), "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_quantiles_transition_tables_feature_scale_or_head": False,
        "target_labels_used": False,
        "label_independent_transfer_tables": True,
        "lagged_ordered_cross_channel_information": True,
        "recurrence_plot_or_rqa": False,
        "supervised_tan_conditional_mutual_information": False,
        "field_order_kt_context_code": False,
        "within_channel_ordinal_motif_transition": False,
        "hilbert_analytic_signal_or_phase_locking": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def supported_source_ba_delta(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index].y.to_numpy(int)
        if len(index) < 40 or np.unique(target).size != 2:
            continue
        deltas[str(source)] = float(balanced_accuracy_score(target, candidate[index] >= 0.5) - balanced_accuracy_score(target, baseline[index] >= 0.5))
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = transfer_entropy_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["static_spec"]["feature_dimension"] == TRANSFER_FEATURE_DIMENSION
        and model_audit["transfer_map_fit"]["label_independent"]
        and model_audit["transfer_map_fit"]["duplicate_balanced"]
        and model_audit["train_transfer_feature"]["train_transition_tables_only"]
        and model_audit["target_transfer_feature"]["train_transition_tables_only"]
        and model_audit["label_independent_transfer_tables"]
        and model_audit["lagged_ordered_cross_channel_information"]
        and not model_audit["target_rows_used_for_robust_scale_quantiles_transition_tables_feature_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["recurrence_plot_or_rqa"]
        and not model_audit["supervised_tan_conditional_mutual_information"]
        and not model_audit["field_order_kt_context_code"]
        and not model_audit["within_channel_ordinal_motif_transition"]
        and not model_audit["hilbert_analytic_signal_or_phase_locking"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 multichannel causal transfer-entropy blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "quantile_alpha_summary_head_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.SEED = SEED
    scaffold.choose_inner = choose_inner
    scaffold.ordinal_direction = transfer_entropy_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v144_model": "v159_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "transfer_entropy_prob", "v144_model": "v159_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V159 TRANSFER ENTROPY] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V159 causal transfer-entropy policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def model_contract(audits: list[dict[str, Any]]) -> bool:
    return all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["feature_dimension"] == TRANSFER_FEATURE_DIMENSION
        and audit[side]["transfer_map_fit"]["label_independent"]
        and audit[side]["transfer_map_fit"]["duplicate_balanced"]
        and audit[side]["train_transfer_feature"]["train_transition_tables_only"]
        and audit[side]["target_transfer_feature"]["train_transition_tables_only"]
        and audit[side]["label_independent_transfer_tables"]
        and audit[side]["lagged_ordered_cross_channel_information"]
        and not audit[side]["target_rows_used_for_robust_scale_quantiles_transition_tables_feature_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["recurrence_plot_or_rqa"]
        and not audit[side]["supervised_tan_conditional_mutual_information"]
        and not audit[side]["field_order_kt_context_code"]
        and not audit[side]["within_channel_ordinal_motif_transition"]
        and not audit[side]["hilbert_analytic_signal_or_phase_locking"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V159 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool)))
    native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "multichannel_causal_transfer_entropy_contract_verified": model_contract(audits),
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V159 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"], "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"], "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "candidate_native_robustness_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V159 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V159 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V159 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V159").resolve(), "V159 output is outside controller research/staging/V159")
    require(out.exists() and out.is_dir(), "V159 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V159 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V159 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V159 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V159 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V159", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V159_DIAGNOSTIC_TRANSFER_ENTROPY": evaluation["candidate_summary"], "V159_FAIL_CLOSED_SELECTED": selected},
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V159", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True,
            "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V159", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V159_MULTICHANNEL_CAUSAL_TRANSFER_ENTROPY_REPORT.json": {"version": "V159", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V159", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V159", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V159", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V159_TRANSFER_ENTROPY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V159_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


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
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V159 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V159 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V159 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "train-only tertile-symbolized causal path, duplicate-balanced label-free ordered channel-pair lagged transition tables, pointwise transfer-entropy summaries and one balanced ridge direction head",
            "rejected_prompt_candidate": {"hypothesis": "MULTICHANNEL_CAUSAL_RECURRENCE_QUANTIFICATION_DIRECTION_V1", "reason": "material exact collision with committed V120 recurrence plots and RQA line statistics"},
            "v90_supervised_tan_cmi_collision": False, "v107_field_kt_collision": False, "v120_rqa_collision_in_retained_model": False,
            "v144_ordinal_motif_collision": False, "v160_hilbert_phase_collision": False,
            "causal_numeric_only": True, "target_row_labels_used": False, "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True, "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            }, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V159 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2,
                "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"],
                "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop,
                "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"],
                "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
                "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"],
                "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False,
                "exact_entire_v69_fallback_contract": True, "output_written": False,
            }), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime,
        "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None,
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False,
    }
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]})
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
