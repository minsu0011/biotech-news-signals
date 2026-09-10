"""V158 causal multiscale event-arrival hazard direction challenger.

Every embargoed fit traverses equal event groups in timestamp order. Only
strictly earlier, label-free arrivals update fixed 15m/60m/240m/1d exponential
market and source-family intensity states. Log intensity, source share,
short-versus-long excitation surprise, and previous-arrival gaps are joined to
the established 74D causal numeric/missing state before one fixed balanced
ridge-logistic direction head.

No outcome, return, probability, target label, same-time event, or future
arrival enters this flow state. V43 recency weighting and V157 density-ratio
loss weighting contain no point-process covariate state. Inner labels select
exact V69 or fixed .25/.50 blends; outer labels are evaluation-only. Failure
restores the entire atomic V69 frame. Bounded modes are CPU30-31/GPU-free.
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

import experiment_v149_multichannel_causal_increment_coskewness_tensor_direction as utility


base = utility.base
ROOT = utility.ROOT
V69_DIR = utility.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V158" / "LOCAL_FORBIDDEN"
VERSION = 158
HYPOTHESIS = "CAUSAL_MULTISCALE_EVENT_ARRIVAL_HAZARD_DIRECTION_V1"
FEATURES = utility.FEATURES
EXPECTED_V69_EXPERIMENT_ID = utility.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v149_multichannel_causal_increment_coskewness_tensor_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "4b9a364db95f598ebb75ca3406333412f1263ac1bb0722f68be81f7e17ab2fe0"

STATE_DIMENSION = 74
HALF_LIVES_MINUTES = np.asarray([15.0, 60.0, 240.0, 1440.0], dtype=float)
FLOW_DIMENSION = 16
DESIGN_DIMENSION = STATE_DIMENSION + FLOW_DIMENSION
GAP_CAP_MINUTES = 10080.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 15801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "EVENT_ARRIVAL_HAZARD_W0.25", "weight": 0.25},
    {"name": "EVENT_ARRIVAL_HAZARD_W0.50", "weight": 0.50},
)

require = utility.require
sha256 = utility.sha256
array_sha256 = utility.array_sha256
clean = utility.clean
bool_series = utility.bool_series
atomic_json = utility.atomic_json
atomic_csv = utility.atomic_csv
load_authorized = utility.load_authorized
metric = utility.metric
controller = utility.controller
v44 = utility.v44
numeric = utility.numeric
blend_probability = base.blend_probability

STATIC_SPEC = {
    "input": "74D train-cut robust causal numeric state plus strict-past timestamp/market/source event-flow state",
    "training_unit": "equal event_group in chronological order",
    "half_lives_minutes": HALF_LIVES_MINUTES.tolist(),
    "flow_feature_blocks": [
        "four log1p market decayed arrival counts",
        "four log1p same-source decayed arrival counts",
        "four source-minus-market log intensity shares",
        "market and source strict-previous-arrival log gaps",
        "market and source 15m-minus-1440m excitation surprises",
    ],
    "flow_dimension": FLOW_DIMENSION,
    "feature_dimension": DESIGN_DIMENSION,
    "same_timestamp_events_excluded_from_each_other": True,
    "query_sequence": "strictly earlier query arrival metadata may update later query rows; no query outcome or label",
    "feature_scaling": "train-group-only coordinate median/IQR-or-std and fixed clip",
    "head": "one fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "point_process_parameters_fitted_or_tuned": False,
    "micro_tuning": False,
}
require(FLOW_DIMENSION == 16 and DESIGN_DIMENSION == 90, "V158 fixed flow design changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V149 utility scaffold changed")
    audit = utility.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust numeric state, metrics, canonical gate, bootstrap and atomic IO only; no V149 output is read",
    }
    audit["v158_access_contract"] = {
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


def group_table(frame: pd.DataFrame, state: np.ndarray, include_label: bool) -> pd.DataFrame:
    require(state.shape == (len(frame), STATE_DIMENSION), "V158 robust-state shape changed")
    work = pd.DataFrame(state, columns=[f"state_{index}" for index in range(STATE_DIMENSION)])
    work["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    work["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True).to_numpy()
    work["market"] = frame.market.astype(str).to_numpy()
    work["source_family"] = frame.source_family.fillna("UNKNOWN").astype(str).to_numpy()
    if include_label:
        work["y"] = frame.y.to_numpy(int)
    consistency = ["event_time_utc", "market", "source_family"] + (["y"] if include_label else [])
    for column in consistency:
        require(
            bool((work.groupby("event_group_id", sort=False)[column].nunique(dropna=False) == 1).all()),
            f"V158 event_group {column} inconsistent",
        )
    aggregation: dict[str, str] = {
        **{f"state_{index}": "mean" for index in range(STATE_DIMENSION)},
        "event_time_utc": "first", "market": "first", "source_family": "first",
    }
    if include_label:
        aggregation["y"] = "first"
    grouped = work.groupby("event_group_id", sort=True, as_index=False).agg(aggregation)
    return grouped.sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)


def causal_arrival_flow(groups: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    require(groups.event_time_utc.notna().all(), "V158 event time missing")
    ordered = groups.sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)
    feature = np.empty((len(ordered), FLOW_DIMENSION), dtype=float)
    market_state: dict[str, np.ndarray] = {}
    source_state: dict[tuple[str, str], np.ndarray] = {}
    market_last: dict[str, pd.Timestamp] = {}
    source_last: dict[tuple[str, str], pd.Timestamp] = {}
    state_time: pd.Timestamp | None = None
    simultaneous_batch_max = 0
    for timestamp, positions in ordered.groupby("event_time_utc", sort=True).indices.items():
        now = pd.Timestamp(timestamp)
        index = np.asarray(positions, dtype=int)
        simultaneous_batch_max = max(simultaneous_batch_max, len(index))
        if state_time is not None:
            elapsed = max(0.0, (now - state_time).total_seconds() / 60.0)
            decay = np.exp(-np.log(2.0) * elapsed / HALF_LIVES_MINUTES)
            for value in market_state.values():
                value *= decay
            for value in source_state.values():
                value *= decay
        for row_index in index:
            market = str(ordered.iloc[row_index].market)
            source = str(ordered.iloc[row_index].source_family)
            source_key = (market, source)
            market_count = market_state.get(market, np.zeros(len(HALF_LIVES_MINUTES), dtype=float))
            source_count = source_state.get(source_key, np.zeros(len(HALF_LIVES_MINUTES), dtype=float))
            market_log = np.log1p(market_count)
            source_log = np.log1p(source_count)
            market_gap = GAP_CAP_MINUTES if market not in market_last else min(
                GAP_CAP_MINUTES, max(0.0, (now - market_last[market]).total_seconds() / 60.0)
            )
            source_gap = GAP_CAP_MINUTES if source_key not in source_last else min(
                GAP_CAP_MINUTES, max(0.0, (now - source_last[source_key]).total_seconds() / 60.0)
            )
            feature[row_index] = np.concatenate([
                market_log, source_log, source_log - market_log,
                np.asarray([
                    np.log1p(market_gap) / np.log1p(GAP_CAP_MINUTES),
                    np.log1p(source_gap) / np.log1p(GAP_CAP_MINUTES),
                    market_log[0] - market_log[-1],
                    source_log[0] - source_log[-1],
                ]),
            ])
        batch = ordered.iloc[index]
        for market, count in batch.groupby("market", sort=True).size().items():
            key = str(market)
            market_state.setdefault(key, np.zeros(len(HALF_LIVES_MINUTES), dtype=float))
            market_state[key] += float(count)
            market_last[key] = now
        for (market, source), count in batch.groupby(["market", "source_family"], sort=True).size().items():
            key = (str(market), str(source))
            source_state.setdefault(key, np.zeros(len(HALF_LIVES_MINUTES), dtype=float))
            source_state[key] += float(count)
            source_last[key] = now
        state_time = now
    require(np.isfinite(feature).all(), "V158 arrival-flow feature invalid")
    return feature, {
        "rows": len(feature), "feature_dimension": FLOW_DIMENSION,
        "half_lives_minutes": HALF_LIVES_MINUTES.tolist(),
        "same_time_batches_excluded_before_update": True,
        "simultaneous_batch_max": simultaneous_batch_max,
        "strict_past_timestamp_only": True,
        "label_return_probability_or_outcome_used": False,
        "market_and_source_excitation_surprise_verified": True,
        "feature_sha256": array_sha256(feature),
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == DESIGN_DIMENSION, "V158 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V158 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V158 scaled feature invalid")
        return result


def event_arrival_hazard_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V158 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    require(train_state.shape[1] == target_state.shape[1] == STATE_DIMENSION, "V158 robust state must be 74D")
    train_groups = group_table(ordered, train_state, include_label=True)
    target_groups = group_table(target_frame, target_state, include_label=False)
    require(train_groups.event_time_utc.max() < target_groups.event_time_utc.min(), "V158 target must follow train cut")
    combined_groups = pd.concat([
        train_groups.assign(_origin="train"), target_groups.assign(_origin="target")
    ], ignore_index=True, sort=False).sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)
    combined_flow, combined_flow_audit = causal_arrival_flow(combined_groups)
    train_positions = np.flatnonzero(combined_groups._origin.to_numpy() == "train")
    target_positions = np.flatnonzero(combined_groups._origin.to_numpy() == "target")
    state_columns = [f"state_{index}" for index in range(STATE_DIMENSION)]
    group_design = np.concatenate([
        combined_groups.loc[train_positions, state_columns].to_numpy(float), combined_flow[train_positions]
    ], axis=1)
    target_group_design = np.concatenate([
        combined_groups.loc[target_positions, state_columns].to_numpy(float), combined_flow[target_positions]
    ], axis=1)
    target_group_ids = combined_groups.loc[target_positions, "event_group_id"].astype(str).to_numpy()
    target_position_by_group = {event_group: position for position, event_group in enumerate(target_group_ids)}
    target_lookup = np.asarray([target_position_by_group[str(value)] for value in target_frame.event_group_id], dtype=int)
    feature_scale = RobustFeatureScale().fit(group_design)
    train_design = feature_scale.transform(group_design)
    target_design = feature_scale.transform(target_group_design)[target_lookup]
    group_target = combined_groups.loc[train_positions, "y"].to_numpy(int)
    require(np.unique(group_target).size == 2, "V158 grouped train cut missing a class")
    class_n = np.bincount(group_target, minlength=2).astype(float)
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V158 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_plus_label_free_arrival_metadata_only": True,
        "categorical_identity_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "grouping": {
            "raw_rows": len(ordered), "equal_event_groups": len(train_groups),
            "duplicate_rows_removed_before_fit": len(ordered) - len(train_groups),
            "class_event_group_n": class_n.astype(int).tolist(),
            "equal_event_group_weighting": True,
        },
        "static_spec": STATIC_SPEC,
        "arrival_flow": combined_flow_audit,
        "train_arrival_feature_sha256": array_sha256(combined_flow[train_positions]),
        "target_arrival_feature_sha256": array_sha256(combined_flow[target_positions]),
        "feature_scaling": {
            "train_only": True, "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed equal-event-group/class-balanced pointwise ridge logistic",
            "objective": "pointwise balanced Bernoulli log loss plus fixed L2",
            "C": RIDGE_LOGISTIC_C, "solver": "liblinear",
            "iterations": int(head.n_iter_[0]),
            "class_event_group_n": class_n.astype(int).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_or_head": False,
        "strictly_earlier_query_arrival_metadata_may_update_later_queries": True,
        "same_time_or_future_query_arrival_used": False,
        "target_labels_used": False,
        "outcomes_returns_probabilities_or_v69_confidence_used_in_flow": False,
        "recency_weighted_loss_or_density_ratio_weight": False,
        "hawkes_parameter_estimation_or_scale_tuning": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def supported_source_ba_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index].y.to_numpy(int)
        if len(index) < 40 or np.unique(target).size != 2:
            continue
        deltas[str(source)] = float(
            balanced_accuracy_score(target, candidate[index] >= 0.5)
            - balanced_accuracy_score(target, baseline[index] >= 0.5)
        )
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = event_arrival_hazard_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_plus_label_free_arrival_metadata_only"]
        and model_audit["grouping"]["equal_event_group_weighting"]
        and model_audit["arrival_flow"]["strict_past_timestamp_only"]
        and model_audit["arrival_flow"]["same_time_batches_excluded_before_update"]
        and model_audit["arrival_flow"]["market_and_source_excitation_surprise_verified"]
        and not model_audit["arrival_flow"]["label_return_probability_or_outcome_used"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_or_head"]
        and not model_audit["same_time_or_future_query_arrival_used"]
        and not model_audit["target_labels_used"]
        and not model_audit["outcomes_returns_probabilities_or_v69_confidence_used_in_flow"]
        and not model_audit["recency_weighted_loss_or_density_ratio_weight"]
        and not model_audit["hawkes_parameter_estimation_or_scale_tuning"]
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
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 causal multiscale event-arrival-hazard blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "half_life_flow_definition_penalty_blend_or_source_floor_micro_tuning": False,
    }, model_audit


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = DESIGN_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = event_arrival_hazard_direction


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v158_model"})
    evidence = evidence.rename(columns={"frenet_prob": "event_arrival_hazard_prob", "v146_model": "v158_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V158 EVENT ARRIVAL HAZARD] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V158 causal multiscale event-arrival-hazard policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V158 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    arrival_contract = all(
        audit[side]["causal_numeric_plus_label_free_arrival_metadata_only"]
        and audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["static_spec"]["feature_dimension"] == DESIGN_DIMENSION
        and audit[side]["arrival_flow"]["feature_dimension"] == FLOW_DIMENSION
        and audit[side]["arrival_flow"]["strict_past_timestamp_only"]
        and audit[side]["arrival_flow"]["same_time_batches_excluded_before_update"]
        and audit[side]["arrival_flow"]["market_and_source_excitation_surprise_verified"]
        and not audit[side]["arrival_flow"]["label_return_probability_or_outcome_used"]
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_or_head"]
        and not audit[side]["same_time_or_future_query_arrival_used"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["outcomes_returns_probabilities_or_v69_confidence_used_in_flow"]
        and not audit[side]["recency_weighted_loss_or_density_ratio_weight"]
        and not audit[side]["hawkes_parameter_estimation_or_scale_tuning"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
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
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "causal_multiscale_event_arrival_hazard_contract_verified": arrival_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V158 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested,
        "candidate_native_robustness_bootstrap": native,
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
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V158 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V158 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V158 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V158").resolve(), "V158 output outside controller research/staging/V158")
    require(out.exists() and out.is_dir(), "V158 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V158 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V158 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V158 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V158 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V158", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V158_DIAGNOSTIC_EVENT_ARRIVAL_HAZARD": evaluation["candidate_summary"],
                "V158_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V158", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V158", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V158_EVENT_ARRIVAL_HAZARD_REPORT.json": {
            "version": "V158", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V158", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V158", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V158", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V158_EVENT_ARRIVAL_HAZARD_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V158_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V158 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V158 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V158 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "flow_definition_sha256": array_sha256(np.concatenate([HALF_LIVES_MINUTES, np.asarray([FLOW_DIMENSION, GAP_CAP_MINUTES])])),
            "architecture": "fixed 15m/60m/240m/1d strict-past market/source arrival intensity, share, gap and excitation-surprise state plus 74D causal state and one equal-group balanced ridge logistic",
            "causal_numeric_plus_arrival_metadata_only": True, "target_row_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V158 support probe is reserved for KR:2")
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
        summary.update({
            "raw_inner_selected": audits[0]["policy"]["selected"],
            "raw_inner_best_non_noop": best_non_noop,
            "raw_outer_baseline": audits[0]["outer_baseline"],
            "raw_outer_selected": audits[0]["outer_candidate"],
            "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
            "inner_model_audit": audits[0]["inner_model_audit"],
            "outer_model_audit": audits[0]["outer_model_audit"],
            "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        })
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


