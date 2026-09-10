"""V109 class-conditional conformal causal-state typicality challenger.

Every same-market past training cut is split chronologically into an earlier
proper-fit segment and a later calibration segment, with an additional fixed
35-minute embargo.  A robust numeric transform is fit on proper-fit rows only.
Rows are then collapsed to equal-weight event-group centroids.  For each class
and causal coordinate, the proper-fit class median defines absolute-residual
nonconformity and later calibration residuals define an exact finite-sample
upper-tail conformal p-value.  Fixed semantic blocks combine log p-values with
equal block weight; the UP-minus-DOWN class-typicality contrast is converted to
probability with one train-calibration-only robust temperature.

There is no coefficient or learned head, covariance/density, discretized
symbol context, KT code, graph/topology, archetype/simplex reconstruction,
neighbor retrieval, tree, kernel, neural model, or return target.  Earlier
committed V69 OOF alone chooses exact V69 or fixed .25/.50 logit blends in the
inner past.  Outer labels are evaluation-only under strict 35-minute nested
chronology; V69 confidence/high_conf remain exact.  Material failure restores
the exact entire atomic V69 DataFrame after canonical 14-gate recomputation.
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
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V109_CLASS_CONDITIONAL_CONFORMAL_STATE_TYPICALITY_V1"
VERSION = 109
HYPOTHESIS = "CLASS_CONDITIONAL_CONFORMAL_STATE_TYPICALITY_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

PROPER_FRACTION = 0.70
TEMPERATURE_FLOOR = 0.05
SMOOTHING_COUNT = 1.0
SEED = 10901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

SEMANTIC_BLOCKS = {
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
    "INTRADAY_STRUCTURE": (
        "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
        "up_fraction_30", "trend_slope_30", "trend_slope_60", "vwap_distance_30",
    ),
    "CLOCK": ("minutes_from_open", "minutes_to_close"),
    "BENCHMARK": (
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "benchmark_ret_30", "benchmark_ret_60",
    ),
}
ARCHITECTURES = (
    {"name": "CONFORMAL_STATE_TYPICALITY_W0.25", "weight": 0.25},
    {"name": "CONFORMAL_STATE_TYPICALITY_W0.50", "weight": 0.50},
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
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_static_spec() -> dict[str, Any]:
    flattened = [name for block in SEMANTIC_BLOCKS.values() for name in block]
    require(len(flattened) == len(FEATURES), "V109 semantic blocks do not cover 37 coordinates")
    require(len(set(flattened)) == len(FEATURES), "V109 semantic blocks repeat a coordinate")
    require(set(flattened) == set(FEATURES), "V109 semantic blocks changed causal feature authority")
    payload = "\n".join(f"{block}|{','.join(names)}" for block, names in SEMANTIC_BLOCKS.items())
    return {
        "feature_n": len(FEATURES), "semantic_block_n": len(SEMANTIC_BLOCKS),
        "semantic_block_sizes": {name: len(items) for name, items in SEMANTIC_BLOCKS.items()},
        "block_assignment_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_features_exactly_once": True,
    }


STATIC_SPEC = verify_static_spec()
FEATURE_INDEX = {name: position for position, name in enumerate(FEATURES)}
BLOCK_INDICES = {
    block: np.asarray([FEATURE_INDEX[name] for name in names], dtype=int)
    for block, names in SEMANTIC_BLOCKS.items()
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, metrics, robust numeric transform, blending, canonical gate, and atomic IO only; no V85 output is read",
    }
    audit["v109_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def event_group_index(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("event_group_id", sort=False)
    require(int(grouped.y.nunique().max()) == 1, "V109 event group crosses direction labels")
    result = grouped.agg(event_time_utc=("event_time_utc", "max"), y=("y", "first")).reset_index()
    result["event_group_id"] = result.event_group_id.astype(str)
    result["event_time_utc"] = pd.to_datetime(result.event_time_utc, utc=True)
    return result.sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)


def proper_calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = event_group_index(train)
    split = int(math.floor(len(groups) * PROPER_FRACTION))
    require(80 <= split < len(groups) - 30, "V109 proper/calibration group split too small")
    calibration_start = pd.Timestamp(groups.iloc[split].event_time_utc)
    proper_groups = groups.loc[groups.event_time_utc < calibration_start - scaffold.EMBARGO].copy()
    calibration_groups = groups.iloc[split:].copy()
    require(len(proper_groups) >= 70 and len(calibration_groups) >= 30, "V109 embargoed proper/calibration support too small")
    proper_ids = set(proper_groups.event_group_id)
    calibration_ids = set(calibration_groups.event_group_id)
    proper_rows = train.loc[train.event_group_id.astype(str).isin(proper_ids)].copy()
    calibration_rows = train.loc[train.event_group_id.astype(str).isin(calibration_ids)].copy()
    require(proper_ids.isdisjoint(calibration_ids), "V109 proper/calibration group overlap")
    proper_max = pd.Timestamp(proper_rows.event_time_utc.max())
    calibration_min = pd.Timestamp(calibration_rows.event_time_utc.min())
    require(proper_max < calibration_min - scaffold.EMBARGO, "V109 internal conformal embargo failed")
    return proper_rows, calibration_rows, {
        "proper_group_n": len(proper_groups), "calibration_group_n": len(calibration_groups),
        "proper_row_n": len(proper_rows), "calibration_row_n": len(calibration_rows),
        "proper_max_time": proper_max.isoformat(), "calibration_min_time": calibration_min.isoformat(),
        "embargo_minutes": 35, "strict_35m_embargo": True,
        "event_group_disjoint": True, "chronological_split": True,
        "proper_fraction_fixed": PROPER_FRACTION,
    }


def group_centroids(frame: pd.DataFrame, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(frame) == len(matrix), "V109 centroid input shape mismatch")
    state = pd.DataFrame(matrix, columns=FEATURES)
    state["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    state["y"] = frame.y.to_numpy(int)
    grouped = state.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V109 centroid label mismatch")
    centroid = grouped[list(FEATURES)].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    require(np.isfinite(centroid).all(), "V109 group centroid is non-finite")
    return centroid, target, group_ids


def fit_class_reference(
    proper_state: np.ndarray, proper_y: np.ndarray,
    calibration_state: np.ndarray, calibration_y: np.ndarray, label: int,
) -> dict[str, Any]:
    fit = proper_state[proper_y == label]
    calibration = calibration_state[calibration_y == label]
    require(len(fit) >= 30 and len(calibration) >= 15, f"V109 class {label} support too small")
    center = np.median(fit, axis=0)
    residual = np.abs(calibration - center)
    sorted_residual = np.sort(residual, axis=0, kind="stable")
    require(np.isfinite(center).all() and np.isfinite(sorted_residual).all(), "V109 class reference invalid")
    return {
        "label": label, "center": center, "sorted_residual": sorted_residual,
        "proper_group_n": len(fit), "calibration_group_n": len(calibration),
        "minimum_p": float(SMOOTHING_COUNT / (len(calibration) + SMOOTHING_COUNT)),
        "center_sha256": array_sha256(center),
        "calibration_residual_sha256": array_sha256(sorted_residual),
    }


def conformal_p_values(matrix: np.ndarray, reference: dict[str, Any]) -> np.ndarray:
    residual = np.abs(matrix - reference["center"])
    sorted_residual = reference["sorted_residual"]
    count = sorted_residual.shape[0]
    result = np.empty_like(residual, dtype=float)
    for feature in range(residual.shape[1]):
        position = np.searchsorted(sorted_residual[:, feature], residual[:, feature], side="left")
        tail_count = count - position
        result[:, feature] = (SMOOTHING_COUNT + tail_count) / (SMOOTHING_COUNT + count)
    lower = float(reference["minimum_p"])
    require(np.isfinite(result).all() and float(result.min()) >= lower - 1e-15, "V109 conformal p-value invalid")
    require(float(result.max()) <= 1.0 + 1e-15, "V109 conformal p-value exceeds one")
    return result


def block_log_typicality(p_values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    block_scores = {
        block: np.mean(np.log(np.clip(p_values[:, indices], 1e-15, 1.0)), axis=1)
        for block, indices in BLOCK_INDICES.items()
    }
    overall = np.mean(np.column_stack(list(block_scores.values())), axis=1)
    require(np.isfinite(overall).all(), "V109 block typicality invalid")
    return overall, block_scores


def class_typicality_contrast(
    matrix: np.ndarray, down_reference: dict[str, Any], up_reference: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    down_p = conformal_p_values(matrix, down_reference)
    up_p = conformal_p_values(matrix, up_reference)
    down_score, down_blocks = block_log_typicality(down_p)
    up_score, up_blocks = block_log_typicality(up_p)
    contrast = up_score - down_score
    return contrast, {
        "down_p_min": float(down_p.min()), "up_p_min": float(up_p.min()),
        "down_p_sha256": array_sha256(down_p), "up_p_sha256": array_sha256(up_p),
        "down_block_mean": {name: float(np.mean(value)) for name, value in down_blocks.items()},
        "up_block_mean": {name: float(np.mean(value)) for name, value in up_blocks.items()},
    }


def conformal_state_typicality_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V109 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V109 market fit mismatch")
    proper_rows, calibration_rows, split_audit = proper_calibration_split(train)
    processor = scaffold.RobustNumericState().fit(proper_rows)
    proper_state, proper_y, proper_groups = group_centroids(proper_rows, processor.transform(proper_rows))
    calibration_state, calibration_y, calibration_groups = group_centroids(
        calibration_rows, processor.transform(calibration_rows),
    )
    target_state = processor.transform(target_frame)
    down_reference = fit_class_reference(proper_state, proper_y, calibration_state, calibration_y, 0)
    up_reference = fit_class_reference(proper_state, proper_y, calibration_state, calibration_y, 1)
    calibration_contrast, calibration_audit = class_typicality_contrast(
        calibration_state, down_reference, up_reference,
    )
    lower = float(np.quantile(calibration_contrast, 0.25))
    upper = float(np.quantile(calibration_contrast, 0.75))
    temperature = max((upper - lower) / 1.349, TEMPERATURE_FLOOR)
    target_contrast, target_audit = class_typicality_contrast(target_state, down_reference, up_reference)
    probability = np.clip(expit(target_contrast / temperature), 1e-5, 1.0 - 1e-5)
    require(np.isfinite(probability).all(), "V109 probability invalid")
    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES), "causal_numeric_only": True,
        "categorical_feature_n": 0, "text_feature_n": 0,
        "source_or_ticker_identity": False,
        "architecture": "proper/calibration class-conditional coordinate conformal p-values with equal semantic-block log typicality",
        "static_spec": STATIC_SPEC, "internal_split": split_audit,
        "proper_group_n": len(proper_groups), "calibration_group_n": len(calibration_groups),
        "proper_scaling_only": True, "class_reference_fit_on_proper_only": True,
        "conformal_residuals_fit_on_later_calibration_only": True,
        "event_group_equal_weight": True,
        "down_reference": {key: value for key, value in down_reference.items() if key not in {"center", "sorted_residual"}},
        "up_reference": {key: value for key, value in up_reference.items() if key not in {"center", "sorted_residual"}},
        "calibration_score_audit": calibration_audit,
        "target_score_audit": target_audit,
        "temperature": temperature, "temperature_fit_on_calibration_only": True,
        "temperature_floor": TEMPERATURE_FLOOR,
        "finite_sample_smoothing_count": SMOOTHING_COUNT,
        "fitted_direction_coefficient_n": 0, "learned_discriminative_head": False,
        "density_or_covariance_model": False, "symbol_context_or_kt_code": False,
        "graph_topology_or_message_passing": False, "archetype_or_simplex_code": False,
        "neighbor_tree_kernel_or_neural": False,
        "target_rows_used_for_scaling_center_residual_or_temperature": False,
        "target_labels_used": False,
        "calibration_contrast_quantiles": {
            str(q): float(np.quantile(calibration_contrast, q)) for q in quantiles
        },
        "target_contrast_quantiles": {str(q): float(np.quantile(target_contrast, q)) for q in quantiles},
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V109 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V109 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = conformal_state_typicality_direction(inner_train, inner_valid)
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
        model_audit["internal_split"]["strict_35m_embargo"]
        and model_audit["internal_split"]["event_group_disjoint"]
        and model_audit["proper_scaling_only"]
        and model_audit["class_reference_fit_on_proper_only"]
        and model_audit["conformal_residuals_fit_on_later_calibration_only"]
        and model_audit["down_reference"]["calibration_group_n"] >= 15
        and model_audit["up_reference"]["calibration_group_n"] >= 15
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["target_rows_used_for_scaling_center_residual_or_temperature"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 conformal-state-typicality logit blend; fixed score and BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v109_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V109 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = conformal_state_typicality_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v109_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["conformal_state_typicality_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v109_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V109 CONFORMAL TYPICALITY] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V109 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V109 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V109 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V109 nested-bootstrap blocks")
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
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        ba_delta.append(float(
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V109 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V109 policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V109/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    conformal_valid = all(
        audit["outer_model_audit"]["causal_numeric_only"]
        and audit["outer_model_audit"]["internal_split"]["strict_35m_embargo"]
        and audit["outer_model_audit"]["internal_split"]["event_group_disjoint"]
        and audit["outer_model_audit"]["proper_scaling_only"]
        and audit["outer_model_audit"]["class_reference_fit_on_proper_only"]
        and audit["outer_model_audit"]["conformal_residuals_fit_on_later_calibration_only"]
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_center_residual_or_temperature"]
        and not audit["outer_model_audit"]["target_labels_used"]
        for audit in audits
    )
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
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "class_conditional_conformal_typicality_contract_verified": conformal_valid,
    }
    material_pass = bool(all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V109 fallback is not exact entire V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
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
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V109 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V109 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V109 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V109 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V109", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V109_DIAGNOSTIC_CONFORMAL_STATE_TYPICALITY": evaluation["candidate_summary"],
                "V109_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V109", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V109", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "CLASS_CONDITIONAL_CONFORMAL_STATE_TYPICALITY_REPORT.json": {
            "version": "V109", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "semantic_blocks": SEMANTIC_BLOCKS,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V109", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": scaffold.now(),
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
        "proper_group_n": audit["outer_model_audit"]["proper_group_n"],
        "calibration_group_n": audit["outer_model_audit"]["calibration_group_n"],
        "strict_outer_35m_embargo": audit["outer_chronology"]["strict_35m_embargo"],
        "strict_internal_35m_embargo": audit["outer_model_audit"]["internal_split"]["strict_35m_embargo"],
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V109_CONFORMAL_STATE_TYPICALITY_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V109_CONFORMAL_STATE_TYPICALITY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V109_DIAGNOSTIC_CONFORMAL_STATE_TYPICALITY_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V109_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V109 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "chronologically split class-conditional conformal state typicality",
            "proper_fraction": PROPER_FRACTION, "internal_embargo_minutes": 35,
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_identity": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "exact_entire_v69_fallback": True,
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0,
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
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "fold_keys": [f"{audit['market']}:{audit['fold']}" for audit in audits],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "inner_baseline": audits[0]["policy"]["baseline"],
        "inner_selected": audits[0]["policy"]["selected"],
        "inner_best_non_noop": best_non_noop,
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_evaluation_only_architectures": audits[0]["outer_architecture_diagnostics_evaluation_only"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "model_audit": audits[0]["outer_model_audit"],
        "current_material_gate": current_material_gate(evaluation),
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    output = write_outputs(args.output, authority, evidence, audits, evaluation)
    summary["output_written"] = True
    summary["output"] = output
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
