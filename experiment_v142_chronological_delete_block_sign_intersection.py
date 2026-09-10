"""V142 chronological delete-block sign-intersection direction challenger.

Every strict-past same-market cut is collapsed to equal event-group rows and
mapped to train-robust 37 causal numeric fields plus 37 missing flags.  Five
fixed chronological blocks define five leave-one-block-out class-balanced ridge
logistic heads.  A feature survives only when all five heads give it the same
nonzero sign; its final magnitude is the minimum absolute delete-block
coefficient.  This conservative sign-intersection lower envelope has no final
refit, intercept, source/environment router, worst-group loss, or hyperparameter
grid.  A fixed train-only robust score scale gives a direct probability.

Unlike V94, heads train on overlapping four-of-five histories and no geometric
median is formed.  Unlike V91/V140, no anchor residual projection or statistical
random-effects meta-analysis is used.  Unlike V81/V131/V136 there is no worst
environment, maximin pairwise, or CVaR objective.  One inner-only joint paired
time-block lower-bound safety check chooses exact V69 noop or the sole full
challenger.  Outer labels are evaluation-only under a strict 35-minute embargo
and event purge; V69 confidence/high_conf stay exact and material failure
restores the exact entire atomic V69 frame.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v138_atomic_v69_offset_logistic_causal_state_correction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V142_CHRONOLOGICAL_DELETE_BLOCK_SIGN_INTERSECTION_V1"
VERSION = 142
HYPOTHESIS = "CHRONOLOGICAL_DELETE_BLOCK_SIGN_INTERSECTION_DIRECTION_V1"
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v138_atomic_v69_offset_logistic_causal_state_correction.py"
EXPECTED_SCAFFOLD_SHA256 = "848758c5920cd2821efd7727669e09bf8027b1c7fb0d98aa810dcc78722eb464"

FEATURES = scaffold.FEATURES
EMBARGO = scaffold.EMBARGO
INNER_VALID_FRACTION = scaffold.INNER_VALID_FRACTION
BLOCKS = 5
MIN_BLOCK_SUPPORT = 30
LOGISTIC_C = 0.5
SIGN_EPSILON = 1e-10
SCORE_SCALE_FLOOR = 0.25
PROBABILITY_EPSILON = 1e-5
SEED = 14201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
INNER_SAFETY_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 64
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 16
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "DELETE_BLOCK_SIGN_INTERSECTION_FULL"},)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
aligned_dev = scaffold.aligned_dev
chronology = scaffold.chronology
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "equal-event-group|37-causal-numeric-plus-37-missing|five-chronological-blocks|min-block-support-30|"
        "five-delete-one-block-four-of-five-ridge-logistic-heads|unanimous-nonzero-sign|"
        "minimum-absolute-coefficient-lower-envelope|no-intercept-no-final-refit|"
        "train-only-robust-score-scale|noop-versus-one-full"
    )
    require(len(FEATURES) == 37 and BLOCKS == 5 and MIN_BLOCK_SUPPORT == 30 and LOGISTIC_C == 0.5, "V142 static spec changed")
    require(len(ARCHITECTURES) == 1, "V142 architecture grid forbidden")
    return {
        "causal_numeric_n": len(FEATURES), "missing_flag_n": len(FEATURES),
        "state_dimension": 2 * len(FEATURES), "chronological_blocks": BLOCKS,
        "minimum_event_groups_per_block": MIN_BLOCK_SUPPORT,
        "delete_block_heads": BLOCKS, "rows_per_head": "four of five chronological blocks",
        "head": "class-balanced L2 logistic C=0.5 liblinear without intercept",
        "aggregation": "unanimous nonzero coefficient sign intersection with minimum absolute magnitude",
        "final_refit": False, "score_scale": "train-only IQR/1.349 with standard deviation fallback and floor 0.25",
        "architecture_candidates_excluding_noop": 1,
        "blend_regularization_block_count_or_threshold_grid": False,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V138 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust numeric state, metrics, canonical gate, bootstrap and atomic IO only; no V138 output is read",
    }
    audit["v142_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
        "v136_staging_read_or_written": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = scaffold.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V142 bounded affinity changed")
    return audit


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference_champion = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference_champion) >= 120, f"V142 {label} reference too small")
    reference = aligned_dev(dev, reference_champion)
    audit = chronology(reference, target_valid, f"V142 {label} delete-block sign intersection")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF IDs joined to immutable V36 causal state"
    audit["target_labels_used_for_fit"] = False
    return reference, audit


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} V69 OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_reference, audit = strict_reference(dev, champion, market, inner_valid, f"inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_valid, inner_champion, inner_reference, audit


def collapse_event_groups(reference: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_group_id", "event_time_utc", "market", "y", *FEATURES]
    work = reference[columns].copy()
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V142 event group crosses labels")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "y": "first",
        **{feature: "mean" for feature in FEATURES},
    }).reset_index().sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)
    require(result.event_group_id.is_unique, "V142 event group collapse invalid")
    return result


def class_balanced_weights(target: np.ndarray, base_weight: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(target, int)
    weight = np.ones(len(target), float) if base_weight is None else np.asarray(base_weight, float).copy()
    require(weight.shape == target.shape and np.all(weight > 0) and np.isfinite(weight).all(), "V142 weights invalid")
    masses = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require(np.all(masses > 0), "V142 class support missing")
    total = float(weight.sum())
    weight *= np.where(target == 1, total / (2.0 * masses[1]), total / (2.0 * masses[0]))
    return weight / float(np.mean(weight))


def chronological_block_ids(rows: int) -> np.ndarray:
    require(rows >= BLOCKS * MIN_BLOCK_SUPPORT, "V142 chronological block support too small")
    block = np.minimum((np.arange(rows) * BLOCKS) // rows, BLOCKS - 1).astype(int)
    counts = np.bincount(block, minlength=BLOCKS)
    require(np.all(counts >= MIN_BLOCK_SUPPORT), "V142 chronological block size changed")
    return block


def fit_one_head(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(np.unique(target).size == 2 and design.shape == (len(target), 2 * len(FEATURES)), "V142 head input invalid")
    model = LogisticRegression(
        penalty="l2", C=LOGISTIC_C, fit_intercept=False, solver="liblinear",
        tol=1e-8, max_iter=500, random_state=SEED,
    )
    model.fit(design, target, sample_weight=weight)
    coefficient = np.asarray(model.coef_[0], float)
    require(np.isfinite(coefficient).all() and int(model.n_iter_[0]) < 500, "V142 ridge head failed")
    return coefficient, {
        "iterations": int(model.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(coefficient)),
        "coefficient_sha256": array_sha256(coefficient),
    }


def fit_sign_intersection(design: np.ndarray, target: np.ndarray, block: np.ndarray, base_weight: np.ndarray | None = None) -> tuple[np.ndarray, float, dict[str, Any]]:
    coefficients: list[np.ndarray] = []
    head_audits: list[dict[str, Any]] = []
    base = np.ones(len(target), float) if base_weight is None else np.asarray(base_weight, float)
    for deleted in range(BLOCKS):
        keep = block != deleted
        weight = class_balanced_weights(target[keep], base[keep])
        coefficient, audit = fit_one_head(design[keep], target[keep], weight)
        coefficients.append(coefficient)
        audit.update({"deleted_block": deleted, "train_n": int(keep.sum()), "deleted_n": int((~keep).sum())})
        head_audits.append(audit)
    matrix = np.vstack(coefficients)
    signs = np.sign(matrix)
    stable = (np.abs(matrix) > SIGN_EPSILON).all(axis=0) & (np.abs(signs.sum(axis=0)) == BLOCKS)
    coefficient = np.zeros(matrix.shape[1], float)
    coefficient[stable] = signs[0, stable] * np.min(np.abs(matrix[:, stable]), axis=0)
    require(int(stable.sum()) > 0 and np.isfinite(coefficient).all(), "V142 sign intersection empty or invalid")
    train_score = design @ coefficient
    lower, upper = np.quantile(train_score, [0.25, 0.75])
    robust = float((upper - lower) / 1.349)
    fallback = float(np.std(train_score))
    scale = max(robust if math.isfinite(robust) and robust > 1e-9 else fallback, SCORE_SCALE_FLOOR)
    sign_inventory = ["".join("+" if value > 0 else "-" if value < 0 else "0" for value in row) for row in signs]
    return coefficient, scale, {
        "head_audits": head_audits,
        "coefficient_matrix_sha256": array_sha256(matrix),
        "sign_inventory_sha256": hashlib.sha256("\n".join(sign_inventory).encode("utf-8")).hexdigest(),
        "stable_feature_n": int(stable.sum()), "stable_feature_fraction": float(stable.mean()),
        "unstable_feature_n": int((~stable).sum()), "coefficient_l2": float(np.linalg.norm(coefficient)),
        "coefficient_sha256": array_sha256(coefficient), "train_score_scale": scale,
        "minimum_absolute_magnitude_rule": True, "unanimous_sign_rule": True,
        "final_refit": False, "intercept_fitted": False,
    }


def native_sign_bootstrap(groups: pd.DataFrame, scaler: Any, design: np.ndarray, target: np.ndarray, block: np.ndarray, target_frame: pd.DataFrame, nominal: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    target_design, _ = scaler.transform(target_frame)
    samples = np.empty((len(target_frame), draws), float)
    retained: list[int] = []
    hashes: list[str] = []
    for draw in range(draws):
        coefficient, scale, audit = fit_sign_intersection(design, target, block, generator.exponential(1.0, len(groups)))
        samples[:, draw] = np.clip(expit((target_design @ coefficient) / scale), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        retained.append(audit["stable_feature_n"])
        hashes.append(audit["coefficient_sha256"])
    std = np.std(samples, axis=1)
    return {
        "contract": "equal-event-group Bayesian bootstrap refit of all five delete-block heads with label-free scaler and chronological boundaries frozen",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "all_delete_block_heads_refit": True, "target_labels_used": False,
        "chronological_boundaries_refit_or_tuned": False, "label_free_scaler_refit": False,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(samples - nominal[:, None]))),
        "row_probability_std_q10_q50_q90": [float(np.quantile(std, q)) for q in (0.1, 0.5, 0.9)],
        "stable_feature_n_q10_q50_q90": [float(np.quantile(retained, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(np.median(samples, axis=1)),
        "coefficient_hash_inventory_sha256": hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest(),
    }


def sign_intersection_direction(reference: pd.DataFrame, target_frame: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, dict[str, Any]]:
    require(reference.market.nunique() == target_frame.market.nunique() == 1 and str(reference.market.iloc[0]) == str(target_frame.market.iloc[0]), "V142 market mismatch")
    groups = collapse_event_groups(reference)
    scaler = scaffold.RobustNumericMissingState().fit(groups)
    design, train_transform = scaler.transform(groups)
    target = groups.y.to_numpy(int)
    block = chronological_block_ids(len(groups))
    coefficient, scale, fit_audit = fit_sign_intersection(design, target, block)
    target_design, target_transform = scaler.transform(target_frame)
    raw_score = target_design @ coefficient
    probability = np.clip(expit(raw_score / scale), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    native = None if native_draws <= 0 else native_sign_bootstrap(groups, scaler, design, target, block, target_frame, probability, native_draws, native_seed)
    return probability, {
        "market": str(reference.market.iloc[0]), "reference_row_n": len(reference),
        "reference_event_group_n": len(groups), "target_n": len(target_frame),
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_ids_only": True,
        "equal_event_group_training": True, "causal_numeric_feature_n": len(FEATURES),
        "missing_flag_n": len(FEATURES), "source_text_ticker_issuer_return_or_v69_probability_feature_n": 0,
        "architecture": "five leave-one-chronological-block-out ridge heads with unanimous sign intersection and minimum absolute coefficient lower envelope",
        "static_spec": STATIC_SPEC, "train_transform": train_transform, "target_transform": target_transform,
        "chronological_block_counts": np.bincount(block, minlength=BLOCKS).astype(int).tolist(),
        "fit_audit": fit_audit, "native_sign_intersection_bootstrap": native,
        "raw_score_q10_q50_q90": [float(np.quantile(raw_score, q)) for q in (0.1, 0.5, 0.9)],
        "probability_sha256": array_sha256(probability),
        "target_rows_used_for_scaler_head_or_sign_selection": False, "target_labels_used": False,
        "delete_block_boundaries_label_independent": True, "worst_environment_group_dro_maximin_or_cvar_objective": False,
        "chronological_anchor_residual_projection": False, "geometric_median_or_random_effects_meta_analysis": False,
        "v69_offset_source_confidence_reliability_or_probability_calibration": False,
        "blend_regularization_block_count_or_sign_threshold_grid": False,
    }


def delta_metric(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, high: np.ndarray) -> dict[str, float]:
    target = frame.y.to_numpy(int)
    returns = frame.fwd_ret_30m.to_numpy(float)
    base_direction, candidate_direction = baseline >= 0.5, candidate >= 0.5
    result = {
        "auc_delta": float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)),
        "balanced_accuracy_delta": float(balanced_accuracy_score(target, candidate_direction) - balanced_accuracy_score(target, base_direction)),
        "all_trade_net_delta": float(np.mean(np.where(candidate_direction, 1.0, -1.0) * returns) - np.mean(np.where(base_direction, 1.0, -1.0) * returns)),
    }
    if int(high.sum()) > 0:
        result["highconf_accuracy_delta"] = float(accuracy_score(target[high], candidate_direction[high]) - accuracy_score(target[high], base_direction[high]))
        result["highconf_net_delta"] = float(np.mean(np.where(candidate_direction[high], 1.0, -1.0) * returns[high]) - np.mean(np.where(base_direction[high], 1.0, -1.0) * returns[high]))
    return result


def inner_joint_lower_bounds(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, high: np.ndarray, draws: int = INNER_SAFETY_DRAWS) -> dict[str, Any]:
    work = frame[["event_time_utc", "y", "fwd_ret_30m"]].copy().reset_index(drop=True)
    work["baseline"] = np.asarray(baseline, float)
    work["candidate"] = np.asarray(candidate, float)
    work["high"] = np.asarray(high, bool)
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    market = str(frame.market.iloc[0])
    work["block"] = timestamp.dt.strftime("%Y-%m") if market == "US" else timestamp.dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby("block", sort=True)]
    if len(blocks) < 4:
        return {
            "method": "paired chronological time-block bootstrap on inner-past validation only",
            "seed": SEED + (0 if market == "US" else 1000), "requested_draws": draws,
            "blocks": len(blocks), "highconf_n": int(high.sum()), "intervals": None,
            "fixed_joint_lower_bound_rule": {
                "auc_delta_lower95_gt": 0.0, "balanced_accuracy_delta_lower95_ge": 0.0,
                "all_trade_net_delta_lower95_ge": 0.0, "highconf_min_n": 10,
                "highconf_accuracy_delta_lower95_ge": -0.005, "highconf_net_delta_lower95_ge": -0.0005,
            },
            "eligible": False,
            "fail_closed_reason": "fewer than four inner chronological validation blocks; support probe cannot authorize a material policy",
        }
    generator = np.random.default_rng(SEED + (0 if market == "US" else 1000))
    values = {key: [] for key in ("auc_delta", "balanced_accuracy_delta", "all_trade_net_delta", "highconf_accuracy_delta", "highconf_net_delta")}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))], ignore_index=True)
        if sample.y.nunique() != 2:
            continue
        delta = delta_metric(sample, sample.baseline.to_numpy(float), sample.candidate.to_numpy(float), sample.high.to_numpy(bool))
        for key, value in delta.items():
            values[key].append(value)
    intervals: dict[str, Any] = {}
    for key, numbers in values.items():
        array = np.asarray(numbers, float)
        intervals[key] = None if len(array) == 0 else {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }
    enough_high = int(high.sum()) >= 10 and intervals["highconf_accuracy_delta"] is not None
    eligible = bool(
        intervals["auc_delta"]["lower95"] > 0.0
        and intervals["balanced_accuracy_delta"]["lower95"] >= 0.0
        and intervals["all_trade_net_delta"]["lower95"] >= 0.0
        and enough_high
        and intervals["highconf_accuracy_delta"]["lower95"] >= -0.005
        and intervals["highconf_net_delta"]["lower95"] >= -0.0005
    )
    return {
        "method": "paired chronological time-block bootstrap on inner-past validation only",
        "seed": SEED + (0 if market == "US" else 1000), "requested_draws": draws,
        "blocks": len(blocks), "highconf_n": int(high.sum()), "intervals": intervals,
        "fixed_joint_lower_bound_rule": {
            "auc_delta_lower95_gt": 0.0, "balanced_accuracy_delta_lower95_ge": 0.0,
            "all_trade_net_delta_lower95_ge": 0.0, "highconf_min_n": 10,
            "highconf_accuracy_delta_lower95_ge": -0.005, "highconf_net_delta_lower95_ge": -0.0005,
        },
        "eligible": eligible,
    }


def choose_inner(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = inner_champion.prob.to_numpy(float)
    candidate, model_audit = sign_intersection_direction(inner_reference, inner_valid)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    base_metric = metric(inner_valid, baseline, confidence, high)
    candidate_metric = metric(inner_valid, candidate, confidence, high)
    raw_delta = delta_metric(inner_valid, baseline, candidate, high)
    safety = inner_joint_lower_bounds(inner_valid, baseline, candidate, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": base_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True,
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": candidate_metric,
        **raw_delta, "eligible": safety["eligible"], "inner_joint_lower_bounds": safety,
    }]
    selected = trials[1] if trials[1]["eligible"] else trials[0]
    return {
        "selection_rule": "inner-past only; exact V69 noop versus one fixed delete-block sign-intersection challenger; full challenger requires all predeclared paired chronological lower-bound guards",
        "baseline": base_metric, "selected": selected, "trials": trials,
        "inner_joint_lower_bounds": safety, "outer_labels_used_for_selection": False,
        "block_count_logistic_C_sign_threshold_or_policy_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v142_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    folds = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else folds):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, f"outer {market} fold {fold}")
        inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_valid, inner_champion, inner_reference)
        full_probability, outer_model_audit = sign_intersection_direction(
            outer_reference, outer_valid,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        baseline = outer_champion.prob.to_numpy(float)
        candidate = baseline.copy() if policy["selected"]["architecture"] is None else full_probability
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        base_metric = metric(outer_valid, baseline, confidence, high)
        full_metric = metric(outer_valid, full_probability, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        lookup = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(lookup)
        diagnostic.loc[positions, "v142_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf"]].copy()
        evidence["baseline_prob"] = baseline
        evidence["full_sign_intersection_prob"] = full_probability
        evidence["candidate_prob"] = candidate
        evidence["v142_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_full_architecture_evaluation_only": full_metric,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V142 SIGN INTERSECTION] market={market} fold={fold} selected={policy['selected']['name']} inner_auc_delta={policy['trials'][1]['auc_delta']:+.6f} outer_full_auc_delta={full_metric['auc'] - base_metric['auc']:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V142 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V142 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V142 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V142 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        values["balanced_accuracy_delta"].append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)))
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "paired market-fold-time block bootstrap over inner-locked V142 sign-intersection policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V142 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V142 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    confidence_exact = bool(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    model_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_ids_only"] and item["equal_event_group_training"]
        and item["delete_block_boundaries_label_independent"] and not item["fit_audit"]["final_refit"]
        and not item["fit_audit"]["intercept_fitted"] and item["fit_audit"]["unanimous_sign_rule"]
        and item["fit_audit"]["minimum_absolute_magnitude_rule"] and not item["target_rows_used_for_scaler_head_or_sign_selection"]
        and not item["target_labels_used"] and not item["worst_environment_group_dro_maximin_or_cvar_objective"]
        and not item["geometric_median_or_random_effects_meta_analysis"] and not item["blend_regularization_block_count_or_sign_threshold_grid"]
        for item in full_models
    )
    native = [item["native_sign_intersection_bootstrap"] for item in full_models]
    native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and item["all_delete_block_heads_refit"]
        and not item["target_labels_used"] and not item["chronological_boundaries_refit_or_tuned"]
        for item in native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_sign_intersection_bootstrap_verified": native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "delete_block_sign_intersection_contract_verified": model_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V142 fallback not exact V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"], "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_sign_intersection_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_sign_intersection_bootstrap": evaluation["model_native_sign_intersection_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V142 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V142 output not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V142").resolve(), "V142 output outside controller staging")
    require(out.exists() and out.is_dir(), "V142 controller stage missing")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing) and not (existing - allowed), "V142 controller prefile conflict")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    reports = {
        "MODEL_COMPARISON.json": {"version": "V142", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V142_DIAGNOSTIC_SIGN_INTERSECTION": evaluation["candidate_summary"], "V142_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V142", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V142", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V142_DELETE_BLOCK_SIGN_INTERSECTION_REPORT.json": {"version": "V142", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V142", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V142_SIGN_INTERSECTION_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V142_DIAGNOSTIC_SIGN_INTERSECTION_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V142_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    policy, inner_audit = choose_inner(inner_valid, inner_champion, inner_reference)
    inner_probability, _ = sign_intersection_direction(inner_reference, inner_valid)
    outer_probability, outer_audit = sign_intersection_direction(outer_reference, outer_valid)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold, "policy": policy,
        "inner_baseline": metric(inner_valid, inner_champion.prob, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_full_sign_intersection": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "outer_baseline": metric(outer_valid, outer_champion.prob, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_full_sign_intersection_evaluation_only": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "inner_model_audit": inner_audit, "outer_model_audit": outer_audit,
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used_for_fit_or_selection": False, "output_written": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe-kr2", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V142 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V142 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
    affinity = preparation_affinity() if bounded else {"reason": "controller-authorized full run is not bounded preparation"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "five four-of-five chronological delete-block ridge heads plus unanimous sign/minimum-magnitude coefficient intersection",
            "causal_pre_event_only": True, "group_dro_maximin_cvar_or_hyperparameter_grid": False,
            "controller_no_arg_contract": {"environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run", "output_bound_to_environment": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True, "model_native_sign_intersection_bootstrap": True, "exact_entire_v69_fallback": True, "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"]},
            "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "output_written": False,
        }), indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe(dev, champion)
            result["preparation_affinity"] = affinity
            result["threadpools"] = threadpool_info()
            result["gpu_model_calls"] = 0
            print(json.dumps(clean(result), indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, args.smoke_test)
    if args.smoke_test:
        print(json.dumps(clean({"status": "SMOKE_OK", "hypothesis": HYPOTHESIS, "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "folds": len(audits), "audits": audits, "outer_baseline": evaluation["outer_baseline"], "outer_candidate": evaluation["outer_candidate"], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"], "fallback": evaluation["fallback"], "output_written": False}), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
