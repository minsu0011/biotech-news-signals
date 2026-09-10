"""V162 cross-fitted temporal delete-block direction-stability confidence.

Atomic V69 probability and predicted direction remain byte-exact.  For each
strict-past same-market cut, immutable V36 causal numeric state is collapsed to
equal event groups and robust-scaled using training rows only.  Five fixed
chronological blocks define five four-of-five class-balanced ridge direction
heads.  Unlike V142, their coefficients are never intersected or aggregated
into a new direction.  Instead, row-level agreement with the frozen V69 side
and relative margin dispersion form an epistemic stability score.  A single
predeclared geometric formula combines stability with frozen V69 confidence;
its high-confidence cutoff is a train-only quantile matching the strict-past
V69 high-confidence coverage.  No V69 correctness or realized return enters
the stability model or cutoff.

V72/V148 fit correctness surfaces, V151 fits signed-return opportunity, V109
and V79 use conformal calibration, and V142 changes direction via coefficient
sign intersection.  V162 does none of those.  Inner OOF labels choose only
exact V69 noop versus this one fixed confidence policy.  Outer labels are
evaluation-only under the 35-minute embargo/event purge.  Any material failure
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v142_chronological_delete_block_sign_intersection as delete_block
import experiment_v148_mondrian_uniform_hoeffding_tail_correctness_lower_bound_selector as contract


ROOT = contract.ROOT
V69_DIR = contract.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V162" / "LOCAL_FORBIDDEN"
VERSION = 162
HYPOTHESIS = "CROSS_FITTED_TEMPORAL_DELETE_BLOCK_DIRECTION_STABILITY_CONFIDENCE_V1"
EXPECTED_MARKET_FOLD_ROWS = contract.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = contract.EXPECTED_V69_EXPERIMENT_ID
CONTRACT_PATH = ROOT / "experiment_v148_mondrian_uniform_hoeffding_tail_correctness_lower_bound_selector.py"
EXPECTED_CONTRACT_SHA256 = "23974cd61b160e394dcb86711b6b26020d66d6da1b1096e494132a41a0fa8215"
DELETE_BLOCK_PATH = ROOT / "experiment_v142_chronological_delete_block_sign_intersection.py"
EXPECTED_DELETE_BLOCK_SHA256 = "96e72094387e7da1fd45f7249e9b3196d31a6ced9666ddde1054c126149fdc4c"

FEATURES = delete_block.FEATURES
EMBARGO = contract.EMBARGO
INNER_VALID_FRACTION = contract.INNER_VALID_FRACTION
BLOCKS = 5
MIN_BLOCK_SUPPORT = 30
LOGISTIC_C = 0.5
ROBUST_CLIP = 7.0
CONFIDENCE_EPSILON = 1e-6
MIN_CALIBRATED_COVERAGE = 0.05
MAX_CALIBRATED_COVERAGE = 0.30
COST = 0.002
SEED = 16201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 64
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 16
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "DELETE_BLOCK_STABILITY_CONFIDENCE_FULL"},)

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
        "strict-past-equal-event-groups|37-causal-numeric-plus-37-missing|"
        "five-fixed-chronological-blocks|five-four-of-five-balanced-ridge-heads-C0.5|"
        "atomic-v69-side-signed-head-margins|sign-agreement-times-exp-negative-relative-MAD|"
        "sqrt-frozen-v69-confidence-times-stability|train-only-v69-HC-coverage-matched-quantile-clipped5to30pct|"
        "no-correctness-no-return-no-conformal-no-direction-change|noop-versus-one-full"
    )
    require(len(FEATURES) == 37 and BLOCKS == 5 and MIN_BLOCK_SUPPORT == 30, "V162 fixed block spec changed")
    require(LOGISTIC_C == 0.5 and ROBUST_CLIP == 7.0 and len(ARCHITECTURES) == 1, "V162 fixed head spec changed")
    require(MIN_CALIBRATED_COVERAGE == 0.05 and MAX_CALIBRATED_COVERAGE == 0.30, "V162 fixed coverage support changed")
    return {
        "input": "37 immutable causal pre-event numeric fields plus deterministic missing flags",
        "training_unit": "equal event group",
        "heads": "five fixed four-of-five chronological delete-block class-balanced L2 logistic heads, C=.5, no intercept",
        "row_stability": "fraction of head margins agreeing with frozen V69 side multiplied by exp(-MAD/(1+median absolute signed margin))",
        "confidence_formula": "sqrt(clipped frozen V69 confidence * clipped row stability)",
        "high_confidence_cutoff": "train-only combined-score quantile matching strict-past frozen V69 high-confidence coverage, clipped to fixed [5%,30%] support range",
        "v69_correctness_used": False, "realized_return_used": False,
        "conformal_or_jackknife_plus_calibration": False,
        "coefficient_intersection_or_direction_ensemble": False,
        "probability_and_direction_structurally_frozen": True,
        "candidate_count_excluding_noop": 1,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "pinned V148 contract scaffold changed")
    require(sha256(DELETE_BLOCK_PATH) == EXPECTED_DELETE_BLOCK_SHA256, "pinned V142 delete-block utility changed")
    audit = contract.verify_authority()
    audit["code_dependencies"] = [
        {"path": CONTRACT_PATH.name, "sha256": EXPECTED_CONTRACT_SHA256, "purpose": "immutable authority, chronology, metrics, canonical gate, bootstrap schema, atomic IO; no V148 output is read"},
        {"path": DELETE_BLOCK_PATH.name, "sha256": EXPECTED_DELETE_BLOCK_SHA256, "purpose": "fixed 37-field robust state and deterministic balanced ridge utilities only; no V142 output is read"},
    ]
    audit["v162_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
        "shared_registry_controller_or_staging_read_for_model": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = contract.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V162 bounded affinity changed")
    return audit


def attach_atomic(dev: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    aligned = aligned_dev(dev, rows).copy()
    require(np.array_equal(aligned.event_id.astype(str), rows.event_id.astype(str)), "V162 atomic alignment order changed")
    for field in ("prob", "confidence_signal", "high_conf", "source_family"):
        aligned[field] = rows[field].to_numpy()
    aligned["high_conf"] = bool_series(aligned.high_conf)
    return aligned


def collapse_equal_event_groups(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["predicted_side"] = (work.prob.to_numpy(float) >= 0.5).astype(int)
    grouped = work.groupby("event_group_id", sort=True)
    for field in ("y", "predicted_side"):
        require(int(grouped[field].nunique().max()) == 1, f"V162 event group crosses {field}")
    aggregate: dict[str, str] = {
        "event_id": "first", "event_time_utc": "min", "market": "first", "y": "first",
        "prob": "mean", "confidence_signal": "mean", "high_conf": "first", "predicted_side": "first",
    }
    aggregate.update({feature: "mean" for feature in FEATURES})
    result = grouped.agg(aggregate).reset_index().sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(result.event_group_id.is_unique, "V162 event-group collapse invalid")
    return result


def chronological_block_ids(rows: int) -> np.ndarray:
    require(rows >= BLOCKS * MIN_BLOCK_SUPPORT, "V162 chronological block support too small")
    block = np.minimum((np.arange(rows) * BLOCKS) // rows, BLOCKS - 1).astype(int)
    require(np.all(np.bincount(block, minlength=BLOCKS) >= MIN_BLOCK_SUPPORT), "V162 block support changed")
    return block


def fit_delete_block_heads(design: np.ndarray, target: np.ndarray, block: np.ndarray, base_weight: np.ndarray | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coefficients: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    base = np.ones(len(target), float) if base_weight is None else np.asarray(base_weight, float)
    require(base.shape == target.shape and np.all(base > 0.0) and np.isfinite(base).all(), "V162 base weights invalid")
    for deleted in range(BLOCKS):
        keep = block != deleted
        weight = delete_block.class_balanced_weights(target[keep], base[keep])
        coefficient, audit = delete_block.fit_one_head(design[keep], target[keep], weight)
        coefficients.append(coefficient)
        audit.update({"deleted_block": deleted, "train_n": int(keep.sum()), "deleted_n": int((~keep).sum())})
        audits.append(audit)
    matrix = np.vstack(coefficients)
    require(matrix.shape == (BLOCKS, design.shape[1]) and np.isfinite(matrix).all(), "V162 coefficient matrix invalid")
    return matrix, audits


def row_stability(head_logits: np.ndarray, frozen_probability: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    frozen_side = np.where(np.asarray(frozen_probability, float) >= 0.5, 1.0, -1.0)
    signed = np.asarray(head_logits, float) * frozen_side[:, None]
    agreement = np.mean(signed >= 0.0, axis=1)
    center = np.median(signed, axis=1)
    dispersion = np.median(np.abs(signed - center[:, None]), axis=1)
    relative_dispersion = dispersion / (1.0 + np.median(np.abs(signed), axis=1))
    stability = np.clip(agreement * np.exp(-relative_dispersion), CONFIDENCE_EPSILON, 1.0)
    require(np.isfinite(stability).all(), "V162 stability invalid")
    return stability, {
        "agreement_q10_q50_q90": [float(np.quantile(agreement, q)) for q in (0.1, 0.5, 0.9)],
        "relative_dispersion_q10_q50_q90": [float(np.quantile(relative_dispersion, q)) for q in (0.1, 0.5, 0.9)],
        "stability_q10_q50_q90": [float(np.quantile(stability, q)) for q in (0.1, 0.5, 0.9)],
        "signed_margin_sha256": array_sha256(signed), "stability_sha256": array_sha256(stability),
    }


def combine_confidence(frozen_confidence: np.ndarray, stability: np.ndarray) -> np.ndarray:
    confidence = np.asarray(frozen_confidence, float)
    require(np.isfinite(confidence).all(), "V162 frozen confidence invalid")
    combined = np.sqrt(np.clip(confidence, CONFIDENCE_EPSILON, 1.0) * np.clip(stability, CONFIDENCE_EPSILON, 1.0))
    require(np.isfinite(combined).all(), "V162 combined confidence invalid")
    return combined


def coverage_cutoff(reference_combined: np.ndarray, reference_high: np.ndarray) -> tuple[float, float]:
    raw_coverage = float(np.mean(np.asarray(reference_high, bool)))
    coverage = float(np.clip(raw_coverage, MIN_CALIBRATED_COVERAGE, MAX_CALIBRATED_COVERAGE))
    cutoff = float(np.quantile(np.asarray(reference_combined, float), 1.0 - coverage, method="higher"))
    require(math.isfinite(cutoff), "V162 coverage cutoff invalid")
    return cutoff, coverage


def native_stability_bootstrap(
    design: np.ndarray,
    target: np.ndarray,
    block: np.ndarray,
    reference_probability: np.ndarray,
    reference_confidence: np.ndarray,
    reference_high: np.ndarray,
    target_design: np.ndarray,
    target_probability: np.ndarray,
    target_confidence: np.ndarray,
    nominal_confidence: np.ndarray,
    nominal_high: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any] | None:
    if draws <= 0:
        return None
    generator = np.random.default_rng(seed)
    confidence_draws = np.empty((len(target_design), draws), float)
    high_draws = np.empty((len(target_design), draws), bool)
    coefficient_hashes: list[str] = []
    for draw in range(draws):
        coefficient, _ = fit_delete_block_heads(design, target, block, generator.exponential(1.0, len(target)))
        reference_stability, _ = row_stability(design @ coefficient.T, reference_probability)
        target_stability, _ = row_stability(target_design @ coefficient.T, target_probability)
        reference_combined = combine_confidence(reference_confidence, reference_stability)
        confidence_draws[:, draw] = combine_confidence(target_confidence, target_stability)
        cutoff, _ = coverage_cutoff(reference_combined, reference_high)
        high_draws[:, draw] = confidence_draws[:, draw] >= cutoff
        coefficient_hashes.append(array_sha256(coefficient))
    return {
        "contract": "equal-event-group Bayesian exponential-weight refit of all five fixed delete-block heads; train-only coverage cutoff recomputed; target labels absent",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "all_five_heads_refit": True, "target_labels_or_returns_used": False,
        "v69_correctness_used": False, "direction_probability_changed": False,
        "median_absolute_confidence_deviation": float(np.median(np.abs(confidence_draws - nominal_confidence[:, None]))),
        "mean_high_confidence_agreement": float(np.mean(high_draws == nominal_high[:, None])),
        "confidence_draw_sha256": array_sha256(confidence_draws),
        "high_draw_sha256": array_sha256(high_draws),
        "coefficient_inventory_sha256": hashlib.sha256("\n".join(coefficient_hashes).encode("utf-8")).hexdigest(),
    }


def stability_selector(dev: pd.DataFrame, reference: pd.DataFrame, target_rows: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_frame = attach_atomic(dev, reference)
    target_frame = attach_atomic(dev, target_rows)
    groups = collapse_equal_event_groups(reference_frame)
    scaler = delete_block.scaffold.RobustNumericMissingState().fit(groups)
    design, train_transform = scaler.transform(groups)
    target_design, target_transform = scaler.transform(target_frame)
    target = groups.y.to_numpy(int)
    block = chronological_block_ids(len(groups))
    coefficients, head_audits = fit_delete_block_heads(design, target, block)
    reference_stability, reference_stability_audit = row_stability(design @ coefficients.T, groups.prob.to_numpy(float))
    target_stability, target_stability_audit = row_stability(target_design @ coefficients.T, target_frame.prob.to_numpy(float))
    reference_combined = combine_confidence(groups.confidence_signal.to_numpy(float), reference_stability)
    combined = combine_confidence(target_frame.confidence_signal.to_numpy(float), target_stability)
    reference_high = bool_series(groups.high_conf).to_numpy(bool)
    raw_reference_coverage = float(np.mean(reference_high))
    cutoff, reference_coverage = coverage_cutoff(reference_combined, reference_high)
    high = combined >= cutoff
    native = native_stability_bootstrap(
        design, target, block, groups.prob.to_numpy(float), groups.confidence_signal.to_numpy(float),
        bool_series(groups.high_conf).to_numpy(bool), target_design, target_frame.prob.to_numpy(float),
        target_frame.confidence_signal.to_numpy(float), combined, high, native_draws, native_seed,
    )
    return combined, high, {
        "architecture": "five temporal delete-block direction heads used only for row-level epistemic stability confidence",
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_only": True,
        "reference_row_n": len(reference), "reference_event_group_n": len(groups), "target_n": len(target_frame),
        "equal_event_group_training": True, "chronological_block_counts": np.bincount(block, minlength=BLOCKS).astype(int).tolist(),
        "train_transform": train_transform, "target_transform": target_transform,
        "head_audits": head_audits, "coefficient_matrix_sha256": array_sha256(coefficients),
        "reference_stability_audit": reference_stability_audit, "target_stability_audit": target_stability_audit,
        "raw_reference_v69_highconf_coverage": raw_reference_coverage,
        "effective_reference_coverage": reference_coverage, "coverage_cutoff": cutoff,
        "target_highconf_coverage": float(np.mean(high)),
        "reference_combined_sha256": array_sha256(reference_combined),
        "target_confidence_sha256": array_sha256(combined), "target_highconf_sha256": array_sha256(high),
        "native_stability_bootstrap": native,
        "probability_exact_v69": True, "predicted_direction_exact_v69": True,
        "target_rows_used_for_scaler_head_or_cutoff": False, "target_labels_used": False,
        "v69_correctness_labels_used": False, "realized_returns_used": False,
        "coverage_cutoff_uses_labels_or_returns": False,
        "coefficient_sign_intersection_or_direction_ensemble": False,
        "conformal_residual_or_jackknife_plus_interval": False,
        "correctness_lcb_beta_binomial_pav_or_expectile": False,
    }


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference = champion.loc[champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference) >= 180, f"V162 {label} reference too small")
    reference_valid = aligned_dev(dev, reference)
    audit = chronology(reference_valid, target_valid, f"V162 {label}")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF IDs joined to immutable V36 causal state"
    audit["target_labels_used_for_fit"] = False
    return reference, audit


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} V69 OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_reference, audit = strict_reference(dev, champion, market, inner_valid, f"inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_valid, inner_champion, inner_reference, audit


def confidence_score(values: dict[str, Any]) -> float:
    fields = (values["confidence_correctness_auc"], values["highconf_accuracy"], values["highconf_mean_signed_net"])
    if any(value is None for value in fields):
        return -1e9
    return float(fields[0] + 0.25 * fields[1] + 5.0 * fields[2])


def choose_inner(dev: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    probability = inner_champion.prob.to_numpy(float)
    baseline_confidence = inner_champion.confidence_signal.to_numpy(float)
    baseline_high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, probability, baseline_confidence, baseline_high)
    stability_confidence, stability_high, model_audit = stability_selector(dev, inner_reference, inner_champion)
    values = metric(inner_valid, probability, stability_confidence, stability_high)
    minimum_high_n = max(12, int(math.ceil(0.05 * len(inner_valid))))
    eligible = bool(
        values["highconf_n"] >= minimum_high_n
        and values["highconf_accuracy"] is not None and values["highconf_mean_signed_net"] is not None
        and (baseline_metric["highconf_accuracy"] is None or values["highconf_accuracy"] >= baseline_metric["highconf_accuracy"] - 0.01)
        and (baseline_metric["highconf_mean_signed_net"] is None or values["highconf_mean_signed_net"] >= baseline_metric["highconf_mean_signed_net"] - 0.0005)
    )
    candidate_delta = None if values["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else values["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"]
    trials = [
        {"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "confidence_auc_delta": 0.0, "highconf_accuracy_delta": 0.0, "highconf_net_delta": 0.0, "eligible": True, "score": confidence_score(baseline_metric)},
        {"name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": values,
         "confidence_auc_delta": candidate_delta,
         "highconf_accuracy_delta": None if baseline_metric["highconf_accuracy"] is None else values["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
         "highconf_net_delta": None if baseline_metric["highconf_mean_signed_net"] is None else values["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
         "eligible": eligible, "minimum_highconf_n": minimum_high_n, "score": confidence_score(values)},
    ]
    selected = max((trial for trial in trials if trial["eligible"]), key=lambda trial: (trial["score"], trial["confidence_auc_delta"] or 0.0, trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past OOF only; exact V69 confidence/high_conf versus one fixed delete-block prediction-stability confidence; fixed coverage/accuracy/net safety; no threshold or formula tuning",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False, "threshold_formula_block_count_or_strength_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v162_model"] = "V69_NOOP"
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
        policy, inner_model_audit = choose_inner(dev, inner_valid, inner_champion, inner_reference)
        confidence, high, outer_model_audit = stability_selector(
            dev, outer_reference, outer_champion,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        probability = outer_champion.prob.to_numpy(float)
        baseline_confidence = outer_champion.confidence_signal.to_numpy(float)
        baseline_high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected_confidence = baseline_confidence if policy["selected"]["architecture"] is None else confidence
        selected_high = baseline_high if policy["selected"]["architecture"] is None else high
        baseline_metric = metric(outer_valid, probability, baseline_confidence, baseline_high)
        surface_metric = metric(outer_valid, probability, confidence, high)
        candidate_metric = metric(outer_valid, probability, selected_confidence, selected_high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        confidence_lookup = dict(zip(outer_valid.event_id, selected_confidence))
        high_lookup = dict(zip(outer_valid.event_id, selected_high))
        diagnostic.loc[positions, "confidence_signal"] = diagnostic.loc[positions, "event_id"].map(confidence_lookup)
        diagnostic.loc[positions, "high_conf"] = diagnostic.loc[positions, "event_id"].map(high_lookup).astype(bool)
        diagnostic.loc[positions, "v162_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m"]].copy()
        evidence["baseline_prob"] = probability
        evidence["candidate_prob"] = probability
        evidence["baseline_confidence"] = baseline_confidence
        evidence["candidate_confidence"] = selected_confidence
        evidence["surface_confidence"] = confidence
        evidence["baseline_high"] = baseline_high
        evidence["candidate_high"] = selected_high
        evidence["surface_high"] = high
        evidence["v162_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": {ARCHITECTURES[0]["name"]: surface_metric},
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_confidence_auc_delta": None if candidate_metric["confidence_correctness_auc"] is None or baseline_metric["confidence_correctness_auc"] is None else candidate_metric["confidence_correctness_auc"] - baseline_metric["confidence_correctness_auc"],
            "outer_highconf_accuracy_delta": None if candidate_metric["highconf_accuracy"] is None or baseline_metric["highconf_accuracy"] is None else candidate_metric["highconf_accuracy"] - baseline_metric["highconf_accuracy"],
            "outer_highconf_net_delta": None if candidate_metric["highconf_mean_signed_net"] is None or baseline_metric["highconf_mean_signed_net"] is None else candidate_metric["highconf_mean_signed_net"] - baseline_metric["highconf_mean_signed_net"],
            "outer_probability_exact_v69": True, "outer_direction_exact_v69": True,
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V162 DELETE-BLOCK STABILITY] market={market} fold={fold} selected={policy['selected']['name']} inner_conf_delta={policy['selected']['confidence_auc_delta'] or 0.0:+.6f} outer_conf_delta={audits[-1]['outer_confidence_auc_delta'] or 0.0:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V162 evidence repeats event")
    require(np.array_equal(champion.prob.to_numpy(float), diagnostic.prob.to_numpy(float)), "V162 changed V69 probability")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V162 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    keys = ("auc_delta", "balanced_accuracy_delta", "all_trade_net_delta", "confidence_auc_delta", "highconf_accuracy_delta", "highconf_net_delta")
    values: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        base_prob, candidate_prob = sample.baseline_prob.to_numpy(float), sample.candidate_prob.to_numpy(float)
        base_conf, candidate_conf = sample.baseline_confidence.to_numpy(float), sample.candidate_confidence.to_numpy(float)
        base_high, candidate_high = bool_series(sample.baseline_high).to_numpy(bool), bool_series(sample.candidate_high).to_numpy(bool)
        correct = (base_prob >= 0.5) == target
        signed = np.where(base_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float) - COST
        values["auc_delta"].append(float(roc_auc_score(target, candidate_prob) - roc_auc_score(target, base_prob)))
        values["balanced_accuracy_delta"].append(float(balanced_accuracy_score(target, candidate_prob >= 0.5) - balanced_accuracy_score(target, base_prob >= 0.5)))
        values["all_trade_net_delta"].append(float(np.mean(np.where(candidate_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float)) - np.mean(np.where(base_prob >= 0.5, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float))))
        if np.unique(correct).size == 2 and base_high.any() and candidate_high.any():
            values["confidence_auc_delta"].append(float(roc_auc_score(correct, candidate_conf) - roc_auc_score(correct, base_conf)))
            values["highconf_accuracy_delta"].append(float(correct[candidate_high].mean() - correct[base_high].mean()))
            values["highconf_net_delta"].append(float(signed[candidate_high].mean() - signed[base_high].mean()))
    require(len(values["auc_delta"]) >= int(0.9 * draws), "V162 bootstrap lost standard draws")
    require(len(values["confidence_auc_delta"]) >= int(0.8 * draws), "V162 bootstrap lost confidence draws")
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "paired market-fold-time block bootstrap over inner-locked V162 confidence policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V162 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V162 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.baseline_confidence, bool_series(evidence.baseline_high))
    candidate = metric(evidence, evidence.candidate_prob, evidence.candidate_confidence, bool_series(evidence.candidate_high))
    require(candidate["balanced_accuracy"] == base["balanced_accuracy"] and candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"], "V162 evaluated direction changed")
    nested = paired_nested_bootstrap(evidence, draws)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    probability_exact = bool(np.array_equal(champion.prob.to_numpy(float), diagnostic_original.prob.to_numpy(float)))
    models = [audit["outer_model_audit"] for audit in audits]
    stability_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_only"]
        and item["probability_exact_v69"] and item["predicted_direction_exact_v69"]
        and item["equal_event_group_training"] and len(item["head_audits"]) == BLOCKS
        and item["reference_event_group_n"] >= BLOCKS * MIN_BLOCK_SUPPORT
        and not item["target_rows_used_for_scaler_head_or_cutoff"] and not item["target_labels_used"]
        and not item["v69_correctness_labels_used"] and not item["realized_returns_used"]
        and not item["coverage_cutoff_uses_labels_or_returns"]
        and not item["coefficient_sign_intersection_or_direction_ensemble"]
        and not item["conformal_residual_or_jackknife_plus_interval"]
        and not item["correctness_lcb_beta_binomial_pav_or_expectile"]
        for item in models
    )
    native = [item["native_stability_bootstrap"] for item in models]
    native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and item["all_five_heads_refit"]
        and not item["target_labels_or_returns_used"] and not item["v69_correctness_used"]
        and not item["direction_probability_changed"] for item in native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    confidence_delta = None if candidate["confidence_correctness_auc"] is None or base["confidence_correctness_auc"] is None else candidate["confidence_correctness_auc"] - base["confidence_correctness_auc"]
    high_accuracy_delta = None if candidate["highconf_accuracy"] is None or base["highconf_accuracy"] is None else candidate["highconf_accuracy"] - base["highconf_accuracy"]
    high_net_delta = None if candidate["highconf_mean_signed_net"] is None or base["highconf_mean_signed_net"] is None else candidate["highconf_mean_signed_net"] - base["highconf_mean_signed_net"]
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_probability_and_direction_exact_v69": probability_exact and all(audit["outer_direction_exact_v69"] for audit in audits),
        "outer_balanced_accuracy_exact_v69": candidate["balanced_accuracy"] == base["balanced_accuracy"],
        "outer_all_trade_net_exact_v69": candidate["all_trade_mean_signed_net"] == base["all_trade_mean_signed_net"],
        "outer_confidence_correctness_auc_delta_gt_0_01": confidence_delta is not None and confidence_delta > 0.01,
        "outer_highconf_accuracy_delta_ge_0": high_accuracy_delta is not None and high_accuracy_delta >= 0.0,
        "outer_highconf_net_delta_ge_0": high_net_delta is not None and high_net_delta >= 0.0,
        "candidate_highconf_coverage_ge_0_05": candidate["highconf_n"] >= int(math.ceil(0.05 * candidate["n"])),
        "nested_bootstrap_confidence_auc_probability_gt_zero_ge_0_75": nested["confidence_auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_highconf_accuracy_probability_gt_zero_ge_0_65": nested["highconf_accuracy_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_highconf_net_probability_gt_zero_ge_0_65": nested["highconf_net_delta"]["probability_gt_zero"] >= 0.65,
        "nested_bootstrap_ba_delta_exact_zero": all(nested["balanced_accuracy_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "nested_bootstrap_net_delta_exact_zero": all(nested["all_trade_net_delta"][key] == 0.0 for key in ("lower95", "median", "upper95")),
        "full_overall_ba_exact_v69": candidate_metrics["balanced_accuracy"] == base_metrics["balanced_accuracy"],
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_stability_bootstrap_verified": native_valid,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "delete_block_stability_confidence_contract_verified": stability_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V162 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": 0.0, "outer_ba_delta": 0.0, "outer_net_delta": 0.0,
        "outer_confidence_auc_delta": confidence_delta,
        "outer_highconf_accuracy_delta": high_accuracy_delta, "outer_highconf_net_delta": high_net_delta,
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_stability_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "immutability": {"probability_exact": probability_exact, "predicted_direction_exact": all(audit["outer_direction_exact_v69"] for audit in audits)},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_stability_bootstrap": evaluation["model_native_stability_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V162 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V162 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V162 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V162").resolve(), "V162 output outside controller staging")
    require(out.exists() and out.is_dir(), "V162 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V162 controller prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V162 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V162 selected native robustness missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V162", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V162_DIAGNOSTIC_DELETE_BLOCK_STABILITY": evaluation["candidate_summary"], "V162_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V162", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V162", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V162_DELETE_BLOCK_DIRECTION_STABILITY_CONFIDENCE_REPORT.json": {"version": "V162", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V162", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V162_DELETE_BLOCK_STABILITY_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V162_DIAGNOSTIC_DELETE_BLOCK_STABILITY_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V162_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    policy, inner_audit = choose_inner(dev, inner_valid, inner_champion, inner_reference)
    outer_confidence, outer_high, outer_audit = stability_selector(dev, outer_reference, outer_champion)
    inner_probability, outer_probability = inner_champion.prob.to_numpy(float), outer_champion.prob.to_numpy(float)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold, "policy": policy,
        "inner_baseline": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_surface": policy["trials"][1]["metrics"],
        "outer_baseline": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_surface_evaluation_only": metric(outer_valid, outer_probability, outer_confidence, outer_high),
        "inner_model_audit": inner_audit, "outer_model_audit": outer_audit,
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used_for_fit_or_selection": False, "probability_and_direction_exact_v69": True, "output_written": False,
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V162 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V162 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
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
            "dev_rows": len(dev), "v69_rows": len(champion), "static_spec": STATIC_SPEC,
            "architecture": "five strict-past temporal delete-block heads provide sign-agreement and margin-dispersion stability only; frozen V69 probability/direction",
            "causal_pre_event_only": True, "outer_retuning": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True,
                "model_native_stability_bootstrap": True, "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            }, "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "output_written": False,
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
        print(json.dumps(clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS, "preparation_affinity": affinity,
            "threadpools": threadpool_info(), "gpu_model_calls": 0, "folds": len(audits), "audits": audits,
            "outer_baseline": evaluation["outer_baseline"], "outer_candidate": evaluation["outer_candidate"],
            "outer_confidence_auc_delta": evaluation["outer_confidence_auc_delta"],
            "outer_highconf_accuracy_delta": evaluation["outer_highconf_accuracy_delta"],
            "outer_highconf_net_delta": evaluation["outer_highconf_net_delta"],
            "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"],
            "fallback": evaluation["fallback"], "output_written": False,
        }), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
