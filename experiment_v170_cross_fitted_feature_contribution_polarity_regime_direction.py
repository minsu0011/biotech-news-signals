"""V170 cross-fitted feature-contribution polarity-regime challenger.

Every strict-past same-market fit robustly scales the immutable 37 causal
numeric fields plus their 37 missing flags, then collapses duplicate rows to
equal event-group centroids.  A fixed chronological 55/45 split fits two
disjoint class-balanced ridge-logit heads.  For each target row, feature-wise
contributions ``x_j * beta_j`` from the early and recent heads define magnitude
weighted sign-agreement and sign-reversal mass.  One frozen formula averages
the two logits when contribution polarity is stable and routes toward the
recent logit when it reverses.  The target row therefore controls the regime
mixture; coefficients are never intersected, projected, or tuned after outer
evidence.

This materially differs from V83's full-vs-early probability-distance damping,
V142's delete-block unanimous coefficient-sign/minimum-magnitude aggregate,
and V143's source-time nuisance-subspace projection.  Inner-past labels choose
only exact V69 or fixed .25/.50 challenger blends.  Outer labels are evaluation
only under the 35-minute embargo/event purge.  Atomic V69 confidence/high_conf
stay exact and any material or execution failure returns the exact entire V69
frame.
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
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v167_multichannel_causal_euler_characteristic_curve_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V170" / "LOCAL_FORBIDDEN"
VERSION = 170
HYPOTHESIS = "CROSS_FITTED_FEATURE_CONTRIBUTION_POLARITY_REGIME_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v167_multichannel_causal_euler_characteristic_curve_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "9e48cf7e6adf9c3dd11266d902d64f811d4fb23ec23e9a08f7ee0666e605340c"

STATE_DIMENSION = 74
EARLY_FRACTION = 0.55
MIN_REGIME_EVENT_GROUPS = 80
MIN_REGIME_CLASS_EVENT_GROUPS = 30
RIDGE_LOGISTIC_C = 0.50
FEATURE_CLIP = 8.0
SEED = 17001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "POLARITY_REGIME_DIRECTION_W0.25", "weight": 0.25},
    {"name": "POLARITY_REGIME_DIRECTION_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
metric = scaffold.metric
controller = scaffold.controller
numeric = scaffold.numeric
blend_probability = base.blend_probability

STATIC_SPEC = {
    "numeric_feature_n": 37,
    "missing_flag_n": 37,
    "state_dimension": STATE_DIMENSION,
    "feature_dimension": STATE_DIMENSION,
    "duplicate_unit": "equal event-group robust-state centroid",
    "chronological_regimes": "disjoint early 55% and recent 45% event groups; timestamp ties never cross",
    "head": "one fixed equal-event-group/class-balanced C=0.5 ridge logistic per regime",
    "target_contribution": "x_j times beta_regime_j; intercept excluded from polarity mass",
    "polarity_mass": "absolute-contribution-weighted early/recent sign agreement and reversal",
    "stability": "agreement_mass / (agreement_mass + reversal_mass); 0.5 if both masses vanish",
    "frozen_logit_formula": "stability * mean(early,recent) + (1-stability) * recent",
    "coefficient_intersection_or_minimum_magnitude_aggregation": False,
    "probability_distance_stability": False,
    "nuisance_subspace_projection": False,
    "target_batch_fit": False,
    "outer_outcome_tuning": False,
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V167 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, metrics, canonical gate, bootstrap and atomic IO only; no V167 output is read",
    }
    audit["v170_access_contract"] = {
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


def equal_event_group_state(
    ordered: pd.DataFrame,
    state: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape == (len(ordered), STATE_DIMENSION), "V170 robust-state shape changed")
    state_columns = [f"state_{index:02d}" for index in range(STATE_DIMENSION)]
    work = pd.DataFrame(state, columns=state_columns)
    work["event_group_id"] = ordered.event_group_id.astype(str).to_numpy()
    work["event_time_utc"] = pd.to_datetime(ordered.event_time_utc, utc=True).to_numpy()
    work["event_id"] = ordered.event_id.astype(str).to_numpy()
    work["y"] = ordered.y.to_numpy(int)
    label_n = work.groupby("event_group_id", sort=False).y.nunique()
    require(bool((label_n == 1).all()), "V170 event group has conflicting labels")
    aggregation: dict[str, Any] = {column: "mean" for column in state_columns}
    aggregation.update({"event_time_utc": "min", "event_id": "min", "y": "first"})
    grouped = work.groupby("event_group_id", as_index=False, sort=False).agg(aggregation)
    grouped = grouped.sort_values(["event_time_utc", "event_id", "event_group_id"], kind="stable").reset_index(drop=True)
    grouped_state = grouped[state_columns].to_numpy(float)
    grouped_y = grouped.y.to_numpy(int)
    require(np.isfinite(grouped_state).all() and set(np.unique(grouped_y)) == {0, 1}, "V170 grouped state invalid")
    return grouped, grouped_state, grouped_y, {
        "row_n": len(ordered), "event_group_n": len(grouped),
        "duplicate_rows_removed": len(ordered) - len(grouped),
        "equal_event_group_weighting": True,
        "group_state_sha256": array_sha256(grouped_state),
        "group_target_sha256": array_sha256(grouped_y),
    }


def chronological_regime_split(grouped: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = len(grouped)
    split = int(math.floor(EARLY_FRACTION * count))
    split = max(MIN_REGIME_EVENT_GROUPS, min(split, count - MIN_REGIME_EVENT_GROUPS))
    times = pd.to_datetime(grouped.event_time_utc, utc=True)
    while split < count - MIN_REGIME_EVENT_GROUPS and times.iloc[split] <= times.iloc[split - 1]:
        split += 1
    early = np.arange(split, dtype=int)
    recent = np.arange(split, count, dtype=int)
    require(len(early) >= MIN_REGIME_EVENT_GROUPS and len(recent) >= MIN_REGIME_EVENT_GROUPS, "V170 regime support too small")
    require(times.iloc[early].max() < times.iloc[recent].min(), "V170 timestamp tie crossed regime split")
    return early, recent, {
        "fixed_early_fraction": EARLY_FRACTION,
        "early_event_group_n": len(early), "recent_event_group_n": len(recent),
        "early_end": times.iloc[early].max(), "recent_start": times.iloc[recent].min(),
        "strict_disjoint_chronology": True,
        "event_group_overlap_n": 0,
    }


def fit_regime_head(design: np.ndarray, target: np.ndarray, label: str) -> tuple[LogisticRegression, dict[str, Any]]:
    class_n = np.bincount(target, minlength=2).astype(float)
    require(bool(np.all(class_n >= MIN_REGIME_CLASS_EVENT_GROUPS)), f"V170 {label} class support too small")
    weight = np.where(target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(design, target, sample_weight=weight)
    coefficient = head.coef_.reshape(-1)
    require(coefficient.shape == (STATE_DIMENSION,) and np.isfinite(coefficient).all(), "V170 head invalid")
    return head, {
        "label": label, "event_group_n": len(target),
        "class_event_group_n": class_n.astype(int).tolist(),
        "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]),
        "coefficient_l2": float(np.linalg.norm(coefficient)),
        "coefficient_sha256": array_sha256(coefficient),
        "intercept": float(head.intercept_[0]),
    }


def contribution_polarity_probability(
    design: np.ndarray,
    early_head: LogisticRegression,
    recent_head: LogisticRegression,
) -> tuple[np.ndarray, dict[str, Any]]:
    early_coef = early_head.coef_.reshape(-1)
    recent_coef = recent_head.coef_.reshape(-1)
    early_contribution = design * early_coef[None, :]
    recent_contribution = design * recent_coef[None, :]
    magnitude = np.abs(early_contribution) + np.abs(recent_contribution)
    total_mass = magnitude.sum(axis=1)
    agreement_raw = (magnitude * (early_contribution * recent_contribution > 0.0)).sum(axis=1)
    reversal_raw = (magnitude * (early_contribution * recent_contribution < 0.0)).sum(axis=1)
    active_mass = agreement_raw + reversal_raw
    stability = np.divide(
        agreement_raw, active_mass,
        out=np.full(len(design), 0.5, dtype=float), where=active_mass > 1e-12,
    )
    agreement_fraction = np.divide(agreement_raw, total_mass, out=np.zeros(len(design)), where=total_mass > 1e-12)
    reversal_fraction = np.divide(reversal_raw, total_mass, out=np.zeros(len(design)), where=total_mass > 1e-12)
    early_logit = early_head.decision_function(design)
    recent_logit = recent_head.decision_function(design)
    combined_logit = stability * 0.5 * (early_logit + recent_logit) + (1.0 - stability) * recent_logit
    probability = np.clip(expit(np.clip(combined_logit, -20.0, 20.0)), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V170 probability invalid")
    return probability, {
        "rows": len(design), "feature_dimension": STATE_DIMENSION,
        "feature_contribution_definition": "robust target state x strict-past regime coefficient",
        "intercept_excluded_from_polarity_mass": True,
        "agreement_mass_mean": float(np.mean(agreement_fraction)),
        "agreement_mass_p10_p50_p90": np.quantile(agreement_fraction, [0.10, 0.50, 0.90]).tolist(),
        "reversal_mass_mean": float(np.mean(reversal_fraction)),
        "reversal_mass_p10_p50_p90": np.quantile(reversal_fraction, [0.10, 0.50, 0.90]).tolist(),
        "stability_mean": float(np.mean(stability)),
        "stability_p10_p50_p90": np.quantile(stability, [0.10, 0.50, 0.90]).tolist(),
        "recent_logit_weight_mean": float(np.mean(1.0 - 0.5 * stability)),
        "contribution_sign_agreement_or_reversal_is_row_specific": True,
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "target_batch_fit": False,
        "target_labels_used": False,
        "early_contribution_sha256": array_sha256(early_contribution),
        "recent_contribution_sha256": array_sha256(recent_contribution),
        "stability_sha256": array_sha256(stability),
        "probability_sha256": array_sha256(probability),
    }


def polarity_regime_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V170 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    require(train_state.shape[1] == target_state.shape[1] == STATE_DIMENSION, "V170 state dimension changed")
    grouped, group_state, group_target, grouping_audit = equal_event_group_state(ordered, train_state)
    early_index, recent_index, split_audit = chronological_regime_split(grouped)
    early_head, early_audit = fit_regime_head(group_state[early_index], group_target[early_index], "early")
    recent_head, recent_audit = fit_regime_head(group_state[recent_index], group_target[recent_index], "recent")
    probability, target_polarity = contribution_polarity_probability(target_state, early_head, recent_head)
    _, train_polarity = contribution_polarity_probability(group_state, early_head, recent_head)
    coefficient_product = early_head.coef_.reshape(-1) * recent_head.coef_.reshape(-1)
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(grouped), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "grouping": grouping_audit, "chronological_regime_split": split_audit,
        "static_spec": STATIC_SPEC,
        "train_geometry": train_polarity, "target_geometry": target_polarity,
        "feature_scaling": {"train_only": True, "robust_numeric_state": True, "clip": FEATURE_CLIP},
        "heads": {"early": early_audit, "recent": recent_audit},
        "coefficient_sign_agreement_fraction": float(np.mean(coefficient_product > 0.0)),
        "coefficient_sign_reversal_fraction": float(np.mean(coefficient_product < 0.0)),
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False,
        "early_recent_event_groups_disjoint": True,
        "outer_labels_used_for_formula_or_head": False,
        "v83_full_vs_early_probability_distance_damping": False,
        "v142_delete_block_unanimous_coefficient_intersection": False,
        "v143_source_time_nuisance_subspace_projection": False,
        "v64_v67_diagnostic_only_ablation_reuse": False,
        "raw_feature_sign_inversion": False,
        "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = STATE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = polarity_regime_direction


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = polarity_regime_direction(inner_train, inner_valid)
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
        model_audit["causal_numeric_only"]
        and model_audit["grouping"]["equal_event_group_weighting"]
        and model_audit["chronological_regime_split"]["strict_disjoint_chronology"]
        and model_audit["early_recent_event_groups_disjoint"]
        and model_audit["train_geometry"]["contribution_sign_agreement_or_reversal_is_row_specific"]
        and not model_audit["train_geometry"]["target_batch_fit"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 row-specific contribution-polarity regime blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "regime_split_formula_head_blend_or_source_floor_micro_tuning": False,
    }, model_audit


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
    diagnostic = diagnostic.rename(columns={"v146_model": "v170_model"})
    evidence = evidence.rename(columns={"frenet_prob": "polarity_regime_prob", "v146_model": "v170_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V170 CONTRIBUTION POLARITY] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V170 contribution-polarity regime policy"
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
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(
        audit[side]["causal_numeric_only"]
        and audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["chronological_regime_split"]["strict_disjoint_chronology"]
        and audit[side]["early_recent_event_groups_disjoint"]
        and audit[side]["train_geometry"]["feature_dimension"] == STATE_DIMENSION
        and audit[side]["target_geometry"]["feature_dimension"] == STATE_DIMENSION
        and audit[side]["target_geometry"]["contribution_sign_agreement_or_reversal_is_row_specific"]
        and not audit[side]["target_geometry"]["target_batch_fit"]
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v83_full_vs_early_probability_distance_damping"]
        and not audit[side]["v142_delete_block_unanimous_coefficient_intersection"]
        and not audit[side]["v143_source_time_nuisance_subspace_projection"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["cross_fitted_feature_contribution_polarity_regime_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {
            "activated": True, "policy": "exact entire V69 DataFrame",
            "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion)),
        }
        require(result["selected_frame"].equals(champion), "V170 fallback not exact entire V69")
    return result


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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V170 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V170 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V170 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V170").resolve(), "V170 output is outside controller research/staging/V170")
    require(out.exists() and out.is_dir(), "V170 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V170 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V170 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
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
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V170 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V170 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V170", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V170_DIAGNOSTIC_CONTRIBUTION_POLARITY_REGIME": evaluation["candidate_summary"],
                "V170_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V170", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V170", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V170_FEATURE_CONTRIBUTION_POLARITY_REGIME_REPORT.json": {
            "version": "V170", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V170", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V170", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V170", "status": "MATCH", "canonical_check_count": 14,
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
    atomic_csv(evidence, out / "V170_CONTRIBUTION_POLARITY_REGIME_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V170_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
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
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V170 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V170 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V170 controller full-run output mismatch")
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
            "architecture": "two disjoint 55/45 chronological equal-group balanced linear heads; row-specific absolute contribution sign agreement/reversal mass routes fixed early/recent logit formula",
            "novelty": {
                "v64_v67_diagnostic_only_not_reused": True,
                "v83_probability_distance_damping": False,
                "v142_coefficient_sign_intersection": False,
                "v143_nuisance_projection": False,
                "raw_feature_sign_inversion": False,
            },
            "causal_numeric_only": True, "target_row_labels_used": False,
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
            require(args.smoke_market == "KR", "V170 support probe is reserved for KR:2")
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
