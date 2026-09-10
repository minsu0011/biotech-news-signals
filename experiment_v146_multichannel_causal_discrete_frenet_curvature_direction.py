"""V146 multichannel causal discrete-Frenet curvature direction challenger.

Every strict-past same-market cut robustly scales the immutable 37 causal
numeric fields and maps them to an observed eight-channel by eight-horizon
past-to-recent path.  A fixed local differential-geometric representation uses
increment speeds, unit tangents, turning cosines, speed-normalized curvature,
oriented tangent-plane bivectors, adjacent-plane rotation (a discrete torsion
proxy), endpoint displacement and elastic-energy summaries.  Train-only robust
feature scaling precedes one fixed duplicate/class-balanced ridge-logit head.

This replaces the proposed Haar scattering family because V95 already
implements it.  V146 uses no wavelet/scattering, global iterated integral,
Fourier spectrum, CUSUM, recurrence, visibility graph, ordinal motif, source
environment, text, return, V69 probability or confidence as a model input.
Inner-past labels choose exact V69 or fixed .25/.50 blends; outer labels are
evaluation-only under strict 35-minute embargo/event purge.  Confidence and
high_conf remain exact, and material failure restores the entire atomic V69
frame.  Audit/support/smoke are CPU30-31, two-thread, no-GPU and no-write.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v144_multichannel_weak_ordinal_motif_transition_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V146" / "LOCAL_FORBIDDEN"
VERSION = 146
HYPOTHESIS = "MULTICHANNEL_CAUSAL_DISCRETE_FRENET_CURVATURE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
SCAFFOLD_PATH = ROOT / "experiment_v144_multichannel_weak_ordinal_motif_transition_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "19817afa20b5243fea7831fd1d26a17cc2159085c5a774a2d9524a82d0f4064c"

CHANNEL_N = 8
HORIZON_N = 8
INCREMENT_N = HORIZON_N - 1
TURN_N = INCREMENT_N - 1
TORSION_N = TURN_N - 1
BIVECTOR_N = CHANNEL_N * (CHANNEL_N - 1) // 2
SUMMARY_N = 5
FRENET_FEATURE_DIMENSION = (
    INCREMENT_N * CHANNEL_N + INCREMENT_N + TURN_N + TURN_N
    + TURN_N * BIVECTOR_N + TORSION_N + TORSION_N
    + CHANNEL_N + SUMMARY_N
)
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
GEOMETRY_EPSILON = 1e-9
SEED = 14601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "DISCRETE_FRENET_CURVATURE_W0.25", "weight": 0.25},
    {"name": "DISCRETE_FRENET_CURVATURE_W0.50", "weight": 0.50},
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
v44 = scaffold.v44
numeric = scaffold.numeric
observed_past_to_recent_path = scaffold.scaffold.observed_past_to_recent_path


CHANNEL_PAIRS = tuple((left, right) for left in range(CHANNEL_N) for right in range(left + 1, CHANNEL_N))
STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "increment_n": INCREMENT_N,
    "turn_n": TURN_N,
    "tangent_plane_bivector_dimension": BIVECTOR_N,
    "torsion_transition_n": TORSION_N,
    "representation": [
        "unit_tangent_coordinates",
        "log1p_increment_speed",
        "adjacent_tangent_turning_cosine",
        "speed_normalized_discrete_curvature",
        "oriented_adjacent_tangent_plane_bivectors",
        "adjacent_unit_bivector_plane_cosine",
        "unit_bivector_difference_norm_discrete_torsion",
        "endpoint_displacement",
        "path_length_chord_efficiency_bending_and_torsion_energy",
    ],
    "feature_dimension": FRENET_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip",
    "head": "fixed duplicate/class-balanced ridge logistic",
    "learned_knot_threshold_wavelet_frequency_graph_motif_or_embedding": False,
}
require(
    INCREMENT_N == 7 and TURN_N == 6 and TORSION_N == 5
    and BIVECTOR_N == 28 and FRENET_FEATURE_DIMENSION == 266,
    "V146 static Frenet dimension changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V144 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending and atomic IO only; no V144 output is read",
    }
    audit["v146_access_contract"] = {
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


def discrete_frenet_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V146 path shape invalid")
    increment = np.diff(path, axis=1)
    speed = np.linalg.norm(increment, axis=2)
    tangent = increment / np.maximum(speed[:, :, None], GEOMETRY_EPSILON)
    next_tangent = tangent[:, 1:, :]
    current_tangent = tangent[:, :-1, :]
    tangent_change = next_tangent - current_tangent
    turning_cosine = np.sum(current_tangent * next_tangent, axis=2)
    local_scale = 0.5 * (speed[:, :-1] + speed[:, 1:])
    curvature = np.linalg.norm(tangent_change, axis=2) / np.maximum(local_scale, GEOMETRY_EPSILON)

    bivector = np.stack([
        current_tangent[:, :, left] * next_tangent[:, :, right]
        - current_tangent[:, :, right] * next_tangent[:, :, left]
        for left, right in CHANNEL_PAIRS
    ], axis=2)
    bivector_norm = np.linalg.norm(bivector, axis=2)
    unit_bivector = bivector / np.maximum(bivector_norm[:, :, None], GEOMETRY_EPSILON)
    plane_cosine = np.sum(unit_bivector[:, :-1, :] * unit_bivector[:, 1:, :], axis=2)
    torsion = np.linalg.norm(np.diff(unit_bivector, axis=1), axis=2)

    endpoint = path[:, -1, :] - path[:, 0, :]
    path_length = speed.sum(axis=1)
    chord = np.linalg.norm(endpoint, axis=1)
    efficiency = chord / np.maximum(path_length, GEOMETRY_EPSILON)
    bending_energy = np.mean(np.square(curvature), axis=1)
    torsion_energy = np.mean(np.square(torsion), axis=1)
    summary = np.stack([path_length, chord, efficiency, bending_energy, torsion_energy], axis=1)
    feature = np.concatenate([
        tangent.reshape(len(path), -1),
        np.log1p(speed),
        turning_cosine,
        curvature,
        bivector.reshape(len(path), -1),
        plane_cosine,
        torsion,
        endpoint,
        summary,
    ], axis=1)
    require(feature.shape == (len(path), FRENET_FEATURE_DIMENSION), "V146 Frenet feature dimension changed")
    require(np.isfinite(feature).all(), "V146 Frenet feature invalid")
    return feature, {
        "rows": len(feature),
        "feature_dimension": feature.shape[1],
        "increment_speed_q10_q50_q90": [float(np.quantile(speed, q)) for q in (0.1, 0.5, 0.9)],
        "turning_cosine_q10_q50_q90": [float(np.quantile(turning_cosine, q)) for q in (0.1, 0.5, 0.9)],
        "curvature_q10_q50_q90": [float(np.quantile(curvature, q)) for q in (0.1, 0.5, 0.9)],
        "bivector_norm_q10_q50_q90": [float(np.quantile(bivector_norm, q)) for q in (0.1, 0.5, 0.9)],
        "torsion_q10_q50_q90": [float(np.quantile(torsion, q)) for q in (0.1, 0.5, 0.9)],
        "path_efficiency_q10_q50_q90": [float(np.quantile(efficiency, q)) for q in (0.1, 0.5, 0.9)],
        "feature_sha256": array_sha256(feature),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "learned_knot_threshold_wavelet_frequency_graph_motif_or_embedding": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == FRENET_FEATURE_DIMENSION, "V146 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        self.scale = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V146 scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V146 scaled Frenet feature invalid")
        return result


def frenet_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V146 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_geometry_audit = discrete_frenet_features(train_path)
    target_feature, target_geometry_audit = discrete_frenet_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V146 probability invalid")
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit,
        "target_path": target_path_audit,
        "train_geometry": train_geometry_audit,
        "target_geometry": target_geometry_audit,
        "feature_scaling": {
            "train_only": True,
            "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C,
            "solver": "liblinear",
            "iterations": int(head.n_iter_[0]),
            "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False,
        "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
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
    challenger, model_audit = frenet_direction(inner_train, inner_valid)
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
        and model_audit["static_spec"]["feature_dimension"] == FRENET_FEATURE_DIMENSION
        and model_audit["train_geometry"]["feature_dimension"] == FRENET_FEATURE_DIMENSION
        and model_audit["train_geometry"]["label_independent_exact_transform"]
        and model_audit["train_geometry"]["local_differential_geometry_not_global_path_integral"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["haar_wavelet_or_scattering"]
        and not model_audit["rough_path_signature_or_levy_area"]
        and not model_audit["fourier_cross_spectrum_or_dmd"]
        and not model_audit["natural_visibility_graph_or_ordinal_motif"]
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 discrete-Frenet curvature blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "geometry_definition_head_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.SEED = SEED
    scaffold.choose_inner = choose_inner
    scaffold.ordinal_direction = frenet_direction
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v144_model": "v146_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "frenet_prob", "v144_model": "v146_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V146 DISCRETE FRENET] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    scaffold.SEED = SEED
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V146 discrete-Frenet curvature policy"
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V146 canonical mismatch")
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
    frenet_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["feature_dimension"] == FRENET_FEATURE_DIMENSION
        and audit[side]["train_geometry"]["feature_dimension"] == FRENET_FEATURE_DIMENSION
        and audit[side]["train_geometry"]["label_independent_exact_transform"]
        and audit[side]["train_geometry"]["local_differential_geometry_not_global_path_integral"]
        and audit[side]["target_geometry"]["feature_dimension"] == FRENET_FEATURE_DIMENSION
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["haar_wavelet_or_scattering"]
        and not audit[side]["rough_path_signature_or_levy_area"]
        and not audit[side]["increment_spd_or_covariance"]
        and not audit[side]["fourier_cross_spectrum_or_dmd"]
        and not audit[side]["cusum_changepoint_or_recurrence"]
        and not audit[side]["natural_visibility_graph_or_ordinal_motif"]
        and not audit[side]["source_time_environment_transport_or_projection"]
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
        "multichannel_discrete_frenet_curvature_contract_verified": frenet_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V146 fallback not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V146 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V146 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V146 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V146").resolve(), "V146 output is outside controller research/staging/V146")
    require(out.exists() and out.is_dir(), "V146 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V146 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V146 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
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
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V146 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V146 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V146", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V146_DIAGNOSTIC_DISCRETE_FRENET_CURVATURE": evaluation["candidate_summary"],
                "V146_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V146", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V146", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V146_MULTICHANNEL_DISCRETE_FRENET_CURVATURE_REPORT.json": {
            "version": "V146", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V146", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V146", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V146", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate,
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
    atomic_csv(evidence, out / "V146_DISCRETE_FRENET_CURVATURE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V146_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V146 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V146 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V146 controller full-run output mismatch")
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
            "architecture": "fixed 266D local multichannel path tangent-plane curvature and discrete-torsion geometry plus balanced ridge logistic",
            "proposed_haar_rejected_due_v95_direct_collision": True,
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
            require(args.smoke_market == "KR", "V146 support probe is reserved for KR:2")
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
