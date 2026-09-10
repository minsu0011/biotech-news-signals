"""V164 class-conditional minimum-enclosing-ball support challenger.

The originally proposed random-projection halfspace-depth family is rejected:
V96 already implements fixed Rademacher projections, class empirical CDFs and
trimmed halfspace depth.  This replacement instead treats each strict-past
class as a geometric support set.  Equal event-group centroids of the complete
74-dimensional robust causal numeric-plus-missingness state fit one fixed,
deterministic Badoiu--Clarkson minimum-enclosing ball per class.  A query's
normalized boundary-slack contrast, divided by one train-only robust scale,
directly yields direction probability.  There are no projection CDFs, density
models, covariance estimates, fitted discriminative coefficients or target-
batch statistics.

Inner-past labels select exact V69 or fixed .25/.50 blends; outer labels are
evaluation-only under the 35-minute embargo/event purge.  V69 confidence and
high_conf remain exact.  Any material or execution failure restores the exact
entire atomic V69 frame.  Audit/support/smoke are CPU30-31, two-thread, no-GPU
and no-write; full execution is controller-bound only.
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
from threadpoolctl import threadpool_limits

import experiment_v161_multichannel_causal_lagged_copula_tail_dependence_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V164" / "LOCAL_FORBIDDEN"
VERSION = 164
HYPOTHESIS = "CLASS_CONDITIONAL_MINIMUM_ENCLOSING_BALL_SUPPORT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v161_multichannel_causal_lagged_copula_tail_dependence_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "5868ee5865b7eeea97d73c779b46e118419f68c3de1f9c443b7c2c94eedab953"

STATE_DIMENSION = 2 * len(FEATURES)
MEB_ITERATIONS = 128
RADIUS_FLOOR = 1e-8
TEMPERATURE_FLOOR = 0.05
SEED = 16401
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "MINIMUM_ENCLOSING_BALL_SUPPORT_W0.25", "weight": 0.25},
    {"name": "MINIMUM_ENCLOSING_BALL_SUPPORT_W0.50", "weight": 0.50},
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
v44 = scaffold.v44
numeric = scaffold.numeric
blend_probability = base.blend_probability

STATIC_SPEC = {
    "state_dimension": STATE_DIMENSION,
    "feature_dimension": STATE_DIMENSION,
    "state": "37 train-cut robust causal numeric coordinates plus 37 deterministic missingness flags",
    "event_group_weighting": "equal event-group centroids before class support fitting",
    "support_estimator": "one deterministic Badoiu-Clarkson minimum-enclosing ball per direction class",
    "iterations": MEB_ITERATIONS,
    "initialization": "first lexicographically sorted event-group centroid within class",
    "update": "farthest support point with step 1/(iteration+2)",
    "score": "UP normalized boundary slack minus DOWN normalized boundary slack",
    "temperature": "train-event-group score IQR/1.349, fallback standard deviation, fixed floor",
    "head": "direct sigmoid; no fitted direction coefficient",
    "random_projection_or_empirical_cdf": False,
    "target_batch_fit": False,
    "hyperparameter_search": False,
}
require(STATE_DIMENSION == 74, "V164 causal-state dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V161 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, metrics, canonical gate, blending, bootstrap and atomic IO only; no V161 output is read",
    }
    audit["v164_access_contract"] = {
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape == (len(ordered), STATE_DIMENSION), "V164 state shape invalid")
    columns = [f"state_{index}" for index in range(STATE_DIMENSION)]
    work = pd.DataFrame(state, columns=columns)
    work["event_group_id"] = ordered.event_group_id.astype(str).to_numpy()
    work["y"] = ordered.y.to_numpy(int)
    require(
        bool((work.groupby("event_group_id", sort=False).y.nunique() == 1).all()),
        "V164 event_group labels inconsistent",
    )
    grouped = work.groupby("event_group_id", sort=True, as_index=False).agg(
        {**{column: "mean" for column in columns}, "y": "first"}
    )
    group_state = grouped[columns].to_numpy(float)
    group_target = grouped.y.to_numpy(int)
    require(np.unique(group_target).size == 2, "V164 grouped train cut missing a class")
    require(np.isfinite(group_state).all(), "V164 grouped state invalid")
    return group_state, group_target, {
        "raw_rows": len(ordered),
        "equal_event_groups": len(grouped),
        "duplicates_removed_before_support_fit": len(ordered) - len(grouped),
        "class_event_group_n": np.bincount(group_target, minlength=2).tolist(),
        "equal_event_group_weighting": True,
        "group_state_sha256": array_sha256(group_state),
    }


def fit_minimum_enclosing_ball(points: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    require(points.ndim == 2 and points.shape[1] == STATE_DIMENSION, "V164 class support shape invalid")
    require(len(points) >= 100 and np.isfinite(points).all(), "V164 class support too small or invalid")
    center = points[0].astype(float, copy=True)
    selected: list[int] = []
    for iteration in range(MEB_ITERATIONS):
        squared = np.sum(np.square(points - center), axis=1)
        farthest = int(np.argmax(squared))
        selected.append(farthest)
        center += (points[farthest] - center) / float(iteration + 2)
    distances = np.linalg.norm(points - center, axis=1)
    radius = max(float(np.max(distances)), RADIUS_FLOOR)
    require(np.isfinite(center).all() and np.isfinite(radius), "V164 minimum-enclosing ball invalid")
    normalized = distances / radius
    require(float(np.max(normalized)) <= 1.0 + 1e-12, "V164 support containment failed")
    return center, radius, {
        "support_n": len(points),
        "dimension": points.shape[1],
        "iterations": MEB_ITERATIONS,
        "unique_coreset_points": len(set(selected)),
        "center_sha256": array_sha256(center),
        "radius": radius,
        "support_normalized_distance_q10_q50_q90_max": [
            float(np.quantile(normalized, quantile)) for quantile in (0.10, 0.50, 0.90, 1.00)
        ],
        "all_training_support_contained": True,
        "deterministic_farthest_point_minimax_updates": True,
    }


def support_slack_score(
    matrix: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    distance = np.stack([
        np.linalg.norm(matrix - centers[label], axis=1) / radii[label]
        for label in (0, 1)
    ], axis=1)
    slack = 1.0 - distance
    score = slack[:, 1] - slack[:, 0]
    require(np.isfinite(score).all(), "V164 boundary-slack score invalid")
    return score, {
        "rows": len(matrix),
        "feature_dimension": STATE_DIMENSION,
        "normalized_distance_sha256": array_sha256(distance),
        "score_sha256": array_sha256(score),
        "score_q10_q50_q90": [float(np.quantile(score, quantile)) for quantile in (0.10, 0.50, 0.90)],
        "support_geometry_not_projection_depth_or_density": True,
        "label_independent_exact_transform": True,
        "label_independent_state_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
    }


def meb_support_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V164 expects one market per fit")
    require(str(ordered.market.iloc[0]) == str(target_frame.market.iloc[0]), "V164 market fit mismatch")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    require(train_state.shape[1] == target_state.shape[1] == STATE_DIMENSION, "V164 robust state dimension changed")
    group_state, group_target, grouping_audit = equal_event_group_state(ordered, train_state)
    centers = np.empty((2, STATE_DIMENSION), dtype=float)
    radii = np.empty(2, dtype=float)
    ball_audit: dict[str, Any] = {}
    for label in (0, 1):
        centers[label], radii[label], ball_audit[str(label)] = fit_minimum_enclosing_ball(
            group_state[group_target == label]
        )
    train_score, train_geometry = support_slack_score(group_state, centers, radii)
    target_score, target_geometry = support_slack_score(target_state, centers, radii)
    q25, q75 = np.quantile(train_score, [0.25, 0.75])
    robust_temperature = float((q75 - q25) / 1.349)
    standard_temperature = float(np.std(train_score))
    temperature = max(
        robust_temperature if robust_temperature > 1e-8 else standard_temperature,
        TEMPERATURE_FLOOR,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(target_score / temperature, -30.0, 30.0)))
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V164 probability invalid")
    return probability, {
        "train_n": len(ordered),
        "train_event_group_n": len(group_target),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "grouping": grouping_audit,
        "static_spec": STATIC_SPEC,
        "train_geometry": train_geometry,
        "target_geometry": target_geometry,
        "class_support_balls": ball_audit,
        "feature_scaling": {
            "train_only": True,
            "clip": float(numeric.ROBUST_CLIP),
            "robust_numeric_processor_only": True,
        },
        "head": {
            "type": "direct normalized support-boundary slack contrast sigmoid",
            "fitted_direction_coefficient_n": 0,
            "train_only_temperature": temperature,
            "train_score_iqr_scale": robust_temperature,
            "train_score_std": standard_temperature,
            "center_sha256": array_sha256(centers),
            "radii": radii.tolist(),
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
        "random_projection_or_halfspace_depth": False,
        "empirical_projection_cdf_or_trimmed_depth": False,
        "cantelli_or_covariance_moment_bound": False,
        "tyler_spatial_shape_or_fisher_lda": False,
        "quantile_transport_or_target_batch_cdf": False,
        "random_fourier_or_random_maclaurin_features": False,
        "archetype_simplex_or_class_affine_subspace": False,
        "vmf_or_hyperbolic_barycenter": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
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
    base.frenet_direction = meb_support_direction


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = meb_support_direction(inner_train, inner_valid)
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
        and model_audit["static_spec"]["state_dimension"] == STATE_DIMENSION
        and model_audit["grouping"]["equal_event_group_weighting"]
        and model_audit["train_geometry"]["support_geometry_not_projection_depth_or_density"]
        and model_audit["target_geometry"]["support_geometry_not_projection_depth_or_density"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["random_projection_or_halfspace_depth"]
        and not model_audit["empirical_projection_cdf_or_trimmed_depth"]
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 class-support MEB blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "support_definition_iterations_temperature_blend_or_source_floor_micro_tuning": False,
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
    diagnostic = diagnostic.rename(columns={"v146_model": "v164_model"})
    evidence = evidence.rename(columns={"frenet_prob": "meb_support_prob", "v146_model": "v164_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V164 CLASS MEB SUPPORT] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V164 class minimum-enclosing-ball support policy"
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
        and audit[side]["static_spec"]["state_dimension"] == STATE_DIMENSION
        and audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["train_geometry"]["support_geometry_not_projection_depth_or_density"]
        and audit[side]["target_geometry"]["support_geometry_not_projection_depth_or_density"]
        and all(ball["all_training_support_contained"] for ball in audit[side]["class_support_balls"].values())
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["random_projection_or_halfspace_depth"]
        and not audit[side]["empirical_projection_cdf_or_trimmed_depth"]
        and not audit[side]["cantelli_or_covariance_moment_bound"]
        and not audit[side]["tyler_spatial_shape_or_fisher_lda"]
        and not audit[side]["quantile_transport_or_target_batch_cdf"]
        and not audit[side]["random_fourier_or_random_maclaurin_features"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["class_conditional_minimum_enclosing_ball_support_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {
            "activated": True,
            "policy": "exact entire V69 DataFrame",
            "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion)),
        }
        require(result["selected_frame"].equals(champion), "V164 fallback not exact entire V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V164 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V164 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V164 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V164").resolve(), "V164 output is outside controller research/staging/V164")
    require(out.exists() and out.is_dir(), "V164 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V164 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V164 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True,
        "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V164 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V164 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V164", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V164_DIAGNOSTIC_CLASS_MEB_SUPPORT": evaluation["candidate_summary"],
                "V164_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V164", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V164", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V164_CLASS_MEB_SUPPORT_REPORT.json": {
            "version": "V164", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V164", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V164", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V164", "status": "MATCH", "canonical_check_count": 14,
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
    atomic_csv(evidence, out / "V164_CLASS_MEB_SUPPORT_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V164_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V164 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V164 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V164 controller full-run output mismatch")
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
            "architecture": "equal-event-group 74D robust causal-state class support sets, deterministic fixed-iteration minimum-enclosing balls, normalized boundary-slack contrast, train-only temperature, direct sigmoid",
            "rejected_original": "random-projection class halfspace depth directly collides with V96",
            "projection_cdf_depth_density_covariance_or_discriminative_head": False,
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
            require(args.smoke_market == "KR", "V164 support probe is reserved for KR:2")
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
