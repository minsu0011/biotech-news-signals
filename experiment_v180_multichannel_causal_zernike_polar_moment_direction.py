"""V180 multichannel causal fixed discrete-Zernike polar-moment direction.

The immutable V36 eight-horizon by eight-semantic-channel causal surface is
scaled only on each strict-past training cut.  On a fixed unit-disk sampling
mask, closed-form Zernike radial polynomials through order six produce complex
polar moments of the signed, positive and negative surface parts.  Oriented
real/imaginary coordinates, rotation-invariant magnitudes and radial-order
energies form a fixed 153-dimensional representation for one balanced ridge
direction head.  There is no learned basis, order, mask or target-window fit.

This is not V178 cubic B-spline functional jets or V179 gradient structure
tensors.  Inner OOF selects exact V69 or fixed .25/.50 blends; outer labels are
evaluation-only after the 35-minute embargo/event purge.  V69 confidence and
high_conf remain exact, and any material failure restores the entire V69 frame.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v177_multichannel_causal_discrete_radon_tomography_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V180" / "LOCAL_FORBIDDEN"
VERSION = 180
HYPOTHESIS = "MULTICHANNEL_CAUSAL_ZERNIKE_POLAR_MOMENT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v177_multichannel_causal_discrete_radon_tomography_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "7cd5e27609b571db176f2c2876a55199b5fc71427a98cc465082c653304f7328"

CHANNEL_N = 8
HORIZON_N = 8
MAX_ORDER = 6
MODES = tuple((n, m) for n in range(MAX_ORDER + 1) for m in range(n + 1) if (n - m) % 2 == 0)
ORIENTED_COORDINATE_N = sum(1 if m == 0 else 2 for _, m in MODES)
MAGNITUDE_N = len(MODES)
RADIAL_ENERGY_N = MAX_ORDER + 1
COMPONENTS = ("SIGNED", "POSITIVE", "NEGATIVE")
FEATURES_PER_COMPONENT = ORIENTED_COORDINATE_N + MAGNITUDE_N + RADIAL_ENERGY_N
ZERNIKE_FEATURE_DIMENSION = len(COMPONENTS) * FEATURES_PER_COMPONENT
CHANNEL_CLIP = 7.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 18001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "ZERNIKE_POLAR_MOMENT_W0.25", "weight": 0.25},
    {"name": "ZERNIKE_POLAR_MOMENT_W0.50", "weight": 0.50},
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
observed_past_to_recent_path = scaffold.observed_past_to_recent_path
equal_event_group_path = scaffold.equal_event_group_path
TrainChannelScale = scaffold.TrainChannelScale
blend_probability = base.blend_probability

STATIC_SPEC = {
    "surface_shape": [HORIZON_N, CHANNEL_N],
    "horizon_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "channel_order": "immutable V36 semantic channel order",
    "unit_disk_grid": "fixed 8x8 cell-center coordinates, corners outside radius one omitted",
    "maximum_radial_order": MAX_ORDER,
    "nonnegative_azimuthal_modes": [list(mode) for mode in MODES],
    "surface_components": list(COMPONENTS),
    "coordinates": "oriented real/imaginary moments, invariant magnitudes, per-order radial energies",
    "feature_dimension": ZERNIKE_FEATURE_DIMENSION,
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_basis_order_mask_or_rotation": False,
    "target_batch_fit": False,
}
require(len(MODES) == 16 and ORIENTED_COORDINATE_N == 28, "V180 Zernike mode dictionary changed")
require(FEATURES_PER_COMPONENT == 51 and ZERNIKE_FEATURE_DIMENSION == 153, "V180 feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V177 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V177 output is read"}
    audit["v180_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.zeros(ZERNIKE_FEATURE_DIMENSION)
        self.scale = np.ones(ZERNIKE_FEATURE_DIMENSION)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.shape[1] == ZERNIKE_FEATURE_DIMENSION, "V180 feature shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V180 scaled feature invalid")
        return result


def radial_polynomial(n: int, m: int, radius: np.ndarray) -> np.ndarray:
    require(0 <= m <= n <= MAX_ORDER and (n - m) % 2 == 0, "V180 invalid Zernike mode")
    result = np.zeros_like(radius, dtype=float)
    for s in range((n - m) // 2 + 1):
        coefficient = ((-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)))
        result += coefficient * np.power(radius, n - 2 * s)
    return result


def fixed_zernike_basis() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coordinate = (np.arange(HORIZON_N, dtype=float) + 0.5) / (HORIZON_N / 2.0) - 1.0
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    radius = np.sqrt(np.square(xx) + np.square(yy))
    angle = np.arctan2(yy, xx)
    mask = radius <= 1.0 + 1e-12
    basis = []
    for n, m in MODES:
        radial = radial_polynomial(n, m, radius[mask])
        basis.append((n + 1.0) * radial * np.exp(-1j * m * angle[mask]) / float(mask.sum()))
    result = np.asarray(basis, dtype=complex)
    require(result.shape == (len(MODES), int(mask.sum())) and np.isfinite(result.real).all() and np.isfinite(result.imag).all(), "V180 basis invalid")
    return mask, result, {"disk_pixel_n": int(mask.sum()), "outside_corner_pixel_n": int(mask.size - mask.sum()), "basis_sha256": array_sha256(np.column_stack((result.real, result.imag))), "fixed_closed_form_radial_polynomial": True}


DISK_MASK, ZERNIKE_BASIS, BASIS_AUDIT = fixed_zernike_basis()


def component_moments(surface: np.ndarray) -> np.ndarray:
    values = surface[DISK_MASK]
    moments = ZERNIKE_BASIS @ values
    oriented: list[float] = []
    for (_, m), value in zip(MODES, moments):
        oriented.append(float(value.real))
        if m > 0:
            oriented.append(float(value.imag))
    magnitudes = np.abs(moments)
    radial = np.asarray([np.sqrt(np.sum(np.square(magnitudes[[index for index, (order, _) in enumerate(MODES) if order == n]]))) for n in range(MAX_ORDER + 1)], dtype=float)
    result = np.concatenate((np.asarray(oriented), magnitudes, radial))
    require(result.shape == (FEATURES_PER_COMPONENT,) and np.isfinite(result).all(), "V180 component moment invalid")
    return result


def zernike_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V180 path shape invalid")
    result = np.empty((len(path), ZERNIKE_FEATURE_DIMENSION), dtype=float)
    for row, surface in enumerate(path):
        components = (surface, np.maximum(surface, 0.0), np.maximum(-surface, 0.0))
        result[row] = np.concatenate([component_moments(component) for component in components])
    require(np.isfinite(result).all() and float(np.std(result)) > 1e-10, "V180 Zernike feature degenerate")
    return result, {"rows": len(result), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result), "basis": BASIS_AUDIT, "maximum_radial_order": MAX_ORDER, "mode_n": len(MODES), "surface_components": list(COMPONENTS), "oriented_real_imag_coordinates": True, "rotation_invariant_magnitudes": True, "radial_order_energies": True, "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": False, "learned_basis_order_mask_or_rotation": False, "target_batch_fit": False, "radon_or_tomography": False, "cubic_bspline_or_functional_jet": False, "finite_difference_gradient_or_structure_tensor": False, "fourier_wavelet_hankel_or_burg": False, "morphological_or_topological_filtration": False}


def zernike_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V180 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_feature, train_zernike = zernike_features(group_path)
    target_feature, target_zernike = zernike_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all() and int(head.n_iter_[0]) < 1000, "V180 head invalid")
    return probability, {"train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_geometry": train_zernike, "target_geometry": target_zernike, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": CHANNEL_CLIP}, "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": FEATURE_CLIP}, "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "class_event_group_n": class_n.astype(int).tolist(), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "intercept": head.intercept_.tolist()}, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "v177_radon_tomography": False, "v178_bspline_functional_jet": False, "v179_structure_tensor": False, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = ZERNIKE_FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = zernike_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = zernike_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and model_audit["train_geometry"]["basis"]["fixed_closed_form_radial_polynomial"] and model_audit["train_geometry"]["oriented_real_imag_coordinates"] and model_audit["train_geometry"]["rotation_invariant_magnitudes"] and model_audit["train_geometry"]["radial_order_energies"] and not model_audit["train_geometry"]["learned_basis_order_mask_or_rotation"] and not model_audit["train_geometry"]["target_batch_fit"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"]); current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]; ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]; net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 Zernike polar-moment blend with fixed safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "order_mode_mask_component_head_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base(); base.choose_inner = choose_inner
    with redirect_stdout(StringIO()): diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v180_model"}); evidence = evidence.rename(columns={"frenet_prob": "zernike_prob", "v146_model": "v180_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V180 ZERNIKE] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base(); result = _BASE_BOOTSTRAP(evidence, draws); result["method"] = "paired market-fold-time block bootstrap over inner-locked V180 Zernike policy"; result["seed"] = SEED; result["standardized"] = True; return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base(); base.paired_nested_bootstrap = paired_nested_bootstrap; result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]; checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"] == ZERNIKE_FEATURE_DIMENSION and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["basis"]["fixed_closed_form_radial_polynomial"] and audit[side]["train_geometry"]["oriented_real_imag_coordinates"] and audit[side]["train_geometry"]["rotation_invariant_magnitudes"] and audit[side]["train_geometry"]["radial_order_energies"] and not audit[side]["train_geometry"]["radon_or_tomography"] and not audit[side]["train_geometry"]["cubic_bspline_or_functional_jet"] and not audit[side]["train_geometry"]["finite_difference_gradient_or_structure_tensor"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit")))
    checks["multichannel_causal_zernike_polar_moment_contract_verified"] = contract; result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]; result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}; require(result["selected_frame"].equals(champion), "V180 fallback not exact V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]; gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}; require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V180 nested key mismatch"); return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V180 output requires controller authority"); require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V180 output target mismatch"); require(out.resolve().parent == (ROOT / "research" / "staging" / "V180").resolve(), "V180 output outside staging/V180"); require(out.exists() and out.is_dir(), "V180 staging child must preexist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}; require(required.issubset(existing), f"V180 prefiles missing: {sorted(required-existing)}"); require(not (existing - allowed), f"V180 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V180 output not controller-bound"); conflict = enforce_output_conflict_fail_closed(out); compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate; require("bootstrap" in selected["robustness"], "V180 native bootstrap missing")
    reports = {"MODEL_COMPARISON.json": {"version": "V180", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V180_DIAGNOSTIC_ZERNIKE": evaluation["candidate_summary"], "V180_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits}, "DEV_ROBUSTNESS_REPORT.json": {"version": "V180", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False}, "SOURCE_TRANSFER_REPORT.json": {"version": "V180", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False}, "V180_ZERNIKE_POLAR_MOMENT_REPORT.json": {"version": "V180", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority}, "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V180", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]}, "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V180", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}, "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V180", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate}, "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()}}
    for name, payload in reports.items(): atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V180_ZERNIKE_OUTER_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V180_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json"); return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); modes.add_argument("--audit-only", action="store_true"); modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--support-probe", action="store_true"); modes.add_argument("--full-run", action="store_true"); parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args()
    if CONTROLLER_OUTPUT: require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V180 full run"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output = DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT), "V180 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT"); require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V180 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args(); bounded = bool(args.audit_only or args.smoke_test or args.support_probe); runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}; authority = verify_authority(); dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "fixed order0-6 signed/positive/negative discrete-Zernike oriented moments, invariant magnitudes and radial energies; balanced ridge", "causal_numeric_only": True, "target_labels_used": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2)); return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V180 support reserved for KR:2"); audit = audits[0]; non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]; best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"])); summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test: summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]}); print(json.dumps(clean(summary), ensure_ascii=False, indent=2)); return
    result = write_outputs(args.output, authority, evidence, audits, evaluation); print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
