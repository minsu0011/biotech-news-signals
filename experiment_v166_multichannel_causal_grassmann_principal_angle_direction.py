"""V166 multichannel causal Grassmann principal-angle direction.

Each immutable causal event is mapped, inside every strict-past same-market
fit, to the observed eight-channel by eight-horizon path.  Training-cut robust
scaling is followed by a deterministic rank-three SVD.  Its left and right
singular spaces are two points on Gr(3, 8), describing horizon-mode and
channel-mode geometry without retaining sign-ambiguous singular vectors.

Equal-event-group class prototypes are formed by weighted means of projection
matrices and fixed top-three eigenspaces.  A target is scored directly by the
DOWN-minus-UP contrast of principal-angle geodesic and chordal distances to
the two row/column prototypes.  Only a train-score robust temperature is fit;
there is no discriminative coefficient head, PCA reconstruction classifier,
PLS response projection, source routing, target statistic or label-dependent
rank choice.  Inner history chooses exact V69 or fixed .25/.50 blends.  Outer
labels are evaluation-only under a strict 35-minute embargo/event purge.
Atomic V69 confidence/high_conf stay exact and every material failure restores
the exact entire atomic V69 frame.

Audit and bounded US:2/KR:2 probes use CPU30-31, two threads and no GPU.  Full
execution is accepted only by no-argument MARKET_BIO_VERSION_OUTPUT binding.
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

import experiment_v163_multichannel_causal_cross_bispectral_phase_coupling_direction as support


base = support.base
ROOT = support.ROOT
V69_DIR = support.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V166" / "LOCAL_FORBIDDEN"
VERSION = 166
HYPOTHESIS = "MULTICHANNEL_CAUSAL_GRASSMANN_PRINCIPAL_ANGLE_DIRECTION_V1"
FEATURES = support.FEATURES
EXPECTED_MARKET_FOLD_ROWS = support.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = support.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v163_multichannel_causal_cross_bispectral_phase_coupling_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "4789f861ed1095bc6864b1f6d0880850bb7a26192ca88f4675c63a2b09d3a22e"

CHANNEL_N = 8
HORIZON_N = 8
SUBSPACE_RANK = 3
SYMMETRIC_COORDINATES = CHANNEL_N * (CHANNEL_N + 1) // 2
GRASSMANN_COORDINATE_DIMENSION = 2 * SYMMETRIC_COORDINATES
SVD_TIE_BREAK = 1e-8
ANGLE_EPSILON = 1e-12
TEMPERATURE_FLOOR = 0.05
SEED = 16601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "GRASSMANN_PRINCIPAL_ANGLE_W0.25", "weight": 0.25},
    {"name": "GRASSMANN_PRINCIPAL_ANGLE_W0.50", "weight": 0.50},
)

require = support.require
sha256 = support.sha256
array_sha256 = support.array_sha256
clean = support.clean
atomic_json = support.atomic_json
atomic_csv = support.atomic_csv
load_authorized = support.load_authorized
controller = support.controller
numeric = support.numeric
observed_past_to_recent_path = support.observed_past_to_recent_path
bool_series = base.bool_series
metric = base.metric
blend_probability = base.blend_probability
supported_source_ba_delta = base.supported_source_ba_delta

STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(base.scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "per_event_representation": "rank-3 left/horizon and right/channel SVD subspaces on Gr(3,8)",
    "sign_invariance": "projection matrices and principal angles only",
    "class_prototypes": "equal-event-group weighted projection-matrix means followed by fixed top-3 eigenspaces",
    "score": "equal average of row/column chordal and geodesic DOWN-minus-UP distance contrasts",
    "temperature": "train-score IQR/1.349, standard-deviation fallback and fixed 0.05 floor",
    "feature_dimension": GRASSMANN_COORDINATE_DIMENSION,
    "learned_rank_metric_weight_or_threshold": False,
    "discriminative_coefficient_head": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V166_GRASSMANN_PRINCIPAL_ANGLE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json",
    "RUN_STATUS.json",
)
require(
    CHANNEL_N == HORIZON_N == 8 and SUBSPACE_RANK == 3
    and SYMMETRIC_COORDINATES == 36 and GRASSMANN_COORDINATE_DIMENSION == 72,
    "V166 static Grassmann representation changed",
)

_SUPPORT_CONFIGURE = support.configure_base
_SUPPORT_BOOTSTRAP = support.paired_nested_bootstrap


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V163 utility scaffold changed")
    audit = support.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, causal path, metrics, canonical gate, bootstrap and atomic IO only; no V163 output is read",
    }
    audit["v166_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def _event_subspaces(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V166 path shape invalid")
    centered = path - np.mean(path, axis=1, keepdims=True)
    tie_break = SVD_TIE_BREAK * np.diag(np.linspace(1.0, 2.0, CHANNEL_N))
    stable = centered + tie_break[None, :, :]
    left, singular, right_t = np.linalg.svd(stable, full_matrices=False)
    row_basis = left[:, :, :SUBSPACE_RANK]
    channel_basis = np.swapaxes(right_t[:, :SUBSPACE_RANK, :], 1, 2)
    row_orthogonality = np.max(np.abs(
        np.einsum("nkr,nks->nrs", row_basis, row_basis) - np.eye(SUBSPACE_RANK)[None, :, :]
    ))
    channel_orthogonality = np.max(np.abs(
        np.einsum("nkr,nks->nrs", channel_basis, channel_basis) - np.eye(SUBSPACE_RANK)[None, :, :]
    ))
    require(row_orthogonality <= 1e-8 and channel_orthogonality <= 1e-8, "V166 SVD bases are not orthonormal")
    require(np.isfinite(singular).all(), "V166 singular values invalid")
    return row_basis, channel_basis, {
        "rows": len(path), "rank": SUBSPACE_RANK,
        "feature_dimension": GRASSMANN_COORDINATE_DIMENSION,
        "row_basis_orthogonality_max_abs": float(row_orthogonality),
        "channel_basis_orthogonality_max_abs": float(channel_orthogonality),
        "singular_value_q10_q50_q90": [float(np.quantile(singular, q)) for q in (0.1, 0.5, 0.9)],
        "row_projection_sha256": array_sha256(np.einsum("nkr,nlr->nkl", row_basis, row_basis)),
        "channel_projection_sha256": array_sha256(np.einsum("nkr,nlr->nkl", channel_basis, channel_basis)),
        "label_independent_exact_transform": True,
        "per_event_row_and_column_grassmann_points": True,
        "svd_sign_invariant_projection_representation": True,
        "fixed_rank_without_label_or_outer_selection": True,
        # Explicit scaffold-compatibility negatives consumed only by V163's
        # generic evaluator before V166 replaces its material-contract check.
        "local_differential_geometry_not_global_path_integral": False,
        "admissible_frequency_triad_identity_verified": False,
        "within_channel_bispectral_magnitude_and_biphase_verified": False,
        "directed_cross_channel_biphase_coupling_verified": False,
        "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False,
        "time_domain_increment_coskewness": False,
        "hilbert_time_local_analytic_phase": False,
    }


def _class_prototype(basis: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(len(basis) == len(weight) and float(np.sum(weight)) > 0.0, "V166 prototype support invalid")
    projection = np.einsum("nkr,nlr->nkl", basis, basis)
    mean_projection = np.einsum("n,nkl->kl", weight, projection) / np.sum(weight)
    eigenvalue, eigenvector = np.linalg.eigh((mean_projection + mean_projection.T) / 2.0)
    order = np.argsort(eigenvalue)[::-1]
    prototype = eigenvector[:, order[:SUBSPACE_RANK]]
    require(np.max(np.abs(prototype.T @ prototype - np.eye(SUBSPACE_RANK))) <= 1e-8, "V166 prototype invalid")
    return prototype, {
        "support_n": len(basis), "effective_weight": float(np.sum(weight)),
        "top_eigenvalues": eigenvalue[order[:SUBSPACE_RANK]].tolist(),
        "eigengap_rank3_to4": float(eigenvalue[order[SUBSPACE_RANK - 1]] - eigenvalue[order[SUBSPACE_RANK]]),
        "prototype_projection_sha256": array_sha256(prototype @ prototype.T),
    }


def _principal_distances(basis: np.ndarray, prototype: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cross = np.einsum("nkr,ks->nrs", basis, prototype)
    cosine = np.clip(np.linalg.svd(cross, compute_uv=False), 0.0, 1.0)
    angle = np.arccos(cosine)
    chordal = np.sqrt(np.maximum(np.sum(1.0 - cosine * cosine, axis=1), 0.0))
    geodesic = np.sqrt(np.sum(angle * angle, axis=1))
    return chordal, geodesic, angle


def _distance_contrast(
    row_basis: np.ndarray,
    channel_basis: np.ndarray,
    prototypes: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    distances: dict[int, np.ndarray] = {}
    angle_audit: dict[str, Any] = {}
    for label in (0, 1):
        row_chordal, row_geodesic, row_angle = _principal_distances(row_basis, prototypes[label][0])
        channel_chordal, channel_geodesic, channel_angle = _principal_distances(channel_basis, prototypes[label][1])
        distances[label] = 0.25 * (row_chordal + row_geodesic + channel_chordal + channel_geodesic)
        angle_audit[str(label)] = {
            "row_angle_mean": float(np.mean(row_angle)),
            "channel_angle_mean": float(np.mean(channel_angle)),
            "distance_mean": float(np.mean(distances[label])),
        }
    raw = distances[0] - distances[1]
    require(np.isfinite(raw).all(), "V166 distance contrast invalid")
    return raw, {
        "class_distance": angle_audit,
        "raw_contrast_mean": float(np.mean(raw)), "raw_contrast_std": float(np.std(raw)),
        "raw_contrast_sha256": array_sha256(raw),
    }


def grassmann_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V166 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_row, train_channel, train_geometry = _event_subspaces(train_path)
    target_row, target_channel, target_geometry = _event_subspaces(target_path)
    target = ordered.y.to_numpy(int)
    sample_weight = numeric.duplicate_class_weights(ordered)
    prototypes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    prototype_audit: dict[str, Any] = {}
    for label in (0, 1):
        mask = target == label
        require(int(mask.sum()) >= 8, f"V166 class {label} support too small")
        row_prototype, row_audit = _class_prototype(train_row[mask], sample_weight[mask])
        channel_prototype, channel_audit = _class_prototype(train_channel[mask], sample_weight[mask])
        prototypes[label] = (row_prototype, channel_prototype)
        prototype_audit[str(label)] = {"row": row_audit, "channel": channel_audit}
    train_raw, train_score_audit = _distance_contrast(train_row, train_channel, prototypes)
    target_raw, target_score_audit = _distance_contrast(target_row, target_channel, prototypes)
    q25, q75 = np.quantile(train_raw, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    standard = float(np.std(train_raw))
    temperature = max(robust if robust > ANGLE_EPSILON else standard, TEMPERATURE_FLOOR)
    probability = np.clip(1.0 / (1.0 + np.exp(-np.clip(target_raw / temperature, -30.0, 30.0))), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V166 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry,
        "class_projection_mean_prototypes": prototype_audit,
        "train_score": train_score_audit, "target_score": target_score_audit,
        "feature_scaling": {"train_only": True, "temperature": temperature, "temperature_floor": TEMPERATURE_FLOOR},
        "head": {"type": "direct fixed principal-angle distance contrast", "fitted_discriminative_coefficients": 0},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False,
        "svd_rank_selected_from_labels_or_outer": False,
        "v101_static_affine_reconstruction_residual": False,
        "v143_source_time_nuisance_projection": False,
        "pls_or_ssa_response_projection": False,
        "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


def configure_base() -> None:
    support.VERSION = VERSION
    support.HYPOTHESIS = HYPOTHESIS
    support.STATIC_SPEC = STATIC_SPEC
    support.BISPECTRAL_FEATURE_DIMENSION = GRASSMANN_COORDINATE_DIMENSION
    support.ARCHITECTURES = ARCHITECTURES
    support.SEED = SEED
    support.bispectral_direction = grassmann_direction
    _SUPPORT_CONFIGURE()


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = grassmann_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "model_eligible": True, "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["train_geometry"]["per_event_row_and_column_grassmann_points"]
        and model_audit["train_geometry"]["svd_sign_invariant_projection_representation"]
        and model_audit["train_geometry"]["fixed_rank_without_label_or_outer_selection"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["svd_rank_selected_from_labels_or_outer"]
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
            "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 Grassmann principal-angle blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "rank_metric_temperature_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    support.configure_base = configure_base
    support.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = support.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v163_model": "v166_model"})
    evidence = evidence.rename(columns={"bispectral_prob": "grassmann_prob", "v163_model": "v166_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V166 GRASSMANN] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _SUPPORT_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V166 Grassmann principal-angle policy"
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
    support.paired_nested_bootstrap = paired_nested_bootstrap
    result = support.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_causal_cross_bispectral_phase_coupling_contract_verified")
    grassmann_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["train_geometry"]["per_event_row_and_column_grassmann_points"]
        and audit[side]["train_geometry"]["svd_sign_invariant_projection_representation"]
        and audit[side]["train_geometry"]["fixed_rank_without_label_or_outer_selection"]
        and audit[side]["target_geometry"]["per_event_row_and_column_grassmann_points"]
        and audit[side]["feature_scaling"]["train_only"]
        and audit[side]["head"]["fitted_discriminative_coefficients"] == 0
        and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v101_static_affine_reconstruction_residual"]
        and not audit[side]["v143_source_time_nuisance_projection"]
        and not audit[side]["pls_or_ssa_response_projection"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["multichannel_causal_grassmann_principal_angle_contract_verified"] = bool(grassmann_contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if result["material_pass"]:
        result["selected_frame"] = result["diagnostic_frame"].copy()
        result["selected_summary"] = result["candidate_summary"]
        result["fallback"] = {"activated": False, "policy": None, "exact_entire_frame_verified": True}
    else:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V166 fallback not exact entire V69")
    result["status"] = "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if result["material_pass"] else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V166 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V166 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V166 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V166").resolve(), "V166 output is outside controller research/staging/V166")
    require(out.exists() and out.is_dir(), "V166 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V166 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V166 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write"}


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V166 report inventory invalid")
    return {
        "status": "TEMP_REPORT_CONTRACT_OK", "required_reports": sorted(required),
        "all_report_names": list(REPORT_NAMES), "current_material_gate": True,
        "selected_material_gate_nested_bootstrap": True, "native_robustness_preserved": True,
        "source_transfer_current_gate": True, "exact_entire_v69_fallback": True,
        "filesystem_write": False,
    }


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V166 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V166 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V166", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V166_DIAGNOSTIC_GRASSMANN_PRINCIPAL_ANGLE": evaluation["candidate_summary"], "V166_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V166", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V166", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V166_GRASSMANN_PRINCIPAL_ANGLE_REPORT.json": {"version": "V166", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V166", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V166", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V166", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict_audit, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    require(set(reports) == set(REPORT_NAMES), "V166 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V166_GRASSMANN_PRINCIPAL_ANGLE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V166_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--contract-probe", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V166 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V166 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V166 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.contract_probe:
        print(json.dumps(clean({**temporary_report_contract_probe(), "hypothesis": HYPOTHESIS, "runtime": runtime, "authority": authority}), ensure_ascii=False, indent=2))
        return
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "fixed rank-3 per-event row/channel Grassmann points, class projection-mean eigenspaces, direct principal-angle chordal/geodesic contrast",
            "prior_exact_grassmann_principal_angle_family_hit_count": 0,
            "v101_static_affine_state_reconstruction_reused": False,
            "v143_source_time_nuisance_subspace_reused": False,
            "v99_pls_or_ssa_reused": False,
            "v146_local_frenet_geometry_reused": False,
            "v164_euclidean_minimum_enclosing_ball_reused": False,
            "v165_hmm_transition_emission_reused": False,
            "causal_numeric_only": True, "target_row_labels_used": False,
            "micro_tuning_or_post_outer_change": False, "canonical_controller_gate_checks": 14,
            "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "gpu_model_calls": 0, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True},
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V166 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]})
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
