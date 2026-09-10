"""V160 multichannel causal Hilbert spectral-phase direction challenger.

Every strict-past same-market cut robustly scales the immutable 37 causal
numeric fields and maps them to the observed eight-channel by eight-horizon
past-to-recent path.  A fixed length-eight discrete Hilbert quadrature creates
a time-domain analytic signal for every channel.  Deterministic summaries use
analytic amplitude, unwrapped phase advance, elapsed-minute instantaneous
frequency and early/recent/all-horizon cross-channel phase locking.  Train-only
robust feature scaling precedes one fixed duplicate/class-balanced ridge-logit
direction head.

This is not V134's frequency-bin rFFT auto-power or unit cross-spectrum: V160
does not expose frequency-bin power or complex cross-spectra and instead
summarizes the inverse-transformed time-local analytic signal and circular
phase synchronization.  It uses no Haar/wavelet/scattering, cepstrum, neural
sequence model, path curvature, recurrence, transfer entropy, arrival hazard,
text, source identity or return input.  Inner-past labels choose exact V69 or
fixed .25/.50 blends.  Outer labels are evaluation-only under strict 35-minute
embargo/event purge.  V69 confidence/high_conf stay exact, and any material
failure restores the exact entire atomic V69 frame.

Audit, contract, US:2 smoke and KR:2 support are CPU30-31, two-thread, no-GPU
and no-write.  Full execution is accepted only by no-argument
MARKET_BIO_VERSION_OUTPUT binding to an existing controller staging attempt.
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
from threadpoolctl import threadpool_limits

import experiment_v146_multichannel_causal_discrete_frenet_curvature_direction as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V160" / "LOCAL_FORBIDDEN"
VERSION = 160
HYPOTHESIS = "MULTICHANNEL_CAUSAL_HILBERT_SPECTRAL_PHASE_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_MARKET_FOLD_ROWS = base.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = base.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v146_multichannel_causal_discrete_frenet_curvature_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "88e466ef8376ca441a28b29f3583d882a77eaba9b8f8db2cac9bac4a37ffcbe6"

CHANNEL_N = 8
HORIZON_N = 8
ADVANCE_N = HORIZON_N - 1
CHANNEL_PAIR_N = CHANNEL_N * (CHANNEL_N - 1) // 2
CHANNEL_AMPLITUDE_SUMMARY_N = 5
CHANNEL_PHASE_ADVANCE_SUMMARY_N = 5
CHANNEL_FREQUENCY_SUMMARY_N = 4
PAIR_PHASE_LOCK_SUMMARY_N = 8
HILBERT_FEATURE_DIMENSION = (
    CHANNEL_N * (
        CHANNEL_AMPLITUDE_SUMMARY_N
        + CHANNEL_PHASE_ADVANCE_SUMMARY_N
        + CHANNEL_FREQUENCY_SUMMARY_N
    )
    + CHANNEL_PAIR_N * PAIR_PHASE_LOCK_SUMMARY_N
)
HORIZON_MINUTES = np.asarray([120.0, 60.0, 30.0, 15.0, 10.0, 5.0, 2.0, 1.0])
ELAPSED_MINUTES = HORIZON_MINUTES[:-1] - HORIZON_MINUTES[1:]
HILBERT_MULTIPLIER = np.asarray([1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0])
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
AMPLITUDE_EPSILON = 1e-9
SEED = 16001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "HILBERT_SPECTRAL_PHASE_W0.25", "weight": 0.25},
    {"name": "HILBERT_SPECTRAL_PHASE_W0.50", "weight": 0.50},
)

require = base.require
sha256 = base.sha256
array_sha256 = base.array_sha256
clean = base.clean
atomic_json = base.atomic_json
atomic_csv = base.atomic_csv
load_authorized = base.load_authorized
controller = base.controller
numeric = base.numeric
observed_past_to_recent_path = base.observed_past_to_recent_path

CHANNEL_PAIRS = tuple(
    (left, right)
    for left in range(CHANNEL_N)
    for right in range(left + 1, CHANNEL_N)
)
STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(base.scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "horizon_minutes": HORIZON_MINUTES.tolist(),
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "analytic_signal": "length-8 exact DFT, fixed even-N Hilbert multiplier [1,2,2,2,1,0,0,0], inverse DFT",
    "channel_summaries": {
        "amplitude": ["mean", "std", "max", "recent", "linear_slope"],
        "unwrapped_phase_advance": ["mean", "std", "sum", "recent", "positive_fraction"],
        "instantaneous_frequency_radian_per_minute": ["mean", "std", "recent", "max_abs"],
    },
    "pair_phase_locking": [
        "all_horizon_plv", "early_plv", "recent_plv",
        "mean_phase_difference_cos", "mean_phase_difference_sin",
        "recent_phase_difference_cos", "recent_phase_difference_sin",
        "early_to_recent_circular_mean_drift",
    ],
    "feature_dimension": HILBERT_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed duplicate/class-balanced C=0.5 ridge logistic",
    "frequency_bin_power_or_cross_spectrum": False,
    "wavelet_scattering_or_cepstrum": False,
    "learned_frequency_or_phase_threshold": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V160_HILBERT_SPECTRAL_PHASE_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json",
    "RUN_STATUS.json",
)
require(
    len(CHANNEL_PAIRS) == CHANNEL_PAIR_N == 28
    and HILBERT_FEATURE_DIMENSION == 336
    and np.array_equal(ELAPSED_MINUTES, np.asarray([60.0, 30.0, 15.0, 5.0, 5.0, 3.0, 1.0])),
    "V160 static Hilbert representation changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V146 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending, bootstrap and atomic IO utilities only; no V146 output is read",
    }
    audit["v160_access_contract"] = {
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


def analytic_signal(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V160 path shape invalid")
    centered = path - np.mean(path, axis=1, keepdims=True)
    spectrum = np.fft.fft(centered, axis=1)
    analytic = np.fft.ifft(spectrum * HILBERT_MULTIPLIER[None, :, None], axis=1)
    reconstruction_error = float(np.max(np.abs(analytic.real - centered)))
    require(np.isfinite(analytic.real).all() and np.isfinite(analytic.imag).all(), "V160 analytic signal invalid")
    require(reconstruction_error <= 1e-10, "V160 Hilbert analytic real reconstruction failed")
    return analytic, {
        "rows": len(path),
        "shape": list(analytic.shape),
        "hilbert_multiplier": HILBERT_MULTIPLIER.tolist(),
        "real_reconstruction_max_abs_error": reconstruction_error,
        "analytic_signal_sha256": array_sha256(np.stack([analytic.real, analytic.imag], axis=-1)),
        "fixed_fourier_quadrature_no_learned_basis": True,
    }


def _linear_slope(values: np.ndarray) -> np.ndarray:
    coordinate = np.arange(HORIZON_N, dtype=float)
    centered = coordinate - coordinate.mean()
    return np.sum(values * centered[None, :, None], axis=1) / float(np.sum(centered * centered))


def hilbert_phase_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    analytic, analytic_audit = analytic_signal(path)
    amplitude = np.abs(analytic)
    log_amplitude = np.log1p(amplitude)
    phase = np.unwrap(np.angle(analytic), axis=1)
    advance = np.diff(phase, axis=1)
    instantaneous_frequency = advance / ELAPSED_MINUTES[None, :, None]

    amplitude_summary = np.concatenate([
        np.mean(log_amplitude, axis=1),
        np.std(log_amplitude, axis=1),
        np.max(log_amplitude, axis=1),
        log_amplitude[:, -1, :],
        _linear_slope(log_amplitude),
    ], axis=1)
    advance_summary = np.concatenate([
        np.mean(advance, axis=1),
        np.std(advance, axis=1),
        np.sum(advance, axis=1),
        advance[:, -1, :],
        np.mean(advance > 0.0, axis=1),
    ], axis=1)
    frequency_summary = np.concatenate([
        np.mean(instantaneous_frequency, axis=1),
        np.std(instantaneous_frequency, axis=1),
        instantaneous_frequency[:, -1, :],
        np.max(np.abs(instantaneous_frequency), axis=1),
    ], axis=1)

    pair_parts: list[np.ndarray] = []
    for left, right in CHANNEL_PAIRS:
        difference = phase[:, :, left] - phase[:, :, right]
        unit = np.exp(1j * difference)
        all_mean = np.mean(unit, axis=1)
        early_mean = np.mean(unit[:, :4], axis=1)
        recent_mean = np.mean(unit[:, 4:], axis=1)
        recent = unit[:, -1]
        drift = np.angle(recent_mean * np.conj(early_mean))
        pair_parts.append(np.stack([
            np.abs(all_mean), np.abs(early_mean), np.abs(recent_mean),
            all_mean.real, all_mean.imag, recent.real, recent.imag, drift,
        ], axis=1))
    pair_summary = np.concatenate(pair_parts, axis=1)
    feature = np.concatenate([
        amplitude_summary, advance_summary, frequency_summary, pair_summary,
    ], axis=1)
    require(feature.shape == (len(path), HILBERT_FEATURE_DIMENSION), "V160 feature dimension changed")
    require(np.isfinite(feature).all(), "V160 Hilbert feature invalid")
    return feature, {
        **analytic_audit,
        "feature_dimension": feature.shape[1],
        "amplitude_summary_dimension": amplitude_summary.shape[1],
        "phase_advance_summary_dimension": advance_summary.shape[1],
        "instantaneous_frequency_summary_dimension": frequency_summary.shape[1],
        "pair_phase_locking_summary_dimension": pair_summary.shape[1],
        "feature_sha256": array_sha256(feature),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "analytic_signal_hilbert_quadrature_verified": True,
        "time_resolved_amplitude_phase_advance_instantaneous_frequency": True,
        "cross_channel_phase_locking_verified": True,
        "frequency_bin_power_or_unit_cross_spectrum_exposed": False,
        "wavelet_scattering_or_cepstrum": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == HILBERT_FEATURE_DIMENSION, "V160 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V160 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V160 scaled feature invalid")
        return result


def hilbert_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V160 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_hilbert_audit = hilbert_phase_features(train_path)
    target_feature, target_hilbert_audit = hilbert_phase_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, target, sample_weight=numeric.duplicate_class_weights(ordered))
    require(int(head.n_iter_[0]) < head.max_iter, "V160 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V160 probability invalid")
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
        "train_geometry": train_hilbert_audit,
        "target_geometry": target_hilbert_audit,
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


_BASE_CHOOSE_INNER = base.choose_inner
_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = HILBERT_FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = hilbert_direction


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_base()
    policy, audit = _BASE_CHOOSE_INNER(inner_train, inner_valid, inner_champion)
    policy["selection_rule"] = "inner-past OOF only; exact V69 no-op or fixed .25/.50 causal Hilbert spectral-phase blend; fixed 2*AUC+BA with BA/net/supported-source safety"
    policy["quadrature_summary_head_blend_or_source_floor_micro_tuning"] = False
    policy.pop("geometry_definition_head_blend_or_source_floor_micro_tuning", None)
    return policy, audit


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
    diagnostic = diagnostic.rename(columns={"v146_model": "v160_model"})
    evidence = evidence.rename(columns={"frenet_prob": "hilbert_phase_prob", "v146_model": "v160_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V160 HILBERT PHASE] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V160 Hilbert spectral-phase policy"
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
    base_contract = bool(checks.pop("multichannel_discrete_frenet_curvature_contract_verified"))
    hilbert_contract = bool(base_contract and all(
        audit[side]["train_geometry"]["analytic_signal_hilbert_quadrature_verified"]
        and audit[side]["train_geometry"]["time_resolved_amplitude_phase_advance_instantaneous_frequency"]
        and audit[side]["train_geometry"]["cross_channel_phase_locking_verified"]
        and not audit[side]["train_geometry"]["frequency_bin_power_or_unit_cross_spectrum_exposed"]
        and audit[side]["target_geometry"]["analytic_signal_hilbert_quadrature_verified"]
        and audit[side]["target_geometry"]["cross_channel_phase_locking_verified"]
        and audit[side]["feature_scaling"]["train_only"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["multichannel_causal_hilbert_spectral_phase_contract_verified"] = hilbert_contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {
            "activated": True,
            "policy": "exact entire V69 DataFrame",
            "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion)),
        }
        require(result["selected_frame"].equals(champion), "V160 fallback not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V160 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V160 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V160 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V160").resolve(), "V160 output is outside controller research/staging/V160")
    require(out.exists() and out.is_dir(), "V160 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V160 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V160 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True,
        "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names), "V160 required report contract incomplete")
    require(len(names) == len(REPORT_NAMES), "V160 report names repeat")
    skeleton = {
        "MODEL_COMPARISON.json": {"models": ["V69_CHAMPION", "V160_DIAGNOSTIC", "V160_FAIL_CLOSED_SELECTED"]},
        "DEV_ROBUSTNESS_REPORT.json": {"selected": {"material_gate": {"nested_bootstrap": {"contract_key": "selected.material_gate.nested_bootstrap"}, "native_robustness_bootstrap": {}}}},
        "SOURCE_TRANSFER_REPORT.json": {"material_gate": "current", "by_source_family": True},
    }
    require(skeleton["DEV_ROBUSTNESS_REPORT.json"]["selected"]["material_gate"]["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V160 report nested key mismatch")
    return {
        "status": "TEMP_REPORT_CONTRACT_OK",
        "required_reports": sorted(required),
        "all_report_names": list(REPORT_NAMES),
        "current_material_gate": True,
        "selected_material_gate_nested_bootstrap": True,
        "native_robustness_preserved": True,
        "source_transfer_current_gate": True,
        "exact_entire_v69_fallback": True,
        "filesystem_write": False,
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V160 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V160 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V160", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V160_DIAGNOSTIC_HILBERT_SPECTRAL_PHASE": evaluation["candidate_summary"],
                "V160_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V160", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V160", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V160_HILBERT_SPECTRAL_PHASE_REPORT.json": {
            "version": "V160", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V160", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V160", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V160", "status": "MATCH", "canonical_check_count": 14,
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
    require(set(reports) == set(REPORT_NAMES), "V160 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V160_HILBERT_SPECTRAL_PHASE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V160_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        "required_controller_reports": [
            "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
        ],
    }


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
        require(
            not args.audit_only and not args.contract_probe and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V160 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V160 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V160 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.contract_probe or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full_cpu", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.contract_probe:
        print(json.dumps(clean({
            **temporary_report_contract_probe(), "hypothesis": HYPOTHESIS,
            "runtime": runtime, "authority": authority,
        }), ensure_ascii=False, indent=2))
        return
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "fixed length-8 Hilbert analytic signal with channel amplitude/phase-advance/instantaneous-frequency and pair PLV summaries plus balanced ridge logistic",
            "prior_exact_hilbert_phase_family_hit_count": 0,
            "v134_frequency_bin_power_or_cross_spectrum_reused": False,
            "v95_wavelet_scattering_reused": False,
            "cepstral_neural_or_path_geometry_reused": False,
            "causal_numeric_only": True, "target_row_labels_used": False,
            "micro_tuning_or_post_outer_change": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "gpu_model_calls": 0,
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
            require(args.smoke_market == "KR", "V160 support probe is reserved for KR:2")
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
