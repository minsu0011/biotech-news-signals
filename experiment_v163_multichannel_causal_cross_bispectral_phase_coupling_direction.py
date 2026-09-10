"""V163 multichannel causal cross-bispectral phase-coupling direction.

Each strict-past same-market cut robustly scales the immutable 37 causal
numeric fields and maps them to the observed 8-channel by 8-horizon
past-to-recent path.  A fixed length-eight real DFT supplies four deterministic
admissible nonzero frequency triads f1+f2=f3.  Within-channel normalized
bispectral magnitude and biphase cosine/sine are retained.  For every unordered
channel pair, both directed cross-bispectra are summarized across the four
triads by normalized magnitude and circular biphase coupling.  A train-only
robust feature scale and one duplicate/class-balanced ridge-logit head produce
the challenger probability.

V163 is third-order frequency-triad coupling, not V134 second-order auto-power
or cross-spectrum, V149 time-domain increment coskewness, or V160 time-local
Hilbert analytic phase.  No frequency grid, threshold, spectral feature or
representation is learned from labels.  Inner-only selection chooses exact
V69 or fixed .25/.50 blends; outer labels are evaluation-only under strict
35-minute embargo/event purge.  Atomic V69 confidence/high_conf remain exact
and any material failure restores the exact entire atomic V69 frame.

Audit, contract, US:2 smoke and KR:2 support run on CPU30-31 with two threads,
GPU disabled and no writes.  Full execution is accepted only through a
no-argument MARKET_BIO_VERSION_OUTPUT controller binding.
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

import experiment_v160_multichannel_causal_hilbert_spectral_phase_direction as support


base = support.base
ROOT = support.ROOT
V69_DIR = support.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V163" / "LOCAL_FORBIDDEN"
VERSION = 163
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CROSS_BISPECTRAL_PHASE_COUPLING_DIRECTION_V1"
FEATURES = support.FEATURES
EXPECTED_MARKET_FOLD_ROWS = support.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = support.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v160_multichannel_causal_hilbert_spectral_phase_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "5759727dab03377abc5ea9e669352f659c1fa1c6e731d83f221de6234a8dfb78"

CHANNEL_N = 8
HORIZON_N = 8
NONZERO_FREQUENCY_N = 4
FREQUENCY_TRIADS = ((1, 1, 2), (1, 2, 3), (1, 3, 4), (2, 2, 4))
TRIAD_N = len(FREQUENCY_TRIADS)
CHANNEL_PAIR_N = CHANNEL_N * (CHANNEL_N - 1) // 2
WITHIN_FEATURES_PER_TRIAD = 3
CROSS_SUMMARIES_PER_DIRECTION = 5
WITHIN_FEATURE_DIMENSION = CHANNEL_N * TRIAD_N * WITHIN_FEATURES_PER_TRIAD
CROSS_FEATURE_DIMENSION = CHANNEL_PAIR_N * 2 * CROSS_SUMMARIES_PER_DIRECTION
BISPECTRAL_FEATURE_DIMENSION = WITHIN_FEATURE_DIMENSION + CROSS_FEATURE_DIMENSION
SPECTRAL_EPSILON = 1e-10
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 16301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CROSS_BISPECTRAL_PHASE_COUPLING_W0.25", "weight": 0.25},
    {"name": "CROSS_BISPECTRAL_PHASE_COUPLING_W0.50", "weight": 0.50},
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

CHANNEL_PAIRS = tuple(
    (left, right)
    for left in range(CHANNEL_N)
    for right in range(left + 1, CHANNEL_N)
)
STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(base.scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "transform": "length-8 orthonormal rFFT after per-event/channel horizon centering",
    "nonzero_frequency_indices": [1, 2, 3, 4],
    "admissible_frequency_triads_f1_plus_f2_equals_f3": [list(item) for item in FREQUENCY_TRIADS],
    "within_channel": "per channel/triad total-energy normalized bispectral magnitude plus biphase cosine and sine",
    "cross_channel": "per unordered pair, both directed source-source-target cross-bispectra summarized across triads by mean/max normalized magnitude, biphase PLV and circular mean cosine/sine",
    "within_feature_dimension": WITHIN_FEATURE_DIMENSION,
    "cross_feature_dimension": CROSS_FEATURE_DIMENSION,
    "feature_dimension": BISPECTRAL_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip +/-8",
    "head": "fixed duplicate/class-balanced C=0.5 ridge logistic",
    "learned_frequency_grid_threshold_or_spectral_feature": False,
    "second_order_auto_or_cross_spectrum": False,
    "time_domain_coskewness": False,
    "hilbert_analytic_phase": False,
}
REPORT_NAMES = (
    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
    "V163_CROSS_BISPECTRAL_PHASE_COUPLING_REPORT.json", "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json",
    "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json",
    "RUN_STATUS.json",
)
require(
    TRIAD_N == 4 and CHANNEL_PAIR_N == 28
    and WITHIN_FEATURE_DIMENSION == 96
    and CROSS_FEATURE_DIMENSION == 280
    and BISPECTRAL_FEATURE_DIMENSION == 376
    and all(first <= second and first + second == third <= NONZERO_FREQUENCY_N for first, second, third in FREQUENCY_TRIADS),
    "V163 static bispectral representation changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V160 utility scaffold changed")
    audit = support.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, causal path interpolation, metrics, canonical gate, blending, bootstrap and atomic IO utilities only; no V160 output is read",
    }
    audit["v163_access_contract"] = {
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


def _bispectrum(
    spectrum: np.ndarray,
    source: int,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for first, second, third in FREQUENCY_TRIADS:
        values.append(
            spectrum[:, first, source]
            * spectrum[:, second, source]
            * np.conj(spectrum[:, third, target])
        )
    bispectrum = np.stack(values, axis=1)
    unit = bispectrum / np.maximum(np.abs(bispectrum), SPECTRAL_EPSILON)
    return bispectrum, unit


def cross_bispectral_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V163 path shape invalid")
    centered = path - np.mean(path, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1, norm="ortho")
    require(spectrum.shape[1:] == (NONZERO_FREQUENCY_N + 1, CHANNEL_N), "V163 rFFT shape changed")
    energy = np.sum(np.abs(spectrum[:, 1:, :]) ** 2, axis=1)
    require(np.isfinite(energy).all(), "V163 spectral energy invalid")

    within_parts: list[np.ndarray] = []
    for channel in range(CHANNEL_N):
        bispectrum, unit = _bispectrum(spectrum, channel, channel)
        normalized_magnitude = np.abs(bispectrum) / np.maximum(
            energy[:, channel, None] ** 1.5, SPECTRAL_EPSILON,
        )
        within_parts.append(np.stack([
            normalized_magnitude, unit.real, unit.imag,
        ], axis=2).reshape(len(path), -1))
    within = np.concatenate(within_parts, axis=1)

    cross_parts: list[np.ndarray] = []
    for left, right in CHANNEL_PAIRS:
        for source, target in ((left, right), (right, left)):
            bispectrum, unit = _bispectrum(spectrum, source, target)
            denominator = energy[:, source, None] * np.sqrt(np.maximum(energy[:, target, None], 0.0))
            normalized_magnitude = np.abs(bispectrum) / np.maximum(denominator, SPECTRAL_EPSILON)
            circular_mean = np.mean(unit, axis=1)
            cross_parts.append(np.stack([
                np.mean(normalized_magnitude, axis=1),
                np.max(normalized_magnitude, axis=1),
                np.abs(circular_mean),
                circular_mean.real,
                circular_mean.imag,
            ], axis=1))
    cross = np.concatenate(cross_parts, axis=1)
    feature = np.concatenate([within, cross], axis=1)
    require(feature.shape == (len(path), BISPECTRAL_FEATURE_DIMENSION), "V163 feature dimension changed")
    require(np.isfinite(feature).all(), "V163 bispectral feature invalid")
    return feature, {
        "rows": len(path),
        "frequency_triads": [list(item) for item in FREQUENCY_TRIADS],
        "within_feature_dimension": within.shape[1],
        "cross_feature_dimension": cross.shape[1],
        "feature_dimension": feature.shape[1],
        "feature_sha256": array_sha256(feature),
        "spectrum_sha256": array_sha256(np.stack([spectrum.real, spectrum.imag], axis=-1)),
        "normalized_magnitude_q10_q50_q90": [
            float(np.quantile(np.abs(within[:, 0::3]), quantile)) for quantile in (0.1, 0.5, 0.9)
        ],
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "admissible_frequency_triad_identity_verified": True,
        "within_channel_bispectral_magnitude_and_biphase_verified": True,
        "directed_cross_channel_biphase_coupling_verified": True,
        "learned_frequency_grid_threshold_or_spectral_feature": False,
        "second_order_frequency_bin_power_or_unit_cross_spectrum": False,
        "time_domain_increment_coskewness": False,
        "hilbert_time_local_analytic_phase": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == BISPECTRAL_FEATURE_DIMENSION, "V163 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V163 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V163 scaled feature invalid")
        return result


def bispectral_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V163 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    train_feature, train_bispectral_audit = cross_bispectral_features(train_path)
    target_feature, target_bispectral_audit = cross_bispectral_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, target, sample_weight=numeric.duplicate_class_weights(ordered))
    require(int(head.n_iter_[0]) < head.max_iter, "V163 ridge logistic did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V163 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_bispectral_audit, "target_geometry": target_bispectral_audit,
        "feature_scaling": {
            "train_only": True, "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C, "solver": "liblinear",
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
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


_BASE_CHOOSE_INNER = base.choose_inner
_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = BISPECTRAL_FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = bispectral_direction


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_base()
    policy, audit = _BASE_CHOOSE_INNER(inner_train, inner_valid, inner_champion)
    policy["selection_rule"] = "inner-past OOF only; exact V69 no-op or fixed .25/.50 cross-bispectral phase-coupling blend; fixed 2*AUC+BA with BA/net/supported-source safety"
    policy["triad_cross_summary_head_blend_or_source_floor_micro_tuning"] = False
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
    diagnostic = diagnostic.rename(columns={"v146_model": "v163_model"})
    evidence = evidence.rename(columns={"frenet_prob": "bispectral_prob", "v146_model": "v163_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V163 CROSS-BISPECTRAL] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V163 cross-bispectral phase-coupling policy"
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
    bispectral_contract = bool(base_contract and all(
        audit[side]["train_geometry"]["admissible_frequency_triad_identity_verified"]
        and audit[side]["train_geometry"]["within_channel_bispectral_magnitude_and_biphase_verified"]
        and audit[side]["train_geometry"]["directed_cross_channel_biphase_coupling_verified"]
        and not audit[side]["train_geometry"]["learned_frequency_grid_threshold_or_spectral_feature"]
        and not audit[side]["train_geometry"]["second_order_frequency_bin_power_or_unit_cross_spectrum"]
        and not audit[side]["train_geometry"]["time_domain_increment_coskewness"]
        and not audit[side]["train_geometry"]["hilbert_time_local_analytic_phase"]
        and audit[side]["target_geometry"]["admissible_frequency_triad_identity_verified"]
        and audit[side]["target_geometry"]["directed_cross_channel_biphase_coupling_verified"]
        and audit[side]["feature_scaling"]["train_only"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["multichannel_causal_cross_bispectral_phase_coupling_contract_verified"] = bispectral_contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {
            "activated": True, "policy": "exact entire V69 DataFrame",
            "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion)),
        }
        require(result["selected_frame"].equals(champion), "V163 fallback not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V163 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V163 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V163 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V163").resolve(), "V163 output is outside controller research/staging/V163")
    require(out.exists() and out.is_dir(), "V163 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V163 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V163 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def temporary_report_contract_probe() -> dict[str, Any]:
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    names = set(REPORT_NAMES)
    require(required.issubset(names) and len(names) == len(REPORT_NAMES), "V163 report inventory invalid")
    return {
        "status": "TEMP_REPORT_CONTRACT_OK",
        "required_reports": sorted(required), "all_report_names": list(REPORT_NAMES),
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
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V163 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V163 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V163", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V163_DIAGNOSTIC_CROSS_BISPECTRAL_PHASE_COUPLING": evaluation["candidate_summary"],
                "V163_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V163", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V163", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V163_CROSS_BISPECTRAL_PHASE_COUPLING_REPORT.json": {
            "version": "V163", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V163", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V163", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V163", "status": "MATCH", "canonical_check_count": 14,
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
    require(set(reports) == set(REPORT_NAMES), "V163 report inventory changed")
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V163_CROSS_BISPECTRAL_PHASE_COUPLING_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V163_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V163 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V163 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V163 controller full-run output mismatch")
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
            "architecture": "fixed four frequency-triad within/cross-channel normalized bispectral magnitude and biphase coupling summaries plus balanced ridge logistic",
            "prior_exact_bispectrum_family_hit_count": 0,
            "v134_second_order_auto_cross_spectrum_reused": False,
            "v149_time_domain_coskewness_reused": False,
            "v160_hilbert_time_local_phase_reused": False,
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
            require(args.smoke_market == "KR", "V163 support probe is reserved for KR:2")
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
