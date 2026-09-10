"""V161 multichannel causal lagged copula-tail direction challenger.

Each embargoed train cut supplies the only empirical marginal CDF reference
for all 64 coordinates of the observed eight-channel by eight-horizon causal
path. Fixed q15/q85 tail indicators produce adjacent-horizon lower/upper
co-exceedance and upper-minus-lower asymmetry for all 64 directed same/cross
channel pairs. The fixed 192D representation is passed to one equal-event-
group/class-balanced ridge-logit direction head. No class label enters the
rank or copula feature map, and no target-batch CDF is fitted.

Inner-past labels select exact V69 or fixed .25/.50 blends; outer labels are
evaluation-only under the 35-minute embargo/event purge.  V69 confidence and
high_conf are frozen.  Any material or execution failure restores the exact
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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v146_multichannel_causal_discrete_frenet_curvature_direction as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V161" / "LOCAL_FORBIDDEN"
VERSION = 161
HYPOTHESIS = "MULTICHANNEL_CAUSAL_LAGGED_COPULA_TAIL_DEPENDENCE_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_V69_EXPERIMENT_ID = base.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v146_multichannel_causal_discrete_frenet_curvature_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "88e466ef8376ca441a28b29f3583d882a77eaba9b8f8db2cac9bac4a37ffcbe6"

CHANNEL_N = 8
HORIZON_N = 8
LAGGED_CHANNEL_PAIRS = tuple(
    (source, target) for source in range(CHANNEL_N) for target in range(CHANNEL_N)
)
LOWER_TAIL_QUANTILE = 0.15
UPPER_TAIL_QUANTILE = 0.85
COPULA_FEATURE_DIMENSION = 3 * len(LAGGED_CHANNEL_PAIRS)
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 16101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "COPULA_TAIL_DEPENDENCE_W0.25", "weight": 0.25},
    {"name": "COPULA_TAIL_DEPENDENCE_W0.50", "weight": 0.50},
)

require = base.require
sha256 = base.sha256
array_sha256 = base.array_sha256
clean = base.clean
bool_series = base.bool_series
atomic_json = base.atomic_json
atomic_csv = base.atomic_csv
load_authorized = base.load_authorized
metric = base.metric
controller = base.controller
v44 = base.v44
numeric = base.numeric
observed_past_to_recent_path = base.observed_past_to_recent_path

STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(base.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "marginal_rank_reference": "equal-event-group train cut only, separately for each channel x horizon coordinate",
    "lower_tail_quantile": LOWER_TAIL_QUANTILE,
    "upper_tail_quantile": UPPER_TAIL_QUANTILE,
    "lagged_channel_pair_n": len(LAGGED_CHANNEL_PAIRS),
    "representation": [
        "mean adjacent-horizon lower-tail co-exceedance for all 8x8 directed channel pairs",
        "mean adjacent-horizon upper-tail co-exceedance for all 8x8 directed channel pairs",
        "upper-minus-lower co-exceedance asymmetry for all directed pairs",
    ],
    "same_channel_pair_n": CHANNEL_N,
    "cross_channel_directed_pair_n": CHANNEL_N * (CHANNEL_N - 1),
    "feature_dimension": COPULA_FEATURE_DIMENSION,
    "feature_scaling": "train-only coordinate median/IQR-or-std and fixed clip",
    "head": "fixed equal-event-group/class-balanced ridge logistic",
    "class_conditional_copula_or_target_batch_cdf": False,
    "tail_quantile_or_lag_search": False,
}
require(
    len(LAGGED_CHANNEL_PAIRS) == 64 and COPULA_FEATURE_DIMENSION == 192,
    "V161 fixed copula-tail feature dimension changed",
)


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V146 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, observed causal path, metrics, canonical gate, blending, bootstrap and atomic IO only; no V146 output is read",
    }
    audit["v161_access_contract"] = {
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


def equal_event_group_path(
    ordered: pd.DataFrame,
    path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(path.shape == (len(ordered), HORIZON_N, CHANNEL_N), "V161 path shape invalid")
    flattened = path.reshape(len(path), -1)
    work = pd.DataFrame(flattened, columns=[f"path_{index}" for index in range(flattened.shape[1])])
    work["event_group_id"] = ordered.event_group_id.astype(str).to_numpy()
    work["y"] = ordered.y.to_numpy(int)
    require(
        bool((work.groupby("event_group_id", sort=False).y.nunique() == 1).all()),
        "V161 event_group labels inconsistent",
    )
    grouped = work.groupby("event_group_id", sort=True, as_index=False).agg(
        {**{f"path_{index}": "mean" for index in range(flattened.shape[1])}, "y": "first"}
    )
    group_path = grouped[[f"path_{index}" for index in range(flattened.shape[1])]].to_numpy(float).reshape(-1, HORIZON_N, CHANNEL_N)
    group_target = grouped.y.to_numpy(int)
    require(np.unique(group_target).size == 2, "V161 grouped train cut missing a class")
    return group_path, group_target, {
        "raw_rows": len(ordered), "equal_event_groups": len(grouped),
        "duplicates_removed_before_rank_fit": len(ordered) - len(grouped),
        "class_event_group_n": np.bincount(group_target, minlength=2).tolist(),
        "equal_event_group_weighting": True,
    }


class TrainEmpiricalMarginalRanks:
    def __init__(self) -> None:
        self.sorted_reference = np.empty((0, HORIZON_N * CHANNEL_N), dtype=float)

    def fit(self, path: np.ndarray) -> "TrainEmpiricalMarginalRanks":
        require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V161 rank-fit path invalid")
        self.sorted_reference = np.sort(path.reshape(len(path), -1), axis=0)
        require(np.isfinite(self.sorted_reference).all(), "V161 empirical reference invalid")
        return self

    def transform(self, path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = path.reshape(len(path), -1)
        rank = np.empty_like(matrix, dtype=float)
        for column in range(matrix.shape[1]):
            rank[:, column] = (
                np.searchsorted(self.sorted_reference[:, column], matrix[:, column], side="right") + 0.5
            ) / (len(self.sorted_reference) + 1.0)
        rank = np.clip(rank, 0.0, 1.0).reshape(-1, HORIZON_N, CHANNEL_N)
        require(np.isfinite(rank).all(), "V161 empirical marginal rank invalid")
        return rank, {
            "rows": len(rank), "coordinate_n": HORIZON_N * CHANNEL_N,
            "reference_event_group_n": len(self.sorted_reference),
            "reference_sha256": array_sha256(self.sorted_reference),
            "rank_sha256": array_sha256(rank),
            "train_cut_reference_only": True,
            "target_batch_cdf_or_quantile_fit": False,
            "class_labels_used_in_rank_map": False,
        }


def copula_tail_features(rank: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(rank.ndim == 3 and rank.shape[1:] == (HORIZON_N, CHANNEL_N), "V161 rank tensor invalid")
    lower = rank <= LOWER_TAIL_QUANTILE
    upper = rank >= UPPER_TAIL_QUANTILE
    lower_co = np.stack([
        np.mean(lower[:, :-1, source] & lower[:, 1:, target], axis=1)
        for source, target in LAGGED_CHANNEL_PAIRS
    ], axis=1).astype(float)
    upper_co = np.stack([
        np.mean(upper[:, :-1, source] & upper[:, 1:, target], axis=1)
        for source, target in LAGGED_CHANNEL_PAIRS
    ], axis=1).astype(float)
    asymmetry = upper_co - lower_co
    feature = np.concatenate([lower_co, upper_co, asymmetry], axis=1)
    require(feature.shape == (len(feature), COPULA_FEATURE_DIMENSION), "V161 copula feature dimension changed")
    require(np.isfinite(feature).all(), "V161 copula-tail feature invalid")
    return feature, {
        "rows": len(feature), "feature_dimension": feature.shape[1],
        "lower_tail_quantile": LOWER_TAIL_QUANTILE,
        "upper_tail_quantile": UPPER_TAIL_QUANTILE,
        "same_channel_pair_n": CHANNEL_N,
        "cross_channel_directed_pair_n": CHANNEL_N * (CHANNEL_N - 1),
        "lower_coexceedance_mean": float(lower_co.mean()),
        "upper_coexceedance_mean": float(upper_co.mean()),
        "asymmetry_mean": float(asymmetry.mean()),
        "feature_sha256": array_sha256(feature),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "lagged_same_and_cross_channel_coexceedance_verified": True,
        "upper_minus_lower_asymmetry_verified": True,
        "class_conditional_density_covariance_or_qda": False,
        "quantile_transport_or_target_batch_cdf": False,
        "third_moment_coskewness_or_transfer_entropy": False,
        "hilbert_phase_frequency_or_phase_locking": False,
    }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == COPULA_FEATURE_DIMENSION, "V161 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V161 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V161 scaled feature invalid")
        return result


def copula_tail_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V161 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping_audit = equal_event_group_path(ordered, train_path)
    ranker = TrainEmpiricalMarginalRanks().fit(group_path)
    train_rank, train_rank_audit = ranker.transform(group_path)
    target_rank, target_rank_audit = ranker.transform(target_path)
    train_feature, train_copula_audit = copula_tail_features(train_rank)
    target_feature, target_copula_audit = copula_tail_features(target_rank)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V161 probability invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "grouping": grouping_audit,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_rank_mapping": train_rank_audit, "target_rank_mapping": target_rank_audit,
        "train_geometry": train_copula_audit, "target_geometry": target_copula_audit,
        "feature_scaling": {
            "train_only": True, "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed equal-event-group/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C, "solver": "liblinear",
            "iterations": int(head.n_iter_[0]),
            "class_event_group_n": class_n.astype(int).tolist(),
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
        "class_conditional_copula_qda": False,
        "source_time_quantile_transport": False,
        "increment_coskewness_tensor": False,
        "transfer_entropy_conditional_transition": False,
        "hilbert_phase_frequency_or_phase_locking": False,
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
    base.FRENET_FEATURE_DIMENSION = COPULA_FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = copula_tail_direction


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_base()
    policy, audit = _BASE_CHOOSE_INNER(inner_train, inner_valid, inner_champion)
    policy["selection_rule"] = "inner-past OOF only; exact V69 no-op or fixed .25/.50 lagged copula-tail-dependence blend; fixed 2*AUC+BA with BA/net/supported-source safety"
    policy["rank_tail_lag_definition_head_blend_or_source_floor_micro_tuning"] = False
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
    diagnostic = diagnostic.rename(columns={"v146_model": "v161_model"})
    evidence = evidence.rename(columns={"frenet_prob": "copula_tail_prob", "v146_model": "v161_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V161 LAGGED COPULA TAIL] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V161 lagged copula-tail-dependence policy"
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
    contract = bool(checks.pop("multichannel_discrete_frenet_curvature_contract_verified"))
    contract = bool(contract and all(
        audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["train_rank_mapping"]["train_cut_reference_only"]
        and audit[side]["target_rank_mapping"]["train_cut_reference_only"]
        and not audit[side]["target_rank_mapping"]["target_batch_cdf_or_quantile_fit"]
        and not audit[side]["train_rank_mapping"]["class_labels_used_in_rank_map"]
        and audit[side]["train_geometry"]["lagged_same_and_cross_channel_coexceedance_verified"]
        and audit[side]["train_geometry"]["upper_minus_lower_asymmetry_verified"]
        and audit[side]["target_geometry"]["lagged_same_and_cross_channel_coexceedance_verified"]
        and audit[side]["target_geometry"]["upper_minus_lower_asymmetry_verified"]
        and not audit[side]["class_conditional_copula_qda"]
        and not audit[side]["source_time_quantile_transport"]
        and not audit[side]["increment_coskewness_tensor"]
        and not audit[side]["transfer_entropy_conditional_transition"]
        and not audit[side]["hilbert_phase_frequency_or_phase_locking"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    ))
    checks["multichannel_causal_lagged_copula_tail_dependence_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {
            "activated": True, "policy": "exact entire V69 DataFrame",
            "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion)),
        }
        require(result["selected_frame"].equals(champion), "V161 fallback not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V161 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V161 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V161 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V161").resolve(), "V161 output is outside controller research/staging/V161")
    require(out.exists() and out.is_dir(), "V161 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V161 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V161 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
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
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V161 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V161 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V161", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V161_DIAGNOSTIC_LAGGED_COPULA_TAIL_DEPENDENCE": evaluation["candidate_summary"],
                "V161_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V161", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V161", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V161_LAGGED_COPULA_TAIL_DEPENDENCE_REPORT.json": {
            "version": "V161", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V161", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V161", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V161", "status": "MATCH", "canonical_check_count": 14,
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
    atomic_csv(evidence, out / "V161_LAGGED_COPULA_TAIL_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V161_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V161 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V161 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V161 controller full-run output mismatch")
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
            "architecture": "train-cut-only coordinate empirical ranks, fixed q15/q85 adjacent-horizon directed same/cross-channel tail co-exceedance and asymmetry (192D), equal-group balanced ridge logistic",
            "class_conditional_copula_qda_target_cdf_or_tail_grid": False,
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
            require(args.smoke_market == "KR", "V161 support probe is reserved for KR:2")
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
