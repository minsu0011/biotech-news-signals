"""V132 fixed leaky echo-state horizon-reservoir direction challenger.

Each embargoed same-market train cut robust-scales only the immutable causal
numeric state, constructs the fixed eight-channel/eight-horizon pre-event path,
and orders it from 120 minutes toward 1 minute before the event.  A deterministic
label-independent 64-state sparse cyclic tanh reservoir with fixed input scale,
spectral radius, leak and zero initialization maps the path to final, temporal-mean
and temporal-RMS echo states.  A train-only robust feature scale and one fixed
duplicate/class-balanced ridge-logistic head predict direction.

There is no source or time environment, pairwise/listwise loss, decision-cell
reliability correction, return-salience weight, DMD/operator eigenspectrum, RQA,
GAF, path signature, CNN, threshold search, calibration or post-outer tuning.
Atomic V69 inner-past labels choose exact V69 or fixed .25/.50 blends.  Outer
labels are evaluation-only under a strict 35-minute embargo, V69 confidence and
high-confidence remain exact, and material failure restores exact entire V69.
Bounded modes use CPU30-31/two threads/no GPU and write nothing.
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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V132_FIXED_LEAKY_ECHO_STATE_RESERVOIR_V1"
VERSION = 132
HYPOTHESIS = "FIXED_LEAKY_ECHO_STATE_HORIZON_RESERVOIR_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

CHANNEL_N = 8
HORIZON_N = 8
RESERVOIR_N = 64
INPUT_SCALE = 0.45
SPECTRAL_RADIUS = 0.82
LEAK = 0.55
BIAS_SCALE = 0.05
REPRESENTATION_N = 3 * RESERVOIR_N
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 13201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "FIXED_LEAKY_ECHO_STATE_RESERVOIR_W0.25", "weight": 0.25},
    {"name": "FIXED_LEAKY_ECHO_STATE_RESERVOIR_W0.50", "weight": 0.50},
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
joint_path = scaffold.joint_path


def fixed_reservoir() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    generator = np.random.default_rng(SEED)
    input_matrix = generator.choice(np.asarray([-1.0, 1.0]), size=(RESERVOIR_N, CHANNEL_N))
    input_matrix *= INPUT_SCALE / np.sqrt(CHANNEL_N)
    bias = generator.choice(np.asarray([-1.0, 1.0]), size=RESERVOIR_N) * BIAS_SCALE
    recurrent = np.zeros((RESERVOIR_N, RESERVOIR_N), dtype=float)
    signs = np.where(np.arange(RESERVOIR_N) % 3 == 0, -1.0, 1.0)
    recurrent[np.arange(RESERVOIR_N), (np.arange(RESERVOIR_N) - 1) % RESERVOIR_N] = signs * SPECTRAL_RADIUS
    spectral = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    require(abs(spectral - SPECTRAL_RADIUS) <= 1e-10, "V132 fixed reservoir spectral radius changed")
    return input_matrix, recurrent, bias, {
        "input_matrix_sha256": array_sha256(input_matrix),
        "recurrent_matrix_sha256": array_sha256(recurrent),
        "bias_sha256": array_sha256(bias),
        "spectral_radius_recomputed": spectral,
        "recurrent_nonzero_n": int(np.count_nonzero(recurrent)),
        "deterministic_seed": SEED,
    }


INPUT_MATRIX, RECURRENT_MATRIX, RESERVOIR_BIAS, RESERVOIR_AUDIT = fixed_reservoir()
STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(scaffold.CHANNEL_NAMES),
    "horizon_n": HORIZON_N,
    "horizon_grid_input_minutes": scaffold.HORIZON_GRID.tolist(),
    "recurrent_order_minutes": scaffold.HORIZON_GRID[::-1].tolist(),
    "reservoir_state_n": RESERVOIR_N,
    "input_scale": INPUT_SCALE,
    "spectral_radius": SPECTRAL_RADIUS,
    "leak": LEAK,
    "bias_scale": BIAS_SCALE,
    "initial_state": "all zeros",
    "summary": ["final_state", "temporal_mean", "temporal_root_mean_square"],
    "representation_n": REPRESENTATION_N,
    "head_C": RIDGE_LOGISTIC_C,
    "reservoir": RESERVOIR_AUDIT,
}
require(len(scaffold.CHANNEL_SPEC) == CHANNEL_N and len(scaffold.HORIZON_GRID) == HORIZON_N, "V132 path geometry changed")
require(REPRESENTATION_N == 192, "V132 representation dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending and atomic IO only; no V119 output is read",
    }
    audit["v132_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def observed_chronological_path(robust_numeric: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    joint, inherited = joint_path(robust_numeric)
    observed = joint[:, 1:, 1:]
    require(observed.shape[1:] == (HORIZON_N, CHANNEL_N), "V132 observed path slice changed")
    chronological = observed[:, ::-1, :].copy()
    return chronological, {
        "rows": len(chronological), "horizons": HORIZON_N, "channels": CHANNEL_N,
        "order_minutes": scaffold.HORIZON_GRID[::-1].tolist(),
        "oldest_to_nearest_pre_event": True,
        "artificial_origin_removed": True, "time_coordinate_removed": True,
        "chronological_path_sha256": array_sha256(chronological),
        "inherited_interpolation_audit": inherited,
    }


def echo_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V132 reservoir path shape invalid")
    state = np.zeros((len(path), RESERVOIR_N), dtype=float)
    trajectory = np.empty((len(path), HORIZON_N, RESERVOIR_N), dtype=float)
    for step in range(HORIZON_N):
        excitation = path[:, step, :] @ INPUT_MATRIX.T + state @ RECURRENT_MATRIX.T + RESERVOIR_BIAS
        state = (1.0 - LEAK) * state + LEAK * np.tanh(excitation)
        trajectory[:, step, :] = state
    representation = np.concatenate([
        trajectory[:, -1, :],
        np.mean(trajectory, axis=1),
        np.sqrt(np.mean(np.square(trajectory), axis=1)),
    ], axis=1)
    require(representation.shape[1] == REPRESENTATION_N and np.isfinite(representation).all(), "V132 echo representation invalid")
    echo_residual = float(np.mean(np.abs(trajectory[:, 1:, :] - trajectory[:, :-1, :])))
    return representation, {
        "rows": len(representation), "representation_n": representation.shape[1],
        "trajectory_sha256": array_sha256(trajectory),
        "representation_sha256": array_sha256(representation),
        "trajectory_min": float(trajectory.min()), "trajectory_max": float(trajectory.max()),
        "mean_absolute_step_change": echo_residual,
        "label_independent_fixed_recurrent_transform": True,
        "zero_initialized_per_row": True,
    }


class RobustEchoScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustEchoScale":
        require(matrix.ndim == 2 and matrix.shape[1] == REPRESENTATION_N, "V132 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        self.scale = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V132 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        transformed = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(transformed).all(), "V132 scaled echo feature invalid")
        return transformed


def reservoir_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V132 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_chronological_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_chronological_path(target_state[:, :len(FEATURES)])
    train_feature, train_echo_audit = echo_features(train_path)
    target_feature, target_echo_audit = echo_features(target_path)
    scale = RobustEchoScale().fit(train_feature)
    train_design = scale.transform(train_feature)
    target_design = scale.transform(target_feature)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    coefficient = np.concatenate([head.coef_.reshape(-1), head.intercept_.reshape(-1)])
    require(np.isfinite(probability).all(), "V132 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_echo": train_echo_audit, "target_echo": target_echo_audit,
        "static_spec": STATIC_SPEC,
        "head": {
            "type": "duplicate/class-balanced fixed L2 ridge logistic",
            "C": RIDGE_LOGISTIC_C, "coefficient_sha256": array_sha256(coefficient),
            "coefficient_norm": float(np.linalg.norm(head.coef_)), "iterations": int(head.n_iter_[0]),
        },
        "target_rows_used_for_scale_reservoir_or_head": False, "target_labels_used": False,
        "threshold_or_calibration_tuned": False, "rank_list_or_pair_loss": False,
        "source_time_environment_or_decision_cell_reliability": False,
        "dmd_rqa_gaf_signature_cnn_or_fitted_recurrent_weights": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "predicted_up_rate": float(np.mean(probability >= 0.5)),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = reservoir_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 echo-state reservoir blend; fixed 2*AUC+BA with BA/net guard",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "reservoir_dimensions_constants_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v132_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V132 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = reservoir_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(baseline, challenger, selected["architecture"]["weight"])
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v132_model"] = selected["name"]
        outer_diagnostics = {architecture["name"]: metric(outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high) for architecture in ARCHITECTURES}
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf"]].copy()
        evidence["baseline_prob"] = baseline
        evidence["reservoir_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v132_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V132 ECHO] market={market} fold={fold} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V132 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V132 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V132 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V132 bootstrap blocks")
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
    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {"contract_key": "selected.material_gate.nested_bootstrap", "method": "paired market-fold-time block bootstrap over inner-locked V132 fixed echo-reservoir policy", "seed": SEED, "requested_draws": draws, "blocks": len(blocks), **{name: interval(items) for name, items in values.items()}}


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    reservoir_contract = all(
        audit[side]["causal_numeric_only"] and audit[side]["train_echo"]["label_independent_fixed_recurrent_transform"]
        and audit[side]["target_echo"]["label_independent_fixed_recurrent_transform"]
        and audit[side]["train_path"]["oldest_to_nearest_pre_event"] and audit[side]["target_path"]["oldest_to_nearest_pre_event"]
        and not audit[side]["target_rows_used_for_scale_reservoir_or_head"] and not audit[side]["target_labels_used"]
        and not audit[side]["threshold_or_calibration_tuned"] and not audit[side]["rank_list_or_pair_loss"]
        and not audit[side]["source_time_environment_or_decision_cell_reliability"]
        and not audit[side]["dmd_rqa_gaf_signature_cnn_or_fitted_recurrent_weights"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
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
        "strict_nested_chronology_all_folds": all(a["outer_chronology"]["strict_35m_embargo"] and a["inner_chronology"]["strict_35m_embargo"] and not a["outer_labels_used_for_selection"] for a in audits),
        "fixed_echo_state_reservoir_contract_verified": reservoir_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V132 fallback not exact V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"], "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "candidate_native_robustness_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(v) for v in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "current_hypothesis_not_inherited_from_v69": True}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V132 nested key mismatch")
    return gate


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V132 output not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V132", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V132_DIAGNOSTIC_FIXED_ECHO_RESERVOIR": evaluation["candidate_summary"], "V132_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V132", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V132", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_v69": not evaluation["material_pass"]},
        "V132_FIXED_LEAKY_ECHO_STATE_RESERVOIR_REPORT.json": {"version": "V132", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V132", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V132_ECHO_RESERVOIR_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V132_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {"output": str(out), "required_controller_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"]}


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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V132 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V132 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "fixed deterministic leaky echo-state reservoir over oldest-to-nearest causal horizon path plus ridge-logistic head",
            "causal_numeric_only": True, "target_row_labels_used": False, "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
            }, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            outer_champion = champion.loc[champion.market.eq("KR") & champion.fold.eq(2)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
            outer_valid = aligned_dev(dev, outer_champion)
            start = pd.Timestamp(outer_valid.event_time_utc.min())
            outer_train = prior_train(dev, "KR", start, outer_valid)
            audit_chronology = chronology(outer_train, outer_valid, "V132 KR2 support")
            probability, model_audit = reservoir_direction(outer_train, outer_valid)
            baseline = outer_champion.prob.to_numpy(float)
            result = {
                "status": "KR2_SUPPORT_OK", "runtime": runtime, "chronology": audit_chronology,
                "baseline": metric(outer_valid, baseline, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
                "candidate_w025_evaluation_only": metric(outer_valid, blend_probability(baseline, probability, 0.25), outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
                "model_audit": model_audit, "output_written": False,
            }
            print(json.dumps(clean(result), ensure_ascii=False, indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS,
        "runtime": runtime, "folds_executed": len(audits), "selected_models": [a["policy"]["selected"]["name"] for a in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": material_gate(evaluation), "output_written": False,
    }
    if args.smoke_test:
        summary.update({
            "raw_inner_baseline": audits[0]["policy"]["baseline"], "raw_inner_selected": audits[0]["policy"]["selected"],
            "raw_inner_best_non_noop": best_non_noop, "raw_outer_baseline": audits[0]["outer_baseline"],
            "raw_outer_selected": audits[0]["outer_candidate"],
            "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
            "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"],
            "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        })
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
