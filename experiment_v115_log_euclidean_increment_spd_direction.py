"""V115 log-Euclidean causal increment-SPD direction challenger.

Each embargoed same-market past cut robustly scales the immutable causal
numeric state and constructs one fixed joint multichannel horizon path per
event.  The event's local path increments form a regularized uncentered
second-moment symmetric-positive-definite matrix.  Exact eigendecomposition
maps that matrix through the principal matrix logarithm into the log-Euclidean
tangent space, whose isometric upper triangle is robustly scaled on training
rows only.  One fixed duplicate/class-balanced ridge-logistic head produces
direction probability.

This is not class covariance or QDA, a path signature/iterated integral/Levy
area, Haar scattering, DTW/prototype, NMF/nonnegative factorization, vMF or
other class density, graph, kernel, tree, or neural representation.  Earlier
committed V69 OOF labels alone choose exact V69 or fixed .25/.50 blends.  Outer
labels are evaluation-only under strict 35-minute nested chronology.  V69
confidence/high-confidence remain exact, and material failure restores the
exact entire atomic V69 DataFrame after canonical 14-gate recomputation.

Audit and US:2 CPU30-31/two-thread smoke write nothing.  With an authorized
MARKET_BIO_VERSION_OUTPUT, no-argument invocation performs the six-fold full
run and writes all required reports/current material-gate evidence.
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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v113_multichannel_rough_path_signature_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V115_LOG_EUCLIDEAN_INCREMENT_SPD_V1"
VERSION = 115
HYPOTHESIS = "LOG_EUCLIDEAN_CAUSAL_INCREMENT_SPD_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v113_multichannel_rough_path_signature_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "649d9c593619b08ab54c818bd208be21a0c2e74bcf38894c79bcfa7463c7f738"

SPD_RIDGE = 0.05
TANGENT_CLIP = 8.0
MATRIX_DIMENSION = scaffold.PATH_DIMENSION
TANGENT_DIMENSION = MATRIX_DIMENSION * (MATRIX_DIMENSION + 1) // 2
RIDGE_LOGISTIC_C = 0.50
SEED = 11501
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "LOG_EUCLIDEAN_INCREMENT_SPD_W0.25", "weight": 0.25},
    {"name": "LOG_EUCLIDEAN_INCREMENT_SPD_W0.50", "weight": 0.50},
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


def verify_static_spec() -> dict[str, Any]:
    require(MATRIX_DIMENSION == 9 and TANGENT_DIMENSION == 45, "V115 SPD tangent dimension changed")
    return {
        "path_input_unique_feature_n": scaffold.STATIC_SPEC["path_input_unique_feature_n"],
        "path_channel_n": scaffold.STATIC_SPEC["channel_n"],
        "path_horizon_n": scaffold.STATIC_SPEC["horizon_n"],
        "increment_n": scaffold.STATIC_SPEC["horizon_n"],
        "spd_matrix_dimension": MATRIX_DIMENSION,
        "log_euclidean_tangent_dimension": TANGENT_DIMENSION,
        "path_channel_spec_sha256": scaffold.STATIC_SPEC["channel_spec_sha256"],
        "label_independent_descriptor_spec": True,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V113 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, fixed causal path construction, metrics, canonical gate, blending, and atomic IO only; no V113 output is read",
    }
    audit["v115_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def increment_spd_tangent(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[2] == MATRIX_DIMENSION, "V115 path shape invalid")
    increments = np.diff(path, axis=1)
    gram = np.einsum("nki,nkj->nij", increments, increments) / increments.shape[1]
    spd = gram + SPD_RIDGE * np.eye(MATRIX_DIMENSION, dtype=float)[None, :, :]
    eigenvalues, eigenvectors = np.linalg.eigh(spd)
    require(np.isfinite(eigenvalues).all() and float(eigenvalues.min()) >= SPD_RIDGE - 1e-10, "V115 SPD eigenvalue invalid")
    log_eigenvalues = np.log(eigenvalues)
    log_matrix = np.einsum("nik,nk,njk->nij", eigenvectors, log_eigenvalues, eigenvectors)
    reconstructed = np.einsum("nik,nk,njk->nij", eigenvectors, np.exp(log_eigenvalues), eigenvectors)
    reconstruction_error = float(np.max(np.abs(reconstructed - spd)))
    symmetry_error = float(np.max(np.abs(log_matrix - np.swapaxes(log_matrix, 1, 2))))
    diagonal = np.diagonal(log_matrix, axis1=1, axis2=2)
    upper_i, upper_j = np.triu_indices(MATRIX_DIMENSION, k=1)
    tangent = np.concatenate([diagonal, np.sqrt(2.0) * log_matrix[:, upper_i, upper_j]], axis=1)
    frobenius_error = float(np.max(np.abs(np.sum(log_matrix * log_matrix, axis=(1, 2)) - np.sum(tangent * tangent, axis=1))))
    require(tangent.shape[1] == TANGENT_DIMENSION and np.isfinite(tangent).all(), "V115 tangent vector invalid")
    require(reconstruction_error <= 1e-10 and symmetry_error <= 1e-10 and frobenius_error <= 1e-9, "V115 log-Euclidean identities failed")
    return tangent, {
        "increment_n": increments.shape[1], "matrix_dimension": MATRIX_DIMENSION,
        "spd_ridge": SPD_RIDGE, "matrix_log": "principal exact symmetric eigendecomposition",
        "minimum_eigenvalue": float(eigenvalues.min()),
        "median_minimum_row_eigenvalue": float(np.median(eigenvalues[:, 0])),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number_median": float(np.median(eigenvalues[:, -1] / eigenvalues[:, 0])),
        "matrix_exp_reconstruction_max_abs_error": reconstruction_error,
        "log_matrix_symmetry_max_abs_error": symmetry_error,
        "isometric_frobenius_max_abs_error": frobenius_error,
        "tangent_dimension": tangent.shape[1],
        "increment_sha256": array_sha256(increments),
        "spd_sha256": array_sha256(spd),
        "log_tangent_sha256": array_sha256(tangent),
    }


class RobustTangentScale:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustTangentScale":
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        standard = np.std(matrix, axis=0)
        self.scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V115 tangent scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -TANGENT_CLIP, TANGENT_CLIP)
        require(np.isfinite(result).all(), "V115 scaled tangent invalid")
        return result


def log_euclidean_spd_direction(
    train: pd.DataFrame, target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V115 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = scaffold.joint_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = scaffold.joint_path(target_state[:, :len(FEATURES)])
    train_tangent, train_spd_audit = increment_spd_tangent(train_path)
    target_tangent, target_spd_audit = increment_spd_tangent(target_path)
    tangent_scale = RobustTangentScale().fit(train_tangent)
    train_design = tangent_scale.transform(train_tangent)
    target_design = tangent_scale.transform(target_tangent)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=500,
        random_state=SEED, n_jobs=1,
    )
    head.fit(train_design, target, sample_weight=head_weight)
    require(int(head.n_iter_[0]) < 500, "V115 ridge-logistic head did not converge")
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    coefficient = np.r_[head.intercept_, head.coef_.ravel()]
    require(np.isfinite(probability).all(), "V115 probability invalid")
    return probability, {
        "market": str(ordered.market.iloc[0]), "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES), "causal_numeric_only": True,
        "categorical_text_source_ticker_feature_n": 0,
        "static_spec": STATIC_SPEC,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_path": train_path_audit, "target_path": target_path_audit,
        "train_spd": train_spd_audit, "target_spd": target_spd_audit,
        "tangent_scaling": {
            "train_only": True, "clip": TANGENT_CLIP,
            "median_sha256": array_sha256(tangent_scale.median),
            "scale_sha256": array_sha256(tangent_scale.scale),
        },
        "head": {
            "architecture": "fixed duplicate/class-balanced ridge logistic on per-event log-Euclidean increment-SPD tangent",
            "ridge_logistic_c": RIDGE_LOGISTIC_C, "iterations": int(head.n_iter_[0]),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(coefficient),
        },
        "target_rows_used_for_scaling_path_spec_tangent_scale_or_head": False,
        "target_labels_used": False,
        "class_covariance_or_qda_density": False,
        "path_signature_iterated_integral_or_levy_area": False,
        "haar_scattering_or_dtw_prototype": False,
        "nmf_ica_pls_or_archetype_factor": False,
        "graph_kernel_tree_or_neural": False,
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = log_euclidean_spd_direction(inner_train, inner_valid)
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
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 log-Euclidean increment-SPD blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "spd_ridge_tangent_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v115_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V115 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = log_euclidean_spd_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(baseline, challenger, selected["architecture"]["weight"])
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v115_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high)
            for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["spd_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v115_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(
            f"[V115 SPD] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V115 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V115 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V115 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V115 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V115 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V115 log-Euclidean increment-SPD policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{name: interval(items) for name, items in values.items()},
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int, smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V115 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    spd_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["static_spec"]["log_euclidean_tangent_dimension"] == TANGENT_DIMENSION
        and audit[side]["train_spd"]["minimum_eigenvalue"] >= SPD_RIDGE - 1e-10
        and audit[side]["target_spd"]["minimum_eigenvalue"] >= SPD_RIDGE - 1e-10
        and audit[side]["train_spd"]["matrix_exp_reconstruction_max_abs_error"] <= 1e-10
        and audit[side]["target_spd"]["matrix_exp_reconstruction_max_abs_error"] <= 1e-10
        and audit[side]["train_spd"]["isometric_frobenius_max_abs_error"] <= 1e-9
        and audit[side]["target_spd"]["isometric_frobenius_max_abs_error"] <= 1e-9
        and audit[side]["tangent_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_scaling_path_spec_tangent_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["class_covariance_or_qda_density"]
        and not audit[side]["path_signature_iterated_integral_or_levy_area"]
        and not audit[side]["haar_scattering_or_dtw_prototype"]
        and not audit[side]["nmf_ica_pls_or_archetype_factor"]
        and not audit[side]["graph_kernel_tree_or_neural"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "log_euclidean_increment_spd_contract_verified": spd_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V115 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap, "candidate_native_robustness_bootstrap": native_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V115 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V115 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V115 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V115", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V115_DIAGNOSTIC_LOG_EUCLIDEAN_INCREMENT_SPD": evaluation["candidate_summary"], "V115_FAIL_CLOSED_SELECTED": selected},
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V115", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V115", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V115_LOG_EUCLIDEAN_INCREMENT_SPD_REPORT.json": {
            "version": "V115", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "spd_ridge": SPD_RIDGE, "tangent_clip": TANGENT_CLIP,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V115", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V115_LOG_EUCLIDEAN_INCREMENT_SPD_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V115_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {"output": str(out), "required_controller_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.full_run and args.output is None, "explicit mode cannot accompany controller output")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V115 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_or_explicit_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES,
            "static_spec": STATIC_SPEC,
            "architecture": "per-event regularized increment second-moment SPD, exact principal matrix logarithm, isometric log-Euclidean tangent vector, fixed ridge head",
            "causal_numeric_only": True, "target_labels_used": False,
            "class_covariance_or_qda_density": False,
            "path_signature_iterated_integral_or_levy_area": False,
            "nmf_ica_pls_or_archetype_factor": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "source_transfer": True, "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if args.smoke_test else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": material_gate(evaluation), "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["inner_model_audit"] = audits[0]["inner_model_audit"]
        summary["outer_model_audit"] = audits[0]["outer_model_audit"]
        summary["candidate_native_robustness_bootstrap"] = evaluation["candidate_native_robustness_bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
