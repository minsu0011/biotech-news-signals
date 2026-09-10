"""V122 semantic-block Legendre polynomial-chaos direction challenger.

For each embargoed same-market past cut, only 37 causal numeric fields are
robustly scaled.  Eight fixed semantic-block medians are clipped onto [-1,1]
and expanded into the exact complete multivariate normalized-Legendre basis of
total degree one through three (164 deterministic terms).  This polynomial
chaos representation contains nonlinear block main effects and all quadratic
and cubic cross-block interactions without learned knots, factors, trees,
kernels, or neural features.  A train-only robust chaos scale and one fixed
duplicate/class-balanced ridge-logistic head produce direction probability.

Atomic V69 inner-past labels select exact V69 or a fixed .25/.50 logit blend.
Outer labels are evaluation-only behind a strict 35-minute embargo.  V69
confidence/high-confidence remain exact.  Any material failure restores the
exact entire V69 DataFrame.

Audit and US:2 smoke use CPU30-31, two threads, no GPU, and write nothing.  A
future authorized no-argument MARKET_BIO_VERSION_OUTPUT invocation, including
research/staging/V122, performs all six folds and writes the three required
reports/current material-gate/nested-bootstrap contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "2"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V122"
VERSION = 122
HYPOTHESIS = "SEMANTIC_BLOCK_LEGENDRE_POLYNOMIAL_CHAOS_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

BLOCKS = (
    ("RETURN", 0, 8), ("VOLATILITY", 8, 12), ("VOLUME", 12, 16),
    ("RANGE", 16, 20), ("CLOSE_POSITION", 20, 23),
    ("INTRADAY_STRUCTURE", 23, 30), ("CLOCK", 30, 32),
    ("BENCHMARK", 32, 37),
)
CHAOS_MAX_DEGREE = 3
CHAOS_VARIABLES = len(BLOCKS)
CHAOS_DIMENSION = 164
BLOCK_DOMAIN_SCALE = 3.0
CHAOS_SCALE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.20
SEED = 12201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "LEGENDRE_POLYNOMIAL_CHAOS_W0.25", "weight": 0.25},
    {"name": "LEGENDRE_POLYNOMIAL_CHAOS_W0.50", "weight": 0.50},
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


def total_degree_multi_indices(variable_n: int, maximum_degree: int) -> tuple[tuple[int, ...], ...]:
    output: list[tuple[int, ...]] = []

    def visit(position: int, remaining: int, current: list[int]) -> None:
        if position == variable_n - 1:
            current.append(remaining)
            output.append(tuple(current))
            current.pop()
            return
        for degree in range(remaining + 1):
            current.append(degree)
            visit(position + 1, remaining - degree, current)
            current.pop()

    for total in range(1, maximum_degree + 1):
        visit(0, total, [])
    return tuple(output)


MULTI_INDICES = total_degree_multi_indices(CHAOS_VARIABLES, CHAOS_MAX_DEGREE)
require(len(MULTI_INDICES) == CHAOS_DIMENSION, "V122 polynomial-chaos dimension changed")
MULTI_INDEX_SHA256 = hashlib.sha256(
    "\n".join(",".join(map(str, item)) for item in MULTI_INDICES).encode("utf-8")
).hexdigest()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust state, metrics, canonical gate, blending, and atomic IO only; no V119 output is read",
    }
    audit["v122_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
    }
    return audit


def semantic_block_state(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(matrix.ndim == 2 and matrix.shape[1] == 2 * len(FEATURES), "V122 robust-state dimension changed")
    numeric_state = matrix[:, :len(FEATURES)]
    block = np.column_stack([
        np.median(numeric_state[:, start:stop], axis=1) for _, start, stop in BLOCKS
    ])
    bounded = np.clip(block / BLOCK_DOMAIN_SCALE, -1.0, 1.0)
    require(bounded.shape[1] == CHAOS_VARIABLES and np.isfinite(bounded).all(), "V122 block state invalid")
    return bounded, {
        "rows": len(bounded), "block_n": bounded.shape[1],
        "blocks": [name for name, _, _ in BLOCKS],
        "domain": [-1.0, 1.0], "robust_domain_divisor": BLOCK_DOMAIN_SCALE,
        "boundary_fraction": float(np.mean(np.abs(bounded) >= 1.0)),
        "block_state_sha256": array_sha256(bounded),
    }


def normalized_legendre_values(block: np.ndarray) -> np.ndarray:
    value = np.empty((len(block), CHAOS_VARIABLES, CHAOS_MAX_DEGREE + 1), dtype=float)
    value[:, :, 0] = 1.0
    value[:, :, 1] = np.sqrt(3.0) * block
    value[:, :, 2] = np.sqrt(5.0) * 0.5 * (3.0 * block ** 2 - 1.0)
    value[:, :, 3] = np.sqrt(7.0) * 0.5 * (5.0 * block ** 3 - 3.0 * block)
    require(np.isfinite(value).all(), "V122 Legendre values invalid")
    return value


def polynomial_chaos_basis(block: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    legendre = normalized_legendre_values(block)
    design = np.ones((len(block), len(MULTI_INDICES)), dtype=float)
    for column, multi_index in enumerate(MULTI_INDICES):
        for variable, degree in enumerate(multi_index):
            if degree:
                design[:, column] *= legendre[:, variable, degree]
    require(design.shape[1] == CHAOS_DIMENSION and np.isfinite(design).all(), "V122 chaos basis invalid")
    degree_count = {
        str(degree): int(sum(sum(index) == degree for index in MULTI_INDICES))
        for degree in range(1, CHAOS_MAX_DEGREE + 1)
    }
    require(degree_count == {"1": 8, "2": 36, "3": 120}, "V122 chaos degree counts changed")
    return design, {
        "dimension": design.shape[1], "maximum_total_degree": CHAOS_MAX_DEGREE,
        "degree_counts": degree_count, "multi_index_sha256": MULTI_INDEX_SHA256,
        "normalization": "sqrt(2d+1) times standard Legendre P_d on [-1,1]",
        "complete_total_degree_basis": True,
        "design_sha256": array_sha256(design),
    }


class RobustChaosScale:
    def __init__(self) -> None:
        self.median = np.empty(0)
        self.scale = np.empty(0)

    def fit(self, matrix: np.ndarray) -> "RobustChaosScale":
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        standard = np.std(matrix, axis=0)
        self.scale = np.where(q75 - q25 > 1e-8, q75 - q25, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V122 chaos scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -CHAOS_SCALE_CLIP, CHAOS_SCALE_CLIP)
        require(np.isfinite(result).all(), "V122 scaled chaos design invalid")
        return result


def polynomial_chaos_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V122 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_block, train_block_audit = semantic_block_state(train_state)
    target_block, target_block_audit = semantic_block_state(target_state)
    train_chaos, train_chaos_audit = polynomial_chaos_basis(train_block)
    target_chaos, target_chaos_audit = polynomial_chaos_basis(target_block)
    scaler = RobustChaosScale().fit(train_chaos)
    train_design = scaler.transform(train_chaos)
    target_design = scaler.transform(target_chaos)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=600,
        random_state=SEED, n_jobs=1,
    )
    head.fit(train_design, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V122 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_block": train_block_audit, "target_block": target_block_audit,
        "train_chaos": train_chaos_audit, "target_chaos": target_chaos_audit,
        "chaos_scaling": {
            "train_only": True, "clip": CHAOS_SCALE_CLIP,
            "median_sha256": array_sha256(scaler.median),
            "scale_sha256": array_sha256(scaler.scale),
        },
        "head": {
            "type": "fixed duplicate/class-balanced L2 logistic",
            "C": RIDGE_LOGISTIC_C, "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_block_basis_scale_or_head": False,
        "target_labels_used": False,
        "source_shape_additive_ramps": False,
        "field_aware_low_rank_bilinear": False,
        "bernstein_partition_or_ulsif": False,
        "recurrence_plot_or_rqa": False,
        "hyperdimensional_binding_or_bundle": False,
        "path_dynamics_signature_spd_or_dmd": False,
        "tree_kernel_graph_or_neural": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = polynomial_chaos_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
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
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 semantic-block Legendre polynomial-chaos blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "block_basis_degree_head_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v122_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V122 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = polynomial_chaos_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(
            baseline, challenger, selected["architecture"]["weight"],
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v122_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]), confidence, high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["polynomial_chaos_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v122_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V122 LEGENDRE-CHAOS] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V122 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V122 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V122 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    result = scaffold.paired_nested_bootstrap(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V122 semantic-block Legendre polynomial-chaos policy"
    return result


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    compatibility_audits = copy.deepcopy(audits)
    for audit in compatibility_audits:
        for side in ("inner_model_audit", "outer_model_audit"):
            model = audit[side]
            model.update({
                "static_path_spec": {"channel_n": 8, "horizon_n": 8},
                "train_dmd": {"rank": scaffold.DMD_RANK, "feature_dimension": scaffold.DMD_FEATURE_DIMENSION, "chronological_adjacent_horizon_transition": True},
                "target_dmd": {"per_row_operator_only": True},
                "dmd_scaling": {"train_only": True},
                "target_rows_used_for_scaling_path_spec_dmd_scale_or_head": False,
                "rough_path_signature_or_levy_area": False,
                "increment_spd_or_matrix_log": False,
                "haar_modulus_or_scattering": False,
                "dtw_or_trajectory_prototype": False,
                "cross_event_dynamic_bayes_filter": False,
                "sinkhorn_or_optimal_transport": False,
                "bernstein_ulsif_density_ratio": False,
                "quantum_density_or_born_fidelity": False,
                "graph_kernel_tree_or_neural": False,
            })
    evaluation = scaffold.evaluate(champion, diagnostic, evidence, compatibility_audits, draws, smoke)
    evaluation["nested_bootstrap"]["method"] = "paired market-fold-time block bootstrap over inner-locked V122 semantic-block Legendre polynomial-chaos policy"
    checks = evaluation["material_checks"]
    checks.pop("per_row_ridge_koopman_dmd_spectral_contract_verified")
    chaos_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["train_block"]["block_n"] == CHAOS_VARIABLES
        and audit[side]["train_chaos"]["dimension"] == CHAOS_DIMENSION
        and audit[side]["train_chaos"]["maximum_total_degree"] == CHAOS_MAX_DEGREE
        and audit[side]["train_chaos"]["degree_counts"] == {"1": 8, "2": 36, "3": 120}
        and audit[side]["train_chaos"]["multi_index_sha256"] == MULTI_INDEX_SHA256
        and audit[side]["target_chaos"]["complete_total_degree_basis"]
        and audit[side]["chaos_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_block_basis_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["source_shape_additive_ramps"]
        and not audit[side]["field_aware_low_rank_bilinear"]
        and not audit[side]["bernstein_partition_or_ulsif"]
        and not audit[side]["recurrence_plot_or_rqa"]
        and not audit[side]["hyperdimensional_binding_or_bundle"]
        and not audit[side]["path_dynamics_signature_spd_or_dmd"]
        and not audit[side]["tree_kernel_graph_or_neural"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    checks["semantic_block_legendre_polynomial_chaos_contract_verified"] = chaos_contract
    material_pass = bool(not smoke and all(checks.values()))
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = evaluation["candidate_summary"] if material_pass else evaluation["baseline_summary"]
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "V122 selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V122 fallback is not exact entire V69")
    evaluation.update({
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "selected_summary": selected_summary,
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected_frame,
        "canonical_gate_audit": {
            **evaluation["canonical_gate_audit"], "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
    })
    return evaluation


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V122 nested bootstrap contract key mismatch",
    )
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V122 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V122 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V122", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V122_DIAGNOSTIC_LEGENDRE_POLYNOMIAL_CHAOS": evaluation["candidate_summary"],
                "V122_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V122", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V122", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V122_SEMANTIC_BLOCK_LEGENDRE_POLYNOMIAL_CHAOS_REPORT.json": {
            "version": "V122", "hypothesis": HYPOTHESIS,
            "blocks": [name for name, _, _ in BLOCKS],
            "maximum_total_degree": CHAOS_MAX_DEGREE,
            "chaos_dimension": CHAOS_DIMENSION,
            "multi_index_sha256": MULTI_INDEX_SHA256,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V122", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V122_LEGENDRE_POLYNOMIAL_CHAOS_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V122_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "output": str(out),
        "required_controller_reports": [
            "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V122 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V122 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V122 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "blocks": [name for name, _, _ in BLOCKS],
            "architecture": "eight robust semantic-block medians on [-1,1], exact normalized Legendre complete total-degree<=3 polynomial-chaos basis, train-only robust chaos scale, fixed ridge head",
            "maximum_total_degree": CHAOS_MAX_DEGREE,
            "chaos_dimension": CHAOS_DIMENSION,
            "degree_counts": {"1": 8, "2": 36, "3": 120},
            "multi_index_sha256": MULTI_INDEX_SHA256,
            "causal_numeric_only": True, "target_labels_used": False,
            "recurrence_plot_or_rqa": False,
            "hyperdimensional_binding_or_bundle": False,
            "bernstein_partition_or_ulsif": False,
            "field_aware_low_rank_bilinear": False,
            "source_shape_additive_ramps": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "exact_output_binding": True,
                "research_staging_v122_compatible": True,
                "explicit_conflict_fail_closed": True,
                "direct_full_without_env_fail_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness": True, "source_transfer": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if args.smoke_test else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
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
        "folds_executed": len(audits),
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
