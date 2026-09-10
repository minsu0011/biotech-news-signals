"""V147 duplicate-balanced sliced-Wasserstein class-barycenter direction.

For every embargoed same-market train cut, the immutable 37 causal numeric
fields plus deterministic missing flags are robust-scaled using train rows only
and collapsed to equal event-group centroids.  A fixed label-independent set of
32 orthonormal spherical directions projects the 74D state.  On a fixed
19-point quantile grid, each direction class is represented by its train-only
empirical projected quantile function.  A query Dirac projection's averaged
regularized 1D W2 distance to the DOWN and UP quantile functions yields a
nonlinear sliced-distribution distance contrast and direct equal-prior
direction probability.  There is no fitted discriminative coefficient head.

This is not V116 within-row semantic-mass Sinkhorn transport, V145 source/time
marginal covariate transport, energy distance, Tyler scatter, nearest retrieval,
copula density, projection depth, or outer-label tuning.  Inner-past evidence
selects exact V69 or fixed .25/.50 blends under fixed BA/net/source safety;
outer labels remain evaluation-only under a strict 35-minute embargo and event
purge.  V69 confidence/high-confidence remain exact and material failure
restores the exact entire atomic V69 DataFrame.

Audit, US:2 smoke and KR:2 support use CPU30-31/two threads/no GPU and write
nothing. Full execution is accepted only through no-argument
MARKET_BIO_VERSION_OUTPUT binding.
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
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v144_multichannel_weak_ordinal_motif_transition_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V147_SLICED_WASSERSTEIN_CLASS_BARYCENTER_V1"
VERSION = 147
HYPOTHESIS = "DUPLICATE_BALANCED_SLICED_WASSERSTEIN_CLASS_BARYCENTER_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v144_multichannel_weak_ordinal_motif_transition_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "19817afa20b5243fea7831fd1d26a17cc2159085c5a774a2d9524a82d0f4064c"

STATE_DIMENSION = 74
SLICE_N = 32
QUANTILE_GRID = np.linspace(0.05, 0.95, 19)
PROJECTED_SCALE_FLOOR = 0.25
CONTRAST_TEMPERATURE_FLOOR = 0.05
SEED = 14701
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "SLICED_WASSERSTEIN_BARYCENTER_W0.25", "weight": 0.25},
    {"name": "SLICED_WASSERSTEIN_BARYCENTER_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44
numeric = scaffold.numeric


def fixed_spherical_directions() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    raw = generator.standard_normal((STATE_DIMENSION, SLICE_N))
    orthogonal, triangular = np.linalg.qr(raw, mode="reduced")
    sign = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    directions = (orthogonal * sign[None, :]).T
    require(
        directions.shape == (SLICE_N, STATE_DIMENSION)
        and np.max(np.abs(directions @ directions.T - np.eye(SLICE_N))) <= 1e-12,
        "V147 fixed directions lost orthonormality",
    )
    return directions


DIRECTIONS = fixed_spherical_directions()
STATIC_SPEC = {
    "state_dimension": STATE_DIMENSION,
    "state": "37 train-cut robust causal numeric plus 37 deterministic missing flags",
    "equal_event_group_centroids": True,
    "slice_n": SLICE_N,
    "directions": "fixed seed-14701 label-independent Gaussian QR with deterministic column sign",
    "directions_sha256": array_sha256(DIRECTIONS),
    "direction_orthonormal_max_abs_error": float(np.max(np.abs(DIRECTIONS @ DIRECTIONS.T - np.eye(SLICE_N)))),
    "quantile_grid": QUANTILE_GRID.tolist(),
    "quantile_n": len(QUANTILE_GRID),
    "projected_scale_floor": PROJECTED_SCALE_FLOOR,
    "contrast_temperature_floor": CONTRAST_TEMPERATURE_FLOOR,
    "distance": "mean over slices and quantiles of squared regularized query-Dirac minus class projected quantile function",
    "class_prior": "equal",
    "fitted_discriminative_head": False,
    "learned_projection_quantile_grid_or_regularization": False,
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V144 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, metrics, canonical gate, blending and atomic IO only; no V144 output is read",
    }
    audit["v147_access_contract"] = {
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
    frame: pd.DataFrame,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(len(frame) == len(state) and state.shape[1] == STATE_DIMENSION, "V147 group-state alignment failed")
    codes, group_id = pd.factorize(frame.event_group_id, sort=True)
    group_n = len(group_id)
    count = np.bincount(codes, minlength=group_n).astype(float)
    grouped = np.zeros((group_n, STATE_DIMENSION), dtype=float)
    np.add.at(grouped, codes, state)
    grouped /= count[:, None]
    target_sum = np.bincount(codes, weights=frame.y.to_numpy(float), minlength=group_n)
    target = np.rint(target_sum / count).astype(int)
    require(
        np.allclose(target_sum, target * count) and np.unique(target).size == 2
        and np.isfinite(grouped).all(),
        "V147 event-group label or centroid contract failed",
    )
    return grouped, target, {
        "row_n": len(frame),
        "event_group_n": group_n,
        "duplicate_rows_removed": len(frame) - group_n,
        "group_size_min": int(count.min()),
        "group_size_max": int(count.max()),
        "class_group_n": np.bincount(target, minlength=2).tolist(),
        "centroid_sha256": array_sha256(grouped),
        "equal_group_weight": True,
    }


def sliced_wasserstein_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V147 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    grouped_state, grouped_target, group_audit = equal_event_group_state(ordered, train_state)
    group_projection = grouped_state @ DIRECTIONS.T
    target_projection = target_state @ DIRECTIONS.T
    pooled_q25, pooled_q75 = np.quantile(group_projection, [0.25, 0.75], axis=0)
    projected_scale = np.maximum((pooled_q75 - pooled_q25) / 1.349, PROJECTED_SCALE_FLOOR)
    class_quantiles = np.stack([
        np.quantile(group_projection[grouped_target == direction], QUANTILE_GRID, axis=0).T
        for direction in (0, 1)
    ], axis=0)
    require(
        class_quantiles.shape == (2, SLICE_N, len(QUANTILE_GRID))
        and np.isfinite(class_quantiles).all() and np.isfinite(projected_scale).all(),
        "V147 class quantile template invalid",
    )

    def distances(projection: np.ndarray) -> np.ndarray:
        scaled = (
            projection[:, None, :, None] - class_quantiles[None, :, :, :]
        ) / projected_scale[None, None, :, None]
        return np.mean(np.square(scaled), axis=(2, 3))

    group_distance = distances(group_projection)
    target_distance = distances(target_projection)
    group_contrast = group_distance[:, 0] - group_distance[:, 1]
    target_contrast = target_distance[:, 0] - target_distance[:, 1]
    q25, q75 = np.quantile(group_contrast, [0.25, 0.75])
    temperature = max(float((q75 - q25) / 1.349), CONTRAST_TEMPERATURE_FLOOR)
    score = np.clip(target_contrast / temperature, -30.0, 30.0)
    probability = np.clip(1.0 / (1.0 + np.exp(-score)), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V147 probability invalid")
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "event_group_training": group_audit,
        "sliced_wasserstein": {
            "static_spec": STATIC_SPEC,
            "group_projection_sha256": array_sha256(group_projection),
            "target_projection_sha256": array_sha256(target_projection),
            "projected_scale_sha256": array_sha256(projected_scale),
            "class_quantile_sha256": array_sha256(class_quantiles),
            "class_quantile_monotone": bool(np.all(np.diff(class_quantiles, axis=2) >= -1e-12)),
            "group_distance_sha256": array_sha256(group_distance),
            "target_distance_sha256": array_sha256(target_distance),
            "group_contrast_quantiles": {
                str(q): float(np.quantile(group_contrast, q)) for q in (0.1, 0.5, 0.9)
            },
            "target_contrast_quantiles": {
                str(q): float(np.quantile(target_contrast, q)) for q in (0.1, 0.5, 0.9)
            },
            "temperature": temperature,
            "target_rows_used_for_projection_scale_class_quantiles_or_temperature": False,
            "target_labels_used": False,
            "label_independent_fixed_directions": True,
            "fixed_quantile_grid_and_regularization": True,
            "fitted_discriminative_head": False,
        },
        "semantic_mass_sinkhorn_transport": False,
        "source_time_marginal_quantile_transport": False,
        "pairwise_sample_energy_distance": False,
        "tyler_scatter_or_fisher_direction": False,
        "nearest_sample_retrieval": False,
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
        base_ba = balanced_accuracy_score(target, baseline[index] >= 0.5)
        current_ba = balanced_accuracy_score(target, candidate[index] >= 0.5)
        deltas[str(source)] = float(current_ba - base_ba)
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = sliced_wasserstein_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
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
            "supported_source_ba_delta": source_deltas,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
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
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 sliced-Wasserstein class-quantile blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "projection_slice_quantile_regularization_temperature_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    original_architectures = scaffold.ARCHITECTURES
    original_choose = scaffold.choose_inner
    original_direction = scaffold.ordinal_direction
    scaffold.ARCHITECTURES = ARCHITECTURES
    scaffold.choose_inner = choose_inner
    scaffold.ordinal_direction = sliced_wasserstein_direction
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            diagnostic, evidence, audits = scaffold.run_nested(dev, champion, smoke, smoke_market)
    finally:
        scaffold.ARCHITECTURES = original_architectures
        scaffold.choose_inner = original_choose
        scaffold.ordinal_direction = original_direction
    diagnostic = diagnostic.rename(columns={"v144_model": "v147_model"})
    evidence = evidence.rename(columns={"ordinal_prob": "sliced_wasserstein_prob", "v144_model": "v147_model"})
    require("v147_model" in diagnostic and "sliced_wasserstein_prob" in evidence, "V147 generic nested-loop rename failed")
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V147 SLICED_WASSERSTEIN] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V147 nested-bootstrap blocks")
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
        values["balanced_accuracy_delta"].append(float(
            balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V147 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V147 sliced-Wasserstein class-barycenter policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{name: interval(items) for name, items in values.items()},
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V147 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    wasserstein_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["event_group_training"]["equal_group_weight"]
        and audit[side]["sliced_wasserstein"]["static_spec"]["slice_n"] == SLICE_N
        and audit[side]["sliced_wasserstein"]["static_spec"]["quantile_n"] == len(QUANTILE_GRID)
        and audit[side]["sliced_wasserstein"]["class_quantile_monotone"]
        and audit[side]["sliced_wasserstein"]["label_independent_fixed_directions"]
        and audit[side]["sliced_wasserstein"]["fixed_quantile_grid_and_regularization"]
        and not audit[side]["sliced_wasserstein"]["target_rows_used_for_projection_scale_class_quantiles_or_temperature"]
        and not audit[side]["sliced_wasserstein"]["target_labels_used"]
        and not audit[side]["sliced_wasserstein"]["fitted_discriminative_head"]
        and not audit[side]["semantic_mass_sinkhorn_transport"]
        and not audit[side]["source_time_marginal_quantile_transport"]
        and not audit[side]["pairwise_sample_energy_distance"]
        and not audit[side]["tyler_scatter_or_fisher_direction"]
        and not audit[side]["nearest_sample_retrieval"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
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
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "duplicate_balanced_sliced_wasserstein_class_barycenter_contract_verified": wasserstein_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V147 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "candidate_native_robustness_bootstrap": native_bootstrap,
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
        "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V147 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V147 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V147 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V147", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V147_DIAGNOSTIC_SLICED_WASSERSTEIN_CLASS_BARYCENTER": evaluation["candidate_summary"],
                "V147_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V147", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V147", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V147_SLICED_WASSERSTEIN_CLASS_BARYCENTER_REPORT.json": {
            "version": "V147", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V147", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V147_SLICED_WASSERSTEIN_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V147_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "output": str(out),
        "required_controller_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
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
            "explicit mode/output conflicts with controller-bound no-argument V147 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V147 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V147 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test or args.support_probe) else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "fixed 32 orthonormal spherical slices and 19 projected class quantiles; direct regularized 1D W2 distance contrast without a fitted head",
            "causal_numeric_only": True, "target_row_labels_used": False,
            "learned_projection_quantile_grid_regularization_or_post_outer_tuning": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v147_compatible": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    bounded = bool(args.smoke_test or args.support_probe)
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V147 support probe is reserved for KR:2")
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
                "nested_bootstrap_reason": "KR:2 support validates model support only; standard material bootstrap remains US smoke/full evaluation",
                "exact_entire_v69_fallback_contract": True, "output_written": False,
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
