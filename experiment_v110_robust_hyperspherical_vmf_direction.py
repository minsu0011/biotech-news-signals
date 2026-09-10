"""V110 robust hyperspherical von Mises--Fisher direction challenger.

Every embargoed same-market training cut robustly scales the 37 immutable
causal numeric fields and appends deterministic missingness flags.  A fixed
neutral coordinate is appended and each row is L2-normalized onto a unit
hypersphere.  For each direction class, event-group duplicate weights estimate
the von Mises--Fisher sufficient statistics: resultant direction, resultant
length, and one fixed analytic concentration approximation.  The normalized
class log-density contrast directly produces direction probability.

There is no discriminative head, covariance estimate, affine subspace,
archetype/simplex reconstruction, conformal p-value, neighbor retrieval,
feature graph, ICA, tree, kernel, or neural representation.  Earlier committed
V69 OOF labels alone choose exact V69 or fixed .25/.50 logit blends.  Outer
labels are evaluation-only under a strict 35-minute embargo.  V69 confidence
and high-confidence membership remain exact.  Any failed material gate returns
the exact entire V69 DataFrame.

Audit and US:2 two-thread smoke write nothing.  With an authorized
MARKET_BIO_VERSION_OUTPUT, no-argument invocation performs the six-fold full
run and writes all required controller reports and the current material gate.
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
from scipy.special import expit, gammaln, ive
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v108_duplicate_balanced_separable_archetype_simplex as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V110_ROBUST_HYPERSPHERICAL_VMF_V1"
VERSION = 110
HYPOTHESIS = "ROBUST_HYPERSPHERICAL_VMF_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v108_duplicate_balanced_separable_archetype_simplex.py"
EXPECTED_SCAFFOLD_SHA256 = "d993c325a393cb477aeca5ff90cdc6cf0da2ef9cd91e33a108376aa2ff93b18d"

NEUTRAL_COORDINATE = 1.0
NORM_FLOOR = 1e-12
KAPPA_FLOOR = 1e-4
KAPPA_CAP = 200.0
LOGIT_TEMPERATURE_PER_DIMENSION = 1.0
BLEND_WEIGHTS = (0.25, 0.50)
SEED = 11001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "HYPERSPHERICAL_VMF_W0.25", "weight": 0.25},
    {"name": "HYPERSPHERICAL_VMF_W0.50", "weight": 0.50},
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


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V108 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, robust-state, metrics, canonical gate, blending, and atomic IO only; no V108 output is read",
    }
    return audit


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    group_size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(group_size, 1.0)
    require(np.isfinite(weight).all() and np.all(weight > 0.0), "V110 duplicate weights invalid")
    return weight


def unit_sphere(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    augmented = np.concatenate([
        np.asarray(matrix, dtype=float),
        np.full((len(matrix), 1), NEUTRAL_COORDINATE, dtype=float),
    ], axis=1)
    norms = np.linalg.norm(augmented, axis=1)
    require(np.isfinite(norms).all() and float(norms.min()) > NORM_FLOOR, "V110 sphere norm invalid")
    sphere = augmented / norms[:, None]
    norm_error = float(np.max(np.abs(np.linalg.norm(sphere, axis=1) - 1.0)))
    require(np.isfinite(sphere).all() and norm_error <= 1e-10, "V110 unit-sphere projection failed")
    return sphere, {
        "dimension": sphere.shape[1],
        "neutral_coordinate": NEUTRAL_COORDINATE,
        "input_norm_min": float(norms.min()),
        "input_norm_median": float(np.median(norms)),
        "input_norm_max": float(norms.max()),
        "unit_norm_max_abs_error": norm_error,
        "sphere_sha256": array_sha256(sphere),
    }


def vmf_log_normalizer(dimension: int, kappa: float) -> float:
    order = 0.5 * dimension - 1.0
    if kappa <= KAPPA_FLOOR:
        return float(gammaln(0.5 * dimension) - np.log(2.0) - 0.5 * dimension * np.log(np.pi))
    scaled_bessel = float(ive(order, kappa))
    require(np.isfinite(scaled_bessel) and scaled_bessel > 0.0, "V110 scaled Bessel value invalid")
    log_bessel = np.log(scaled_bessel) + abs(kappa)
    result = order * np.log(kappa) - 0.5 * dimension * np.log(2.0 * np.pi) - log_bessel
    require(np.isfinite(result), "V110 vMF log normalizer invalid")
    return float(result)


def fit_vmf_class(sphere: np.ndarray, weight: np.ndarray, label: int) -> dict[str, Any]:
    normalized_weight = weight / float(weight.sum())
    resultant = np.sum(sphere * normalized_weight[:, None], axis=0)
    resultant_length = float(np.linalg.norm(resultant))
    require(np.isfinite(resultant_length) and 0.0 <= resultant_length <= 1.0 + 1e-10, "V110 resultant invalid")
    if resultant_length <= NORM_FLOOR:
        mean_direction = np.zeros(sphere.shape[1], dtype=float)
        mean_direction[-1] = 1.0
    else:
        mean_direction = resultant / resultant_length
    clipped_resultant = min(max(resultant_length, 0.0), 1.0 - 1e-8)
    dimension = sphere.shape[1]
    raw_kappa = clipped_resultant * (dimension - clipped_resultant ** 2) / max(1.0 - clipped_resultant ** 2, 1e-8)
    kappa = float(np.clip(raw_kappa, KAPPA_FLOOR, KAPPA_CAP))
    log_normalizer = vmf_log_normalizer(dimension, kappa)
    return {
        "label": label,
        "mean_direction": mean_direction,
        "kappa": kappa,
        "log_normalizer": log_normalizer,
        "effective_duplicate_weight_n": float(weight.sum()),
        "audit": {
            "label": label,
            "row_n": len(sphere),
            "dimension": dimension,
            "resultant_length": resultant_length,
            "raw_kappa": float(raw_kappa),
            "kappa": kappa,
            "kappa_was_capped": bool(raw_kappa > KAPPA_CAP),
            "log_normalizer": log_normalizer,
            "mean_direction_norm": float(np.linalg.norm(mean_direction)),
            "mean_direction_sha256": array_sha256(mean_direction),
            "effective_duplicate_weight_n": float(weight.sum()),
        },
    }


def vmf_log_density(sphere: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    value = model["log_normalizer"] + model["kappa"] * (sphere @ model["mean_direction"])
    require(np.isfinite(value).all(), "V110 vMF log density invalid")
    return value


def hyperspherical_vmf_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V110 expects one market per fit")
    processor = scaffold.scaffold.scaffold.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_sphere, train_sphere_audit = unit_sphere(train_state)
    target_sphere, target_sphere_audit = unit_sphere(target_state)
    target = ordered.y.to_numpy(int)
    duplicate_weight = duplicate_unit_weights(ordered)
    models: dict[int, dict[str, Any]] = {}
    class_weight: dict[int, float] = {}
    for label in (0, 1):
        index = target == label
        require(int(index.sum()) >= 100, f"V110 class {label} support too small")
        class_weight[label] = float(duplicate_weight[index].sum())
        models[label] = fit_vmf_class(train_sphere[index], duplicate_weight[index], label)
    total_weight = class_weight[0] + class_weight[1]
    log_down = vmf_log_density(target_sphere, models[0]) + np.log((class_weight[0] + 0.5) / (total_weight + 1.0))
    log_up = vmf_log_density(target_sphere, models[1]) + np.log((class_weight[1] + 0.5) / (total_weight + 1.0))
    dimension = target_sphere.shape[1]
    contrast = np.clip((log_up - log_down) / (dimension * LOGIT_TEMPERATURE_PER_DIMENSION), -30.0, 30.0)
    probability = np.clip(expit(contrast), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V110 direction probability invalid")
    return probability, {
        "market": str(ordered.market.iloc[0]),
        "train_n": len(ordered), "target_n": len(target_frame),
        "raw_numeric_feature_n": len(FEATURES),
        "robust_state_dimension": train_state.shape[1],
        "sphere_dimension": dimension,
        "train_transform": train_transform, "target_transform": target_transform,
        "train_sphere": train_sphere_audit, "target_sphere": target_sphere_audit,
        "classes": {str(label): models[label]["audit"] for label in (0, 1)},
        "class_effective_duplicate_weights": {str(label): class_weight[label] for label in (0, 1)},
        "logit_temperature": dimension * LOGIT_TEMPERATURE_PER_DIMENSION,
        "contrast_quantiles": np.quantile(contrast, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "probability_sha256": array_sha256(probability),
        "causal_numeric_only": True,
        "target_rows_used_for_scaling_or_density_fit": False,
        "target_labels_used": False,
        "discriminative_head": False,
        "covariance_or_subspace": False,
        "archetype_or_simplex_reconstruction": False,
        "conformal_nonconformity_or_pvalue": False,
        "feature_graph_or_persistence": False,
        "ica_or_negentropy": False,
        "nearest_neighbor_or_retrieval": False,
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = hyperspherical_vmf_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 vMF logit blend; fixed 2*AUC+BA score with BA/net eligibility",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "density_scale_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v110_model"] = "V69_NOOP"
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
        outer_chronology = chronology(outer_train, outer_valid, f"V110 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = hyperspherical_vmf_direction(outer_train, outer_valid)
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
        diagnostic.loc[positions, "v110_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["vmf_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v110_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V110 VMF] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V110 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V110 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V110 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V110 nested-bootstrap blocks")
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
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V110 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V110 hyperspherical-vMF policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V110 canonical mismatch")
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
    model_contract = all(
        audit[side]["causal_numeric_only"]
        and not audit[side]["target_rows_used_for_scaling_or_density_fit"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["discriminative_head"]
        and not audit[side]["covariance_or_subspace"]
        and not audit[side]["archetype_or_simplex_reconstruction"]
        and not audit[side]["conformal_nonconformity_or_pvalue"]
        and not audit[side]["feature_graph_or_persistence"]
        and not audit[side]["ica_or_negentropy"]
        and not audit[side]["nearest_neighbor_or_retrieval"]
        and audit[side]["train_sphere"]["unit_norm_max_abs_error"] <= 1e-10
        and audit[side]["target_sphere"]["unit_norm_max_abs_error"] <= 1e-10
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
        "robust_hyperspherical_vmf_contract_verified": model_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V110 fallback is not exact entire V69")
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
    return {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V110", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V110_DIAGNOSTIC_HYPERSPHERICAL_VMF": evaluation["candidate_summary"],
                "V110_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V110", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V110", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V110_HYPERSPHERICAL_VMF_REPORT.json": {
            "version": "V110", "hypothesis": HYPOTHESIS,
            "features": FEATURES, "neutral_coordinate": NEUTRAL_COORDINATE,
            "kappa_floor": KAPPA_FLOOR, "kappa_cap": KAPPA_CAP,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V110", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V110_HYPERSPHERICAL_VMF_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V110_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.smoke_test and not args.full_run and args.output is None, "explicit mode cannot accompany controller output")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()
    runtime = scaffold.scaffold.scaffold.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
        "mode": "controller_or_explicit_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES,
            "architecture": "robust numeric+missing state, fixed neutral coordinate/unit sphere, duplicate-balanced class vMF sufficient statistics and direct normalized log-density contrast",
            "fixed_kappa_approximation": "Rbar*(p-Rbar^2)/(1-Rbar^2)",
            "causal_numeric_only": True, "target_labels_used": False,
            "discriminative_head": False, "covariance_or_subspace": False,
            "archetype_or_simplex_reconstruction": False,
            "conformal_nonconformity_or_pvalue": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "source_transfer": True, "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
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
