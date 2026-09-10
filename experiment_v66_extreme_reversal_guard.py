"""V66: sparse nonlinear extreme-tail confidence guard for US_NEWS.

Material difference from V65: this runner has no P(correct) meta-model and no
continuous all-row contribution inversion.  It starts from the frozen V58
confidence itself and modifies it only when causal categorical-context or
volatility-range late-expert contribution enters an inner-past-defined extreme
tail.  Penalty, clipping, and saturation candidates never alter a raw feature,
feature sign, direction probability, or direction prediction.

Selection and high-confidence thresholds are nested chronological inner-only.
Failure of any predeclared material gate returns the exact V58 confidence and
mask as a no-op.  Seal, final reserve, DEV extension, Massive, KIS, and Naver
inputs are outside this runner's data graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import experiment_v65_us_news_confidence_reversal as v65
import runtime_limits


ROOT = Path(__file__).resolve().parent
V65_EVIDENCE = ROOT / "staging" / "V65_US_NEWS_CONFIDENCE_REVERSAL_GUARD_V1"
OUT = Path(os.environ.get(
    "MARKET_BIO_VERSION_OUTPUT",
    str(ROOT / "staging" / "V66_EXTREME_REVERSAL_GUARD_V1"),
))
HYPOTHESIS = "EXTREME_REVERSAL_GUARD_V1"
FAMILY = "EXTREME_CONFIDENCE_REVERSAL"
EXPECTED_V65_EVIDENCE_SHA256 = {
    "CONFIDENCE_COMPARISON.csv": "0da001dbfcfbb6840da38addff8886b52decc23de71c3cb50f55bb54de1953b8",
    "CONFIDENCE_META_MODEL_REPORT.json": "e547d7a1c8235172a172cc20d5b27c96b005e75627f890b49f97e571f2949e26",
    "DEV_ROBUSTNESS_REPORT.json": "14f9dce58506283844566a497f8a3c3215c34e0d6fdedbfd6b3e3975fe6026d0",
    "MODEL_COMPARISON.json": "cf033380eb0346fd826ffcc0181c806dc4e312ed6865a2247e62d0b99553a27d",
    "RISK_COVERAGE_REPORT.csv": "a9a8e2111fad82eeaca694a2603a7d09d0516a5856fa6de4a6bab1118239d6fb",
    "SOURCE_TRANSFER_REPORT.json": "a094aaad36b16b70fe6c722f7291c86d57b0f08728039e879afd7dfb7b9ace9b",
    "US_NEWS_CONFIDENCE_META_OOF.csv.gz": "4550a7c80d76b6bf40b959455b526e51ec375b426985ab0680be8da7ada1a246",
    "US_NEWS_CONFIDENCE_REVERSAL_REPORT.json": "808ee721493d91d04aa6dfc38c4f8920a839834735bb3d5f754c92382c1d52ae",
}
TAIL_QUANTILES = (0.80, 0.90)
HC_QUANTILES = (0.70, 0.80, 0.85, 0.90)
SEED = 6601


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def verify_prior_evidence() -> dict[str, Any]:
    base = v65.verify_inputs_and_evidence()
    hashes = {}
    for relative, expected in EXPECTED_V65_EVIDENCE_SHA256.items():
        path = V65_EVIDENCE / relative
        require(path.is_file(), f"required V65 evidence missing: {relative}")
        actual = sha256(path)
        require(actual == expected, f"pinned V65 evidence changed: {relative}")
        hashes[relative] = actual
    comparison = json.loads(
        (V65_EVIDENCE / "MODEL_COMPARISON.json").read_text(encoding="utf-8")
    )
    require(comparison.get("hypothesis") == "US_NEWS_CONFIDENCE_REVERSAL_GUARD_V1",
            "V65 evidence hypothesis mismatch")
    require(comparison.get("status") == "MATERIAL_EXPERIMENT_FAIL_NOOP",
            "V65 is no longer the pinned failed no-op evidence")
    require(comparison.get("direction_probability_changed") is False,
            "V65 direction immutability evidence changed")
    checks = comparison["us_news_confidence_evaluation"]["material_checks"]
    require(checks["confidence_auc_delta_ge_0_010"] is False,
            "V65 AUC failure evidence changed")
    require(checks["highconf_accuracy_delta_ge_0_010"] is False,
            "V65 HC accuracy failure evidence changed")
    require(checks["highconf_net_not_worse"] is False,
            "V65 HC net failure evidence changed")
    return {
        **base,
        "v65_evidence_sha256": hashes,
        "v65_material_status": "MATERIAL_EXPERIMENT_FAIL_NOOP",
        "v65_failed_material_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "v66_material_difference": (
            "V58-confidence sparse extreme-tail nonlinear guard; no P(correct) meta base "
            "and no continuous all-row contribution inverse"
        ),
        "massive_accessed": False,
        "kis_accessed": False,
        "naver_accessed": False,
        "raw_feature_sign_flip_performed": False,
    }


def candidate_specs() -> list[dict[str, Any]]:
    specs = [{
        "name": "V58_FROZEN_CONFIDENCE_NOOP", "mode": "noop",
        "tail": "none", "tail_quantile": None,
    }]
    for tail in ("positive", "absolute"):
        for quantile in TAIL_QUANTILES:
            for penalty in (0.10, 0.20):
                specs.append({
                    "name": f"{tail.upper()}_Q{int(100*quantile)}_PENALTY_{int(100*penalty):02d}",
                    "mode": "penalty", "tail": tail,
                    "tail_quantile": quantile, "penalty": penalty,
                })
            for cap_quantile in (0.60, 0.75):
                specs.append({
                    "name": f"{tail.upper()}_Q{int(100*quantile)}_CLIP_C{int(100*cap_quantile)}",
                    "mode": "clip", "tail": tail,
                    "tail_quantile": quantile, "cap_quantile": cap_quantile,
                })
            for saturation in (0.25, 0.50):
                specs.append({
                    "name": f"{tail.upper()}_Q{int(100*quantile)}_SAT_{int(100*saturation)}",
                    "mode": "saturation", "tail": tail,
                    "tail_quantile": quantile, "cap_quantile": 0.60,
                    "saturation": saturation,
                })
    return specs


def contribution_ranks(
    reference: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    tail: str,
) -> np.ndarray:
    rows = []
    for block in v65.NEGATIVE_BLOCKS:
        if tail == "positive":
            ref = reference[block]
            target = values[block]
        elif tail == "absolute":
            ref = np.abs(reference[block])
            target = np.abs(values[block])
        else:
            raise RuntimeError(f"unsupported tail: {tail}")
        rows.append(v37.rank_against(ref, target))
    return np.max(np.vstack(rows), axis=0)


def transform_confidence(
    spec: dict[str, Any],
    reference_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    reference_contributions: dict[str, np.ndarray],
    target_contributions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = target_frame.confidence_signal.to_numpy(float).copy()
    if spec["mode"] == "noop":
        return confidence, np.zeros(len(target_frame), dtype=bool), np.zeros(len(target_frame), dtype=float)
    ranks = contribution_ranks(
        reference_contributions, target_contributions, str(spec["tail"]),
    )
    quantile = float(spec["tail_quantile"])
    active = ranks >= quantile
    strength = np.clip((ranks - quantile) / max(1e-12, 1.0 - quantile), 0.0, 1.0)
    if spec["mode"] == "penalty":
        confidence[active] = np.clip(
            confidence[active] - float(spec["penalty"]) * strength[active], 0.0, 1.0,
        )
    else:
        cap = float(np.quantile(
            reference_frame.confidence_signal.to_numpy(float),
            float(spec["cap_quantile"]),
        ))
        above = active & (confidence > cap)
        if spec["mode"] == "clip":
            confidence[above] = cap
        elif spec["mode"] == "saturation":
            confidence[above] = cap + (
                confidence[above] - cap
            ) * float(spec["saturation"])
        else:
            raise RuntimeError(f"unsupported guard mode: {spec['mode']}")
    return confidence, active, strength


def inner_policy(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    correctness = frame.correct_target.to_numpy(int)
    auc = v65.safe_auc(correctness, score)
    ranks = v37.rank_self(score)
    base_accuracy = float(correctness.mean())
    thresholds = []
    for quantile in HC_QUANTILES:
        high = ranks >= quantile
        accuracy = float(correctness[high].mean())
        lift = accuracy - base_accuracy
        selection_score = float(auc or 0.5) + 0.50 * accuracy + 0.25 * lift
        thresholds.append({
            "quantile": quantile, "highconf_n": int(high.sum()),
            "highconf_coverage": float(high.mean()), "highconf_accuracy": accuracy,
            "highconf_correctness_lift": lift, "selection_score": selection_score,
        })
    thresholds.sort(key=lambda row: (
        row["selection_score"], -abs(row["quantile"] - 0.80),
    ), reverse=True)
    return {
        "confidence_correctness_auc": auc,
        "base_correctness_rate": base_accuracy,
        "selected_threshold": thresholds[0],
        "threshold_candidates": thresholds,
    }


def run_nested_oof(frame: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    specs = candidate_specs()
    outputs = []
    audits = []
    for train, valid, fold in v65.chronological_outer_folds(frame, smoke=smoke):
        inner_train, inner_valid = v65.inner_past_split(train)
        inner_time = v65.time_audit(inner_train, inner_valid, f"V66 inner fold {fold}")
        outer_time = v65.time_audit(train, valid, f"V66 outer fold {fold}")

        # Inner-valid contributions are predictions of inner-train experts, so
        # their empirical distribution is a label-safe inner-only tail reference.
        inner_valid_contribution = v65.late_block_contributions(
            inner_train, inner_valid, SEED + 100 + fold,
        )
        inner_reference_contribution = inner_valid_contribution
        reference_frame = inner_valid
        policies = {}
        inner_scores = {}
        inner_activation = {}
        inner_strength = {}
        for spec in specs:
            score, active, strength = transform_confidence(
                spec, reference_frame, inner_valid,
                inner_reference_contribution, inner_valid_contribution,
            )
            inner_scores[spec["name"]] = score
            inner_activation[spec["name"]] = active
            inner_strength[spec["name"]] = strength
            policy = inner_policy(inner_valid, score)
            policy["guard_spec"] = spec
            policy["inner_activation_coverage"] = float(active.mean())
            policies[spec["name"]] = policy
        selected = sorted(
            policies,
            key=lambda name: (
                policies[name]["selected_threshold"]["selection_score"],
                policies[name]["guard_spec"]["mode"] == "noop",
            ),
            reverse=True,
        )[0]

        outer_valid_contribution = v65.late_block_contributions(
            train, valid, SEED + 1100 + fold,
        )
        output = valid[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "y", "fwd_ret_30m", "prob", "correct_target",
            "confidence_signal", "v58_high_conf", "cluster_weight",
        ]].copy()
        output["guard_outer_fold"] = int(fold)
        output["selected_candidate"] = selected
        output["selected_mode"] = policies[selected]["guard_spec"]["mode"]
        for spec in specs:
            name = spec["name"]
            outer_score, outer_active, outer_strength = transform_confidence(
                spec, inner_valid, valid,
                inner_valid_contribution, outer_valid_contribution,
            )
            normalized = v37.rank_against(inner_scores[name], outer_score)
            quantile = float(policies[name]["selected_threshold"]["quantile"])
            output[f"confidence__{name}"] = normalized
            output[f"high__{name}"] = normalized >= quantile
            if name == selected:
                output["selected_confidence"] = normalized
                output["selected_high_conf"] = normalized >= quantile
                output["selected_guard_active"] = outer_active
                output["selected_guard_strength"] = outer_strength
        for block in v65.NEGATIVE_BLOCKS:
            output[f"late_contribution__{block}"] = outer_valid_contribution[block]
        outputs.append(output)
        audits.append({
            "fold": int(fold), "outer": outer_time, "inner": inner_time,
            "tail_reference": {
                "rows": len(inner_valid),
                "source": "held-out inner validation late contributions",
                "labels_used_for_tail_threshold": False,
            },
            "selected_candidate": selected,
            "selected_mode": policies[selected]["guard_spec"]["mode"],
            "candidate_policies": policies,
            "direction_probability_changed": False,
            "raw_feature_sign_flip_performed": False,
        })
        print(
            f"[V66 EXTREME] fold={fold} train={len(train)} valid={len(valid)} "
            f"selected={selected} active={output.selected_guard_active.mean():.3f}",
            flush=True,
        )
    oof = pd.concat(outputs, ignore_index=True)
    original = frame.set_index("event_id").loc[oof.event_id]
    require(np.array_equal(oof.prob.to_numpy(float), original.prob.to_numpy(float)),
            "V66 changed direction probabilities")
    require(np.array_equal(oof.prob.ge(0.5).to_numpy(), original.prob.ge(0.5).to_numpy()),
            "V66 changed direction predictions")
    return oof, audits


def evaluate(oof: pd.DataFrame, audits: list[dict[str, Any]]) -> dict[str, Any]:
    specs = candidate_specs()
    candidate_metrics = {}
    for spec in specs:
        name = spec["name"]
        metrics = v65.confidence_metrics(
            oof, oof[f"confidence__{name}"].to_numpy(float),
            oof[f"high__{name}"].to_numpy(bool),
        )
        active = []
        for _, group in oof.groupby("guard_outer_fold"):
            if str(group.selected_candidate.iloc[0]) == name:
                active.extend(group.selected_guard_active.to_numpy(bool).tolist())
        metrics["guard_spec"] = spec
        metrics["selected_fold_activation_coverage"] = (
            float(np.mean(active)) if active else None
        )
        candidate_metrics[name] = metrics
    selected = v65.confidence_metrics(
        oof, oof.selected_confidence.to_numpy(float), oof.selected_high_conf.to_numpy(bool),
    )
    selected["guard_activation_n"] = int(oof.selected_guard_active.sum())
    selected["guard_activation_coverage"] = float(oof.selected_guard_active.mean())
    selected["candidate_by_fold"] = {
        str(key): str(value) for key, value in oof.groupby("guard_outer_fold").selected_candidate.first().items()
    }
    baseline_name = "V58_FROZEN_CONFIDENCE_NOOP"
    baseline = candidate_metrics[baseline_name]
    frozen_original = v65.confidence_metrics(
        oof, oof.confidence_signal.to_numpy(float), oof.v58_high_conf.to_numpy(bool),
    )
    fold_deltas = []
    for fold, group in oof.groupby("guard_outer_fold"):
        selected_auc = v65.safe_auc(
            group.correct_target.to_numpy(int), group.selected_confidence.to_numpy(float),
        )
        baseline_auc = v65.safe_auc(
            group.correct_target.to_numpy(int), group[f"confidence__{baseline_name}"].to_numpy(float),
        )
        fold_deltas.append({
            "fold": int(fold), "selected_auc": selected_auc,
            "baseline_auc": baseline_auc,
            "auc_delta": None if selected_auc is None or baseline_auc is None else selected_auc - baseline_auc,
            "candidate": str(group.selected_candidate.iloc[0]),
        })
    positive_fraction = float(np.mean([
        row["auc_delta"] is not None and row["auc_delta"] > 0 for row in fold_deltas
    ]))
    guard_fraction = float(np.mean([audit["selected_mode"] != "noop" for audit in audits]))
    checks = {
        "confidence_auc_delta_ge_0_010": (
            selected["confidence_correctness_auc"] >= baseline["confidence_correctness_auc"] + 0.010
        ),
        "highconf_accuracy_delta_ge_0_010": (
            selected["highconf_accuracy"] is not None and baseline["highconf_accuracy"] is not None
            and selected["highconf_accuracy"] >= baseline["highconf_accuracy"] + 0.010
        ),
        "highconf_net_not_worse": (
            selected["highconf_mean_signed_net"] is not None
            and baseline["highconf_mean_signed_net"] is not None
            and selected["highconf_mean_signed_net"] >= baseline["highconf_mean_signed_net"]
        ),
        "positive_fold_auc_fraction_ge_0_60": positive_fraction >= 0.60,
        "guard_selected_fold_fraction_ge_0_60": guard_fraction >= 0.60,
        "extreme_activation_coverage_0_05_to_0_30": (
            0.05 <= selected["guard_activation_coverage"] <= 0.30
        ),
        "highconf_risk_coverage_0_10_to_0_35": 0.10 <= selected["highconf_coverage"] <= 0.35,
    }
    material_pass = bool(all(checks.values()))
    # Reuse the V65 day-block bootstrap with identical comparison column names.
    bootstrap_frame = oof.copy()
    bootstrap_frame["confidence__V58_FROZEN_CONFIDENCE"] = bootstrap_frame[
        f"confidence__{baseline_name}"
    ]
    bootstrap_frame["high__V58_FROZEN_CONFIDENCE"] = bootstrap_frame[f"high__{baseline_name}"]
    risk_rows = []
    risk_rows.extend(v65.risk_coverage_rows(
        oof, oof.selected_confidence.to_numpy(float), "INNER_SELECTED_EXTREME_GUARD",
    ))
    risk_rows.extend(v65.risk_coverage_rows(
        oof, oof[f"confidence__{baseline_name}"].to_numpy(float), "V58_FROZEN_CONFIDENCE",
    ))
    return {
        "eligible_oof_n": len(oof),
        "eligible_oof_coverage_of_us_news": float(len(oof) / v65.EXPECTED_US_NEWS_ROWS),
        "burn_in_n": int(v65.EXPECTED_US_NEWS_ROWS - len(oof)),
        "candidate_metrics": candidate_metrics,
        "inner_selected_metrics": selected,
        "v58_inner_threshold_baseline_metrics": baseline,
        "v58_original_mask_metrics": frozen_original,
        "fold_auc_deltas": fold_deltas,
        "positive_fold_auc_fraction": positive_fraction,
        "guard_selected_fold_fraction": guard_fraction,
        "material_checks": checks,
        "material_pass": material_pass,
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_NOOP",
        "bootstrap": v65.bootstrap_comparison(bootstrap_frame),
        "risk_coverage_rows": risk_rows,
    }


def final_policy_frame(all_oof: pd.DataFrame, guard_oof: pd.DataFrame, material_pass: bool) -> pd.DataFrame:
    final = all_oof.copy()
    final["prob_before_v66"] = final.prob.to_numpy(float)
    final["confidence_before_v66"] = final.confidence_signal.to_numpy(float)
    final["high_conf_before_v66"] = final.v58_high_conf.to_numpy(bool)
    if material_pass:
        lookup = guard_oof.set_index("event_id")
        eligible = final.event_id.isin(lookup.index) & final.source_family.eq("US_NEWS")
        ids = final.loc[eligible, "event_id"]
        final.loc[eligible, "confidence_signal"] = lookup.loc[ids, "selected_confidence"].to_numpy(float)
        final.loc[eligible, "high_conf"] = lookup.loc[ids, "selected_high_conf"].to_numpy(bool)
        final["family"] = "V66_EXTREME_REVERSAL_GUARD"
    else:
        final["confidence_signal"] = final.confidence_before_v66.to_numpy(float)
        final["high_conf"] = final.high_conf_before_v66.to_numpy(bool)
        final["family"] = "V58_LOCKED_SOURCE_DGP_ROUTING_NOOP"
    require(np.array_equal(final.prob.to_numpy(float), final.prob_before_v66.to_numpy(float)),
            "V66 final policy changed direction probability")
    require(np.array_equal(final.prob.ge(0.5).to_numpy(), final.prob_before_v66.ge(0.5).to_numpy()),
            "V66 final policy changed direction prediction")
    if not material_pass:
        require(np.array_equal(final.confidence_signal.to_numpy(float),
                               final.confidence_before_v66.to_numpy(float)),
                "V66 no-op confidence differs from V58")
        require(np.array_equal(final.high_conf.to_numpy(bool),
                               final.high_conf_before_v66.to_numpy(bool)),
                "V66 no-op mask differs from V58")
    return final


def build_reports(
    all_oof: pd.DataFrame,
    guard_oof: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    material_pass = bool(evaluation["material_pass"])
    final = final_policy_frame(all_oof, guard_oof, material_pass)
    baseline_summary = v44.summarize(all_oof)
    final_summary = v44.summarize(final)
    status = evaluation["status"]
    validation = (
        "V58 direction and base confidence are frozen. Only inner-past-defined extreme "
        "late-contribution tails can compress US_NEWS confidence. Nested folds use strict "
        "35-minute embargoes; raw features/signs are unchanged; no seal/final/extension data."
    )
    comparison = {
        "version": "V66", "hypothesis": HYPOTHESIS, "family": FAMILY,
        "status": status, "material_pass": material_pass,
        "selected_policy": "V66_EXTREME_REVERSAL_GUARD" if material_pass else "V58_EXACT_NOOP",
        "direction_probability_changed": False, "direction_prediction_changed": False,
        "models": {"V58_BASELINE": baseline_summary, "V66_FAIL_CLOSED_SELECTED": final_summary},
        "us_news_guard_evaluation": evaluation, "input_audit": input_audit,
        "validation": validation, "seal_authorized": False,
    }
    robustness = {
        "version": "V66", "hypothesis": HYPOTHESIS, "family": FAMILY,
        "status": status, "opened_dev_only": True, "seal_outcomes_loaded": False,
        "extension_loaded": False, "selected": final_summary, "baseline": baseline_summary,
        "material_checks": evaluation["material_checks"], "bootstrap": evaluation["bootstrap"],
        "fallback_is_exact_v58_noop": not material_pass,
        "direction_probability_changed": False, "seal_authorized": False,
    }
    transfer = {
        "version": "V66", "hypothesis": HYPOTHESIS, "family": FAMILY,
        "status": status, "seal_outcomes_loaded": False, "extension_loaded": False,
        "selected_oof_by_source": final_summary["metrics"]["by_source_family"],
        "selected_oof_by_market": final_summary["metrics"]["by_market"],
        "us_news_extreme_guard": {
            "eligible_oof_n": evaluation["eligible_oof_n"],
            "selected": evaluation["inner_selected_metrics"],
            "baseline": evaluation["v58_inner_threshold_baseline_metrics"],
        },
        "method": validation,
    }
    diagnostic = {
        "version": "V66", "hypothesis": HYPOTHESIS, "family": FAMILY,
        "status": status, "material_difference_from_v65": input_audit["v66_material_difference"],
        "guard_constraints": {
            "target_source": "US_NEWS", "blocks": list(v65.NEGATIVE_BLOCKS),
            "modes": ["penalty", "clip", "saturation"],
            "tail_quantiles": list(TAIL_QUANTILES),
            "raw_feature_sign_flip_allowed": False,
            "raw_feature_sign_flip_performed": False,
        },
        "evaluation": evaluation, "nested_selection_audits": audits,
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V58 confidence/high_conf no-op" if not material_pass else None,
        },
        "input_audit": input_audit, "validation": validation,
    }
    selection = {
        "version": "V66", "hypothesis": HYPOTHESIS, "status": status,
        "candidate_count": len(candidate_specs()), "candidate_specs": candidate_specs(),
        "selection_audits": audits, "material_checks": evaluation["material_checks"],
    }
    comparison_rows = []
    for name, metrics in evaluation["candidate_metrics"].items():
        row = {key: value for key, value in metrics.items() if key != "guard_spec"}
        comparison_rows.append({"candidate": name, **metrics["guard_spec"], **row})
    export_columns = [
        "event_id", "event_group_id", "event_time_utc", "ticker", "source_family",
        "y", "fwd_ret_30m", "prob", "correct_target", "guard_outer_fold",
        "selected_candidate", "selected_mode", "selected_confidence",
        "selected_high_conf", "selected_guard_active", "selected_guard_strength",
        "confidence_signal", "v58_high_conf",
    ] + [f"late_contribution__{block}" for block in v65.NEGATIVE_BLOCKS]
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "EXTREME_REVERSAL_GUARD_REPORT.json": diagnostic,
        "EXTREME_GUARD_SELECTION_REPORT.json": selection,
    }
    table_reports = {
        "GUARD_CANDIDATE_COMPARISON.csv": pd.DataFrame(comparison_rows),
        "RISK_COVERAGE_REPORT.csv": pd.DataFrame(evaluation["risk_coverage_rows"]),
        "US_NEWS_EXTREME_GUARD_OOF.csv.gz": guard_oof[export_columns].copy(),
    }
    return json_reports, table_reports


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.name.endswith(".csv.gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    input_audit = verify_prior_evidence()
    all_oof, us_news = v65.load_frame()
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "material_difference": input_audit["v66_material_difference"],
            "input_audit": input_audit, "us_news_n": len(us_news),
            "direction_probability_changed": False, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    guard_oof, audits = run_nested_oof(us_news, smoke=args.smoke_test)
    evaluation = evaluate(guard_oof, audits)
    if args.smoke_test:
        print(json.dumps(json_clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS,
            "outer_oof_n": len(guard_oof),
            "selected_candidate": audits[0]["selected_candidate"],
            "selected_metrics": evaluation["inner_selected_metrics"],
            "direction_probability_changed": False, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    json_reports, table_reports = build_reports(
        all_oof, guard_oof, audits, evaluation, input_audit,
    )
    json_reports = {name: json_clean(payload) for name, payload in json_reports.items()}
    output_names = sorted([*json_reports, *table_reports])
    if args.dry_run:
        print(json.dumps(json_clean({
            "status": "DRY_RUN_OK", "hypothesis": HYPOTHESIS,
            "material_status": evaluation["status"],
            "material_checks": evaluation["material_checks"],
            "inner_selected_metrics": evaluation["inner_selected_metrics"],
            "v58_baseline_metrics": evaluation["v58_inner_threshold_baseline_metrics"],
            "candidate_by_fold": evaluation["inner_selected_metrics"]["candidate_by_fold"],
            "fallback_is_exact_v58_noop": not evaluation["material_pass"],
            "direction_probability_changed": False,
            "would_write": output_names, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    for name, payload in json_reports.items():
        v37.atomic_json(payload, OUT / name)
    for name, payload in table_reports.items():
        atomic_csv(payload, OUT / name)
    print(json.dumps({
        "status": evaluation["status"], "hypothesis": HYPOTHESIS,
        "output_dir": str(OUT), "files": output_names,
        "fallback_is_exact_v58_noop": not evaluation["material_pass"],
        "direction_probability_changed": False, "seal_authorized": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
