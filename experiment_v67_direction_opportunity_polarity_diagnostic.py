"""V67: separate direction and opportunity feature-block polarity diagnostics.

This audit is diagnostic-only.  It does not alter V58 probability, direction,
confidence, or high-confidence membership.  For each supported source family,
fixed linear probes estimate chronological leave-one-block-out contribution to
(a) direction and (b) absolute-move opportunity.  All preprocessing, direction
cutoffs, material-move thresholds, and opportunity top-slice references are fit
inside past-only nested folds with a strict 35-minute embargo.

No raw feature sign is flipped.  A polarity is non-zero only when at least 70%
of eligible outer-fold signs agree.  KR_KIND is forced to conservative zero
because its immutable DEV support is below the declared source minimum.
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
from scipy import sparse
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import experiment_v64_feature_polarity_diagnostic as v64
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V58_OOF_PATH = ROOT / "cache" / "v58_locked_source_routing_oof.csv.gz"
REGISTRY_PATH = ROOT / "research" / "FEATURE_BLOCK_REGISTRY.json"
OUT = Path(os.environ.get(
    "MARKET_BIO_VERSION_OUTPUT",
    str(ROOT / "staging" / "V67_DIRECTION_OPPORTUNITY_POLARITY_DIAGNOSTIC_V1"),
))
HYPOTHESIS = "DIRECTION_OPPORTUNITY_POLARITY_DIAGNOSTIC_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V58_OOF_PATH: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
    REGISTRY_PATH: "e34fcb63993dc774b2d2df7ac2a4103f438df05cc094a6cd423885185b91b290",
}
EXPECTED_DEV_ROWS = 9462
EXPECTED_V58_ROWS = 7568
SOURCES = ("US_SEC", "US_NEWS", "KR_NEWS", "KR_KIND")
SOURCE_MINIMUM = 200
MINIMUM_FOLDS = 3
STABILITY_THRESHOLD = 0.70
DIRECTION_EPSILON = 0.0015
OPPORTUNITY_EPSILON = 0.0015
DIRECTION_CUTOFFS = (0.48, 0.50, 0.52)
MATERIAL_QUANTILE = 0.70
OPPORTUNITY_TOP_QUANTILE = 0.80
COST = float(v36.COST)
SEED = 6701


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


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V67 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V67 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    require(registry.get("hypothesis") == "FEATURE_POLARITY_DIAGNOSTIC_V1",
            "feature registry hypothesis mismatch")
    require(registry.get("registry_name") == "FEATURE_BLOCK_REGISTRY",
            "feature registry name mismatch")
    registry_blocks = registry.get("blocks", {})
    require(list(registry_blocks) == list(v64.BLOCK_REGISTRY),
            "feature registry block order changed")
    for name, spec in v64.BLOCK_REGISTRY.items():
        require(registry_blocks[name]["kind"] == spec["kind"],
                f"feature registry kind changed: {name}")
        require(registry_blocks[name]["columns"] == spec["columns"],
                f"feature registry columns changed: {name}")
    return {
        "input_sha256": hashes,
        "authorized_inputs": [str(path.relative_to(ROOT)) for path in EXPECTED_SHA256],
        "dev_extension_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
        "v58_predictions_changed": False,
        "v58_confidence_changed": False,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
    }, registry


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    v58 = pd.read_csv(
        V58_OOF_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_DEV_ROWS and dev.event_id.is_unique,
            "immutable V36 DEV contract changed")
    require(len(v58) == EXPECTED_V58_ROWS and v58.event_id.is_unique,
            "frozen V58 OOF contract changed")
    require(set(v58.event_id).issubset(set(dev.event_id)), "V58 is not a V36 DEV subset")
    dev["event_time_utc"] = pd.to_datetime(dev.event_time_utc, utc=True)
    dev["text_cluster_id"] = dev.apply(v37.normalized_cluster, axis=1)
    dev["cluster_weight"] = 1.0 / dev.groupby("text_cluster_id").event_id.transform("size")
    lookup = v58.set_index("event_id")
    dev["v58_prob"] = dev.event_id.map(lookup.prob)
    dev["v58_confidence"] = dev.event_id.map(lookup.confidence_signal)
    return dev, v58.copy()


def chronology_audit(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - pd.Timedelta(minutes=35),
            f"{label} violates strict 35-minute embargo")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end.isoformat(), "valid_start": valid_start.isoformat(),
        "strict_35m_embargo_verified": True,
    }


def stack(matrices: dict[str, sparse.csr_matrix], blocks: list[str]) -> sparse.csr_matrix:
    require(bool(blocks), "cannot build an empty feature design")
    return sparse.hstack([matrices[name] for name in blocks], format="csr", dtype=np.float32)


def fit_direction(
    train_matrix: sparse.csr_matrix,
    train: pd.DataFrame,
    target_matrix: sparse.csr_matrix,
    seed: int,
) -> np.ndarray:
    return v64.fit_probe(
        train_matrix, train.y.to_numpy(int), train.cluster_weight.to_numpy(float),
        target_matrix, seed,
    )


def fit_opportunity(
    train_matrix: sparse.csr_matrix,
    train: pd.DataFrame,
    target_matrix: sparse.csr_matrix,
    seed: int,
) -> np.ndarray:
    target = np.log1p(np.abs(train.fwd_ret_30m.to_numpy(float)) * 10000.0)
    model = SGDRegressor(
        loss="huber", epsilon=1.35, penalty="l2", alpha=3e-4,
        max_iter=1400, tol=1e-3, average=True, random_state=seed,
    )
    model.fit(train_matrix, target, sample_weight=train.cluster_weight.to_numpy(float))
    return model.predict(target_matrix)


def direction_metrics(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    prediction = np.asarray(probability, float) >= cutoff
    accuracy = float((prediction == y).mean())
    base = float(y.mean())
    balanced = float(balanced_accuracy_score(y, prediction)) if np.unique(y).size == 2 else None
    auc = safe_auc(y, np.asarray(probability, float))
    edge = accuracy - max(base, 1.0 - base)
    score = None if balanced is None or auc is None else float(
        0.45 * balanced + 0.45 * auc + 0.10 * (0.5 + edge)
    )
    return {
        "n": len(frame), "cutoff": float(cutoff), "accuracy": accuracy,
        "balanced_accuracy": balanced, "auc": auc, "edge_vs_naive": edge,
        "late_score": score,
    }


def choose_direction_cutoff(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    candidates = [direction_metrics(frame, probability, cutoff) for cutoff in DIRECTION_CUTOFFS]
    candidates.sort(key=lambda row: (
        row["late_score"] if row["late_score"] is not None else -1.0,
        -abs(row["cutoff"] - 0.5),
    ), reverse=True)
    return {"selected_cutoff": candidates[0]["cutoff"], "candidates": candidates}


def economic_robustness(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, Any]:
    eligible = np.asarray(selected, bool) & frame.v58_prob.notna().to_numpy()
    if not eligible.any():
        return {"eligible_n": 0, "mean_signed_net": None, "median_signed_net": None,
                "winsorized_net": None, "top_1_removed_net": None, "top_5_removed_net": None}
    direction = frame.v58_prob.to_numpy(float)[eligible] >= 0.5
    signed = np.where(direction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float)[eligible] - COST
    low, high = np.quantile(signed, [0.01, 0.99]) if len(signed) >= 5 else (signed.min(), signed.max())
    order = np.argsort(np.abs(signed))[::-1]

    def removed(count: int) -> float | None:
        keep = np.ones(len(signed), dtype=bool)
        keep[order[:min(count, len(signed))]] = False
        return float(signed[keep].mean()) if keep.any() else None

    return {
        "eligible_n": int(len(signed)),
        "mean_signed_net": float(signed.mean()),
        "median_signed_net": float(np.median(signed)),
        "winsorized_net": float(np.clip(signed, low, high).mean()),
        "top_1_removed_net": removed(1), "top_5_removed_net": removed(5),
    }


def opportunity_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    material_threshold: float,
    score_reference: np.ndarray,
) -> dict[str, Any]:
    magnitude = np.abs(frame.fwd_ret_30m.to_numpy(float))
    score = np.asarray(score, float)
    material = (magnitude >= material_threshold).astype(int)
    magnitude_rank = finite_or_none(pd.Series(score).corr(pd.Series(magnitude), method="spearman"))
    material_auc = safe_auc(material, score)
    opportunity_rank = v37.rank_against(score_reference, score)
    selected = opportunity_rank >= OPPORTUNITY_TOP_QUANTILE
    economic = economic_robustness(frame, selected)
    rank_unit = None if magnitude_rank is None else 0.5 * (magnitude_rank + 1.0)
    economic_unit = (
        0.5 if economic["mean_signed_net"] is None
        else 0.5 + np.clip(float(economic["mean_signed_net"]), -0.01, 0.01) / 0.02
    )
    late_score = None if rank_unit is None or material_auc is None else float(
        0.40 * rank_unit + 0.40 * material_auc + 0.20 * economic_unit
    )
    return {
        "n": len(frame), "material_move_threshold_abs_return": material_threshold,
        "material_move_rate": float(material.mean()),
        "magnitude_rank_spearman": magnitude_rank,
        "material_move_auc": material_auc,
        "opportunity_top_coverage": float(selected.mean()),
        "economic_robustness": economic,
        "late_score": late_score,
    }


def model_pair(
    blocks: list[str],
    inner_train_matrices: dict[str, sparse.csr_matrix],
    inner_valid_matrices: dict[str, sparse.csr_matrix],
    outer_train_matrices: dict[str, sparse.csr_matrix],
    outer_valid_matrices: dict[str, sparse.csr_matrix],
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    x_inner_train = stack(inner_train_matrices, blocks)
    x_inner_valid = stack(inner_valid_matrices, blocks)
    x_outer_train = stack(outer_train_matrices, blocks)
    x_outer_valid = stack(outer_valid_matrices, blocks)

    inner_direction = fit_direction(x_inner_train, inner_train, x_inner_valid, seed)
    direction_policy = choose_direction_cutoff(inner_valid, inner_direction)
    outer_direction = fit_direction(x_outer_train, outer_train, x_outer_valid, seed + 1000)
    direction = direction_metrics(
        outer_valid, outer_direction, float(direction_policy["selected_cutoff"]),
    )

    inner_opportunity = fit_opportunity(x_inner_train, inner_train, x_inner_valid, seed + 2000)
    outer_opportunity = fit_opportunity(x_outer_train, outer_train, x_outer_valid, seed + 3000)
    material_threshold = float(np.quantile(
        np.abs(outer_train.fwd_ret_30m.to_numpy(float)), MATERIAL_QUANTILE,
    ))
    opportunity = opportunity_metrics(
        outer_valid, outer_opportunity, material_threshold, inner_opportunity,
    )
    return {
        "blocks": blocks,
        "direction_inner_policy": direction_policy,
        "direction_outer_metrics": direction,
        "opportunity_inner": {
            "material_threshold_from_inner_train": float(np.quantile(
                np.abs(inner_train.fwd_ret_30m.to_numpy(float)), MATERIAL_QUANTILE,
            )),
            "score_reference_n": len(inner_opportunity),
        },
        "opportunity_outer_metrics": opportunity,
    }


def fold_sign(delta: float, epsilon: float) -> str:
    return "+" if delta > epsilon else "-" if delta < -epsilon else "0"


def run_fold_audits(
    dev: pd.DataFrame,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    block_names = list(v64.BLOCK_REGISTRY)
    sources = ["US_NEWS"] if smoke else list(SOURCES)
    if smoke:
        block_names = block_names[:2]
    direction_rows = []
    opportunity_rows = []
    nested = {"sources": {}, "smoke_test": smoke}
    for source_index, source in enumerate(sources):
        source_frame = dev[dev.source_family.eq(source)].copy()
        folds = v37.chronological_folds(source_frame)
        supported = len(source_frame) >= SOURCE_MINIMUM and len(folds) >= MINIMUM_FOLDS
        nested["sources"][source] = {
            "rows": len(source_frame), "eligible_folds": len(folds),
            "minimum_rows": SOURCE_MINIMUM, "minimum_folds": MINIMUM_FOLDS,
            "supported": supported, "folds": [],
        }
        if not supported:
            for block in block_names:
                sentinel = {
                    "source_family": source, "feature_block": block, "fold": None,
                    "support_status": "INSUFFICIENT_SUPPORT_CONSERVATIVE_ZERO",
                    "source_n": len(source_frame), "eligible_fold_n": len(folds),
                    "fold_polarity": "0", "full_late_score": None,
                    "ablated_late_score": None, "late_score_delta": None,
                }
                direction_rows.append(dict(sentinel))
                opportunity_rows.append(dict(sentinel))
            continue
        if smoke:
            folds = folds[:1]
        for train, valid, fold in folds:
            inner_train, inner_valid = v37.split_inner(train)
            require(len(inner_train) >= 100 and len(inner_valid) >= 30,
                    f"V67 nested rows insufficient: {source} fold {fold}")
            require(inner_train.y.nunique() == 2 and inner_valid.y.nunique() == 2,
                    f"V67 nested classes insufficient: {source} fold {fold}")
            inner_time = chronology_audit(inner_train, inner_valid, f"{source} inner fold {fold}")
            outer_time = chronology_audit(train, valid, f"{source} outer fold {fold}")
            inner_train_matrices, inner_valid_matrices, inner_dimensions = v64.fit_block_matrices(
                inner_train, inner_valid, block_names,
            )
            outer_train_matrices, outer_valid_matrices, outer_dimensions = v64.fit_block_matrices(
                train, valid, block_names,
            )
            seed = SEED + source_index * 100 + int(fold)
            full = model_pair(
                block_names, inner_train_matrices, inner_valid_matrices,
                outer_train_matrices, outer_valid_matrices,
                inner_train, inner_valid, train, valid, seed,
            )
            nested["sources"][source]["folds"].append({
                "fold": int(fold), "inner": inner_time, "outer": outer_time,
                "inner_feature_dimensions": inner_dimensions,
                "outer_feature_dimensions": outer_dimensions,
                "full_model": full,
            })
            for block_index, block in enumerate(block_names):
                ablated_blocks = [name for name in block_names if name != block]
                ablated = model_pair(
                    ablated_blocks, inner_train_matrices, inner_valid_matrices,
                    outer_train_matrices, outer_valid_matrices,
                    inner_train, inner_valid, train, valid, seed + 10 + block_index,
                )
                full_direction = full["direction_outer_metrics"]
                ablated_direction = ablated["direction_outer_metrics"]
                direction_delta = float(full_direction["late_score"] - ablated_direction["late_score"])
                direction_rows.append({
                    "source_family": source, "feature_block": block, "fold": int(fold),
                    "support_status": "SUPPORTED", "source_n": len(source_frame),
                    "eligible_fold_n": len(folds),
                    "full_late_score": full_direction["late_score"],
                    "ablated_late_score": ablated_direction["late_score"],
                    "late_score_delta": direction_delta,
                    "fold_polarity": fold_sign(direction_delta, DIRECTION_EPSILON),
                    "full_auc": full_direction["auc"], "ablated_auc": ablated_direction["auc"],
                    "auc_delta": full_direction["auc"] - ablated_direction["auc"],
                    "full_balanced_accuracy": full_direction["balanced_accuracy"],
                    "ablated_balanced_accuracy": ablated_direction["balanced_accuracy"],
                    "balanced_accuracy_delta": (
                        full_direction["balanced_accuracy"] - ablated_direction["balanced_accuracy"]
                    ),
                    "full_edge": full_direction["edge_vs_naive"],
                    "ablated_edge": ablated_direction["edge_vs_naive"],
                    "edge_delta": full_direction["edge_vs_naive"] - ablated_direction["edge_vs_naive"],
                    "inner_selected_cutoff_full": full["direction_inner_policy"]["selected_cutoff"],
                    "inner_selected_cutoff_ablated": ablated["direction_inner_policy"]["selected_cutoff"],
                    "raw_feature_sign_flip_performed": False,
                })
                full_opportunity = full["opportunity_outer_metrics"]
                ablated_opportunity = ablated["opportunity_outer_metrics"]
                opportunity_delta = float(full_opportunity["late_score"] - ablated_opportunity["late_score"])
                opportunity_rows.append({
                    "source_family": source, "feature_block": block, "fold": int(fold),
                    "support_status": "SUPPORTED", "source_n": len(source_frame),
                    "eligible_fold_n": len(folds),
                    "full_late_score": full_opportunity["late_score"],
                    "ablated_late_score": ablated_opportunity["late_score"],
                    "late_score_delta": opportunity_delta,
                    "fold_polarity": fold_sign(opportunity_delta, OPPORTUNITY_EPSILON),
                    "full_magnitude_rank": full_opportunity["magnitude_rank_spearman"],
                    "ablated_magnitude_rank": ablated_opportunity["magnitude_rank_spearman"],
                    "magnitude_rank_delta": (
                        full_opportunity["magnitude_rank_spearman"]
                        - ablated_opportunity["magnitude_rank_spearman"]
                    ),
                    "full_material_move_auc": full_opportunity["material_move_auc"],
                    "ablated_material_move_auc": ablated_opportunity["material_move_auc"],
                    "material_move_auc_delta": (
                        full_opportunity["material_move_auc"]
                        - ablated_opportunity["material_move_auc"]
                    ),
                    "full_economic_eligible_n": full_opportunity["economic_robustness"]["eligible_n"],
                    "full_economic_mean_net": full_opportunity["economic_robustness"]["mean_signed_net"],
                    "full_economic_winsorized_net": full_opportunity["economic_robustness"]["winsorized_net"],
                    "full_economic_top5_removed_net": full_opportunity["economic_robustness"]["top_5_removed_net"],
                    "ablated_economic_eligible_n": ablated_opportunity["economic_robustness"]["eligible_n"],
                    "ablated_economic_mean_net": ablated_opportunity["economic_robustness"]["mean_signed_net"],
                    "ablated_economic_winsorized_net": ablated_opportunity["economic_robustness"]["winsorized_net"],
                    "ablated_economic_top5_removed_net": ablated_opportunity["economic_robustness"]["top_5_removed_net"],
                    "material_threshold_from_outer_train": full_opportunity[
                        "material_move_threshold_abs_return"
                    ],
                    "opportunity_top_quantile": OPPORTUNITY_TOP_QUANTILE,
                    "raw_feature_sign_flip_performed": False,
                })
            print(
                f"[V67 DIRECTION/OPPORTUNITY] source={source} fold={fold} "
                f"train={len(train)} valid={len(valid)} blocks={len(block_names)}",
                flush=True,
            )
    return direction_rows, opportunity_rows, nested


def aggregate_sign(rows: pd.DataFrame, source: str, block: str) -> dict[str, Any]:
    group = rows[
        rows.source_family.eq(source) & rows.feature_block.eq(block)
        & rows.support_status.eq("SUPPORTED")
    ].copy()
    if group.empty:
        return {
            "polarity": "0", "fold_n": 0, "stability": 0.0,
            "positive_fraction": 0.0, "zero_fraction": 1.0, "negative_fraction": 0.0,
            "median_late_score_delta": None,
            "fallback_reason": "INSUFFICIENT_SUPPORT_CONSERVATIVE_ZERO",
        }
    median = float(group.late_score_delta.median())
    median_sign = fold_sign(median, DIRECTION_EPSILON)
    stability = float((group.fold_polarity == median_sign).mean()) if median_sign != "0" else 0.0
    if median_sign in {"+", "-"} and stability >= STABILITY_THRESHOLD:
        polarity = median_sign
        fallback = None
    else:
        polarity = "0"
        fallback = (
            "median_within_epsilon" if median_sign == "0" else "stability_below_0.70"
        )
    return {
        "polarity": polarity, "fold_n": int(len(group)), "stability": stability,
        "positive_fraction": float(group.fold_polarity.eq("+").mean()),
        "zero_fraction": float(group.fold_polarity.eq("0").mean()),
        "negative_fraction": float(group.fold_polarity.eq("-").mean()),
        "median_late_score_delta": median, "fallback_reason": fallback,
    }


def aggregate_reports(
    direction_rows: list[dict[str, Any]],
    opportunity_rows: list[dict[str, Any]],
    registry: dict[str, Any],
    support: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    direction = pd.DataFrame(direction_rows)
    opportunity = pd.DataFrame(opportunity_rows)
    mapping: dict[str, Any] = {}
    stability_rows = []
    drilldown = []
    for source in SOURCES:
        mapping[source] = {}
        for block in v64.BLOCK_REGISTRY:
            direction_result = aggregate_sign(direction, source, block)
            opportunity_result = aggregate_sign(opportunity, source, block)
            reversal = (
                direction_result["polarity"] in {"+", "-"}
                and opportunity_result["polarity"] in {"+", "-"}
                and direction_result["polarity"] != opportunity_result["polarity"]
            )
            stable_evidence = (
                direction_result["polarity"] != "0" or opportunity_result["polarity"] != "0"
            )
            mapping[source][block] = {
                "direction": direction_result,
                "opportunity": opportunity_result,
                "direction_opportunity_reversal": reversal,
                "source_supported": bool(support[source]["supported"]),
            }
            stability_rows.append({
                "source_family": source, "feature_block": block,
                "source_n": support[source]["rows"],
                "source_supported": support[source]["supported"],
                "direction_polarity": direction_result["polarity"],
                "direction_stability": direction_result["stability"],
                "direction_median_delta": direction_result["median_late_score_delta"],
                "direction_fallback_reason": direction_result["fallback_reason"],
                "opportunity_polarity": opportunity_result["polarity"],
                "opportunity_stability": opportunity_result["stability"],
                "opportunity_median_delta": opportunity_result["median_late_score_delta"],
                "opportunity_fallback_reason": opportunity_result["fallback_reason"],
                "direction_opportunity_reversal": reversal,
                "raw_feature_sign_flip_performed": False,
            })
            if stable_evidence or reversal:
                spec = registry["blocks"][block]
                drilldown.append({
                    "source_family": source, "feature_block": block,
                    "evidence_reason": "BLOCK_REVERSAL" if reversal else "STABLE_BLOCK_POLARITY",
                    "direction_polarity": direction_result["polarity"],
                    "opportunity_polarity": opportunity_result["polarity"],
                    "feature_kind": spec["kind"], "features": spec["columns"],
                    "feature_level_polarity_inferred": False,
                    "note": (
                        "Drill-down is an inventory of the evidenced block only; individual raw "
                        "feature signs are not inferred or flipped."
                    ),
                })
    return mapping, pd.DataFrame(stability_rows), drilldown


def build_reports(
    dev: pd.DataFrame,
    v58: pd.DataFrame,
    input_audit: dict[str, Any],
    registry: dict[str, Any],
    direction_rows: list[dict[str, Any]],
    opportunity_rows: list[dict[str, Any]],
    nested: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    support = {
        source: {
            key: value for key, value in nested["sources"][source].items() if key != "folds"
        }
        for source in SOURCES
    }
    mapping, stability, drilldown = aggregate_reports(
        direction_rows, opportunity_rows, registry, support,
    )
    v58_summary = v44.summarize(v58)
    require(abs(v58_summary["metrics"]["balanced_accuracy"] - 0.525511873859576) < 1e-12,
            "V67 V58 BA changed")
    require(abs(v58_summary["metrics"]["auc"] - 0.5229106843440796) < 1e-12,
            "V67 V58 AUC changed")
    validation = (
        "Diagnostic only; V58 OOF prediction/confidence is unchanged. Source-specific outer "
        "chronology and inner-past preprocessing/cutoff/opportunity references use strict 35-minute "
        "embargoes. Block contribution is full minus leave-one-block-out late score. Stability below "
        "70% and KR_KIND insufficient support map to zero. No raw sign flip or reserved labels."
    )
    map_report = {
        "version": "V67", "hypothesis": HYPOTHESIS,
        "diagnostic_only": True, "stability_threshold": STABILITY_THRESHOLD,
        "direction_epsilon": DIRECTION_EPSILON,
        "opportunity_epsilon": OPPORTUNITY_EPSILON,
        "source_support": support, "polarity_map": mapping,
        "feature_drilldown_for_stable_or_reversal_only": drilldown,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
    }
    diagnostic = {
        "version": "V67", "hypothesis": HYPOTHESIS,
        "status": "DIAGNOSTIC_ONLY", "input_audit": input_audit,
        "source_support": support,
        "direction_target": "y = (fwd_ret_30m > 0)",
        "opportunity_target": "log1p(abs(fwd_ret_30m)*10000)",
        "direction_metrics": ["AUC", "balanced_accuracy", "edge_vs_naive"],
        "opportunity_metrics": [
            "magnitude_rank_spearman", "past-threshold material_move_auc",
            "V58-direction economic robustness on inner-ranked top opportunity slice",
        ],
        "nested_chronology_audit": nested,
        "feature_drilldown": drilldown,
        "validation": validation,
    }
    comparison = {
        "version": "V67", "hypothesis": HYPOTHESIS,
        "status": "DIAGNOSTIC_ONLY", "champion": "V58_LOCKED_SOURCE_DGP_ROUTING",
        "champion_unchanged": True, "models": {"V58_UNCHANGED": v58_summary},
        "direction_opportunity_polarity_map": mapping,
        "input_audit": input_audit, "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V67", "hypothesis": HYPOTHESIS,
        "status": "DIAGNOSTIC_ONLY", "opened_dev_only": True,
        "seal_outcomes_loaded": False, "dev_extension_loaded": False,
        "selected": v58_summary, "champion_unchanged": True,
        "source_support": support, "seal_authorized": False,
    }
    transfer = {
        "version": "V67", "hypothesis": HYPOTHESIS,
        "status": "DIAGNOSTIC_ONLY", "seal_outcomes_loaded": False,
        "dev_extension_loaded": False,
        "selected_oof_by_source": v58_summary["metrics"]["by_source_family"],
        "selected_oof_by_market": v58_summary["metrics"]["by_market"],
        "direction_opportunity_polarity_by_source": mapping,
        "method": validation,
    }
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json": map_report,
        "DIRECTION_OPPORTUNITY_DIAGNOSTIC_REPORT.json": diagnostic,
    }
    table_reports = {
        "DIRECTION_POLARITY_AUDIT.csv": pd.DataFrame(direction_rows),
        "OPPORTUNITY_POLARITY_AUDIT.csv": pd.DataFrame(opportunity_rows),
        "POLARITY_STABILITY_REPORT.csv": stability,
    }
    return json_reports, table_reports


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
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
    input_audit, registry = verify_inputs()
    dev, v58 = load_frames()
    source_counts = {source: int(dev.source_family.eq(source).sum()) for source in SOURCES}
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "input_audit": input_audit, "source_counts": source_counts,
            "kr_kind_forced_zero": source_counts["KR_KIND"] < SOURCE_MINIMUM,
            "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    direction_rows, opportunity_rows, nested = run_fold_audits(dev, smoke=args.smoke_test)
    if args.smoke_test:
        print(json.dumps(json_clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS,
            "direction_rows": len(direction_rows),
            "opportunity_rows": len(opportunity_rows),
            "source_support": {
                source: {key: value for key, value in row.items() if key != "folds"}
                for source, row in nested["sources"].items()
            },
            "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    json_reports, table_reports = build_reports(
        dev, v58, input_audit, registry, direction_rows, opportunity_rows, nested,
    )
    json_reports = {name: json_clean(payload) for name, payload in json_reports.items()}
    output_names = sorted([*json_reports, *table_reports])
    stability = table_reports["POLARITY_STABILITY_REPORT.csv"]
    summary = {
        "stable_direction_nonzero": int(stability.direction_polarity.ne("0").sum()),
        "stable_opportunity_nonzero": int(stability.opportunity_polarity.ne("0").sum()),
        "direction_opportunity_reversals": int(stability.direction_opportunity_reversal.sum()),
        "kr_kind_all_zero": bool(stability.loc[
            stability.source_family.eq("KR_KIND"),
            ["direction_polarity", "opportunity_polarity"],
        ].eq("0").all().all()),
    }
    if args.dry_run:
        print(json.dumps(json_clean({
            "status": "DRY_RUN_OK", "hypothesis": HYPOTHESIS,
            "diagnostic_summary": summary,
            "v58_predictions_changed": False, "v58_confidence_changed": False,
            "would_write": output_names, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    for name, payload in json_reports.items():
        v37.atomic_json(payload, OUT / name)
    for name, payload in table_reports.items():
        atomic_csv(payload, OUT / name)
    print(json.dumps({
        "status": "DIAGNOSTIC_WRITTEN", "hypothesis": HYPOTHESIS,
        "output_dir": str(OUT), "files": output_names,
        "diagnostic_summary": summary,
        "v58_predictions_changed": False, "v58_confidence_changed": False,
        "seal_authorized": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
