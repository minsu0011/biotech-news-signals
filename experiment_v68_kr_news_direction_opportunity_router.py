"""V68: KR_NEWS direction/opportunity polarity late router.

V67 found 75%-stable direction/opportunity reversal for the event-clock and
text-content blocks on KR_NEWS.  This material intervention fits separate block
experts and lets an embargoed inner-past split choose independent +/0/- late
fusion coefficients for direction and opportunity, plus an opportunity top
threshold.  Raw feature values, text, and raw feature signs are never changed.

Only chronological outer-OOF evidence can pass the predeclared material gates.
If any gate fails, the final output is an exact V58 probability, confidence,
and high-confidence no-op.  KR_KIND is untouched.  No DEV extension, research
seal, or final reserve is in the input graph.
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
from scipy.special import expit, logit
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
V67_DIR = ROOT / "output_V67"
OUT = Path(os.environ.get(
    "MARKET_BIO_VERSION_OUTPUT",
    str(ROOT / "staging" / "V68_KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_V1"),
))
HYPOTHESIS = "KR_NEWS_DIRECTION_OPPORTUNITY_POLARITY_ROUTER_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V58_OOF_PATH: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
    V67_DIR / "COMMIT.json": "de29762eb7a6cbf32fb645313da24fa99c175a7f894f70396d78f2437f266802",
    V67_DIR / "DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json": "d1a8a113896f88b1d4c8353bdc8e5555c0521f08bb9ed1f4353e0b530cb3906e",
    V67_DIR / "DIRECTION_POLARITY_AUDIT.csv": "546fc3726ec5a34c17fa7d6f6689df9374a356699e020945186cb56138cdc465",
    V67_DIR / "OPPORTUNITY_POLARITY_AUDIT.csv": "614ec07113ee8186f068ab7ff68d223920b2ba31518d90e57845b4261e2287a0",
    V67_DIR / "POLARITY_STABILITY_REPORT.csv": "898fffed86c407251d1ed1097b033416739bbf7fd461185e7173fc9a5648a47b",
}
EXPECTED_MANIFEST_SHA256 = "8273a7cb933471bc1e2228ee7dc55b8de4eaaff01e4e5a193e21dc55f6fae865"
EXPECTED_V58_ROWS = 7568
EXPECTED_KR_NEWS_ROWS = 1139
BLOCKS = ("event_clock", "text_content")
COEFFICIENTS = (-0.35, 0.0, 0.35)
DIRECTION_CUTOFFS = (0.48, 0.50, 0.52)
OPPORTUNITY_QUANTILES = (0.70, 0.80, 0.85, 0.90)
MATERIAL_QUANTILE = 0.70
EMBARGO = pd.Timedelta(minutes=35)
COST = float(v36.COST)
SEED = 6801


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


def verify_inputs() -> dict[str, Any]:
    hashes = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V68 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V68 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    commit = json.loads((V67_DIR / "COMMIT.json").read_text(encoding="utf-8"))
    require(commit.get("version") == 67, "V67 commit version mismatch")
    require(commit.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            "V67 commit manifest pointer mismatch")
    require(sha256(V67_DIR / "ARTIFACT_MANIFEST.json") == EXPECTED_MANIFEST_SHA256,
            "V67 artifact manifest changed")
    manifest = json.loads((V67_DIR / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    for name in (
        "DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json",
        "DIRECTION_POLARITY_AUDIT.csv", "OPPORTUNITY_POLARITY_AUDIT.csv",
        "POLARITY_STABILITY_REPORT.csv",
    ):
        require(manifest["files"][name]["sha256"] == EXPECTED_SHA256[V67_DIR / name],
                f"V67 manifest evidence hash mismatch: {name}")
    polarity_map = json.loads(
        (V67_DIR / "DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json").read_text(encoding="utf-8")
    )
    require(polarity_map.get("hypothesis") == "DIRECTION_OPPORTUNITY_POLARITY_DIAGNOSTIC_V1",
            "V67 map hypothesis mismatch")
    for block in BLOCKS:
        evidence = polarity_map["polarity_map"]["KR_NEWS"][block]
        require(evidence["direction"]["polarity"] == "-",
                f"V67 direction polarity changed: {block}")
        require(evidence["opportunity"]["polarity"] == "+",
                f"V67 opportunity polarity changed: {block}")
        require(evidence["direction_opportunity_reversal"] is True,
                f"V67 reversal evidence missing: {block}")
        require(float(evidence["direction"]["stability"]) >= 0.70,
                f"V67 direction stability insufficient: {block}")
        require(float(evidence["opportunity"]["stability"]) >= 0.70,
                f"V67 opportunity stability insufficient: {block}")
    return {
        "input_sha256": hashes,
        "v67_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "v67_reversal_blocks": {
            block: {"direction": "-", "opportunity": "+", "minimum_stability": 0.70}
            for block in BLOCKS
        },
        "scope": "KR_NEWS only; KR_KIND and every other source unchanged",
        "dev_extension_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
    }


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    v58 = pd.read_csv(
        V58_OOF_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(v58) == EXPECTED_V58_ROWS and v58.event_id.is_unique,
            "V58 OOF contract changed")
    require(set(v58.event_id).issubset(set(dev.event_id)), "V58 is not a V36 DEV subset")
    raw = dev.set_index("event_id").loc[v58.event_id].reset_index()
    for column in v58.columns:
        if column not in raw.columns or column in {
            "prob", "high_conf", "confidence_signal", "fold", "family",
            "new_issuer", "text_cluster_id", "cluster_weight", "routed_from",
        }:
            raw[column] = v58[column].to_numpy()
    raw["event_time_utc"] = pd.to_datetime(raw.event_time_utc, utc=True)
    raw["text_cluster_id"] = raw.apply(v37.normalized_cluster, axis=1)
    raw["cluster_weight"] = 1.0 / raw.groupby("text_cluster_id").event_id.transform("size")
    kr_news = raw[raw.source_family.eq("KR_NEWS")].copy()
    require(len(kr_news) == EXPECTED_KR_NEWS_ROWS, f"KR_NEWS V58 rows changed: {len(kr_news)}")
    require(kr_news.y.nunique() == 2, "KR_NEWS direction target has one class")
    return v58, kr_news


def chronology_audit(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label} violates strict 35-minute embargo")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end.isoformat(), "valid_start": valid_start.isoformat(),
        "strict_35m_embargo_verified": True,
    }


def fit_block_experts(
    train: pd.DataFrame,
    target: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    train_matrices, target_matrices, dimensions = v64.fit_block_matrices(
        train, target, list(BLOCKS),
    )
    direction = {}
    opportunity = {}
    magnitude_target = np.log1p(np.abs(train.fwd_ret_30m.to_numpy(float)) * 10000.0)
    for index, block in enumerate(BLOCKS):
        direction[block] = v64.fit_probe(
            train_matrices[block], train.y.to_numpy(int),
            train.cluster_weight.to_numpy(float), target_matrices[block], seed + index,
        )
        model = SGDRegressor(
            loss="huber", epsilon=1.35, penalty="l2", alpha=3e-4,
            max_iter=1400, tol=1e-3, average=True, random_state=seed + 100 + index,
        )
        model.fit(
            train_matrices[block], magnitude_target,
            sample_weight=train.cluster_weight.to_numpy(float),
        )
        opportunity[block] = model.predict(target_matrices[block])
    return direction, opportunity, dimensions


def direction_candidates(
    base_probability: np.ndarray,
    expert_probability: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    base_logit = logit(np.clip(np.asarray(base_probability, float), 1e-6, 1.0 - 1e-6))
    centered = {
        block: np.clip(logit(np.clip(expert_probability[block], 1e-6, 1.0 - 1e-6)), -3.0, 3.0)
        for block in BLOCKS
    }
    candidates = {}
    for clock_weight in COEFFICIENTS:
        for text_weight in COEFFICIENTS:
            name = f"DIR_CLOCK_{clock_weight:+.2f}_TEXT_{text_weight:+.2f}"
            probability = expit(
                base_logit + clock_weight * centered["event_clock"]
                + text_weight * centered["text_content"]
            )
            candidates[name] = {
                "probability": probability,
                "clock_weight": clock_weight, "text_weight": text_weight,
            }
    return candidates


def direction_metrics(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    prediction = np.asarray(probability, float) >= cutoff
    accuracy = float((prediction == y).mean())
    base = float(y.mean())
    balanced = float(balanced_accuracy_score(y, prediction))
    auc = safe_auc(y, np.asarray(probability, float))
    edge = accuracy - max(base, 1.0 - base)
    score = float(0.45 * balanced + 0.45 * float(auc) + 0.10 * (0.5 + edge))
    return {
        "n": len(frame), "cutoff": cutoff, "accuracy": accuracy,
        "balanced_accuracy": balanced, "auc": auc,
        "edge_vs_naive": edge, "selection_score": score,
    }


def choose_direction_policy(
    frame: pd.DataFrame,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for name, candidate in candidates.items():
        for cutoff in DIRECTION_CUTOFFS:
            metrics = direction_metrics(frame, candidate["probability"], cutoff)
            rows.append({
                "candidate": name, "clock_weight": candidate["clock_weight"],
                "text_weight": candidate["text_weight"], "cutoff": cutoff,
                "metrics": metrics, "selection_score": metrics["selection_score"],
            })
    rows.sort(key=lambda row: (
        row["selection_score"],
        -(abs(row["clock_weight"]) + abs(row["text_weight"])),
        -abs(row["cutoff"] - 0.5),
    ), reverse=True)
    return {"selected": rows[0], "candidates": rows}


def opportunity_candidates(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    expert_reference: dict[str, np.ndarray],
    expert_target: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    base = v37.rank_against(
        reference.opportunity_confidence.to_numpy(float),
        target.opportunity_confidence.to_numpy(float),
    )
    expert_rank = {
        block: v37.rank_against(expert_reference[block], expert_target[block])
        for block in BLOCKS
    }
    candidates = {}
    for clock_weight in COEFFICIENTS:
        for text_weight in COEFFICIENTS:
            name = f"OPP_CLOCK_{clock_weight:+.2f}_TEXT_{text_weight:+.2f}"
            score = np.clip(
                base + clock_weight * (expert_rank["event_clock"] - 0.5)
                + text_weight * (expert_rank["text_content"] - 0.5),
                0.0, 1.0,
            )
            candidates[name] = {
                "score": score, "clock_weight": clock_weight,
                "text_weight": text_weight,
            }
    return candidates


def signed_net(frame: pd.DataFrame, probability: np.ndarray, selected: np.ndarray) -> np.ndarray:
    direction = np.asarray(probability, float) >= 0.5
    net = np.where(direction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return net[np.asarray(selected, bool)]


def opportunity_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    high: np.ndarray,
    direction_probability: np.ndarray,
    material_threshold: float,
) -> dict[str, Any]:
    magnitude = np.abs(frame.fwd_ret_30m.to_numpy(float))
    material = (magnitude >= material_threshold).astype(int)
    score = np.asarray(score, float)
    high = np.asarray(high, bool)
    rank = finite_or_none(pd.Series(score).corr(pd.Series(magnitude), method="spearman"))
    auc = safe_auc(material, score)
    net = signed_net(frame, direction_probability, high)
    material_precision = float(material[high].mean()) if high.any() else None
    mean_net = float(net.mean()) if len(net) else None
    economic_unit = 0.5 if mean_net is None else 0.5 + np.clip(mean_net, -0.01, 0.01) / 0.02
    rank_unit = 0.5 if rank is None else 0.5 * (rank + 1.0)
    selection_score = float(
        0.30 * rank_unit + 0.35 * float(auc or 0.5)
        + 0.20 * float(material_precision or 0.0) + 0.15 * economic_unit
    )
    return {
        "n": len(frame), "magnitude_rank_spearman": rank,
        "material_move_auc": auc, "material_move_rate": float(material.mean()),
        "material_precision_high": material_precision,
        "highconf_n": int(high.sum()), "highconf_coverage": float(high.mean()),
        "highconf_accuracy": float(
            ((np.asarray(direction_probability) >= 0.5)[high] == frame.y.to_numpy(int)[high]).mean()
        ) if high.any() else None,
        "highconf_mean_signed_net": mean_net,
        "highconf_median_signed_net": float(np.median(net)) if len(net) else None,
        "selection_score": selection_score,
    }


def choose_opportunity_policy(
    frame: pd.DataFrame,
    candidates: dict[str, dict[str, Any]],
    direction_probability: np.ndarray,
    material_threshold: float,
) -> dict[str, Any]:
    rows = []
    for name, candidate in candidates.items():
        ranks = v37.rank_self(candidate["score"])
        for quantile in OPPORTUNITY_QUANTILES:
            high = ranks >= quantile
            metrics = opportunity_metrics(
                frame, candidate["score"], high, direction_probability, material_threshold,
            )
            rows.append({
                "candidate": name, "clock_weight": candidate["clock_weight"],
                "text_weight": candidate["text_weight"], "quantile": quantile,
                "metrics": metrics, "selection_score": metrics["selection_score"],
            })
    rows.sort(key=lambda row: (
        row["selection_score"],
        -(abs(row["clock_weight"]) + abs(row["text_weight"])),
        -abs(row["quantile"] - 0.80),
    ), reverse=True)
    return {"selected": rows[0], "candidates": rows}


def run_outer_oof(frame: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    folds = v37.chronological_folds(frame)
    require(len(folds) == 4, f"V68 KR_NEWS eligible folds changed: {len(folds)}")
    if smoke:
        folds = folds[:1]
    outputs = []
    audits = []
    for train, valid, fold in folds:
        inner_train, inner_valid = v37.split_inner(train)
        require(len(inner_train) >= 100 and len(inner_valid) >= 30,
                f"V68 inner support insufficient: fold {fold}")
        require(inner_train.y.nunique() == 2 and inner_valid.y.nunique() == 2,
                f"V68 inner classes insufficient: fold {fold}")
        inner_time = chronology_audit(inner_train, inner_valid, f"V68 inner fold {fold}")
        outer_time = chronology_audit(train, valid, f"V68 outer fold {fold}")
        inner_direction_expert, inner_opportunity_expert, inner_dimensions = fit_block_experts(
            inner_train, inner_valid, SEED + fold,
        )
        inner_direction_candidates = direction_candidates(
            inner_valid.prob.to_numpy(float), inner_direction_expert,
        )
        direction_policy = choose_direction_policy(inner_valid, inner_direction_candidates)
        selected_inner_direction_raw = inner_direction_candidates[
            direction_policy["selected"]["candidate"]
        ]["probability"]
        selected_inner_direction = expit(
            logit(np.clip(selected_inner_direction_raw, 1e-6, 1.0 - 1e-6))
            - logit(np.asarray(float(direction_policy["selected"]["cutoff"])))
        )

        # inner_valid is strictly held out from the inner expert fit.  Its own
        # score distribution is therefore a leakage-safe reference for inner
        # policy selection (and avoids an under-supported second inner split).
        inner_opportunity_candidates = opportunity_candidates(
            inner_valid, inner_valid,
            inner_opportunity_expert, inner_opportunity_expert,
        )
        material_threshold_inner = float(np.quantile(
            np.abs(inner_train.fwd_ret_30m.to_numpy(float)), MATERIAL_QUANTILE,
        ))
        opportunity_policy = choose_opportunity_policy(
            inner_valid, inner_opportunity_candidates,
            selected_inner_direction, material_threshold_inner,
        )

        outer_direction_expert, outer_opportunity_expert, outer_dimensions = fit_block_experts(
            train, valid, SEED + 1000 + fold,
        )
        outer_direction_candidates = direction_candidates(
            valid.prob.to_numpy(float), outer_direction_expert,
        )
        direction_name = direction_policy["selected"]["candidate"]
        router_probability_raw = outer_direction_candidates[direction_name]["probability"]
        direction_cutoff = float(direction_policy["selected"]["cutoff"])
        router_probability = expit(
            logit(np.clip(router_probability_raw, 1e-6, 1.0 - 1e-6))
            - logit(np.asarray(direction_cutoff))
        )
        baseline_probability = valid.prob.to_numpy(float)

        outer_opportunity_candidates = opportunity_candidates(
            inner_valid, valid, inner_opportunity_expert, outer_opportunity_expert,
        )
        opportunity_name = opportunity_policy["selected"]["candidate"]
        inner_opportunity_score = inner_opportunity_candidates[opportunity_name]["score"]
        outer_opportunity_score_raw = outer_opportunity_candidates[opportunity_name]["score"]
        outer_opportunity_rank = v37.rank_against(
            inner_opportunity_score, outer_opportunity_score_raw,
        )
        opportunity_quantile = float(opportunity_policy["selected"]["quantile"])
        router_high = outer_opportunity_rank >= opportunity_quantile
        baseline_opportunity_rank = v37.rank_against(
            inner_valid.opportunity_confidence.to_numpy(float),
            valid.opportunity_confidence.to_numpy(float),
        )
        baseline_high = baseline_opportunity_rank >= opportunity_quantile
        material_threshold_outer = float(np.quantile(
            np.abs(train.fwd_ret_30m.to_numpy(float)), MATERIAL_QUANTILE,
        ))
        candidate_direction_metrics = direction_metrics(valid, router_probability, 0.5)
        baseline_direction_metrics = direction_metrics(valid, baseline_probability, 0.5)
        candidate_opportunity_metrics = opportunity_metrics(
            valid, outer_opportunity_rank, router_high, router_probability,
            material_threshold_outer,
        )
        baseline_opportunity_metrics = opportunity_metrics(
            valid, baseline_opportunity_rank, baseline_high, baseline_probability,
            material_threshold_outer,
        )
        output = valid[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "y", "fwd_ret_30m", "prob", "high_conf",
            "confidence_signal", "opportunity_confidence", "cluster_weight",
        ]].copy()
        output["router_outer_fold"] = int(fold)
        output["router_prob_candidate"] = router_probability
        output["router_direction_candidate"] = direction_name
        output["router_direction_clock_weight"] = direction_policy["selected"]["clock_weight"]
        output["router_direction_text_weight"] = direction_policy["selected"]["text_weight"]
        output["router_direction_inner_cutoff"] = direction_cutoff
        output["router_opportunity_score"] = outer_opportunity_rank
        output["baseline_opportunity_score"] = baseline_opportunity_rank
        output["router_opportunity_high"] = router_high
        output["baseline_opportunity_high"] = baseline_high
        output["router_opportunity_candidate"] = opportunity_name
        output["router_opportunity_clock_weight"] = opportunity_policy["selected"]["clock_weight"]
        output["router_opportunity_text_weight"] = opportunity_policy["selected"]["text_weight"]
        output["router_opportunity_inner_quantile"] = opportunity_quantile
        output["material_move"] = (
            np.abs(valid.fwd_ret_30m.to_numpy(float)) >= material_threshold_outer
        )
        outputs.append(output)
        audits.append({
            "fold": int(fold), "inner": inner_time, "outer": outer_time,
            "inner_dimensions": inner_dimensions, "outer_dimensions": outer_dimensions,
            "direction_policy": direction_policy,
            "opportunity_policy": opportunity_policy,
            "outer_candidate_direction_metrics": candidate_direction_metrics,
            "outer_baseline_direction_metrics": baseline_direction_metrics,
            "outer_candidate_opportunity_metrics": candidate_opportunity_metrics,
            "outer_baseline_opportunity_metrics": baseline_opportunity_metrics,
            "direction_probability_input_changed": False,
            "raw_feature_sign_flip_performed": False,
        })
        print(
            f"[V68 KR ROUTER] fold={fold} train={len(train)} valid={len(valid)} "
            f"direction=({direction_policy['selected']['clock_weight']:+.2f},"
            f"{direction_policy['selected']['text_weight']:+.2f}) "
            f"opportunity=({opportunity_policy['selected']['clock_weight']:+.2f},"
            f"{opportunity_policy['selected']['text_weight']:+.2f}) "
            f"q={opportunity_quantile:.2f}", flush=True,
        )
    oof = pd.concat(outputs, ignore_index=True)
    require(oof.event_id.is_unique, "V68 duplicate KR_NEWS outer OOF event")
    require(np.array_equal(oof.prob.to_numpy(float),
                           frame.set_index("event_id").loc[oof.event_id].prob.to_numpy(float)),
            "V68 mutated frozen V58 input probability")
    return oof, audits


def aggregate_direction_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    return direction_metrics(frame, probability, 0.5)


def aggregate_opportunity_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    high: np.ndarray,
    direction_probability: np.ndarray,
) -> dict[str, Any]:
    material = frame.material_move.to_numpy(int)
    magnitude = np.abs(frame.fwd_ret_30m.to_numpy(float))
    score = np.asarray(score, float)
    high = np.asarray(high, bool)
    net = signed_net(frame, direction_probability, high)
    return {
        "n": len(frame),
        "magnitude_rank_spearman": finite_or_none(
            pd.Series(score).corr(pd.Series(magnitude), method="spearman")
        ),
        "material_move_auc": safe_auc(material, score),
        "material_precision_high": float(material[high].mean()) if high.any() else None,
        "highconf_n": int(high.sum()), "highconf_coverage": float(high.mean()),
        "highconf_accuracy": float(
            ((np.asarray(direction_probability) >= 0.5)[high] == frame.y.to_numpy(int)[high]).mean()
        ) if high.any() else None,
        "highconf_mean_signed_net": float(net.mean()) if len(net) else None,
        "highconf_median_signed_net": float(np.median(net)) if len(net) else None,
        "highconf_winsorized_net": float(
            np.clip(net, *np.quantile(net, [0.01, 0.99])).mean()
        ) if len(net) >= 5 else (float(net.mean()) if len(net) else None),
    }


def risk_coverage(frame: pd.DataFrame, probability: np.ndarray, score: np.ndarray, name: str) -> list[dict[str, Any]]:
    order = np.argsort(-np.asarray(score, float), kind="stable")
    rows = []
    for coverage in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00):
        count = max(1, int(math.ceil(len(frame) * coverage)))
        selected = order[:count]
        prediction = np.asarray(probability)[selected] >= 0.5
        y = frame.y.to_numpy(int)[selected]
        net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float)[selected] - COST
        rows.append({
            "policy": name, "target_coverage": coverage,
            "actual_coverage": float(count / len(frame)), "n": count,
            "accuracy": float((prediction == y).mean()),
            "risk": float(1.0 - (prediction == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction))
                if np.unique(y).size == 2 else None,
            "mean_signed_net": float(net.mean()),
            "material_move_rate": float(frame.material_move.to_numpy(bool)[selected].mean()),
        })
    return rows


def candidate_full_frame(v58: pd.DataFrame, router_oof: pd.DataFrame) -> pd.DataFrame:
    candidate = v58.copy()
    lookup = router_oof.set_index("event_id")
    eligible = candidate.event_id.isin(lookup.index) & candidate.source_family.eq("KR_NEWS")
    ids = candidate.loc[eligible, "event_id"]
    candidate.loc[eligible, "prob"] = lookup.loc[ids, "router_prob_candidate"].to_numpy(float)
    candidate.loc[eligible, "confidence_signal"] = lookup.loc[ids, "router_opportunity_score"].to_numpy(float)
    candidate.loc[eligible, "high_conf"] = lookup.loc[ids, "router_opportunity_high"].to_numpy(bool)
    candidate["family"] = "V68_KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_CANDIDATE"
    return candidate


def evaluate(v58: pd.DataFrame, oof: pd.DataFrame, audits: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_direction = aggregate_direction_metrics(
        oof, oof.router_prob_candidate.to_numpy(float),
    )
    baseline_direction = aggregate_direction_metrics(oof, oof.prob.to_numpy(float))
    candidate_opportunity = aggregate_opportunity_metrics(
        oof, oof.router_opportunity_score.to_numpy(float),
        oof.router_opportunity_high.to_numpy(bool),
        oof.router_prob_candidate.to_numpy(float),
    )
    baseline_opportunity = aggregate_opportunity_metrics(
        oof, oof.baseline_opportunity_score.to_numpy(float),
        oof.high_conf.to_numpy(bool), oof.prob.to_numpy(float),
    )
    direction_fold_positive = float(np.mean([
        audit["outer_candidate_direction_metrics"]["selection_score"]
        > audit["outer_baseline_direction_metrics"]["selection_score"]
        for audit in audits
    ]))
    opportunity_fold_positive = float(np.mean([
        audit["outer_candidate_opportunity_metrics"]["selection_score"]
        > audit["outer_baseline_opportunity_metrics"]["selection_score"]
        for audit in audits
    ]))
    baseline_full = v44.summarize(v58)
    candidate_frame = candidate_full_frame(v58, oof)
    candidate_full = v44.summarize(candidate_frame)
    overall_ba_delta = (
        candidate_full["metrics"]["balanced_accuracy"] - baseline_full["metrics"]["balanced_accuracy"]
    )
    overall_auc_delta = candidate_full["metrics"]["auc"] - baseline_full["metrics"]["auc"]
    candidate_kr = candidate_full["metrics"]["by_market"]["KR"]
    baseline_kr = baseline_full["metrics"]["by_market"]["KR"]
    kr_market_ba_delta = candidate_kr["balanced_accuracy"] - baseline_kr["balanced_accuracy"]
    kr_market_auc_delta = candidate_kr["auc"] - baseline_kr["auc"]
    opportunity_or_economic = bool(
        candidate_opportunity["material_move_auc"]
        >= baseline_opportunity["material_move_auc"] + 0.010
        or candidate_opportunity["magnitude_rank_spearman"]
        >= baseline_opportunity["magnitude_rank_spearman"] + 0.020
        or (
            candidate_opportunity["highconf_accuracy"]
            >= baseline_opportunity["highconf_accuracy"] + 0.010
            and candidate_opportunity["highconf_mean_signed_net"]
            >= baseline_opportunity["highconf_mean_signed_net"] + 0.001
        )
    )
    checks = {
        "kr_news_balanced_accuracy_delta_ge_0_010": (
            candidate_direction["balanced_accuracy"] >= baseline_direction["balanced_accuracy"] + 0.010
        ),
        "kr_news_auc_delta_ge_0_010": (
            candidate_direction["auc"] >= baseline_direction["auc"] + 0.010
        ),
        "opportunity_or_hc_economic_improvement": opportunity_or_economic,
        "candidate_highconf_net_gt_0": candidate_opportunity["highconf_mean_signed_net"] > 0.0,
        "direction_positive_fold_fraction_ge_0_75": direction_fold_positive >= 0.75,
        "opportunity_positive_fold_fraction_ge_0_75": opportunity_fold_positive >= 0.75,
        "candidate_highconf_coverage_0_10_to_0_35": (
            0.10 <= candidate_opportunity["highconf_coverage"] <= 0.35
        ),
        "overall_ba_no_regression_below_minus_0_002": overall_ba_delta >= -0.002,
        "overall_auc_no_regression_below_minus_0_002": overall_auc_delta >= -0.002,
        "kr_market_ba_no_regression_below_minus_0_005": kr_market_ba_delta >= -0.005,
        "kr_market_auc_no_regression_below_minus_0_005": kr_market_auc_delta >= -0.005,
    }
    material_pass = bool(all(checks.values()))
    return {
        "eligible_outer_oof_n": len(oof),
        "coverage_of_v58_kr_news": float(len(oof) / EXPECTED_KR_NEWS_ROWS),
        "burn_in_n": EXPECTED_KR_NEWS_ROWS - len(oof),
        "candidate_direction": candidate_direction,
        "baseline_direction": baseline_direction,
        "candidate_opportunity": candidate_opportunity,
        "baseline_opportunity": baseline_opportunity,
        "direction_positive_fold_fraction": direction_fold_positive,
        "opportunity_positive_fold_fraction": opportunity_fold_positive,
        "candidate_full_summary": candidate_full,
        "baseline_full_summary": baseline_full,
        "overall_ba_delta": overall_ba_delta,
        "overall_auc_delta": overall_auc_delta,
        "kr_market_ba_delta": kr_market_ba_delta,
        "kr_market_auc_delta": kr_market_auc_delta,
        "material_checks": checks,
        "material_pass": material_pass,
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_NOOP",
    }


def final_frame(v58: pd.DataFrame, oof: pd.DataFrame, material_pass: bool) -> pd.DataFrame:
    final = v58.copy()
    original_prob = final.prob.to_numpy(float).copy()
    original_confidence = final.confidence_signal.to_numpy(float).copy()
    original_high = final.high_conf.to_numpy(bool).copy()
    if material_pass:
        final = candidate_full_frame(v58, oof)
    else:
        final["family"] = "V58_LOCKED_SOURCE_DGP_ROUTING_NOOP"
    require(np.array_equal(v58.loc[v58.source_family.ne("KR_NEWS"), "prob"].to_numpy(float),
                           final.loc[final.source_family.ne("KR_NEWS"), "prob"].to_numpy(float)),
            "V68 changed a non-KR_NEWS probability")
    require(np.array_equal(v58.loc[v58.source_family.eq("KR_KIND"), "prob"].to_numpy(float),
                           final.loc[final.source_family.eq("KR_KIND"), "prob"].to_numpy(float)),
            "V68 changed KR_KIND")
    if not material_pass:
        require(np.array_equal(final.prob.to_numpy(float), original_prob),
                "V68 failure is not an exact V58 probability no-op")
        require(np.array_equal(final.confidence_signal.to_numpy(float), original_confidence),
                "V68 failure is not an exact V58 confidence no-op")
        require(np.array_equal(final.high_conf.to_numpy(bool), original_high),
                "V68 failure is not an exact V58 high-conf no-op")
    return final


def build_reports(
    v58: pd.DataFrame,
    oof: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    material_pass = bool(evaluation["material_pass"])
    final = final_frame(v58, oof, material_pass)
    final_summary = v44.summarize(final)
    status = evaluation["status"]
    validation = (
        "KR_NEWS-only block-expert late fusion. Direction and opportunity +/-/0 weights and "
        "opportunity threshold are selected on embargoed inner-past rows; outer OOF is untouched. "
        "No raw feature/text sign flip. KR_KIND and all other sources remain frozen. No reserved labels."
    )
    comparison = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "material_pass": material_pass,
        "selected_policy": "V68_KR_NEWS_ROUTER" if material_pass else "V58_EXACT_NOOP",
        "models": {
            "V58_BASELINE": evaluation["baseline_full_summary"],
            "V68_CANDIDATE_DIAGNOSTIC": evaluation["candidate_full_summary"],
            "V68_FAIL_CLOSED_SELECTED": final_summary,
        },
        "router_evaluation": evaluation, "input_audit": input_audit,
        "validation": validation, "seal_authorized": False,
    }
    robustness = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "opened_dev_only": True, "seal_outcomes_loaded": False,
        "dev_extension_loaded": False, "selected": final_summary,
        "candidate": evaluation["candidate_full_summary"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v58_noop": not material_pass,
        "seal_authorized": False,
    }
    transfer = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "seal_outcomes_loaded": False, "dev_extension_loaded": False,
        "selected_oof_by_source": final_summary["metrics"]["by_source_family"],
        "selected_oof_by_market": final_summary["metrics"]["by_market"],
        "kr_news_router": {
            "candidate_direction": evaluation["candidate_direction"],
            "baseline_direction": evaluation["baseline_direction"],
            "candidate_opportunity": evaluation["candidate_opportunity"],
            "baseline_opportunity": evaluation["baseline_opportunity"],
        },
        "kr_kind_unchanged": True, "method": validation,
    }
    router_report = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "scope": "KR_NEWS only", "kr_kind_unchanged": True,
        "v67_evidence": input_audit["v67_reversal_blocks"],
        "direction_coefficients": list(COEFFICIENTS),
        "opportunity_coefficients": list(COEFFICIENTS),
        "opportunity_quantiles": list(OPPORTUNITY_QUANTILES),
        "evaluation": evaluation, "nested_selection_audits": audits,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V58 prob/confidence/high_conf" if not material_pass else None,
        },
        "validation": validation,
    }
    selection_report = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "selection_space": {
            "blocks": list(BLOCKS), "direction_coefficients": list(COEFFICIENTS),
            "direction_cutoffs": list(DIRECTION_CUTOFFS),
            "opportunity_coefficients": list(COEFFICIENTS),
            "opportunity_quantiles": list(OPPORTUNITY_QUANTILES),
        },
        "folds": audits, "material_checks": evaluation["material_checks"],
    }
    risk_rows = []
    risk_rows.extend(risk_coverage(
        oof, oof.router_prob_candidate.to_numpy(float),
        oof.router_opportunity_score.to_numpy(float), "V68_ROUTER_CANDIDATE",
    ))
    risk_rows.extend(risk_coverage(
        oof, oof.prob.to_numpy(float),
        oof.confidence_signal.to_numpy(float), "V58_FROZEN_BASELINE",
    ))
    fold_rows = []
    for audit in audits:
        fold_rows.append({
            "fold": audit["fold"],
            "direction_candidate": audit["direction_policy"]["selected"]["candidate"],
            "direction_clock_weight": audit["direction_policy"]["selected"]["clock_weight"],
            "direction_text_weight": audit["direction_policy"]["selected"]["text_weight"],
            "direction_cutoff": audit["direction_policy"]["selected"]["cutoff"],
            "opportunity_candidate": audit["opportunity_policy"]["selected"]["candidate"],
            "opportunity_clock_weight": audit["opportunity_policy"]["selected"]["clock_weight"],
            "opportunity_text_weight": audit["opportunity_policy"]["selected"]["text_weight"],
            "opportunity_quantile": audit["opportunity_policy"]["selected"]["quantile"],
            "candidate_direction_ba": audit["outer_candidate_direction_metrics"]["balanced_accuracy"],
            "baseline_direction_ba": audit["outer_baseline_direction_metrics"]["balanced_accuracy"],
            "candidate_direction_auc": audit["outer_candidate_direction_metrics"]["auc"],
            "baseline_direction_auc": audit["outer_baseline_direction_metrics"]["auc"],
            "candidate_opportunity_auc": audit["outer_candidate_opportunity_metrics"]["material_move_auc"],
            "baseline_opportunity_auc": audit["outer_baseline_opportunity_metrics"]["material_move_auc"],
            "candidate_hc_net": audit["outer_candidate_opportunity_metrics"]["highconf_mean_signed_net"],
            "baseline_hc_net": audit["outer_baseline_opportunity_metrics"]["highconf_mean_signed_net"],
            "raw_feature_sign_flip_performed": False,
        })
    economic = {
        "version": "V68", "hypothesis": HYPOTHESIS, "status": status,
        "candidate": evaluation["candidate_opportunity"],
        "baseline": evaluation["baseline_opportunity"],
        "risk_coverage_rows": risk_rows,
    }
    export_columns = [
        "event_id", "event_group_id", "event_time_utc", "ticker", "source_family",
        "y", "fwd_ret_30m", "prob", "high_conf", "confidence_signal",
        "router_outer_fold", "router_prob_candidate", "router_direction_candidate",
        "router_direction_clock_weight", "router_direction_text_weight",
        "router_direction_inner_cutoff", "router_opportunity_score",
        "baseline_opportunity_score", "router_opportunity_high",
        "baseline_opportunity_high", "router_opportunity_candidate",
        "router_opportunity_clock_weight", "router_opportunity_text_weight",
        "router_opportunity_inner_quantile", "material_move",
    ]
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "POLARITY_ROUTER_SELECTION_REPORT.json": selection_report,
        "KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_REPORT.json": router_report,
        "ECONOMIC_ROBUSTNESS_REPORT.json": economic,
    }
    table_reports = {
        "KR_NEWS_ROUTER_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "RISK_COVERAGE_REPORT.csv": pd.DataFrame(risk_rows),
        "KR_NEWS_ROUTER_OOF.csv.gz": oof[export_columns].copy(),
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
    input_audit = verify_inputs()
    v58, kr_news = load_frames()
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "input_audit": input_audit, "kr_news_n": len(kr_news),
            "kr_kind_unchanged": True, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    router_oof, audits = run_outer_oof(kr_news, smoke=args.smoke_test)
    evaluation = evaluate(v58, router_oof, audits)
    if args.smoke_test:
        print(json.dumps(json_clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS,
            "outer_oof_n": len(router_oof),
            "direction_policy": audits[0]["direction_policy"]["selected"],
            "opportunity_policy": audits[0]["opportunity_policy"]["selected"],
            "kr_kind_unchanged": True, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    json_reports, table_reports = build_reports(
        v58, router_oof, audits, evaluation, input_audit,
    )
    json_reports = {name: json_clean(payload) for name, payload in json_reports.items()}
    output_names = sorted([*json_reports, *table_reports])
    summary = {
        "material_status": evaluation["status"],
        "kr_news_candidate_ba": evaluation["candidate_direction"]["balanced_accuracy"],
        "kr_news_baseline_ba": evaluation["baseline_direction"]["balanced_accuracy"],
        "kr_news_candidate_auc": evaluation["candidate_direction"]["auc"],
        "kr_news_baseline_auc": evaluation["baseline_direction"]["auc"],
        "candidate_hc_net": evaluation["candidate_opportunity"]["highconf_mean_signed_net"],
        "baseline_hc_net": evaluation["baseline_opportunity"]["highconf_mean_signed_net"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v58_noop": not evaluation["material_pass"],
    }
    if args.dry_run:
        print(json.dumps(json_clean({
            "status": "DRY_RUN_OK", "hypothesis": HYPOTHESIS,
            **summary, "kr_kind_unchanged": True,
            "would_write": output_names, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return
    for name, payload in json_reports.items():
        v37.atomic_json(payload, OUT / name)
    for name, payload in table_reports.items():
        atomic_csv(payload, OUT / name)
    print(json.dumps(json_clean({
        "status": evaluation["status"], "hypothesis": HYPOTHESIS,
        "output_dir": str(OUT), "files": output_names, **summary,
        "kr_kind_unchanged": True, "seal_authorized": False,
    }), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
