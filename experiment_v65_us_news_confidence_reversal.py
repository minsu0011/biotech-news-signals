"""V65: fail-closed US_NEWS confidence-reversal guard.

The V58 direction probability and prediction are immutable in this experiment.
Only confidence is researched, using the correctness of the already outer-OOF
V58 direction as the meta-target.  Candidate selection and high-confidence
thresholds are inner-past only.  The two V64 blocks with 100% stable negative
US_NEWS contribution are used as late expert-contribution gates; raw features
and raw feature signs are never inverted.

If the predeclared outer-OOF improvement checks do not all pass, V65 records a
material experiment failure and returns the original V58 confidence policy as
a no-op.  No seal, final reserve, or DEV extension is opened.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
V64_EVIDENCE = ROOT / "staging" / "V64_FEATURE_POLARITY_DIAGNOSTIC_V1"
OUT = Path(os.environ.get(
    "MARKET_BIO_VERSION_OUTPUT",
    str(ROOT / "staging" / "V65_US_NEWS_CONFIDENCE_REVERSAL_GUARD_V1"),
))

DEV_PATH = DATA / "dev_contract_v36_labeled.csv.gz"
V58_OOF_PATH = CACHE / "v58_locked_source_routing_oof.csv.gz"
EXPECTED_INPUT_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V58_OOF_PATH: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
}
EXPECTED_V64_EVIDENCE_SHA256 = {
    "CONFIDENCE_BIN_REPORT.csv": "69d4ee5eba09ff436bc5195d605e9c18a31419bd0fe1897788b78c5bc1c6950c",
    "CONFIDENCE_RELIABILITY_REPORT.json": "76cbc01b46ae3dd1e494af12c0df5946fab303b9d0d8dd33b0e7058ec811b8c1",
    "EXTREME_REVERSAL_REPORT.json": "92ee67f7ddef99c41bdb0b4a2f0b6f09319f07638b8d605e7a7345db378b8dde",
    "FEATURE_BLOCK_POLARITY_AUDIT.csv": "3cc7df9e98b73fa3abe2822168fe9d45f7e86d067991032a4bcca2c52e7d5651",
    "POLARITY_STABILITY_REPORT.csv": "ae0f20278271700bedd91243c1c3444ffee4e3cbae4e556f05d160b1deefae41",
    "SOURCE_POLARITY_AUDIT.csv": "bd050651babc47ed151ecddcbda36dda2b475a8ab945d24774b07561d4430c7f",
    "research/FEATURE_BLOCK_REGISTRY.json": "e34fcb63993dc774b2d2df7ac2a4103f438df05cc094a6cd423885185b91b290",
}

EXPECTED_DEV_ROWS = 9462
EXPECTED_V58_ROWS = 7568
EXPECTED_US_NEWS_ROWS = 954
HYPOTHESIS = "US_NEWS_CONFIDENCE_REVERSAL_GUARD_V1"
FAMILIES = ("CONFIDENCE_META_MODEL", "EXTREME_CONFIDENCE_REVERSAL")
EMBARGO = pd.Timedelta(minutes=35)
COST = float(v36.COST)
SEED = 6501
OUTER_BLOCKS = 6
HC_QUANTILES = (0.70, 0.80, 0.85, 0.90)

NEGATIVE_BLOCKS = {
    "categorical_context": {
        "kind": "categorical",
        "columns": ["market", "source", "source_family", "event_type", "form"],
    },
    "volatility_range": {
        "kind": "numeric",
        "columns": [
            "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
            "range_5m", "range_10m", "range_30m", "range_60m",
            "entry_bar_range", "abs_pre_ret_2", "abs_pre_ret_5",
            "abs_pre_ret_15", "abs_pre_ret_30", "abs_pre_ret_60",
        ],
    },
}

META_RAW_COLUMNS = [
    "prob", "confidence_signal", "opportunity_confidence",
    "confidence_opportunity_direction", "confidence_event_form_direction",
    "confidence_fixed_numeric_direction", "prob_opportunity_direction",
    "prob_event_form_direction", "prob_fixed_numeric_direction",
]
CANDIDATE_FAMILY = {
    "V58_FROZEN_CONFIDENCE": "BASELINE",
    "ABS_DIRECTION_MARGIN": "BASELINE",
    "OPPORTUNITY_CONFIDENCE_BLEND": "OPPORTUNITY_BASELINE",
    "CONFIDENCE_META_MODEL_ZERO": "CONFIDENCE_META_MODEL",
    "EXTREME_CONFIDENCE_REVERSAL_PENALTY": "EXTREME_CONFIDENCE_REVERSAL",
    "EXTREME_CONFIDENCE_REVERSAL_INVERSE": "EXTREME_CONFIDENCE_REVERSAL",
}
MATERIAL_FAMILIES = set(FAMILIES)


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


def verify_inputs_and_evidence() -> dict[str, Any]:
    input_hashes = {}
    for path, expected in EXPECTED_INPUT_SHA256.items():
        require(path.is_file(), f"required V65 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V65 input changed: {path.name}")
        input_hashes[str(path.relative_to(ROOT))] = actual
    evidence_hashes = {}
    for relative, expected in EXPECTED_V64_EVIDENCE_SHA256.items():
        path = V64_EVIDENCE / relative
        require(path.is_file(), f"required V64 evidence missing: {relative}")
        actual = sha256(path)
        require(actual == expected, f"pinned V64 evidence changed: {relative}")
        evidence_hashes[relative] = actual

    reversal = json.loads(
        (V64_EVIDENCE / "EXTREME_REVERSAL_REPORT.json").read_text(encoding="utf-8")
    )
    require(reversal.get("hypothesis") == "FEATURE_POLARITY_DIAGNOSTIC_V1",
            "V64 reversal evidence hypothesis mismatch")
    require(reversal.get("us_news_high_confidence_reversal") is True,
            "V64 no longer establishes US_NEWS confidence reversal")
    stability = pd.read_csv(V64_EVIDENCE / "POLARITY_STABILITY_REPORT.csv")
    observed_negative = {}
    for block in NEGATIVE_BLOCKS:
        row = stability[
            stability.source_family.eq("US_NEWS") & stability.block.eq(block)
        ]
        require(len(row) == 1, f"V64 polarity evidence missing for {block}")
        item = row.iloc[0]
        require(str(item.polarity) == "-", f"V64 polarity is not negative for {block}")
        require(float(item.negative_fraction) == 1.0,
                f"V64 polarity is not 100% stable for {block}")
        observed_negative[block] = {
            "polarity": "-", "negative_fraction": 1.0,
            "fold_n": int(item.fold_n),
            "median_combined_contribution": float(item.median_combined_contribution),
        }
    registry = json.loads(
        (V64_EVIDENCE / "research" / "FEATURE_BLOCK_REGISTRY.json").read_text(encoding="utf-8")
    )
    for block, spec in NEGATIVE_BLOCKS.items():
        registered = registry["blocks"][block]
        require(registered["kind"] == spec["kind"], f"V64 block kind changed: {block}")
        require(registered["columns"] == spec["columns"], f"V64 block columns changed: {block}")
    return {
        "input_sha256": input_hashes,
        "v64_evidence_sha256": evidence_hashes,
        "v64_us_news_high_confidence_reversal": True,
        "v64_stable_negative_blocks": observed_negative,
        "authorized_data_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(V58_OOF_PATH.relative_to(ROOT)),
        ],
        "extension_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
    }


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    oof = pd.read_csv(
        V58_OOF_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_DEV_ROWS, f"V36 DEV row count changed: {len(dev)}")
    require(len(oof) == EXPECTED_V58_ROWS, f"V58 OOF row count changed: {len(oof)}")
    require(dev.event_id.is_unique and oof.event_id.is_unique, "input event_id is not unique")
    require(set(oof.event_id).issubset(set(dev.event_id)), "V58 OOF is not a V36 DEV subset")
    aligned = dev.set_index("event_id").loc[oof.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), oof.y.to_numpy(int)),
            "V58 and V36 labels differ")
    require(np.allclose(aligned.fwd_ret_30m.to_numpy(float),
                        oof.fwd_ret_30m.to_numpy(float), rtol=0.0, atol=1e-15),
            "V58 and V36 forward returns differ")
    for column in oof.columns:
        if column not in aligned.columns or column in {
            "prob", "high_conf", "confidence_signal", "fold", "family",
            "new_issuer", "text_cluster_id", "cluster_weight", "routed_from",
        }:
            aligned[column] = oof[column].to_numpy()
    aligned["event_time_utc"] = pd.to_datetime(aligned.event_time_utc, utc=True)
    aligned["v58_high_conf"] = aligned.high_conf.map(
        lambda value: value if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes"}
    ).astype(bool)
    aligned["v58_probability_copy"] = aligned.prob.to_numpy(float)
    aligned["correct_target"] = (
        (aligned.prob.to_numpy(float) >= 0.5) == aligned.y.to_numpy(int)
    ).astype(int)
    # Reconstruct duplicate controls from immutable raw DEV, not the anomalous
    # historical OOF text-cluster serialization.
    aligned["text_cluster_id"] = aligned.apply(v37.normalized_cluster, axis=1)
    aligned["cluster_weight"] = 1.0 / aligned.groupby("text_cluster_id").event_id.transform("size")
    us_news = aligned[aligned.source_family.eq("US_NEWS")].copy()
    require(len(us_news) == EXPECTED_US_NEWS_ROWS,
            f"US_NEWS OOF row count changed: {len(us_news)}")
    require(us_news.correct_target.nunique() == 2, "US_NEWS correctness target has one class")
    require(np.array_equal(aligned.prob.to_numpy(float), aligned.v58_probability_copy.to_numpy(float)),
            "V65 changed direction probabilities during input assembly")
    return aligned, us_news


def chronological_outer_folds(frame: pd.DataFrame, smoke: bool = False) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    ordered = frame.sort_values("event_time_utc").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(ordered)), OUTER_BLOCKS)
    folds = []
    for fold in range(1, OUTER_BLOCKS):
        valid = ordered.iloc[blocks[fold]].copy()
        boundary = valid.event_time_utc.min()
        train = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
        train = train[~train.text_cluster_id.isin(set(valid.text_cluster_id))].copy()
        if len(train) >= 120 and len(valid) >= 100 and train.correct_target.nunique() == 2:
            folds.append((train, valid, fold))
    require(len(folds) == OUTER_BLOCKS - 1,
            f"V65 eligible outer folds changed: {len(folds)}")
    return folds[:1] if smoke else folds


def inner_past_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values("event_time_utc").reset_index(drop=True)
    split = max(100, int(0.72 * len(ordered)))
    valid = ordered.iloc[split:].copy()
    boundary = valid.event_time_utc.min()
    base = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
    base = base[~base.text_cluster_id.isin(set(valid.text_cluster_id))].copy()
    require(len(base) >= 80 and len(valid) >= 30, "V65 inner split is too small")
    require(base.correct_target.nunique() == 2 and valid.correct_target.nunique() == 2,
            "V65 inner correctness classes are insufficient")
    return base, valid


def time_audit(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label} violates strict 35-minute embargo")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end.isoformat(), "valid_start": valid_start.isoformat(),
        "strict_35m_embargo_verified": True,
    }


def meta_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[META_RAW_COLUMNS].apply(pd.to_numeric, errors="coerce").copy()
    x["direction_margin"] = np.abs(x.prob - 0.5)
    x["predicted_up"] = (x.prob >= 0.5).astype(float)
    base_up = x.prob >= 0.5
    auxiliary = [
        ("opportunity", "prob_opportunity_direction", "confidence_opportunity_direction"),
        ("event_form", "prob_event_form_direction", "confidence_event_form_direction"),
        ("fixed_numeric", "prob_fixed_numeric_direction", "confidence_fixed_numeric_direction"),
    ]
    agreement_strength = []
    for name, probability, confidence in auxiliary:
        agree = np.where((x[probability] >= 0.5) == base_up, 1.0, -1.0)
        x[f"{name}_agreement"] = agree
        x[f"{name}_agreement_strength"] = agree * x[confidence]
        agreement_strength.append(x[f"{name}_agreement_strength"].to_numpy(float))
    x["mean_auxiliary_agreement_strength"] = np.mean(agreement_strength, axis=0)
    return x.replace([np.inf, -np.inf], np.nan)


def fit_meta_probability(train: pd.DataFrame, target: pd.DataFrame, seed: int) -> np.ndarray:
    x_train = meta_feature_frame(train)
    x_target = meta_feature_frame(target)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_array = imputer.fit_transform(x_train)
    target_array = imputer.transform(x_target)
    scaler = StandardScaler()
    train_array = scaler.fit_transform(train_array)
    target_array = scaler.transform(target_array)
    model = LogisticRegression(
        C=0.30, solver="liblinear", max_iter=700, class_weight="balanced",
        random_state=seed,
    )
    model.fit(
        train_array, train.correct_target.to_numpy(int),
        sample_weight=train.cluster_weight.to_numpy(float),
    )
    return np.clip(model.predict_proba(target_array)[:, 1], 1e-6, 1.0 - 1e-6)


def volatility_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for suffix in ("2", "5", "15", "30", "60"):
        x[f"abs_pre_ret_{suffix}"] = pd.to_numeric(
            x[f"pre_ret_{suffix}"], errors="coerce",
        ).abs()
    columns = NEGATIVE_BLOCKS["volatility_range"]["columns"]
    return x[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_block_probability(train: pd.DataFrame, target: pd.DataFrame, block: str, seed: int) -> np.ndarray:
    spec = NEGATIVE_BLOCKS[block]
    if spec["kind"] == "numeric":
        x_train = volatility_frame(train)
        x_target = volatility_frame(target)
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        train_array = imputer.fit_transform(x_train)
        target_array = imputer.transform(x_target)
        scaler = StandardScaler(with_mean=False)
        train_matrix = sparse.csr_matrix(scaler.fit_transform(train_array), dtype=np.float32)
        target_matrix = sparse.csr_matrix(scaler.transform(target_array), dtype=np.float32)
    else:
        columns = spec["columns"]
        x_train = train[columns].fillna("").astype(str)
        x_target = target[columns].fillna("").astype(str)
        encoder = OneHotEncoder(
            handle_unknown="ignore", min_frequency=3, sparse_output=True,
            dtype=np.float32,
        )
        train_matrix = encoder.fit_transform(x_train).tocsr()
        target_matrix = encoder.transform(x_target).tocsr()
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=3e-4, max_iter=1200,
        tol=1e-3, average=True, class_weight="balanced", random_state=seed,
    )
    model.fit(
        train_matrix, train.y.to_numpy(int),
        sample_weight=train.cluster_weight.to_numpy(float),
    )
    return np.clip(model.predict_proba(target_matrix)[:, 1], 1e-6, 1.0 - 1e-6)


def late_block_contributions(train: pd.DataFrame, target: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    base_up = target.prob.to_numpy(float) >= 0.5
    contributions = {}
    for block_index, block in enumerate(NEGATIVE_BLOCKS):
        probability = fit_block_probability(train, target, block, seed + block_index)
        agreement = np.where((probability >= 0.5) == base_up, 1.0, -1.0)
        # This is a late expert agreement contribution.  No input feature and no
        # direction probability is sign-flipped.
        contributions[block] = agreement * (2.0 * np.abs(probability - 0.5))
    contributions["mean_negative_block_contribution"] = np.mean(
        np.vstack([contributions[block] for block in NEGATIVE_BLOCKS]), axis=0,
    )
    return contributions


def candidate_scores(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    meta_probability: np.ndarray,
    contributions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    margin_reference = np.abs(reference.prob.to_numpy(float) - 0.5)
    margin_target = np.abs(target.prob.to_numpy(float) - 0.5)
    opportunity_reference = reference.opportunity_confidence.to_numpy(float)
    opportunity_target = target.opportunity_confidence.to_numpy(float)
    frozen_reference = reference.confidence_signal.to_numpy(float)
    frozen_target = target.confidence_signal.to_numpy(float)
    opportunity_blend = (
        0.40 * v37.rank_against(frozen_reference, frozen_target)
        + 0.30 * v37.rank_against(margin_reference, margin_target)
        + 0.30 * v37.rank_against(opportunity_reference, opportunity_target)
    )
    contribution = contributions["mean_negative_block_contribution"]
    meta_logit = logit(np.clip(meta_probability, 1e-6, 1.0 - 1e-6))
    return {
        "V58_FROZEN_CONFIDENCE": frozen_target,
        "ABS_DIRECTION_MARGIN": margin_target,
        "OPPORTUNITY_CONFIDENCE_BLEND": opportunity_blend,
        "CONFIDENCE_META_MODEL_ZERO": meta_probability,
        "EXTREME_CONFIDENCE_REVERSAL_PENALTY": expit(
            meta_logit - np.maximum(contribution, 0.0)
        ),
        "EXTREME_CONFIDENCE_REVERSAL_INVERSE": expit(meta_logit - contribution),
    }


def candidate_inner_policy(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    correctness = frame.correct_target.to_numpy(int)
    auc = safe_auc(correctness, score)
    score_rank = v37.rank_self(score)
    base_accuracy = float(correctness.mean())
    rows = []
    for quantile in HC_QUANTILES:
        high = score_rank >= quantile
        accuracy = float(correctness[high].mean())
        lift = accuracy - base_accuracy
        selection_score = float(auc or 0.5) + 0.50 * accuracy + 0.25 * lift
        rows.append({
            "quantile": quantile, "highconf_n": int(high.sum()),
            "highconf_coverage": float(high.mean()), "highconf_accuracy": accuracy,
            "highconf_correctness_lift": lift, "selection_score": selection_score,
        })
    rows.sort(key=lambda row: (
        row["selection_score"], -abs(row["quantile"] - 0.80),
    ), reverse=True)
    return {
        "confidence_correctness_auc": auc,
        "base_correctness_rate": base_accuracy,
        "selected_threshold": rows[0],
        "threshold_candidates": rows,
    }


def run_nested_oof(frame: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs = []
    audits = []
    for train, valid, fold in chronological_outer_folds(frame, smoke=smoke):
        inner_train, inner_valid = inner_past_split(train)
        inner_time = time_audit(inner_train, inner_valid, f"V65 inner fold {fold}")
        outer_time = time_audit(train, valid, f"V65 outer fold {fold}")

        inner_meta = fit_meta_probability(inner_train, inner_valid, SEED + fold)
        inner_contributions = late_block_contributions(
            inner_train, inner_valid, SEED + 100 + fold,
        )
        inner_scores = candidate_scores(
            inner_train, inner_valid, inner_meta, inner_contributions,
        )
        policies = {
            name: candidate_inner_policy(inner_valid, score)
            for name, score in inner_scores.items()
        }
        selection = sorted(
            policies,
            key=lambda name: (
                policies[name]["selected_threshold"]["selection_score"],
                name == "CONFIDENCE_META_MODEL_ZERO",
            ),
            reverse=True,
        )[0]

        outer_meta = fit_meta_probability(train, valid, SEED + 1000 + fold)
        outer_contributions = late_block_contributions(
            train, valid, SEED + 1100 + fold,
        )
        outer_scores = candidate_scores(train, valid, outer_meta, outer_contributions)
        output = valid[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "y", "fwd_ret_30m", "prob", "v58_high_conf",
            "confidence_signal", "correct_target", "cluster_weight",
        ]].copy()
        output["meta_outer_fold"] = int(fold)
        output["selected_candidate"] = selection
        output["selected_family"] = CANDIDATE_FAMILY[selection]
        for name in CANDIDATE_FAMILY:
            normalized = v37.rank_against(inner_scores[name], outer_scores[name])
            quantile = float(policies[name]["selected_threshold"]["quantile"])
            output[f"confidence__{name}"] = normalized
            output[f"high__{name}"] = normalized >= quantile
        output["selected_confidence"] = output[f"confidence__{selection}"].to_numpy(float)
        output["selected_high_conf"] = output[f"high__{selection}"].to_numpy(bool)
        for block in NEGATIVE_BLOCKS:
            output[f"late_contribution__{block}"] = outer_contributions[block]
        outputs.append(output)
        audits.append({
            "fold": int(fold), "outer": outer_time, "inner": inner_time,
            "selected_candidate": selection,
            "selected_family": CANDIDATE_FAMILY[selection],
            "candidate_policies": policies,
            "direction_probability_changed": False,
            "raw_feature_sign_flip_performed": False,
        })
        print(
            f"[V65 CONFIDENCE] fold={fold} train={len(train)} valid={len(valid)} "
            f"selected={selection} q={policies[selection]['selected_threshold']['quantile']:.2f}",
            flush=True,
        )
    oof = pd.concat(outputs, ignore_index=True)
    original = frame.set_index("event_id").loc[oof.event_id]
    require(np.array_equal(oof.prob.to_numpy(float), original.prob.to_numpy(float)),
            "V65 changed V58 direction probabilities")
    require(np.array_equal(oof.prob.ge(0.5).to_numpy(), original.prob.ge(0.5).to_numpy()),
            "V65 changed V58 direction predictions")
    return oof, audits


def confidence_metrics(frame: pd.DataFrame, score: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    correctness = frame.correct_target.to_numpy(int)
    prediction = frame.prob.to_numpy(float) >= 0.5
    signed = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    high = np.asarray(high, bool)
    return {
        "n": len(frame),
        "confidence_correctness_auc": safe_auc(correctness, np.asarray(score, float)),
        "direction_accuracy": float(correctness.mean()),
        "direction_balanced_accuracy": float(balanced_accuracy_score(frame.y, prediction)),
        "direction_auc": safe_auc(frame.y.to_numpy(int), frame.prob.to_numpy(float)),
        "highconf_n": int(high.sum()),
        "highconf_coverage": float(high.mean()),
        "highconf_accuracy": float(correctness[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(signed[high].mean()) if high.any() else None,
        "all_mean_signed_net": float(signed.mean()),
    }


def risk_coverage_rows(frame: pd.DataFrame, score: np.ndarray, name: str) -> list[dict[str, Any]]:
    score = np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    rows = []
    for coverage in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00):
        count = max(1, int(math.ceil(len(frame) * coverage)))
        selected = frame.iloc[order[:count]]
        metrics = confidence_metrics(
            selected, score[order[:count]], np.ones(count, dtype=bool),
        )
        rows.append({
            "confidence_policy": name, "target_coverage": coverage,
            "actual_coverage": float(count / len(frame)),
            "n": count, "accuracy": metrics["direction_accuracy"],
            "risk": float(1.0 - metrics["direction_accuracy"]),
            "balanced_accuracy": metrics["direction_balanced_accuracy"],
            "mean_signed_net": metrics["highconf_mean_signed_net"],
            "mean_confidence": float(score[order[:count]].mean()),
        })
    return rows


def bootstrap_comparison(frame: pd.DataFrame, nboot: int = 500) -> dict[str, Any]:
    x = frame.copy().reset_index(drop=True)
    x["day"] = pd.to_datetime(x.event_time_utc, utc=True).dt.date
    groups = {day: np.asarray(index, dtype=int) for day, index in x.groupby("day").indices.items()}
    days = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(SEED + 99)
    auc_delta = []
    accuracy_delta = []
    net_delta = []
    for _ in range(nboot):
        positions = np.concatenate([groups[day] for day in rng.choice(days, len(days), replace=True)])
        sample = x.iloc[positions]
        correct = sample.correct_target.to_numpy(int)
        selected_auc = safe_auc(correct, sample.selected_confidence.to_numpy(float))
        baseline_auc = safe_auc(
            correct, sample["confidence__V58_FROZEN_CONFIDENCE"].to_numpy(float),
        )
        if selected_auc is not None and baseline_auc is not None:
            auc_delta.append(selected_auc - baseline_auc)
        selected_high = sample.selected_high_conf.to_numpy(bool)
        baseline_high = sample["high__V58_FROZEN_CONFIDENCE"].to_numpy(bool)
        if selected_high.any() and baseline_high.any():
            accuracy_delta.append(
                float(correct[selected_high].mean() - correct[baseline_high].mean())
            )
            prediction = sample.prob.to_numpy(float) >= 0.5
            signed = np.where(prediction, 1.0, -1.0) * sample.fwd_ret_30m.to_numpy(float) - COST
            net_delta.append(float(signed[selected_high].mean() - signed[baseline_high].mean()))

    def interval(values: list[float]) -> dict[str, Any]:
        return {
            "n_boot": len(values),
            "mean": float(np.mean(values)) if values else None,
            "lower95": float(np.quantile(values, 0.025)) if values else None,
            "upper95": float(np.quantile(values, 0.975)) if values else None,
        }
    return {
        "resampling_unit": "event_day",
        "confidence_correctness_auc_delta": interval(auc_delta),
        "highconf_accuracy_delta": interval(accuracy_delta),
        "highconf_mean_signed_net_delta": interval(net_delta),
    }


def evaluate(oof: pd.DataFrame, audits: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {}
    for name in CANDIDATE_FAMILY:
        candidates[name] = confidence_metrics(
            oof, oof[f"confidence__{name}"].to_numpy(float),
            oof[f"high__{name}"].to_numpy(bool),
        )
        candidates[name]["family"] = CANDIDATE_FAMILY[name]
    selected = confidence_metrics(
        oof, oof.selected_confidence.to_numpy(float), oof.selected_high_conf.to_numpy(bool),
    )
    selected["selected_candidate_by_fold"] = {
        str(key): str(value) for key, value in oof.groupby("meta_outer_fold").selected_candidate.first().items()
    }
    frozen_original = confidence_metrics(
        oof, oof.confidence_signal.to_numpy(float), oof.v58_high_conf.to_numpy(bool),
    )
    baseline = candidates["V58_FROZEN_CONFIDENCE"]
    fold_deltas = []
    for fold, group in oof.groupby("meta_outer_fold"):
        selected_auc = safe_auc(group.correct_target.to_numpy(int), group.selected_confidence.to_numpy(float))
        baseline_auc = safe_auc(
            group.correct_target.to_numpy(int),
            group["confidence__V58_FROZEN_CONFIDENCE"].to_numpy(float),
        )
        fold_deltas.append({
            "fold": int(fold), "selected_auc": selected_auc, "baseline_auc": baseline_auc,
            "auc_delta": None if selected_auc is None or baseline_auc is None else selected_auc - baseline_auc,
            "selected_candidate": str(group.selected_candidate.iloc[0]),
        })
    positive_fraction = float(np.mean([
        row["auc_delta"] is not None and row["auc_delta"] > 0 for row in fold_deltas
    ]))
    material_fraction = float(np.mean([
        audit["selected_family"] in MATERIAL_FAMILIES for audit in audits
    ]))
    checks = {
        "confidence_auc_delta_ge_0_010": (
            selected["confidence_correctness_auc"]
            >= baseline["confidence_correctness_auc"] + 0.010
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
        "risk_coverage_between_0_10_and_0_35": 0.10 <= selected["highconf_coverage"] <= 0.35,
        "material_candidate_fold_fraction_ge_0_60": material_fraction >= 0.60,
    }
    material_pass = bool(all(checks.values()))
    risk_rows = []
    risk_rows.extend(risk_coverage_rows(
        oof, oof.selected_confidence.to_numpy(float), "INNER_SELECTED_META_GUARD",
    ))
    risk_rows.extend(risk_coverage_rows(
        oof, oof["confidence__V58_FROZEN_CONFIDENCE"].to_numpy(float),
        "V58_FROZEN_CONFIDENCE_INNER_THRESHOLD",
    ))
    risk_rows.extend(risk_coverage_rows(
        oof, np.abs(oof.prob.to_numpy(float) - 0.5), "ABS_DIRECTION_MARGIN",
    ))
    return {
        "eligible_oof_n": len(oof),
        "eligible_oof_coverage_of_us_news": float(len(oof) / EXPECTED_US_NEWS_ROWS),
        "burn_in_n": int(EXPECTED_US_NEWS_ROWS - len(oof)),
        "candidate_metrics": candidates,
        "inner_selected_metrics": selected,
        "v58_original_mask_metrics": frozen_original,
        "v58_inner_threshold_baseline_metrics": baseline,
        "fold_auc_deltas": fold_deltas,
        "positive_fold_auc_fraction": positive_fraction,
        "material_candidate_fold_fraction": material_fraction,
        "material_checks": checks,
        "material_pass": material_pass,
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_NOOP",
        "bootstrap": bootstrap_comparison(oof),
        "risk_coverage_rows": risk_rows,
    }


def final_policy_frame(all_oof: pd.DataFrame, meta_oof: pd.DataFrame, material_pass: bool) -> pd.DataFrame:
    final = all_oof.copy()
    final["prob_before_v65"] = final.prob.to_numpy(float)
    final["confidence_before_v65"] = final.confidence_signal.to_numpy(float)
    final["high_conf_before_v65"] = final.v58_high_conf.to_numpy(bool)
    if material_pass:
        lookup = meta_oof.set_index("event_id")
        eligible = final.event_id.isin(lookup.index) & final.source_family.eq("US_NEWS")
        ids = final.loc[eligible, "event_id"]
        final.loc[eligible, "confidence_signal"] = lookup.loc[ids, "selected_confidence"].to_numpy(float)
        final.loc[eligible, "high_conf"] = lookup.loc[ids, "selected_high_conf"].to_numpy(bool)
        final["family"] = "V65_US_NEWS_CONFIDENCE_REVERSAL_GUARD"
    else:
        final["confidence_signal"] = final.confidence_before_v65.to_numpy(float)
        final["high_conf"] = final.high_conf_before_v65.to_numpy(bool)
        final["family"] = "V58_LOCKED_SOURCE_DGP_ROUTING_NOOP"
    require(np.array_equal(final.prob.to_numpy(float), final.prob_before_v65.to_numpy(float)),
            "V65 final policy changed direction probability")
    require(np.array_equal(final.prob.ge(0.5).to_numpy(), final.prob_before_v65.ge(0.5).to_numpy()),
            "V65 final policy changed direction prediction")
    if not material_pass:
        require(np.array_equal(final.confidence_signal.to_numpy(float),
                               final.confidence_before_v65.to_numpy(float)),
                "V65 fail-closed confidence is not a V58 no-op")
        require(np.array_equal(final.high_conf.to_numpy(bool),
                               final.high_conf_before_v65.to_numpy(bool)),
                "V65 fail-closed mask is not a V58 no-op")
    return final


def output_reports(
    all_oof: pd.DataFrame,
    meta_oof: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    material_pass = bool(evaluation["material_pass"])
    final = final_policy_frame(all_oof, meta_oof, material_pass)
    v58_summary = v44.summarize(all_oof)
    final_summary = v44.summarize(final)
    status = evaluation["status"]
    validation = (
        "V58 direction probabilities/predictions are immutable. US_NEWS P(correct) and both "
        "negative-block late experts are fit on past rows only with strict 35-minute outer and "
        "inner embargoes. Candidate and HC threshold selection use inner validation only. Raw "
        "features are never sign-flipped. Seal/final/DEV_EXTENSION are not read."
    )
    comparison = {
        "version": "V65", "hypothesis": HYPOTHESIS, "families": list(FAMILIES),
        "status": status, "material_pass": material_pass,
        "selected_policy": (
            "V65_US_NEWS_CONFIDENCE_REVERSAL_GUARD" if material_pass
            else "V58_LOCKED_SOURCE_DGP_ROUTING_NOOP"
        ),
        "direction_probability_changed": False,
        "direction_prediction_changed": False,
        "models": {
            "V58_BASELINE": v58_summary,
            "V65_FAIL_CLOSED_SELECTED": final_summary,
        },
        "us_news_confidence_evaluation": evaluation,
        "input_audit": input_audit,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V65", "hypothesis": HYPOTHESIS, "status": status,
        "opened_dev_only": True, "seal_outcomes_loaded": False,
        "extension_loaded": False, "selected": final_summary,
        "baseline": v58_summary, "material_checks": evaluation["material_checks"],
        "bootstrap": evaluation["bootstrap"], "fallback_is_exact_v58_noop": not material_pass,
        "direction_probability_changed": False, "seal_authorized": False,
    }
    transfer = {
        "version": "V65", "hypothesis": HYPOTHESIS, "status": status,
        "seal_outcomes_loaded": False, "extension_loaded": False,
        "selected_oof_by_source": final_summary["metrics"]["by_source_family"],
        "selected_oof_by_market": final_summary["metrics"]["by_market"],
        "us_news_confidence": {
            "eligible_oof_n": evaluation["eligible_oof_n"],
            "inner_selected": evaluation["inner_selected_metrics"],
            "v58_baseline": evaluation["v58_inner_threshold_baseline_metrics"],
            "risk_coverage": evaluation["risk_coverage_rows"],
        },
        "method": validation,
    }
    diagnostic = {
        "version": "V65", "hypothesis": HYPOTHESIS, "status": status,
        "families": list(FAMILIES), "input_audit": input_audit,
        "negative_block_guard": {
            "blocks": NEGATIVE_BLOCKS,
            "candidate_modes": ["zero", "positive-contribution penalty", "late-contribution inverse"],
            "raw_feature_sign_flip_allowed": False,
            "raw_feature_sign_flip_performed": False,
        },
        "evaluation": evaluation,
        "nested_selection_audits": audits,
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V58 confidence_signal and high_conf no-op" if not material_pass else None,
        },
        "validation": validation,
    }
    meta_model = {
        "version": "V65", "hypothesis": HYPOTHESIS,
        "family": "CONFIDENCE_META_MODEL",
        "target": "correct_target = ((V58 prob >= 0.5) == y)",
        "target_n": EXPECTED_US_NEWS_ROWS,
        "outer_oof_n": len(meta_oof),
        "meta_features": list(meta_feature_frame(all_oof.head(1)).columns),
        "candidate_family": CANDIDATE_FAMILY,
        "selection_audits": audits,
        "direction_probability_changed": False,
    }
    comparison_rows = []
    for candidate, metrics in evaluation["candidate_metrics"].items():
        comparison_rows.append({"candidate": candidate, **metrics})
    comparison_rows.extend([
        {"candidate": "INNER_SELECTED_META_GUARD", **evaluation["inner_selected_metrics"]},
        {"candidate": "V58_ORIGINAL_HIGH_CONF_MASK", **evaluation["v58_original_mask_metrics"]},
    ])
    meta_export_columns = [
        "event_id", "event_group_id", "event_time_utc", "ticker", "source_family",
        "y", "fwd_ret_30m", "prob", "correct_target", "meta_outer_fold",
        "selected_candidate", "selected_family", "selected_confidence",
        "selected_high_conf", "confidence_signal", "v58_high_conf",
    ] + [f"late_contribution__{block}" for block in NEGATIVE_BLOCKS]
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "US_NEWS_CONFIDENCE_REVERSAL_REPORT.json": diagnostic,
        "CONFIDENCE_META_MODEL_REPORT.json": meta_model,
    }
    table_reports = {
        "CONFIDENCE_COMPARISON.csv": pd.DataFrame(comparison_rows),
        "RISK_COVERAGE_REPORT.csv": pd.DataFrame(evaluation["risk_coverage_rows"]),
        "US_NEWS_CONFIDENCE_META_OOF.csv.gz": meta_oof[meta_export_columns].copy(),
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
    modes.add_argument("--audit-only", action="store_true",
                       help="verify pinned input/evidence only; do not fit or write")
    modes.add_argument("--smoke-test", action="store_true",
                       help="run one outer fold; do not write")
    modes.add_argument("--dry-run", action="store_true",
                       help="run all folds and reports; do not write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    input_audit = verify_inputs_and_evidence()
    all_oof, us_news = load_frame()
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "input_audit": input_audit, "us_news_n": len(us_news),
            "direction_probability_changed": False, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return

    meta_oof, audits = run_nested_oof(us_news, smoke=args.smoke_test)
    evaluation = evaluate(meta_oof, audits)
    if args.smoke_test:
        print(json.dumps(json_clean({
            "status": "SMOKE_OK", "hypothesis": HYPOTHESIS,
            "outer_oof_n": len(meta_oof),
            "selected_candidate": audits[0]["selected_candidate"],
            "selected_metrics": evaluation["inner_selected_metrics"],
            "direction_probability_changed": False, "output_written": False,
        }), indent=2, ensure_ascii=False))
        return

    json_reports, table_reports = output_reports(
        all_oof, meta_oof, audits, evaluation, input_audit,
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
            "selected_candidate_by_fold": evaluation["inner_selected_metrics"]["selected_candidate_by_fold"],
            "direction_probability_changed": False,
            "fallback_is_exact_v58_noop": not evaluation["material_pass"],
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
        "direction_probability_changed": False,
        "fallback_is_exact_v58_noop": not evaluation["material_pass"],
        "seal_authorized": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
