"""V64 diagnostic: confidence correctness and causal feature-block polarity.

This runner is deliberately diagnostic-only.  It reads the immutable V36 DEV
contract and the frozen V58 outer-OOF predictions, and it never opens an
extension, research-seal pool, or final reserve.  The V58 predictions and
confidence policy remain unchanged.

Feature polarity is not inferred from raw correlations and no feature is sign
flipped.  A fixed linear probe measures block-only and leave-one-block-out
late-fold contributions.  Preprocessing, fitting, and cutoff selection are
repeated inside each chronological outer fold; a non-zero polarity is emitted
only when the fold sign is at least 70 percent stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
OUT = Path(os.environ.get(
    "MARKET_BIO_VERSION_OUTPUT",
    str(ROOT / "staging" / "V64_FEATURE_POLARITY_DIAGNOSTIC_V1"),
))

DEV_PATH = DATA / "dev_contract_v36_labeled.csv.gz"
V58_OOF_PATH = CACHE / "v58_locked_source_routing_oof.csv.gz"
EXPECTED_SHA256 = {
    DEV_PATH.name: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V58_OOF_PATH.name: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
}
EXPECTED_DEV_ROWS = 9462
EXPECTED_OOF_ROWS = 7568
EXPECTED_FAMILY = "V58_LOCKED_SOURCE_DGP_ROUTING"
ROUTES = {
    "US_SEC": "SEC_NUMERIC",
    "US_NEWS": "US_NEWS_MULTILINGUAL",
    "KR_NEWS": "KR_DIVERSE_CONSENSUS",
    "KR_KIND": "KR_DIVERSE_CONSENSUS",
}
SCOPES = ("OVERALL", "US_SEC", "US_NEWS", "KR_NEWS", "KR_KIND")
POLARITY_SOURCES = ("US_NEWS", "US_SEC")
EMBARGO = pd.Timedelta(minutes=35)
COST = float(v36.COST)
SEED = 6401
STABILITY_THRESHOLD = 0.70
POLARITY_EPSILON = 0.001
DIRECTION_CUTOFFS = (0.46, 0.48, 0.50, 0.52, 0.54)


RETURN_PATH = [
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_ret_90", "pre_ret_120", "ret_slope_2_15", "ret_slope_5_30",
    "entry_bar_ret", "intraday_ret_open",
]
VOLATILITY_RANGE = [
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "range_5m", "range_10m", "range_30m", "range_60m", "entry_bar_range",
    "abs_pre_ret_2", "abs_pre_ret_5", "abs_pre_ret_15", "abs_pre_ret_30",
    "abs_pre_ret_60",
]
VOLUME_ACTIVITY = [
    "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "volume_ratio_10_60",
]
TREND_POSITION = [
    "close_position_10", "close_position_30", "close_position_60",
    "return_autocorr_30", "up_fraction_10", "up_fraction_30",
    "trend_slope_30", "trend_slope_60", "vwap_distance_30",
]
MARKET_RELATIVE = [
    "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
    "benchmark_ret_30", "benchmark_ret_60", "excess_ret_5", "excess_ret_15",
    "excess_ret_30", "excess_ret_60",
]
EVENT_CLOCK = [
    "minutes_from_open", "minutes_to_close", "event_hour", "event_weekday",
    "event_month", "event_hour_sin", "event_hour_cos",
]
EVENT_LEXICAL_FLAGS = [
    "event_positive_kw", "event_negative_kw", "event_financing_kw",
    "event_trial_kw", "event_regulatory_kw", "event_ma_kw",
    "event_earnings_kw", "headline_len", "body_len", "kw_sentiment",
]

BLOCK_REGISTRY: dict[str, dict[str, Any]] = {
    "return_path": {
        "kind": "numeric", "columns": RETURN_PATH,
        "interpretation": "pre-event return path and entry-bar direction",
    },
    "volatility_range": {
        "kind": "numeric", "columns": VOLATILITY_RANGE,
        "interpretation": "pre-event volatility, ranges, and absolute returns",
    },
    "volume_activity": {
        "kind": "numeric", "columns": VOLUME_ACTIVITY,
        "interpretation": "pre-event relative trading activity",
    },
    "trend_position": {
        "kind": "numeric", "columns": TREND_POSITION,
        "interpretation": "trend, autocorrelation, price position, and VWAP distance",
    },
    "market_relative": {
        "kind": "numeric", "columns": MARKET_RELATIVE,
        "interpretation": "benchmark path and label-independent excess returns",
    },
    "event_clock": {
        "kind": "numeric", "columns": EVENT_CLOCK,
        "interpretation": "event timing and cyclical clock encoding",
    },
    "event_lexical_flags": {
        "kind": "numeric", "columns": EVENT_LEXICAL_FLAGS,
        "interpretation": "precomputed event keyword and text-length features",
    },
    "price_scale": {
        "kind": "numeric", "columns": ["log_entry_price"],
        "interpretation": "entry price scale available at inference",
    },
    "categorical_context": {
        "kind": "categorical", "columns": ["market", "source", "source_family", "event_type", "form"],
        "interpretation": "source and event context known at event time",
    },
    "text_content": {
        "kind": "text", "columns": ["semantic_text"],
        "interpretation": "headline/body semantic segment fitted by inner-past TF-IDF",
    },
}

DERIVED_COLUMNS = {
    "event_hour", "event_weekday", "event_month", "event_hour_sin",
    "event_hour_cos", "abs_pre_ret_2", "abs_pre_ret_5", "abs_pre_ret_15",
    "abs_pre_ret_30", "abs_pre_ret_60", "ret_slope_2_15", "ret_slope_5_30",
    "kw_sentiment", "excess_ret_5", "excess_ret_15", "excess_ret_30",
    "excess_ret_60", "semantic_text",
}
FORBIDDEN_LABEL_OR_FUTURE = {
    "y", "fwd_ret_30m", "exit_target_utc", "actual_exit_time_utc",
    "exit_time_utc", "exit_slippage_sec", "actual_hold_seconds", "exit_price",
}
IDENTIFIER_OR_PROVENANCE = {
    "event_id", "event_group_id", "ticker", "company", "url", "article_id",
    "origin_file", "contract_version", "price_source", "price_provider",
    "price_cadence_sec", "bar_timestamp_semantics", "raw_file_sha256",
    "timestamp_quality", "entry_target_utc", "actual_entry_time_utc",
    "entry_time_utc", "entry_slippage_sec", "event_time_utc",
}


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
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def verify_inputs() -> dict[str, str]:
    actual = {}
    for path in (DEV_PATH, V58_OOF_PATH):
        require(path.is_file(), f"required V64 diagnostic input missing: {path.name}")
        actual[path.name] = sha256(path)
        require(actual[path.name] == EXPECTED_SHA256[path.name],
                f"pinned V64 diagnostic input changed: {path.name}")
    return actual


def load_authorized_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes = verify_inputs()
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    oof = pd.read_csv(
        V58_OOF_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_DEV_ROWS, f"immutable V36 DEV rows changed: {len(dev)}")
    require(len(oof) == EXPECTED_OOF_ROWS, f"frozen V58 OOF rows changed: {len(oof)}")
    require(dev.event_id.is_unique, "immutable V36 DEV event_id is not unique")
    require(oof.event_id.is_unique, "frozen V58 OOF event_id is not unique")
    require(set(oof.event_id).issubset(set(dev.event_id)), "V58 OOF is not a V36 DEV subset")
    require(set(oof.source_family) == set(ROUTES), "V58 source-family contract changed")
    require(oof.family.eq(EXPECTED_FAMILY).all(), "V58 family contract changed")
    require(pd.to_numeric(oof.prob, errors="coerce").between(0.0, 1.0).all(),
            "V58 probability is missing or outside [0,1]")
    require(pd.to_numeric(oof.confidence_signal, errors="coerce").between(0.0, 1.0).all(),
            "V58 confidence signal is missing or outside [0,1]")
    for source, route in ROUTES.items():
        require(oof.loc[oof.source_family.eq(source), "routed_from"].eq(route).all(),
                f"V58 frozen route changed for {source}")

    aligned = dev.set_index("event_id").loc[oof.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), oof.y.to_numpy(int)),
            "V58 labels differ from immutable V36 DEV")
    require(np.allclose(aligned.fwd_ret_30m.to_numpy(float),
                        oof.fwd_ret_30m.to_numpy(float), rtol=0.0, atol=1e-15),
            "V58 forward returns differ from immutable V36 DEV")
    require(np.array_equal(aligned.source_family.astype(str).to_numpy(),
                           oof.source_family.astype(str).to_numpy()),
            "V58 source identities differ from immutable V36 DEV")

    # Start from the immutable DEV rows and overlay only frozen OOF policy fields.
    policy_columns = [column for column in oof.columns if column not in aligned.columns]
    for column in policy_columns:
        aligned[column] = oof[column].to_numpy()
    for column in ("prob", "high_conf", "confidence_signal", "fold", "family",
                   "new_issuer", "text_cluster_id", "cluster_weight", "routed_from"):
        if column in oof.columns:
            aligned[column] = oof[column].to_numpy()
    aligned["event_time_utc"] = pd.to_datetime(aligned.event_time_utc, utc=True)
    aligned["high_conf"] = aligned.high_conf.map(
        lambda value: value if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes"}
    ).astype(bool)
    audit = {
        "authorized_inputs": [str(DEV_PATH.relative_to(ROOT)), str(V58_OOF_PATH.relative_to(ROOT))],
        "input_sha256": hashes,
        "dev_rows": len(dev),
        "v58_oof_rows": len(oof),
        "v58_sources": {str(key): int(value) for key, value in oof.source_family.value_counts().items()},
        "v58_family": EXPECTED_FAMILY,
        "fixed_routes_verified": ROUTES,
        "forbidden_roles_loaded": [],
        "extension_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
    }
    return aligned, dev, audit


def prepared_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = app.model_frame(frame).copy()
    for column in ("source_family", "form"):
        prepared[column] = frame[column].fillna("").astype(str).to_numpy()
    prepared["semantic_text"] = frame.apply(v37.semantic_segment, axis=1).fillna("").astype(str)
    for column in app.NUMERIC_COLS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in BLOCK_REGISTRY["categorical_context"]["columns"]:
        prepared[column] = prepared[column].fillna("").astype(str)
    return prepared


def dev_probe_frame(dev: pd.DataFrame) -> pd.DataFrame:
    """Attach only label-independent duplicate controls to immutable DEV."""
    frame = dev.copy()
    frame["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True)
    frame["text_cluster_id"] = frame.apply(v37.normalized_cluster, axis=1)
    frame["cluster_weight"] = 1.0 / frame.groupby("text_cluster_id").event_id.transform("size")
    require(frame.text_cluster_id.notna().all(), "DEV probe cluster construction failed")
    require(frame.cluster_weight.gt(0.0).all(), "DEV probe cluster weights are invalid")
    return frame


def validate_registry(prepared: pd.DataFrame) -> None:
    registered_numeric = []
    registered_columns = []
    for spec in BLOCK_REGISTRY.values():
        require(set(spec["columns"]).issubset(prepared.columns),
                f"feature block has unavailable columns: {set(spec['columns']) - set(prepared.columns)}")
        registered_columns.extend(spec["columns"])
        if spec["kind"] == "numeric":
            registered_numeric.extend(spec["columns"])
    require(len(registered_columns) == len(set(registered_columns)),
            "feature blocks overlap")
    require(set(registered_numeric) == set(app.NUMERIC_COLS),
            f"numeric registry mismatch: missing={set(app.NUMERIC_COLS)-set(registered_numeric)}, "
            f"extra={set(registered_numeric)-set(app.NUMERIC_COLS)}")


def value_available(series: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    return series.fillna("").astype(str).str.strip().ne("")


def feature_inventory(frame: pd.DataFrame, dev_columns: Iterable[str]) -> dict[str, Any]:
    prepared = prepared_feature_frame(frame)
    validate_registry(prepared)
    blocks = {}
    for name, spec in BLOCK_REGISTRY.items():
        kind = str(spec["kind"])
        column_rows = []
        for column in spec["columns"]:
            available = value_available(prepared[column], kind)
            by_source = {}
            for source, index in frame.groupby("source_family").groups.items():
                by_source[str(source)] = float(available.loc[index].mean())
            column_rows.append({
                "column": column,
                "origin": "derived_label_independent" if column in DERIVED_COLUMNS else "raw_event_time_available",
                "non_null_or_nonempty_rate": float(available.mean()),
                "availability_by_source": by_source,
            })
        blocks[name] = {
            **spec,
            "audit_enabled": True,
            "raw_sign_flip_allowed": False,
            "raw_sign_flip_performed": False,
            "features": column_rows,
        }
    raw_columns = set(dev_columns)
    blocked = sorted(raw_columns & (FORBIDDEN_LABEL_OR_FUTURE | IDENTIFIER_OR_PROVENANCE))
    registered_raw = {
        column for spec in BLOCK_REGISTRY.values() for column in spec["columns"]
        if column not in DERIVED_COLUMNS
    }
    unregistered = sorted(raw_columns - registered_raw - FORBIDDEN_LABEL_OR_FUTURE - IDENTIFIER_OR_PROVENANCE)
    return {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "rows": len(frame),
        "blocks": blocks,
        "blocked_label_or_future_columns": sorted(raw_columns & FORBIDDEN_LABEL_OR_FUTURE),
        "blocked_identifier_or_provenance_columns": sorted(raw_columns & IDENTIFIER_OR_PROVENANCE),
        "all_blocked_columns": blocked,
        "unregistered_raw_columns_not_used_by_probe": unregistered,
        "registry_checks": {
            "all_model_frame_numeric_features_registered_once": True,
            "blocks_are_disjoint": True,
            "raw_sign_flip_allowed": False,
            "raw_sign_flip_performed": False,
        },
    }


def registry_document() -> dict[str, Any]:
    return {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "registry_name": "FEATURE_BLOCK_REGISTRY",
        "blocks": BLOCK_REGISTRY,
        "derived_label_independent_columns": sorted(DERIVED_COLUMNS),
        "prohibited_from_model": {
            "label_or_future": sorted(FORBIDDEN_LABEL_OR_FUTURE),
            "identifier_or_provenance": sorted(IDENTIFIER_OR_PROVENANCE),
        },
        "polarity_semantics": {
            "+": "stable positive nested late-fold contribution",
            "0": "weak, mixed, or less than 70 percent stable; conservative fallback",
            "-": "stable negative nested late-fold contribution",
            "raw_sign_flip_allowed": False,
        },
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None


def diagnostic_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    probability = frame.prob.to_numpy(float)
    prediction = probability >= 0.5
    returns = frame.fwd_ret_30m.to_numpy(float)
    signed_net = np.where(prediction, 1.0, -1.0) * returns - COST
    meaningful = bool(len(frame) >= 20 and np.unique(y).size == 2 and np.bincount(y, minlength=2).min() >= 5)
    return {
        "n": int(len(frame)),
        "accuracy": float((prediction == y).mean()) if len(frame) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if meaningful else None,
        "ba_meaningful": meaningful,
        "mean_signed_net": float(signed_net.mean()) if len(frame) else None,
        "mean_abs_return": float(np.abs(returns).mean()) if len(frame) else None,
        "mean_probability_margin": float(np.abs(probability - 0.5).mean()) if len(frame) else None,
        "pred_up": float(prediction.mean()) if len(frame) else None,
    }


def quantile_bin_frame(frame: pd.DataFrame, confidence_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([confidence_column, "event_time_utc", "event_id"], kind="mergesort").copy()
    positions = np.arange(len(ordered), dtype=int)
    ordered["confidence_bin"] = np.minimum(9, (positions * 10) // max(1, len(ordered))) + 1
    return ordered


def confidence_scope_report(frame: pd.DataFrame, confidence_column: str) -> dict[str, Any]:
    work = frame.copy().reset_index(drop=True)
    confidence = work[confidence_column].to_numpy(float)
    correctness = ((work.prob.to_numpy(float) >= 0.5) == work.y.to_numpy(int)).astype(int)
    bins_frame = quantile_bin_frame(work, confidence_column)
    bins = []
    for bin_number, group in bins_frame.groupby("confidence_bin", sort=True):
        row = diagnostic_metrics(group)
        row.update({
            "bin": int(bin_number),
            "confidence_quantile_low": float((int(bin_number) - 1) / 10.0),
            "confidence_quantile_high": float(int(bin_number) / 10.0),
            "mean_confidence": float(group[confidence_column].mean()),
        })
        bins.append(row)
    top = {}
    order = np.argsort(-confidence, kind="stable")
    for percentage in (5, 10, 20):
        count = max(1, int(math.ceil(len(work) * percentage / 100.0)))
        selected = work.iloc[order[:count]].copy()
        metrics = diagnostic_metrics(selected)
        metrics["coverage"] = float(count / len(work))
        metrics["mean_confidence"] = float(selected[confidence_column].mean())
        top[f"top_{percentage}_percent"] = metrics

    high = work[work.high_conf].copy()
    low = work[~work.high_conf].copy()
    high_metrics = diagnostic_metrics(high) if len(high) else {"n": 0}
    low_metrics = diagnostic_metrics(low) if len(low) else {"n": 0}
    accuracy_by_bin = pd.Series([row["accuracy"] for row in bins], dtype=float)
    net_by_bin = pd.Series([row["mean_signed_net"] for row in bins], dtype=float)
    abs_return_by_bin = pd.Series([row["mean_abs_return"] for row in bins], dtype=float)
    bin_index = pd.Series(np.arange(1, len(bins) + 1), dtype=float)
    accuracy_steps = np.diff(accuracy_by_bin.to_numpy()) if len(bins) > 1 else np.array([])
    top20 = top["top_20_percent"]
    bottom_count = max(1, int(math.ceil(len(work) * 0.20)))
    bottom20 = diagnostic_metrics(work.iloc[np.argsort(confidence, kind="stable")[:bottom_count]])
    auc = safe_auc(correctness, confidence)
    reversals = {
        "confidence_correctness_auc_below_0_5": bool(auc is not None and auc < 0.5),
        "top20_accuracy_not_above_bottom20": bool(
            top20.get("accuracy") is not None and bottom20.get("accuracy") is not None
            and top20["accuracy"] <= bottom20["accuracy"]
        ),
        "top20_net_not_above_bottom20": bool(
            top20.get("mean_signed_net") is not None and bottom20.get("mean_signed_net") is not None
            and top20["mean_signed_net"] <= bottom20["mean_signed_net"]
        ),
        "frozen_highconf_accuracy_not_above_other": bool(
            high_metrics.get("accuracy") is not None and low_metrics.get("accuracy") is not None
            and high_metrics["accuracy"] <= low_metrics["accuracy"]
        ),
    }
    return {
        "n": len(work),
        "confidence_column": confidence_column,
        "confidence_correctness_auc": auc,
        "quantile_bins": bins,
        "high_confidence_top_slices": top,
        "bottom_20_percent": bottom20,
        "frozen_v58_high_confidence": high_metrics,
        "frozen_v58_non_high_confidence": low_metrics,
        "monotonicity": {
            "accuracy_spearman_by_bin": finite_or_none(bin_index.corr(accuracy_by_bin, method="spearman")),
            "mean_net_spearman_by_bin": finite_or_none(bin_index.corr(net_by_bin, method="spearman")),
            "abs_return_spearman_by_bin": finite_or_none(bin_index.corr(abs_return_by_bin, method="spearman")),
            "nondecreasing_accuracy_steps": int((accuracy_steps >= 0.0).sum()),
            "accuracy_step_count": int(len(accuracy_steps)),
            "strict_monotonic_accuracy": bool(len(accuracy_steps) > 0 and (accuracy_steps >= 0.0).all()),
        },
        "reversal_checks": {
            **reversals,
            "high_confidence_reversal": bool(any(reversals.values())),
        },
    }


def confidence_report(frame: pd.DataFrame) -> dict[str, Any]:
    reports = {}
    for scope in SCOPES:
        subset = frame if scope == "OVERALL" else frame[frame.source_family.eq(scope)]
        require(len(subset) > 0, f"required confidence diagnostic scope absent: {scope}")
        reports[scope] = {
            "frozen_v58_confidence": confidence_scope_report(subset, "confidence_signal"),
            "model_probability_margin": confidence_scope_report(
                subset.assign(probability_margin=np.abs(subset.prob.to_numpy(float) - 0.5)),
                "probability_margin",
            ),
        }
    return {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "diagnostic_only": True,
        "v58_predictions_changed": False,
        "transaction_cost": COST,
        "scopes": reports,
    }


def fit_block_matrices(
    train: pd.DataFrame,
    target: pd.DataFrame,
    block_names: list[str],
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix], dict[str, int]]:
    x_train = prepared_feature_frame(train)
    x_target = prepared_feature_frame(target)
    train_matrices: dict[str, sparse.csr_matrix] = {}
    target_matrices: dict[str, sparse.csr_matrix] = {}
    dimensions = {}
    for name in block_names:
        spec = BLOCK_REGISTRY[name]
        columns = list(spec["columns"])
        kind = spec["kind"]
        if kind == "numeric":
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            train_array = imputer.fit_transform(x_train[columns])
            target_array = imputer.transform(x_target[columns])
            scaler = StandardScaler(with_mean=False)
            train_matrix = scaler.fit_transform(train_array)
            target_matrix = scaler.transform(target_array)
            train_matrix = sparse.csr_matrix(train_matrix, dtype=np.float32)
            target_matrix = sparse.csr_matrix(target_matrix, dtype=np.float32)
        elif kind == "categorical":
            encoder = OneHotEncoder(
                handle_unknown="ignore", min_frequency=3, sparse_output=True,
                dtype=np.float32,
            )
            train_matrix = encoder.fit_transform(x_train[columns]).tocsr()
            target_matrix = encoder.transform(x_target[columns]).tocsr()
        elif kind == "text":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=3, max_df=0.997,
                max_features=15000, sublinear_tf=True, dtype=np.float32,
                token_pattern=r"(?u)\b\w+\b",
            )
            try:
                train_matrix = vectorizer.fit_transform(x_train[columns[0]]).tocsr()
                target_matrix = vectorizer.transform(x_target[columns[0]]).tocsr()
            except ValueError as error:
                if "empty vocabulary" not in str(error).lower():
                    raise
                train_matrix = sparse.csr_matrix((len(train), 1), dtype=np.float32)
                target_matrix = sparse.csr_matrix((len(target), 1), dtype=np.float32)
        else:
            raise RuntimeError(f"unsupported block kind: {kind}")
        require(train_matrix.shape[1] == target_matrix.shape[1], f"matrix width mismatch: {name}")
        train_matrices[name] = train_matrix
        target_matrices[name] = target_matrix
        dimensions[name] = int(train_matrix.shape[1])
    return train_matrices, target_matrices, dimensions


def stack_blocks(matrices: dict[str, sparse.csr_matrix], names: list[str]) -> sparse.csr_matrix:
    require(bool(names), "probe cannot fit an empty block list")
    return sparse.hstack([matrices[name] for name in names], format="csr", dtype=np.float32)


def fit_probe(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_target: sparse.csr_matrix,
    seed: int,
) -> np.ndarray:
    require(np.unique(y_train).size == 2, "probe training requires both labels")
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=3e-4, max_iter=1200,
        tol=1e-3, average=True, class_weight="balanced", random_state=seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return np.clip(model.predict_proba(x_target)[:, 1], 1e-6, 1.0 - 1e-6)


def late_metrics(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    prediction = probability >= cutoff
    auc = safe_auc(y, probability)
    balanced = float(balanced_accuracy_score(y, prediction)) if np.unique(y).size == 2 else None
    signed = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    score = None if auc is None or balanced is None else float(0.5 * auc + 0.5 * balanced)
    return {
        "n": len(frame),
        "cutoff": float(cutoff),
        "balanced_accuracy": balanced,
        "auc": auc,
        "late_score": score,
        "accuracy": float((prediction == y).mean()),
        "mean_signed_net": float(signed.mean()),
    }


def choose_cutoff(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = []
    for cutoff in DIRECTION_CUTOFFS:
        metrics = late_metrics(frame, probability, cutoff)
        bounded_net = max(-0.01, min(0.01, float(metrics["mean_signed_net"])))
        selection_score = float(metrics["late_score"] or 0.0) + 2.0 * bounded_net
        candidates.append({**metrics, "selection_score": selection_score})
    candidates.sort(key=lambda row: (row["selection_score"], -abs(row["cutoff"] - 0.5)), reverse=True)
    return float(candidates[0]["cutoff"]), {"selected": candidates[0], "candidates": candidates}


def subset_probe(
    names: list[str],
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
    inner_probability = fit_probe(
        stack_blocks(inner_train_matrices, names), inner_train.y.to_numpy(int),
        inner_train.cluster_weight.to_numpy(float),
        stack_blocks(inner_valid_matrices, names), seed,
    )
    cutoff, selection = choose_cutoff(inner_valid, inner_probability)
    outer_probability = fit_probe(
        stack_blocks(outer_train_matrices, names), outer_train.y.to_numpy(int),
        outer_train.cluster_weight.to_numpy(float),
        stack_blocks(outer_valid_matrices, names), seed + 1000,
    )
    return {
        "blocks": names,
        "inner_cutoff_selection": selection,
        "outer_metrics": late_metrics(outer_valid, outer_probability, cutoff),
    }


def assert_nested_times(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label} violates strict 35-minute embargo")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end.isoformat(), "valid_start": valid_start.isoformat(),
        "strict_35m_embargo_verified": True,
    }


def polarity_folds(frame: pd.DataFrame, smoke: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_names = list(BLOCK_REGISTRY)
    sources = list(POLARITY_SOURCES)
    if smoke:
        block_names = block_names[:2]
        sources = sources[:1]
    rows = []
    audit = {"sources": {}, "smoke_test": smoke}
    for source_index, source in enumerate(sources):
        source_frame = frame[frame.source_family.eq(source)].copy()
        folds = v37.chronological_folds(source_frame)
        require(bool(folds), f"no eligible chronological polarity folds for {source}")
        if smoke:
            folds = folds[:1]
        audit["sources"][source] = []
        for train, valid, fold in folds:
            inner_train, inner_valid = v37.split_inner(train)
            require(len(inner_train) >= 100 and len(inner_valid) >= 30,
                    f"insufficient nested rows for {source} fold {fold}")
            require(inner_train.y.nunique() == 2 and inner_valid.y.nunique() == 2,
                    f"nested labels insufficient for {source} fold {fold}")
            outer_time = assert_nested_times(train, valid, f"{source} outer fold {fold}")
            inner_time = assert_nested_times(inner_train, inner_valid, f"{source} inner fold {fold}")

            inner_train_matrices, inner_valid_matrices, inner_dimensions = fit_block_matrices(
                inner_train, inner_valid, block_names,
            )
            outer_train_matrices, outer_valid_matrices, outer_dimensions = fit_block_matrices(
                train, valid, block_names,
            )
            seed = SEED + source_index * 100 + int(fold)
            full = subset_probe(
                block_names, inner_train_matrices, inner_valid_matrices,
                outer_train_matrices, outer_valid_matrices,
                inner_train, inner_valid, train, valid, seed,
            )
            full_score = float(full["outer_metrics"]["late_score"])
            fold_audit = {
                "fold": int(fold), "outer": outer_time, "inner": inner_time,
                "inner_feature_dimensions": inner_dimensions,
                "outer_feature_dimensions": outer_dimensions,
                "full_probe": full,
            }
            audit["sources"][source].append(fold_audit)
            for block_index, block in enumerate(block_names):
                block_only = subset_probe(
                    [block], inner_train_matrices, inner_valid_matrices,
                    outer_train_matrices, outer_valid_matrices,
                    inner_train, inner_valid, train, valid, seed + 10 + block_index,
                )
                ablated_names = [name for name in block_names if name != block]
                require(bool(ablated_names), "polarity diagnostic needs at least two blocks")
                ablated = subset_probe(
                    ablated_names, inner_train_matrices, inner_valid_matrices,
                    outer_train_matrices, outer_valid_matrices,
                    inner_train, inner_valid, train, valid, seed + 100 + block_index,
                )
                ablated_score = float(ablated["outer_metrics"]["late_score"])
                block_score = float(block_only["outer_metrics"]["late_score"])
                ablation_delta = full_score - ablated_score
                standalone_delta = block_score - 0.5
                contribution = 0.5 * ablation_delta + 0.5 * standalone_delta
                sign = "+" if contribution > POLARITY_EPSILON else "-" if contribution < -POLARITY_EPSILON else "0"
                rows.append({
                    "source_family": source, "fold": int(fold), "block": block,
                    "full_late_score": full_score,
                    "ablated_late_score": ablated_score,
                    "block_only_late_score": block_score,
                    "ablation_delta": ablation_delta,
                    "block_only_delta_vs_0_5": standalone_delta,
                    "combined_contribution": contribution,
                    "fold_polarity": sign,
                    "block_only": block_only,
                    "ablated": ablated,
                })
            print(
                f"[V64 POLARITY] source={source} fold={fold} "
                f"train={len(train)} valid={len(valid)} blocks={len(block_names)}",
                flush=True,
            )
    return rows, audit


def aggregate_polarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame([{
        "source_family": row["source_family"], "block": row["block"],
        "fold": row["fold"], "combined_contribution": row["combined_contribution"],
        "fold_polarity": row["fold_polarity"],
    } for row in rows])
    aggregated = {}
    for (source, block), group in table.groupby(["source_family", "block"], sort=True):
        counts = group.fold_polarity.value_counts().to_dict()
        median = float(group.combined_contribution.median())
        median_sign = "+" if median > POLARITY_EPSILON else "-" if median < -POLARITY_EPSILON else "0"
        stability = float((group.fold_polarity == median_sign).mean()) if median_sign != "0" else 0.0
        if median_sign in {"+", "-"} and stability >= STABILITY_THRESHOLD:
            polarity = median_sign
            fallback_reason = None
        else:
            polarity = "0"
            fallback_reason = (
                "median_contribution_within_epsilon" if median_sign == "0"
                else "stability_below_0.70"
            )
        aggregated.setdefault(str(source), {})[str(block)] = {
            "polarity": polarity,
            "fold_n": int(len(group)),
            "positive_fraction": float((group.fold_polarity == "+").mean()),
            "zero_fraction": float((group.fold_polarity == "0").mean()),
            "negative_fraction": float((group.fold_polarity == "-").mean()),
            "median_combined_contribution": median,
            "stability_to_median_sign": stability,
            "stability_threshold": STABILITY_THRESHOLD,
            "fallback_reason": fallback_reason,
            "fold_sign_counts": {sign: int(counts.get(sign, 0)) for sign in ("+", "0", "-")},
        }
    return aggregated


def polarity_report(frame: pd.DataFrame, smoke: bool = False) -> dict[str, Any]:
    rows, nested_audit = polarity_folds(frame, smoke=smoke)
    return {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "diagnostic_only": True,
        "method": (
            "fixed SGD log-loss probe; preprocessing fitted on inner/outer train only; "
            "inner-past cutoff selection; chronological outer late-score; equal-weight "
            "ablation and block-only contribution"
        ),
        "polarity_epsilon": POLARITY_EPSILON,
        "stability_threshold": STABILITY_THRESHOLD,
        "raw_sign_flip_allowed": False,
        "raw_sign_flip_performed": False,
        "sources_required": list(POLARITY_SOURCES),
        "aggregate": aggregate_polarity(rows),
        "fold_results": rows,
        "nested_chronology_audit": nested_audit,
    }


def standard_reports(
    frame: pd.DataFrame,
    input_audit: dict[str, Any],
    confidence: dict[str, Any],
    polarity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selected = v44.summarize(frame)
    validation = (
        "V58 predictions and confidence are frozen chronological outer OOF. V64 is audit-only; "
        "feature contribution probes use source-specific outer chronology, strict 35-minute "
        "embargoes, and inner-past-only cutoff selection. No seal, final reserve, or extension read."
    )
    comparison = {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "status": "DIAGNOSTIC_ONLY",
        "champion": "V58_LOCKED_SOURCE_DGP_ROUTING",
        "champion_unchanged": True,
        "models": {"V58_LOCKED_SOURCE_DGP_ROUTING_UNCHANGED": selected},
        "confidence_reversal_by_scope": {
            scope: report["frozen_v58_confidence"]["reversal_checks"]
            for scope, report in confidence["scopes"].items()
        },
        "feature_polarity": polarity["aggregate"],
        "input_audit": input_audit,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V64",
        "status": "DIAGNOSTIC_ONLY",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "opened_dev_only": True,
        "v36plus_seal_outcomes_loaded": False,
        "extension_loaded": False,
        "selected": selected,
        "champion_unchanged": True,
        "seal_authorized": False,
        "validation": validation,
    }
    transfer = {
        "version": "V64",
        "status": "DIAGNOSTIC_ONLY",
        "seal_outcomes_loaded": False,
        "extension_loaded": False,
        "selected_oof_by_source": selected["metrics"]["by_source_family"],
        "selected_oof_by_market": selected["metrics"]["by_market"],
        "selected_oof_new_issuer": selected["metrics"]["new_issuer"],
        "confidence_correctness_auc_by_source": {
            source: confidence["scopes"][source]["frozen_v58_confidence"]["confidence_correctness_auc"]
            for source in ROUTES
        },
        "polarity_by_source": polarity["aggregate"],
        "method": validation,
    }
    return {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
    }


def confidence_bin_table(confidence: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for scope, scope_report in confidence["scopes"].items():
        for confidence_name, report in scope_report.items():
            for row in report["quantile_bins"]:
                rows.append({"scope": scope, "confidence_measure": confidence_name, **row})
    return pd.DataFrame(rows)


def feature_block_polarity_table(polarity: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in polarity["fold_results"]:
        rows.append({
            "source_family": row["source_family"],
            "fold": row["fold"],
            "block": row["block"],
            "full_late_score": row["full_late_score"],
            "ablated_late_score": row["ablated_late_score"],
            "block_only_late_score": row["block_only_late_score"],
            "ablation_delta": row["ablation_delta"],
            "block_only_delta_vs_0_5": row["block_only_delta_vs_0_5"],
            "combined_contribution": row["combined_contribution"],
            "fold_polarity": row["fold_polarity"],
            "block_only_inner_cutoff": row["block_only"]["inner_cutoff_selection"]["selected"]["cutoff"],
            "ablated_inner_cutoff": row["ablated"]["inner_cutoff_selection"]["selected"]["cutoff"],
            "raw_sign_flip_performed": False,
        })
    return pd.DataFrame(rows)


def polarity_stability_table(polarity: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for source, blocks in polarity["aggregate"].items():
        for block, report in blocks.items():
            counts = report["fold_sign_counts"]
            scalar_report = {key: value for key, value in report.items() if key != "fold_sign_counts"}
            rows.append({
                "source_family": source, "block": block, **scalar_report,
                "positive_fold_n": counts["+"], "zero_fold_n": counts["0"],
                "negative_fold_n": counts["-"],
            })
    return pd.DataFrame(rows)


def source_polarity_table(polarity: dict[str, Any]) -> pd.DataFrame:
    fold_table = feature_block_polarity_table(polarity)
    stability = polarity_stability_table(polarity)
    rows = []
    for source, group in fold_table.groupby("source_family", sort=True):
        source_stability = stability[stability.source_family.eq(source)]
        rows.append({
            "source_family": source,
            "fold_block_observations": int(len(group)),
            "eligible_fold_n": int(group.fold.nunique()),
            "block_n": int(group.block.nunique()),
            "mean_combined_contribution": float(group.combined_contribution.mean()),
            "median_combined_contribution": float(group.combined_contribution.median()),
            "stable_positive_blocks": int(source_stability.polarity.eq("+").sum()),
            "stable_zero_fallback_blocks": int(source_stability.polarity.eq("0").sum()),
            "stable_negative_blocks": int(source_stability.polarity.eq("-").sum()),
            "raw_sign_flip_performed": False,
        })
    return pd.DataFrame(rows)


def extreme_reversal_report(confidence: dict[str, Any]) -> dict[str, Any]:
    scopes = {}
    for scope, scope_report in confidence["scopes"].items():
        scopes[scope] = {}
        for confidence_name, report in scope_report.items():
            top = report["high_confidence_top_slices"]["top_20_percent"]
            bottom = report["bottom_20_percent"]
            scopes[scope][confidence_name] = {
                "confidence_correctness_auc": report["confidence_correctness_auc"],
                "top20_accuracy": top.get("accuracy"),
                "bottom20_accuracy": bottom.get("accuracy"),
                "top20_minus_bottom20_accuracy": (
                    None if top.get("accuracy") is None or bottom.get("accuracy") is None
                    else float(top["accuracy"] - bottom["accuracy"])
                ),
                "top20_mean_signed_net": top.get("mean_signed_net"),
                "bottom20_mean_signed_net": bottom.get("mean_signed_net"),
                "top20_minus_bottom20_mean_signed_net": (
                    None if top.get("mean_signed_net") is None or bottom.get("mean_signed_net") is None
                    else float(top["mean_signed_net"] - bottom["mean_signed_net"])
                ),
                "reversal_checks": report["reversal_checks"],
            }
    return {
        "version": "V64",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "diagnostic_only": True,
        "scopes": scopes,
        "us_news_high_confidence_reversal": scopes["US_NEWS"]["frozen_v58_confidence"]["reversal_checks"]["high_confidence_reversal"],
    }


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true",
                       help="verify pinned inputs and feature registry; do not fit or write")
    modes.add_argument("--smoke-test", action="store_true",
                       help="run one source/fold and two blocks; do not write")
    modes.add_argument("--dry-run", action="store_true",
                       help="run the complete diagnostic; do not write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    frame, dev, input_audit = load_authorized_inputs()
    probe_frame = dev_probe_frame(dev)
    inventory = feature_inventory(probe_frame, dev.columns)
    registry = registry_document()
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
            "input_audit": input_audit,
            "registry_checks": inventory["registry_checks"],
            "output_written": False,
        }), indent=2, ensure_ascii=False))
        return

    confidence = confidence_report(frame)
    polarity = polarity_report(probe_frame, smoke=args.smoke_test)
    if args.smoke_test:
        print(json.dumps(json_clean({
            "status": "SMOKE_OK", "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
            "sources": list(polarity["aggregate"]),
            "aggregate": polarity["aggregate"],
            "output_written": False,
        }), indent=2, ensure_ascii=False))
        return

    json_reports = standard_reports(frame, input_audit, confidence, polarity)
    json_reports.update({
        "CONFIDENCE_RELIABILITY_REPORT.json": confidence,
        "EXTREME_REVERSAL_REPORT.json": extreme_reversal_report(confidence),
        "research/FEATURE_BLOCK_REGISTRY.json": {**registry, "feature_inventory": inventory},
    })
    json_reports = {name: json_clean(payload) for name, payload in json_reports.items()}
    csv_reports = {
        "CONFIDENCE_BIN_REPORT.csv": confidence_bin_table(confidence),
        "FEATURE_BLOCK_POLARITY_AUDIT.csv": feature_block_polarity_table(polarity),
        "SOURCE_POLARITY_AUDIT.csv": source_polarity_table(polarity),
        "POLARITY_STABILITY_REPORT.csv": polarity_stability_table(polarity),
    }
    output_names = sorted([*json_reports, *csv_reports])
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_OK", "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
            "output_written": False,
            "would_write": output_names,
            "confidence_correctness_auc": {
                scope: confidence["scopes"][scope]["frozen_v58_confidence"]["confidence_correctness_auc"]
                for scope in SCOPES
            },
            "us_news_high_confidence_reversal": confidence["scopes"]["US_NEWS"][
                "frozen_v58_confidence"
            ]["reversal_checks"]["high_confidence_reversal"],
            "us_news_reversal_checks": confidence["scopes"]["US_NEWS"][
                "frozen_v58_confidence"
            ]["reversal_checks"],
            "polarity": polarity["aggregate"],
        }, indent=2, ensure_ascii=False))
        return

    for name, payload in json_reports.items():
        v37.atomic_json(payload, OUT / name)
    for name, payload in csv_reports.items():
        atomic_csv(payload, OUT / name)
    print(json.dumps({
        "status": "DIAGNOSTIC_WRITTEN",
        "hypothesis": "FEATURE_POLARITY_DIAGNOSTIC_V1",
        "output_dir": str(OUT),
        "files": output_names,
        "champion_unchanged": True,
        "seal_authorized": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
