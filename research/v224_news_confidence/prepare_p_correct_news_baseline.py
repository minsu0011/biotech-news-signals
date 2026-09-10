"""Audit and prepare a strict-OOF P_CORRECT_NEWS baseline.

Authorized data are limited to the immutable V36 DEV contract/events and the
atomically committed V69 outer-OOF direction predictions.  This module never
opens DEV_EXTENSION, role assignments, Research Seal, or Final Meta data.

The correctness target is I[(V69 outer-OOF direction >= .5) == y].  A row is
never evaluated by a correctness model fitted on that row.  Correctness-derived
history features are either prequential or frozen from an earlier outer-train
window, and require a strict 35-minute outcome-availability lag.  Raw URLs and
article IDs are never model inputs.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_LABEL_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
DEV_EVENT_PATH = ROOT / "data" / "dev_contract_v36_events.csv.gz"
DEV_STATUS_PATH = ROOT / "data" / "DEV_CONTRACT_V36_STATUS.json"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
V69_COMMIT_PATH = V69_DIR / "COMMIT.json"
V69_MANIFEST_PATH = V69_DIR / "ARTIFACT_MANIFEST.json"

EXPECTED_DEV_LABEL_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_DEV_EVENT_SHA256 = "6b49d4545f4d2cc95f76c5bfbc27e4f2836b23b2e3ff2488473d0fd1a7d1e748"
EXPECTED_V69_SHA256 = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
EXPECTED_V69_EXPERIMENT = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_ROWS = {"US_NEWS": 954, "KR_NEWS": 1139}
NEWS_SOURCES = ("US_NEWS", "KR_NEWS")
EMBARGO = pd.Timedelta(minutes=35)
OUTER_BLOCKS = 5
SMOOTHING = 12.0
SEED = 22431

EXPERT_PROBABILITY_COLUMNS = (
    "prob", "prob_opportunity_direction", "prob_event_form_direction",
    "prob_fixed_numeric_direction",
)
EXPERT_CONFIDENCE_COLUMNS = (
    "confidence_signal", "opportunity_confidence",
    "confidence_opportunity_direction", "confidence_event_form_direction",
    "confidence_fixed_numeric_direction",
)
HISTORY_KEYS = (
    "provider_class", "publisher_host_class", "event_type_clean", "ticker_clean",
    "article_key",
)
IDENTIFIER_COLUMNS = {
    "event_id", "event_group_id", "url", "article_id", "article_key",
    "headline", "body", "text", "origin_file",
}
FORBIDDEN_MODEL_COLUMNS = {
    "y", "fwd_ret_30m", "correct_target", "high_conf", "opportunity_high",
    "exit_target_utc", "actual_exit_time_utc", "exit_time_utc", "exit_price",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
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
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression_options: str | dict[str, Any] | None = compression
    if compression == "gzip":
        compression_options = {"method": "gzip", "mtime": 0}
    frame.to_csv(temporary, index=False, compression=compression_options)
    temporary.replace(path)


def configure_runtime(cpu_ids: Sequence[int]) -> dict[str, Any]:
    ids = tuple(dict.fromkeys(int(value) for value in cpu_ids))
    available = os.cpu_count() or 1
    require(ids and all(0 <= value < available for value in ids),
            f"requested CPU ids {ids} unavailable; logical CPUs={available}")
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        mask = sum(1 << value for value in ids)
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(ids))
    return {
        "cpu_ids": list(ids), "thread_count": 2, "gpu_hidden": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }


def verify_authority() -> dict[str, Any]:
    for path in (DEV_LABEL_PATH, DEV_EVENT_PATH, DEV_STATUS_PATH, V69_PATH,
                 V69_COMMIT_PATH, V69_MANIFEST_PATH):
        require(path.is_file(), f"authorized input missing: {path}")
    dev_status = json.loads(DEV_STATUS_PATH.read_text(encoding="utf-8"))
    commit = json.loads(V69_COMMIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(V69_MANIFEST_PATH.read_text(encoding="utf-8"))
    hashes = {
        "dev_labeled": sha256(DEV_LABEL_PATH), "dev_events": sha256(DEV_EVENT_PATH),
        "v69_oof": sha256(V69_PATH), "v69_commit": sha256(V69_COMMIT_PATH),
        "v69_manifest": sha256(V69_MANIFEST_PATH),
    }
    require(hashes["dev_labeled"] == EXPECTED_DEV_LABEL_SHA256, "DEV labeled hash changed")
    require(hashes["dev_events"] == EXPECTED_DEV_EVENT_SHA256, "DEV event hash changed")
    require(hashes["v69_oof"] == EXPECTED_V69_SHA256, "V69 selected OOF hash changed")
    require(dev_status.get("events_sha256") == hashes["dev_events"],
            "DEV status is not bound to event metadata")
    require(dev_status.get("labeled_sha256") == hashes["dev_labeled"],
            "DEV status is not bound to labels")
    require(commit.get("data_sha") == hashes["dev_labeled"], "V69 is not bound to DEV labels")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT, "V69 experiment changed")
    require(commit.get("manifest_sha256") == hashes["v69_manifest"], "V69 manifest changed")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT, "V69 manifest authority changed")
    selected = manifest.get("files", {}).get(V69_PATH.name)
    require(bool(selected), "V69 selected OOF absent from manifest")
    require(selected["sha256"] == hashes["v69_oof"] and int(selected["bytes"]) == V69_PATH.stat().st_size,
            "V69 selected OOF manifest mismatch")
    # Verify every manifest entry without parsing any non-authorized outcome artifact.
    for name, spec in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested file missing: {name}")
        require(path.stat().st_size == int(spec["bytes"]), f"V69 byte size changed: {name}")
        require(sha256(path) == spec["sha256"], f"V69 manifested hash changed: {name}")
    return {
        "authority": "V36 immutable DEV + atomic V69 outer OOF",
        "authorized_inputs": [
            str(DEV_LABEL_PATH.relative_to(ROOT)), str(DEV_EVENT_PATH.relative_to(ROOT)),
            str(DEV_STATUS_PATH.relative_to(ROOT)), str(V69_COMMIT_PATH.relative_to(ROOT)),
            str(V69_MANIFEST_PATH.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
        ],
        "sha256": hashes, "v69_manifest_files_verified": len(manifest["files"]),
        "dev_extension_loaded": False, "role_assignment_loaded": False,
        "research_seal_loaded": False, "final_meta_loaded": False,
    }


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes"}
    ).astype(bool)


def present(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def url_host(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "<MISSING>"
    try:
        host = (urlparse(text).hostname or "").lower()
    except ValueError:
        host = ""
    return host.removeprefix("www.") or "<INVALID>"


def host_class(host: str) -> str:
    if host.endswith("reuters.com"):
        return "REUTERS_HOST"
    if host.endswith("naver.com"):
        return "NAVER_AGGREGATOR_HOST"
    if host.endswith("news.google.com") or host.endswith("google.com"):
        return "GOOGLE_NEWS_AGGREGATOR_HOST"
    if host in {"<MISSING>", "<INVALID>"}:
        return host
    return "OTHER_HOST"


def provider_class(value: Any) -> str:
    source = "" if pd.isna(value) else str(value).upper()
    if "REUTERS" in source:
        return "REUTERS"
    if "NAVER" in source:
        return "NAVER"
    if "EXACT" in source or "GOOGLE" in source:
        return "GOOGLE_NEWS"
    return "OTHER_NEWS"


def stable_article_key(article_id: Any, url: Any) -> str:
    article = "" if pd.isna(article_id) else str(article_id).strip().lower()
    raw_url = "" if pd.isna(url) else str(url).strip().lower()
    normalized_url = raw_url.split("?", 1)[0].rstrip("/")
    payload = f"article|{article}" if article else f"url|{normalized_url}"
    if payload in {"article|", "url|"}:
        return "<MISSING>"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_event_type(value: Any) -> str:
    text = "" if pd.isna(value) else re.sub(r"\s+", "_", str(value).strip().upper())
    return text[:80] or "<MISSING>"


def load_news_frame() -> pd.DataFrame:
    oof = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    labels = pd.read_csv(
        DEV_LABEL_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        usecols=["event_id", "source_family", "y", "fwd_ret_30m", "entry_target_utc"],
    )
    metadata_columns = [
        "event_id", "event_group_id", "ticker", "event_time_utc", "source",
        "source_family", "form", "event_type", "timestamp_quality", "url",
        "article_id", "headline", "body", "text", "origin_file",
    ]
    metadata = pd.read_csv(
        DEV_EVENT_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        usecols=metadata_columns,
    )
    require(oof.event_id.is_unique and labels.event_id.is_unique and metadata.event_id.is_unique,
            "authorized event_id uniqueness failed")
    news = oof[oof.source_family.isin(NEWS_SOURCES)].copy()
    news = news.merge(
        labels.rename(columns={"source_family": "label_source_family", "y": "dev_y",
                               "fwd_ret_30m": "dev_fwd_ret_30m"}),
        on="event_id", how="left", validate="one_to_one",
    )
    news = news.merge(
        metadata.rename(columns={"source_family": "metadata_source_family",
                                 "ticker": "metadata_ticker", "event_group_id": "metadata_event_group_id",
                                 "event_time_utc": "metadata_event_time_utc", "form": "metadata_form"}),
        on="event_id", how="left", validate="one_to_one",
    )
    require(news.source.notna().all(), "news metadata join is incomplete")
    require(np.array_equal(news.y.to_numpy(int), news.dev_y.to_numpy(int)), "V69/DEV labels differ")
    require(np.allclose(news.fwd_ret_30m, news.dev_fwd_ret_30m, rtol=0.0, atol=1e-15),
            "V69/DEV returns differ")
    require(news.source_family.equals(news.label_source_family), "label source family differs")
    require(news.source_family.equals(news.metadata_source_family), "metadata source family differs")
    for source, expected in EXPECTED_ROWS.items():
        require(int(news.source_family.eq(source).sum()) == expected, f"{source} row count changed")
    news["event_time_utc"] = pd.to_datetime(news.event_time_utc, utc=True)
    news["metadata_event_time_utc"] = pd.to_datetime(news.metadata_event_time_utc, utc=True)
    require(news.event_time_utc.equals(news.metadata_event_time_utc), "metadata event time differs")
    news["entry_target_utc"] = pd.to_datetime(news.entry_target_utc, utc=True)
    news["decision_delay_seconds"] = (
        news.entry_target_utc - news.event_time_utc
    ).dt.total_seconds()
    require(news.decision_delay_seconds.eq(120.0).all(), "decision target is not exactly t+2")
    news["high_conf"] = bool_series(news.high_conf)
    news["direction_prediction"] = news.prob.to_numpy(float) >= 0.5
    news["correct_target"] = (
        news.direction_prediction.to_numpy(bool) == news.y.to_numpy(int).astype(bool)
    ).astype(int)
    news["url_host"] = news.url.map(url_host)
    news["publisher_host_class"] = news.url_host.map(host_class)
    news["provider_class"] = news.source.map(provider_class)
    news["article_key"] = [
        stable_article_key(article, url) for article, url in zip(news.article_id, news.url)
    ]
    news["event_type_clean"] = news.event_type.map(clean_event_type)
    news["ticker_clean"] = news.ticker.fillna("<MISSING>").astype(str).str.upper()
    return news.sort_values(["source_family", "event_time_utc", "event_id"], kind="mergesort").reset_index(drop=True)


def text_shape(series: pd.Series, prefix: str) -> pd.DataFrame:
    text = series.fillna("").astype(str)
    length = text.str.len().astype(float)
    denominator = length.clip(lower=1.0)
    return pd.DataFrame({
        f"{prefix}_present": text.str.strip().ne("").astype(float),
        f"{prefix}_char_n": length,
        f"{prefix}_word_n": text.str.count(r"\S+").astype(float),
        f"{prefix}_digit_fraction": text.str.count(r"[0-9]").astype(float) / denominator,
        f"{prefix}_hangul_fraction": text.str.count(r"[가-힣]").astype(float) / denominator,
        f"{prefix}_uppercase_fraction": text.str.count(r"[A-Z]").astype(float) / denominator,
        f"{prefix}_question_n": text.str.count(r"\?").astype(float),
        f"{prefix}_exclamation_n": text.str.count("!").astype(float),
    }, index=series.index)


def static_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    probabilities = []
    aligned = []
    base_up = frame.direction_prediction.to_numpy(bool)
    for column in EXPERT_PROBABILITY_COLUMNS:
        value = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        require(np.isfinite(value).all(), f"non-finite strict OOF probability: {column}")
        output[f"oof_{column}"] = value
        output[f"margin_{column}"] = 2.0 * np.abs(value - 0.5)
        output[f"agrees_direction_{column}"] = ((value >= 0.5) == base_up).astype(float)
        probabilities.append(value)
        aligned.append(np.where(base_up, value, 1.0 - value))
    matrix = np.column_stack(probabilities)
    aligned_matrix = np.column_stack(aligned)
    output["expert_probability_std"] = matrix.std(axis=1)
    output["expert_probability_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    output["expert_aligned_mean"] = aligned_matrix.mean(axis=1)
    output["expert_aligned_min"] = aligned_matrix.min(axis=1)
    output["direction_margin"] = 2.0 * np.abs(frame.prob.to_numpy(float) - 0.5)
    p = np.clip(frame.prob.to_numpy(float), 1e-8, 1.0 - 1e-8)
    output["direction_entropy"] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    output["predicted_up"] = base_up.astype(float)
    for column in EXPERT_CONFIDENCE_COLUMNS:
        output[f"oof_{column}"] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    output["url_available"] = present(frame.url).astype(float)
    output["article_id_available"] = present(frame.article_id).astype(float)
    output["timestamp_exact"] = frame.timestamp_quality.fillna("").astype(str).str.upper().eq("EXACT").astype(float)
    output = pd.concat([
        output, text_shape(frame.headline, "headline"), text_shape(frame.body, "body")
    ], axis=1)
    timestamp = pd.to_datetime(frame.event_time_utc, utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    output["event_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    output["event_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    output["event_weekday_sin"] = np.sin(2.0 * np.pi * timestamp.dt.weekday / 7.0)
    output["event_weekday_cos"] = np.cos(2.0 * np.pi * timestamp.dt.weekday / 7.0)
    output["provider_class"] = frame.provider_class.to_numpy()
    output["publisher_host_class"] = frame.publisher_host_class.to_numpy()
    output["event_type_clean"] = frame.event_type_clean.to_numpy()
    output["source_family_feature"] = frame.source_family.to_numpy()
    return output.replace([np.inf, -np.inf], np.nan)


def history_feature_frame(
    history: pd.DataFrame,
    target: pd.DataFrame,
    keys: Sequence[str] = HISTORY_KEYS,
    smoothing: float = SMOOTHING,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build outcome-history features using only labels at least 35m old.

    The history may equal target for prequential training features.  The strict
    time predicate prevents the current row or an outcome not yet observable at
    its t+2 decision from entering any feature.
    """
    history_ordered = history.sort_values(["event_time_utc", "event_id"], kind="mergesort")
    target_ordered = target.sort_values(["event_time_utc", "event_id"], kind="mergesort")
    success: dict[str, defaultdict[str, float]] = {
        key: defaultdict(float) for key in keys
    }
    weight_sum: dict[str, defaultdict[str, float]] = {
        key: defaultdict(float) for key in keys
    }
    row_count: dict[str, defaultdict[str, int]] = {
        key: defaultdict(int) for key in keys
    }
    global_success = 0.0
    global_weight = 0.0
    global_count = 0
    pointer = 0
    history_records = list(history_ordered.itertuples(index=False))
    output_rows: dict[int, dict[str, float]] = {}
    minimum_gap_minutes: float | None = None
    same_event_reference_n = 0
    for target_index, row in target_ordered.iterrows():
        cutoff = pd.Timestamp(row.event_time_utc) - EMBARGO
        while pointer < len(history_records):
            candidate = history_records[pointer]
            candidate_time = pd.Timestamp(candidate.event_time_utc)
            if not candidate_time < cutoff:
                break
            # A future full-sample duplicate count would itself leak.  History
            # priors therefore use unit weights; model fitting separately uses
            # duplicate weights computed from its outer-train rows only.
            candidate_weight = 1.0
            candidate_target = int(candidate.correct_target)
            global_success += candidate_weight * candidate_target
            global_weight += candidate_weight
            global_count += 1
            gap = (pd.Timestamp(row.event_time_utc) - candidate_time).total_seconds() / 60.0
            minimum_gap_minutes = gap if minimum_gap_minutes is None else min(minimum_gap_minutes, gap)
            if str(candidate.event_id) == str(row.event_id):
                same_event_reference_n += 1
            for key in keys:
                value = str(getattr(candidate, key))
                success[key][value] += candidate_weight * candidate_target
                weight_sum[key][value] += candidate_weight
                row_count[key][value] += 1
            pointer += 1
        features: dict[str, float] = {
            "history_global_count": float(global_count),
            "history_global_correct_rate": (
                (global_success + smoothing * 0.5) / (global_weight + smoothing)
            ),
        }
        for key in keys:
            value = str(row[key])
            denominator = weight_sum[key][value]
            features[f"history_count__{key}"] = float(row_count[key][value])
            features[f"history_correct_rate__{key}"] = (
                success[key][value] + smoothing * 0.5
            ) / (denominator + smoothing)
        output_rows[int(target_index)] = features
    result = pd.DataFrame.from_dict(output_rows, orient="index").reindex(target.index)
    audit = {
        "history_rows": len(history), "target_rows": len(target),
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "minimum_used_history_gap_minutes": minimum_gap_minutes,
        "same_event_reference_n": same_event_reference_n,
        "same_row_correctness_used": False,
        "strict_less_than_cutoff": True,
    }
    require(same_event_reference_n == 0, "same event leaked into history features")
    require(minimum_gap_minutes is None or minimum_gap_minutes > 35.0,
            "outcome-history feature violates strict 35-minute lag")
    return result, audit


def frozen_history_feature_frame(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    keys: Sequence[str] = HISTORY_KEYS,
    smoothing: float = SMOOTHING,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # The generic as-of builder adds only train labels old enough for each
    # validation row.  No validation target is accepted by the function.
    return history_feature_frame(train, valid, keys=keys, smoothing=smoothing)


def feature_frames(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_static = static_feature_frame(train)
    valid_static = static_feature_frame(valid)
    train_history, train_audit = history_feature_frame(train, train)
    valid_history, valid_audit = frozen_history_feature_frame(train, valid)
    x_train = pd.concat([train_static, train_history], axis=1)
    x_valid = pd.concat([valid_static, valid_history], axis=1)
    require(list(x_train.columns) == list(x_valid.columns), "train/valid feature schema differs")
    require(not (set(x_train.columns) & FORBIDDEN_MODEL_COLUMNS), "outcome column entered model features")
    require(not (set(x_train.columns) & IDENTIFIER_COLUMNS), "raw identifier/content entered model features")
    return x_train, x_valid, {"train_history": train_audit, "valid_history": valid_audit}


def purged_outer_folds(frame: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int, dict[str, Any]]]:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="mergesort").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(ordered)), OUTER_BLOCKS)
    output = []
    for fold in range(1, OUTER_BLOCKS):
        valid = ordered.iloc[blocks[fold]].copy()
        boundary = pd.Timestamp(valid.event_time_utc.min())
        train = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
        duplicate_keys = set(valid.article_key.astype(str)) - {"<MISSING>"}
        before_purge = len(train)
        train = train[~train.article_key.astype(str).isin(duplicate_keys)].copy()
        overlap = (set(train.article_key.astype(str)) - {"<MISSING>"}) & duplicate_keys
        require(not overlap, "article URL/ID cluster crosses outer train/valid")
        require(len(train) >= 100 and len(valid) >= 100, f"outer fold {fold} support insufficient")
        require(train.correct_target.nunique() == 2 and valid.correct_target.nunique() == 2,
                f"outer fold {fold} correctness class support insufficient")
        train_end = pd.Timestamp(train.event_time_utc.max())
        valid_start = pd.Timestamp(valid.event_time_utc.min())
        require(train_end < valid_start - EMBARGO, "outer fold violates strict embargo")
        output.append((train.reset_index(drop=True), valid.reset_index(drop=True), fold, {
            "fold": fold, "train_n": len(train), "valid_n": len(valid),
            "article_purge_n": before_purge - len(train),
            "article_overlap_n": 0, "train_end": train_end,
            "valid_start": valid_start, "embargo_minutes": 35.0,
        }))
    return output


def safe_auc(target: Iterable[int], score: Iterable[float]) -> float | None:
    y = np.asarray(list(target), dtype=int)
    value = np.asarray(list(score), dtype=float)
    return float(roc_auc_score(y, value)) if len(y) and np.unique(y).size == 2 else None


def ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    target = frame.correct_target.to_numpy(int)
    score = frame[score_column].to_numpy(float)
    order = np.argsort(-score, kind="stable")

    def top(fraction: float) -> dict[str, Any]:
        count = max(1, int(math.ceil(len(frame) * fraction)))
        chosen = order[:count]
        return {"n": count, "coverage": count / len(frame),
                "accuracy": float(target[chosen].mean())}

    bins = pd.DataFrame({"target": target, "score": score}).sort_values(
        "score", kind="mergesort"
    ).reset_index(drop=True)
    bins["bin"] = np.minimum(9, np.arange(len(bins)) * 10 // len(bins)) + 1
    bin_rows = [{
        "bin": int(number), "n": len(group),
        "mean_score": float(group.score.mean()), "accuracy": float(group.target.mean()),
    } for number, group in bins.groupby("bin", sort=True)]
    return {
        "n": len(frame), "correctness_rate": float(target.mean()),
        "correctness_auc": safe_auc(target, score),
        "brier": float(brier_score_loss(target, np.clip(score, 0.0, 1.0))),
        "top10": top(0.10), "top20": top(0.20), "decile_bins": bin_rows,
    }


def run_strict_outer_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = []
    audits: dict[str, Any] = {}
    for source_index, source in enumerate(NEWS_SOURCES):
        source_frame = frame[frame.source_family.eq(source)].copy()
        audits[source] = {"rows": len(source_frame), "folds": []}
        for train, valid, fold, chronology in purged_outer_folds(source_frame):
            x_train, x_valid, feature_audit = feature_frames(train, valid)
            categorical = [
                "provider_class", "publisher_host_class", "event_type_clean",
                "source_family_feature",
            ]
            numeric = [column for column in x_train.columns if column not in categorical]
            transformer = ColumnTransformer([
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                             keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
            ])
            model = Pipeline([
                ("features", transformer),
                ("model", LogisticRegression(
                    C=0.25, solver="liblinear", class_weight="balanced",
                    max_iter=1000, random_state=SEED + source_index * 100 + fold,
                )),
            ])
            train_article_weight = (
                1.0 / train.groupby("article_key").event_id.transform("size").clip(lower=1)
            ).to_numpy(float)
            model.fit(
                x_train, train.correct_target.to_numpy(int),
                model__sample_weight=train_article_weight,
            )
            probability = np.clip(model.predict_proba(x_valid)[:, 1], 1e-6, 1.0 - 1e-6)
            output = valid[[
                "event_id", "event_time_utc", "source_family", "fold", "prob",
                "confidence_signal", "direction_prediction", "correct_target",
                "provider_class", "publisher_host_class",
            ]].copy()
            output["p_correct_news"] = probability
            output["p_correct_outer_fold"] = fold
            output["correctness_model_fit_contains_event"] = False
            predictions.append(output)
            fold_metrics = ranking_metrics(output, "p_correct_news")
            baseline_metrics = ranking_metrics(output, "confidence_signal")
            audits[source]["folds"].append({
                "outer_fold": fold, "chronology": chronology,
                "feature_history": feature_audit,
                "feature_n": len(x_train.columns), "numeric_feature_n": len(numeric),
                "categorical_feature_n": len(categorical),
                "fixed_model": "L2_LOGISTIC_C_0.25_BALANCED",
                "selection_on_outer_labels": False,
                "same_row_fitted_correctness": False,
                "p_correct_metrics": fold_metrics,
                "v69_confidence_metrics_same_rows": baseline_metrics,
            })
    evidence = pd.concat(predictions, ignore_index=True)
    require(not evidence.correctness_model_fit_contains_event.any(),
            "same-row fitted correctness detected")
    for source in NEWS_SOURCES:
        part = evidence[evidence.source_family.eq(source)]
        audits[source]["evaluated_rows"] = len(part)
        audits[source]["p_correct_metrics"] = ranking_metrics(part, "p_correct_news")
        audits[source]["v69_confidence_metrics_same_rows"] = ranking_metrics(
            part, "confidence_signal"
        )
    audits["OVERALL"] = {
        "evaluated_rows": len(evidence),
        "p_correct_metrics": ranking_metrics(evidence, "p_correct_news"),
        "v69_confidence_metrics_same_rows": ranking_metrics(evidence, "confidence_signal"),
    }
    return evidence, audits


def inventory_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = ["OVERALL", *NEWS_SOURCES]
    fields = [
        ("publisher", None, "ABSENT_CANONICAL_COLUMN"),
        ("publisher_host_proxy", "publisher_host_class", "DERIVED_URL_HOST_NOT_ORIGINAL_PUBLISHER"),
        ("provider", "provider_class", "SANITIZED_FROM_SOURCE_PROVENANCE"),
        ("url", "url", "PRESENT_BUT_RAW_IDENTIFIER_EXCLUDED_FROM_MODEL"),
        ("article_id", "article_id", "SPARSE_URL_HASH_FALLBACK_FOR_PURGE_ONLY"),
        ("event_timestamp", "event_time_utc", "EXACT_EVENT_TIME_CONTRACT"),
        ("published_at", None, "ABSENT_SEPARATE_COLUMN"),
        ("ingested_at", None, "ABSENT_CAPTURE_PROOF"),
        ("headline", "headline", "CONTRACT_TPLUS2_CONTENT_NO_INGEST_TIMESTAMP"),
        ("body", "body", "CONTRACT_TPLUS2_CONTENT_NO_INGEST_TIMESTAMP"),
        ("text", "text", "CONTRACT_TPLUS2_CONTENT_NO_INGEST_TIMESTAMP"),
        ("timestamp_quality", "timestamp_quality", "EXACT_FLAG_AVAILABLE"),
    ]
    for scope in scopes:
        part = frame if scope == "OVERALL" else frame[frame.source_family.eq(scope)]
        for name, column, status in fields:
            if column is None:
                count = unique = 0
                rate = 0.0
                median_chars = p90_chars = None
            else:
                available = part[column].notna() if pd.api.types.is_datetime64_any_dtype(part[column]) else present(part[column])
                count = int(available.sum())
                unique = int(part.loc[available, column].nunique())
                rate = float(available.mean())
                if name in {"headline", "body", "text"}:
                    lengths = part.loc[available, column].astype(str).str.len()
                    median_chars = finite(lengths.median())
                    p90_chars = finite(lengths.quantile(0.90))
                else:
                    median_chars = p90_chars = None
            rows.append({
                "scope": scope, "field": name, "rows": len(part),
                "present_n": count, "present_rate": rate, "unique_n": unique,
                "median_chars": median_chars, "p90_chars": p90_chars,
                "audit_status": status,
            })
    return pd.DataFrame(rows)


def feature_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    sample = frame.iloc[:min(10, len(frame))].copy()
    static = static_feature_frame(sample)
    history_names = ["history_global_count", "history_global_correct_rate"]
    for key in HISTORY_KEYS:
        history_names.extend([f"history_count__{key}", f"history_correct_rate__{key}"])
    rows = []
    categorical = {
        "provider_class", "publisher_host_class", "event_type_clean", "source_family_feature"
    }
    for column in static.columns:
        if column.startswith("headline_") or column.startswith("body_"):
            timing = "TPLUS2_CONTRACT_ELIGIBLE_CAPTURE_TIMESTAMP_NOT_PROVEN"
            origin = "immutable DEV event content shape"
        elif column in categorical:
            timing = "TPLUS2_EVENT_PROVENANCE"
            origin = "sanitized event metadata"
        elif column.startswith("oof_") or column.startswith("margin_") or column.startswith("agrees_"):
            timing = "STRICT_V69_OUTER_OOF_AT_DECISION"
            origin = "frozen direction/expert output"
        elif column.startswith("expert_") or column.startswith("direction_") or column == "predicted_up":
            timing = "STRICT_V69_OUTER_OOF_AT_DECISION"
            origin = "derived from frozen direction/expert output"
        else:
            timing = "TPLUS2_EVENT_METADATA"
            origin = "immutable DEV event metadata"
        rows.append({
            "feature": column, "kind": "categorical" if column in categorical else "numeric",
            "origin": origin, "decision_time_contract": timing,
            "correctness_label_dependency": "NONE",
            "leakage_control": "target-free row feature",
            "baseline_included": True,
        })
    for column in history_names:
        key = column.split("__", 1)[1] if "__" in column else "GLOBAL"
        rows.append({
            "feature": column, "kind": "numeric",
            "origin": f"matured prior correctness history: {key}",
            "decision_time_contract": "OUTCOME_AVAILABLE_STRICTLY_MORE_THAN_35_MINUTES_EARLIER",
            "correctness_label_dependency": "PAST_ONLY",
            "leakage_control": "prequential training; frozen outer-train mapping for validation",
            "baseline_included": True,
        })
    for excluded, reason in (
        ("url", "raw identifier"), ("article_id", "raw identifier"),
        ("article_key", "purge/history key only"), ("headline", "raw content"),
        ("body", "raw content"), ("text", "raw content"),
        ("source", "contains historical allocation/version tokens"),
        ("y", "direction outcome"), ("fwd_ret_30m", "future return"),
        ("correct_target", "correctness outcome"),
    ):
        rows.append({
            "feature": excluded, "kind": "excluded", "origin": reason,
            "decision_time_contract": "NOT_A_MODEL_FEATURE",
            "correctness_label_dependency": "FORBIDDEN" if excluded in FORBIDDEN_MODEL_COLUMNS else "NONE",
            "leakage_control": "excluded",
            "baseline_included": False,
        })
    return pd.DataFrame(rows)


def prepared_feature_export(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces = []
    audits = {}
    for source in NEWS_SOURCES:
        part = frame[frame.source_family.eq(source)].copy()
        static = static_feature_frame(part)
        history, audit = history_feature_frame(part, part)
        features = pd.concat([static, history], axis=1)
        output = part[[
            "event_id", "event_time_utc", "source_family", "fold",
            "direction_prediction", "correct_target",
        ]].copy()
        output = pd.concat([output, features], axis=1)
        output["history_recipe"] = "PREQUENTIAL_STRICT_GT_35M"
        output["same_row_correctness_used"] = False
        pieces.append(output)
        audits[source] = audit
    result = pd.concat(pieces, ignore_index=True)
    require(not result.same_row_correctness_used.any(), "prepared export includes same-row correctness")
    return result, audits


def inventory_summary(frame: pd.DataFrame, inventory: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source in NEWS_SOURCES:
        part = frame[frame.source_family.eq(source)]
        source_inventory = inventory[inventory.scope.eq(source)].set_index("field")
        summary[source] = {
            "rows": len(part), "time_start": part.event_time_utc.min(),
            "time_end": part.event_time_utc.max(),
            "v69_fold_counts": {str(k): int(v) for k, v in part.fold.value_counts().sort_index().items()},
            "provider_class_counts": {
                str(k): int(v) for k, v in part.provider_class.value_counts().items()
            },
            "publisher_host_class_counts": {
                str(k): int(v) for k, v in part.publisher_host_class.value_counts().items()
            },
            "availability": {
                field: {
                    "present_n": int(source_inventory.loc[field, "present_n"]),
                    "present_rate": float(source_inventory.loc[field, "present_rate"]),
                    "unique_n": int(source_inventory.loc[field, "unique_n"]),
                    "status": str(source_inventory.loc[field, "audit_status"]),
                } for field in (
                    "publisher", "publisher_host_proxy", "url", "article_id",
                    "event_timestamp", "published_at", "ingested_at", "headline", "body",
                )
            },
            "article_key_unique_n": int(part.article_key.nunique()),
            "article_key_duplicate_row_n": int(part.article_key.duplicated(False).sum()),
            "correctness_rate": float(part.correct_target.mean()),
        }
    return summary


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V224 News Confidence Inventory and Strict-OOF Baseline",
        "",
        f"Status: `{report['status']}`",
        "",
        "Only immutable V36 DEV metadata/labels and the atomically committed V69 outer-OOF "
        "direction frame were opened. DEV_EXTENSION, role assignment, Research Seal, and Final "
        "Meta were not opened.",
        "",
        "## Leakage contract",
        "",
        "- Correctness target: `I[(V69 outer-OOF probability >= 0.5) == y]`.",
        "- Every reported P_CORRECT_NEWS value is from a later purged outer fold.",
        "- Correctness-history features use labels strictly more than 35 minutes old.",
        "- Validation history is built from outer-train rows only.",
        "- Raw URL, article ID/key, headline, body, source-version token, labels, and returns are not model inputs.",
        "- Article URL/ID keys are used only for duplicate purge and past support counts.",
        "",
        "## Inventory blockers",
        "",
    ]
    for blocker in report["blockers"]:
        lines.append(f"- `{blocker['code']}`: {blocker['detail']}")
    lines.extend(["", "## Baseline", ""])
    for source in (*NEWS_SOURCES, "OVERALL"):
        result = report["baseline"][source]
        candidate = result["p_correct_metrics"]
        baseline = result["v69_confidence_metrics_same_rows"]
        lines.append(
            f"- {source}: n={result['evaluated_rows']}, P_CORRECT AUC={candidate['correctness_auc']}, "
            f"top20 accuracy={candidate['top20']['accuracy']}; V69 confidence AUC on the same rows="
            f"{baseline['correctness_auc']}."
        )
    lines.extend([
        "", "This baseline is diagnostic only. Missing canonical publisher identity and capture/ingest "
        "timestamps block a production publisher/body discriminator and independent t+2 provenance proof.", "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="inventory only; skip outer baseline")
    parser.add_argument("--cpu-ids", default="30,31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_ids = tuple(int(piece.strip()) for piece in args.cpu_ids.split(",") if piece.strip())
    runtime = configure_runtime(cpu_ids)
    authority = verify_authority()
    frame = load_news_frame()
    inventory = inventory_table(frame)
    catalog = feature_catalog(frame)
    prepared, preparation_audit = prepared_feature_export(frame)
    if args.audit_only:
        evidence = pd.DataFrame()
        baseline: dict[str, Any] = {}
    else:
        evidence, baseline = run_strict_outer_baseline(frame)
    blockers = [
        {
            "code": "CANONICAL_PUBLISHER_ABSENT",
            "detail": (
                "DEV has no publisher field. reuters.com is direct, but Naver/Google hosts are "
                "aggregators and cannot identify the original publisher without new metadata."
            ),
        },
        {
            "code": "NO_CAPTURE_OR_INGEST_TIMESTAMP",
            "detail": (
                "event_time_utc is EXACT and entry_target is exactly t+2, but no separate fetched_at/"
                "ingested_at exists; URL/headline/body availability by t+2 is contractual, not independently provable."
            ),
        },
        {
            "code": "US_ARTICLE_ID_SPARSE",
            "detail": "US_NEWS article_id coverage is 115/954; normalized URL hashes are purge keys only.",
        },
        {
            "code": "NO_FRESH_NEWS_REPLICATION",
            "detail": "This run is immutable old DEV only; no extension, Seal, or Final data were opened.",
        },
    ]
    report = {
        "version": "V224_NEWS_CONFIDENCE_INVENTORY_BASELINE_V1",
        "status": (
            "AUDIT_ONLY_BLOCKED_FOR_PRODUCTION" if args.audit_only
            else "STRICT_OUTER_OOF_BASELINE_COMPLETE_BLOCKED_FOR_PRODUCTION"
        ),
        "diagnostic_only": True, "authority": authority, "runtime": runtime,
        "contract": {
            "direction": "atomically committed V69 outer OOF; immutable",
            "correctness_target": "I[(V69 outer-OOF probability >= 0.5) == y]",
            "decision_time": "entry_target_utc = event_time_utc + 2 minutes",
            "outcome_history_lag": "strictly more than 35 minutes in event time",
            "same_row_fitted_correctness_allowed": False,
            "same_row_fitted_correctness_observed": False,
            "outer_labels_used_for_training_or_selection": False,
            "raw_url_or_article_id_model_input": False,
            "raw_headline_or_body_model_input": False,
        },
        "inventory": inventory_summary(frame, inventory),
        "feature_preparation_audit": preparation_audit,
        "baseline": baseline, "blockers": blockers,
        "production_authorized": False, "seal_authorized": False,
    }
    atomic_csv(inventory, HERE / "NEWS_DEV_FIELD_INVENTORY.csv")
    atomic_csv(catalog, HERE / "NEWS_TPLUS2_FEATURE_CATALOG.csv")
    atomic_csv(prepared, HERE / "P_CORRECT_NEWS_FEATURE_PREPARATION.csv.gz", compression="gzip")
    if not evidence.empty:
        atomic_csv(evidence, HERE / "P_CORRECT_NEWS_STRICT_OUTER_OOF.csv.gz", compression="gzip")
    atomic_json(report, HERE / "P_CORRECT_NEWS_BASELINE_REPORT.json")
    markdown = HERE / "P_CORRECT_NEWS_AUDIT.md"
    temporary = markdown.with_suffix(markdown.suffix + ".tmp")
    temporary.write_text(audit_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    output_names = [
        "NEWS_DEV_FIELD_INVENTORY.csv", "NEWS_TPLUS2_FEATURE_CATALOG.csv",
        "P_CORRECT_NEWS_FEATURE_PREPARATION.csv.gz", "P_CORRECT_NEWS_BASELINE_REPORT.json",
        "P_CORRECT_NEWS_AUDIT.md",
    ]
    if not evidence.empty:
        output_names.append("P_CORRECT_NEWS_STRICT_OUTER_OOF.csv.gz")
    artifact_hashes = {name: sha256(HERE / name) for name in sorted(output_names)}
    manifest = {
        "version": report["version"], "status": report["status"],
        "script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "tests": {
            "path": "test_prepare_p_correct_news_baseline.py",
            "sha256": sha256(HERE / "test_prepare_p_correct_news_baseline.py"),
        },
        "authorized_input_sha256": authority["sha256"],
        "artifacts": artifact_hashes,
        "invariants": {
            "same_row_fitted_correctness": False, "outer_labels_used_for_selection": False,
            "seal_or_final_opened": False, "existing_output_or_registry_modified": False,
        },
    }
    atomic_json(manifest, HERE / "ARTIFACT_MANIFEST.json")
    print(json.dumps(json_clean({
        "status": report["status"], "news_rows": len(frame),
        "evaluated_outer_oof_rows": len(evidence), "baseline": baseline,
        "blockers": [row["code"] for row in blockers],
        "artifact_sha256": {**artifact_hashes, "ARTIFACT_MANIFEST.json": sha256(HERE / "ARTIFACT_MANIFEST.json")},
        "same_row_fitted_correctness": False, "seal_or_final_opened": False,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
