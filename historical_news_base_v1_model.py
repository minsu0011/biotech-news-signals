"""Strict nested-OOF P(correct) research on the frozen news BASE V1.

The source membership and outer source-family folds must already be frozen.
This script reads only immutable V36 DEV outcomes and immutable V69 outer-OOF
direction predictions.  It never changes direction probabilities, the V224
prospective namespace, Research Seal, or Final Meta.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
RESEARCH = ROOT / "research" / "high_conf_news_base_v1"
V69_OOF = ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
EMBED_ROOT = ROOT / "data" / "historical_news_backfill" / "embeddings"
CANDIDATE_SPLIT = RESEARCH / "HIST_NEWS_MODEL_CANDIDATE_SPLIT_MANIFEST.parquet"
CANDIDATE_SPLIT_FREEZE = RESEARCH / "MODEL_CANDIDATE_SPLIT_FREEZE.json"
NEWS_SOURCES = ("US_NEWS", "KR_NEWS")
SEED = 20260831
TRADE_COST = 0.002
HC_QUANTILE = 0.80
ALPHAS = (-1.0, 0.0, 0.5, 1.0, 1.5)
ALPHA_TIE_ORDER = {1.0: 5, 0.5: 4, 0.0: 3, -1.0: 2, 1.5: 1}

MODEL_INPUT_FILES = (
    "SOURCE_FREEZE.json",
    "SHA256_MANIFEST.csv",
    "POST_FREEZE_LABEL_JOIN_STATUS.json",
    "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet",
    "ARTICLES_NORMALIZED.parquet",
    "EMBEDDING_INDEX.parquet",
    "HIST_NEWS_DEV_SPLIT_MANIFEST.parquet",
)


class ModelResearchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelResearchError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    temp.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    frame.to_csv(temp, index=False, lineterminator="\n")
    temp.replace(path)


def safe_auc(target: Iterable[int], score: Iterable[float]) -> float | None:
    y = np.asarray(list(target), dtype=int)
    p = np.asarray(list(score), dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def expected_calibration_error(target: Sequence[int], score: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(target, dtype=int)
    p = np.asarray(score, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def signed_net(frame: pd.DataFrame) -> np.ndarray:
    direction = np.where(frame["direction_prediction"].astype(bool), 1.0, -1.0)
    return direction * frame["fwd_ret_30m"].astype(float).to_numpy() - TRADE_COST


def bootstrap_mean(values: Sequence[float], *, seed: int, draws: int = 2000) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"draws": draws, "lower95": None, "median": None, "upper95": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, median, high = np.quantile(samples, [0.025, 0.5, 0.975])
    return {"draws": draws, "lower95": float(low), "median": float(median), "upper95": float(high)}


def score_metrics(frame: pd.DataFrame, score_column: str, high_column: str, *, seed: int) -> dict[str, Any]:
    y = frame["correct_target"].astype(int).to_numpy()
    p = np.clip(frame[score_column].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    high = frame[high_column].astype(bool).to_numpy()
    net = frame["signed_net"].astype(float).to_numpy()
    selected_net = net[high]
    top_sorted = np.sort(selected_net)[::-1]
    winsor = np.clip(
        selected_net,
        np.quantile(selected_net, 0.01) if len(selected_net) else 0.0,
        np.quantile(selected_net, 0.99) if len(selected_net) else 0.0,
    )
    quantile_rows = []
    order = np.argsort(-p)
    for coverage in (0.15, 0.20, 0.30, 0.50):
        count = max(1, int(math.ceil(len(frame) * coverage))) if len(frame) else 0
        chosen = order[:count]
        quantile_rows.append(
            {
                "coverage_target": coverage,
                "n": int(count),
                "accuracy": float(y[chosen].mean()) if count else None,
                "mean_net": float(net[chosen].mean()) if count else None,
            }
        )
    return {
        "n": int(len(frame)),
        "correct_rate": float(y.mean()) if len(y) else None,
        "correctness_auc": safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "ece_10": expected_calibration_error(y, p),
        "hc_n": int(high.sum()),
        "hc_coverage": float(high.mean()) if len(high) else None,
        "hc_accuracy": float(y[high].mean()) if high.any() else None,
        "hc_net": float(selected_net.mean()) if len(selected_net) else None,
        "bootstrap_hc_net": bootstrap_mean(selected_net, seed=seed),
        "hc_winsorized_net": float(winsor.mean()) if len(winsor) else None,
        "hc_top1_removed_net": float(top_sorted[1:].mean()) if len(top_sorted) > 1 else None,
        "hc_top5_removed_net": float(top_sorted[5:].mean()) if len(top_sorted) > 5 else None,
        "risk_coverage": quantile_rows,
    }


def verify_authority() -> dict[str, Any]:
    for name in MODEL_INPUT_FILES:
        require((DATA / name).is_file(), f"missing frozen model input: {name}")
    freeze = json.loads((DATA / "SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    join = json.loads((DATA / "POST_FREEZE_LABEL_JOIN_STATUS.json").read_text(encoding="utf-8"))
    require(bool(freeze.get("source_membership_frozen")), "source membership is not frozen")
    require(bool(freeze.get("outcome_blind")), "source freeze is not outcome-blind")
    require(bool(freeze.get("qa_pass")), "source freeze QA did not pass")
    require(bool(join.get("source_freeze_unchanged")), "post-freeze join changed source freeze")
    require(bool(join.get("model_ready")), "post-freeze label join is not model-ready")
    manifest = pd.read_csv(DATA / "SHA256_MANIFEST.csv")
    for row in manifest.to_dict(orient="records"):
        path = DATA / str(row["file"])
        require(path.is_file(), f"manifest member missing: {path.name}")
        require(sha256_file(path) == str(row["sha256"]), f"manifest hash mismatch: {path.name}")
    require(V69_OOF.is_file(), "immutable V69 outer-OOF file missing")
    require(CANDIDATE_SPLIT.is_file() and CANDIDATE_SPLIT_FREEZE.is_file(), "label-blind model-candidate split missing")
    candidate_freeze = json.loads(CANDIDATE_SPLIT_FREEZE.read_text(encoding="utf-8"))
    require(bool(candidate_freeze.get("label_blind")), "model-candidate split is not label-blind")
    require(candidate_freeze.get("outcome_columns_read") == [], "model-candidate split read outcomes")
    require(candidate_freeze.get("manifest_file_sha256") == sha256_file(CANDIDATE_SPLIT), "model-candidate split hash mismatch")
    require(candidate_freeze.get("source_freeze_sha256") == sha256_file(DATA / "SOURCE_FREEZE.json"), "candidate split/source freeze mismatch")
    return {
        "source_freeze_sha256": sha256_file(DATA / "SOURCE_FREEZE.json"),
        "split_sha256": freeze.get("split_manifest_sha256"),
        "labeled_sha256": sha256_file(DATA / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet"),
        "v69_outer_oof_sha256": sha256_file(V69_OOF),
        "model_candidate_split_sha256": sha256_file(CANDIDATE_SPLIT),
        "model_candidate_split_semantic_sha256": candidate_freeze.get("manifest_semantic_sha256"),
        "source_manifest_verified_files": int(len(manifest)),
        "research_seal_loaded": False,
        "final_meta_loaded": False,
        "v224_prospective_loaded": False,
    }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _ticker_key(value: Any, market: Any) -> str:
    ticker = _clean(value).upper()
    if _clean(market).upper() == "KR" and ticker.isdigit():
        return ticker.zfill(6)
    return ticker


def map_v69_to_frozen_events(direction: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = events.copy()
    lookup: dict[str, set[int]] = {}
    for idx, row in events.iterrows():
        for column in ("canonical_event_id", "existing_event_id", "join_key_event_id"):
            key = _clean(row.get(column))
            if key:
                lookup.setdefault(key, set()).add(int(idx))
    mapped_rows = []
    outside = 0
    collision_key_resolved = 0
    collision_exact_resolved = 0
    for row in direction.to_dict(orient="records"):
        raw_id = _clean(row.get("event_id"))
        candidates = events.loc[sorted(lookup.get(raw_id, set()))]
        if candidates.empty:
            outside += 1
            continue
        if len(candidates) > 1:
            market = _clean(row.get("market")).upper()
            ticker = _ticker_key(row.get("ticker"), market)
            event_time = pd.to_datetime(row.get("event_time_utc"), utc=True, errors="coerce")
            ticker_match = pd.Series(
                [_ticker_key(value, candidate_market) for value, candidate_market in zip(candidates["ticker"], candidates["market"])],
                index=candidates.index,
            ).eq(ticker)
            candidates = candidates.loc[
                candidates["market"].astype(str).str.upper().eq(market)
                & ticker_match
                & ((pd.to_datetime(candidates["event_time_utc"], utc=True) - event_time).abs() <= pd.Timedelta(seconds=1))
            ]
            if len(candidates) == 1:
                collision_key_resolved += 1
            elif len(candidates) > 1:
                exact = candidates.loc[candidates["exact_price_eligible"].fillna(False).astype(bool)]
                if len(exact) == 1:
                    candidates = exact
                    collision_exact_resolved += 1
            require(len(candidates) == 1, f"ambiguous V69 direction event mapping: {raw_id}")
        output = dict(row)
        output["direction_event_id_raw"] = raw_id
        output["canonical_event_id"] = str(candidates.iloc[0]["canonical_event_id"])
        mapped_rows.append(output)
    mapped = pd.DataFrame(mapped_rows)
    require(not mapped["canonical_event_id"].duplicated().any(), "multiple V69 OOF rows map to one frozen event")
    return mapped, {
        "direction_rows_input": int(len(direction)),
        "direction_rows_mapped": int(len(mapped)),
        "direction_rows_outside_freeze": int(outside),
        "collision_rows_resolved_by_market_ticker_time": int(collision_key_resolved),
        "collision_rows_resolved_by_exact_eligibility": int(collision_exact_resolved),
    }


def build_event_article_view(articles: pd.DataFrame) -> pd.DataFrame:
    causal = articles.loc[articles["causal_t2_eligible"].fillna(False).astype(bool)].copy()
    causal["published_at_utc"] = pd.to_datetime(causal["published_at_utc"], utc=True, errors="coerce")
    causal = causal.sort_values(["canonical_event_id", "published_at_utc", "article_uid"], kind="mergesort")
    representatives = causal.drop_duplicates(["canonical_event_id", "syndication_root_id"], keep="first")
    rows = []
    for event_id, group in representatives.groupby("canonical_event_id", sort=False):
        first = group.iloc[0]
        headlines = [_clean(value) for value in group["headline"] if _clean(value)]
        bodies = [_clean(value) for value in group["body"] if _clean(value)]
        publishers = [_clean(value) for value in group["origin_publisher"] if _clean(value)]
        topics = [_clean(value) for value in group["primary_topic"] if _clean(value)]
        rows.append(
            {
                "canonical_event_id": str(event_id),
                "publisher_primary": _clean(first.get("origin_publisher")) or "UNKNOWN",
                "publisher_class_primary": _clean(first.get("publisher_class")) or "OTHER",
                "topic_primary": _clean(first.get("primary_topic")) or "OTHER",
                "publisher_topic": f"{_clean(first.get('origin_publisher')) or 'UNKNOWN'}|{_clean(first.get('primary_topic')) or 'OTHER'}",
                "headline_text": " [SEP] ".join(headlines),
                "body_text": " [SEP] ".join(bodies),
                "text_all": " [SEP] ".join(headlines + bodies),
                "causal_article_uids": list(group["article_uid"].astype(str)),
                "causal_article_count": int(len(group)),
                "publisher_count_view": int(len(set(publishers))),
                "topic_count_view": int(len(set(topics))),
            }
        )
    return pd.DataFrame(rows)


def load_event_embeddings(event_view: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    event_by_article = {}
    for row in event_view[["canonical_event_id", "causal_article_uids"]].to_dict(orient="records"):
        for article_uid in row["causal_article_uids"]:
            event_by_article[str(article_uid)] = str(row["canonical_event_id"])
    needed = index.loc[index["article_uid"].astype(str).isin(event_by_article)].copy()
    require(len(needed) == len(event_by_article), "embedding index missing causal model articles")
    event_vectors: dict[str, list[np.ndarray]] = {}
    for shard_path, group in needed.groupby("shard_path", sort=True):
        path = EMBED_ROOT / str(shard_path)
        require(path.is_file(), f"embedding shard missing: {shard_path}")
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        for row in group.to_dict(orient="records"):
            vector = np.asarray(vectors[int(row["shard_index"])], dtype=np.float32)
            event_vectors.setdefault(event_by_article[str(row["article_uid"])], []).append(vector)
    rows = []
    for event_id, vectors in event_vectors.items():
        mean = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        rows.append({"canonical_event_id": event_id, "embedding_vector": mean.astype(np.float32)})
    return pd.DataFrame(rows)


def load_model_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(DATA / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet")
    split = pd.read_parquet(DATA / "HIST_NEWS_DEV_SPLIT_MANIFEST.parquet")
    model_split = pd.read_parquet(CANDIDATE_SPLIT)
    articles = pd.read_parquet(DATA / "ARTICLES_NORMALIZED.parquet")
    frozen_events = pd.read_parquet(
        DATA / "EVENT_MASTER.parquet",
        columns=[
            "canonical_event_id", "existing_event_id", "join_key_event_id", "market", "ticker",
            "event_time_utc", "exact_price_eligible",
        ],
    )
    index = pd.read_parquet(DATA / "EMBEDDING_INDEX.parquet")
    direction = pd.read_csv(V69_OOF, low_memory=False)
    direction = direction.loc[direction["source_family"].isin(NEWS_SOURCES)].copy()
    require(not direction["event_id"].duplicated().any(), "V69 NEWS outer-OOF event IDs are not unique")
    direction, direction_mapping_audit = map_v69_to_frozen_events(direction, frozen_events)
    article_view = build_event_article_view(articles)
    event_embeddings = load_event_embeddings(article_view, index)
    frame = matrix.merge(
        split[["canonical_event_id", "source_family_fold", "fold"]].rename(
            columns={"source_family_fold": "global_source_family_fold", "fold": "global_fold"}
        ),
        on="canonical_event_id", how="left", validate="one_to_one",
    )
    frame = frame.merge(
        model_split[["canonical_event_id", "model_event_group_id", "model_source_fold"]],
        on="canonical_event_id", how="inner", validate="one_to_one",
    )
    frame = frame.merge(article_view, on="canonical_event_id", how="left", validate="one_to_one")
    frame = frame.merge(event_embeddings, on="canonical_event_id", how="left", validate="one_to_one")
    direction_columns = [
        "canonical_event_id", "direction_event_id_raw", "event_time_utc", "source_family", "y", "fwd_ret_30m", "prob", "confidence_signal",
        "fold", "family", "new_issuer", "opportunity_confidence", "prob_opportunity_direction",
        "prob_event_form_direction", "prob_fixed_numeric_direction", "text_cluster_id", "cluster_weight",
    ]
    frame = frame.merge(
        direction[direction_columns].rename(columns={"fold": "direction_outer_fold"}),
        on="canonical_event_id", how="inner", validate="one_to_one", suffixes=("_label", "_direction"),
    )
    if "y_direction" in frame:
        require(np.array_equal(frame["y_label"].astype(int), frame["y_direction"].astype(int)), "DEV/V69 y mismatch")
        frame["y"] = frame["y_direction"].astype(int)
    else:
        frame["y"] = frame["y"].astype(int)
    if "fwd_ret_30m_direction" in frame:
        delta = (frame["fwd_ret_30m_label"].astype(float) - frame["fwd_ret_30m_direction"].astype(float)).abs()
        require(bool((delta < 1e-12).all()), "DEV/V69 forward-return mismatch")
        frame["fwd_ret_30m"] = frame["fwd_ret_30m_direction"].astype(float)
    causal = frame[["tierA_article_count_t2", "tierBFull_article_count_t2"]].fillna(0).sum(axis=1) > 0
    frame = frame.loc[causal].copy()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc_direction"], utc=True, errors="raise")
    frame["source_freeze_event_group_id"] = frame["event_group_id"]
    frame["event_group_id"] = frame["model_event_group_id"]
    frame["source_family_fold"] = frame["model_source_fold"].astype(int)
    frame["direction_prediction"] = frame["prob"].astype(float) >= 0.5
    frame["correct_target"] = (frame["direction_prediction"].astype(int) == frame["y"].astype(int)).astype(int)
    frame["signed_net"] = signed_net(frame)
    frame["event_year"] = frame["event_time_utc"].dt.year.astype(int)
    frame["event_month"] = frame["event_time_utc"].dt.month.astype(int)
    frame["event_hour"] = frame["event_time_utc"].dt.hour.astype(int)
    frame["event_dayofweek"] = frame["event_time_utc"].dt.dayofweek.astype(int)
    for column in ("publisher_primary", "publisher_class_primary", "topic_primary", "publisher_topic", "text_all"):
        frame[column] = frame[column].fillna("UNKNOWN" if "publisher" in column else "OTHER" if column == "topic_primary" else "")
    require(frame["embedding_vector"].map(lambda value: isinstance(value, np.ndarray)).all(), "model-ready event embedding missing")
    require(frame["source_family_fold"].notna().all(), "source-family frozen fold missing")
    frame = frame.sort_values(["source_family", "event_time_utc", "canonical_event_id"], kind="mergesort").reset_index(drop=True)
    fold_authority = {}
    frame["validation_fold"] = -1
    frame["validation_fold_authority"] = ""
    for source, source_rows in frame.groupby("source_family"):
        candidate_fold_count = int(source_rows["model_source_fold"].nunique())
        direction_fold_count = int(source_rows["direction_outer_fold"].nunique())
        if candidate_fold_count >= 4:
            chosen_column = "model_source_fold"
            authority_name = "LABEL_BLIND_MODEL_CANDIDATE_SPLIT"
        else:
            require(direction_fold_count >= 3, f"insufficient frozen chronological folds for {source}")
            chosen_column = "direction_outer_fold"
            authority_name = "IMMUTABLE_V69_SOURCE_OUTER_FOLD"
        mask_source = frame["source_family"] == source
        frame.loc[mask_source, "validation_fold"] = frame.loc[mask_source, chosen_column].astype(int)
        frame.loc[mask_source, "validation_fold_authority"] = authority_name
        fold_authority[source] = {
            "authority": authority_name,
            "candidate_fold_count": candidate_fold_count,
            "direction_fold_count": direction_fold_count,
            "selection_uses_outcome": False,
        }
    inventory = {
        "model_ready_events": int(len(frame)),
        "by_source": {str(k): int(v) for k, v in frame["source_family"].value_counts().items()},
        "by_source_fold": {
            f"{source}|{int(fold)}": int(count)
            for (source, fold), count in frame.groupby(["source_family", "validation_fold"]).size().items()
        },
        "tier_a_events": int((frame["tierA_article_count_t2"] > 0).sum()),
        "tier_b_full_events": int((frame["tierBFull_article_count_t2"] > 0).sum()),
        "direction_oof_fit_contains_same_event": False,
        "direction_mapping_audit": direction_mapping_audit,
        "validation_fold_authority": fold_authority,
    }
    return frame, inventory


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    families: tuple[str, ...]


CANDIDATES = (
    CandidateSpec("SIMPLE_METADATA", ("NUMERIC_META", "CATEGORICAL")),
    CandidateSpec("TEXT_WORD", ("TEXT_WORD",)),
    CandidateSpec("TEXT_CHAR", ("TEXT_CHAR",)),
    CandidateSpec("EMBEDDING_LINEAR", ("EMBEDDING",)),
    CandidateSpec("META_TEXT_WORD", ("NUMERIC_META", "CATEGORICAL", "TEXT_WORD")),
    CandidateSpec("META_TEXT_CHAR", ("NUMERIC_META", "CATEGORICAL", "TEXT_CHAR")),
    CandidateSpec("META_EMBEDDING", ("NUMERIC_META", "CATEGORICAL", "EMBEDDING")),
    CandidateSpec("TEXT_WORD_CHAR", ("TEXT_WORD", "TEXT_CHAR")),
    CandidateSpec("FOLDSAFE_RELIABILITY", ("TARGET_PRIORS", "CATEGORICAL")),
    CandidateSpec("MODEL_AGREEMENT", ("MODEL_AGREEMENT",)),
    CandidateSpec("NOVELTY_ONLY", ("NOVELTY",)),
    CandidateSpec("STRUCTURED_TIMESTAMP", ("STRUCTURED", "TIMESTAMP")),
    CandidateSpec("PUBLISHER_TOPIC", ("CATEGORICAL",)),
    CandidateSpec("MARKET_CONTEXT", ("MARKET_CONTEXT",)),
    CandidateSpec(
        "ALL_LINEAR_FUSION",
        (
            "NUMERIC_META", "CATEGORICAL", "TEXT_WORD", "TEXT_CHAR", "EMBEDDING",
            "TARGET_PRIORS", "MODEL_AGREEMENT", "MARKET_CONTEXT",
        ),
    ),
)

AGREEMENT_COLUMNS = (
    "prob", "confidence_signal", "opportunity_confidence", "prob_opportunity_direction",
    "prob_event_form_direction", "prob_fixed_numeric_direction",
)
CATEGORICAL_COLUMNS = (
    "market", "source_family", "publisher_primary", "publisher_class_primary",
    "topic_primary", "publisher_topic", "new_issuer",
)
MARKET_CONTEXT_COLUMNS = (
    "event_year", "event_month", "event_hour", "event_dayofweek", "ticker",
)


def _catalog_numeric_columns(frame: pd.DataFrame, blocks: set[str] | None = None) -> list[str]:
    catalog = pd.read_csv(DATA / "FEATURE_CATALOG.csv")
    if blocks is not None:
        catalog = catalog.loc[catalog["block"].isin(blocks)]
    names = []
    for name in catalog["feature"].astype(str):
        if name not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            names.append(name)
    return sorted(set(names))


def _prequential_target_priors(train: pd.DataFrame, valid: pd.DataFrame, strength: float = 20.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = ("publisher_primary", "topic_primary", "publisher_topic")
    ordered = train.sort_values(["event_time_utc", "canonical_event_id"], kind="mergesort")
    train_values = pd.DataFrame(index=train.index)
    global_sum = 0.0
    global_n = 0
    stats: dict[str, dict[str, list[float]]] = {key: {} for key in keys}
    for idx, row in ordered.iterrows():
        global_prior = (global_sum + strength * 0.5) / (global_n + strength)
        for key in keys:
            token = _clean(row[key]) or "UNKNOWN"
            total, count = stats[key].get(token, [0.0, 0.0])
            posterior = (total + strength * global_prior) / (count + strength)
            train_values.loc[idx, f"prior_{key}"] = posterior
            train_values.loc[idx, f"support_{key}"] = math.log1p(count)
        target = float(row["correct_target"])
        for key in keys:
            token = _clean(row[key]) or "UNKNOWN"
            total, count = stats[key].get(token, [0.0, 0.0])
            stats[key][token] = [total + target, count + 1.0]
        global_sum += target
        global_n += 1
    global_prior = (global_sum + strength * 0.5) / (global_n + strength)
    valid_values = pd.DataFrame(index=valid.index)
    for idx, row in valid.iterrows():
        for key in keys:
            token = _clean(row[key]) or "UNKNOWN"
            total, count = stats[key].get(token, [0.0, 0.0])
            valid_values.loc[idx, f"prior_{key}"] = (total + strength * global_prior) / (count + strength)
            valid_values.loc[idx, f"support_{key}"] = math.log1p(count)
    columns = [item for key in keys for item in (f"prior_{key}", f"support_{key}")]
    return (
        train_values.loc[train.index, columns].to_numpy(dtype=float),
        valid_values.loc[valid.index, columns].to_numpy(dtype=float),
        columns,
    )


def build_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    families: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    train_parts: list[sparse.csr_matrix] = []
    valid_parts: list[sparse.csr_matrix] = []
    audit: dict[str, Any] = {"families": list(families), "feature_counts": {}}

    def add_dense(name: str, train_array: np.ndarray, valid_array: np.ndarray, *, scale: bool = True) -> None:
        train_array = np.asarray(train_array, dtype=float)
        valid_array = np.asarray(valid_array, dtype=float)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_array = imputer.fit_transform(train_array)
        valid_array = imputer.transform(valid_array)
        if scale:
            scaler = StandardScaler()
            train_array = scaler.fit_transform(train_array)
            valid_array = scaler.transform(valid_array)
        train_parts.append(sparse.csr_matrix(train_array))
        valid_parts.append(sparse.csr_matrix(valid_array))
        audit["feature_counts"][name] = int(train_array.shape[1])

    family_set = set(families)
    if "NUMERIC_META" in family_set:
        columns = _catalog_numeric_columns(frame=train)
        add_dense("NUMERIC_META", train[columns].to_numpy(), valid[columns].to_numpy())
    if "NOVELTY" in family_set:
        columns = _catalog_numeric_columns(frame=train, blocks={"NOVELTY"})
        add_dense("NOVELTY", train[columns].to_numpy(), valid[columns].to_numpy())
    if "STRUCTURED" in family_set:
        columns = _catalog_numeric_columns(frame=train, blocks={"STRUCTURED_CONFIRMATION"})
        add_dense("STRUCTURED", train[columns].to_numpy(), valid[columns].to_numpy())
    if "TIMESTAMP" in family_set:
        columns = _catalog_numeric_columns(frame=train, blocks={"TIMESTAMP_QUALITY"})
        add_dense("TIMESTAMP", train[columns].to_numpy(), valid[columns].to_numpy())
    if "MODEL_AGREEMENT" in family_set:
        columns = [name for name in AGREEMENT_COLUMNS if name in train.columns]
        add_dense("MODEL_AGREEMENT", train[columns].to_numpy(), valid[columns].to_numpy())
    if "MARKET_CONTEXT" in family_set:
        numeric = ["event_year", "event_month", "event_hour", "event_dayofweek"]
        add_dense("MARKET_CONTEXT_NUMERIC", train[numeric].to_numpy(), valid[numeric].to_numpy())
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        train_encoded = encoder.fit_transform(train[["ticker"]].fillna("UNKNOWN").astype(str)).tocsr()
        valid_encoded = encoder.transform(valid[["ticker"]].fillna("UNKNOWN").astype(str)).tocsr()
        train_parts.append(train_encoded)
        valid_parts.append(valid_encoded)
        audit["feature_counts"]["MARKET_CONTEXT_TICKER"] = int(train_encoded.shape[1])
    if "CATEGORICAL" in family_set:
        columns = [name for name in CATEGORICAL_COLUMNS if name in train.columns]
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        train_encoded = encoder.fit_transform(train[columns].fillna("UNKNOWN").astype(str)).tocsr()
        valid_encoded = encoder.transform(valid[columns].fillna("UNKNOWN").astype(str)).tocsr()
        train_parts.append(train_encoded)
        valid_parts.append(valid_encoded)
        audit["feature_counts"]["CATEGORICAL"] = int(train_encoded.shape[1])
    categorical_family_columns = {
        "PUBLISHER_ONLY": ["publisher_primary"],
        "PUBLISHER_CLASS_ONLY": ["publisher_class_primary"],
        "TOPIC_ONLY": ["topic_primary"],
        "PUBLISHER_TOPIC_ONLY": ["publisher_topic"],
    }
    for family, columns in categorical_family_columns.items():
        if family not in family_set:
            continue
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        train_encoded = encoder.fit_transform(train[columns].fillna("UNKNOWN").astype(str)).tocsr()
        valid_encoded = encoder.transform(valid[columns].fillna("UNKNOWN").astype(str)).tocsr()
        train_parts.append(train_encoded)
        valid_parts.append(valid_encoded)
        audit["feature_counts"][family] = int(train_encoded.shape[1])
    if "TEXT_WORD" in family_set:
        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
            max_df=0.98, max_features=25000, sublinear_tf=True,
        )
        train_text = vectorizer.fit_transform(train["text_all"].fillna("").astype(str)).tocsr()
        valid_text = vectorizer.transform(valid["text_all"].fillna("").astype(str)).tocsr()
        train_parts.append(train_text)
        valid_parts.append(valid_text)
        audit["feature_counts"]["TEXT_WORD"] = int(train_text.shape[1])
    if "TEXT_CHAR" in family_set:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
            sublinear_tf=True,
        )
        train_text = vectorizer.fit_transform(train["text_all"].fillna("").astype(str)).tocsr()
        valid_text = vectorizer.transform(valid["text_all"].fillna("").astype(str)).tocsr()
        train_parts.append(train_text)
        valid_parts.append(valid_text)
        audit["feature_counts"]["TEXT_CHAR"] = int(train_text.shape[1])
    if "EMBEDDING" in family_set:
        train_embedding = np.stack(train["embedding_vector"].to_numpy()).astype(float)
        valid_embedding = np.stack(valid["embedding_vector"].to_numpy()).astype(float)
        add_dense("EMBEDDING", train_embedding, valid_embedding)
    if "TARGET_PRIORS" in family_set:
        train_prior, valid_prior, names = _prequential_target_priors(train, valid)
        add_dense("TARGET_PRIORS", train_prior, valid_prior)
        audit["target_prior_columns"] = names
        audit["same_row_target_used"] = False
        audit["validation_target_used"] = False
    require(bool(train_parts), f"candidate has no usable feature parts: {families}")
    train_matrix = sparse.hstack(train_parts, format="csr")
    valid_matrix = sparse.hstack(valid_parts, format="csr")
    audit["total_features"] = int(train_matrix.shape[1])
    return train_matrix, valid_matrix, audit


def fit_predict_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: CandidateSpec,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_matrix, valid_matrix, audit = build_matrices(train, valid, spec.families)
    target = train["correct_target"].astype(int).to_numpy()
    if len(np.unique(target)) < 2:
        probability = np.full(len(valid), float(target.mean()) if len(target) else 0.5)
        audit["model"] = "CONSTANT_PRIOR"
        return probability, audit
    model = LogisticRegression(
        C=0.25, penalty="l2", solver="liblinear", class_weight="balanced",
        random_state=seed, max_iter=2000,
    )
    model.fit(train_matrix, target)
    audit["model"] = "L2_LOGISTIC_C_0.25_BALANCED"
    audit["converged"] = int(model.n_iter_[0]) < int(model.max_iter)
    return model.predict_proba(valid_matrix)[:, 1], audit


def purge_ticker_embargo(train: pd.DataFrame, valid: pd.DataFrame, days: int = 7) -> tuple[pd.DataFrame, int]:
    earliest_valid = valid.groupby("ticker")["event_time_utc"].min().to_dict()
    keep = []
    for row in train[["ticker", "event_time_utc"]].itertuples(index=False):
        boundary = earliest_valid.get(row.ticker)
        keep.append(boundary is None or row.event_time_utc < boundary - pd.Timedelta(days=days))
    keep_mask = np.asarray(keep, dtype=bool)
    return train.loc[keep_mask].copy(), int((~keep_mask).sum())


def inner_chronological_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = (
        train.groupby("event_group_id", as_index=False)
        .agg(event_time_utc=("event_time_utc", "min"), rows=("canonical_event_id", "size"))
        .sort_values(["event_time_utc", "event_group_id"], kind="mergesort")
    )
    target_rows = max(1, int(math.floor(len(train) * 0.70)))
    cumulative = groups["rows"].cumsum()
    split_position = int(np.searchsorted(cumulative.to_numpy(), target_rows, side="left"))
    split_position = min(max(split_position, 0), max(len(groups) - 2, 0))
    train_groups = set(groups.iloc[: split_position + 1]["event_group_id"].astype(str))
    inner_train = train.loc[train["event_group_id"].astype(str).isin(train_groups)].copy()
    inner_valid = train.loc[~train["event_group_id"].astype(str).isin(train_groups)].copy()
    inner_train, ticker_purge_n = purge_ticker_embargo(inner_train, inner_valid)
    require(len(inner_train) >= 40 and len(inner_valid) >= 20, "inner chronological support insufficient")
    require(inner_train["event_time_utc"].max() <= inner_valid["event_time_utc"].min(), "inner chronology overlap")
    require(not set(inner_train["event_group_id"]) & set(inner_valid["event_group_id"]), "inner event-group leakage")
    return inner_train, inner_valid, {
        "inner_train_n": int(len(inner_train)),
        "inner_valid_n": int(len(inner_valid)),
        "inner_train_end": inner_train["event_time_utc"].max(),
        "inner_valid_start": inner_valid["event_time_utc"].min(),
        "inner_event_group_overlap_n": 0,
        "inner_ticker_embargo_days": 7,
        "inner_ticker_embargo_purge_n": ticker_purge_n,
    }


def outer_splits(frame: pd.DataFrame) -> list[tuple[str, int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    splits = []
    for source in NEWS_SOURCES:
        source_frame = frame.loc[frame["source_family"] == source].copy()
        for fold in sorted(source_frame["validation_fold"].astype(int).unique()):
            if fold <= int(source_frame["validation_fold"].min()):
                continue
            train = source_frame.loc[source_frame["validation_fold"].astype(int) < fold].copy()
            valid = source_frame.loc[source_frame["validation_fold"].astype(int) == fold].copy()
            train, ticker_purge_n = purge_ticker_embargo(train, valid)
            if len(train) < 60 or len(valid) < 30 or train["correct_target"].nunique() < 2 or valid["correct_target"].nunique() < 2:
                continue
            require(train["event_time_utc"].max() <= valid["event_time_utc"].min(), f"outer chronology overlap: {source}/{fold}")
            require(not set(train["event_group_id"]) & set(valid["event_group_id"]), f"outer event-group leakage: {source}/{fold}")
            splits.append(
                (
                    source, int(fold), train, valid,
                    {
                        "outer_train_n": int(len(train)), "outer_valid_n": int(len(valid)),
                        "outer_train_end": train["event_time_utc"].max(),
                        "outer_valid_start": valid["event_time_utc"].min(),
                        "outer_event_group_overlap_n": 0,
                        "outer_ticker_embargo_days": 7,
                        "outer_ticker_embargo_purge_n": ticker_purge_n,
                    },
                )
            )
    require(bool(splits), "no strict source-family outer splits are usable")
    return splits


def _candidate_selection_key(row: Mapping[str, Any], order: Mapping[str, int]) -> tuple[float, float, int]:
    auc = -math.inf if row.get("inner_auc") is None else float(row["inner_auc"])
    brier = math.inf if row.get("inner_brier") is None else float(row["inner_brier"])
    return auc, -brier, -int(order[str(row["candidate"])])


def run_nested_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    candidate_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    order = {spec.name: index for index, spec in enumerate(CANDIDATES)}
    for split_index, (source, outer_fold, train, valid, outer_audit) in enumerate(outer_splits(frame), start=1):
        inner_train, inner_valid, inner_audit = inner_chronological_split(train)
        inner_predictions: dict[str, np.ndarray] = {}
        inner_thresholds: dict[str, float] = {}
        inner_model_audits: dict[str, dict[str, Any]] = {}
        for candidate_index, spec in enumerate(CANDIDATES):
            probability, audit = fit_predict_candidate(
                inner_train, inner_valid, spec,
                seed=SEED + split_index * 1000 + candidate_index,
            )
            inner_predictions[spec.name] = probability
            inner_thresholds[spec.name] = float(np.quantile(probability, HC_QUANTILE))
            inner_model_audits[spec.name] = audit
            inner_rows.append(
                {
                    "source_family": source,
                    "outer_fold": outer_fold,
                    "candidate": spec.name,
                    "inner_train_n": len(inner_train),
                    "inner_valid_n": len(inner_valid),
                    "inner_auc": safe_auc(inner_valid["correct_target"], probability),
                    "inner_brier": float(brier_score_loss(inner_valid["correct_target"], probability)),
                    "inner_hc_threshold_q80": inner_thresholds[spec.name],
                    "inner_hc_accuracy": float(
                        inner_valid.loc[probability >= inner_thresholds[spec.name], "correct_target"].mean()
                    ),
                    "outer_labels_used": False,
                    "feature_count": audit["total_features"],
                }
            )
        current_inner = [row for row in inner_rows if row["source_family"] == source and row["outer_fold"] == outer_fold]
        selected_candidate = max(current_inner, key=lambda row: _candidate_selection_key(row, order))["candidate"]
        opportunity_cutoff = float(np.quantile(inner_valid["opportunity_confidence"].astype(float), HC_QUANTILE))
        for candidate_index, spec in enumerate(CANDIDATES):
            probability, audit = fit_predict_candidate(
                train, valid, spec,
                seed=SEED + split_index * 1000 + 500 + candidate_index,
            )
            output = valid[
                [
                    "canonical_event_id", "event_group_id", "event_time_utc", "source_family", "ticker",
                    "y", "fwd_ret_30m", "prob", "direction_prediction", "correct_target", "signed_net",
                    "publisher_primary", "publisher_class_primary", "topic_primary", "publisher_topic",
                    "opportunity_confidence", "cluster_weight", "text_cluster_id",
                ]
            ].copy()
            output["outer_fold"] = outer_fold
            output["candidate"] = spec.name
            output["p_correct_news"] = probability
            output["hc_threshold"] = inner_thresholds[spec.name]
            output["news_high_conf"] = probability >= inner_thresholds[spec.name]
            output["selected_by_inner"] = spec.name == selected_candidate
            output["correctness_model_fit_contains_event"] = False
            output["outer_labels_used_for_candidate_or_threshold"] = False
            output["direction_probability_output"] = output["prob"]
            output["direction_prediction_output"] = output["direction_prediction"]
            output["opportunity_threshold"] = opportunity_cutoff
            output["dual_gate_high"] = output["news_high_conf"] & (
                output["opportunity_confidence"].astype(float) >= opportunity_cutoff
            )
            candidate_rows.append(output)
            if spec.name == selected_candidate:
                selected_rows.append(output.copy())
        selected_inner = next(row for row in current_inner if row["candidate"] == selected_candidate)
        fold_audits.append(
            {
                "source_family": source,
                "outer_fold": outer_fold,
                "selected_candidate": selected_candidate,
                "selected_inner_auc": selected_inner["inner_auc"],
                "selected_inner_brier": selected_inner["inner_brier"],
                "selected_inner_hc_threshold": selected_inner["inner_hc_threshold_q80"],
                "opportunity_threshold": opportunity_cutoff,
                **outer_audit,
                **inner_audit,
                "outer_labels_used_for_fit_selection_or_threshold": False,
                "same_event_in_correctness_fit": False,
                "direction_immutable": True,
            }
        )
        print(
            f"[BASE_V1_MODEL] source={source} outer_fold={outer_fold} "
            f"train={len(train)} valid={len(valid)} selected={selected_candidate}",
            flush=True,
        )
    all_candidates = pd.concat(candidate_rows, ignore_index=True)
    selected = pd.concat(selected_rows, ignore_index=True)
    inner = pd.DataFrame(inner_rows)
    require(not selected["canonical_event_id"].duplicated().any(), "selected nested OOF event duplicated")
    require(not selected["correctness_model_fit_contains_event"].any(), "same event appears in P(correct) fit")
    require(np.array_equal(selected["prob"].to_numpy(), selected["direction_probability_output"].to_numpy()), "direction probability changed")
    require(np.array_equal(selected["direction_prediction"].to_numpy(), selected["direction_prediction_output"].to_numpy()), "direction prediction changed")
    return all_candidates, selected, inner, fold_audits


POLARITY_BLOCKS = {
    "PUBLISHER": ("PUBLISHER_ONLY",),
    "PUBLISHER_CLASS": ("PUBLISHER_CLASS_ONLY",),
    "TOPIC": ("TOPIC_ONLY",),
    "PUBLISHER_TOPIC": ("PUBLISHER_TOPIC_ONLY",),
    "TEXT": ("TEXT_WORD",),
    "NOVELTY": ("NOVELTY",),
    "TIMESTAMP_QUALITY": ("TIMESTAMP",),
    "STRUCTURED_CONFIRMATION": ("STRUCTURED",),
    "MARKET_CONTEXT": ("MARKET_CONTEXT",),
    "MODEL_AGREEMENT": ("MODEL_AGREEMENT",),
}


def _as_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5))


def run_polarity_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    outer_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    block_names = list(POLARITY_BLOCKS)
    for split_index, (source, outer_fold, train, valid, outer_audit) in enumerate(outer_splits(frame), start=1):
        inner_train, inner_valid, inner_audit = inner_chronological_split(train)
        inner_logits: dict[str, np.ndarray] = {}
        outer_logits: dict[str, np.ndarray] = {}
        for block_index, (block, families) in enumerate(POLARITY_BLOCKS.items()):
            spec = CandidateSpec(f"POLARITY_{block}", tuple(families))
            inner_probability, _ = fit_predict_candidate(
                inner_train, inner_valid, spec, seed=SEED + 50000 + split_index * 100 + block_index,
            )
            outer_probability, _ = fit_predict_candidate(
                train, valid, spec, seed=SEED + 60000 + split_index * 100 + block_index,
            )
            inner_logits[block] = _as_logit(inner_probability)
            outer_logits[block] = _as_logit(outer_probability)
        inner_stack = np.stack([inner_logits[name] for name in block_names])
        outer_stack = np.stack([outer_logits[name] for name in block_names])
        inner_full = inner_stack.mean(axis=0)
        outer_full = outer_stack.mean(axis=0)
        for block_index, block in enumerate(block_names):
            inner_without = np.delete(inner_stack, block_index, axis=0).mean(axis=0)
            outer_without = np.delete(outer_stack, block_index, axis=0).mean(axis=0)
            candidates = []
            for alpha in ALPHAS:
                inner_probability = expit(inner_without + alpha * (inner_full - inner_without))
                threshold = float(np.quantile(inner_probability, HC_QUANTILE))
                candidates.append(
                    {
                        "alpha": alpha,
                        "inner_auc": safe_auc(inner_valid["correct_target"], inner_probability),
                        "inner_brier": float(brier_score_loss(inner_valid["correct_target"], inner_probability)),
                        "inner_hc_threshold": threshold,
                    }
                )
            chosen = max(
                candidates,
                key=lambda row: (
                    -math.inf if row["inner_auc"] is None else float(row["inner_auc"]),
                    -float(row["inner_brier"]), ALPHA_TIE_ORDER[float(row["alpha"])],
                ),
            )
            selection_rows.extend(
                {
                    "source_family": source, "outer_fold": outer_fold, "feature_block": block,
                    **row, "chosen_by_inner": row["alpha"] == chosen["alpha"], "outer_labels_used": False,
                }
                for row in candidates
            )
            outer_probability = expit(outer_without + float(chosen["alpha"]) * (outer_full - outer_without))
            output = valid[
                [
                    "canonical_event_id", "event_group_id", "event_time_utc", "source_family", "ticker",
                    "correct_target", "signed_net", "direction_prediction", "prob",
                ]
            ].copy()
            output["outer_fold"] = outer_fold
            output["feature_block"] = block
            output["selected_alpha"] = float(chosen["alpha"])
            output["p_correct_polarity"] = outer_probability
            output["hc_threshold"] = float(chosen["inner_hc_threshold"])
            output["news_high_conf"] = outer_probability >= float(chosen["inner_hc_threshold"])
            output["outer_labels_used_for_alpha_or_threshold"] = False
            outer_rows.append(output)
        fold_audits.append(
            {
                "source_family": source, "outer_fold": outer_fold,
                **outer_audit, **inner_audit,
                "block_count": len(block_names), "alpha_grid": list(ALPHAS),
                "outer_labels_used_for_alpha_or_threshold": False,
            }
        )
        print(f"[BASE_V1_POLARITY] source={source} outer_fold={outer_fold}", flush=True)
    return pd.concat(outer_rows, ignore_index=True), pd.DataFrame(selection_rows), fold_audits


def candidate_comparison(evidence: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    details: dict[str, Any] = {}
    for candidate in [spec.name for spec in CANDIDATES]:
        candidate_frame = evidence.loc[evidence["candidate"] == candidate]
        for source in (*NEWS_SOURCES, "ALL_NEWS"):
            subset = candidate_frame if source == "ALL_NEWS" else candidate_frame.loc[candidate_frame["source_family"] == source]
            if subset.empty:
                continue
            metrics = score_metrics(subset, "p_correct_news", "news_high_conf", seed=SEED + len(rows))
            details[f"{candidate}|{source}"] = metrics
            rows.append(
                {
                    "candidate": candidate,
                    "source_family": source,
                    "feature_families": json.dumps(next(spec.families for spec in CANDIDATES if spec.name == candidate)),
                    **{key: value for key, value in metrics.items() if key not in {"bootstrap_hc_net", "risk_coverage"}},
                    "bootstrap_hc_net_lower95": metrics["bootstrap_hc_net"]["lower95"],
                }
            )
    return pd.DataFrame(rows), details


def polarity_summary(outer: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, block), subset in outer.groupby(["source_family", "feature_block"], sort=True):
        metrics = score_metrics(subset, "p_correct_polarity", "news_high_conf", seed=SEED + len(rows) + 700)
        chosen = selection.loc[
            (selection["source_family"] == source)
            & (selection["feature_block"] == block)
            & selection["chosen_by_inner"].astype(bool)
        ]
        counts = chosen["alpha"].value_counts()
        modal_alpha = float(
            max(counts.index, key=lambda alpha: (int(counts.loc[alpha]), ALPHA_TIE_ORDER[float(alpha)]))
        )
        stability = float(counts.max() / counts.sum())
        if modal_alpha == -1.0:
            classification = "POLARITY_REVERSED"
        elif modal_alpha == 0.0:
            classification = "REMOVED"
        elif modal_alpha == 0.5:
            classification = "SHRINK"
        elif modal_alpha == 1.5:
            classification = "AMPLIFY"
        else:
            classification = "NORMAL"
        rows.append(
            {
                "source_family": source,
                "feature_block": block,
                "modal_inner_alpha": modal_alpha,
                "classification": classification,
                "stability": stability,
                "support_outer_folds": int(len(chosen)),
                "inner_chosen_counts": json.dumps({str(float(k)): int(v) for k, v in counts.items()}, sort_keys=True),
                "correctness_auc": metrics["correctness_auc"],
                "brier": metrics["brier"],
                "hc_coverage": metrics["hc_coverage"],
                "hc_accuracy": metrics["hc_accuracy"],
                "hc_net": metrics["hc_net"],
                "bootstrap_hc_net_lower95": metrics["bootstrap_hc_net"]["lower95"],
                "top5_removed_net": metrics["hc_top5_removed_net"],
                "selection_basis": "INNER_CHRONOLOGICAL_ONLY",
            }
        )
    return pd.DataFrame(rows)


def calibration_report(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for source in (*NEWS_SOURCES, "ALL_NEWS"):
        subset = frame if source == "ALL_NEWS" else frame.loc[frame["source_family"] == source]
        if subset.empty:
            continue
        ranked = subset.sort_values("p_correct_news").copy()
        ranked["confidence_bin"] = pd.qcut(
            np.arange(len(ranked)), q=min(10, len(ranked)), labels=False, duplicates="drop"
        )
        bins = (
            ranked.groupby("confidence_bin")
            .agg(n=("correct_target", "size"), mean_p=("p_correct_news", "mean"), accuracy=("correct_target", "mean"))
            .reset_index()
        )
        accuracy = bins["accuracy"].to_numpy()
        report[source] = {
            "ece_10": expected_calibration_error(subset["correct_target"], subset["p_correct_news"]),
            "confidence_bin_non_decreasing_fraction": float(np.mean(np.diff(accuracy) >= 0)) if len(accuracy) > 1 else None,
            "bins": bins.to_dict(orient="records"),
        }
    return report


def risk_coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in (*NEWS_SOURCES, "ALL_NEWS"):
        subset = frame if source == "ALL_NEWS" else frame.loc[frame["source_family"] == source]
        subset = subset.sort_values("p_correct_news", ascending=False)
        for coverage in np.arange(0.05, 1.001, 0.05):
            count = max(1, int(math.ceil(len(subset) * coverage)))
            selected = subset.head(count)
            rows.append(
                {
                    "source_family": source,
                    "coverage_target": float(round(coverage, 2)),
                    "n": int(len(selected)),
                    "coverage_actual": float(len(selected) / len(subset)),
                    "accuracy": float(selected["correct_target"].mean()),
                    "mean_net": float(selected["signed_net"].mean()),
                    "mean_p_correct": float(selected["p_correct_news"].mean()),
                }
            )
    return pd.DataFrame(rows)


def publisher_topic_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, source_frame in frame.groupby("source_family"):
        for key in ("publisher_primary", "topic_primary", "publisher_topic"):
            for value, group in source_frame.groupby(key, dropna=False):
                rows.append(
                    {
                        "source_family": source,
                        "reliability_key": key,
                        "reliability_value": _clean(value) or "UNKNOWN",
                        "outer_oof_n": int(len(group)),
                        "outer_oof_correct_rate": float(group["correct_target"].mean()),
                        "outer_oof_mean_p_correct": float(group["p_correct_news"].mean()),
                        "outer_oof_hc_rate": float(group["news_high_conf"].mean()),
                        "same_row_target_encoding": False,
                        "feature_recipe": "past-only train-fold Bayesian shrinkage when FOLDSAFE_RELIABILITY selected",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["source_family", "reliability_key", "outer_oof_n"], ascending=[True, True, False]
    )


def balanced_accuracy(target: Sequence[int], prediction: Sequence[bool]) -> float | None:
    y = np.asarray(target, dtype=int)
    pred = np.asarray(prediction, dtype=bool)
    values = []
    for label in (0, 1):
        mask = y == label
        if not mask.any():
            return None
        values.append(float((pred[mask] == bool(label)).mean()))
    return float(np.mean(values))


def bootstrap_balanced_accuracy(frame: pd.DataFrame, *, seed: int, draws: int = 2000) -> dict[str, Any]:
    if frame.empty:
        return {"draws": draws, "lower95": None, "median": None, "upper95": None}
    rng = np.random.default_rng(seed)
    values = []
    y = frame["y"].astype(int).to_numpy()
    pred = frame["direction_prediction"].astype(bool).to_numpy()
    for _ in range(draws):
        idx = rng.integers(0, len(frame), len(frame))
        value = balanced_accuracy(y[idx], pred[idx])
        if value is not None:
            values.append(value)
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {"draws": draws, "lower95": float(low), "median": float(median), "upper95": float(high)}


def integrated_gate_report(frame: pd.DataFrame) -> dict[str, Any]:
    direction_ba = balanced_accuracy(frame["y"], frame["direction_prediction"])
    direction_auc = safe_auc(frame["y"], frame["prob"])
    hc = frame.loc[frame["news_high_conf"]]
    bootstrap_ba = bootstrap_balanced_accuracy(frame, seed=SEED + 900)
    hc_metrics = score_metrics(frame, "p_correct_news", "news_high_conf", seed=SEED + 901)
    source_ba = {
        source: balanced_accuracy(group["y"], group["direction_prediction"])
        for source, group in frame.groupby("source_family")
    }
    fold_ba = {
        f"{source}|{int(fold)}": balanced_accuracy(group["y"], group["direction_prediction"])
        for (source, fold), group in frame.groupby(["source_family", "outer_fold"])
    }
    checks = {
        "BA_GE_0_540": direction_ba is not None and direction_ba >= 0.540,
        "AUC_GE_0_560": direction_auc is not None and direction_auc >= 0.560,
        "EDGE_GE_0_020": direction_ba is not None and direction_ba - 0.5 >= 0.020,
        "HC_COVERAGE_GE_0_15": hc_metrics["hc_coverage"] is not None and hc_metrics["hc_coverage"] >= 0.15,
        "HC_ACCURACY_GE_0_600": hc_metrics["hc_accuracy"] is not None and hc_metrics["hc_accuracy"] >= 0.600,
        "HC_NET_GT_0": hc_metrics["hc_net"] is not None and hc_metrics["hc_net"] > 0,
        "BOOTSTRAP_BA_LOWER_GE_0_500": bootstrap_ba["lower95"] is not None and bootstrap_ba["lower95"] >= 0.500,
        "BOOTSTRAP_HC_NET_LOWER_GT_0": hc_metrics["bootstrap_hc_net"]["lower95"] is not None and hc_metrics["bootstrap_hc_net"]["lower95"] > 0,
        "US_BA_GE_0_510": source_ba.get("US_NEWS") is not None and source_ba["US_NEWS"] >= 0.510,
        "KR_BA_GE_0_510": source_ba.get("KR_NEWS") is not None and source_ba["KR_NEWS"] >= 0.510,
        "WORST_FOLD_GE_0_490": bool(fold_ba) and min(value for value in fold_ba.values() if value is not None) >= 0.490,
        "MAJOR_SOURCE_BA_GE_0_490": bool(source_ba) and min(value for value in source_ba.values() if value is not None) >= 0.490,
        "WINSORIZED_HC_NET_GT_0": hc_metrics["hc_winsorized_net"] is not None and hc_metrics["hc_winsorized_net"] > 0,
        "TOP5_REMOVED_HC_NET_GT_0": hc_metrics["hc_top5_removed_net"] is not None and hc_metrics["hc_top5_removed_net"] > 0,
    }
    dual = frame.copy()
    dual["dual_gate_high"] = dual["dual_gate_high"].astype(bool)
    dual_metrics = score_metrics(dual, "p_correct_news", "dual_gate_high", seed=SEED + 902)
    return {
        "scope": "MODEL_READY_STRICT_NESTED_OOF_NEWS_ONLY",
        "official_pass_claimed": False,
        "direction_probability_delta_max": float(
            np.max(np.abs(frame["prob"] - frame["direction_probability_output"]))
        ),
        "direction_prediction_mismatch": int(
            (frame["direction_prediction"] != frame["direction_prediction_output"]).sum()
        ),
        "direction_ba": direction_ba,
        "direction_auc": direction_auc,
        "direction_edge": None if direction_ba is None else direction_ba - 0.5,
        "bootstrap_direction_ba": bootstrap_ba,
        "source_ba": source_ba,
        "fold_ba": fold_ba,
        "confidence_gate": hc_metrics,
        "opportunity_dual_gate": dual_metrics,
        "canonical_14_checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "status": "DEV_RESEARCH_ONLY",
    }


def survivor_decision(selected: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    metrics = score_metrics(selected, "p_correct_news", "news_high_conf", seed=SEED + 950)
    selected_candidates = selected.groupby(["source_family", "outer_fold"])["candidate"].first()
    stability = float(selected_candidates.value_counts().max() / len(selected_candidates))
    checks = {
        "correctness_auc_ge_0_55": metrics["correctness_auc"] is not None and metrics["correctness_auc"] >= 0.55,
        "hc_accuracy_ge_0_60": metrics["hc_accuracy"] is not None and metrics["hc_accuracy"] >= 0.60,
        "hc_coverage_ge_0_15": metrics["hc_coverage"] is not None and metrics["hc_coverage"] >= 0.15,
        "bootstrap_hc_net_lower_gt_0": metrics["bootstrap_hc_net"]["lower95"] is not None and metrics["bootstrap_hc_net"]["lower95"] > 0,
        "top5_removed_net_gt_0": metrics["hc_top5_removed_net"] is not None and metrics["hc_top5_removed_net"] > 0,
        "multiple_fold_candidate_stability_ge_0_50": stability >= 0.50,
    }
    return {
        "status": "RESEARCH_SURVIVOR" if all(checks.values()) else "RESEARCH_FAIL",
        "official_pass": False,
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "selected_candidate_stability": stability,
        "metrics": metrics,
    }


def append_experiment_ledger(comparison: pd.DataFrame, authority: Mapping[str, Any]) -> Path:
    ledger_path = ROOT / "research" / "HIGH_CONF_NEWS_EXPERIMENT_LEDGER.csv"
    records = []
    for row in comparison.to_dict(orient="records"):
        payload = {
            "data_freeze_sha": authority["source_freeze_sha256"],
            "split_sha": authority["split_sha256"],
            "candidate": row["candidate"],
            "source_family": row["source_family"],
            "validation": "SOURCE_FAMILY_FROZEN_OUTER_PLUS_INNER_CHRONOLOGICAL",
        }
        experiment_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        records.append(
            {
                "experiment_id": experiment_id,
                "data_freeze_sha": authority["source_freeze_sha256"],
                "market_source": row["source_family"],
                "feature_families": row["feature_families"],
                "model": row["candidate"],
                "correctness_auc": row.get("correctness_auc"),
                "brier": row.get("brier"),
                "logloss": row.get("log_loss"),
                "ece": row.get("ece_10"),
                "hc_coverage": row.get("hc_coverage"),
                "hc_accuracy": row.get("hc_accuracy"),
                "hc_net": row.get("hc_net"),
                "bootstrap_lower": row.get("bootstrap_hc_net_lower95"),
                "polarity_intervention": "NONE_BASELINE_CANDIDATE",
                "material_pass": False,
            }
        )
    new = pd.DataFrame(records)
    if ledger_path.is_file():
        old = pd.read_csv(ledger_path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates("experiment_id", keep="last")
    write_csv(ledger_path, new.sort_values("experiment_id").reset_index(drop=True))
    return ledger_path


def build_research_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name == "MODEL_ARTIFACT_MANIFEST.json" or not path.is_file():
            continue
        rows.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(
        output_dir / "MODEL_ARTIFACT_MANIFEST.json",
        {
            "version": "HIST_HIGH_CONF_NEWS_BASE_V1_MODEL_MANIFEST",
            "files": rows,
            "research_seal_touched": False,
            "final_meta_touched": False,
            "v224_prospective_touched": False,
        },
    )


def main() -> int:
    authority = verify_authority()
    frame, inventory = load_model_frame()
    all_candidates, selected, inner, candidate_fold_audits = run_nested_candidates(frame)
    polarity_outer, polarity_selection, polarity_fold_audits = run_polarity_audit(frame)
    comparison, comparison_details = candidate_comparison(all_candidates)
    polarity = polarity_summary(polarity_outer, polarity_selection)
    calibration = calibration_report(selected)
    risk_coverage = risk_coverage_table(selected)
    reliability = publisher_topic_audit(selected)
    integrated = integrated_gate_report(selected)
    survivor = survivor_decision(selected, comparison)

    RESEARCH.mkdir(parents=True, exist_ok=True)
    write_parquet(RESEARCH / "CORRECTNESS_OOF_PREDICTIONS.parquet", selected)
    write_parquet(RESEARCH / "ALL_CANDIDATE_OOF_PREDICTIONS.parquet", all_candidates)
    write_parquet(RESEARCH / "POLARITY_OUTER_OOF.parquet", polarity_outer)
    write_csv(RESEARCH / "MODEL_COMPARISON.csv", comparison)
    write_csv(RESEARCH / "RISK_COVERAGE_REPORT.csv", risk_coverage)
    write_csv(RESEARCH / "FEATURE_POLARITY_AUDIT.csv", polarity)
    write_csv(RESEARCH / "PUBLISHER_TOPIC_RELIABILITY_AUDIT.csv", reliability)
    write_csv(RESEARCH / "INNER_SELECTION_AUDIT.csv", inner)
    write_csv(RESEARCH / "POLARITY_INNER_SELECTION_AUDIT.csv", polarity_selection)
    write_json(RESEARCH / "CONFIDENCE_CALIBRATION_REPORT.json", calibration)
    write_json(RESEARCH / "HC_ROBUSTNESS_REPORT.json", survivor)
    write_json(RESEARCH / "INTEGRATED_GATE_REPORT.json", integrated)
    write_json(
        RESEARCH / "BASELINE_RESULTS.json",
        {
            "version": "HIST_HIGH_CONF_NEWS_BASE_V1_STRICT_NESTED_OOF",
            "authority": authority,
            "inventory": inventory,
            "experiment_budget": {"material_candidates": len(CANDIDATES), "candidates": [spec.name for spec in CANDIDATES]},
            "contract": {
                "target": "I[(immutable V69 outer-OOF direction >= .5) == immutable V36 y]",
                "outer": "frozen source_family_fold expanding chronology",
                "inner": "outer-train-only chronological event-group split",
                "threshold": "inner score q80 only",
                "target_priors": "past-only prequential train; frozen outer-train mapping for validation",
                "direction_changed": False,
                "outer_labels_used_for_fit_selection_or_threshold": False,
            },
            "candidate_fold_audits": candidate_fold_audits,
            "candidate_metrics": comparison_details,
            "selected_nested_oof": score_metrics(selected, "p_correct_news", "news_high_conf", seed=SEED + 999),
            "survivor": survivor,
            "status": "DEV_RESEARCH_ONLY",
        },
    )
    write_json(
        RESEARCH / "FEATURE_POLARITY_REPORT.json",
        {
            "alpha_grid": list(ALPHAS),
            "blocks": list(POLARITY_BLOCKS),
            "selection": "inner chronological AUC, then Brier, fixed conservative tie order",
            "fold_audits": polarity_fold_audits,
            "outer_labels_used_for_alpha_or_threshold": False,
            "summary": polarity.to_dict(orient="records"),
        },
    )
    ledger_path = append_experiment_ledger(comparison, authority)
    state = {
        "schema_version": "HIGH_CONF_NEWS_MODEL_STATE_V1",
        "source_frozen": True,
        "split_frozen": True,
        "dev_label_joined": True,
        "model_ready": True,
        "model_ready_events": inventory["model_ready_events"],
        "nested_oof_evaluated_events": int(len(selected)),
        "research_status": survivor["status"],
        "official_pass": False,
        "frozen_research_model_created": False,
        "frozen_research_model_reason": (
            "ROBUST_SURVIVOR_REQUIRES_FINAL_SCORER_SERIALIZATION"
            if survivor["status"] == "RESEARCH_SURVIVOR"
            else "NO_ROBUST_RESEARCH_SURVIVOR"
        ),
        "experiment_ledger": str(ledger_path.resolve()),
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_prospective_touched": False,
    }
    write_json(ROOT / "state" / "HIGH_CONF_NEWS_MODEL_STATE.json", state)
    build_research_manifest(RESEARCH)
    print(
        json.dumps(
            {
                "status": survivor["status"],
                "model_ready_events": inventory["model_ready_events"],
                "nested_oof_events": len(selected),
                "correctness_auc": survivor["metrics"]["correctness_auc"],
                "hc_accuracy": survivor["metrics"]["hc_accuracy"],
                "hc_coverage": survivor["metrics"]["hc_coverage"],
                "bootstrap_hc_net_lower95": survivor["metrics"]["bootstrap_hc_net"]["lower95"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelResearchError as exc:
        print(f"FAIL_CLOSED: {exc}", flush=True)
        raise SystemExit(2)
