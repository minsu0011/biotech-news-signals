"""Build the outcome-blind historical high-confidence news source freeze.

This module deliberately performs no model fitting and reads no outcomes.  It
consumes only the two collection-stage tables in ``data/historical_news_backfill``
and writes a deterministic, label-independent source freeze.

The strictest invariant is temporal: an article contributes to a t+2 feature
only when its original publication time is no later than the event cutoff.  If
a genuine historical first-seen time exists, that time must also precede the
cutoff.  Current crawl time is never imputed as historical first-seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from historical_news_embedding_cache import MODEL_ID as EMBEDDING_MODEL_ID
from historical_news_embedding_cache import text_sha256 as embedding_text_sha256
from historical_news_quality_contract import (
    CAUSAL_TIERS,
    CONTRACT_VERSION as QUALITY_CONTRACT_VERSION,
    TIER_A,
    TIER_B_FULL,
    TIER_B_HEADLINE,
    TIER_C,
    classify_historical_article_quality,
    publisher_is_known,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "historical_news_backfill"
DEFAULT_OUTPUT = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
DEFAULT_RESEARCH = ROOT / "research" / "high_conf_news_base_v1"

FORMAT_VERSION = "HIST_HIGH_CONF_NEWS_BASE_V1"
TOPIC_TAXONOMY_VERSION = "BIO_NEWS_RULES_BASE_V1_20260831"
EMBARGO_DAYS = 7
FOLD_COUNT = 5

CODE_RECIPE_FILES = (
    "historical_news_quality_contract.py",
    "historical_news_event_inventory.py",
    "historical_news_article_backfill.py",
    "historical_news_build_freeze.py",
    "runtime_limits.py",
    "historical_news_10h_orchestrator.py",
    "historical_news_embedding_cache.py",
)

POST_FREEZE_LABEL_COLUMNS = (
    "y",
    "fwd_ret_30m",
    "contract_version",
    "price_source",
    "price_cadence_sec",
    "bar_timestamp_semantics",
    "entry_target_utc",
    "actual_entry_time_utc",
    "entry_time_utc",
    "entry_slippage_sec",
    "exit_target_utc",
    "actual_exit_time_utc",
    "exit_time_utc",
    "exit_slippage_sec",
    "actual_hold_seconds",
    "entry_price",
    "exit_price",
)

FORBIDDEN_PATH_PARTS = {
    "v224_prospective_confirmation",
    "v224_prospective_causal_confirmation",
    "research_seal_pool",
    "final_meta_reserve",
    "cert",
    "cert_research",
}
FORBIDDEN_COLUMN_PATTERNS = (
    re.compile(r"(^|_)(fwd|forward)_(ret|return)($|_)", re.I),
    re.compile(r"(^|_)(strategy_)?return($|_)", re.I),
    re.compile(r"(^|_)(is_)?correct(ness)?($|_)", re.I),
    re.compile(r"(^|_)p_correct($|_)", re.I),
    re.compile(r"(^|_)direction_label($|_)", re.I),
    re.compile(r"(^|_)up_down($|_)", re.I),
    re.compile(r"(^|_)target($|_)", re.I),
    re.compile(r"(^|_)outcome($|_)", re.I),
)

PRECISION_ORDER = {
    "UNKNOWN": 0,
    "DATE_ONLY": 1,
    "HOUR_ONLY": 2,
    "EXACT_MINUTE": 3,
    "EXACT_SECOND": 4,
}
CONFIDENCE_ORDER = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

PUBLISHER_CLASSES = (
    "WIRE",
    "MAJOR_FINANCIAL",
    "MAJOR_GENERAL",
    "BIOTECH_SPECIALIST",
    "ISSUER_OFFICIAL",
    "PR_DISTRIBUTOR",
    "REGULATORY",
    "AGGREGATOR",
    "OTHER",
)

TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FDA_REJECTION_CRL", ("complete response letter", " crl ", "fda reject", "approval denied", "fda 거절", "허가 거절", "보완요구서")),
    ("FDA_APPROVAL", ("fda approval", "fda approves", "approved by the fda", "regulatory approval", "fda 승인", "품목허가", "신약 허가")),
    ("CLINICAL_HOLD", ("clinical hold", "partial clinical hold", "trial hold", "임상 중단", "임상시험 중단", "임상 보류")),
    ("ENDPOINT_MISSED", ("missed endpoint", "did not meet", "failed to meet", "endpoint was not met", "평가지표 미충족", "목표 미달", "유효성 입증 실패")),
    ("ENDPOINT_MET", ("met endpoint", "achieved endpoint", "endpoint was met", "평가지표 충족", "1차 지표 충족", "유효성 입증")),
    ("TRIAL_NEGATIVE", ("trial failed", "negative results", "no statistically significant", "study discontinued", "임상 실패", "부정적 결과", "통계적 유의성 미달")),
    ("TRIAL_POSITIVE", ("positive results", "positive topline", "statistically significant", "promising results", "임상 성공", "긍정적 결과", "통계적 유의성 확보")),
    ("SAFETY_SIGNAL", ("safety signal", "adverse event", "serious adverse", "toxicity", "patient death", "안전성 우려", "이상반응", "중대한 이상", "독성")),
    ("PHASE3", ("phase 3", "phase iii", "pivotal trial", "임상 3상", "3상 임상")),
    ("PHASE2", ("phase 2", "phase ii", "임상 2상", "2상 임상")),
    ("PHASE1", ("phase 1", "phase i", "first-in-human", "임상 1상", "1상 임상")),
    ("OFFERING", ("public offering", "registered direct offering", "underwritten offering", "유상증자", "제3자배정", "주식 공모")),
    ("ATM", ("at-the-market", "at the market offering", "sales agreement", "시장가 발행")),
    ("CONVERTIBLE", ("convertible note", "convertible senior", "convertible debt", "전환사채", "교환사채")),
    ("WARRANT", ("warrant exercise", "purchase warrant", "pre-funded warrant", "신주인수권", "워런트")),
    ("DILUTION", ("dilution", "dilutive", "shares issued", "주식 희석", "신주 발행")),
    ("FINANCING", ("financing", "capital raise", "private placement", "자금 조달", "투자 유치", "사모 발행")),
    ("LICENSING", ("license agreement", "licensing agreement", "exclusive license", "기술이전", "라이선스 계약", "독점 실시권")),
    ("PARTNERSHIP", ("partnership", "collaboration agreement", "strategic collaboration", "공동개발", "업무협약", "전략적 협력")),
    ("MILESTONE", ("milestone payment", "development milestone", "commercial milestone", "마일스톤", "단계별 기술료")),
    ("M_AND_A", ("acquisition", "merger agreement", "to acquire", "takeover", "인수합병", "합병 계약", "경영권 인수")),
    ("EARNINGS", ("quarterly results", "financial results", "earnings", "net loss", "분기 실적", "영업이익", "영업손실", "순손실")),
    ("GUIDANCE", ("guidance", "outlook", "financial forecast", "실적 전망", "매출 전망", "가이던스")),
    ("MANAGEMENT", ("chief executive officer", "chief financial officer", "appoints", "resignation", "대표이사", "최고경영자", "최고재무책임자", "선임", "사임")),
    ("PATENT", ("patent", "intellectual property", "patent office", "특허 등록", "특허 출원", "지식재산권")),
    ("MANUFACTURING", ("manufacturing", "production facility", "contract manufacturer", "cmc", "생산시설", "위탁생산", "제조시설", "공장 증설")),
    ("RECALL", ("recall", "withdrawal from market", "product withdrawal", "제품 회수", "판매 중단", "자진 회수")),
)

POSITIVE_TOPICS = {"FDA_APPROVAL", "TRIAL_POSITIVE", "ENDPOINT_MET"}
NEGATIVE_TOPICS = {"FDA_REJECTION_CRL", "TRIAL_NEGATIVE", "ENDPOINT_MISSED", "CLINICAL_HOLD", "SAFETY_SIGNAL", "RECALL"}


class FreezeError(RuntimeError):
    """Raised when an invariant requires fail-closed behavior."""


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {str(x): str(x) for x in items}
        self.rank = {str(x): 0 for x in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


@dataclass(frozen=True)
class BuildPaths:
    input_dir: Path
    output_dir: Path
    research_dir: Path


@dataclass
class EmbeddingBundle:
    vectors: dict[str, np.ndarray]
    index_rows: dict[str, dict[str, Any]]
    model_metadata: dict[str, Any]
    source_hashes: dict[str, Any]
    required_articles: int
    available_articles: int
    stale_text_articles: int
    missing_articles: int
    errors: list[str]

    @property
    def complete(self) -> bool:
        return not self.errors and self.available_articles == self.required_articles

    @property
    def coverage_ratio(self) -> float:
        return self.available_articles / self.required_articles if self.required_articles else 1.0


def _clean_string(value: Any) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _ticker_comparison_key(value: Any, market: Any) -> str:
    """Normalize ticker representation only for deterministic key comparison."""
    ticker = _clean_string(value).upper()
    market_key = _clean_string(market).upper()
    if market_key == "KR" and ticker.isdigit():
        return ticker.zfill(6)
    return ticker


def _norm_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", _clean_string(value).lower()).strip()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, *parts: Any, size: int = 20) -> str:
    payload = "\x1f".join(_clean_string(x) for x in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:size]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    work = frame.loc[:, list(columns)] if columns is not None else frame
    digest = hashlib.sha256()
    for row in work.to_dict(orient="records"):
        normalized: dict[str, Any] = {}
        for key, value in sorted(row.items()):
            if isinstance(value, pd.Timestamp):
                normalized[key] = value.isoformat()
            elif value is pd.NA or (isinstance(value, float) and math.isnan(value)):
                normalized[key] = None
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        digest.update(_stable_json(normalized).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_safe_path(path: Path) -> None:
    parts = {p.lower() for p in path.resolve().parts}
    blocked = sorted(parts & FORBIDDEN_PATH_PARTS)
    if blocked:
        raise FreezeError(f"forbidden source namespace: {path} ({','.join(blocked)})")


def _assert_outcome_blind_columns(frame: pd.DataFrame, source: Path) -> None:
    blocked = [
        str(column)
        for column in frame.columns
        if any(pattern.search(str(column)) for pattern in FORBIDDEN_COLUMN_PATTERNS)
    ]
    if blocked:
        raise FreezeError(f"outcome-like columns in {source}: {blocked}")


def _read_parquet(path: Path) -> pd.DataFrame:
    _assert_safe_path(path)
    if not path.is_file():
        raise FreezeError(f"required input missing: {path}")
    frame = pd.read_parquet(path)
    _assert_outcome_blind_columns(frame, path)
    return frame


def load_embedding_bundle(input_dir: Path, articles: pd.DataFrame, *, require_complete: bool) -> EmbeddingBundle:
    embedding_dir = (input_dir / "embeddings").resolve()
    index_path = embedding_dir / "EMBEDDING_INDEX.parquet"
    metadata_path = embedding_dir / "EMBEDDING_MODEL_FREEZE.json"
    required = int(len(articles))
    errors: list[str] = []
    source_hashes: dict[str, Any] = {
        "EMBEDDING_INDEX.parquet": sha256_file(index_path) if index_path.is_file() else "MISSING",
        "EMBEDDING_MODEL_FREEZE.json": sha256_file(metadata_path) if metadata_path.is_file() else "MISSING",
    }
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"EMBEDDING_METADATA_INVALID:{type(error).__name__}")
    else:
        errors.append("EMBEDDING_METADATA_MISSING")
    if not index_path.is_file():
        errors.append("EMBEDDING_INDEX_MISSING")
        bundle = EmbeddingBundle({}, {}, metadata, source_hashes, required, 0, 0, required, errors)
        if require_complete:
            raise FreezeError("embedding cache incomplete: " + ";".join(errors))
        return bundle
    index = pd.read_parquet(index_path)
    required_columns = {
        "article_uid", "text_sha256", "model_id", "model_revision", "dimension",
        "dtype", "normalized", "shard_path", "shard_index", "shard_sha256",
    }
    missing_columns = sorted(required_columns - set(index.columns))
    if missing_columns:
        errors.append("EMBEDDING_INDEX_COLUMNS_MISSING:" + ",".join(missing_columns))
    if index.get("article_uid", pd.Series(dtype=str)).astype(str).duplicated().any():
        errors.append("EMBEDDING_INDEX_DUPLICATE_ARTICLE_UID")
    if metadata.get("model_id") != EMBEDDING_MODEL_ID:
        errors.append("EMBEDDING_MODEL_ID_MISMATCH")
    if metadata.get("target_or_outcome_training") is not False or metadata.get("outcome_fields_read") not in ([], None):
        errors.append("EMBEDDING_OUTCOME_BLIND_GATE_FAILED")
    if errors and missing_columns:
        bundle = EmbeddingBundle({}, {}, metadata, source_hashes, required, 0, 0, required, errors)
        if require_complete:
            raise FreezeError("embedding cache invalid: " + ";".join(errors))
        return bundle

    index = index.copy()
    index["article_uid"] = index["article_uid"].astype(str)
    index_by_uid = {str(row["article_uid"]): row for row in index.to_dict(orient="records")}
    expected_text = {
        str(uid): embedding_text_sha256(headline, body)
        for uid, headline, body in articles[["article_uid", "headline", "body"]].itertuples(index=False, name=None)
    }
    eligible_rows: dict[str, dict[str, Any]] = {}
    stale = 0
    for uid, digest in expected_text.items():
        row = index_by_uid.get(uid)
        if row is None:
            continue
        if str(row.get("text_sha256")) != digest:
            stale += 1
            continue
        if str(row.get("model_id")) != str(metadata.get("model_id")) or str(row.get("model_revision")) != str(metadata.get("model_revision")):
            errors.append(f"EMBEDDING_MODEL_REVISION_MISMATCH:{uid}")
            continue
        eligible_rows[uid] = row

    vectors: dict[str, np.ndarray] = {}
    shard_cache: dict[str, np.ndarray] = {}
    shard_hashes: list[str] = []
    for uid, row in eligible_rows.items():
        relative = Path(str(row["shard_path"]))
        shard = (embedding_dir / relative).resolve()
        if not shard.is_relative_to(embedding_dir) or not shard.is_file() or shard.is_symlink():
            errors.append(f"EMBEDDING_SHARD_UNSAFE_OR_MISSING:{uid}")
            continue
        declared_sha = str(row["shard_sha256"])
        key = shard.as_posix()
        if key not in shard_cache:
            observed_sha = sha256_file(shard)
            if observed_sha != declared_sha:
                errors.append(f"EMBEDDING_SHARD_HASH_MISMATCH:{relative.as_posix()}")
                continue
            try:
                shard_cache[key] = np.load(shard, allow_pickle=False, mmap_mode="r")
            except Exception as error:
                errors.append(f"EMBEDDING_SHARD_LOAD_FAILED:{type(error).__name__}")
                continue
            shard_hashes.append(declared_sha)
        matrix = shard_cache.get(key)
        if matrix is None:
            continue
        position = int(row["shard_index"])
        dimension = int(row["dimension"])
        if matrix.ndim != 2 or position < 0 or position >= len(matrix) or matrix.shape[1] != dimension:
            errors.append(f"EMBEDDING_SHARD_SHAPE_INVALID:{uid}")
            continue
        vector = np.asarray(matrix[position], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 0 or abs(norm - 1.0) > 0.01:
            errors.append(f"EMBEDDING_VECTOR_INVALID:{uid}")
            continue
        vectors[uid] = vector / norm
    source_hashes["EMBEDDING_SHARDS_SEMANTIC_SHA256"] = hashlib.sha256(
        "\n".join(sorted(set(shard_hashes))).encode("ascii")
    ).hexdigest()
    available = len(vectors)
    bundle = EmbeddingBundle(
        vectors=vectors,
        index_rows=eligible_rows,
        model_metadata=metadata,
        source_hashes=source_hashes,
        required_articles=required,
        available_articles=available,
        stale_text_articles=stale,
        missing_articles=max(0, required - available),
        errors=errors,
    )
    if require_complete and not bundle.complete:
        detail = errors[:20] + [f"MISSING_OR_STALE:{bundle.missing_articles}"]
        raise FreezeError("embedding cache incomplete or corrupt: " + ";".join(detail))
    return bundle


def attach_embeddings(articles: pd.DataFrame, bundle: EmbeddingBundle) -> pd.DataFrame:
    articles = articles.copy()
    articles["embedding_available"] = articles["article_uid"].map(lambda uid: str(uid) in bundle.vectors)
    articles["embedding_vector"] = articles["article_uid"].map(lambda uid: bundle.vectors.get(str(uid)))
    articles["embedding_model_id"] = str(bundle.model_metadata.get("model_id") or "")
    articles["embedding_model_revision"] = str(bundle.model_metadata.get("model_revision") or "")
    return articles


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _ensure_columns(frame: pd.DataFrame, names: Iterable[str], default: Any = "") -> pd.DataFrame:
    frame = frame.copy()
    for name in names:
        if name not in frame.columns:
            frame[name] = default
    return frame


def _canonical_url(value: Any) -> str:
    raw = _clean_string(value)
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        host = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
        keep = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm_") and k.lower() not in {"ref", "source", "output"}]
        return urlunsplit((parsed.scheme.lower() or "https", host, path, urlencode(sorted(keep)), ""))
    except Exception:
        return raw


def _publisher_info(raw: Any, url: Any, source_type: Any, origin: Any = "") -> tuple[str, str, str]:
    raw_text = _norm_token(origin or raw)
    host = urlsplit(_canonical_url(url)).netloc.lower()
    source = _norm_token(source_type)
    patterns: tuple[tuple[str, tuple[str, ...], str], ...] = (
        ("Reuters", ("reuters",), "WIRE"),
        ("Associated Press", ("associated press", " ap news", "apnews.com"), "WIRE"),
        ("Bloomberg", ("bloomberg",), "MAJOR_FINANCIAL"),
        ("The Wall Street Journal", ("wall street journal", "wsj.com"), "MAJOR_FINANCIAL"),
        ("CNBC", ("cnbc",), "MAJOR_FINANCIAL"),
        ("Financial Times", ("financial times", "ft.com"), "MAJOR_FINANCIAL"),
        ("Stat News", ("stat news", "statnews"), "BIOTECH_SPECIALIST"),
        ("BioPharma Dive", ("biopharma dive", "biopharmadive"), "BIOTECH_SPECIALIST"),
        ("Fierce Biotech", ("fierce biotech", "fiercebiotech"), "BIOTECH_SPECIALIST"),
        ("BioSpace", ("biospace",), "BIOTECH_SPECIALIST"),
        ("GlobeNewswire", ("globenewswire",), "PR_DISTRIBUTOR"),
        ("Business Wire", ("business wire", "businesswire"), "PR_DISTRIBUTOR"),
        ("PR Newswire", ("pr newswire", "prnewswire"), "PR_DISTRIBUTOR"),
        ("U.S. SEC", ("sec.gov", " sec ", "edgar"), "REGULATORY"),
        ("KIND", ("kind.krx", " kind ", "dart.fss"), "REGULATORY"),
        ("Yahoo Finance", ("yahoo finance", "finance.yahoo"), "AGGREGATOR"),
        ("MSN", (" msn ", "msn.com"), "AGGREGATOR"),
        ("Naver News", ("naver news", "news.naver"), "AGGREGATOR"),
    )
    haystack = f" {raw_text} {host} {source} "
    for canonical, aliases, publisher_class in patterns:
        if any(alias in haystack for alias in aliases):
            return canonical, publisher_class, host
    if any(token in source for token in ("issuer", "company pr", "official press")):
        canonical = _clean_string(origin or raw) or "Issuer official"
        return canonical, "ISSUER_OFFICIAL", host
    cleaned = _clean_string(origin or raw)
    if cleaned:
        return cleaned, "OTHER", host
    if host:
        return host, "OTHER", host
    return "UNKNOWN", "OTHER", ""


def _infer_origin(row: Mapping[str, Any]) -> str:
    explicit = _clean_string(row.get("origin_publisher"))
    if explicit:
        return explicit
    text = f" {_norm_token(row.get('headline'))} {_norm_token(row.get('body'))} "
    attribution = (
        ("Reuters", (" by reuters ", " reporting by reuters ", " reuters ")),
        ("Associated Press", (" associated press ", " by ap ")),
    )
    for publisher, needles in attribution:
        if any(needle in text for needle in needles):
            return publisher
    return _clean_string(row.get("publisher_canonical") or row.get("publisher_raw"))


def normalize_events(source: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_event_id", "market", "ticker", "event_time_utc"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise FreezeError(f"EVENT_MASTER missing required columns: {missing}")
    events = _ensure_columns(
        source,
        (
            "event_group_id", "cik", "issuer_id", "issuer_name", "event_time_local",
            "event_timezone", "source_family", "source_file", "source_row_number",
            "source_row_sha256", "existing_article_id", "existing_url",
            "exact_price_eligible", "decision_cutoff_utc", "inventory_priority_year",
            "inventory_priority_rank", "join_key_market_ticker_event_time", "provenance_json",
        ),
    )
    events["canonical_event_id"] = events["canonical_event_id"].map(_clean_string)
    events["market"] = events["market"].map(lambda x: _clean_string(x).upper())
    events["ticker"] = events["ticker"].map(lambda x: _clean_string(x).upper())
    events["event_time_utc"] = _to_utc(events["event_time_utc"])
    supplied_cutoff = _to_utc(events["decision_cutoff_utc"])
    expected_cutoff = events["event_time_utc"] + pd.Timedelta(minutes=2)
    mismatched = supplied_cutoff.notna() & ((supplied_cutoff - expected_cutoff).abs() > pd.Timedelta(seconds=1))
    if bool(mismatched.any()):
        raise FreezeError(f"decision cutoff differs from event+2m for {int(mismatched.sum())} events")
    events["decision_cutoff_utc"] = expected_cutoff
    if events["canonical_event_id"].eq("").any() or events["event_time_utc"].isna().any():
        raise FreezeError("EVENT_MASTER contains empty ID or invalid UTC event time")
    duplicate_ids = events["canonical_event_id"].duplicated(keep=False)
    if bool(duplicate_ids.any()):
        raise FreezeError(f"duplicate canonical_event_id: {events.loc[duplicate_ids, 'canonical_event_id'].head(10).tolist()}")
    events["event_group_id"] = events["event_group_id"].map(_clean_string)
    missing_group = events["event_group_id"].eq("")
    if bool(missing_group.any()):
        events.loc[missing_group, "event_group_id"] = [
            _stable_id("eg", market, ticker, event_time.floor("D").isoformat())
            for market, ticker, event_time in events.loc[
                missing_group, ["market", "ticker", "event_time_utc"]
            ].itertuples(index=False, name=None)
        ]
    events["source_event_group_id"] = events["event_group_id"]
    events["event_year"] = events["event_time_utc"].dt.year.astype("Int64")
    events = events.sort_values(["event_time_utc", "canonical_event_id"], kind="mergesort").reset_index(drop=True)
    return events


def _link_missing_articles(articles: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_ids = set(events["canonical_event_id"])
    event_keys = _ensure_columns(events, ("existing_event_id", "join_key_event_id"))
    raw_lookup: dict[str, set[int]] = defaultdict(set)
    for idx, event in event_keys.iterrows():
        for column in ("canonical_event_id", "existing_event_id", "join_key_event_id"):
            key = _clean_string(event.get(column))
            if key:
                raw_lookup[key].add(int(idx))
    indexed: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in events.groupby(["market", "ticker"], dropna=False):
        indexed[(str(key[0]), str(key[1]))] = group.sort_values("event_time_utc")
    linked_ids: list[str] = []
    methods: list[str] = []
    qualities: list[str] = []
    quarantine_reasons: list[str] = []
    for row in articles.to_dict(orient="records"):
        explicit = _clean_string(row.get("canonical_event_id"))
        raw_candidates = sorted(raw_lookup.get(explicit, set())) if explicit else []
        if raw_candidates:
            candidates = event_keys.loc[raw_candidates]
            if len(candidates) > 1:
                market = _clean_string(row.get("market")).upper()
                ticker = _clean_string(row.get("ticker")).upper()
                source_event_time = row.get("source_event_time_utc")
                if isinstance(source_event_time, pd.Timestamp) and not pd.isna(source_event_time):
                    candidates = candidates.loc[
                        (candidates["market"].map(lambda x: _clean_string(x).upper()) == market)
                        & (candidates["ticker"].map(lambda x: _clean_string(x).upper()) == ticker)
                        & ((candidates["event_time_utc"] - source_event_time).abs() <= pd.Timedelta(seconds=1))
                    ]
                else:
                    candidates = candidates.iloc[0:0]
                if len(candidates) != 1:
                    linked_ids.append("")
                    methods.append("AMBIGUOUS_EVENT_MAPPING")
                    qualities.append("LOW")
                    quarantine_reasons.append("RAW_EVENT_ID_NOT_UNIQUE_AFTER_MARKET_TICKER_EVENT_TIME")
                    continue
                linked_ids.append(str(candidates.iloc[0]["canonical_event_id"]))
                methods.append("RAW_EVENT_ID_COLLISION_RESOLVED_BY_MARKET_TICKER_EVENT_TIME")
                qualities.append("HIGH")
                quarantine_reasons.append("")
                continue
            linked_ids.append(str(candidates.iloc[0]["canonical_event_id"]))
            methods.append("EXPLICIT_CANONICAL_OR_RAW_EVENT_ID")
            qualities.append("HIGH")
            quarantine_reasons.append("")
            continue
        # A source event timestamp is stronger than article-publication proximity
        # and is used before the bounded fallback below.
        market = _clean_string(row.get("market")).upper()
        ticker = _clean_string(row.get("ticker")).upper()
        source_event_time = row.get("source_event_time_utc")
        direct = indexed.get((market, ticker))
        if isinstance(source_event_time, pd.Timestamp) and not pd.isna(source_event_time) and direct is not None:
            exact = direct.loc[(direct["event_time_utc"] - source_event_time).abs() <= pd.Timedelta(seconds=1)]
            if len(exact) == 1:
                linked_ids.append(str(exact.iloc[0]["canonical_event_id"]))
                methods.append("MARKET_TICKER_SOURCE_EVENT_TIME")
                qualities.append("HIGH")
                quarantine_reasons.append("")
                continue
            if len(exact) > 1:
                linked_ids.append("")
                methods.append("AMBIGUOUS_EVENT_MAPPING")
                qualities.append("LOW")
                quarantine_reasons.append("MARKET_TICKER_SOURCE_EVENT_TIME_NOT_UNIQUE")
                continue
        pub = row.get("published_at_utc")
        candidates = indexed.get((market, ticker))
        if not isinstance(pub, pd.Timestamp) or pd.isna(pub) or candidates is None or candidates.empty:
            linked_ids.append("")
            methods.append("UNLINKED")
            qualities.append("LOW")
            quarantine_reasons.append("NO_EVENT_LINK_KEYS")
            continue
        delta = (candidates["event_time_utc"] - pub).abs()
        in_window = delta <= pd.Timedelta(hours=24)
        if not bool(in_window.any()):
            linked_ids.append("")
            methods.append("UNLINKED_OUTSIDE_24H")
            qualities.append("LOW")
            quarantine_reasons.append("NO_EVENT_WITHIN_24H")
            continue
        viable = candidates.loc[in_window].copy()
        viable["_delta"] = (viable["event_time_utc"] - pub).abs()
        viable = viable.sort_values(["_delta", "event_time_utc", "canonical_event_id"])
        if len(viable) != 1:
            linked_ids.append("")
            methods.append("AMBIGUOUS_EVENT_MAPPING")
            qualities.append("LOW")
            quarantine_reasons.append("MULTIPLE_MARKET_TICKER_EVENTS_WITHIN_24H")
            continue
        linked_ids.append(str(viable.iloc[0]["canonical_event_id"]))
        methods.append("INFERRED_MARKET_TICKER_NEAREST_24H")
        qualities.append("MEDIUM")
        quarantine_reasons.append("")
    articles["canonical_event_id"] = linked_ids
    articles["event_link_method"] = methods
    articles["event_match_quality"] = qualities
    articles["event_mapping_quarantine_reason"] = quarantine_reasons
    return articles


def normalize_articles(source: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    required = {"article_uid", "published_at_utc", "headline"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise FreezeError(f"ARTICLES_NORMALIZED missing required columns: {missing}")
    source = source.copy()
    if "event_time_utc" in source.columns:
        source = source.rename(columns={"event_time_utc": "source_event_time_utc"})
    if "decision_cutoff_utc" in source.columns:
        source = source.rename(columns={"decision_cutoff_utc": "source_decision_cutoff_utc"})
    columns = (
        "canonical_event_id", "market", "ticker", "issuer", "canonical_url", "raw_url",
        "publisher_raw", "publisher_canonical", "origin_publisher", "hosting_publisher",
        "publisher_class", "source_provider", "source_type", "published_at_source",
        "published_at_precision", "published_at_confidence", "timezone_name",
        "timezone_inferred", "first_seen_at_utc", "first_seen_source", "first_seen_observed", "captured_at_utc",
        "body", "language", "article_id", "content_sha256", "raw_capture_path",
        "raw_capture_sha256", "raw_mime_type", "discovery_window", "raw_capture_present",
        "tier_precondition_status", "tier_precondition_reasons",
        "source_event_time_utc", "source_decision_cutoff_utc",
    )
    articles = _ensure_columns(source, columns)
    articles["article_uid"] = articles["article_uid"].map(_clean_string)
    articles["canonical_event_id"] = articles["canonical_event_id"].map(_clean_string)
    articles["market"] = articles["market"].map(lambda x: _clean_string(x).upper())
    articles["ticker"] = articles["ticker"].map(lambda x: _clean_string(x).upper())
    articles["headline"] = articles["headline"].map(_clean_string)
    articles["body"] = articles["body"].map(_clean_string)
    articles["published_at_utc_source_value"] = articles["published_at_utc"].map(_clean_string)
    articles["published_at_utc"] = _to_utc(articles["published_at_utc"])
    articles["first_seen_at_utc"] = _to_utc(articles["first_seen_at_utc"])
    articles["captured_at_utc"] = _to_utc(articles["captured_at_utc"])
    articles["source_event_time_utc"] = _to_utc(articles["source_event_time_utc"])
    articles["source_decision_cutoff_utc"] = _to_utc(articles["source_decision_cutoff_utc"])
    articles["published_at_precision"] = articles["published_at_precision"].map(lambda x: _clean_string(x).upper() or "UNKNOWN")
    articles["published_at_confidence"] = articles["published_at_confidence"].map(lambda x: _clean_string(x).upper() or "UNKNOWN")
    articles["published_at_precision"] = articles["published_at_precision"].where(articles["published_at_precision"].isin(PRECISION_ORDER), "UNKNOWN")
    articles["published_at_confidence"] = articles["published_at_confidence"].where(articles["published_at_confidence"].isin(CONFIDENCE_ORDER), "UNKNOWN")
    # A provider date later than the collection capture is not silently repaired.
    # Preserve the supplied value for audit, clear the usable timestamp, and force
    # the row into the non-causal auxiliary path.  This catches date parsers that
    # mistake maturity years or other numbers in a headline for publication time.
    invalid_future_vs_capture = (
        articles["published_at_utc"].notna()
        & articles["captured_at_utc"].notna()
        & (articles["published_at_utc"] > articles["captured_at_utc"] + pd.Timedelta(minutes=5))
    )
    articles["published_at_invalid_future_vs_capture"] = invalid_future_vs_capture.astype(bool)
    articles["timestamp_quarantine_reason"] = ""
    articles.loc[invalid_future_vs_capture, "timestamp_quarantine_reason"] = "PUBLICATION_AFTER_CAPTURE"
    articles.loc[invalid_future_vs_capture, "published_at_utc"] = pd.NaT
    articles.loc[invalid_future_vs_capture, "published_at_precision"] = "UNKNOWN"
    articles.loc[invalid_future_vs_capture, "published_at_confidence"] = "LOW"
    articles["first_seen_observed"] = articles["first_seen_observed"].map(
        lambda x: bool(x) if not isinstance(x, str) else x.strip().lower() in {"1", "true", "yes"}
    )
    articles.loc[articles["first_seen_at_utc"].isna(), "first_seen_observed"] = False
    articles["canonical_url"] = [
        _canonical_url(canonical or raw)
        for canonical, raw in zip(articles["canonical_url"], articles["raw_url"])
    ]
    origins, canonicals, classes, domains = [], [], [], []
    for row in articles.to_dict(orient="records"):
        origin = _infer_origin(row)
        canonical, publisher_class, domain = _publisher_info(
            row.get("publisher_canonical") or row.get("publisher_raw"),
            row.get("canonical_url") or row.get("raw_url"),
            row.get("source_type"),
            origin,
        )
        origins.append(canonical if origin else canonical)
        canonicals.append(canonical)
        classes.append(publisher_class if publisher_class in PUBLISHER_CLASSES else "OTHER")
        domains.append(domain)
    articles["origin_publisher"] = origins
    articles["publisher_canonical"] = canonicals
    articles["publisher_class"] = classes
    articles["publisher_domain"] = domains
    articles["hosting_publisher"] = [
        _clean_string(host) or _publisher_info(raw, url, source)[0]
        for host, raw, url, source in zip(
            articles["hosting_publisher"], articles["publisher_raw"], articles["canonical_url"], articles["source_type"]
        )
    ]
    articles = _link_missing_articles(articles, events)
    # Exact duplicate UID rows may be harmless append-only repeats; conflicting ones are not.
    if articles["article_uid"].eq("").any():
        raise FreezeError("empty article_uid")
    duplicate_uid = articles["article_uid"].duplicated(keep=False)
    if bool(duplicate_uid.any()):
        compare = ["canonical_url", "content_sha256", "published_at_utc", "headline"]
        conflicts = []
        for uid, group in articles.loc[duplicate_uid].groupby("article_uid"):
            if any(group[col].astype(str).nunique(dropna=False) > 1 for col in compare):
                conflicts.append(uid)
        if conflicts:
            raise FreezeError(f"conflicting duplicate article_uid: {conflicts[:10]}")
        articles = articles.drop_duplicates("article_uid", keep="first")
    # The normalized frozen article table contains one row per canonical URL. Copies at
    # different URLs remain and are handled by syndication roots.
    nonempty_url = articles["canonical_url"].ne("")
    dup_url = nonempty_url & articles["canonical_url"].duplicated(keep=False)
    if bool(dup_url.any()):
        articles = articles.sort_values(
            ["canonical_url", "published_at_utc", "article_uid"], na_position="last", kind="mergesort"
        ).drop_duplicates("canonical_url", keep="first")
    articles["normalized_text"] = [
        _norm_token(f"{headline} {body}") for headline, body in zip(articles["headline"], articles["body"])
    ]
    articles["text_fingerprint"] = articles["normalized_text"].map(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest() if x else ""
    )
    articles = articles.sort_values(["published_at_utc", "article_uid"], na_position="last", kind="mergesort").reset_index(drop=True)
    return articles


def _word_set(text: Any) -> set[str]:
    return {token for token in _norm_token(text).split() if len(token) > 1}


def _jaccard(left: Any, right: Any) -> float:
    a, b = _word_set(left), _word_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_syndication(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = articles["article_uid"].tolist()
    uf = UnionFind(ids)
    exact_keys = ("canonical_url", "article_id", "content_sha256", "text_fingerprint")
    for key in exact_keys:
        for value, group in articles.loc[articles[key].map(_clean_string).ne("")].groupby(key):
            members = group["article_uid"].tolist()
            for other in members[1:]:
                uf.union(members[0], other)
    # Near duplicate comparisons are bounded to event/ticker and a 48h support window.
    bucket_column = "canonical_event_id"
    for _, group in articles.groupby(bucket_column, dropna=False):
        group = group.sort_values(["published_at_utc", "article_uid"], na_position="last")
        records = group.to_dict(orient="records")
        for i, left in enumerate(records):
            if not _clean_string(left.get(bucket_column)):
                continue
            for right in records[i + 1 :]:
                ltime, rtime = left.get("published_at_utc"), right.get("published_at_utc")
                if isinstance(ltime, pd.Timestamp) and isinstance(rtime, pd.Timestamp) and rtime - ltime > pd.Timedelta(hours=48):
                    break
                headline_sim = _jaccard(left.get("headline"), right.get("headline"))
                body_sim = _jaccard(left.get("normalized_text"), right.get("normalized_text"))
                if headline_sim >= 0.88 and (body_sim >= 0.72 or not _clean_string(left.get("body")) or not _clean_string(right.get("body"))):
                    uf.union(str(left["article_uid"]), str(right["article_uid"]))
    roots: dict[str, list[str]] = defaultdict(list)
    for article_uid in ids:
        roots[uf.find(article_uid)].append(article_uid)
    root_id_by_article: dict[str, str] = {}
    cluster_rows: list[dict[str, Any]] = []
    indexed = articles.set_index("article_uid", drop=False)
    for members in sorted(roots.values(), key=lambda x: sorted(x)[0]):
        member_frame = indexed.loc[sorted(members)].reset_index(drop=True).sort_values(
            ["published_at_utc", "article_uid"], na_position="last"
        )
        origin = member_frame.iloc[0]
        root_id = _stable_id("syn", *sorted(members))
        for uid in members:
            root_id_by_article[uid] = root_id
        cluster_rows.append(
            {
                "syndication_root_id": root_id,
                "canonical_event_id": _stable_json(sorted(set(member_frame["canonical_event_id"].map(_clean_string)) - {""})),
                "origin_article_uid": str(origin["article_uid"]),
                "origin_publisher": str(origin["origin_publisher"]),
                "member_count": int(len(member_frame)),
                "member_article_uids": _stable_json(sorted(members)),
                "member_publishers": _stable_json(sorted(set(member_frame["publisher_canonical"].astype(str)))),
                "first_published_at": member_frame["published_at_utc"].min(),
            }
        )
    result = articles.copy()
    result["syndication_root_id"] = result["article_uid"].map(root_id_by_article)
    clusters = pd.DataFrame(cluster_rows).sort_values("syndication_root_id").reset_index(drop=True)
    return result, clusters


def classify_topics(articles: pd.DataFrame) -> pd.DataFrame:
    result = articles.copy()
    primary, all_topics, provenance = [], [], []
    for row in result.to_dict(orient="records"):
        text = f" {_norm_token(row.get('headline'))} {_norm_token(row.get('body'))} "
        matches: list[str] = []
        evidence: list[dict[str, Any]] = []
        for topic, phrases in TOPIC_RULES:
            found = [phrase.strip() for phrase in phrases if phrase in text]
            if found:
                matches.append(topic)
                evidence.append({"topic": topic, "method": "label_free_keyword_rule", "matched": sorted(found)})
        if not matches:
            matches = ["OTHER"]
            evidence = [{"topic": "OTHER", "method": "label_free_default", "matched": []}]
        primary.append(matches[0])
        all_topics.append(_stable_json(matches))
        provenance.append(_stable_json(evidence))
    result["primary_topic"] = primary
    result["topics_json"] = all_topics
    result["topic_provenance_json"] = provenance
    result["topic_taxonomy_version"] = TOPIC_TAXONOMY_VERSION
    return result


def build_topic_qa_sample(articles: pd.DataFrame, per_topic: int = 5) -> pd.DataFrame:
    """Create deterministic label-blind text evidence samples per primary topic."""

    rows: list[dict[str, Any]] = []
    for topic, group in articles.groupby("primary_topic", dropna=False):
        topic_name = _clean_string(topic) or "OTHER"
        candidates = group.copy()
        candidates["_sample_key"] = candidates["article_uid"].map(
            lambda uid: hashlib.sha256(f"{TOPIC_TAXONOMY_VERSION}|{topic_name}|{uid}".encode("utf-8")).hexdigest()
        )
        for row in candidates.sort_values("_sample_key").head(per_topic).to_dict(orient="records"):
            body = _clean_string(row.get("body"))
            rows.append(
                {
                    "topic_taxonomy_version": TOPIC_TAXONOMY_VERSION,
                    "primary_topic": topic_name,
                    "article_uid": row.get("article_uid"),
                    "market": row.get("market"),
                    "publisher_canonical": row.get("publisher_canonical"),
                    "headline": row.get("headline"),
                    "body_evidence_excerpt": body[:500],
                    "topics_json": row.get("topics_json"),
                    "topic_provenance_json": row.get("topic_provenance_json"),
                    "selection_method": "DETERMINISTIC_LABEL_BLIND_HASH_SAMPLE",
                }
            )
    return pd.DataFrame(rows).sort_values(["primary_topic", "article_uid"], kind="mergesort").reset_index(drop=True)


def build_quality(articles: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = articles.merge(
        events[["canonical_event_id", "event_time_utc", "decision_cutoff_utc"]],
        on="canonical_event_id",
        how="left",
        validate="many_to_one",
    )
    audit_rows: list[dict[str, Any]] = []
    tiers, causal_flags, headline_flags, unobserved_flags, aux_flags = [], [], [], [], []
    for row in result.to_dict(orient="records"):
        precision = _clean_string(row.get("published_at_precision")).upper() or "UNKNOWN"
        confidence = _clean_string(row.get("published_at_confidence")).upper() or "UNKNOWN"
        event_quality = _clean_string(row.get("event_match_quality")).upper()
        raw_present = bool(row.get("raw_capture_present")) and bool(_clean_string(row.get("raw_capture_sha256") or row.get("content_sha256")))
        published = row.get("published_at_utc")
        cutoff = row.get("decision_cutoff_utc")
        first_seen = row.get("first_seen_at_utc")
        captured = row.get("captured_at_utc")
        decision = classify_historical_article_quality(
            canonical_publisher=row.get("publisher_canonical"),
            published_at_precision=precision,
            published_at_confidence=confidence,
            published_at=published,
            decision_cutoff=cutoff,
            event_match_quality=event_quality,
            headline=row.get("headline"),
            body=row.get("body"),
            first_seen_at=first_seen,
            first_seen_observed=row.get("first_seen_observed"),
            captured_at=captured,
            provenance_available=raw_present,
        )
        tier = decision.tier
        causal = decision.causal_eligible
        aux_only = not causal
        tiers.append(tier)
        causal_flags.append(causal)
        headline_flags.append(decision.headline_only_aux)
        unobserved_flags.append(decision.first_seen_unobserved)
        aux_flags.append(aux_only)
        audit_rows.append(
            {
                "article_uid": row["article_uid"],
                "quality_contract_version": QUALITY_CONTRACT_VERSION,
                "tier": tier,
                "publisher_quality": "KNOWN" if decision.publisher_known else "UNKNOWN",
                "timestamp_quality": f"{precision}:{confidence}",
                "content_quality": "BODY_SUBSTANTIVE" if decision.body_substantive else ("HEADLINE_ONLY" if decision.headline_substantive else "INSUFFICIENT"),
                "raw_provenance_available": bool(raw_present),
                "event_match_quality": event_quality or "LOW",
                "causal_eligible": bool(causal),
                "headline_only_aux": bool(decision.headline_only_aux),
                "published_before_cutoff": bool(decision.published_before_cutoff),
                "first_seen_observed": bool(decision.first_seen_observed),
                "first_seen_unobserved": bool(decision.first_seen_unobserved),
                "first_seen_before_cutoff": bool(decision.first_seen_before_cutoff),
                "current_crawl_as_historical_first_seen": bool(decision.current_crawl_as_historical_first_seen),
                "reason": "|".join(decision.reasons) if decision.reasons else "STRICT_CAUSAL_PASS",
            }
        )
    result["quality_contract_version"] = QUALITY_CONTRACT_VERSION
    result["tier"] = tiers
    result["quality_tier"] = tiers
    result["causal_t2_eligible"] = causal_flags
    result["headline_only_aux"] = headline_flags
    result["first_seen_unobserved"] = unobserved_flags
    result["retrospective_aux_only"] = aux_flags
    quality = pd.DataFrame(audit_rows).sort_values("article_uid").reset_index(drop=True)
    return result, quality


def build_publisher_registry(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for publisher, group in articles.groupby("publisher_canonical", dropna=False):
        classes = group["publisher_class"].map(_clean_string)
        publisher_class = classes.value_counts().index[0] if len(classes.value_counts()) else "OTHER"
        aliases = sorted(set(group["publisher_raw"].map(_clean_string)) - {""})
        domains = sorted(set(group["publisher_domain"].map(_clean_string)) - {""})
        source_types = sorted(set(group["source_type"].map(_clean_string)) - {""})
        languages = sorted(set(group["language"].map(_clean_string)) - {""})
        market_set = set(group["market"].astype(str))
        country = "KR" if market_set == {"KR"} else ("US" if market_set == {"US"} else "MULTI_OR_UNKNOWN")
        rows.append(
            {
                "publisher_canonical": _clean_string(publisher) or "UNKNOWN",
                "publisher_class": publisher_class if publisher_class in PUBLISHER_CLASSES else "OTHER",
                "domains": _stable_json(domains),
                "aliases": _stable_json(aliases),
                "source_type": _stable_json(source_types),
                "country": country,
                "language": _stable_json(languages),
                "article_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("publisher_canonical").reset_index(drop=True)


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _pairwise_jaccard(texts: Sequence[str]) -> list[float]:
    values: list[float] = []
    for i, left in enumerate(texts):
        for right in texts[i + 1 :]:
            values.append(_jaccard(left, right))
    return values


def _pairwise_cosine(vectors: Sequence[np.ndarray]) -> list[float]:
    values: list[float] = []
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            values.append(float(np.clip(np.dot(left, right), -1.0, 1.0)))
    return values


def _structured_kind(row: Mapping[str, Any]) -> str:
    haystack = " ".join(
        _norm_token(row.get(key))
        for key in ("source_type", "source_provider", "publisher_class", "publisher_canonical", "canonical_url")
    )
    if "sec.gov" in haystack or re.search(r"\b(sec|edgar)\b", haystack):
        return "SEC"
    if "kind.krx" in haystack or "dart.fss" in haystack or re.search(r"\bkind\b", haystack):
        return "KIND"
    if "issuer" in haystack or "official" in haystack:
        return "ISSUER_PR"
    return ""


def _build_novelty_history_index(articles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index article history once so each event does not rescan the full corpus.

    This is a performance-only transformation.  The indexed columns and filters are
    identical to the former per-event ``all_articles`` scan.
    """
    history = articles.loc[articles["published_at_utc"].notna()].copy()
    history["_ticker_clean"] = history["ticker"].map(_clean_string)
    history = history.loc[history["_ticker_clean"].ne("")]
    return {
        str(ticker): group.sort_values("published_at_utc").reset_index(drop=True)
        for ticker, group in history.groupby("_ticker_clean", sort=False)
    }


def _novelty_features(
    event_id: str,
    event_articles: pd.DataFrame,
    ticker_history: pd.DataFrame,
    event_time: pd.Timestamp,
) -> dict[str, Any]:
    current = event_articles.loc[event_articles["causal_t2_eligible"]]
    if current.empty:
        return {
            "similarity_to_prior_24h": np.nan,
            "similarity_to_prior_7d": np.nan,
            "similarity_to_prior_30d": np.nan,
            "near_duplicate_prior_news": False,
            "news_recycling_class": "NO_CAUSAL_ARTICLE",
        }
    prior = ticker_history.loc[
        (ticker_history["canonical_event_id"].map(_clean_string) != event_id)
        & (ticker_history["published_at_utc"] < event_time)
        & (ticker_history["published_at_utc"] >= event_time - pd.Timedelta(days=30))
    ]
    current_texts = current.sort_values("published_at_utc").drop_duplicates("syndication_root_id")["normalized_text"].tolist()
    values: dict[str, float] = {}
    for name, delta in (("24h", pd.Timedelta(hours=24)), ("7d", pd.Timedelta(days=7)), ("30d", pd.Timedelta(days=30))):
        corpus = prior.loc[prior["published_at_utc"] >= event_time - delta, "normalized_text"].tolist()
        values[name] = max((_jaccard(x, y) for x in current_texts for y in corpus), default=np.nan)
    sim30 = values["30d"]
    duplicate = bool(not pd.isna(sim30) and sim30 >= 0.85)
    if duplicate:
        recycle = "NEAR_DUPLICATE_PRIOR_NEWS"
    elif not pd.isna(sim30) and sim30 >= 0.55:
        recycle = "MINOR_UPDATE_OR_REPUBLICATION"
    else:
        recycle = "NOVEL_STORY"
    return {
        "similarity_to_prior_24h": values["24h"],
        "similarity_to_prior_7d": values["7d"],
        "similarity_to_prior_30d": values["30d"],
        "near_duplicate_prior_news": duplicate,
        "news_recycling_class": recycle,
    }


def _empty_event_feature_row(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact feature defaults for an event with no linked articles."""
    event_id = str(event["canonical_event_id"])
    row: dict[str, Any] = {
        "canonical_event_id": event_id,
        "event_group_id": str(event["event_group_id"]),
        "market": event["market"],
        "ticker": event["ticker"],
        "event_time_utc": event["event_time_utc"],
        "decision_cutoff_utc": event["decision_cutoff_utc"],
        "tierA_article_count_t2": 0,
        "tierB_article_count_t2": 0,
        "tierBFull_article_count_t2": 0,
        "tierBHeadline_article_count_t2": 0,
        "independent_source_count_t2": 0,
        "unique_publisher_count_t2": 0,
        "syndication_root_count_t2": 0,
        "publisher_class_counts_t2": _stable_json({}),
        "structured_confirmation_count_t2": 0,
        "official_source_count_t2": 0,
        "issuer_pr_present_t2": False,
        "SEC_present_t2": False,
        "KIND_present_t2": False,
        "structured_text_similarity": np.nan,
        "unique_origin_publisher_count": 0,
        "unique_publisher_class_count": 0,
        "publisher_entropy": 0.0,
        "source_type_diversity": 0,
        "wire_present": False,
        "major_financial_present": False,
        "biotech_specialist_present": False,
        "issuer_official_present": False,
        "mean_pairwise_similarity": np.nan,
        "median_similarity": np.nan,
        "minimum_similarity": np.nan,
        "semantic_dispersion": np.nan,
        "semantic_consensus_available": False,
        "semantic_similarity_method": "TOKEN_JACCARD_FALLBACK",
        "embedding_article_count_t2": 0,
        "embedding_coverage_t2": np.nan,
        "embedding_model_revision": "",
        "topic_agreement_rate": np.nan,
        "topic_conflict_count": 0,
        "semantic_contradiction_proxy": 0.0,
        "primary_topics_t2": _stable_json([]),
        "origin_publishers_t2": _stable_json([]),
        "source_types_t2": _stable_json([]),
        "published_time_precision_mean": np.nan,
        "published_time_confidence_mean": np.nan,
        "minutes_from_event_to_publish_min": np.nan,
        "minutes_from_event_to_publish_max": np.nan,
        "published_pre_event_count": 0,
        "published_post_event_count": 0,
        "first_seen_known_count": 0,
        "headline_length_mean": np.nan,
        "body_length_mean": np.nan,
        "body_available_rate": np.nan,
        "headline_body_similarity_mean": np.nan,
        "market_context_join_key": event_id,
        "model_agreement_join_key": event_id,
        "similarity_to_prior_24h": np.nan,
        "similarity_to_prior_7d": np.nan,
        "similarity_to_prior_30d": np.nan,
        "near_duplicate_prior_news": False,
        "news_recycling_class": "NO_CAUSAL_ARTICLE",
    }
    for publisher_class in PUBLISHER_CLASSES:
        row[f"publisher_class_{publisher_class.lower()}_count_t2"] = 0
    return row


def build_event_features(events: pd.DataFrame, articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    by_event = {key: group.copy() for key, group in articles.groupby("canonical_event_id") if _clean_string(key)}
    novelty_history = _build_novelty_history_index(articles)
    empty_history = articles.iloc[0:0].copy()
    for event in events.to_dict(orient="records"):
        event_id = str(event["canonical_event_id"])
        linked = by_event.get(event_id)
        if linked is None:
            rows.append(_empty_event_feature_row(event))
            cluster_rows.append(
                {
                    "canonical_event_id": event_id,
                    "event_group_id": event["event_group_id"],
                    "article_count_all_windows": 0,
                    "causal_article_count_t2": 0,
                    "syndication_root_ids": _stable_json([]),
                    "article_uids": _stable_json([]),
                    "topics": _stable_json([]),
                }
            )
            continue
        causal = linked.loc[linked["causal_t2_eligible"]].copy()
        if causal.empty:
            row = _empty_event_feature_row(event)
            row["tierBHeadline_article_count_t2"] = int(
                (linked["quality_tier"] == TIER_B_HEADLINE).sum()
            )
            rows.append(row)
            cluster_rows.append(
                {
                    "canonical_event_id": event_id,
                    "event_group_id": event["event_group_id"],
                    "article_count_all_windows": int(len(linked)),
                    "causal_article_count_t2": 0,
                    "syndication_root_ids": _stable_json(sorted(set(linked["syndication_root_id"].astype(str)))),
                    "article_uids": _stable_json(sorted(set(linked["article_uid"].astype(str)))),
                    "topics": _stable_json(sorted(set(linked["primary_topic"].astype(str)))),
                }
            )
            continue
        root_reps = causal.sort_values(["published_at_utc", "article_uid"]).drop_duplicates("syndication_root_id")
        publishers = root_reps["origin_publisher"].map(_clean_string).tolist()
        pub_classes = root_reps["publisher_class"].map(_clean_string).tolist()
        source_types = root_reps["source_type"].map(_clean_string).tolist()
        topics = root_reps["primary_topic"].map(_clean_string).tolist()
        texts = root_reps["normalized_text"].map(_clean_string).tolist()
        embedding_vectors = [
            value for value in root_reps.get("embedding_vector", pd.Series(dtype=object)).tolist()
            if isinstance(value, np.ndarray)
        ]
        if len(embedding_vectors) >= 2:
            similarities = _pairwise_cosine(embedding_vectors)
            similarity_method = "MULTILINGUAL_E5_COSINE"
        else:
            similarities = _pairwise_jaccard(texts)
            similarity_method = "TOKEN_JACCARD_FALLBACK"
        topic_counts = Counter(topics)
        topic_agreement = max(topic_counts.values()) / len(topics) if topics else np.nan
        structured = root_reps.copy()
        structured["structured_kind"] = [
            _structured_kind(row) for row in structured.to_dict(orient="records")
        ] if len(structured) else []
        structured = structured.loc[structured["structured_kind"].ne("")]
        sec_count = int((structured["structured_kind"] == "SEC").sum()) if len(structured) else 0
        kind_count = int((structured["structured_kind"] == "KIND").sum()) if len(structured) else 0
        issuer_count = int((structured["structured_kind"] == "ISSUER_PR").sum()) if len(structured) else 0
        official_texts = structured["normalized_text"].tolist() if len(structured) else []
        nonofficial_texts = root_reps.loc[~root_reps["article_uid"].isin(structured["article_uid"]), "normalized_text"].tolist() if len(structured) else []
        structured_similarity = max((_jaccard(a, b) for a in official_texts for b in nonofficial_texts), default=np.nan)
        positive = sum(topic in POSITIVE_TOPICS for topic in topics)
        negative = sum(topic in NEGATIVE_TOPICS for topic in topics)
        contradiction_count = int(min(positive, negative))
        ticker = _clean_string(causal.iloc[0]["ticker"]) if len(causal) else ""
        novelty = _novelty_features(
            event_id,
            linked,
            novelty_history.get(ticker, empty_history),
            event["event_time_utc"],
        )
        precision_scores = [PRECISION_ORDER.get(x, 0) for x in root_reps["published_at_precision"].astype(str)]
        confidence_scores = [CONFIDENCE_ORDER.get(x, 0) for x in root_reps["published_at_confidence"].astype(str)]
        minutes = (
            (root_reps["published_at_utc"] - event["event_time_utc"]).dt.total_seconds() / 60.0
            if len(root_reps) else pd.Series(dtype=float)
        )
        class_counts = Counter(pub_classes)
        row = {
            "canonical_event_id": event_id,
            "event_group_id": str(event["event_group_id"]),
            "market": event["market"],
            "ticker": event["ticker"],
            "event_time_utc": event["event_time_utc"],
            "decision_cutoff_utc": event["decision_cutoff_utc"],
            "tierA_article_count_t2": int((causal["quality_tier"] == TIER_A).sum()),
            "tierB_article_count_t2": int((causal["quality_tier"] == TIER_B_FULL).sum()),
            "tierBFull_article_count_t2": int((causal["quality_tier"] == TIER_B_FULL).sum()),
            "tierBHeadline_article_count_t2": int((linked["quality_tier"] == TIER_B_HEADLINE).sum()),
            "independent_source_count_t2": int(root_reps["syndication_root_id"].nunique()),
            "unique_publisher_count_t2": int(root_reps["origin_publisher"].nunique()),
            "syndication_root_count_t2": int(root_reps["syndication_root_id"].nunique()),
            "publisher_class_counts_t2": _stable_json(dict(sorted(class_counts.items()))),
            "structured_confirmation_count_t2": int(len(structured)),
            "official_source_count_t2": int(len(structured)),
            "issuer_pr_present_t2": bool(issuer_count),
            "SEC_present_t2": bool(sec_count),
            "KIND_present_t2": bool(kind_count),
            "structured_text_similarity": structured_similarity,
            "unique_origin_publisher_count": int(len(set(publishers))),
            "unique_publisher_class_count": int(len(set(pub_classes) - {""})),
            "publisher_entropy": _entropy(publishers),
            "source_type_diversity": int(len(set(source_types) - {""})),
            "wire_present": bool("WIRE" in pub_classes),
            "major_financial_present": bool("MAJOR_FINANCIAL" in pub_classes),
            "biotech_specialist_present": bool("BIOTECH_SPECIALIST" in pub_classes),
            "issuer_official_present": bool("ISSUER_OFFICIAL" in pub_classes),
            "mean_pairwise_similarity": float(np.mean(similarities)) if similarities else np.nan,
            "median_similarity": float(np.median(similarities)) if similarities else np.nan,
            "minimum_similarity": float(np.min(similarities)) if similarities else np.nan,
            "semantic_dispersion": float(1.0 - np.mean(similarities)) if similarities else np.nan,
            "semantic_consensus_available": bool(similarities),
            "semantic_similarity_method": similarity_method,
            "embedding_article_count_t2": int(len(embedding_vectors)),
            "embedding_coverage_t2": float(len(embedding_vectors) / len(root_reps)) if len(root_reps) else np.nan,
            "embedding_model_revision": _clean_string(root_reps["embedding_model_revision"].iloc[0]) if len(root_reps) and "embedding_model_revision" in root_reps else "",
            "topic_agreement_rate": topic_agreement,
            "topic_conflict_count": contradiction_count,
            "semantic_contradiction_proxy": float(contradiction_count / max(len(topics), 1)),
            "primary_topics_t2": _stable_json(sorted(set(topics) - {""})),
            "origin_publishers_t2": _stable_json(sorted(set(publishers) - {""})),
            "source_types_t2": _stable_json(sorted(set(source_types) - {""})),
            "published_time_precision_mean": float(np.mean(precision_scores)) if precision_scores else np.nan,
            "published_time_confidence_mean": float(np.mean(confidence_scores)) if confidence_scores else np.nan,
            "minutes_from_event_to_publish_min": float(minutes.min()) if len(minutes) else np.nan,
            "minutes_from_event_to_publish_max": float(minutes.max()) if len(minutes) else np.nan,
            "published_pre_event_count": int((minutes <= 0).sum()) if len(minutes) else 0,
            "published_post_event_count": int((minutes > 0).sum()) if len(minutes) else 0,
            "first_seen_known_count": int(root_reps["first_seen_observed"].sum()) if len(root_reps) else 0,
            "headline_length_mean": float(root_reps["headline"].str.len().mean()) if len(root_reps) else np.nan,
            "body_length_mean": float(root_reps["body"].str.len().mean()) if len(root_reps) else np.nan,
            "body_available_rate": float(root_reps["body"].ne("").mean()) if len(root_reps) else np.nan,
            "headline_body_similarity_mean": float(np.mean([_jaccard(h, b) for h, b in zip(root_reps["headline"], root_reps["body"])])) if len(root_reps) else np.nan,
            "market_context_join_key": event_id,
            "model_agreement_join_key": event_id,
            **novelty,
        }
        for publisher_class in PUBLISHER_CLASSES:
            row[f"publisher_class_{publisher_class.lower()}_count_t2"] = int(class_counts.get(publisher_class, 0))
        rows.append(row)
        cluster_rows.append(
            {
                "canonical_event_id": event_id,
                "event_group_id": event["event_group_id"],
                "article_count_all_windows": int(len(linked)),
                "causal_article_count_t2": int(len(causal)),
                "syndication_root_ids": _stable_json(sorted(set(linked["syndication_root_id"].astype(str))) if len(linked) else []),
                "article_uids": _stable_json(sorted(set(linked["article_uid"].astype(str))) if len(linked) else []),
                "topics": _stable_json(sorted(set(linked["primary_topic"].astype(str))) if len(linked) else []),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["event_time_utc", "canonical_event_id"]).reset_index(drop=True),
        pd.DataFrame(cluster_rows).sort_values("canonical_event_id").reset_index(drop=True),
    )


def build_split_manifest(events: pd.DataFrame, articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.sort_values(["event_time_utc", "canonical_event_id"]).reset_index(drop=True).copy()
    ids = events["canonical_event_id"].astype(str).tolist()
    uf = UnionFind(ids)
    # Preserve inventory event groups.
    duplicate_base_group = events["source_event_group_id"].duplicated(keep=False)
    for _, group in events.loc[duplicate_base_group].groupby("source_event_group_id"):
        members = group["canonical_event_id"].astype(str).tolist()
        for other in members[1:]:
            uf.union(members[0], other)
    # Ticker/time embargo is a grouping constraint, not a label-aware purge.
    for _, group in events.groupby(["market", "ticker"], dropna=False):
        group = group.sort_values("event_time_utc")
        records = group[["canonical_event_id", "event_time_utc"]].to_dict(orient="records")
        for left, right in zip(records, records[1:]):
            if right["event_time_utc"] - left["event_time_utc"] <= pd.Timedelta(days=EMBARGO_DAYS):
                uf.union(str(left["canonical_event_id"]), str(right["canonical_event_id"]))
    # One syndication root must never cross folds.
    linked = articles.loc[articles["canonical_event_id"].ne("")]
    for _, group in linked.groupby("syndication_root_id"):
        members = sorted(set(group["canonical_event_id"].astype(str)))
        for other in members[1:]:
            uf.union(members[0], other)
    components: dict[str, list[str]] = defaultdict(list)
    for event_id in ids:
        components[uf.find(event_id)].append(event_id)
    index = {event_id: i for i, event_id in enumerate(ids)}
    intervals = sorted((min(index[x] for x in values), max(index[x] for x in values), values) for values in components.values())
    blocks: list[tuple[int, int, list[str]]] = []
    for start, end, members in intervals:
        if blocks and start <= blocks[-1][1]:
            old_start, old_end, old_members = blocks[-1]
            blocks[-1] = (old_start, max(old_end, end), old_members + members)
        else:
            blocks.append((start, end, list(members)))
    # Any indices enclosed by a connected interval become the same chronological
    # block. This prevents a long event group from straddling a fold boundary.
    block_by_event: dict[str, str] = {}
    effective_group: dict[str, str] = {}
    component_id: dict[str, str] = {}
    for component_members in components.values():
        stable_component_id = _stable_id("eg", *sorted(component_members))
        for event_id in component_members:
            component_id[event_id] = stable_component_id
    for block_no, (start, end, _) in enumerate(blocks):
        block_members = ids[start : end + 1]
        block_id = _stable_id("chron", *block_members)
        for event_id in block_members:
            block_by_event[event_id] = block_id
            effective_group[event_id] = component_id[event_id]
    # Assign whole chronological blocks in chronological order, balancing by rows.
    fold_by_block: dict[str, int] = {}
    total = max(len(events), 1)
    seen = 0
    for start, end, _ in blocks:
        block_id = block_by_event[ids[start]]
        midpoint = seen + (end - start + 1) / 2
        fold = min(FOLD_COUNT - 1, int(midpoint * FOLD_COUNT / total))
        fold_by_block[block_id] = fold
        seen += end - start + 1
    events["event_group_id"] = events["canonical_event_id"].map(effective_group)
    events["chronological_block_id"] = events["canonical_event_id"].map(block_by_event)
    manifest = pd.DataFrame(
        {
            "canonical_event_id": events["canonical_event_id"],
            "event_group_id": events["event_group_id"],
            "market": events["market"],
            "ticker": events["ticker"],
            "source_family": events["source_family"],
            "event_time": events["event_time_utc"],
            "fold": events["chronological_block_id"].map(fold_by_block).astype(int),
            "role": "HISTORICAL_HIGH_CONF_NEWS_DEV",
            "chronological_block_id": events["chronological_block_id"],
            "ticker_embargo_days": EMBARGO_DAYS,
        }
    )
    # Freeze a second, source-family-specific chronological fold before any label
    # is read.  NEWS sources have very different eras (for example historical US
    # vs recent KR), so the global inventory fold alone can leave a source with no
    # usable inner/outer sequence.  Effective event groups are assigned as whole
    # units and therefore remain isolated.
    source_fold_by_event: dict[str, int] = {}
    for source_family, source_rows in manifest.groupby("source_family", sort=True, dropna=False):
        source_rows = source_rows.sort_values(["event_time", "canonical_event_id"], kind="mergesort").reset_index(drop=True)
        source_ids = source_rows["canonical_event_id"].astype(str).tolist()
        source_index = {event_id: index for index, event_id in enumerate(source_ids)}
        source_intervals = []
        for _, group in source_rows.groupby("event_group_id", sort=False):
            positions = [source_index[str(event_id)] for event_id in group["canonical_event_id"]]
            source_intervals.append((min(positions), max(positions)))
        source_intervals.sort()
        source_blocks: list[tuple[int, int]] = []
        for start, end in source_intervals:
            if source_blocks and start <= source_blocks[-1][1]:
                old_start, old_end = source_blocks[-1]
                source_blocks[-1] = (old_start, max(old_end, end))
            else:
                source_blocks.append((start, end))
        total_source_rows = max(len(source_rows), 1)
        seen_source_rows = 0
        for start, end in source_blocks:
            block_size = end - start + 1
            midpoint = seen_source_rows + block_size / 2
            source_fold = min(FOLD_COUNT - 1, int(midpoint * FOLD_COUNT / total_source_rows))
            for event_id in source_ids[start : end + 1]:
                source_fold_by_event[event_id] = source_fold
            seen_source_rows += block_size
    manifest["source_family_fold"] = manifest["canonical_event_id"].map(source_fold_by_event).astype(int)
    manifest = manifest.sort_values(["event_time", "canonical_event_id"]).reset_index(drop=True)
    # Critical safety validation.
    if manifest.groupby("event_group_id")["fold"].nunique().max() > 1:
        raise FreezeError("event group crosses folds")
    if manifest.groupby("event_group_id")["source_family_fold"].nunique().max() > 1:
        raise FreezeError("event group crosses source-family folds")
    article_folds = linked[["canonical_event_id", "syndication_root_id"]].merge(
        manifest[["canonical_event_id", "fold"]], on="canonical_event_id", how="inner"
    )
    if len(article_folds) and article_folds.groupby("syndication_root_id")["fold"].nunique().max() > 1:
        raise FreezeError("syndication root crosses folds")
    article_source_folds = linked[["canonical_event_id", "syndication_root_id"]].merge(
        manifest[["canonical_event_id", "source_family_fold"]], on="canonical_event_id", how="inner"
    )
    if len(article_source_folds) and article_source_folds.groupby("syndication_root_id")["source_family_fold"].nunique().max() > 1:
        raise FreezeError("syndication root crosses source-family folds")
    fold_ranges = manifest.groupby("fold")["event_time"].agg(["min", "max"]).sort_index()
    prior_max: pd.Timestamp | None = None
    for row in fold_ranges.itertuples():
        if prior_max is not None and row.min < prior_max:
            raise FreezeError("chronological fold ranges overlap")
        prior_max = row.max
    for source_family, source_rows in manifest.groupby("source_family", sort=True, dropna=False):
        source_ranges = source_rows.groupby("source_family_fold")["event_time"].agg(["min", "max"]).sort_index()
        source_prior_max: pd.Timestamp | None = None
        for row in source_ranges.itertuples():
            if source_prior_max is not None and row.min < source_prior_max:
                raise FreezeError(f"source-family chronological fold ranges overlap: {source_family}")
            source_prior_max = row.max
    return events, manifest


def build_feature_catalog(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    block_map = {
        "publisher": "PUBLISHER",
        "publisher_class": "PUBLISHER_CLASS",
        "topic": "TOPIC",
        "consensus": "SOURCE_CONSENSUS",
        "similarity": "SOURCE_CONSENSUS",
        "dispersion": "SOURCE_CONSENSUS",
        "contradiction": "SOURCE_CONSENSUS",
        "source": "SOURCE_DIVERSITY",
        "wire": "SOURCE_DIVERSITY",
        "biotech": "SOURCE_DIVERSITY",
        "issuer_official": "SOURCE_DIVERSITY",
        "syndication": "SYNDICATION",
        "novel": "NOVELTY",
        "prior_": "NOVELTY",
        "recycl": "NOVELTY",
        "structured": "STRUCTURED_CONFIRMATION",
        "sec_": "STRUCTURED_CONFIRMATION",
        "kind_": "STRUCTURED_CONFIRMATION",
        "first_seen": "TIMESTAMP_QUALITY",
        "published_": "TIMESTAMP_QUALITY",
        "minutes_": "TIMESTAMP_QUALITY",
        "headline": "HEADLINE_BODY_AGREEMENT",
        "body": "TEXT_CONTENT",
        "market_context": "MARKET_CONTEXT_JOIN",
        "model_agreement": "MODEL_AGREEMENT_JOIN",
    }
    rows: list[dict[str, Any]] = []
    for column in feature_matrix.columns:
        lower = column.lower()
        block = "EVENT_KEY"
        for needle, candidate in block_map.items():
            if needle in lower:
                block = candidate
                break
        causal = column not in {"canonical_event_id", "event_group_id", "market", "ticker", "event_time_utc", "decision_cutoff_utc"}
        rows.append(
            {
                "feature": column,
                "block": block,
                "dtype": str(feature_matrix[column].dtype),
                "source": "CAUSAL_EVENT_NEWS" if block != "EVENT_KEY" else "EVENT_MASTER",
                "available_at": "EVENT_T_PLUS_2" if causal else "EVENT_TIME",
                "causal_t2_eligible": bool(causal),
                "requires_first_seen": bool("first_seen" in lower),
                "quality_tier": "A_PLUS_B" if causal else "ALL",
                "target_dependent": False,
                "description": "Outcome-blind historical news source feature; target priors are intentionally absent.",
            }
        )
    return pd.DataFrame(rows).sort_values(["block", "feature"]).reset_index(drop=True)


def run_qa(events: pd.DataFrame, articles: pd.DataFrame, quality: pd.DataFrame, causal: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if events["canonical_event_id"].duplicated().any():
        failures.append("DUPLICATE_EVENT_ID")
    if articles["article_uid"].duplicated().any():
        failures.append("DUPLICATE_ARTICLE_UID")
    if "embedding_available" not in articles or not bool(articles["embedding_available"].all()):
        missing_embeddings = int((~articles.get("embedding_available", pd.Series(False, index=articles.index))).astype(bool)).sum()
        failures.append(f"EMBEDDING_COVERAGE_INCOMPLETE:{missing_embeddings}")
    nonempty_urls = articles.loc[articles["canonical_url"].ne(""), "canonical_url"]
    if nonempty_urls.duplicated().any():
        failures.append("DUPLICATE_CANONICAL_URL")
    future_vs_capture = (
        articles["published_at_utc"].notna() & articles["captured_at_utc"].notna()
        & (articles["published_at_utc"] > articles["captured_at_utc"] + pd.Timedelta(minutes=5))
    )
    if bool(future_vs_capture.any()):
        failures.append(f"PUBLICATION_AFTER_CAPTURE:{int(future_vs_capture.sum())}")
    future_causal = articles["causal_t2_eligible"] & (articles["published_at_utc"] > articles["decision_cutoff_utc"])
    if bool(future_causal.fillna(False).any()):
        failures.append(f"FUTURE_ARTICLE_IN_T2:{int(future_causal.fillna(False).sum())}")
    first_seen_future = (
        articles["causal_t2_eligible"] & articles["first_seen_observed"]
        & (articles["first_seen_at_utc"] > articles["decision_cutoff_utc"])
    )
    if bool(first_seen_future.fillna(False).any()):
        failures.append(f"FUTURE_FIRST_SEEN_IN_T2:{int(first_seen_future.fillna(False).sum())}")
    misused = quality["current_crawl_as_historical_first_seen"] & quality["causal_eligible"]
    if bool(misused.any()):
        failures.append(f"CURRENT_CRAWL_AS_FIRST_SEEN_IN_T2:{int(misused.sum())}")
    if articles.loc[articles["causal_t2_eligible"], "published_at_source"].map(_clean_string).eq("").any():
        failures.append("MISSING_TIMESTAMP_PROVENANCE_IN_CAUSAL_SET")
    if split.groupby("event_group_id")["fold"].nunique().max() > 1:
        failures.append("EVENT_GROUP_SPLIT_LEAKAGE")
    if split.groupby("event_group_id")["source_family_fold"].nunique().max() > 1:
        failures.append("EVENT_GROUP_SOURCE_FOLD_LEAKAGE")
    article_folds = articles.loc[articles["canonical_event_id"].ne(""), ["canonical_event_id", "syndication_root_id"]].merge(
        split[["canonical_event_id", "fold"]], on="canonical_event_id", how="inner"
    )
    if len(article_folds) and article_folds.groupby("syndication_root_id")["fold"].nunique().max() > 1:
        failures.append("SYNDICATION_ROOT_SPLIT_LEAKAGE")
    article_source_folds = articles.loc[articles["canonical_event_id"].ne(""), ["canonical_event_id", "syndication_root_id"]].merge(
        split[["canonical_event_id", "source_family_fold"]], on="canonical_event_id", how="inner"
    )
    if len(article_source_folds) and article_source_folds.groupby("syndication_root_id")["source_family_fold"].nunique().max() > 1:
        failures.append("SYNDICATION_ROOT_SOURCE_FOLD_LEAKAGE")
    if articles["headline"].eq("").any():
        warnings.append(f"EMPTY_HEADLINE:{int(articles['headline'].eq('').sum())}")
    if articles["body"].eq("").any():
        warnings.append(f"EMPTY_BODY:{int(articles['body'].eq('').sum())}")
    quarantined_timestamps = articles.get(
        "published_at_invalid_future_vs_capture", pd.Series(False, index=articles.index)
    ).astype(bool)
    if bool(quarantined_timestamps.any()):
        warnings.append(f"QUARANTINED_INVALID_PUBLICATION_TIMESTAMP:{int(quarantined_timestamps.sum())}")
    unknown_publishers = ~articles["publisher_canonical"].map(publisher_is_known)
    if bool(unknown_publishers.any()):
        warnings.append(f"UNKNOWN_PUBLISHER:{int(unknown_publishers.sum())}")
    if articles["canonical_event_id"].eq("").any():
        warnings.append(f"UNLINKED_ARTICLE:{int(articles['canonical_event_id'].eq('').sum())}")
    causal_count_check = articles.loc[articles["causal_t2_eligible"]].groupby("canonical_event_id")["article_uid"].count()
    reported = causal.set_index("canonical_event_id")[["tierA_article_count_t2", "tierB_article_count_t2"]].sum(axis=1)
    aligned = reported.index.to_series().map(causal_count_check).fillna(0).astype(int)
    if not np.array_equal(reported.astype(int).to_numpy(), aligned.to_numpy()):
        failures.append("CAUSAL_AGGREGATE_COUNT_MISMATCH")
    return {
        "qa_pass": not failures,
        "critical_failures": failures,
        "warnings": warnings,
        "checks": {
            "no_outcome_aware_columns": True,
            "no_future_article_in_causal_features": not any("FUTURE_ARTICLE" in x for x in failures),
            "no_future_first_seen_in_causal_features": not any("FUTURE_FIRST_SEEN" in x for x in failures),
            "no_current_crawl_as_historical_first_seen": not any("CURRENT_CRAWL" in x for x in failures),
            "event_group_split_safe": "EVENT_GROUP_SPLIT_LEAKAGE" not in failures,
            "event_group_source_fold_safe": "EVENT_GROUP_SOURCE_FOLD_LEAKAGE" not in failures,
            "syndication_group_safe": "SYNDICATION_ROOT_SPLIT_LEAKAGE" not in failures,
            "syndication_source_fold_safe": "SYNDICATION_ROOT_SOURCE_FOLD_LEAKAGE" not in failures,
            "timestamp_provenance_recorded": "MISSING_TIMESTAMP_PROVENANCE_IN_CAUSAL_SET" not in failures,
            "multilingual_embedding_coverage_complete": not any("EMBEDDING_COVERAGE" in x for x in failures),
            "research_seal_touched": False,
            "final_meta_touched": False,
            "v224_prospective_touched": False,
        },
    }


def _coverage_markdown(events: pd.DataFrame, articles: pd.DataFrame, causal: pd.DataFrame, qa: Mapping[str, Any]) -> str:
    tiers = articles["quality_tier"].value_counts()
    causal_events = causal["tierA_article_count_t2"] + causal["tierB_article_count_t2"] > 0
    lines = [
        "# Historical High-Confidence News Coverage Report",
        "",
        "This report is outcome-blind. No direction, return, correctness, Seal, Final, or V224 prospective data was read.",
        "",
        "## Coverage",
        "",
        f"- Total events: {len(events)}",
        f"- US events: {int((events['market'] == 'US').sum())}",
        f"- KR events: {int((events['market'] == 'KR').sum())}",
        f"- Total normalized articles: {len(articles)}",
        f"- Tier A: {int(tiers.get(TIER_A, 0))}",
        f"- Tier B_FULL: {int(tiers.get(TIER_B_FULL, 0))}",
        f"- Tier B_HEADLINE: {int(tiers.get(TIER_B_HEADLINE, 0))}",
        f"- Tier C: {int(tiers.get(TIER_C, 0))}",
        f"- Events with at least one t+2 causal article: {int(causal_events.sum())}",
        f"- Events with at least two independent sources: {int((causal['independent_source_count_t2'] >= 2).sum())}",
        f"- Events with at least three independent sources: {int((causal['independent_source_count_t2'] >= 3).sum())}",
        f"- Events with structured confirmation: {int((causal['structured_confirmation_count_t2'] > 0).sum())}",
        f"- Body coverage: {float(articles['body'].ne('').mean()) if len(articles) else 0.0:.4f}",
        f"- Exact-minute-or-better timestamp coverage: {float(articles['published_at_precision'].map(PRECISION_ORDER).ge(3).mean()) if len(articles) else 0.0:.4f}",
        f"- Multilingual embedding coverage: {float(articles['embedding_available'].mean()) if len(articles) and 'embedding_available' in articles else 0.0:.4f}",
        "",
        "## Year-by-year",
        "",
        "| Year | Events | Articles | Tier A | Tier B_FULL | Tier B_HEADLINE | Multi-source events |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    event_year = events.set_index("canonical_event_id")["event_time_utc"].dt.year
    article_year = articles["canonical_event_id"].map(event_year)
    causal_year = causal["canonical_event_id"].map(event_year)
    for year in range(2018, 2027):
        lines.append(
            f"| {year} | {int((events['event_time_utc'].dt.year == year).sum())} | "
            f"{int((article_year == year).sum())} | {int(((article_year == year) & (articles['quality_tier'] == TIER_A)).sum())} | "
            f"{int(((article_year == year) & (articles['quality_tier'] == TIER_B_FULL)).sum())} | "
            f"{int(((article_year == year) & (articles['quality_tier'] == TIER_B_HEADLINE)).sum())} | "
            f"{int(((causal_year == year) & (causal['independent_source_count_t2'] >= 2)).sum())} |"
        )
    lines.extend(["", "## Publisher distribution", ""])
    for publisher, count in articles["publisher_canonical"].value_counts().head(25).items():
        lines.append(f"- {publisher}: {int(count)}")
    lines.extend(["", "## Topic distribution", ""])
    for topic, count in articles["primary_topic"].value_counts().items():
        lines.append(f"- {topic}: {int(count)}")
    lines.extend(
        [
            "",
            "## Data views",
            "",
            f"- STRICT_A events: {int((causal['tierA_article_count_t2'] > 0).sum())}",
            f"- A_PLUS_B_FULL events: {int(causal_events.sum())}",
            f"- MULTISOURCE_ONLY events: {int((causal['independent_source_count_t2'] >= 2).sum())}",
            f"- STRUCTURED_CONFIRM_ONLY events: {int((causal['structured_confirmation_count_t2'] > 0).sum())}",
            f"- US_NEWS events: {int((events['market'] == 'US').sum())}",
            f"- KR_NEWS events: {int((events['market'] == 'KR').sum())}",
            "",
            "## QA",
            "",
            f"- QA pass: {str(bool(qa['qa_pass'])).upper()}",
            f"- Critical failures: {len(qa['critical_failures'])}",
            f"- Warnings: {len(qa['warnings'])}",
            "",
            "MODEL_READY = FALSE (source freeze is label-independent; the explicitly post-freeze DEV label join remains pending).",
            "SOURCE_FROZEN = TRUE" if qa["qa_pass"] else "SOURCE_FROZEN = FALSE",
            "LABEL_SELECTION_LEAKAGE = FALSE",
            "RESEARCH_SEAL_TOUCHED = FALSE",
            "FINAL_META_TOUCHED = FALSE",
            "",
        ]
    )
    return "\n".join(lines)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _fold_safe_spec() -> dict[str, Any]:
    return {
        "version": "FOLD_SAFE_TARGET_PRIORS_SPEC_V1",
        "implemented_in_this_data_session": False,
        "target_dependent_feature_values_frozen": False,
        "allowed_next_session_features": [
            "publisher_historical_correctness", "topic_historical_correctness", "publisher_topic_reliability"
        ],
        "requirements": {
            "past_only": True,
            "train_fold_only": True,
            "strict_nested_oof": True,
            "bayesian_shrinkage": True,
            "global_fallback": True,
            "source_fallback": True,
            "validation_or_test_rows_never_contribute": True,
        },
        "forbidden": [
            "full_dataset_target_encoding", "in_sample_correctness", "label_aware_source_filter",
            "label_aware_topic_filter", "source_membership_change_after_freeze",
        ],
    }


def _prepare_output(paths: BuildPaths, source_hashes: Mapping[str, str]) -> tuple[Path, Path]:
    paths.output_dir.parent.mkdir(parents=True, exist_ok=True)
    paths.research_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = paths.output_dir / "SOURCE_FREEZE.json"
    if freeze_path.is_file():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing.get("source_input_sha256") != dict(source_hashes):
            raise FreezeError("existing immutable source freeze has different input hashes")
        manifest = paths.output_dir / "SHA256_MANIFEST.csv"
        if not manifest.is_file():
            raise FreezeError("existing freeze is missing SHA256_MANIFEST.csv")
        for row in pd.read_csv(manifest).to_dict(orient="records"):
            file_path = paths.output_dir / str(row["file"])
            if not file_path.is_file() or sha256_file(file_path) != row["sha256"]:
                raise FreezeError(f"existing frozen artifact integrity failure: {file_path.name}")
        return paths.output_dir, paths.research_dir
    stage = Path(
        tempfile.mkdtemp(
            dir=str(paths.output_dir.parent.resolve()),
            prefix=f".{paths.output_dir.name}.stage_",
        )
    ).resolve()
    if stage.parent != paths.output_dir.parent.resolve() or stage.is_symlink():
        raise FreezeError(f"unsafe staging directory resolution: {stage}")
    return stage, paths.research_dir


def _read_label_table(path: Path) -> pd.DataFrame:
    """Read an explicitly authorized post-freeze DEV label table.

    Outcome columns are allowed only in this function. Source-defining fields are
    rejected so a label file cannot silently alter article membership or metadata.
    """
    _assert_safe_path(path)
    if not path.is_file():
        raise FreezeError(f"post-freeze label table missing: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        labels = pd.read_parquet(path)
    elif suffix == ".csv" or path.name.lower().endswith(".csv.gz"):
        labels = pd.read_csv(path, low_memory=False)
    else:
        raise FreezeError("post-freeze labels must be parquet, CSV, or CSV.GZ")
    authority_key = "canonical_event_id" if "canonical_event_id" in labels.columns else "event_id" if "event_id" in labels.columns else ""
    if not authority_key:
        raise FreezeError("post-freeze labels require canonical_event_id or event_id authority key")
    key_columns = [authority_key] + [column for column in ("market", "ticker", "event_time_utc") if column in labels.columns]
    retained = key_columns + [column for column in POST_FREEZE_LABEL_COLUMNS if column in labels.columns]
    ignored = sorted(set(labels.columns) - set(retained))
    labels = labels.loc[:, retained].copy()
    labels[authority_key] = labels[authority_key].map(_clean_string)
    if labels[authority_key].eq("").any():
        raise FreezeError("label authority keys must be nonempty")
    labels.attrs["authority_key"] = authority_key
    labels.attrs["ignored_columns"] = ignored
    return labels.sort_values(authority_key, kind="mergesort").reset_index(drop=True)


def _map_label_authority_to_frozen_events(labels: pd.DataFrame, frozen_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    authority_key = str(labels.attrs.get("authority_key", "canonical_event_id"))
    ignored_columns = list(labels.attrs.get("ignored_columns", []))
    events = _ensure_columns(frozen_events, ("existing_event_id", "join_key_event_id"))
    events = events.copy()
    for column in ("canonical_event_id", "existing_event_id", "join_key_event_id", "market", "ticker"):
        events[column] = events[column].map(_clean_string)
    events["event_time_utc"] = _to_utc(events["event_time_utc"])
    if authority_key == "canonical_event_id":
        mapped = labels.rename(columns={authority_key: "canonical_event_id"}).copy()
        if mapped["canonical_event_id"].duplicated().any():
            raise FreezeError("canonical label authority keys must be unique")
        in_scope = mapped["canonical_event_id"].isin(set(events["canonical_event_id"]))
        label_columns = [column for column in POST_FREEZE_LABEL_COLUMNS if column in mapped.columns]
        result = mapped.loc[in_scope, ["canonical_event_id", *label_columns]].copy()
        return result, {
            "label_rows_total": int(len(labels)),
            "label_rows_mapped": int(in_scope.sum()),
            "label_rows_outside_frozen_membership": int((~in_scope).sum()),
            "collision_rows_resolved_by_keys": 0,
            "collision_rows_resolved_by_exact_eligibility": 0,
            "ignored_label_columns": ignored_columns,
        }
    lookup: dict[str, set[int]] = defaultdict(set)
    for idx, row in events.iterrows():
        for key in (row["canonical_event_id"], row["existing_event_id"], row["join_key_event_id"]):
            if key:
                lookup[key].add(int(idx))
    mapped_rows: list[dict[str, Any]] = []
    outside = 0
    collision_resolved = 0
    collision_exact_eligibility = 0
    for row in labels.to_dict(orient="records"):
        raw_id = _clean_string(row.get("event_id"))
        candidate_idx = sorted(lookup.get(raw_id, set()))
        if not candidate_idx:
            outside += 1
            continue
        candidates = events.loc[candidate_idx]
        if len(candidates) > 1:
            if all(column in labels.columns for column in ("market", "ticker", "event_time_utc")):
                market = _clean_string(row.get("market")).upper()
                ticker = _ticker_comparison_key(row.get("ticker"), market)
                event_time = pd.to_datetime(row.get("event_time_utc"), errors="coerce", utc=True)
                candidates = candidates.loc[
                    (candidates["market"].str.upper() == market)
                    & pd.Series(
                        [
                            _ticker_comparison_key(candidate_ticker, candidate_market)
                            for candidate_ticker, candidate_market in zip(candidates["ticker"], candidates["market"])
                        ],
                        index=candidates.index,
                    ).eq(ticker)
                    & ((candidates["event_time_utc"] - event_time).abs() <= pd.Timedelta(seconds=1))
                ]
            if len(candidates) > 1 and "exact_price_eligible" in candidates.columns:
                exact_mask = candidates["exact_price_eligible"].map(
                    lambda value: bool(value) if not isinstance(value, str)
                    else value.strip().lower() in {"1", "true", "yes", "y"}
                )
                exact_candidates = candidates.loc[exact_mask]
                if len(exact_candidates) == 1:
                    candidates = exact_candidates
                    collision_exact_eligibility += 1
            if len(candidates) != 1:
                raise FreezeError(f"ambiguous collided raw event_id cannot be resolved by market/ticker/time: {raw_id}")
            collision_resolved += 1
        chosen = candidates.iloc[0]
        out = {key: value for key, value in row.items() if key not in {"event_id", "market", "ticker", "event_time_utc"}}
        out["canonical_event_id"] = chosen["canonical_event_id"]
        mapped_rows.append(out)
    retained_label_columns = [column for column in POST_FREEZE_LABEL_COLUMNS if column in labels.columns]
    mapped = pd.DataFrame(mapped_rows, columns=retained_label_columns + ["canonical_event_id"])
    if len(mapped) and mapped["canonical_event_id"].duplicated().any():
        raise FreezeError("multiple DEV label rows map to one frozen canonical_event_id")
    return mapped, {
        "label_rows_total": int(len(labels)),
        "label_rows_mapped": int(len(mapped)),
        "label_rows_outside_frozen_membership": int(outside),
        "collision_rows_resolved_by_keys": int(collision_resolved),
        "collision_rows_resolved_by_exact_eligibility": int(collision_exact_eligibility),
        "ignored_label_columns": ignored_columns,
    }


def join_dev_labels_after_freeze(paths: BuildPaths, labels_path: Path) -> dict[str, Any]:
    """Perform a membership-preserving DEV label join after source freeze.

    This is intentionally separate from :func:`build_freeze`; no caller can join
    outcomes accidentally by running the default command. Missing labels remain
    null and never remove a frozen event.
    """
    freeze_path = paths.output_dir / "SOURCE_FREEZE.json"
    manifest_path = paths.output_dir / "SHA256_MANIFEST.csv"
    matrix_path = paths.output_dir / "HIST_HIGH_CONF_NEWS_FEATURE_MATRIX.parquet"
    if not freeze_path.is_file() or not manifest_path.is_file() or not matrix_path.is_file():
        raise FreezeError("source freeze must complete before explicit DEV label join")
    freeze_before = sha256_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("source_membership_frozen") or not freeze.get("outcome_blind") or not freeze.get("qa_pass"):
        raise FreezeError("source freeze is not eligible for post-freeze label join")
    labels_path = labels_path.resolve()
    labels = _read_label_table(labels_path)
    frozen_events = pd.read_parquet(paths.output_dir / "EVENT_MASTER.parquet")
    labels, mapping_audit = _map_label_authority_to_frozen_events(labels, frozen_events)
    matrix = pd.read_parquet(matrix_path)
    if matrix["canonical_event_id"].duplicated().any():
        raise FreezeError("frozen feature matrix event membership is not unique")
    frozen_ids = matrix["canonical_event_id"].map(_clean_string)
    before_order = frozen_ids.tolist()
    labeled = matrix.merge(labels, on="canonical_event_id", how="left", validate="one_to_one", sort=False)
    if labeled["canonical_event_id"].map(_clean_string).tolist() != before_order or len(labeled) != len(matrix):
        raise FreezeError("post-freeze label join changed source membership or order")
    label_columns = [column for column in labels.columns if column != "canonical_event_id"]
    labeled_path = paths.output_dir / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet"
    status_path = paths.output_dir / "POST_FREEZE_LABEL_JOIN_STATUS.json"
    label_sha = sha256_file(labels_path)
    if status_path.is_file() and labeled_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("label_input_sha256") != label_sha:
            raise FreezeError("an immutable post-freeze label join already uses a different label input")
        if existing.get("labeled_file_sha256") != sha256_file(labeled_path):
            raise FreezeError("post-freeze labeled artifact integrity failure")
        return existing
    _write_parquet(labeled, labeled_path)
    matched = int(labeled[label_columns].notna().any(axis=1).sum()) if label_columns else 0
    status = {
        "version": "POST_FREEZE_DEV_LABEL_JOIN_V1",
        "source_freeze_sha256_before_join": freeze_before,
        "source_freeze_sha256_after_join": sha256_file(freeze_path),
        "source_freeze_unchanged": sha256_file(freeze_path) == freeze_before,
        "source_membership_sha256": hashlib.sha256("\n".join(before_order).encode("utf-8")).hexdigest(),
        "label_input_path": str(labels_path),
        "label_input_sha256": label_sha,
        "label_columns": label_columns,
        **mapping_audit,
        "frozen_event_rows": int(len(matrix)),
        "label_matched_rows": matched,
        "label_unmatched_rows": int(len(matrix) - matched),
        "membership_or_order_changed": False,
        "label_aware_filtering": False,
        "target_priors_built": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_prospective_touched": False,
        "labeled_file_sha256": sha256_file(labeled_path),
        "model_ready": True,
    }
    _write_json(status_path, status)
    # Extend the file integrity ledger. The source-freeze JSON itself remains byte
    # identical, and the join status records its before/after digest.
    manifest_rows = []
    for file_path in sorted(paths.output_dir.iterdir(), key=lambda x: x.name):
        if file_path.name == "SHA256_MANIFEST.csv" or not file_path.is_file():
            continue
        manifest_rows.append({"file": file_path.name, "sha256": sha256_file(file_path), "bytes": file_path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, lineterminator="\n")
    _write_json(paths.research_dir / "POST_FREEZE_LABEL_JOIN_REPORT.json", status)
    return status


def build_freeze(paths: BuildPaths) -> dict[str, Any]:
    event_path = paths.input_dir / "EVENT_MASTER.parquet"
    article_path = paths.input_dir / "ARTICLES_NORMALIZED.parquet"
    _assert_safe_path(paths.input_dir)
    _assert_safe_path(paths.output_dir)
    raw_events = _read_parquet(event_path)
    raw_articles = _read_parquet(article_path)
    embedding_bundle = load_embedding_bundle(paths.input_dir, raw_articles, require_complete=True)
    source_hashes = {
        "EVENT_MASTER.parquet": sha256_file(event_path) if event_path.is_file() else "MISSING",
        "ARTICLES_NORMALIZED.parquet": sha256_file(article_path) if article_path.is_file() else "MISSING",
        **embedding_bundle.source_hashes,
    }
    if "MISSING" in source_hashes.values():
        raise FreezeError(f"required source missing: {source_hashes}")
    stage, research_dir = _prepare_output(paths, source_hashes)
    if stage == paths.output_dir and (paths.output_dir / "SOURCE_FREEZE.json").is_file():
        return json.loads((paths.output_dir / "SOURCE_FREEZE.json").read_text(encoding="utf-8"))

    events = normalize_events(raw_events)
    articles = normalize_articles(raw_articles, events)
    articles = attach_embeddings(articles, embedding_bundle)
    articles, syndication = build_syndication(articles)
    articles = classify_topics(articles)
    articles, quality = build_quality(articles, events)
    topic_qa_sample = build_topic_qa_sample(articles)
    events, split = build_split_manifest(events, articles)
    # Carry the effective leak-safe groups into downstream tables.
    group_map = events.set_index("canonical_event_id")["event_group_id"]
    articles["event_group_id"] = articles["canonical_event_id"].map(group_map).fillna("")
    causal, event_clusters = build_event_features(events, articles)
    causal["event_group_id"] = causal["canonical_event_id"].map(group_map)
    event_clusters["event_group_id"] = event_clusters["canonical_event_id"].map(group_map)
    publisher_registry = build_publisher_registry(articles)
    feature_matrix = causal.copy()
    feature_catalog = build_feature_catalog(feature_matrix)
    qa = run_qa(events, articles, quality, causal, split)
    if not qa["qa_pass"]:
        research_dir.mkdir(parents=True, exist_ok=True)
        _write_json(research_dir / "HIST_NEWS_QA_REPORT.json", qa)
        raise FreezeError(f"freeze QA failed closed: {qa['critical_failures']}")

    tables = {
        "EVENT_MASTER.parquet": events,
        "ARTICLES_NORMALIZED.parquet": articles.drop(columns=["normalized_text", "embedding_vector"], errors="ignore"),
        "EMBEDDING_INDEX.parquet": pd.DataFrame(embedding_bundle.index_rows.values()).sort_values("article_uid").reset_index(drop=True),
        "ARTICLE_QUALITY_AUDIT.parquet": quality,
        "PUBLISHER_REGISTRY.parquet": publisher_registry,
        "SYNDICATION_CLUSTERS.parquet": syndication,
        "EVENT_NEWS_CLUSTERS.parquet": event_clusters,
        "CAUSAL_EVENT_NEWS.parquet": causal,
        "HIST_NEWS_DEV_SPLIT_MANIFEST.parquet": split,
        "HIST_HIGH_CONF_NEWS_FEATURE_MATRIX.parquet": feature_matrix,
    }
    for name, frame in tables.items():
        _write_parquet(frame, stage / name)
    feature_catalog.to_csv(stage / "FEATURE_CATALOG.csv", index=False, lineterminator="\n")
    topic_qa_sample.to_csv(stage / "TOPIC_QA_SAMPLE.csv", index=False, lineterminator="\n")
    coverage = _coverage_markdown(events, articles, causal, qa)
    (stage / "COVERAGE_REPORT.md").write_text(coverage, encoding="utf-8", newline="\n")
    code_recipe_rows = []
    for relative_name in CODE_RECIPE_FILES:
        recipe_path = ROOT / relative_name
        if recipe_path.is_file():
            code_recipe_rows.append(
                {
                    "file": relative_name,
                    "sha256": sha256_file(recipe_path),
                    "bytes": recipe_path.stat().st_size,
                    "recipe_role": "DATA_ONLY_NO_MODEL",
                }
            )
    code_recipe = pd.DataFrame(code_recipe_rows)
    code_recipe.to_csv(stage / "CODE_RECIPE_MANIFEST.csv", index=False, lineterminator="\n")
    _write_json(research_dir / "FOLD_SAFE_TARGET_PRIORS_SPEC.json", _fold_safe_spec())
    _write_json(research_dir / "HIST_NEWS_QA_REPORT.json", qa)

    semantic_hashes = {
        "article_set_sha256": semantic_sha(tables["ARTICLES_NORMALIZED.parquet"]),
        "event_set_sha256": semantic_sha(events),
        "publisher_registry_sha256": semantic_sha(publisher_registry),
        "syndication_clusters_sha256": semantic_sha(syndication),
        "causal_reconstruction_sha256": semantic_sha(causal),
        "split_manifest_sha256": semantic_sha(split),
        "feature_catalog_sha256": sha256_file(stage / "FEATURE_CATALOG.csv"),
        "topic_qa_sample_sha256": sha256_file(stage / "TOPIC_QA_SAMPLE.csv"),
        "code_recipe_manifest_sha256": sha256_file(stage / "CODE_RECIPE_MANIFEST.csv"),
        "embedding_index_semantic_sha256": semantic_sha(tables["EMBEDDING_INDEX.parquet"]),
    }
    tier_counts = articles["quality_tier"].value_counts()
    causal_mask = (causal["tierA_article_count_t2"] + causal["tierBFull_article_count_t2"]) > 0
    causal_market_counts = causal.loc[causal_mask, "market"].value_counts()
    multisource_count = int((causal["independent_source_count_t2"] >= 2).sum())
    ambiguous_count = int((articles["event_link_method"] == "AMBIGUOUS_EVENT_MAPPING").sum())
    other_topic_rate = float((articles["primary_topic"] == "OTHER").mean()) if len(articles) else 0.0
    freeze = {
        "format_version": FORMAT_VERSION,
        "source_input_sha256": source_hashes,
        **semantic_hashes,
        "topic_taxonomy_version": TOPIC_TAXONOMY_VERSION,
        "code_recipe_sha256": {row["file"]: row["sha256"] for row in code_recipe_rows},
        "embedding_model": embedding_bundle.model_metadata,
        "embedding_cache_root": str((paths.input_dir / "embeddings").resolve()),
        "embedding_coverage": {
            "required_articles": embedding_bundle.required_articles,
            "available_articles": embedding_bundle.available_articles,
            "coverage_ratio": embedding_bundle.coverage_ratio,
            "stale_text_articles": embedding_bundle.stale_text_articles,
            "missing_articles": embedding_bundle.missing_articles,
        },
        "source_membership_frozen": True,
        "outcome_blind": True,
        "label_join_complete": False,
        "model_ready": False,
        "model_ready_blocker": "EXPLICIT_POST_FREEZE_DEV_LABEL_JOIN_PENDING",
        "qa_pass": True,
        "counter_semantics": {
            "canonical_event_inventory_count": int(len(events)),
            "article_candidates_scanned": int(len(raw_articles)),
            "canonical_articles_count": int(len(articles)),
            "exact_eligible_events": int(events["exact_price_eligible"].fillna(False).astype(bool).sum()),
        },
        "freeze_gates": {
            "NO_OUTCOME_AWARE_COLLECTION": True,
            "NO_FUTURE_ARTICLE_IN_CAUSAL_FEATURES": True,
            "NO_ROLE_CONTAMINATION": True,
            "NO_RESEARCH_SEAL_ROWS": True,
            "NO_FINAL_META_ROWS": True,
            "NO_V224_PROSPECTIVE_ROWS": True,
            "EVENT_GROUP_SPLIT_SAFE": True,
            "SYNDICATION_GROUP_SAFE": True,
            "TIMESTAMP_PROVENANCE_RECORDED": True,
            "MULTILINGUAL_EMBEDDING_COVERAGE_COMPLETE": embedding_bundle.complete,
        },
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "quality_tiers": {
            TIER_A: int(tier_counts.get(TIER_A, 0)),
            TIER_B_FULL: int(tier_counts.get(TIER_B_FULL, 0)),
            TIER_B_HEADLINE: int(tier_counts.get(TIER_B_HEADLINE, 0)),
            TIER_C: int(tier_counts.get(TIER_C, 0)),
        },
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "causal_t2_event_count": int(causal_mask.sum()),
        "causal_t2_event_count_by_market": {str(k): int(v) for k, v in causal_market_counts.items()},
        "multi_source_t2_event_count": multisource_count,
        "multisource_features_base_v1_usable": bool(multisource_count >= 100 and multisource_count / max(int(causal_mask.sum()), 1) >= 0.01),
        "ambiguous_event_mapping_quarantined": ambiguous_count,
        "unlinked_articles": int(articles["canonical_event_id"].eq("").sum()),
        "other_topic_rate": other_topic_rate,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_prospective_touched": False,
    }
    _write_json(
        stage / "ARTICLE_EMBEDDINGS_MANIFEST.json",
        {
            "schema_version": "ARTICLE_EMBEDDINGS_BASE_V1",
            "index_file": "EMBEDDING_INDEX.parquet",
            "index_semantic_sha256": semantic_hashes["embedding_index_semantic_sha256"],
            "model": embedding_bundle.model_metadata,
            "coverage": freeze["embedding_coverage"],
            "outcome_blind": True,
        },
    )
    _write_json(stage / "SOURCE_FREEZE.json", freeze)
    manifest_rows = []
    for file_path in sorted(stage.iterdir(), key=lambda x: x.name):
        if file_path.name == "SHA256_MANIFEST.csv" or not file_path.is_file():
            continue
        manifest_rows.append({"file": file_path.name, "sha256": sha256_file(file_path), "bytes": file_path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(stage / "SHA256_MANIFEST.csv", index=False, lineterminator="\n")

    if paths.output_dir.exists():
        # An output directory without a freeze file is an incomplete previous build;
        # preserve it for inspection rather than deleting it.
        if any(paths.output_dir.iterdir()):
            raise FreezeError(f"non-empty unfrozen output directory requires manual review: {paths.output_dir}")
        paths.output_dir.rmdir()
    stage.replace(paths.output_dir)
    build_report = {
        "status": "SOURCE_FROZEN",
        "output_dir": str(paths.output_dir),
        "source_freeze_sha256": sha256_file(paths.output_dir / "SOURCE_FREEZE.json"),
        "manifest_sha256": sha256_file(paths.output_dir / "SHA256_MANIFEST.csv"),
        "row_counts": freeze["row_counts"],
        "quality_tiers": freeze["quality_tiers"],
        "causal_t2_event_count": freeze["causal_t2_event_count"],
        "model_ready": False,
        "label_join_pending": True,
    }
    _write_json(research_dir / "HIST_NEWS_BUILD_REPORT.json", build_report)
    return freeze


def build_preview(input_dir: Path, preview_dir: Path) -> dict[str, Any]:
    """Refresh bounded derived QA without claiming or creating a source freeze."""
    input_dir = input_dir.resolve()
    preview_dir = preview_dir.resolve()
    _assert_safe_path(input_dir)
    event_path = input_dir / "EVENT_MASTER.parquet"
    article_path = input_dir / "ARTICLES_NORMALIZED.parquet"
    if not event_path.is_file() or not article_path.is_file():
        raise FreezeError("preview requires EVENT_MASTER.parquet and ARTICLES_NORMALIZED.parquet")
    raw_articles = _read_parquet(article_path)
    events = normalize_events(_read_parquet(event_path))
    articles = normalize_articles(raw_articles, events)
    embedding_bundle = load_embedding_bundle(input_dir, raw_articles, require_complete=False)
    articles = attach_embeddings(articles, embedding_bundle)
    articles, syndication = build_syndication(articles)
    articles = classify_topics(articles)
    articles, quality = build_quality(articles, events)
    causal_articles = articles.loc[articles["causal_t2_eligible"]]
    causal_counts = (
        causal_articles.groupby("canonical_event_id", as_index=False)
        .agg(
            tierA_article_count_t2=("quality_tier", lambda x: int((x == TIER_A).sum())),
            tierB_article_count_t2=("quality_tier", lambda x: int((x == TIER_B_FULL).sum())),
            tierBFull_article_count_t2=("quality_tier", lambda x: int((x == TIER_B_FULL).sum())),
            independent_source_count_t2=("syndication_root_id", "nunique"),
            unique_publisher_count_t2=("origin_publisher", "nunique"),
        )
        .sort_values("canonical_event_id")
        .reset_index(drop=True)
    )
    future_causal = articles["causal_t2_eligible"] & (articles["published_at_utc"] > articles["decision_cutoff_utc"])
    future_first_seen = (
        articles["causal_t2_eligible"] & articles["first_seen_observed"]
        & (articles["first_seen_at_utc"] > articles["decision_cutoff_utc"])
    )
    misused = quality["causal_eligible"] & quality["current_crawl_as_historical_first_seen"]
    critical = []
    if bool(future_causal.fillna(False).any()):
        critical.append(f"FUTURE_ARTICLE_IN_T2:{int(future_causal.fillna(False).sum())}")
    if bool(future_first_seen.fillna(False).any()):
        critical.append(f"FUTURE_FIRST_SEEN_IN_T2:{int(future_first_seen.fillna(False).sum())}")
    if bool(misused.any()):
        critical.append(f"CURRENT_CRAWL_AS_FIRST_SEEN_IN_T2:{int(misused.sum())}")
    tier_counts = quality["tier"].value_counts()
    causal_markets = causal_counts[["canonical_event_id"]].merge(
        events[["canonical_event_id", "market"]], on="canonical_event_id", how="left", validate="one_to_one"
    ) if len(causal_counts) else pd.DataFrame(columns=["canonical_event_id", "market"])
    link_counts = articles["event_link_method"].value_counts()
    topic_counts = articles["primary_topic"].value_counts()
    preview = {
        "status": "DERIVED_PREVIEW_NOT_FROZEN",
        "source_frozen": False,
        "model_ready": False,
        "outcome_columns_read": False,
        "source_input_sha256": {
            "EVENT_MASTER.parquet": sha256_file(event_path),
            "ARTICLES_NORMALIZED.parquet": sha256_file(article_path),
            **embedding_bundle.source_hashes,
        },
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "counter_semantics": {
            "canonical_event_inventory_count": int(len(events)),
            "article_candidates_scanned": int(len(raw_articles)),
            "canonical_articles_count": int(len(articles)),
            "exact_eligible_events": int(events["exact_price_eligible"].fillna(False).astype(bool).sum()),
        },
        "row_counts": {
            "events": int(len(events)),
            "articles_before_dedup": int(len(raw_articles)),
            "articles_after_dedup": int(len(articles)),
            "syndication_roots": int(len(syndication)),
        },
        "quality_tiers": {
            TIER_A: int(tier_counts.get(TIER_A, 0)),
            TIER_B_FULL: int(tier_counts.get(TIER_B_FULL, 0)),
            TIER_B_HEADLINE: int(tier_counts.get(TIER_B_HEADLINE, 0)),
            TIER_C: int(tier_counts.get(TIER_C, 0)),
        },
        "causal_t2_event_count": int(causal_counts["canonical_event_id"].nunique()) if len(causal_counts) else 0,
        "causal_t2_event_count_by_market": {str(k): int(v) for k, v in causal_markets["market"].value_counts().items()},
        "multi_source_t2_event_count": int((causal_counts["independent_source_count_t2"] >= 2).sum()) if len(causal_counts) else 0,
        "article_link_methods": {str(k): int(v) for k, v in link_counts.items()},
        "ambiguous_event_mapping_quarantined": int(link_counts.get("AMBIGUOUS_EVENT_MAPPING", 0)),
        "unlinked_articles": int(articles["canonical_event_id"].eq("").sum()),
        "publisher_distribution": {str(k): int(v) for k, v in articles["publisher_canonical"].value_counts().head(50).items()},
        "topic_distribution": {str(k): int(v) for k, v in topic_counts.items()},
        "other_topic_rate": float(topic_counts.get("OTHER", 0) / len(articles)) if len(articles) else 0.0,
        "timestamp_precision_distribution": {str(k): int(v) for k, v in articles["published_at_precision"].value_counts().items()},
        "first_seen_observed_count": int(articles["first_seen_observed"].sum()),
        "embedding": {
            "model_id": embedding_bundle.model_metadata.get("model_id"),
            "model_revision": embedding_bundle.model_metadata.get("model_revision"),
            "required_articles": embedding_bundle.required_articles,
            "available_articles": embedding_bundle.available_articles,
            "coverage_ratio": embedding_bundle.coverage_ratio,
            "stale_text_articles": embedding_bundle.stale_text_articles,
            "missing_articles": embedding_bundle.missing_articles,
            "errors": embedding_bundle.errors[:50],
            "complete": embedding_bundle.complete,
        },
        "qa_pass": not critical,
        "critical_failures": critical,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_prospective_touched": False,
        "preview_semantic_sha256": {
            "quality": semantic_sha(quality),
            "causal_counts": semantic_sha(causal_counts),
        },
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    quality_path = preview_dir / "PREVIEW_ARTICLE_QUALITY.parquet"
    causal_path = preview_dir / "PREVIEW_CAUSAL_COUNTS.parquet"
    status_path = preview_dir / "PREVIEW_STATUS.json"
    for frame, target in ((quality, quality_path), (causal_counts, causal_path)):
        temp = preview_dir / f".{target.name}.tmp_{os.getpid()}"
        _write_parquet(frame, temp)
        temp.replace(target)
    temp_status = preview_dir / f".{status_path.name}.tmp_{os.getpid()}"
    _write_json(temp_status, preview)
    temp_status.replace(status_path)
    return preview


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--resume", action="store_true", help="Harmless compatibility flag; immutable outputs are verified and reused.")
    parser.add_argument("--preview", action="store_true", help="Refresh non-frozen bounded dedup/topic/t+2 QA aggregates only.")
    parser.add_argument("--preview-dir", type=Path, help="Preview output; defaults to INPUT/derived_preview.")
    parser.add_argument(
        "--join-dev-labels-after-freeze",
        action="store_true",
        help="Explicitly run the membership-preserving label join; never enabled by default.",
    )
    parser.add_argument("--labels-path", type=Path, help="Parquet/CSV DEV label authority for the explicit post-freeze join.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = BuildPaths(args.input_dir.resolve(), args.output_dir.resolve(), args.research_dir.resolve())
        if args.preview:
            preview_dir = args.preview_dir or (args.input_dir / "derived_preview")
            preview = build_preview(args.input_dir, preview_dir)
            print(_stable_json(preview))
            return 0
        if args.join_dev_labels_after_freeze:
            if args.labels_path is None:
                raise FreezeError("--labels-path is required with --join-dev-labels-after-freeze")
            status = join_dev_labels_after_freeze(paths, args.labels_path)
            print(_stable_json({"status": "POST_FREEZE_LABEL_JOIN_COMPLETE", **status}))
            return 0
        freeze = build_freeze(paths)
    except FreezeError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        _stable_json(
            {
                "status": "SOURCE_FROZEN",
                "output_dir": str(args.output_dir.resolve()),
                "events": freeze["row_counts"]["EVENT_MASTER.parquet"],
                "articles": freeze["row_counts"]["ARTICLES_NORMALIZED.parquet"],
                "causal_t2_events": freeze["causal_t2_event_count"],
                "model_ready": freeze["model_ready"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
