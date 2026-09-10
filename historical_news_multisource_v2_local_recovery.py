from __future__ import annotations

import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V1 = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
EMBED_ROOT = ROOT / "data" / "historical_news_backfill" / "embeddings"
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
POLICY_MANIFEST = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
OUT = ROOT / "research" / "multisource_v2"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"


NAVER_OID_REGISTRY: dict[str, tuple[str, str]] = {
    "001": ("연합뉴스", "MAJOR_WIRE"),
    "003": ("뉴시스", "MAJOR_WIRE"),
    "008": ("머니투데이", "MAJOR_FINANCIAL"),
    "009": ("매일경제", "MAJOR_FINANCIAL"),
    "011": ("서울경제", "MAJOR_FINANCIAL"),
    "014": ("파이낸셜뉴스", "MAJOR_FINANCIAL"),
    "015": ("한국경제", "MAJOR_FINANCIAL"),
    "016": ("헤럴드경제", "MAJOR_GENERAL"),
    "018": ("이데일리", "MAJOR_FINANCIAL"),
    "020": ("동아일보", "MAJOR_GENERAL"),
    "021": ("문화일보", "MAJOR_GENERAL"),
    "022": ("세계일보", "MAJOR_GENERAL"),
    "023": ("조선일보", "MAJOR_GENERAL"),
    "025": ("중앙일보", "MAJOR_GENERAL"),
    "028": ("한겨레", "MAJOR_GENERAL"),
    "029": ("디지털타임스", "MAJOR_FINANCIAL"),
    "030": ("전자신문", "MAJOR_FINANCIAL"),
    "031": ("아이뉴스24", "MAJOR_GENERAL"),
    "032": ("경향신문", "MAJOR_GENERAL"),
    "052": ("YTN", "MAJOR_GENERAL"),
    "055": ("SBS", "MAJOR_GENERAL"),
    "056": ("KBS", "MAJOR_GENERAL"),
    "057": ("MBN", "MAJOR_GENERAL"),
    "082": ("부산일보", "LOCAL_OTHER_MEDIA"),
    "119": ("데일리안", "MAJOR_GENERAL"),
    "214": ("MBC", "MAJOR_GENERAL"),
    "215": ("한국경제TV", "MAJOR_FINANCIAL"),
    "277": ("아시아경제", "MAJOR_FINANCIAL"),
    "366": ("조선비즈", "MAJOR_FINANCIAL"),
    "417": ("머니S", "MAJOR_FINANCIAL"),
    "421": ("뉴스1", "MAJOR_WIRE"),
}

PUBLISHER_CLASSES: dict[str, str] = {
    "Reuters": "MAJOR_WIRE",
    "Bloomberg": "MAJOR_FINANCIAL",
    "CNBC": "MAJOR_FINANCIAL",
    "MarketWatch": "MAJOR_FINANCIAL",
    "The Wall Street Journal": "MAJOR_FINANCIAL",
    "Barron's": "MAJOR_FINANCIAL",
    "Financial Times": "MAJOR_FINANCIAL",
    "Associated Press": "MAJOR_WIRE",
    "STAT": "BIOTECH_SPECIALIST",
    "Endpoints News": "BIOTECH_SPECIALIST",
    "Fierce Biotech": "BIOTECH_SPECIALIST",
    "Fierce Pharma": "PHARMA_SPECIALIST",
    "BioSpace": "BIOTECH_SPECIALIST",
    "연합뉴스": "MAJOR_WIRE",
    "뉴시스": "MAJOR_WIRE",
    "뉴스1": "MAJOR_WIRE",
    "한국경제": "MAJOR_FINANCIAL",
    "매일경제": "MAJOR_FINANCIAL",
    "서울경제": "MAJOR_FINANCIAL",
    "머니투데이": "MAJOR_FINANCIAL",
    "이데일리": "MAJOR_FINANCIAL",
    "파이낸셜뉴스": "MAJOR_FINANCIAL",
    "아시아경제": "MAJOR_FINANCIAL",
    "한국경제TV": "MAJOR_FINANCIAL",
    "전자신문": "MAJOR_FINANCIAL",
    "조선비즈": "MAJOR_FINANCIAL",
    "데일리팜": "PHARMA_SPECIALIST",
    "히트뉴스": "PHARMA_SPECIALIST",
    "청년의사": "BIOTECH_SPECIALIST",
    "의학신문": "PHARMA_SPECIALIST",
}

AGGREGATORS = {
    "", "UNKNOWN", "Naver News", "Google News", "Yahoo Finance", "v.daum.net",
    "Daum", "네이트", "Nate", "Investing.com 한국어",
}

LINK_THRESHOLDS = {
    "embedding_min": 0.88,
    "sequence_ratio_min": 0.58,
    "char3_jaccard_min": 0.30,
    "discovery_lookback_hours": 24,
    "causal_cutoff_minutes": 2,
    "body_min_chars_for_strict_independence": 80,
    "syndication_sequence_ratio": 0.92,
    "syndication_char3_jaccard": 0.82,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def normalized_headline(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def char_ngrams(value: Any, n: int = 3) -> set[str]:
    text = normalized_headline(value)
    return {text[index:index + n] for index in range(max(0, len(text) - n + 1))}


def lexical_similarity(left: Any, right: Any) -> tuple[float, float]:
    left_text = normalized_headline(left)
    right_text = normalized_headline(right)
    sequence = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    left_grams = char_ngrams(left_text)
    right_grams = char_ngrams(right_text)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return float(sequence), float(jaccard)


def naver_oid(url: Any, event_id: Any = "") -> str:
    match = re.search(r"/article/(\d{3})/", str(url or ""))
    if match:
        return match.group(1)
    match = re.search(r"^NAVER:[^:]+:(\d{3})", str(event_id or ""))
    return match.group(1) if match else ""


def publisher_class(publisher: Any) -> str:
    value = str(publisher or "").strip()
    if value in AGGREGATORS:
        return "AGGREGATOR"
    return PUBLISHER_CLASSES.get(value, "LOCAL_OTHER_MEDIA")


def resolve_current_origin(row: pd.Series) -> dict[str, Any]:
    raw_origin = str(row.get("origin_publisher") or "").strip()
    host = str(row.get("hosting_publisher") or "").strip()
    oid = naver_oid(row.get("canonical_url"), row.get("canonical_event_id"))
    if host == "Naver News" or raw_origin == "Naver News":
        if oid in NAVER_OID_REGISTRY:
            origin, category = NAVER_OID_REGISTRY[oid]
            return {
                "origin_publisher_resolved": origin,
                "publisher_class_resolved": category,
                "origin_resolution_method": "NAVER_OID_STATIC_PUBLIC_REGISTRY",
                "origin_resolution_confidence": "HIGH",
                "naver_oid": oid,
            }
        return {
            "origin_publisher_resolved": "UNKNOWN",
            "publisher_class_resolved": "AGGREGATOR",
            "origin_resolution_method": "NAVER_OID_UNRESOLVED",
            "origin_resolution_confidence": "LOW",
            "naver_oid": oid,
        }
    return {
        "origin_publisher_resolved": raw_origin or "UNKNOWN",
        "publisher_class_resolved": publisher_class(raw_origin),
        "origin_resolution_method": "V1_ORIGIN_PROVENANCE",
        "origin_resolution_confidence": "HIGH" if raw_origin and raw_origin not in AGGREGATORS else "LOW",
        "naver_oid": oid,
    }


def classify_relation(
    alternate_origin: str,
    current_origin: str,
    body_length: int,
    sequence_ratio: float,
    char3_jaccard: float,
) -> tuple[str, bool]:
    if alternate_origin in AGGREGATORS or publisher_class(alternate_origin) == "AGGREGATOR":
        return "HOST_OR_AGGREGATOR_ONLY", False
    if not current_origin or current_origin == "UNKNOWN":
        return "CURRENT_ORIGIN_UNRESOLVED", False
    if alternate_origin == current_origin:
        return "SAME_ORIGIN_MIRROR", False
    if (
        sequence_ratio >= LINK_THRESHOLDS["syndication_sequence_ratio"]
        or char3_jaccard >= LINK_THRESHOLDS["syndication_char3_jaccard"]
    ):
        return "POTENTIAL_SYNDICATION_OR_COMMON_INFORMATION_ROOT", False
    if body_length < LINK_THRESHOLDS["body_min_chars_for_strict_independence"]:
        return "HEADLINE_ONLY_DISTINCT_MEDIA_CANDIDATE", False
    return "STRICT_DISTINCT_MEDIA_ROOT", True


def load_vectors(article_uids: Iterable[str], index: pd.DataFrame) -> dict[str, np.ndarray]:
    wanted = {str(value) for value in article_uids}
    selected = index.loc[index["article_uid"].astype(str).isin(wanted)]
    vectors: dict[str, np.ndarray] = {}
    for shard_path, group in selected.groupby("shard_path", sort=True):
        array = np.load(EMBED_ROOT / str(shard_path), mmap_mode="r", allow_pickle=False)
        for row in group.itertuples(index=False):
            vectors[str(row.article_uid)] = np.asarray(array[int(row.shard_index)], dtype=np.float32)
    return vectors


def build_current_provenance(queue: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    queue_ids = set(queue["canonical_event_id"].astype(str))
    current = articles.loc[
        articles["canonical_event_id"].astype(str).isin(queue_ids)
        & articles["causal_t2_eligible"].fillna(False).astype(bool)
    ].copy()
    current = current.sort_values(["canonical_event_id", "published_at_utc", "article_uid"], kind="mergesort")
    rows = []
    for event_id, group in current.groupby("canonical_event_id", sort=False):
        row = group.iloc[0]
        resolution = resolve_current_origin(row)
        rows.append(
            {
                "canonical_event_id": str(event_id),
                "current_article_uid": str(row["article_uid"]),
                "hosting_publisher": str(row.get("hosting_publisher") or ""),
                "origin_publisher_v1": str(row.get("origin_publisher") or ""),
                **resolution,
                "current_published_at_utc": pd.to_datetime(row["published_at_utc"], utc=True),
                "current_headline": str(row.get("headline") or ""),
                "current_body_length": len(str(row.get("body") or "")),
                "base_v1_mutated": False,
            }
        )
    result = pd.DataFrame(rows)
    if result["canonical_event_id"].duplicated().any():
        raise RuntimeError("current provenance event duplication")
    return result


def build_local_link_audit(
    queue: pd.DataFrame,
    articles: pd.DataFrame,
    embedding_index: pd.DataFrame,
    current_provenance: pd.DataFrame,
) -> pd.DataFrame:
    queue = queue.copy()
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    queue["decision_cutoff_utc"] = queue["event_time_utc"] + pd.Timedelta(
        minutes=LINK_THRESHOLDS["causal_cutoff_minutes"]
    )
    unlinked = articles.loc[
        (articles["canonical_event_id"].isna() | articles["canonical_event_id"].astype(str).str.strip().eq(""))
        & ~articles["origin_publisher"].fillna("").astype(str).isin({"Reuters", "Naver News"})
    ].copy()
    unlinked["published_at_utc"] = pd.to_datetime(unlinked["published_at_utc"], utc=True, errors="coerce")
    unlinked = unlinked.loc[unlinked["published_at_utc"].notna()].copy()

    current_articles = articles.merge(
        current_provenance[["canonical_event_id", "current_article_uid", "origin_publisher_resolved"]],
        left_on=["canonical_event_id", "article_uid"],
        right_on=["canonical_event_id", "current_article_uid"],
        how="inner",
        validate="many_to_one",
    )
    needed = set(unlinked["article_uid"].astype(str)) | set(current_articles["article_uid"].astype(str))
    vectors = load_vectors(needed, embedding_index)
    current_by_event = {str(key): group for key, group in current_articles.groupby("canonical_event_id", sort=False)}
    origin_by_event = current_provenance.set_index("canonical_event_id")["origin_publisher_resolved"].to_dict()

    pairs: list[dict[str, Any]] = []
    for alternate in unlinked.itertuples(index=False):
        alternate_uid = str(alternate.article_uid)
        if alternate_uid not in vectors:
            continue
        candidates = queue.loc[
            queue["market"].astype(str).eq(str(alternate.market))
            & queue["ticker"].astype(str).eq(str(alternate.ticker))
        ]
        candidates = candidates.loc[
            (alternate.published_at_utc >= (
                candidates["event_time_utc"] - pd.Timedelta(hours=LINK_THRESHOLDS["discovery_lookback_hours"])
            ))
            & (alternate.published_at_utc <= candidates["decision_cutoff_utc"])
        ]
        for event in candidates.itertuples(index=False):
            group = current_by_event.get(str(event.canonical_event_id))
            if group is None:
                continue
            best: tuple[float, float, float, str, str] | None = None
            for current in group.itertuples(index=False):
                current_uid = str(current.article_uid)
                if current_uid not in vectors:
                    continue
                embedding = float(np.dot(vectors[alternate_uid], vectors[current_uid]))
                sequence, jaccard = lexical_similarity(alternate.headline, current.headline)
                score = 0.45 * embedding + 0.30 * sequence + 0.25 * jaccard
                record = (score, embedding, sequence, current_uid, str(current.headline))
                if best is None or record[0] > best[0]:
                    best = record
            if best is None:
                continue
            score, embedding, sequence, current_uid, current_headline = best
            _, jaccard = lexical_similarity(alternate.headline, current_headline)
            accepted_semantic_match = bool(
                embedding >= LINK_THRESHOLDS["embedding_min"]
                and sequence >= LINK_THRESHOLDS["sequence_ratio_min"]
                and jaccard >= LINK_THRESHOLDS["char3_jaccard_min"]
            )
            if not accepted_semantic_match:
                continue
            alternate_origin = str(alternate.origin_publisher or "").strip()
            current_origin = str(origin_by_event.get(str(event.canonical_event_id), "UNKNOWN"))
            body_length = len(str(alternate.body or ""))
            relation, strict_independent = classify_relation(
                alternate_origin, current_origin, body_length, sequence, jaccard
            )
            pairs.append(
                {
                    "canonical_event_id": str(event.canonical_event_id),
                    "market": str(event.market),
                    "ticker": str(event.ticker),
                    "event_time_utc": event.event_time_utc,
                    "decision_cutoff_utc": event.decision_cutoff_utc,
                    "alternate_article_uid": alternate_uid,
                    "current_article_uid": current_uid,
                    "alternate_origin_publisher": alternate_origin,
                    "current_origin_publisher_resolved": current_origin,
                    "alternate_publisher_class": publisher_class(alternate_origin),
                    "alternate_hosting_publisher": str(alternate.hosting_publisher or ""),
                    "alternate_source_provider": str(alternate.source_provider or ""),
                    "alternate_url": str(alternate.canonical_url or ""),
                    "alternate_published_at_utc": alternate.published_at_utc,
                    "alternate_body_length": body_length,
                    "alternate_headline": str(alternate.headline or ""),
                    "current_headline": current_headline,
                    "embedding_similarity": embedding,
                    "headline_sequence_ratio": sequence,
                    "headline_char3_jaccard": jaccard,
                    "match_score": score,
                    "absolute_event_delta_minutes": abs(
                        (alternate.published_at_utc - event.event_time_utc).total_seconds() / 60.0
                    ),
                    "causal_published_by_t2": True,
                    "relation_class": relation,
                    "strict_independent_root_accepted": strict_independent,
                    "requires_body_recovery": body_length < LINK_THRESHOLDS["body_min_chars_for_strict_independence"],
                    "event_match_method": "TICKER_TIME_EMBEDDING_AND_LEXICAL_V2",
                    "event_match_quality_v2": "HIGH",
                    "outcome_fields_read": "",
                }
            )
    if not pairs:
        return pd.DataFrame()
    frame = pd.DataFrame(pairs)
    frame = frame.sort_values(
        ["alternate_article_uid", "absolute_event_delta_minutes", "match_score", "canonical_event_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).drop_duplicates("alternate_article_uid", keep="first")
    return frame.sort_values(["canonical_event_id", "alternate_article_uid"], kind="mergesort").reset_index(drop=True)


def build_publisher_registry(current: pd.DataFrame, local_links: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in current.itertuples(index=False):
        publisher = str(row.origin_publisher_resolved)
        rows.append(
            {
                "publisher_canonical": publisher,
                "publisher_class": str(row.publisher_class_resolved),
                "is_major_media": str(row.publisher_class_resolved).startswith("MAJOR_"),
                "is_aggregator": str(row.publisher_class_resolved) == "AGGREGATOR",
                "registry_evidence": str(row.origin_resolution_method),
            }
        )
    if not local_links.empty:
        for row in local_links.itertuples(index=False):
            rows.append(
                {
                    "publisher_canonical": str(row.alternate_origin_publisher),
                    "publisher_class": str(row.alternate_publisher_class),
                    "is_major_media": str(row.alternate_publisher_class).startswith("MAJOR_"),
                    "is_aggregator": str(row.alternate_publisher_class) == "AGGREGATOR",
                    "registry_evidence": "PUBLIC_RSS_SOURCE_METADATA",
                }
            )
    registry = pd.DataFrame(rows).sort_values(
        ["publisher_canonical", "registry_evidence"], kind="mergesort"
    ).drop_duplicates("publisher_canonical", keep="first")
    registry["reliability_hardcoded"] = False
    registry["outcome_fields_read"] = ""
    return registry.reset_index(drop=True)


def main() -> int:
    if not POLICY_MANIFEST.is_file():
        raise RuntimeError("V2 policy must be frozen before local recovery")
    policy = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("V2 policy does not permit outcome-blind collection")

    queue = pd.read_parquet(QUEUE)
    articles = pd.read_parquet(V1 / "ARTICLES_NORMALIZED.parquet")
    embedding_index = pd.read_parquet(V1 / "EMBEDDING_INDEX.parquet")
    current = build_current_provenance(queue, articles)
    links = build_local_link_audit(queue, articles, embedding_index, current)
    registry = build_publisher_registry(current, links)

    OUT.mkdir(parents=True, exist_ok=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    current_path = WORKING / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
    link_path = WORKING / "V2_LOCAL_CACHE_LINK_AUDIT.parquet"
    publisher_path = WORKING / "V2_PUBLISHER_REGISTRY.parquet"
    atomic_parquet(current_path, current)
    atomic_parquet(link_path, links)
    atomic_parquet(publisher_path, registry)

    unresolved = int(current["origin_publisher_resolved"].eq("UNKNOWN").sum())
    relation_counts = (
        {str(key): int(value) for key, value in links["relation_class"].value_counts().items()}
        if not links.empty else {}
    )
    summary = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_LOCAL_RECOVERY_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "queue_events": int(len(queue)),
        "current_source_provenance_rows": int(len(current)),
        "current_origin_resolved": int(len(current) - unresolved),
        "current_origin_unresolved": unresolved,
        "current_origin_resolution_rate": float((len(current) - unresolved) / len(current)) if len(current) else 0.0,
        "local_alternate_articles_high_match": int(len(links)),
        "local_alternate_events_high_match": int(links["canonical_event_id"].nunique()) if not links.empty else 0,
        "strict_independent_roots_added": int(links["strict_independent_root_accepted"].sum()) if not links.empty else 0,
        "headline_only_body_recovery_candidates": int(links["requires_body_recovery"].sum()) if not links.empty else 0,
        "relation_counts": relation_counts,
        "thresholds": LINK_THRESHOLDS,
        "base_v1_mutated": False,
        "source_freeze_ready": False,
        "model_training_allowed": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {
            current_path.name: sha256_file(current_path),
            link_path.name: sha256_file(link_path),
            publisher_path.name: sha256_file(publisher_path),
        },
    }
    atomic_json(OUT / "LOCAL_CACHE_RECOVERY_REPORT.json", summary)

    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update(
        {
            "updated_at_utc": utc_now(),
            "status": "LOCAL_PROVENANCE_RECOVERED_NETWORK_COLLECTION_REQUIRED",
            "phase": "OUTCOME_BLIND_ALTERNATE_SOURCE_COLLECTION",
            "current_origin_resolved": summary["current_origin_resolved"],
            "current_origin_unresolved": summary["current_origin_unresolved"],
            "local_alternate_events_high_match": summary["local_alternate_events_high_match"],
            "strict_independent_roots_added": summary["strict_independent_roots_added"],
            "headline_only_body_recovery_candidates": summary["headline_only_body_recovery_candidates"],
            "labels_read": False,
            "source_frozen": False,
            "split_frozen": False,
            "model_training_allowed": False,
            "base_v1_membership_mutable": False,
            "research_seal_touched": False,
            "final_meta_touched": False,
            "v224_touched": False,
        }
    )
    atomic_json(STATE, state)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
