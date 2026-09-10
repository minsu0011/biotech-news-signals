from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
V1 = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
RESEARCH_V1 = ROOT / "research" / "high_conf_news_base_v1"
OUT = ROOT / "research" / "hist_news_multisource_v2"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

CANDIDATE_SPLIT = RESEARCH_V1 / "HIST_NEWS_MODEL_CANDIDATE_SPLIT_MANIFEST.parquet"
QUEUE = OUT / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
SUMMARY = OUT / "QUEUE_SUMMARY.json"
MANIFEST = OUT / "SHA256_MANIFEST.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def _joined(values: pd.Series) -> str:
    cleaned = sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})
    return "|".join(cleaned)


def build_queue(
    candidates: pd.DataFrame,
    causal: pd.DataFrame,
    events: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    required_candidate = {
        "canonical_event_id", "model_event_group_id", "market", "ticker", "source_family",
        "event_time_utc", "candidate_contract",
    }
    missing = sorted(required_candidate - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate split missing columns: {missing}")

    causal_columns = [
        "canonical_event_id", "independent_source_count_t2", "unique_publisher_count_t2",
        "syndication_root_count_t2", "structured_confirmation_count_t2",
    ]
    queue = candidates.merge(causal[causal_columns], on="canonical_event_id", how="left", validate="one_to_one")
    event_columns = ["canonical_event_id", "issuer_name", "event_type", "source_provider"]
    queue = queue.merge(events[event_columns], on="canonical_event_id", how="left", validate="one_to_one")

    causal_articles = articles.loc[articles["causal_t2_eligible"].fillna(False).astype(bool)].copy()
    article_summary = (
        causal_articles.groupby("canonical_event_id", sort=False)
        .agg(
            current_article_count_t2=("article_uid", "nunique"),
            current_publishers_t2=("origin_publisher", _joined),
            current_providers_t2=("source_provider", _joined),
            current_source_types_t2=("source_type", _joined),
        )
        .reset_index()
    )
    queue = queue.merge(article_summary, on="canonical_event_id", how="left", validate="one_to_one")

    numeric = [
        "independent_source_count_t2", "unique_publisher_count_t2", "syndication_root_count_t2",
        "structured_confirmation_count_t2", "current_article_count_t2",
    ]
    for column in numeric:
        queue[column] = pd.to_numeric(queue[column], errors="coerce").fillna(0).astype(int)
    for column in ("current_publishers_t2", "current_providers_t2", "current_source_types_t2"):
        queue[column] = queue[column].fillna("").astype(str)

    queue = queue.loc[queue["independent_source_count_t2"] < 2].copy()
    queue["schema_version"] = "HIST_NEWS_MULTISOURCE_V2_PRIORITY_V1"
    queue["outcome_blind"] = True
    queue["append_only_v2"] = True
    queue["v1_membership_mutable"] = False
    queue["target_independent_source_count"] = 2
    queue["additional_independent_sources_needed"] = 2 - queue["independent_source_count_t2"]
    queue["priority_bucket"] = queue["independent_source_count_t2"].map(
        {1: "P0_ADD_SECOND_INDEPENDENT_SOURCE", 0: "P1_RECOVER_FIRST_CAUSAL_SOURCE"}
    ).fillna("P2_REVIEW")
    queue["collection_status"] = "NEEDS_ALTERNATE_PROVIDER"
    queue["allowed_discovery_routes"] = "PUBLIC_RSS|PUBLISHER_ARCHIVE|ISSUER_PR_ARCHIVE|PUBLIC_API|EXISTING_PROVIDER"
    queue["same_url_retry_allowed"] = False
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")

    priority_order = {
        "P0_ADD_SECOND_INDEPENDENT_SOURCE": 0,
        "P1_RECOVER_FIRST_CAUSAL_SOURCE": 1,
        "P2_REVIEW": 2,
    }
    queue["_priority_order"] = queue["priority_bucket"].map(priority_order).astype(int)
    queue = queue.sort_values(
        ["_priority_order", "market", "event_time_utc", "ticker", "canonical_event_id"],
        kind="mergesort",
    ).drop(columns="_priority_order").reset_index(drop=True)
    queue.insert(0, "priority_rank", range(1, len(queue) + 1))
    return queue


def main() -> None:
    required = [
        V1 / "SOURCE_FREEZE.json",
        V1 / "CAUSAL_EVENT_NEWS.parquet",
        V1 / "EVENT_MASTER.parquet",
        V1 / "ARTICLES_NORMALIZED.parquet",
        CANDIDATE_SPLIT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V1 authority files: {missing}")

    source_freeze_before = sha256_file(V1 / "SOURCE_FREEZE.json")
    candidates = pd.read_parquet(CANDIDATE_SPLIT)
    causal = pd.read_parquet(V1 / "CAUSAL_EVENT_NEWS.parquet")
    events = pd.read_parquet(V1 / "EVENT_MASTER.parquet")
    articles = pd.read_parquet(V1 / "ARTICLES_NORMALIZED.parquet")
    queue = build_queue(candidates, causal, events, articles)
    write_parquet(QUEUE, queue)

    source_freeze_after = sha256_file(V1 / "SOURCE_FREEZE.json")
    if source_freeze_before != source_freeze_after:
        raise RuntimeError("BASE V1 source freeze changed while creating V2 queue")

    bucket_counts = {str(key): int(value) for key, value in queue["priority_bucket"].value_counts().items()}
    market_counts = {str(key): int(value) for key, value in queue["market"].value_counts().items()}
    summary = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_QUEUE_SUMMARY_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "QUEUE_READY_COLLECTION_NOT_STARTED",
        "outcome_blind": True,
        "append_only_v2": True,
        "base_v1_source_freeze_unchanged": True,
        "base_v1_source_freeze_sha256": source_freeze_after,
        "candidate_split_sha256": sha256_file(CANDIDATE_SPLIT),
        "candidate_rows": int(len(candidates)),
        "queue_rows": int(len(queue)),
        "priority_bucket_counts": bucket_counts,
        "market_counts": market_counts,
        "already_multisource_candidates_excluded": int(len(candidates) - len(queue)),
        "same_url_retry_allowed": False,
        "collection_network_requests": 0,
        "queue_sha256": sha256_file(QUEUE),
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
    }
    write_json(SUMMARY, summary)

    manifest = pd.DataFrame(
        [
            {"file": QUEUE.name, "sha256": sha256_file(QUEUE), "bytes": QUEUE.stat().st_size},
            {"file": SUMMARY.name, "sha256": sha256_file(SUMMARY), "bytes": SUMMARY.stat().st_size},
        ]
    )
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    write_json(
        STATE,
        {
            **summary,
            "queue_path": str(QUEUE),
            "manifest_path": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
