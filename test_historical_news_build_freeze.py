from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_news_embedding_cache import MODEL_ID, text_sha256

from historical_news_build_freeze import (
    BuildPaths,
    FreezeError,
    _map_label_authority_to_frozen_events,
    _ticker_comparison_key,
    build_freeze,
    build_preview,
    join_dev_labels_after_freeze,
    normalize_articles,
    normalize_events,
)


def _event_rows() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2024-01-02T15:00:00Z",
            "2024-03-02T15:00:00Z",
            "2024-05-02T15:00:00Z",
            "2024-07-02T15:00:00Z",
            "2024-09-02T15:00:00Z",
            "2024-11-02T15:00:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "canonical_event_id": [f"E{i}" for i in range(1, 7)],
            "event_group_id": [f"G{i}" for i in range(1, 7)],
            "existing_event_id": ["RAW_COLLISION", "RAW_COLLISION", "RAW3", "RAW4", "RAW5", "RAW6"],
            "join_key_event_id": ["RAW_COLLISION", "RAW_COLLISION", "RAW3", "RAW4", "RAW5", "RAW6"],
            "market": ["US", "US", "KR", "KR", "US", "US"],
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "issuer_name": ["Issuer"] * 6,
            "event_time_utc": times,
            "source_family": ["US_NEWS", "US_NEWS", "KR_NEWS", "KR_NEWS", "US_NEWS", "US_NEWS"],
            "exact_price_eligible": [True] * 6,
            "decision_cutoff_utc": times + pd.Timedelta(minutes=2),
        }
    )


def _article_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_uid": "A1",
                "canonical_event_id": "RAW_COLLISION",
                "market": "US",
                "ticker": "AAA",
                "event_time_utc": "2024-01-02T15:00:00Z",
                "canonical_url": "https://reuters.com/a?utm_source=x",
                "publisher_raw": "REUTERS",
                "publisher_canonical": "Reuters",
                "source_provider": "local",
                "source_type": "WIRE",
                "published_at_utc": "2024-01-02T15:00:30Z",
                "published_at_source": "JSON-LD",
                "published_at_precision": "EXACT_SECOND",
                "published_at_confidence": "HIGH",
                "first_seen_at_utc": "2024-01-02T15:00:45Z",
                "first_seen_observed": True,
                "captured_at_utc": "2024-01-02T15:01:00Z",
                "headline": "FDA approves phase 3 therapy",
                "body": "The phase 3 trial met its primary endpoint with statistically significant results. Reporting by Reuters with full clinical context.",
                "language": "en",
                "content_sha256": "1" * 64,
                "raw_capture_path": "raw/A1.json.gz",
                "raw_capture_sha256": "a" * 64,
                "raw_capture_present": True,
            },
            {
                "article_uid": "A2",
                "canonical_event_id": "E1",
                "market": "US",
                "ticker": "AAA",
                "canonical_url": "https://finance.yahoo.com/a",
                "publisher_raw": "Yahoo Finance",
                "origin_publisher": "Reuters",
                "source_provider": "local",
                "source_type": "AGGREGATOR",
                "published_at_utc": "2024-01-02T15:00:40Z",
                "published_at_source": "RSS",
                "published_at_precision": "EXACT_SECOND",
                "published_at_confidence": "HIGH",
                "first_seen_at_utc": None,
                "first_seen_observed": False,
                "captured_at_utc": "2024-01-02T15:01:30Z",
                "headline": "FDA approves phase 3 therapy",
                "body": "The phase 3 trial met its primary endpoint with statistically significant results. Reporting by Reuters with full clinical context.",
                "language": "en",
                "content_sha256": "1" * 64,
                "raw_capture_path": "raw/A2.json.gz",
                "raw_capture_sha256": "b" * 64,
                "raw_capture_present": True,
            },
            {
                "article_uid": "A3",
                "canonical_event_id": "E2",
                "market": "US",
                "ticker": "BBB",
                "canonical_url": "https://businesswire.com/b",
                "publisher_raw": "Business Wire",
                "source_provider": "local",
                "source_type": "ISSUER_PR",
                "published_at_utc": "2024-03-02T15:01:00Z",
                "published_at_source": "HTML_META",
                "published_at_precision": "EXACT_MINUTE",
                "published_at_confidence": "HIGH",
                "first_seen_at_utc": None,
                "first_seen_observed": False,
                "captured_at_utc": "2024-03-02T15:01:30Z",
                "headline": "Company announces licensing partnership",
                "body": "An exclusive license agreement and strategic collaboration includes development, regulatory, and commercial milestone obligations.",
                "language": "en",
                "content_sha256": "3" * 64,
                "raw_capture_path": "raw/A3.json.gz",
                "raw_capture_sha256": "c" * 64,
                "raw_capture_present": True,
            },
            {
                "article_uid": "A4",
                "canonical_event_id": "E3",
                "market": "KR",
                "ticker": "CCC",
                "canonical_url": "https://news.example.kr/c",
                "publisher_raw": "Example Korea",
                "source_provider": "local",
                "source_type": "NEWS",
                "published_at_utc": "2024-05-02T15:03:00Z",
                "published_at_source": "PROVIDER_API",
                "published_at_precision": "EXACT_SECOND",
                "published_at_confidence": "HIGH",
                "first_seen_at_utc": None,
                "first_seen_observed": False,
                "captured_at_utc": "2024-05-02T15:04:00Z",
                "headline": "Trial positive results",
                "body": "Positive topline results were announced.",
                "language": "ko",
                "content_sha256": "4" * 64,
                "raw_capture_path": "raw/A4.json.gz",
                "raw_capture_sha256": "d" * 64,
                "raw_capture_present": True,
            },
            {
                "article_uid": "A5",
                "canonical_event_id": "E4",
                "market": "KR",
                "ticker": "DDD",
                "canonical_url": "https://news.example.kr/d",
                "publisher_raw": "Example Korea",
                "source_provider": "local",
                "source_type": "NEWS",
                "published_at_utc": "2024-07-02T15:01:00Z",
                "published_at_source": "PROVIDER_API",
                "published_at_precision": "EXACT_SECOND",
                "published_at_confidence": "HIGH",
                "first_seen_at_utc": "2024-07-02T15:03:00Z",
                "first_seen_observed": True,
                "captured_at_utc": "2024-07-02T15:03:00Z",
                "headline": "Offering announced",
                "body": "Registered direct offering announced.",
                "language": "ko",
                "content_sha256": "5" * 64,
                "raw_capture_path": "raw/A5.json.gz",
                "raw_capture_sha256": "e" * 64,
                "raw_capture_present": True,
            },
        ]
    )


def _fixture_paths(tmp_path: Path) -> BuildPaths:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "frozen"
    research_dir = tmp_path / "research"
    input_dir.mkdir()
    _event_rows().to_parquet(input_dir / "EVENT_MASTER.parquet", index=False)
    article_rows = _article_rows()
    article_rows.to_parquet(input_dir / "ARTICLES_NORMALIZED.parquet", index=False)
    embedding_dir = input_dir / "embeddings"
    shard_dir = embedding_dir / "shards"
    shard_dir.mkdir(parents=True)
    vectors = np.asarray(
        [[1.0, index + 1.0, 2.0, 3.0] for index in range(len(article_rows))],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.astype(np.float16)
    shard = shard_dir / "TEST.npy"
    with shard.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    index_rows = []
    for position, row in enumerate(article_rows.to_dict(orient="records")):
        index_rows.append(
            {
                "schema_version": "HIST_NEWS_MULTILINGUAL_EMBEDDING_V1",
                "article_uid": row["article_uid"],
                "text_sha256": text_sha256(row["headline"], row["body"]),
                "model_id": MODEL_ID,
                "model_revision": "test-revision",
                "dimension": 4,
                "dtype": "float16",
                "normalized": True,
                "shard_path": "shards/TEST.npy",
                "shard_index": position,
                "shard_sha256": shard_sha,
                "embedded_at_utc": "2026-08-30T00:00:00Z",
                "gpu_used": False,
            }
        )
    pd.DataFrame(index_rows).to_parquet(embedding_dir / "EMBEDDING_INDEX.parquet", index=False)
    (embedding_dir / "EMBEDDING_MODEL_FREEZE.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": "test-revision",
                "target_or_outcome_training": False,
                "outcome_fields_read": [],
                "dimension": 4,
                "dtype": "float16",
                "normalized": True,
            }
        ),
        encoding="utf-8",
    )
    return BuildPaths(input_dir, output_dir, research_dir)


def test_build_is_outcome_blind_causal_and_idempotent(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    freeze = build_freeze(paths)
    assert freeze["source_membership_frozen"] is True
    assert freeze["outcome_blind"] is True
    assert freeze["model_ready"] is False
    assert freeze["label_join_complete"] is False
    assert not (paths.output_dir / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet").exists()

    articles = pd.read_parquet(paths.output_dir / "ARTICLES_NORMALIZED.parquet")
    quality = pd.read_parquet(paths.output_dir / "ARTICLE_QUALITY_AUDIT.parquet").set_index("article_uid")
    causal = pd.read_parquet(paths.output_dir / "CAUSAL_EVENT_NEWS.parquet").set_index("canonical_event_id")
    assert quality.loc["A1", "tier"] == "A"
    assert quality.loc["A2", "tier"] == "B_FULL"
    assert "FIRST_SEEN_UNOBSERVED" in quality.loc["A2", "reason"]
    assert quality.loc["A4", "tier"] == "C"  # after event+2m
    assert quality.loc["A4", "causal_eligible"] == False
    assert quality.loc["A5", "tier"] == "C"  # first_seen after cutoff/current-crawl-like
    assert causal.loc["E1", "syndication_root_count_t2"] == 1
    assert causal.loc["E1", "tierA_article_count_t2"] == 1
    assert causal.loc["E1", "tierB_article_count_t2"] == 1
    assert causal.loc["E3", "tierA_article_count_t2"] + causal.loc["E3", "tierB_article_count_t2"] == 0
    assert articles.loc[articles["article_uid"] == "A4", "causal_t2_eligible"].item() == False

    first_manifest = (paths.output_dir / "SHA256_MANIFEST.csv").read_bytes()
    second = build_freeze(paths)
    assert second == freeze
    assert (paths.output_dir / "SHA256_MANIFEST.csv").read_bytes() == first_manifest


def test_split_is_chronological_group_and_syndication_safe(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    build_freeze(paths)
    split = pd.read_parquet(paths.output_dir / "HIST_NEWS_DEV_SPLIT_MANIFEST.parquet")
    articles = pd.read_parquet(paths.output_dir / "ARTICLES_NORMALIZED.parquet")
    assert split.groupby("event_group_id")["fold"].nunique().max() == 1
    joined = articles[["canonical_event_id", "syndication_root_id"]].merge(
        split[["canonical_event_id", "fold"]], on="canonical_event_id", how="inner"
    )
    assert joined.groupby("syndication_root_id")["fold"].nunique().max() == 1
    ranges = split.groupby("fold")["event_time"].agg(["min", "max"]).sort_index()
    assert all(ranges.iloc[i]["max"] <= ranges.iloc[i + 1]["min"] for i in range(len(ranges) - 1))


def test_outcome_like_input_fails_closed(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    articles = pd.read_parquet(paths.input_dir / "ARTICLES_NORMALIZED.parquet")
    articles["prediction_correctness"] = [1, 1, 0, 1, 0]
    articles.to_parquet(paths.input_dir / "ARTICLES_NORMALIZED.parquet", index=False)
    with pytest.raises(FreezeError, match="outcome-like columns"):
        build_freeze(paths)


def test_final_freeze_requires_complete_embedding_cache_but_preview_reports_gap(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    index_path = paths.input_dir / "embeddings/EMBEDDING_INDEX.parquet"
    partial = pd.read_parquet(index_path).iloc[:-1].copy()
    partial.to_parquet(index_path, index=False)
    preview = build_preview(paths.input_dir, tmp_path / "preview")
    assert preview["embedding"]["complete"] is False
    assert preview["embedding"]["missing_articles"] == 1
    with pytest.raises(FreezeError, match="embedding cache incomplete"):
        build_freeze(paths)


def test_wrong_cutoff_fails_closed() -> None:
    events = _event_rows()
    events.loc[0, "decision_cutoff_utc"] = events.loc[0, "event_time_utc"] + pd.Timedelta(minutes=3)
    with pytest.raises(FreezeError, match="decision cutoff"):
        normalize_events(events)


def test_manifest_hashes_every_frozen_artifact(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    build_freeze(paths)
    manifest = pd.read_csv(paths.output_dir / "SHA256_MANIFEST.csv")
    names = set(manifest["file"])
    assert "SOURCE_FREEZE.json" in names
    assert "HIST_HIGH_CONF_NEWS_FEATURE_MATRIX.parquet" in names
    assert "FEATURE_CATALOG.csv" in names
    assert "TOPIC_QA_SAMPLE.csv" in names
    assert "ARTICLE_EMBEDDINGS_MANIFEST.json" in names
    assert "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet" not in names
    frozen = json.loads((paths.output_dir / "SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    assert frozen["freeze_gates"]["NO_FUTURE_ARTICLE_IN_CAUSAL_FEATURES"] is True
    assert frozen["freeze_gates"]["NO_RESEARCH_SEAL_ROWS"] is True


def test_explicit_post_freeze_label_join_preserves_membership(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    build_freeze(paths)
    freeze_path = paths.output_dir / "SOURCE_FREEZE.json"
    freeze_before = freeze_path.read_bytes()
    labels_path = tmp_path / "dev_labels.parquet"
    pd.DataFrame(
        {
            "canonical_event_id": ["E1", "E3"],
            "y": [1, 0],
            "fwd_ret_30m": [0.1, -0.2],
        }
    ).to_parquet(labels_path, index=False)
    status = join_dev_labels_after_freeze(paths, labels_path)
    labeled = pd.read_parquet(paths.output_dir / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet")
    matrix = pd.read_parquet(paths.output_dir / "HIST_HIGH_CONF_NEWS_FEATURE_MATRIX.parquet")
    assert labeled["canonical_event_id"].tolist() == matrix["canonical_event_id"].tolist()
    assert len(labeled) == len(matrix)
    assert status["label_matched_rows"] == 2
    assert status["source_freeze_unchanged"] is True
    assert freeze_path.read_bytes() == freeze_before


def test_post_freeze_label_join_discards_source_fields(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    build_freeze(paths)
    labels_path = tmp_path / "bad_labels.parquet"
    pd.DataFrame(
        {"canonical_event_id": ["E1"], "y": [1], "publisher": ["drop this source"]}
    ).to_parquet(labels_path, index=False)
    status = join_dev_labels_after_freeze(paths, labels_path)
    labeled = pd.read_parquet(paths.output_dir / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet")
    assert "publisher" not in labeled.columns
    assert "publisher" in status["ignored_label_columns"]


def test_raw_event_id_collision_resolves_only_by_market_ticker_time(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    build_freeze(paths)
    labels_path = tmp_path / "labels.csv.gz"
    pd.DataFrame(
        {
            "event_id": ["RAW_COLLISION"],
            "market": ["US"],
            "ticker": ["BBB"],
            "event_time_utc": ["2024-03-02T15:00:00Z"],
            "y": [0],
            "fwd_ret_30m": [-0.05],
            "headline": ["must never be joined"],
        }
    ).to_csv(labels_path, index=False, compression="gzip")
    status = join_dev_labels_after_freeze(paths, labels_path)
    labeled = pd.read_parquet(paths.output_dir / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet").set_index("canonical_event_id")
    assert labeled.loc["E2", "y"] == 0
    assert pd.isna(labeled.loc["E1", "y"])
    assert status["collision_rows_resolved_by_keys"] == 1
    assert "headline" not in labeled.columns


def test_kr_ticker_comparison_preserves_leading_zero_identity() -> None:
    assert _ticker_comparison_key("001060", "KR") == "001060"
    assert _ticker_comparison_key(1060, "KR") == "001060"
    assert _ticker_comparison_key("AAPL", "US") == "AAPL"


def test_label_collision_prefers_unique_exact_price_eligible_candidate() -> None:
    events = _event_rows().iloc[:2].copy()
    events.loc[1, ["market", "ticker", "event_time_utc"]] = events.loc[0, ["market", "ticker", "event_time_utc"]]
    events["exact_price_eligible"] = [True, False]
    labels = pd.DataFrame(
        {
            "event_id": ["RAW_COLLISION"],
            "market": ["US"],
            "ticker": ["AAA"],
            "event_time_utc": [events.loc[0, "event_time_utc"]],
            "y": [1],
        }
    )
    labels.attrs["authority_key"] = "event_id"
    labels.attrs["ignored_columns"] = []
    mapped, audit = _map_label_authority_to_frozen_events(labels, events)
    assert mapped.iloc[0]["canonical_event_id"] == "E1"
    assert audit["collision_rows_resolved_by_exact_eligibility"] == 1


def test_ambiguous_market_ticker_event_mapping_is_quarantined() -> None:
    events = _event_rows()
    duplicate_time = events.iloc[[0]].copy()
    duplicate_time["canonical_event_id"] = "E7"
    duplicate_time["event_group_id"] = "G7"
    duplicate_time["existing_event_id"] = "RAW7"
    duplicate_time["join_key_event_id"] = "RAW7"
    events = normalize_events(pd.concat([events, duplicate_time], ignore_index=True))
    article = _article_rows().iloc[[0]].copy()
    article["article_uid"] = "AMBIGUOUS"
    article["canonical_event_id"] = "NO_EXPLICIT_MATCH"
    normalized = normalize_articles(article, events).iloc[0]
    assert normalized["canonical_event_id"] == ""
    assert normalized["event_link_method"] == "AMBIGUOUS_EVENT_MAPPING"
    assert normalized["event_match_quality"] == "LOW"
    assert normalized["event_mapping_quarantine_reason"] == "MARKET_TICKER_SOURCE_EVENT_TIME_NOT_UNIQUE"


def test_publication_after_capture_is_preserved_but_timestamp_quarantined() -> None:
    events = normalize_events(_event_rows())
    article = _article_rows().iloc[[2]].copy()
    article["article_uid"] = "BAD_PROVIDER_DATE"
    article["published_at_utc"] = "2027-01-04T00:00:00Z"
    article["captured_at_utc"] = "2026-08-30T12:02:00Z"
    normalized = normalize_articles(article, events).iloc[0]
    assert normalized["article_uid"] == "BAD_PROVIDER_DATE"
    assert normalized["published_at_utc_source_value"] == "2027-01-04T00:00:00Z"
    assert pd.isna(normalized["published_at_utc"])
    assert normalized["published_at_invalid_future_vs_capture"] == True
    assert normalized["timestamp_quarantine_reason"] == "PUBLICATION_AFTER_CAPTURE"
    assert normalized["published_at_precision"] == "UNKNOWN"


def test_preview_never_claims_or_creates_source_freeze(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    preview_dir = tmp_path / "derived_preview"
    status = build_preview(paths.input_dir, preview_dir)
    assert status["status"] == "DERIVED_PREVIEW_NOT_FROZEN"
    assert status["source_frozen"] is False
    assert status["model_ready"] is False
    assert status["qa_pass"] is True
    assert (preview_dir / "PREVIEW_STATUS.json").is_file()
    assert (preview_dir / "PREVIEW_ARTICLE_QUALITY.parquet").is_file()
    assert (preview_dir / "PREVIEW_CAUSAL_COUNTS.parquet").is_file()
    assert not paths.output_dir.exists()
