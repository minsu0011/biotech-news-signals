import pandas as pd

from historical_news_multisource_v2_source_graph import (
    build_event_time_manifest,
    char5_containment,
    entropy,
    independence_root_map,
    shares_base_information_root,
    stable_id,
)


def test_stable_id_is_deterministic_and_namespaced() -> None:
    assert stable_id("NODE_", "event", "article") == stable_id("NODE_", "event", "article")
    assert stable_id("NODE_", "event", "article").startswith("NODE_")
    assert stable_id("NODE_", "event", "article") != stable_id("NODE_", "event", "other")


def test_entropy_is_zero_for_one_publisher_and_positive_for_two() -> None:
    assert entropy(["Reuters", "Reuters"]) == 0.0
    assert entropy(["Reuters", "Bloomberg"]) > 0.0


def test_event_time_uses_deterministic_official_identifier_before_media() -> None:
    events = pd.DataFrame([{
        "canonical_event_id": "E1", "market": "US", "ticker": "ABC",
        "event_time_utc": pd.Timestamp("2026-01-01T12:00:00Z"),
    }])
    recovered = pd.DataFrame([
        {
            "canonical_event_id": "E1", "event_match_high": True,
            "timestamp_precision": "EXACT_SECOND", "timestamp_confidence": "MEDIUM_DETERMINISTIC_IDENTIFIER",
            "source_root_type": "OFFICIAL_ROOT", "published_at_utc": pd.Timestamp("2026-01-01T10:00:00Z"),
            "canonical_url": "https://example.com/medium",
            "official_root_accepted": True, "strict_independent_media_root_accepted": False,
        },
        {
            "canonical_event_id": "E1", "event_match_high": True,
            "timestamp_precision": "EXACT_SECOND", "timestamp_confidence": "HIGH",
            "source_root_type": "MEDIA_REPORTING_ROOT", "published_at_utc": pd.Timestamp("2026-01-01T11:00:00Z"),
            "canonical_url": "https://example.org/high",
            "official_root_accepted": False, "strict_independent_media_root_accepted": True,
        },
    ])
    result = build_event_time_manifest(events, recovered).iloc[0]
    assert result["v2_resolved_event_time_utc"] == pd.Timestamp("2026-01-01T10:00:00Z")
    assert result["resolution_method"] == "EARLIEST_HIGH_CONFIDENCE_OFFICIAL_DISCLOSURE"


def test_event_time_preserves_base_for_date_only_source() -> None:
    events = pd.DataFrame([{
        "canonical_event_id": "E1", "market": "KR", "ticker": "001000",
        "event_time_utc": pd.Timestamp("2026-01-01T12:00:00Z"),
    }])
    recovered = pd.DataFrame([{
        "canonical_event_id": "E1", "event_match_high": True,
        "timestamp_precision": "DATE_ONLY", "timestamp_confidence": "HIGH",
        "source_root_type": "OFFICIAL_ROOT", "published_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
        "canonical_url": "https://example.com/date-only",
        "official_root_accepted": True, "strict_independent_media_root_accepted": False,
    }])
    result = build_event_time_manifest(events, recovered).iloc[0]
    assert not result["event_time_changed"]
    assert result["resolution_method"] == "PRESERVE_BASE_V1_EVENT_TIME"


def test_pairwise_near_duplicates_share_one_independence_root() -> None:
    event_times = pd.DataFrame([{
        "canonical_event_id": "E1", "v2_decision_cutoff_utc": pd.Timestamp("2026-01-01T12:02:00Z")
    }])
    common = "Company announced phase three results with detailed efficacy and safety findings. " * 5
    recovered = pd.DataFrame([
        {
            "canonical_event_id": "E1", "strict_independent_media_root_accepted": True,
            "published_at_utc": pd.Timestamp("2026-01-01T12:00:00Z"), "timestamp_precision": "EXACT_SECOND",
            "body": common, "raw_capture_sha256": "A",
        },
        {
            "canonical_event_id": "E1", "strict_independent_media_root_accepted": True,
            "published_at_utc": pd.Timestamp("2026-01-01T12:01:00Z"), "timestamp_precision": "EXACT_SECOND",
            "body": common + " Syndicated copy.", "raw_capture_sha256": "B",
        },
    ])
    mapping = independence_root_map(recovered, event_times)
    assert mapping[0][0] == mapping[1][0]
    assert mapping[0][1] and mapping[1][1]


def test_char5_containment_catches_short_copy_inside_long_article() -> None:
    copied = "The company reported revenue growth of 45 percent and improved cost efficiency. " * 4
    long_article = copied + ("Additional background about unrelated prior projects and market history. " * 20)
    assert char5_containment(copied, long_article) >= 0.55


def test_short_base_digest_near_simultaneous_article_cannot_prove_independence() -> None:
    base = (
        "Other item one Other item two Food safety ministry sends a halal support "
        "delegation to Indonesia..."
    )
    candidate = (
        "The food safety ministry said it formed a public private delegation to help "
        "companies respond to Indonesia's halal labeling requirement. " * 12
    )
    shared, reason = shares_base_information_root(base, candidate, 0.74, 120)
    assert shared
    assert reason == "BASE_SHORT_EXCERPT_NEAR_SIMULTANEOUS_INDEPENDENCE_UNPROVEN"


def test_long_distinct_base_article_is_not_merged_only_for_close_timing() -> None:
    base = "Independent reporting with interviews and trial context. " * 20
    candidate = "Separate reporting based on a different analyst investigation. " * 20
    shared, reason = shares_base_information_root(base, candidate, 0.74, 120)
    assert not shared
    assert reason == ""
