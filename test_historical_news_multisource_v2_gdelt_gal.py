import gzip
import json

from historical_news_multisource_v2_gdelt_gal import (
    bucket_name,
    gal_bucket_times,
    gal_item_is_semantic_candidate,
    parse_gal_payload,
)


def test_gal_buckets_follow_quarter_hour_plus_one_cadence() -> None:
    buckets = gal_bucket_times("2021-02-21T18:56:24Z", before_minutes=15, after_minutes=20)
    assert [bucket_name(value) for value in buckets] == [
        "20210221184600.gal.json.gz",
        "20210221190100.gal.json.gz",
        "20210221191600.gal.json.gz",
    ]


def test_parse_gal_json_lines_preserves_url_title_and_seen_time() -> None:
    rows = [
        {"date": "2021-02-21T19:01:00.000Z", "url": "https://example.com/a", "title": "Trial result"},
        {"date": "2021-02-21T19:01:00.000Z", "url": "", "title": "Invalid"},
    ]
    payload = gzip.compress("\n".join(json.dumps(row) for row in rows).encode("utf-8"))
    parsed = parse_gal_payload(payload)
    assert len(parsed) == 1
    assert parsed[0]["url"] == "https://example.com/a"


def test_gal_candidate_filter_requires_semantic_headline_match() -> None:
    accepted, sequence, jaccard = gal_item_is_semantic_candidate(
        "Company reports positive phase 3 trial results",
        "Company reports positive phase 3 trial results",
    )
    assert accepted and sequence == 1.0 and jaccard == 1.0
    assert not gal_item_is_semantic_candidate(
        "Company reports positive phase 3 trial results",
        "Central bank changes interest rates",
    )[0]


def test_gal_candidate_filter_rejects_broad_topic_without_named_anchor() -> None:
    assert not gal_item_is_semantic_candidate(
        "Japan begins COVID-19 vaccination in first major step to halt pandemic",
        "Michigan hospital cancels Covid vaccinations due to major shortage",
        "Pfizer vaccine supply was reduced this week.",
    )[0]
