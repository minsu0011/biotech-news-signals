import pandas as pd

from historical_news_multisource_v2_queue import build_queue


def test_build_queue_is_outcome_blind_and_excludes_existing_multisource() -> None:
    candidates = pd.DataFrame(
        [
            {
                "canonical_event_id": "e1",
                "model_event_group_id": "g1",
                "market": "US",
                "ticker": "AAA",
                "source_family": "US_NEWS",
                "event_time_utc": "2025-01-01T12:00:00Z",
                "candidate_contract": "EXACT_PRICE_ELIGIBLE_AND_CAUSAL_TIER_A_OR_B_FULL",
            },
            {
                "canonical_event_id": "e2",
                "model_event_group_id": "g2",
                "market": "KR",
                "ticker": "000001",
                "source_family": "KR_NEWS",
                "event_time_utc": "2025-01-02T12:00:00Z",
                "candidate_contract": "EXACT_PRICE_ELIGIBLE_AND_CAUSAL_TIER_A_OR_B_FULL",
            },
        ]
    )
    causal = pd.DataFrame(
        [
            {
                "canonical_event_id": "e1",
                "independent_source_count_t2": 1,
                "unique_publisher_count_t2": 1,
                "syndication_root_count_t2": 1,
                "structured_confirmation_count_t2": 0,
            },
            {
                "canonical_event_id": "e2",
                "independent_source_count_t2": 2,
                "unique_publisher_count_t2": 2,
                "syndication_root_count_t2": 2,
                "structured_confirmation_count_t2": 0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {"canonical_event_id": "e1", "issuer_name": "A", "event_type": "NEWS", "source_provider": "P1"},
            {"canonical_event_id": "e2", "issuer_name": "B", "event_type": "NEWS", "source_provider": "P2"},
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "canonical_event_id": "e1",
                "article_uid": "a1",
                "causal_t2_eligible": True,
                "origin_publisher": "Wire A",
                "source_provider": "Provider A",
                "source_type": "WIRE",
            },
            {
                "canonical_event_id": "e2",
                "article_uid": "a2",
                "causal_t2_eligible": True,
                "origin_publisher": "Wire B",
                "source_provider": "Provider B",
                "source_type": "WIRE",
            },
        ]
    )

    queue = build_queue(candidates, causal, events, articles)

    assert queue["canonical_event_id"].tolist() == ["e1"]
    assert queue.loc[0, "priority_bucket"] == "P0_ADD_SECOND_INDEPENDENT_SOURCE"
    assert bool(queue.loc[0, "outcome_blind"])
    assert not bool(queue.loc[0, "same_url_retry_allowed"])
    assert queue.loc[0, "additional_independent_sources_needed"] == 1
    assert not any(column in queue for column in ("y", "return", "correct_target", "signed_net"))
