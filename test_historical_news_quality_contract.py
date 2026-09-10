from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from historical_news_quality_contract import (
    CONTRACT_VERSION,
    TIER_A,
    TIER_B_FULL,
    TIER_B_HEADLINE,
    TIER_C,
    classify_historical_article_quality,
    publisher_is_known,
)


EVENT = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
CUTOFF = EVENT + timedelta(minutes=2)
PUBLISHED = EVENT + timedelta(seconds=30)
BODY = "Substantive historical article body with enough factual detail for the strict BASE V1 full-text contract."


def classify(**overrides):
    values = {
        "canonical_publisher": "Reuters",
        "published_at_precision": "EXACT_SECOND",
        "published_at_confidence": "HIGH",
        "published_at": PUBLISHED,
        "decision_cutoff": CUTOFF,
        "event_match_quality": "HIGH",
        "headline": "Company announces material Phase 3 clinical result",
        "body": BODY,
        "first_seen_at": None,
        "first_seen_observed": False,
        "captured_at": EVENT + timedelta(days=30),
        "provenance_available": True,
    }
    values.update(overrides)
    return classify_historical_article_quality(**values)


@pytest.mark.parametrize("publisher", ["", "Unknown", "UNKNOWN", "unknown", "N/A", "NA", "None", None])
def test_unknown_publisher_never_strict_or_full(publisher) -> None:
    assert publisher_is_known(publisher) is False
    decision = classify(canonical_publisher=publisher)
    assert decision.tier == TIER_C
    assert decision.causal_eligible is False


def test_medium_event_match_never_b_full() -> None:
    decision = classify(event_match_quality="MEDIUM")
    assert decision.tier == TIER_C
    assert decision.causal_eligible is False


def test_short_body_is_headline_only_aux() -> None:
    decision = classify(body="short body")
    assert decision.tier == TIER_B_HEADLINE
    assert decision.headline_only_aux is True
    assert decision.causal_eligible is False


def test_published_after_t2_never_causal() -> None:
    decision = classify(published_at=CUTOFF + timedelta(seconds=1))
    assert decision.tier == TIER_C
    assert decision.causal_eligible is False


def test_current_crawl_time_never_becomes_historical_first_seen() -> None:
    captured = EVENT + timedelta(days=30)
    decision = classify(
        first_seen_at=captured,
        first_seen_observed=True,
        captured_at=captured,
    )
    assert decision.tier == TIER_C
    assert decision.current_crawl_as_historical_first_seen is True


def test_precise_known_high_match_substantive_body_is_b_full() -> None:
    decision = classify()
    assert decision.contract_version == CONTRACT_VERSION
    assert decision.tier == TIER_B_FULL
    assert decision.causal_eligible is True
    assert decision.first_seen_unobserved is True


def test_genuine_historical_first_seen_is_tier_a() -> None:
    decision = classify(
        first_seen_at=EVENT + timedelta(seconds=45),
        first_seen_observed=True,
        captured_at=EVENT + timedelta(seconds=50),
        provenance_available=True,
    )
    assert decision.tier == TIER_A
    assert decision.causal_eligible is True
