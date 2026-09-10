from __future__ import annotations

import pytest

from historical_news_base_v1_pending_audit import classify_pending_fetch


@pytest.mark.parametrize("url", ["", None])
def test_missing_url_is_terminal(url) -> None:
    result = classify_pending_fetch(raw_url=url)
    assert result["category"] == "URL_NOT_AVAILABLE"
    assert result["retry_same_url"] is False


def test_google_aggregator_requires_alternate_provider() -> None:
    result = classify_pending_fetch(raw_url="https://news.google.com/rss/articles/abc", matched_quality_tier="B_HEADLINE")
    assert result["category"] == "NEEDS_ALTERNATE_PROVIDER"
    assert result["retry_same_url"] is False
    assert result["needs_alternate_provider"] is True


def test_direct_headline_only_article_is_body_recovery() -> None:
    result = classify_pending_fetch(raw_url="https://n.news.naver.com/article/1", matched_quality_tier="B_HEADLINE")
    assert result["category"] == "BODY_RECOVERY_ONLY"
    assert result["retry_same_url"] is True


def test_full_article_is_already_covered() -> None:
    result = classify_pending_fetch(raw_url="https://example.com/a", matched_quality_tier="B_FULL")
    assert result["category"] == "ALREADY_COVERED"
    assert result["terminal_for_same_url"] is True


def test_tier_c_is_low_value_aux_only() -> None:
    result = classify_pending_fetch(raw_url="https://example.com/a", matched_quality_tier="C")
    assert result["category"] == "LOW_VALUE_AUX_ONLY"
    assert result["retry_same_url"] is False
