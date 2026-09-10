from types import SimpleNamespace

import pandas as pd

from historical_news_multisource_v2_body_recovery import (
    char_ngrams,
    domain_is,
    english_distinctive_anchor_consistent,
    recovery_status,
    nonarticle_body_reason,
    attribution_anchor_consistent,
    choose_timestamp,
    resolve_publisher,
    salient_token_overlap,
    text_similarity,
)


def test_text_similarity_distinguishes_exact_from_unrelated() -> None:
    exact = text_similarity("GC녹십자 헌터라제 품목허가", "GC녹십자 헌터라제 품목허가")
    unrelated = text_similarity("GC녹십자 헌터라제 품목허가", "미국 금리와 반도체 전망")
    assert exact == (1.0, 1.0)
    assert unrelated[0] < 0.3
    assert unrelated[1] == 0.0


def test_domain_resolution_is_direct_and_aggregator_safe() -> None:
    assert domain_is("www.newspim.com", "newspim.com")
    assert not domain_is("newspim.com.example.org", "newspim.com")
    publisher, publisher_class, method = resolve_publisher(
        "https://www.newspim.com/news/view/1", "", "UNKNOWN", ""
    )
    assert (publisher, publisher_class, method) == ("뉴스핌", "MAJOR_FINANCIAL", "DIRECT_DOMAIN_REGISTRY")


def test_salient_token_overlap_matches_official_body() -> None:
    count, ratio = salient_token_overlap(
        "JW중외제약, 2주 1회 비만약 보팡글루타이드 국내 3상 신청",
        "JW중외제약은 비만 치료 신약후보물질 보팡글루타이드의 임상 3상 시험계획을 신청했다. 2주 1회 투여한다.",
    )
    assert count >= 4
    assert ratio >= 0.5


def test_recovery_status_keeps_post_t2_auxiliary() -> None:
    row = {
        "fetch_succeeded": True,
        "event_match_high": True,
        "body_length": 500,
        "origin_distinct_from_current": True,
        "published_by_t2": False,
        "source_root_type": "MEDIA_REPORTING_ROOT",
        "likely_same_information_root": False,
    }
    assert recovery_status(row) == "RETROSPECTIVE_AUX_POST_T2"


def test_recovery_status_rejects_syndication_as_independent() -> None:
    row = {
        "fetch_succeeded": True,
        "event_match_high": True,
        "body_length": 500,
        "origin_distinct_from_current": True,
        "published_by_t2": True,
        "source_root_type": "MEDIA_REPORTING_ROOT",
        "likely_same_information_root": True,
    }
    assert recovery_status(row) == "CAUSAL_DISTINCT_PUBLISHER_SYNDICATION_ROOT"


def test_recovery_status_rejects_rss_only_timestamp_as_causal_root() -> None:
    row = {
        "fetch_succeeded": True,
        "event_match_high": True,
        "body_length": 500,
        "origin_distinct_from_current": True,
        "published_by_t2": True,
        "timestamp_acceptable_for_root": False,
        "source_root_type": "MEDIA_REPORTING_ROOT",
        "likely_same_information_root": False,
    }
    assert recovery_status(row) == "TIMESTAMP_CONFIDENCE_INSUFFICIENT"


def test_recovery_status_rejects_headline_only_page_shell() -> None:
    row = {
        "fetch_succeeded": True,
        "event_match_high": True,
        "body_event_match_high": False,
        "body_length": 800,
        "origin_distinct_from_current": True,
        "published_by_t2": True,
        "timestamp_acceptable_for_root": True,
        "source_root_type": "MEDIA_REPORTING_ROOT",
        "likely_same_information_root": False,
    }
    assert recovery_status(row) == "BODY_EVENT_MATCH_REJECTED"


def test_short_repeated_photo_caption_is_not_an_independent_article() -> None:
    caption = (
        "[Agency] The company said second-quarter operating income turned profitable. "
        "The image is a short supplied caption only. (Photo=Company provided) Redistribution prohibited. "
    )
    body = caption + caption
    assert nonarticle_body_reason(body) == "SHORT_REPEATED_CAPTION"
    row = {
        "fetch_succeeded": True, "event_match_high": True, "body_event_match_high": True,
        "body_content_usable": False, "body_length": len(body), "origin_distinct_from_current": True,
        "published_by_t2": True, "timestamp_acceptable_for_root": True,
        "source_root_type": "MEDIA_REPORTING_ROOT", "likely_same_information_root": False,
    }
    assert recovery_status(row) == "BODY_CONTENT_NOT_ARTICLE"


def test_broker_research_attribution_anchor_must_match() -> None:
    headline = '한투證 "기업, 반도체·태양광 투자 확대 수혜"'
    assert attribution_anchor_consistent(headline, "한국투자증권 연구원은 성장 전망을 제시했다")[0]
    consistent, missing = attribution_anchor_consistent(headline, "그로쓰리서치 연구원은 성장 전망을 제시했다")
    assert not consistent
    assert missing
    assert missing == "한투"


def test_gdelt_seen_time_is_never_promoted_to_origin_page_confidence() -> None:
    candidate = SimpleNamespace(
        discovery_timestamp_utc=pd.Timestamp("2021-02-21T19:01:00Z"),
        discovery_timestamp_provenance="GDELT_GAL_SEEN_TIME_DISCOVERY_ONLY",
    )
    stamp, provenance, precision, confidence = choose_timestamp({}, candidate)
    assert stamp == pd.Timestamp("2021-02-21T19:01:00Z")
    assert provenance == "GDELT_GAL_SEEN_TIME_DISCOVERY_ONLY"
    assert precision == "EXACT_SECOND"
    assert confidence == "LOW_GDELT_SEEN_TIME_NOT_PUBLICATION"


def test_english_named_event_anchor_rejects_broad_covid_topic_match() -> None:
    reference = "Japan begins COVID-19 vaccination in first major step to halt pandemic"
    wrong = "Michigan hospital cancels vaccinations after a Pfizer vaccine supply shortage"
    consistent, missing = english_distinctive_anchor_consistent(reference, wrong)
    assert not consistent
    assert "japan" in missing


def test_english_named_event_anchor_accepts_same_event_rephrasing() -> None:
    reference = "Japan begins COVID-19 vaccination in first major step to halt pandemic"
    same = "Japan launched its coronavirus immunization campaign at hospitals on Wednesday."
    assert english_distinctive_anchor_consistent(reference, same)[0]


def test_multiple_named_anchors_reject_related_but_different_country_event() -> None:
    reference = "Germany sees rapid EU approval of AstraZeneca's COVID-19 vaccine"
    wrong = "UK approves AstraZeneca Oxford Covid-19 vaccine for emergency use"
    consistent, missing = english_distinctive_anchor_consistent(reference, wrong)
    assert not consistent
    assert "germany" in missing and "eu" in missing


def test_multiple_named_anchors_accept_matching_entities_in_article_lead() -> None:
    reference = "Germany sees rapid EU approval of AstraZeneca's COVID-19 vaccine"
    lead = "AstraZeneca said Germany expects EU regulators to review the vaccine rapidly."
    assert english_distinctive_anchor_consistent(reference, lead)[0]
