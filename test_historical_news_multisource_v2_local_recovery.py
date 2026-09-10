import pandas as pd

from historical_news_multisource_v2_local_recovery import (
    classify_relation,
    lexical_similarity,
    naver_oid,
    resolve_current_origin,
)


def test_naver_host_is_separate_from_origin() -> None:
    row = pd.Series(
        {
            "origin_publisher": "Naver News",
            "hosting_publisher": "Naver News",
            "canonical_url": "https://n.news.naver.com/mnews/article/008/0005401105",
            "canonical_event_id": "NAVER:041920:0080005401105::x",
        }
    )
    resolved = resolve_current_origin(row)
    assert naver_oid(row["canonical_url"]) == "008"
    assert resolved["origin_publisher_resolved"] == "머니투데이"
    assert resolved["publisher_class_resolved"] == "MAJOR_FINANCIAL"
    assert resolved["origin_resolution_confidence"] == "HIGH"


def test_same_origin_and_headline_only_are_not_independent() -> None:
    relation, accepted = classify_relation("머니투데이", "머니투데이", 500, 0.7, 0.4)
    assert relation == "SAME_ORIGIN_MIRROR"
    assert accepted is False
    relation, accepted = classify_relation("뉴스1", "아시아경제", 5, 0.75, 0.5)
    assert relation == "HEADLINE_ONLY_DISTINCT_MEDIA_CANDIDATE"
    assert accepted is False


def test_body_supported_distinct_origin_can_be_candidate_root() -> None:
    relation, accepted = classify_relation("뉴스1", "아시아경제", 500, 0.75, 0.5)
    assert relation == "STRICT_DISTINCT_MEDIA_ROOT"
    assert accepted is True


def test_lexical_similarity_separates_matching_and_unrelated_headlines() -> None:
    match = lexical_similarity(
        "JW중외제약 2주 1회 비만약 국내 3상 신청",
        "JW중외제약, 2주 1회 비만 치료제 국내 임상 3상 신청",
    )
    unrelated = lexical_similarity(
        "JW중외제약 2주 1회 비만약 국내 3상 신청",
        "유한양행 공식몰 할인 판매 기획전",
    )
    assert match[0] > unrelated[0]
    assert match[1] > unrelated[1]
