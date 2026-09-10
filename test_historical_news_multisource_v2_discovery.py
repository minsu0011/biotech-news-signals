from historical_news_multisource_v2_discovery import (
    direct_url_from_rss,
    inferred_origin,
    parse_rss,
    provider_url,
    query_text,
)


def test_query_is_metadata_only() -> None:
    value = query_text("Company reports positive phase 3 results", "ABCD", "US")
    assert "ABCD" in value
    assert "return" not in value.lower()
    assert "correct" not in value.lower()
    exact = query_text("Company reports positive phase 3 results", "ABCD", "US", "EXACT_HEADLINE_ONLY_V2")
    assert "ABCD" not in exact
    assert exact.startswith('"') and exact.endswith('"')
    entity = query_text(
        "Company reports positive phase 3 results", "ABCD", "US", "ENTITY_TERMS_DATE_V3",
        "2026-08-24T14:00:00Z",
    )
    assert '"' not in entity
    assert "ABCD" in entity
    assert "2026-08-24" in entity
    official = query_text(
        "투자판단 관련 주요경영사항 CTP55 미국 품목허가 신청", "068270", "KR",
        "KIND_OFFICIAL_TERMS_DATE_V4", "2026-08-24T00:00:00Z",
    )
    assert "CTP55" in official
    assert "068270" in official
    assert "2026-08-24" in official
    specific = query_text(
        "투자판단 관련 주요경영사항(CTP55 코센틱스 바이오시밀러 미국 품목허가 신청)",
        "068270", "KR", "KIND_OFFICIAL_SPECIFIC_V5", "2026-08-24T00:00:00Z",
    )
    assert specific.startswith("CTP55")
    assert "투자판단" not in specific


def test_direct_domain_can_resolve_origin_without_rss_source() -> None:
    origin, method = inferred_origin("UNKNOWN", "https://www.newstap.co.kr/news/articleView.html?idxno=1")
    assert origin == "뉴스탭"
    assert method == "DIRECT_URL_DOMAIN_REGISTRY"


def test_provider_urls_are_public_rss() -> None:
    google = provider_url("GOOGLE_NEWS_PUBLIC_RSS", '"headline" ABCD', "US")
    bing = provider_url("BING_NEWS_PUBLIC_RSS", '"headline" ABCD', "US")
    assert google.startswith("https://news.google.com/rss/search?")
    assert "format=rss" in bing


def test_parse_rss_preserves_source_and_second_timestamp() -> None:
    payload = b"""<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel><item><title>Example headline - Reuters</title>
    <link>https://example.com/a</link><pubDate>Mon, 18 Aug 2026 00:01:02 GMT</pubDate>
    <source>Reuters</source></item></channel></rss>"""
    rows = parse_rss(payload)
    assert len(rows) == 1
    assert rows[0]["headline"] == "Example headline"
    assert rows[0]["publisher_rss"] == "Reuters"
    assert rows[0]["published_at_utc"].endswith("+00:00")


def test_bing_redirect_can_expose_direct_public_url() -> None:
    link = "https://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fexample.com%2Farticle"
    assert direct_url_from_rss(link) == "https://example.com/article"
