from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import historical_news_article_backfill as mod


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(**overrides):
    value = {
        "event_id": "E1", "market": "US", "ticker": "ABCD", "company": "Acme Bio",
        "event_time_utc": "2025-01-02T14:30:10+00:00", "source": "REUTERS_QIU",
        "headline": "Acme Bio announces positive Phase 3 results",
        "body": "NEW YORK (Reuters) - " + ("The company reported trial results and safety details. " * 4),
        "timestamp_quality": "EXACT",
        "url": "https://www.reuters.com/world/acme/?utm_source=test&x=1",
        "article_id": "R1",
    }
    value.update(overrides)
    return value


def test_local_first_normalization_raw_hash_and_no_fake_first_seen(tmp_path: Path):
    root = tmp_path / "project"
    source = root / "data" / "events_us_news_real.csv"
    _write_csv(source, [_row()])
    out = root / "data" / "historical_news_backfill"
    engine = mod.BackfillEngine(root, out, [mod.SourceSpec("US_NEWS_REAL", source, "csv")])
    summary = engine.run_once(resume=False, max_requests=0, max_local_items=10)
    assert summary["requests_used"] == 0
    frame = pd.read_parquet(out / "ARTICLES_NORMALIZED.parquet")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["origin_publisher"] == "Reuters"
    assert row["canonical_url"] == "https://www.reuters.com/world/acme?x=1"
    assert row["published_at_precision"] == "EXACT_SECOND"
    assert row["published_at_confidence"] == "HIGH"
    assert row["first_seen_observed"] == False
    assert pd.isna(row["first_seen_at_utc"])
    assert row["tier"] == "B_FULL"
    raw_path = out / row["raw_capture_path"]
    with gzip.open(raw_path, "rb") as handle:
        assert mod.sha256_bytes(handle.read()) == row["raw_capture_sha256"]


def test_missing_body_is_persisted_for_later_bounded_fetch(tmp_path: Path):
    root = tmp_path / "project"
    source = root / "source.csv"
    _write_csv(source, [_row(body="")])
    out = root / "data" / "historical_news_backfill"
    engine = mod.BackfillEngine(root, out, [mod.SourceSpec("S", source, "csv")])
    summary = engine.run_once(resume=False, max_requests=0, max_local_items=10)
    assert summary["requests_used"] == 0
    backlog = list(mod.read_jsonl(out / "ledgers" / "FETCH_BACKLOG.jsonl"))
    assert len(backlog) == 1
    assert backlog[0]["status"] == "PENDING"
    state = json.loads((out / "state" / "ARTICLE_BACKFILL_WORKER_STATE.json").read_text(encoding="utf-8"))
    assert state["pending_article_fetches"] == 1


def test_canonical_state_merge_preserves_orchestrator_fields_and_runtime(tmp_path: Path):
    root = tmp_path / "project"
    source = root / "source.csv"
    _write_csv(source, [_row()])
    canonical_path = root / "state" / "HIST_NEWS_BACKFILL_STATE.json"
    canonical_path.parent.mkdir(parents=True)
    canonical_path.write_text(json.dumps({
        "productive_seconds": 123.5,
        "current_phase": "ROOT_OWNED_PHASE",
        "orchestrator_sentinel": "KEEP",
    }), encoding="utf-8")
    out = root / "data" / "historical_news_backfill"
    engine = mod.BackfillEngine(root, out, [mod.SourceSpec("S", source, "csv")])
    summary = engine.run_once(resume=False, max_requests=0, max_local_items=10)
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert summary["state_path"] == str(canonical_path)
    assert canonical["productive_seconds"] == 123.5
    assert canonical["current_phase"] == "ROOT_OWNED_PHASE"
    assert canonical["orchestrator_sentinel"] == "KEEP"
    assert canonical["article_backfill"]["canonical_productive_seconds_modified"] is False


def test_resume_dedupes_request_event_source_and_article(tmp_path: Path):
    root = tmp_path / "project"
    source = root / "source.csv"
    _write_csv(source, [_row()])
    out = root / "data" / "historical_news_backfill"
    spec = [mod.SourceSpec("S", source, "csv")]
    first = mod.BackfillEngine(root, out, spec)
    assert first.run_once(resume=False, max_requests=0, max_local_items=10)["new_canonical_articles"] == 1
    second = mod.BackfillEngine(root, out, spec)
    summary = second.run_once(resume=True, max_requests=0, max_local_items=10)
    assert summary["status"] == "NO_PROGRESS"
    assert summary["new_canonical_articles"] == 0
    assert len(list(mod.read_jsonl(out / "ledgers" / "EVENT_SOURCE_REGISTRY.jsonl"))) == 1
    assert len(pd.read_parquet(out / "ARTICLES_NORMALIZED.parquet")) == 1


def test_current_capture_never_becomes_historical_first_seen(tmp_path: Path):
    candidate = mod.source_candidate(_row(first_seen_at_utc="2025-01-02T14:30:00Z"), "TEST")
    row = mod.build_normalized(candidate, tmp_path, raw=b"{}", raw_mime="application/json",
                               raw_kind="LOCAL_PROVIDER_RECORD")
    assert row["first_seen_at_utc"] is None
    assert row["first_seen_observed"] is False
    assert "FIRST_SEEN_UNOBSERVED" in row["tier_precondition_reasons"]


def test_explicit_historical_first_seen_can_satisfy_tier_a(tmp_path: Path):
    material = _row()
    material.update({
        "first_seen_at_utc": "2025-01-02T14:30:05Z",
        "first_seen_source": "ARCHIVED_PROVIDER_RECEIPT",
        "first_seen_genuine_historical": True,
    })
    candidate = mod.source_candidate(material, "TEST")
    row = mod.build_normalized(candidate, tmp_path, raw=b"{}", raw_mime="application/json",
                               raw_kind="LOCAL_PROVIDER_RECORD")
    assert row["first_seen_observed"] is True
    assert row["tier"] == "A"


def test_tier_c_when_timestamp_is_imprecise_or_origin_unknown(tmp_path: Path):
    material = _row(
        event_time_utc="2025-01-02T00:00:00Z", timestamp_quality="",
        source="MYSTERY", url="https://unknown.example/a", body="short",
    )
    material["published_at"] = "2025-01-02"
    material.pop("event_time_utc")
    candidate = mod.source_candidate(material, "MYSTERY")
    row = mod.build_normalized(candidate, tmp_path, raw=b"{}", raw_mime="application/json",
                               raw_kind="LOCAL_PROVIDER_RECORD")
    assert row["tier"] == "C"
    assert row["causal_eligible_precondition"] is False


def test_request_registry_quarantines_third_identical_failure(tmp_path: Path):
    registry = mod.RequestRegistry(tmp_path)
    url = "https://example.com/article"
    digest = registry.digest("ARTICLE", url)
    for _ in range(3):
        registry.record(digest=digest, kind="ARTICLE", url=url, status="RETRY",
                        error="Timeout", retry_after_seconds=0)
    assert registry.latest[digest]["status"] == "QUARANTINED"
    allowed, _ = registry.may_attempt("ARTICLE", url, datetime.now(timezone.utc))
    assert allowed is False


def test_request_registry_refreshes_successful_robots_but_not_article(tmp_path: Path):
    registry = mod.RequestRegistry(tmp_path)
    robots_url = "https://example.com/robots.txt"
    article_url = "https://example.com/article"
    robots_digest = registry.digest("ROBOTS", robots_url)
    article_digest = registry.digest("ARTICLE", article_url)
    registry.record(digest=robots_digest, kind="ROBOTS", url=robots_url, status="SUCCESS", http_status=200)
    registry.record(digest=article_digest, kind="ARTICLE", url=article_url, status="SUCCESS", http_status=200)
    now = datetime.now(timezone.utc)
    assert registry.may_attempt("ROBOTS", robots_url, now)[0] is True
    assert registry.may_attempt("ARTICLE", article_url, now)[0] is False


def test_public_url_guard_and_html_timestamp_provenance():
    assert not mod.safe_public_url("http://127.0.0.1/private")
    assert not mod.safe_public_url("file:///tmp/a")
    assert mod.safe_public_url("https://example.com/news")
    raw = b'''<html><head><meta property="og:title" content="Trial update">
    <meta property="og:site_name" content="Reuters">
    <meta property="article:published_time" content="2024-06-01T09:03:12-04:00"></head>
    <body><article><p>This is a sufficiently descriptive historical article paragraph.</p></article></body></html>'''
    parsed = mod.parse_html_article(raw, "https://example.com/a", "TEST")
    assert parsed["published_at_source"] == "HTML_META:article:published_time"
    assert parsed["published_at_precision"] == "EXACT_SECOND"
    assert mod.iso_utc(parsed["published_at"]) == "2024-06-01T13:03:12Z"


def test_us_timezone_abbreviation_is_explicitly_normalized_without_warning():
    stamp, precision, confidence, inferred = mod.parse_published(
        "4:53 AM EST February 19, 2021", "GDELT_GAL_DIRECT_RECOVERY"
    )
    assert mod.iso_utc(stamp) == "2021-02-19T09:53:00Z"
    assert precision == "EXACT_MINUTE"
    assert confidence == "HIGH"
    assert inferred is False


def test_plain_url_text_is_not_sent_through_html_parser():
    assert mod.normalize_text("https://example.com/a?x=1&y=2") == "https://example.com/a?x=1&y=2"
    assert mod.normalize_text("<b>Trial</b>&nbsp;result") == "Trial result"


def test_rss_query_is_label_blind_deterministic_and_time_bounded():
    event = {
        "canonical_event_id": "E", "market": "US", "ticker": "ABCD",
        "issuer_name": "Acme Bio", "event_time_utc": "2025-01-02T14:30:00Z",
        "fwd_ret": 99, "correctness": True,
    }
    url = mod.BackfillEngine._rss_query_url(event)
    assert "Acme+Bio" in url and "ABCD" in url
    assert "after%3A2025-01-01" in url and "before%3A2025-01-04" in url
    assert "99" not in url and "correct" not in url


def test_no_event_master_records_real_discovery_pending_state(tmp_path: Path):
    root = tmp_path / "project"
    out = root / "data" / "historical_news_backfill"
    engine = mod.BackfillEngine(root, out, [])
    summary = engine.run_once(resume=False, max_requests=2, max_local_items=0,
                              max_discovery_queries=1)
    state = json.loads((out / "state" / "ARTICLE_BACKFILL_WORKER_STATE.json").read_text(encoding="utf-8"))
    assert summary["requests_used"] == 0
    assert state["discovery_expansion_pending"] is True
    assert state["discovery_expansion_reason"] == "EVENT_MASTER_NOT_AVAILABLE"


def test_bounded_rss_discovery_persists_raw_query_and_event_source(tmp_path: Path):
    root = tmp_path / "project"
    out = root / "data" / "historical_news_backfill"
    out.mkdir(parents=True)
    pd.DataFrame([{
        "canonical_event_id": "E-RSS", "market": "US", "ticker": "ABCD",
        "issuer_name": "Acme Bio", "event_time_utc": "2025-01-02T14:30:00Z",
    }]).to_parquet(out / "EVENT_MASTER.parquet", index=False)
    rss = b'''<?xml version="1.0"?><rss><channel><item>
    <title>Acme Bio announces Phase 3 trial update - Reuters</title>
    <link>https://www.reuters.com/world/acme-trial</link>
    <pubDate>Thu, 02 Jan 2025 14:29:30 GMT</pubDate><source>Reuters</source>
    </item></channel></rss>'''

    class FakeFetcher:
        max_requests = 2
        requests_used = 0

        def document(self, _url, _kind, _mimes):
            self.requests_used += 1
            return rss, "application/rss+xml"

    engine = mod.BackfillEngine(root, out, [])
    queries, discovered, operations = engine._discover_from_event_master(FakeFetcher(), 1)
    assert (queries, discovered, operations) == (1, 1, 1)
    article = next(iter(engine.article_latest.values()))
    assert article["canonical_event_id"] == "E-RSS"
    assert article["event_match_quality"] == "HIGH"
    assert article["discovery_window"] == "CAUSAL_T2_CANDIDATE"
    query = list(mod.read_jsonl(out / "ledgers" / "DISCOVERY_QUERY_REGISTRY.jsonl"))[-1]
    assert query["status"] == "SUCCESS"
    raw_path = out / query["raw_capture_path"]
    with gzip.open(raw_path, "rb") as handle:
        assert mod.sha256_bytes(handle.read()) == query["raw_capture_sha256"]
    event_source = list(mod.read_jsonl(out / "ledgers" / "EVENT_SOURCE_REGISTRY.jsonl"))
    assert len(event_source) == 1


def test_cli_is_bounded_and_no_progress_returns_zero(tmp_path: Path, capsys):
    root = tmp_path / "empty"
    out = root / "data" / "historical_news_backfill"
    code = mod.main(["--once", "--resume", "--max-requests", "0", "--max-local-items", "0",
                     "--root", str(root), "--output-root", str(out)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bounded"] is True
    assert payload["status"] == "NO_PROGRESS"
