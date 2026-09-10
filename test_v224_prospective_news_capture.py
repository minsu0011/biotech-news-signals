"""Deterministic tests for V224 Track B prospective news capture."""
from __future__ import annotations

import gzip
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import v224_prospective_news_capture as capture


NOW = datetime(2026, 8, 30, 0, 1, 0, tzinfo=timezone.utc)


def rss_payload() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>test</title>
<item><title>AbbVie clinical trial succeeds - Reuters</title>
<link>https://news.google.com/rss/articles/TEST1</link>
<guid>gnews-test-1</guid><pubDate>Sun, 30 Aug 2026 00:00:00 GMT</pubDate>
<description>&lt;b&gt;AbbVie&lt;/b&gt; reported a clinical trial update.</description>
<source url="https://www.reuters.com">Reuters</source></item>
<item><title>Biopharma financing update - Business Wire</title>
<link>https://news.google.com/rss/articles/TEST2</link>
<guid>gnews-test-2</guid><pubDate>Sat, 29 Aug 2026 23:00:00 GMT</pubDate>
<description>Company announces financing.</description>
<source url="https://www.businesswire.com">Business Wire</source></item>
</channel></rss>"""


def broad_task() -> dict:
    return {
        "query_id": "BROAD_US_TEST", "kind": "BROAD", "market": "US",
        "ticker": None, "company": None, "query": "test when:1d",
        "query_sha256": capture.query_hash("test when:1d"), "status": "READY",
        "cadence_seconds": capture.BROAD_SUCCESS_CADENCE_SECONDS,
        "next_due_at_utc": capture.iso_utc(NOW), "attempts": 0,
        "consecutive_failures": 0, "deterministic_failure_count": 0,
        "deterministic_failure_fingerprint": None, "last_attempt_at_utc": None,
        "last_success_at_utc": None, "last_error_code": None,
    }


def raw_metadata() -> dict:
    body = rss_payload()
    return {
        "capture_id": "CAP:test", "raw_capture_path": "raw/2026/08/30/test.xml.gz",
        "raw_capture_sha256": capture.sha256_bytes(body), "raw_gzip_sha256": "gzip-test",
    }


def test_parse_feed_preserves_publisher_and_enforces_tplus2() -> None:
    policy, _ = capture.load_policy()
    _, policy_sha = capture.load_policy()
    _, aliases = capture.load_universes(policy)
    records, invalid = capture.parse_feed(
        rss_payload(), broad_task(), NOW, raw_metadata(), aliases,
        capture.POLICY_VERSION, policy_sha,
    )
    assert invalid == []
    assert len(records) == 2
    timely, late = records
    assert timely["publisher"] == "Reuters"
    assert timely["query_policy_version"] == 2
    assert timely["query_policy_sha256"] == policy_sha
    assert timely["publisher_url"] == "https://www.reuters.com"
    assert timely["publisher_class"] == "WIRE_SERVICE"
    assert timely["headline"] == "AbbVie clinical trial succeeds"
    assert timely["ticker"] == "ABBV"
    assert timely["ticker_mapping_evidence"] == "CONTENT_ALIAS_EXACT"
    assert timely["t_plus_2_usable"] is True
    assert timely["causal_eligibility_reason"] == "PROVEN_LOCAL_FIRST_SEEN_BY_T_PLUS_2"
    assert timely["canonical_url"] is None
    assert late["publisher_class"] == "PR_DISTRIBUTION"
    assert late["t_plus_2_usable"] is False
    assert late["historical_backfill"] is True
    assert late["causal_eligibility_reason"] == "FIRST_SEEN_AFTER_DECISION_CUTOFF_HISTORICAL_ONLY"


def test_one_shot_persists_raw_and_append_only_ledgers_resume_safe(tmp_path: Path) -> None:
    namespace = tmp_path / "namespace"
    state_path = tmp_path / "state" / "NEWS_PROSPECTIVE_CAPTURE_STATE.json"

    def fake_fetch(task: dict, policy: dict, timeout: float) -> dict:
        assert task["kind"] == "BROAD"
        return {"ok": True, "status": 200, "body": rss_payload(),
                "headers": {"Content-Type": "application/rss+xml"},
                "error_code": None, "deterministic": False, "request_url": "https://example.invalid/rss"}

    result = capture.run_once(
        resume=False, max_queries=1, timeout_seconds=1, min_interval_seconds=0,
        namespace=namespace, state_path=state_path, fetcher=fake_fetch,
        now_fn=lambda: NOW, sleep_fn=lambda _: None,
    )
    assert result["status"] == "READY"
    assert result["requests"] == 1
    assert result["new_articles"] == 2
    assert result["t_plus_2_usable"] == 1
    assert result["historical_only"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["research_seal_or_final_rows_read"] is False
    assert state["last_run"]["credentials_logged"] is False
    assert len(state["queue"]) == 4 + 796 + 413
    raw_rows = capture.load_jsonl(namespace / "ledger" / "raw_capture_ledger.jsonl")
    news_rows = capture.load_jsonl(namespace / "ledger" / "news_records.jsonl")
    observation_rows = capture.load_jsonl(namespace / "ledger" / "capture_observations.jsonl")
    assert len(raw_rows) == 1 and len(news_rows) == 2 and len(observation_rows) == 2
    raw_path = namespace / raw_rows[0]["raw_capture_path"]
    assert gzip.decompress(raw_path.read_bytes()) == rss_payload()
    assert capture.sha256_file(raw_path) == raw_rows[0]["raw_gzip_sha256"]
    with pytest.raises(RuntimeError, match="STATE_EXISTS_USE_RESUME"):
        capture.run_once(
            resume=False, max_queries=0, namespace=namespace, state_path=state_path,
            fetcher=fake_fetch, now_fn=lambda: NOW, sleep_fn=lambda _: None,
        )


def test_three_identical_deterministic_failures_quarantine_and_explicit_resume() -> None:
    task = broad_task()
    for attempt in range(3):
        quarantined = capture.schedule_failure(task, NOW + timedelta(minutes=attempt), "HTTP_403", True, "raw")
    assert quarantined is True
    assert task["status"] == "QUARANTINED"
    assert task["deterministic_failure_count"] == 3
    state = {"queue": [task]}
    rows = capture.resume_quarantined(state, NOW + timedelta(hours=1))
    assert len(rows) == 1
    assert task["status"] == "READY"
    assert task["deterministic_failure_count"] == 0


def test_changed_deterministic_error_class_does_not_accumulate() -> None:
    task = broad_task()
    capture.schedule_failure(task, NOW, "HTTP_403", True, "raw-a")
    capture.schedule_failure(task, NOW, "HTTP_400", True, "raw-b")
    assert task["deterministic_failure_count"] == 1
    assert task["status"] == "BACKOFF"


def test_transient_backoff_and_circuit_breaker_thresholds() -> None:
    task = broad_task()
    capture.schedule_failure(task, NOW, "HTTP_503", False)
    first_due = capture.parse_utc(task["next_due_at_utc"])
    capture.schedule_failure(task, NOW, "HTTP_503", False)
    second_due = capture.parse_utc(task["next_due_at_utc"])
    assert second_due > first_due
    assert task["status"] == "BACKOFF"
    assert task["deterministic_failure_count"] == 0
    assert capture.should_open_circuit(3, 3, 3) is False
    assert capture.should_open_circuit(4, 3, 3) is True
    assert capture.should_open_circuit(5, 1, 5) is True


def test_record_schema_rejects_causal_misstatement() -> None:
    policy, policy_sha = capture.load_policy()
    _, aliases = capture.load_universes(policy)
    record = capture.parse_feed(
        rss_payload(), broad_task(), NOW, raw_metadata(), aliases,
        capture.POLICY_VERSION, policy_sha,
    )[0][0]
    schema = json.loads(capture.SCHEMA_PATH.read_text(encoding="utf-8"))
    capture.validate_record(record, schema)
    record["t_plus_2_usable"] = False
    with pytest.raises(RuntimeError, match="TPLUS2_CAUSAL_INVARIANT"):
        capture.validate_record(record, schema)


def test_v1_policy_and_raw_bytes_are_unchanged() -> None:
    assert capture.sha256_file(capture.V1_POLICY_PATH) == capture.V1_POLICY_SHA256
    expected = {
        "0a2126dbb13d8392dbd427f32330850f9f5c50b5da4e33efddb878424da5b213.xml.gz":
            "64a0625392e30fa35aa6d9494dede55c0a5ecb2480213295e0e598333d4d3f3e",
        "e07ea2d707ef050ca978ee3eb72c74a686b288dfb478d95e2e58f956da829bb2.xml.gz":
            "860d743972ce26a6eedd65fe7f518fe07b7e2189e236504787183e2b8da1c425",
    }
    raw_root = capture.NAMESPACE / "raw" / "2026" / "08" / "29"
    for name, digest in expected.items():
        assert capture.sha256_file(raw_root / name) == digest


def test_v2_broad_cadence_issuer_cadence_and_selector_priority() -> None:
    policy, _ = capture.load_policy()
    universes, _ = capture.load_universes(policy)
    queue = capture.make_queue(policy, universes, NOW)
    broad = next(item for item in queue if item["kind"] == "BROAD")
    issuer = next(item for item in queue if item["kind"] == "ISSUER")
    capture.schedule_success(broad, NOW)
    capture.schedule_success(issuer, NOW)
    assert capture.parse_utc(broad["next_due_at_utc"]) - NOW == timedelta(seconds=90)
    assert capture.parse_utc(broad["next_due_at_utc"]) <= NOW + timedelta(minutes=2)
    assert capture.parse_utc(issuer["next_due_at_utc"]) - NOW == timedelta(hours=8)
    all_due = capture.make_queue(policy, universes, NOW)
    selected = capture.select_due_queries({"queue": all_due}, NOW, 6)
    assert [item["kind"] for item in selected[:4]] == ["BROAD"] * 4
    assert [item["kind"] for item in selected[4:]] == ["ISSUER"] * 2


def test_v1_to_v2_migration_preserves_ledgers_totals_and_attempts(tmp_path: Path) -> None:
    policy, policy_sha = capture.load_policy()
    universes, _ = capture.load_universes(policy)
    new_queue = capture.make_queue(policy, universes, NOW)
    old_queue = copy.deepcopy(new_queue)
    for number, item in enumerate(old_queue):
        item.pop("cadence_seconds")
        item["attempts"] = number % 4
        if item["kind"] == "BROAD":
            item["next_due_at_utc"] = capture.iso_utc(NOW + timedelta(hours=8))
    old_state = capture.initial_state(capture.V1_POLICY_SHA256, old_queue, NOW - timedelta(hours=1))
    old_state["schema_version"] = 1
    old_state.pop("query_policy_version")
    old_state.pop("migration_history_path")
    old_state["totals"]["requests"] = 17
    old_state["totals"]["new_articles"] = 9
    expected_totals = copy.deepcopy(old_state["totals"])
    expected_attempts = {item["query_id"]: item["attempts"] for item in old_queue}
    state_path = tmp_path / "state.json"
    capture.atomic_json(state_path, old_state)
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    immutable = ledger_dir / "news_records.jsonl"
    immutable.write_bytes(b'{"article_id":"v1"}\n')
    immutable_sha = capture.sha256_file(immutable)
    history = ledger_dir / "state_migration_history.jsonl"
    migrated = capture.load_or_initialize_state(
        state_path, policy_sha, new_queue, NOW, True, history,
    )
    assert migrated["query_policy_version"] == 2
    assert migrated["query_policy_sha256"] == policy_sha
    assert migrated["totals"] == expected_totals
    assert {item["query_id"]: item["attempts"] for item in migrated["queue"]} == expected_attempts
    assert capture.sha256_file(immutable) == immutable_sha
    assert len(capture.load_jsonl(history)) == 1
    assert all(
        capture.parse_utc(item["next_due_at_utc"]) <= NOW + timedelta(minutes=2)
        for item in migrated["queue"] if item["kind"] == "BROAD"
    )


def test_zero_request_cycle_reads_no_outcome_price_seal_or_final_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    accessed: list[str] = []
    original_open = Path.open

    def audited_open(self: Path, *args, **kwargs):
        accessed.append(str(self).casefold())
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    result = capture.run_once(
        resume=False, max_queries=0, min_interval_seconds=0,
        namespace=tmp_path / "namespace", state_path=tmp_path / "state.json",
        now_fn=lambda: NOW, sleep_fn=lambda _: None,
    )
    assert result["requests"] == 0
    forbidden = ("seal", "final", "price", "label", "outcome", "fwd_ret")
    assert not [path for path in accessed if any(token in path for token in forbidden)]
