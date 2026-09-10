from __future__ import annotations

import pandas as pd

import acquire_v37plus_sources as sources
import v59_provider_acquisition as provider


def test_defer_old_pending_without_touching_roles_or_recent(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "MASSIVE_CACHE", tmp_path)
    frame = pd.DataFrame(
        {
            "event_id": ["old", "recent", "already_failed"],
            "ticker": ["OLD", "NEW", "BAD"],
            "event_time_local": pd.to_datetime(
                ["2017-01-03T15:00:00Z", "2025-01-03T15:00:00Z", "2017-01-04T15:00:00Z"],
                utc=True,
            ),
            "status": ["PENDING", "PENDING", "PROVIDER_FAILED"],
            "last_error": ["", "", "HTTP_403"],
            "attempt_count": [0, 0, 1],
            "source_family": ["US_SEC", "US_SEC", "US_SEC"],
        }
    )
    output, count = provider.defer_outside_massive_entitlement(
        frame, pd.Timestamp("2024-09-03T00:00:00Z")
    )
    assert count == 1
    assert output.loc[0, "status"] == "DEFERRED_OUTSIDE_VERIFIED_ENTITLEMENT_ANCHOR"
    assert output.loc[1, "status"] == "PENDING"
    assert output.loc[2, "status"] == "PROVIDER_FAILED"
    assert output.loc[0, "attempt_count"] == 0


def test_old_pending_with_existing_cache_is_not_deferred(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "MASSIVE_CACHE", tmp_path)
    cache = tmp_path / "2017" / "OLD" / "2017-01-03.parquet"
    cache.parent.mkdir(parents=True)
    cache.touch()
    frame = pd.DataFrame(
        {
            "event_id": ["old"],
            "ticker": ["OLD"],
            "event_time_local": pd.to_datetime(["2017-01-03T15:00:00Z"], utc=True),
            "status": ["PENDING"],
            "last_error": [""],
            "attempt_count": [0],
            "source_family": ["US_SEC"],
        }
    )
    output, count = provider.defer_outside_massive_entitlement(
        frame, pd.Timestamp("2024-09-03T00:00:00Z")
    )
    assert count == 0
    assert output.loc[0, "status"] == "PENDING"


def test_required_sec_and_frozen_dev_precede_newer_unrelated_news():
    frame = pd.DataFrame(
        {
            "ticker": ["NEWS", "SEAL", "DEV"],
            "trade_date": ["2026-08-27", "2026-08-20", "2025-01-02"],
            "status": ["PENDING", "PENDING", "PENDING"],
            "_data_role": ["DEV_EXTENSION", "RESEARCH_SEAL_POOL", "DEV_EXTENSION"],
            "source_family": ["US_NEWS", "US_SEC", "US_SEC"],
        }
    )
    schedule = provider.massive_pending_group_schedule(frame)
    assert schedule.ticker.tolist() == ["DEV", "SEAL", "NEWS"]


def test_sec_archive_overlap_selection_is_safe_and_end_exclusive():
    payload = {
        "filings": {
            "files": [
                {"name": "CIK0000000001-submissions-001.json", "filingFrom": "2023-01-01", "filingTo": "2024-08-31"},
                {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2024-09-01", "filingTo": "2025-01-01"},
                {"name": "../escape.json", "filingFrom": "2024-09-01", "filingTo": "2025-01-01"},
                {"name": "CIK0000000001-submissions-003.json", "filingFrom": "2026-08-28", "filingTo": "2026-09-01"},
            ]
        }
    }
    selected = sources.sec_archive_files_for_interval(
        payload,
        pd.Timestamp("2024-09-01T00:00:00Z"),
        pd.Timestamp("2026-08-28T00:00:00Z"),
    )
    assert selected == ["CIK0000000001-submissions-002.json"]
