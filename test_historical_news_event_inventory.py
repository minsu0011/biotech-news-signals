from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import historical_news_event_inventory as inventory


def _write_sources(root: Path, unsafe_pool: bool = False) -> None:
    data = root / "data"
    pool = data / "untouched_pool"
    pool.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "US:NEW",
                "event_group_id": "G:NEW",
                "market": "US",
                "ticker": "NEW",
                "company": "New Bio",
                "event_time_utc": "2026-02-01T14:30:00Z",
                "source": "REUTERS_QIU",
                "source_family": "US_NEWS",
                "url": "https://example.com/new",
                "article_id": "n1",
                "y": 1,
                "fwd_ret": 999.0,
            },
            {
                "event_id": "US:SEALED",
                "event_group_id": "G:SEALED",
                "market": "US",
                "ticker": "BAD",
                "company": "Bad Bio",
                "event_time_utc": "2026-02-01T14:31:00Z",
                "source": "US_NEWS_SEAL",
                "source_family": "US_NEWS",
                "url": "https://example.com/sealed",
                "article_id": "bad",
                "y": 0,
                "fwd_ret": -999.0,
            },
            {
                "event_id": "SEC:NOT_NEWS",
                "event_group_id": "G:SEC",
                "market": "US",
                "ticker": "SEC",
                "company": "SEC Bio",
                "event_time_utc": "2026-02-01T14:32:00Z",
                "source": "SEC",
                "source_family": "US_SEC",
                "url": "https://example.com/sec",
                "article_id": "sec",
                "y": 1,
                "fwd_ret": 500.0,
            },
        ]
    ).to_csv(data / "dev_contract_v36_events.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "event_id": "US:OLD",
                "market": "US",
                "ticker": "OLD",
                "company": "Old Bio",
                "event_time_utc": "2024-01-01T14:30:00Z",
                "source": "REUTERS_QIU",
                "form": "",
                "headline": "old",
                "body": "body",
                "event_type": "TRIAL",
                "timestamp_quality": "EXACT",
                "url": "https://example.com/old",
                "article_id": "o1",
                "prediction_correct": True,
                "strategy_return": 123.0,
            }
        ]
    ).to_csv(data / "events_us_news_real.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "KR:POOL",
                "market": "KR",
                "ticker": "000001",
                "company": "KR Bio",
                "event_time_utc": "2025-06-01T01:30:00Z",
                "source": "GOOGLE_NEWS_RSS_KR",
                "form": "",
                "headline": "kr",
                "body": "body",
                "event_type": "OTHER",
                "timestamp_quality": "EXACT",
                "url": "https://example.kr/pool",
                "article_id": "k1",
                "selection_uses_labels": unsafe_pool,
                "label": 1,
            }
        ]
    ).to_csv(pool / "gnews_kr_fixture.csv.gz", index=False, compression="gzip")


def test_inventory_is_outcome_blind_recent_first_and_resume_safe(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    output = tmp_path / "data/historical_news_backfill"
    state = tmp_path / "state/HIST_NEWS_BACKFILL_STATE.json"

    first = inventory.build_inventory(tmp_path, output, state, resume=True)
    assert first["action"] == "BUILT"
    frame = pd.read_parquet(output / "EVENT_MASTER.parquet")
    assert frame["canonical_event_id"].tolist() == ["US:NEW", "KR:POOL", "US:OLD"]
    assert not any(column.casefold() in inventory.FORBIDDEN_COLUMN_NAMES for column in frame.columns)
    assert "US:SEALED" not in set(frame["canonical_event_id"])
    assert "SEC:NOT_NEWS" not in set(frame["canonical_event_id"])
    assert frame.loc[frame["canonical_event_id"] == "US:NEW", "exact_price_eligible"].item()
    assert frame["inventory_priority_year"].tolist() == [2026, 2025, 2024]
    assert frame["event_time_utc"].dt.tz is not None

    event_sha = inventory.sha256_file(output / "EVENT_MASTER.parquet")
    second = inventory.build_inventory(tmp_path, output, state, resume=True)
    assert second["action"] == "NO_INPUT_CHANGE"
    assert inventory.sha256_file(output / "EVENT_MASTER.parquet") == event_sha
    loaded_state = json.loads(state.read_text(encoding="utf-8"))
    assert set(loaded_state) == {"event_inventory"}
    assert loaded_state["event_inventory"]["status"] == "NO_INPUT_CHANGE"


def test_declared_label_aware_pool_fails_closed(tmp_path: Path) -> None:
    _write_sources(tmp_path, unsafe_pool=True)
    with pytest.raises(RuntimeError, match="outcome-aware selection"):
        inventory.build_inventory(
            tmp_path,
            tmp_path / "data/historical_news_backfill",
            tmp_path / "state/HIST_NEWS_BACKFILL_STATE.json",
        )


def test_forbidden_namespaces_and_state_merge(tmp_path: Path) -> None:
    root = tmp_path
    forbidden = root / "cert_research/V224_PROSPECTIVE_CONFIRMATION/events.csv"
    allowed = root / "data/events_us_news_real.csv"
    assert inventory.is_forbidden_path(forbidden, root)
    assert not inventory.is_forbidden_path(allowed, root)

    state_path = root / "state/HIST_NEWS_BACKFILL_STATE.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"article_worker": {"status": "RUNNING"}}), encoding="utf-8")
    inventory.merge_inventory_state(state_path, {"status": "COMPLETE", "events_total": 3})
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["article_worker"] == {"status": "RUNNING"}
    assert state["event_inventory"]["events_total"] == 3


def test_raw_event_id_collision_preserves_each_join_mapping() -> None:
    base = {
        "canonical_event_id": "WIRE:1",
        "existing_event_id": "WIRE:1",
        "event_group_id": "WIRE:1",
        "market": "US",
        "cik": None,
        "issuer_name": None,
        "event_timezone": "America/New_York",
        "source_family": "US_NEWS",
        "source_provider": "WIRE",
        "source_file": "data/events_us_news_real.csv",
        "source_file_sha256": "a" * 64,
        "source_row_sha256": "b" * 64,
        "existing_article_id": "1",
        "existing_url": "https://example.com/1",
        "event_type": "OTHER",
        "timestamp_quality": "EXACT",
        "exact_price_eligible": False,
        "inventory_priority_year": 2026,
        "join_key_event_id": "WIRE:1",
        "provenance_count": 1,
        "_source_role": "CURATED_LOCAL",
    }
    records = []
    for row_number, ticker, minute in ((2, "AAA", 30), (3, "BBB", 31)):
        timestamp = pd.Timestamp(f"2026-01-01T14:{minute}:00Z")
        record = dict(base)
        record.update(
            {
                "ticker": ticker,
                "issuer_id": f"US:{ticker}",
                "event_time_utc": timestamp,
                "event_time_local": timestamp.tz_convert("America/New_York").isoformat(),
                "decision_cutoff_utc": timestamp + pd.Timedelta(minutes=2),
                "source_row_number": row_number,
                "join_key_market_ticker_event_time": (
                    f"US|{ticker}|{timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"
                ),
            }
        )
        records.append(record)
    frame, audit = inventory.collapse_candidates(records)
    assert len(frame) == 2
    assert frame["canonical_event_id"].nunique() == 2
    assert set(frame["existing_event_id"]) == {"WIRE:1"}
    assert all(value.startswith("WIRE:1::") for value in frame["canonical_event_id"])
    assert audit["raw_event_id_multi_join_conflicts_resolved"] == 1
    assert audit["canonical_id_conflicts"] == 0
