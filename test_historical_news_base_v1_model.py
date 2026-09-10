from __future__ import annotations

import numpy as np
import pandas as pd

from historical_news_base_v1_model import (
    _prequential_target_priors,
    map_v69_to_frozen_events,
    purge_ticker_embargo,
)


def _prior_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_event_id": ["E1", "E2", "E3"],
            "event_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"], utc=True
            ),
            "publisher_primary": ["P", "P", "P"],
            "topic_primary": ["T", "T", "T"],
            "publisher_topic": ["P|T", "P|T", "P|T"],
            "correct_target": [1, 0, 1],
        }
    )


def test_target_priors_are_prequential_and_do_not_use_same_row() -> None:
    train = _prior_frame().iloc[:2].copy()
    valid = _prior_frame().iloc[[2]].copy()
    train_prior, valid_prior, columns = _prequential_target_priors(train, valid, strength=2.0)
    publisher_prior = columns.index("prior_publisher_primary")
    publisher_support = columns.index("support_publisher_primary")
    assert train_prior[0, publisher_prior] == 0.5
    assert train_prior[0, publisher_support] == 0.0
    assert train_prior[1, publisher_prior] > 0.5  # row 1 sees only row 0's target
    assert valid_prior[0, publisher_support] == np.log1p(2)


def test_ticker_embargo_purges_only_near_boundary_same_ticker() -> None:
    train = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "event_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-09T00:00:00Z", "2024-01-09T00:00:00Z"], utc=True
            ),
        }
    )
    valid = pd.DataFrame(
        {"ticker": ["AAA"], "event_time_utc": pd.to_datetime(["2024-01-10T00:00:00Z"], utc=True)}
    )
    purged, count = purge_ticker_embargo(train, valid, days=7)
    assert count == 1
    assert len(purged) == 2
    assert ((purged["ticker"] == "AAA") & (purged["event_time_utc"] == pd.Timestamp("2024-01-01T00:00:00Z"))).any()
    assert (purged["ticker"] == "BBB").any()


def test_v69_collision_maps_unique_exact_eligible_kr_ticker() -> None:
    raw_id = "NAVER:000100:ROW"
    events = pd.DataFrame(
        {
            "canonical_event_id": ["EXACT", "AUX"],
            "existing_event_id": [raw_id, raw_id],
            "join_key_event_id": [raw_id, raw_id],
            "market": ["KR", "KR"],
            "ticker": ["000100", "100"],
            "event_time_utc": pd.to_datetime(["2026-08-12T02:04:00Z"] * 2, utc=True),
            "exact_price_eligible": [True, False],
        }
    )
    direction = pd.DataFrame(
        {
            "event_id": [raw_id],
            "market": ["KR"],
            "ticker": ["000100"],
            "event_time_utc": ["2026-08-12T02:04:00Z"],
        }
    )
    mapped, audit = map_v69_to_frozen_events(direction, events)
    assert mapped.iloc[0]["canonical_event_id"] == "EXACT"
    assert audit["collision_rows_resolved_by_exact_eligibility"] == 1
