from __future__ import annotations

from collections import Counter

import pandas as pd

import alpaca_backfill_v221 as backfill


def test_timing_only_exact_contract_passes_without_prices_or_outcomes():
    timestamps = pd.Series(pd.date_range("2023-06-15T13:30:00Z", periods=90, freq="1min"))
    passed, reason, detail = backfill.timing_only_eligibility(
        pd.Timestamp("2023-06-15T13:35:30Z"), timestamps
    )
    assert passed is True
    assert reason == "EXACT_TIMING_ELIGIBLE"
    assert detail == {
        "entry_slippage_sec": 30.0,
        "exit_slippage_sec": 0.0,
        "hold_seconds": 1800.0,
    }


def test_timing_only_exact_contract_rejects_missing_exit():
    timestamps = pd.Series(pd.date_range("2023-06-15T13:30:00Z", periods=10, freq="1min"))
    passed, reason, _ = backfill.timing_only_eligibility(
        pd.Timestamp("2023-06-15T13:35:30Z"), timestamps
    )
    assert passed is False
    assert reason == "BAR_MISSING"


def test_backfill_is_fail_closed_when_parity_is_not_passed():
    allowed, reason = backfill.authorized_feed("iex")
    assert allowed is False
    assert reason in {"PROVIDER_PARITY_FAIL", "PROVIDER_PARITY_FEED_MISMATCH"}


def test_settled_group_is_not_retried():
    settled = pd.DataFrame({"alpaca_sip_status": ["PROVIDER_FAILED", "FETCHED_VALID"]})
    pending = pd.DataFrame({"alpaca_sip_status": ["PROVIDER_FAILED", "BLOCKED_AUTH_HTTP_401"]})
    assert backfill.group_is_settled(settled, "sip") is True
    assert backfill.group_is_settled(pending, "sip") is False


def test_no_bar_group_gets_one_distinct_asof_retry():
    group = pd.DataFrame(
        {"alpaca_sip_status": ["PROVIDER_FAILED"], "last_error": ["NO_BAR_OR_QA_FAIL"]}
    )
    exhausted = pd.DataFrame(
        {"alpaca_sip_status": ["PROVIDER_FAILED"], "last_error": ["ASOF_NO_BAR_OR_QA_FAIL"]}
    )
    assert backfill.group_is_settled(group, "sip") is True
    assert backfill.group_is_settled(group, "sip", retry_asof=True) is False
    assert backfill.group_is_settled(exhausted, "sip", retry_asof=True) is True


def test_pre_2016_rows_are_deferred_without_attempt():
    frame = pd.DataFrame(
        {
            "event_time_utc": ["2015-12-31T15:00:00Z", "2016-01-04T15:00:00Z"],
            "alpaca_sip_status": ["BLOCKED_AUTH_HTTP_401", "BLOCKED_AUTH_HTTP_401"],
            "last_error": ["", ""],
        }
    )
    result, count = backfill.defer_outside_alpaca_anchor(frame, "sip")
    assert count == 1
    assert result.loc[0, "alpaca_sip_status"] == "DEFERRED_OUTSIDE_ALPACA_COVERAGE_ANCHOR"
    assert result.loc[1, "alpaca_sip_status"] == "BLOCKED_AUTH_HTTP_401"


def test_targets_do_not_double_count_existing_alpaca_contribution():
    targets = backfill.absolute_alpaca_targets(
        us_dev_total=78,
        eligible_by_role={
            "US:RESEARCH_SEAL_POOL": 251,
            "US:FINAL_META_RESERVE": 245,
        },
        alpaca_achieved=Counter(
            {
                "DEV_EXTENSION": 38,
                "RESEARCH_SEAL_POOL": 15,
                "FINAL_META_RESERVE": 25,
            }
        ),
    )
    assert targets == {
        "DEV_EXTENSION": 80,
        "RESEARCH_SEAL_POOL": 14,
        "FINAL_META_RESERVE": 30,
    }
