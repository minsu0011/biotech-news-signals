import pandas as pd

from historical_news_multisource_v2_price_audit import exact_times, pre_event_features


def test_exact_times_enforces_t2_and_actual_entry_plus_30m() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01T14:00:00Z", periods=61, freq="1min"))
    eligible, reason, timing = exact_times(pd.Timestamp("2026-01-01T14:00:00Z"), timestamps)
    assert eligible
    assert reason == "EXACT_TIMING_ELIGIBLE"
    assert timing["actual_entry_time_utc"] == pd.Timestamp("2026-01-01T14:02:00Z")
    assert timing["actual_exit_time_utc"] == pd.Timestamp("2026-01-01T14:32:00Z")


def test_exact_times_rejects_entry_gap_over_60_seconds() -> None:
    timestamps = pd.Series([
        pd.Timestamp("2026-01-01T14:00:00Z"),
        pd.Timestamp("2026-01-01T14:04:00Z"),
        pd.Timestamp("2026-01-01T14:34:00Z"),
    ])
    eligible, reason, _ = exact_times(pd.Timestamp("2026-01-01T14:00:00Z"), timestamps)
    assert not eligible
    assert reason == "ENTRY_TIMING_MISS"


def test_pre_event_features_never_include_entry_or_future_bars(tmp_path) -> None:
    timestamps = pd.date_range("2026-01-01T14:00:00Z", periods=40, freq="1min")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "open": range(100, 140), "high": range(101, 141), "low": range(99, 139),
        "close": range(100, 140), "volume": range(1, 41),
    })
    path = tmp_path / "bars.parquet"
    frame.to_parquet(path, index=False)
    entry = pd.Timestamp("2026-01-01T14:20:00Z")
    result = pre_event_features(path, entry)
    assert result["pre_event_feature_status"] == "PASS"
    assert result["latest_completed_bar_end_utc"] <= entry
    assert result["completed_pre_entry_bar_count"] == 20
