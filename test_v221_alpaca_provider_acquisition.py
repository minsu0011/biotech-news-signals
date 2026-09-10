from __future__ import annotations

import json

import pandas as pd

import alpaca_provider_acquisition as alpaca


def good_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-06-15T13:30:00Z", periods=3, freq="1min"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 20, 30],
        }
    )


def test_bar_qa_accepts_valid_open_timestamped_minute_bars():
    result = alpaca.bar_qa(good_bars())
    assert result["pass"] is True
    assert result["median_cadence_sec"] == 60.0
    assert result["duplicates"] == 0


def test_bar_qa_rejects_duplicate_or_invalid_prices():
    frame = good_bars()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    frame.loc[2, "open"] = 0
    result = alpaca.bar_qa(frame)
    assert result["pass"] is False
    assert result["duplicates"] == 1
    assert result["invalid_price"] == 1


def test_credential_presence_never_serializes_values(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "unit-key-value")
    monkeypatch.setenv("ALPACA_API_SECRET", "unit-secret-value")
    serialized = json.dumps(alpaca.credential_presence())
    assert "unit-key-value" not in serialized
    assert "unit-secret-value" not in serialized
    assert set(alpaca.credential_presence().values()) == {"PRESENT"}


def test_compatibility_aliases_are_accepted(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "unit-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "unit-secret")
    assert alpaca.alpaca_secret_values() == ("unit-key", "unit-secret")


def test_incomplete_standard_pair_never_mixes_with_compatibility(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "standard-key-only")
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "compatibility-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "compatibility-secret")
    assert alpaca.alpaca_secret_values() == ("compatibility-key", "compatibility-secret")
    assert alpaca.alpaca_pair_status()["selected_pair_source"] == "COMPATIBILITY"


def test_complete_standard_pair_has_explicit_priority(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "standard-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "standard-secret")
    monkeypatch.setenv("APCA_API_KEY_ID", "compatibility-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "compatibility-secret")
    assert alpaca.alpaca_secret_values() == ("standard-key", "standard-secret")
    assert alpaca.alpaca_pair_status()["selected_pair_source"] == "STANDARD"
