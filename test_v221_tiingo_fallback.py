from __future__ import annotations

import json

import pytest

import tiingo_fallback_v221 as tiingo


def test_missing_token_is_fail_closed_and_never_serialized(tmp_path, monkeypatch):
    monkeypatch.setattr(tiingo, "PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(tiingo, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(
        tiingo,
        "credential_status",
        lambda: {"TIINGO_API_TOKEN": "MISSING"},
    )
    preflight, plan = tiingo.write_preflight()
    assert preflight["status"] == "TIINGO_CREDENTIAL_MISSING"
    assert preflight["network_attempted"] is False
    assert preflight["values_serialized"] is False
    assert plan["backfill_authorized"] is False
    assert plan["parity"]["executed"] is False


def test_present_token_still_requires_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(tiingo, "PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(tiingo, "PLAN_PATH", tmp_path / "plan.json")
    secret = "unit-test-secret-value"
    monkeypatch.setattr(
        tiingo,
        "credential_status",
        lambda: {"TIINGO_API_TOKEN": "PRESENT"},
    )
    preflight, plan = tiingo.write_preflight()
    serialized = json.dumps([preflight, plan])
    assert preflight["status"] == "CREDENTIAL_PRESENT_PARITY_REQUIRED"
    assert plan["backfill_authorized"] is False
    assert secret not in serialized


def test_fetch_without_token_never_calls_network(monkeypatch):
    monkeypatch.setattr(
        tiingo,
        "credential_status",
        lambda: {"TIINGO_API_TOKEN": "MISSING"},
    )
    monkeypatch.setattr(tiingo.requests, "get", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(RuntimeError, match="TIINGO_CREDENTIAL_MISSING"):
        tiingo.fetch_1m_bars("AAPL", "2024-01-01", "2024-01-02")
