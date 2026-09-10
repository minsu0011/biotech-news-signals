from __future__ import annotations

import json
import os

import provider_env_runtime as runtime


def test_presence_never_returns_values(monkeypatch):
    for index, name in enumerate(runtime.PROVIDER_ENV_NAMES):
        monkeypatch.setenv(name, f"unit-test-provider-value-{index}")
    report = runtime.presence()
    assert set(report) == set(runtime.PROVIDER_ENV_NAMES)
    assert set(report.values()) == {"PRESENT"}
    serialized = json.dumps(report)
    assert "unit-test-provider-value" not in serialized


def test_allowlist_rejects_unrelated_environment_name():
    try:
        runtime.import_user_environment_if_missing(("PATH",))
    except ValueError as error:
        assert "not allowlisted" in str(error)
    else:
        raise AssertionError("unrelated environment name was accepted")


def test_all_present_tracks_markers(monkeypatch):
    for name in runtime.PROVIDER_ENV_NAMES:
        monkeypatch.setenv(name, "test-only")
    assert runtime.all_present()
    monkeypatch.delenv(runtime.PROVIDER_ENV_NAMES[0])
    monkeypatch.setattr(runtime, "_windows_user_value", lambda name: None)
    assert runtime.import_user_environment_if_missing()[runtime.PROVIDER_ENV_NAMES[0]] == "MISSING"
    assert not runtime.all_present()


def test_alpaca_standard_and_compatibility_names_are_allowlisted(monkeypatch):
    alpaca_names = (
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    )
    for name in alpaca_names:
        monkeypatch.setenv(name, "test-only")
    assert runtime.presence(alpaca_names) == {name: "PRESENT" for name in alpaca_names}


def test_refresh_user_environment_removes_stale_missing_values(monkeypatch):
    name = runtime.ALPACA_ENV_NAMES[0]
    monkeypatch.setenv(name, "stale-process-value")
    monkeypatch.setattr(runtime, "_windows_user_value", lambda candidate: None)
    assert runtime.refresh_user_environment((name,))[name] == "MISSING"
    assert name not in os.environ
