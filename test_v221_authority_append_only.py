import pandas as pd
import pytest

import refresh_current_data_authority as authority


def policy():
    return {
        "selection_uses_labels": False,
        "existing_roles_must_remain_unchanged": True,
        "new_row_allocation": {
            "DEV_EXTENSION": 80,
            "RESEARCH_SEAL_POOL": 10,
            "FINAL_META_RESERVE": 10,
        },
    }


def role_row(event_id: str, role: str) -> dict:
    role_hash = authority.hashlib.sha256(
        f"{authority.ROLE_SEED}|{event_id}".encode()
    ).hexdigest()
    return {
        "event_id": event_id,
        "role": role,
        "role_hash": role_hash,
        "role_frozen_at": "2026-08-29T00:00:00+09:00",
        "labels_opened": False,
    }


def deterministic_role(event_id: str) -> str:
    role_hash = authority.hashlib.sha256(
        f"{authority.ROLE_SEED}|{event_id}".encode()
    ).hexdigest()
    bucket = int(role_hash[:8], 16) % 100
    if bucket < 80:
        return "DEV_EXTENSION"
    if bucket < 90:
        return "RESEARCH_SEAL_POOL"
    return "FINAL_META_RESERVE"


def test_append_only_role_contract_accepts_valid_new_row(monkeypatch):
    baseline = pd.DataFrame([role_row("old", "FINAL_META_RESERVE")])
    current = pd.concat(
        [baseline, pd.DataFrame([role_row("new", deterministic_role("new"))])],
        ignore_index=True,
    )
    monkeypatch.setattr(authority, "sha256", lambda _path: "test-sha")
    result = authority.validate_append_only_roles(current, baseline, policy())
    assert result["baseline_roles_unchanged"] is True
    assert result["appended_rows"] == 1


def test_append_only_role_contract_rejects_changed_frozen_role(monkeypatch):
    baseline = pd.DataFrame([role_row("old", "FINAL_META_RESERVE")])
    current = baseline.copy()
    current.loc[0, "role"] = "DEV_EXTENSION"
    monkeypatch.setattr(authority, "sha256", lambda _path: "test-sha")
    with pytest.raises(RuntimeError, match="FROZEN_ROLE_BASELINE_CHANGED"):
        authority.validate_append_only_roles(current, baseline, policy())
