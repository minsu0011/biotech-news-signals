from pathlib import Path

import pytest

import activate_v221_new_data_epoch as activation


def test_activation_refuses_not_ready_epoch():
    with pytest.raises(RuntimeError, match="EPOCH_ACTIVATION_NOT_READY"):
        activation.build_contract(
            {"status": "NOT_READY"},
            {"counts": {"DEV_EXTENSION": {"US_SEC": 117}}},
        )


def test_activation_builds_canonical_contract_only_after_all_gates(monkeypatch, tmp_path: Path):
    readiness_file = tmp_path / "readiness.json"
    readiness_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(activation, "READINESS_PATH", readiness_file)
    assets = {
        name: {"path": str(path), "sha256": f"sha-{name}", "bytes": 1}
        for name, path in activation.controller.DATA_EPOCH_ASSETS.items()
    }
    counts = {
        "total": 427,
        "by_source_family": {
            "US_SEC": {
                "n": 120, "y_0": 71, "y_1": 49,
                "unique_dates": 107, "unique_tickers": 62,
                "chronological_blocks": [],
            },
            "KR_NEWS": {
                "n": 307, "y_0": 181, "y_1": 126,
                "unique_dates": 57, "unique_tickers": 113,
                "chronological_blocks": [],
            },
        },
    }
    support = {
        "total_n_ge_240": True,
        "only_required_source_families": True,
        "by_source_family": {
            "US_SEC": {"all": True},
            "KR_NEWS": {"all": True},
        },
    }
    monkeypatch.setattr(activation.controller, "extension_asset_metadata", lambda: assets)
    monkeypatch.setattr(
        activation.controller,
        "inspect_extension_support",
        lambda _path: (counts, support, []),
    )
    readiness = {
        "status": "READY_TO_FREEZE",
        "seal_or_final_labels_opened": False,
        "blocking_reasons": [],
        "checks": {"total_n": True, "source:US_SEC": True, "source:KR_NEWS": True},
        "files": assets,
    }
    authority = {
        "counts": {"DEV_EXTENSION": {"US_SEC": 120}},
        "invariants": {"v36": {"match": True}, "v69": {"match": True}},
        "role_assignment_invariant": {
            "baseline_roles_unchanged": True,
            "all_role_hashes_valid": True,
            "appended_roles_match_frozen_policy": True,
        },
    }
    contract, _ = activation.build_contract(readiness, authority)
    assert contract["status"] == "FROZEN"
    assert contract["frozen"] is True
    assert contract["counts"] == counts
    assert len(contract["data_epoch_sha256"]) == 64
    assert contract["reserved_labels_opened"] is False
