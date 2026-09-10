from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import v224_prospective_checkpoint_watcher as watcher
import v224_prospective_role_freeze as role_freeze


SOURCE_ROOT = Path(__file__).resolve().parent


def prepare_workspace(tmp_path: Path) -> Path:
    policy = json.loads((SOURCE_ROOT / role_freeze.POLICY_REL).read_text(encoding="utf-8"))
    files = [
        contract["path"] for contract in policy["protected_authorities"].values()
    ] + [
        value for key, value in policy["v224_authority"].items() if key.endswith("_path")
    ]
    for relative in files:
        source = SOURCE_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    policy_path = tmp_path / role_freeze.POLICY_REL
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / role_freeze.POLICY_REL, policy_path)
    return tmp_path


def event(event_id: str, timestamp: str, ticker: str = "ABCD") -> dict[str, str]:
    return {
        "event_id": event_id,
        "market": "US",
        "ticker": ticker,
        "company": "Example Biopharma Inc.",
        "event_time_utc": timestamp,
        "source": "SEC_V224_PROSPECTIVE",
        "form": "8-K",
        "headline": "8-K 8.01",
        "body": "",
        "event_type": "SEC_8-K",
        "timestamp_quality": "EXACT",
        "url": f"https://www.sec.gov/Archives/{event_id}",
    }


def test_zero_row_bootstrap_is_concrete_and_one_shot(tmp_path: Path) -> None:
    root = prepare_workspace(tmp_path)
    before = role_freeze.sha256(root / "data/V59_DATA_ROLE_ASSIGNMENT.parquet")
    status = watcher.run(root)
    after = role_freeze.sha256(root / "data/V59_DATA_ROLE_ASSIGNMENT.parquet")

    assert status["status"] == "BOOTSTRAPPED_ZERO_PROSPECTIVE_ROWS"
    assert status["one_shot"] is True
    assert status["poll_or_sleep"] is False
    assert status["prospective_role_rows"] == 0
    assert status["exact_rows"] == 0
    assert status["next_checkpoint_rows"] == 20
    assert status["acquisition"]["requests_used"] == 0
    assert before == after
    assert (root / "research/V224_PROSPECTIVE_CONFIRMATION/ROLE_FREEZE_STATUS.json").exists()
    assert (root / "research/V224_PROSPECTIVE_CONFIRMATION/WATCH_STATUS.json").exists()
    assert (root / "data/V224_PROSPECTIVE_CONFIRMATION_ROLE_FREEZE.csv.gz").exists()
    assert (root / "data/V224_PROSPECTIVE_CONFIRMATION_EXACT_ROWS.csv.gz").exists()


def test_role_freeze_appends_without_changing_existing_prefix(tmp_path: Path) -> None:
    root = prepare_workspace(tmp_path)
    raw = root / "data/events_sec_v224_prospective_test.csv"
    first = event("SEC:1:0001", "2026-08-31T14:00:05Z")
    pd.DataFrame([first]).to_csv(raw, index=False)
    central_before = role_freeze.sha256(root / "data/V59_DATA_ROLE_ASSIGNMENT.parquet")

    first_status = role_freeze.run(root)
    ledger_path = root / "data/V224_PROSPECTIVE_CONFIRMATION_ROLE_FREEZE.csv.gz"
    first_ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    first_record = first_ledger.iloc[0].to_dict()

    second = event("SEC:2:0002", "2026-08-31T15:00:05Z", ticker="EFGH")
    pd.DataFrame([first, second]).to_csv(raw, index=False)
    second_status = role_freeze.run(root)
    second_ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)

    assert first_status["rows_appended"] == 1
    assert second_status["rows_appended"] == 1
    assert len(second_ledger) == 2
    assert second_ledger.iloc[0].to_dict() == first_record
    assert second_ledger.assignment_order.tolist() == ["1", "2"]
    assert set(second_ledger.data_role) == {"V224_PROSPECTIVE_CONFIRMATION"}
    assert not {"RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"}.intersection(second_ledger.data_role)
    assert second_status["central_role_table_content_loaded"] is False
    assert second_status["central_role_rows_read"] == 0
    assert role_freeze.sha256(root / "data/V59_DATA_ROLE_ASSIGNMENT.parquet") == central_before


def synthetic_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2026-08-31T13:30:00Z", "2026-08-31T20:00:00Z", freq="1min")
    close = 10.0 + np.arange(len(timestamps)) * 0.001
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": close + 0.01,
        "low": close - 0.01,
        "close": close + 0.002,
        "volume": np.full(len(timestamps), 1000),
        "transactions": np.full(len(timestamps), 10),
        "vwap": close + 0.001,
    })


def test_locked_exact_t2_30m_and_slippage_guard() -> None:
    source = pd.Series(event("SEC:3:0003", "2026-08-31T14:00:00Z"))
    row, reason = watcher.label_event(source, synthetic_bars(), "TEST_MASSIVE_1M")
    assert reason == "OK"
    assert row is not None
    assert row["entry_slippage_sec"] == 0
    assert row["exit_slippage_sec"] == 0
    assert row["actual_hold_seconds"] == 1800
    assert str(row["contract_version"]) == "EXACT_T2_30M_V36"

    bars = synthetic_bars()
    bars = bars[~bars.timestamp.isin(pd.to_datetime(["2026-08-31T14:02:00Z", "2026-08-31T14:03:00Z"]))]
    rejected, rejected_reason = watcher.label_event(source, bars, "TEST_MASSIVE_1M")
    assert rejected is None
    assert rejected_reason == "ENTRY_TIMING_MISS"


def test_exact_20_row_checkpoint_is_immutable(tmp_path: Path) -> None:
    root = prepare_workspace(tmp_path)
    policy, _, policy_sha = role_freeze.load_policy(root)
    rows = pd.DataFrame([
        {
            "prospective_observation_order": index,
            "event_id": f"SEC:CP:{index:03d}",
            "data_role": role_freeze.ROLE,
            "role_policy_sha256": policy_sha,
            "entry_slippage_sec": 0,
            "exit_slippage_sec": 0,
            "actual_hold_seconds": 1800,
        }
        for index in range(1, 21)
    ])
    created = watcher.create_checkpoints(root, rows, policy, policy_sha)
    verified = watcher.create_checkpoints(root, rows, policy, policy_sha)
    manifest_path = root / "research/V224_PROSPECTIVE_CONFIRMATION/checkpoints/CHECKPOINT_020.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert created[0]["action"] == "CREATED"
    assert verified[0]["action"] == "VERIFIED_EXISTING"
    assert manifest["rows"] == 20
    assert manifest["cumulative_rows"] == 20
    assert manifest["block_start_observation_order"] == 1
    assert manifest["block_end_observation_order"] == 20
    assert manifest["protocol_change_authority"] is False
    assert manifest["seal_or_final_open_authority"] is False
    assert manifest["v224_evaluation_run_by_watcher"] is False
    assert not (root / "research/V224_PROSPECTIVE_CONFIRMATION/checkpoints/CHECKPOINT_040.json").exists()
