from __future__ import annotations

import json

import pandas as pd

import alpaca_backfill_v221 as worker
import autonomous_v71plus as controller


def configure(tmp_path, monkeypatch, frame: pd.DataFrame, targets: dict[str, int]) -> None:
    queue = tmp_path / "queue.parquet"
    parity = tmp_path / "parity.json"
    frame.to_parquet(queue, index=False)
    parity.write_text(
        json.dumps(
            {
                "backfill_authorized": True,
                "sealed_role_labels_opened": False,
                "result": {"status": "PASS", "feed": "SIP"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller, "ALPACA_QUEUE_PATH", queue)
    monkeypatch.setattr(controller, "ALPACA_PARITY_PATH", parity)
    monkeypatch.setattr(worker, "role_targets", lambda _: targets)


def row(role: str, status: str, error: str = "") -> dict[str, str]:
    return {
        "role": role,
        "exact_contract_status": "NOT_EXACT",
        "alpaca_sip_status": status,
        "last_error": error,
    }


def test_satisfied_role_stale_rows_do_not_schedule(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            row("RESEARCH_SEAL_POOL", "BLOCKED_AUTH_HTTP_401", "BLOCKED_AUTH_HTTP_401"),
            row("DEV_EXTENSION", "PROVIDER_FAILED", "ASOF_NO_BAR_OR_QA_FAIL"),
        ]
    )
    configure(
        tmp_path,
        monkeypatch,
        frame,
        {"DEV_EXTENSION": 1, "RESEARCH_SEAL_POOL": 0, "FINAL_META_RESERVE": 0},
    )
    assert controller.alpaca_backfill_command(10) is None


def test_no_bar_failure_schedules_distinct_asof_retry(tmp_path, monkeypatch):
    frame = pd.DataFrame([row("DEV_EXTENSION", "PROVIDER_FAILED", "NO_BAR_OR_QA_FAIL")])
    configure(
        tmp_path,
        monkeypatch,
        frame,
        {"DEV_EXTENSION": 1, "RESEARCH_SEAL_POOL": 0, "FINAL_META_RESERVE": 0},
    )
    command = controller.alpaca_backfill_command(10)
    assert command is not None
    assert "--retry-asof" in command


def test_epoch_gate_requires_activation_and_unopened_seals(tmp_path, monkeypatch):
    readiness = tmp_path / "readiness.json"
    monkeypatch.setattr(controller, "READINESS_PATH", readiness)
    readiness.write_text(
        json.dumps(
            {
                "status": "READY",
                "activation_contract_written": True,
                "seal_or_final_labels_opened": False,
            }
        ),
        encoding="utf-8",
    )
    assert controller.new_data_epoch_ready() is True
    readiness.write_text(
        json.dumps(
            {
                "status": "READY",
                "activation_contract_written": True,
                "seal_or_final_labels_opened": True,
            }
        ),
        encoding="utf-8",
    )
    assert controller.new_data_epoch_ready() is False
