"""Low-load future SEC/Massive epoch watcher for the frozen Aug-Sep window."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
LOGS = ROOT / "logs"
LOCK_PATH = STATE / "V221_FUTURE_EPOCH_WATCH.lock.json"
STATUS_PATH = STATE / "V221_FUTURE_EPOCH_WATCH_STATUS.json"
LOG_PATH = LOGS / "v221_future_epoch_watch.log"
READINESS_PATH = ROOT / "research" / "V70_NEW_DATA_EPOCH_READINESS.json"
ACTIVATION_PATH = ROOT / "research" / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
PREDICTION_MANIFEST = ROOT / "research" / "zero_tune_inputs" / "V221" / "manifest.json"
POLICY_PATH = ROOT / "data" / "V221_EXPANDED_FRESH_SEC_COLLECTION_20260831_20260930_POLICY_V1.json"
START = "2026-08-31"
END = "2026-09-30"
ET = ZoneInfo("America/New_York")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.is_file():
        try:
            previous = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if pid_alive(int(previous.get("pid", -1))):
                raise SystemExit(0)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    atomic_json({"pid": os.getpid(), "started_at": now_iso()}, LOCK_PATH)


def append_log(message: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def run_command(arguments: list[str]) -> int:
    command = [sys.executable, *arguments]
    append_log("RUN " + " ".join(arguments))
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, timeout=3600, check=False,
        )
    append_log(f"EXIT {completed.returncode} " + " ".join(arguments))
    return int(completed.returncode)


def readiness() -> tuple[str, int]:
    if not READINESS_PATH.is_file():
        return "MISSING", 0
    value = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    count = int(
        (((value.get("counts") or {}).get("by_source_family") or {}).get("US_SEC") or {}).get("n", 0)
    )
    return str(value.get("status", "UNKNOWN")), count


def scheduled_days() -> list[datetime]:
    start = datetime.fromisoformat(START).date()
    end = datetime.fromisoformat(END).date()
    values: list[datetime] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            values.append(datetime(day.year, day.month, day.day, 16, 20, tzinfo=ET))
        day += timedelta(days=1)
    return values


def execute_epoch_cycle(day: str) -> bool:
    steps = [
        ["collect_fresh_sec_v221.py", "--start", START, "--end", END,
         "--expanded-universe", "--policy", str(POLICY_PATH.relative_to(ROOT))],
        ["v59_provider_acquisition.py", "--build-queues", "--assign-roles"],
        ["v59_provider_acquisition.py", "--fetch-massive", "--max-requests", "200"],
        ["alpaca_massive_failure_fallback_v221.py"],
        ["alpaca_recent_timing_fallback_v221.py"],
        ["v59_provider_acquisition.py", "--build-dev-labels", "--audit-timing"],
        ["audit_v70_data_epoch_readiness.py"],
        ["refresh_current_data_authority.py"],
        ["sync_v221_final_data_state.py"],
    ]
    for step in steps:
        code = run_command(step)
        status, count = readiness()
        atomic_json({
            "schema_version": 1, "pid": os.getpid(), "state": "RUNNING_CYCLE",
            "cycle_day_et": day, "last_step": step, "last_exit_code": code,
            "readiness_status": status, "us_sec_dev": count,
            "updated_at": now_iso(), "credentials_logged": False,
        }, STATUS_PATH)
        if code != 0:
            append_log(f"CYCLE_FAILED day={day} step={step[0]}")
            return False
    return True


def main() -> int:
    acquire_lock()
    processed: set[str] = set()
    if STATUS_PATH.is_file():
        try:
            processed = set(json.loads(STATUS_PATH.read_text(encoding="utf-8")).get("processed_days_et", []))
        except Exception:
            processed = set()
    try:
        append_log(f"WATCH_STARTED pid={os.getpid()}")
        while True:
            status, count = readiness()
            if status in {"READY_TO_FREEZE", "READY", "NEW_DATA_EPOCH_READY"} and count >= 120:
                transition_steps = []
                if status == "READY_TO_FREEZE":
                    transition_steps.extend([
                        ["refresh_current_data_authority.py"],
                        ["activate_v221_new_data_epoch.py"],
                        ["refresh_current_data_authority.py"],
                        ["sync_v221_final_data_state.py"],
                    ])
                    transition_steps.append([
                        "experiment_v221_new_data_zero_tune_and_polarity_transfer.py",
                        "--mode", "scaffold-audit",
                    ])
                transition_ok = True
                for step in transition_steps:
                    if run_command(step) != 0:
                        transition_ok = False
                        append_log(f"POST_EPOCH_TRANSITION_FAILED step={step[0]}")
                        break
                status, count = readiness()
                if transition_ok and status == "NEW_DATA_EPOCH_READY" and PREDICTION_MANIFEST.is_file():
                    evaluation_code = run_command([
                        "experiment_v221_new_data_zero_tune_and_polarity_transfer.py",
                        "--mode", "evaluate", "--prediction-manifest",
                        str(PREDICTION_MANIFEST.relative_to(ROOT)),
                    ])
                    atomic_json({
                        "schema_version": 1, "pid": os.getpid(),
                        "state": (
                            "ZERO_TUNE_TRANSFER_EVALUATED"
                            if evaluation_code == 0 else "ZERO_TUNE_TRANSFER_FAILED_CLOSED"
                        ),
                        "readiness_status": status, "us_sec_dev": count,
                        "processed_days_et": sorted(processed), "updated_at": now_iso(),
                        "prediction_manifest": str(PREDICTION_MANIFEST.relative_to(ROOT)),
                        "last_exit_code": evaluation_code, "credentials_logged": False,
                    }, STATUS_PATH)
                    return 0 if evaluation_code == 0 else 4
                atomic_json({
                    "schema_version": 1, "pid": os.getpid(),
                    "state": (
                        "NEW_DATA_EPOCH_READY_AWAITING_FROZEN_SCORER_BUNDLE"
                        if transition_ok and status == "NEW_DATA_EPOCH_READY"
                        else "POST_EPOCH_TRANSITION_FAILED"
                    ),
                    "readiness_status": status, "us_sec_dev": count,
                    "processed_days_et": sorted(processed), "updated_at": now_iso(),
                    "activation_contract_written": ACTIVATION_PATH.is_file(),
                    "next_action": (
                        "Supply the separately frozen label-blind V69 scorer bundle; "
                        "retraining to imitate V69 is forbidden by zero-tune."
                    ),
                    "credentials_logged": False,
                }, STATUS_PATH)
                append_log(f"POST_EPOCH_TRANSITION status={status} us_sec_dev={count}")
                if not transition_ok:
                    return 3
                time.sleep(60)
                continue
            now_et = datetime.now(ET)
            due = [value for value in scheduled_days() if value <= now_et and value.date().isoformat() not in processed]
            if due:
                cycle = due[0]
                day = cycle.date().isoformat()
                ok = execute_epoch_cycle(day)
                if ok:
                    processed.add(day)
                else:
                    time.sleep(60)
                    continue
            future = [value for value in scheduled_days() if value > now_et]
            if not future and not due:
                atomic_json({
                    "schema_version": 1, "pid": os.getpid(), "state": "WINDOW_EXHAUSTED_NOT_READY",
                    "readiness_status": status, "us_sec_dev": count,
                    "processed_days_et": sorted(processed), "updated_at": now_iso(),
                    "credentials_logged": False,
                }, STATUS_PATH)
                return 2
            next_run = min(future).isoformat() if future else None
            atomic_json({
                "schema_version": 1, "pid": os.getpid(), "state": "WAITING_FOR_FROZEN_FUTURE_WINDOW",
                "readiness_status": status, "us_sec_dev": count,
                "processed_days_et": sorted(processed), "next_run_at_et": next_run,
                "updated_at": now_iso(), "credentials_logged": False,
            }, STATUS_PATH)
            time.sleep(60)
    finally:
        try:
            if LOCK_PATH.is_file():
                LOCK_PATH.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
