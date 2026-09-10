"""Low-load autonomous loop for the isolated V224 prospective track only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
FIRST_DATE = date(2026, 8, 31)
STATE_DIR = ROOT / "state"
LOG_DIR = ROOT / "logs"
LOCK_PATH = STATE_DIR / "V224_AUTONOMOUS_PROSPECTIVE.lock.json"
STATUS_PATH = STATE_DIR / "V224_AUTONOMOUS_PROSPECTIVE_STATUS.json"
LOG_PATH = LOG_DIR / "v224_autonomous_prospective.log"
WATCH_STATUS = ROOT / "research/V224_PROSPECTIVE_CONFIRMATION/WATCH_STATUS.json"
FORBIDDEN_COMMAND_FRAGMENTS = (
    "v59_provider_acquisition", "assign-roles", "build-dev-labels",
    "activate_v221", "research_seal", "final_meta", "refresh_current_data_authority",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            previous = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if pid_alive(int(previous.get("pid", -1))):
                raise SystemExit(0)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    atomic_json({"pid": os.getpid(), "started_at_utc": now_utc()}, LOCK_PATH)


def append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def assert_isolated_command(arguments: list[str]) -> None:
    joined = " ".join(arguments).lower()
    bad = [fragment for fragment in FORBIDDEN_COMMAND_FRAGMENTS if fragment in joined]
    if bad:
        raise RuntimeError("V224_LOOP_FORBIDDEN_COMMAND:" + ",".join(bad))


def run_command(arguments: list[str], timeout: int = 3600) -> int:
    assert_isolated_command(arguments)
    append_log("RUN " + " ".join(arguments))
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, "-B", *arguments], cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
            timeout=timeout, check=False,
        )
    append_log(f"EXIT {completed.returncode} " + " ".join(arguments))
    return int(completed.returncode)


def due_business_days(now_et: datetime, processed: set[str]) -> list[str]:
    last_collectable = now_et.date() if now_et.time() >= datetime.strptime("16:20", "%H:%M").time() else now_et.date() - timedelta(days=1)
    values: list[str] = []
    day = FIRST_DATE
    while day <= last_collectable:
        key = day.isoformat()
        if day.weekday() < 5 and key not in processed:
            values.append(key)
        day += timedelta(days=1)
    return values


def next_run_at(now_et: datetime) -> str:
    candidate = now_et.replace(hour=16, minute=20, second=0, microsecond=0)
    if candidate <= now_et:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.isoformat()


def exact_rows() -> int:
    if not WATCH_STATUS.exists():
        return 0
    try:
        return int(json.loads(WATCH_STATUS.read_text(encoding="utf-8")).get("exact_rows", 0))
    except Exception:
        return 0


def cycle(day: str, max_price_requests: int) -> tuple[bool, str]:
    steps = [
        ["collect_v224_prospective_sec.py", "--date", day],
        ["v224_prospective_role_freeze.py"],
        ["v224_prospective_checkpoint_watcher.py", "--collect-prices", "--max-requests", str(max_price_requests)],
    ]
    evaluator = ROOT / "v224_prospective_locked_evaluation.py"
    if evaluator.exists():
        steps.append([evaluator.name])
    for step in steps:
        code = run_command(step)
        if code != 0:
            return False, step[0]
    return True, "COMPLETE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-price-requests", type=int, default=200)
    parser.add_argument("--one-shot", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.max_price_requests < 0:
        raise SystemExit("invalid loop limits")
    acquire_lock()
    processed: set[str] = set()
    if STATUS_PATH.exists():
        try:
            processed = set(json.loads(STATUS_PATH.read_text(encoding="utf-8")).get("processed_days_et", []))
        except Exception:
            processed = set()
    try:
        append_log(f"START pid={os.getpid()}")
        while True:
            rows = exact_rows()
            if rows >= 100:
                atomic_json({
                    "schema_version": 1, "track": "V224_PROSPECTIVE_CONFIRMATION",
                    "state": "TARGET_100_REACHED_LOOP_STOPPED", "pid": os.getpid(),
                    "exact_rows": rows, "processed_days_et": sorted(processed),
                    "updated_at_utc": now_utc(), "seal_or_final_opened": False,
                }, STATUS_PATH)
                return 0
            now_et = datetime.now(ET)
            due = due_business_days(now_et, processed)
            last_error = None
            if due:
                day = due[0]
                ok, failed_step = cycle(day, args.max_price_requests)
                if ok:
                    processed.add(day)
                else:
                    last_error = f"{day}:{failed_step}"
            state = "WAITING_FOR_NEXT_US_COLLECTION_WINDOW" if not last_error else "CYCLE_FAILED_WILL_RETRY"
            atomic_json({
                "schema_version": 1, "track": "V224_PROSPECTIVE_CONFIRMATION",
                "state": state, "pid": os.getpid(), "exact_rows": exact_rows(),
                "processed_days_et": sorted(processed), "last_error": last_error,
                "next_run_at_et": next_run_at(datetime.now(ET)),
                "poll_seconds": args.poll_seconds, "max_price_requests_per_cycle": args.max_price_requests,
                "central_role_or_epoch_commands_allowed": False,
                "seal_or_final_opened": False, "credentials_logged": False,
                "updated_at_utc": now_utc(),
            }, STATUS_PATH)
            if args.one_shot:
                return 0 if not last_error else 2
            time.sleep(args.poll_seconds)
    finally:
        try:
            if LOCK_PATH.exists():
                LOCK_PATH.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
