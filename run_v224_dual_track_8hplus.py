#!/usr/bin/env python3
"""Resume-safe V224 dual-track scheduler.

Only successful data/research command runtime is credited as useful runtime.
Sleeping, integrity checks, and status rendering are never credited.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "research" / "V224_DUAL_TRACK_8HPLUS_POLICY_V1.json"
STATE_PATH = ROOT / "state" / "DUAL_TRACK_RUNTIME_STATE.json"
LOCK_PATH = ROOT / "state" / "V224_DUAL_TRACK_8HPLUS.lock.json"
LOG_PATH = ROOT / "logs" / "v224_dual_track_8hplus.log"
SNAPSHOT_PATH = ROOT / "research" / "V224_DUAL_TRACK_RUNTIME_SNAPSHOT.json"
FINAL_JSON_PATH = ROOT / "research" / "V224_DUAL_TRACK_8HPLUS_FINAL_REPORT.json"
FINAL_MD_PATH = ROOT / "research" / "V224_DUAL_TRACK_8HPLUS_FINAL_REPORT.md"
TRACK_A_STATE_PATH = ROOT / "state" / "V224_PROSPECTIVE_STATE.json"
TRACK_A_STATUS_PATH = ROOT / "research" / "V224_PROSPECTIVE_CAUSAL_CONFIRMATION" / "TRACK_A_STATUS.json"
TRACK_B_RESEARCH_STATE_PATH = ROOT / "state" / "HIGH_CONFIDENCE_NEWS_RESEARCH_STATE.json"
ET = ZoneInfo("America/New_York")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{iso_now()} {message}\n")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        try:
            old = load_json(LOCK_PATH)
            old_pid = int(old.get("pid", 0))
        except Exception:
            old_pid = 0
        if pid_alive(old_pid):
            raise RuntimeError(f"dual-track scheduler already running as PID {old_pid}")
    atomic_json(LOCK_PATH, {"pid": os.getpid(), "started_at_utc": iso_now()})


def release_lock() -> None:
    if not LOCK_PATH.exists():
        return
    try:
        lock = load_json(LOCK_PATH)
        if int(lock.get("pid", -1)) != os.getpid():
            return
        LOCK_PATH.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def observed_fixed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def nth_weekday(year: int, month: int, weekday: int, ordinal: int) -> date:
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + timedelta(days=delta + 7 * (ordinal - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def easter_sunday(year: int) -> date:
    # Gregorian computus (Meeus/Jones/Butcher).
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def nyse_full_holidays(year: int) -> set[date]:
    holidays = {
        observed_fixed_holiday(date(year, 1, 1)),
        nth_weekday(year, 1, 0, 3),       # MLK
        nth_weekday(year, 2, 0, 3),       # Washington's Birthday
        easter_sunday(year) - timedelta(days=2),
        last_weekday(year, 5, 0),         # Memorial Day
        observed_fixed_holiday(date(year, 7, 4)),
        nth_weekday(year, 9, 0, 1),       # Labor Day
        nth_weekday(year, 11, 3, 4),      # Thanksgiving
        observed_fixed_holiday(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(observed_fixed_holiday(date(year, 6, 19)))
    # A following year's New Year's Day can be observed on Dec 31.
    next_new_year = observed_fixed_holiday(date(year + 1, 1, 1))
    if next_new_year.year == year:
        holidays.add(next_new_year)
    return holidays


def is_nyse_session(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_full_holidays(day.year)


def eligible_track_a_dates(policy: dict[str, Any], state: dict[str, Any]) -> list[str]:
    now_et = datetime.now(ET)
    last_eligible = now_et.date()
    cutoff = policy["track_a"]["collection_cutoff_et"]
    cutoff_hour, cutoff_minute = (int(part) for part in cutoff.split(":"))
    if (now_et.hour, now_et.minute) < (cutoff_hour, cutoff_minute):
        last_eligible -= timedelta(days=1)
    first = date.fromisoformat(policy["track_a"]["first_prospective_date"])
    processed = set(state["track_a"].get("processed_session_dates", []))
    pending: list[str] = []
    current = first
    while current <= last_eligible:
        rendered = current.isoformat()
        if is_nyse_session(current) and rendered not in processed:
            pending.append(rendered)
        current += timedelta(days=1)
    return pending


def fresh_state(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"v224-dual-{utc_now().strftime('%Y%m%dT%H%M%SZ')}",
        "status": "RUNNING",
        "started_at_utc": iso_now(),
        "updated_at_utc": iso_now(),
        "minimum_wall_runtime_seconds": policy["minimum_wall_runtime_seconds"],
        "minimum_useful_runtime_seconds": policy["minimum_useful_runtime_seconds"],
        "useful_runtime_seconds": 0.0,
        "successful_task_runs": 0,
        "failed_task_runs": 0,
        "track_a": {"processed_session_dates": [], "attempts": {}},
        "track_b": {
            "last_capture_at_utc": None,
            "capture_runs": 0,
            "research_completed": [],
        },
        "tasks": {},
        "quarantined_tasks": [],
        "last_error": None,
        "integrity": {"status": "NOT_CHECKED", "checked_at_utc": None},
        "seal_opened": False,
        "final_opened": False,
    }


def load_or_create_state(policy: dict[str, Any], resume: bool) -> dict[str, Any]:
    if resume and STATE_PATH.exists():
        state = load_json(STATE_PATH)
        state["status"] = "RUNNING"
        state["resumed_at_utc"] = iso_now()
        state["updated_at_utc"] = iso_now()
        return state
    return fresh_state(policy)


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = iso_now()
    started = datetime.fromisoformat(state["started_at_utc"])
    state["wall_runtime_seconds"] = max(
        0.0, (utc_now() - started).total_seconds()
    )
    atomic_json(STATE_PATH, state)
    snapshot = {
        key: value
        for key, value in state.items()
        if key not in {"tasks"}
    }
    snapshot["pid"] = os.getpid()
    atomic_json(SNAPSHOT_PATH, snapshot)
    sync_component_states(state)


def sync_component_states(state: dict[str, Any]) -> None:
    if TRACK_A_STATUS_PATH.is_file():
        status = load_json(TRACK_A_STATUS_PATH)
        track_a_payload = {
            "schema_version": 1,
            "updated_at_utc": iso_now(),
            "authority": "research/V224_PROSPECTIVE_CAUSAL_CONFIRMATION/TRACK_A_STATUS.json",
            "authority_sha256": sha256_file(TRACK_A_STATUS_PATH),
            "status": status.get("status", "UNKNOWN"),
            "role_rows": status.get("role_rows", 0),
            "causal_feature_rows": status.get("causal_feature_rows", 0),
            "prediction_rows": status.get("prediction_rows", 0),
            "outcome_rows_opened": status.get("outcome_rows_opened", 0),
            "completed_blocks": status.get("completed_blocks", 0),
            "prediction_before_outcomes": status.get("prediction_before_outcomes", False),
            "protocol_changed": status.get("protocol_changed", False),
            "seal_opened": False,
            "final_opened": False,
        }
    else:
        track_a_payload = {
            "schema_version": 1,
            "updated_at_utc": iso_now(),
            "status": "WAITING_ELIGIBLE_WINDOW",
            "role_rows": 0,
            "prediction_rows": 0,
            "outcome_rows_opened": 0,
            "prediction_before_outcomes": True,
            "protocol_changed": False,
            "seal_opened": False,
            "final_opened": False,
        }
    atomic_json(TRACK_A_STATE_PATH, track_a_payload)

    research_state: dict[str, Any] = {
        "schema_version": 1,
        "updated_at_utc": iso_now(),
        "status": "BOUNDED_RESEARCH_ACTIVE",
        "completed_experiments": state["track_b"].get("research_completed", []),
        "prospective_capture_runs": state["track_b"].get("capture_runs", 0),
        "promotion_allowed": False,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
    }
    report_path = ROOT / "research" / "v224_news_confidence" / "V228_POLARITY_AUDIT" / "V228_POLARITY_REPORT.json"
    if report_path.is_file():
        research_state["latest_report"] = str(report_path.relative_to(ROOT)).replace("\\", "/")
        research_state["latest_report_sha256"] = sha256_file(report_path)
    atomic_json(TRACK_B_RESEARCH_STATE_PATH, research_state)


def verify_integrity(policy: dict[str, Any], state: dict[str, Any]) -> None:
    mismatches: list[dict[str, str]] = []
    for item in policy["pinned_files"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != item["sha256"]:
            mismatches.append(
                {"path": item["path"], "expected": item["sha256"], "actual": actual}
            )
    state["integrity"] = {
        "status": "PASS" if not mismatches else "FAIL",
        "checked_at_utc": iso_now(),
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError(f"frozen integrity failure: {mismatches}")


def error_signature(returncode: int, stderr: str, stdout: str) -> str:
    tail = (stderr.strip() or stdout.strip())[-2000:]
    digest = hashlib.sha256(tail.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"exit={returncode}:tail={digest}"


def expanded_command(command: list[str], substitutions: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for arg in command:
        for key, value in substitutions.items():
            arg = arg.replace("{" + key + "}", value)
        rendered.append(arg)
    return rendered


def assert_allowed_command(command: list[str], policy: dict[str, Any]) -> None:
    joined = " ".join(command).lower()
    for forbidden in policy["forbidden_command_fragments"]:
        if forbidden.lower() in joined:
            raise ValueError(f"forbidden command fragment: {forbidden}")


def run_task(
    name: str,
    command: list[str],
    policy: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    command = [sys.executable if arg == "{PYTHON}" else arg for arg in command]
    if name in set(state.get("quarantined_tasks", [])):
        append_log(f"SKIP quarantined task={name}")
        return False
    assert_allowed_command(command, policy)
    task_state = state["tasks"].setdefault(
        name,
        {
            "runs": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_same_signature": 0,
            "last_error_signature": None,
        },
    )
    task_state["runs"] += 1
    task_state["last_started_at_utc"] = iso_now()
    save_state(state)
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
        check=False,
    )
    elapsed = max(0.0, time.monotonic() - started)
    task_log_dir = ROOT / "logs" / "v224_dual_track_tasks"
    task_log_dir.mkdir(parents=True, exist_ok=True)
    task_log_path = task_log_dir / f"{name}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.log"
    task_log_path.write_text(
        f"command={json.dumps(command, ensure_ascii=False)}\n"
        f"returncode={proc.returncode}\n"
        f"elapsed_seconds={elapsed:.6f}\n"
        "--- stdout ---\n"
        f"{proc.stdout}\n"
        "--- stderr ---\n"
        f"{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    task_state["last_finished_at_utc"] = iso_now()
    task_state["last_elapsed_seconds"] = elapsed
    task_state["last_returncode"] = proc.returncode
    task_state["last_log"] = str(task_log_path.relative_to(ROOT))
    if proc.returncode == 0:
        task_state["successes"] += 1
        task_state["consecutive_same_signature"] = 0
        task_state["last_error_signature"] = None
        state["successful_task_runs"] += 1
        state["useful_runtime_seconds"] += elapsed
        append_log(f"PASS task={name} elapsed={elapsed:.3f}s")
        save_state(state)
        return True
    signature = error_signature(proc.returncode, proc.stderr, proc.stdout)
    if signature == task_state.get("last_error_signature"):
        task_state["consecutive_same_signature"] += 1
    else:
        task_state["last_error_signature"] = signature
        task_state["consecutive_same_signature"] = 1
    task_state["failures"] += 1
    state["failed_task_runs"] += 1
    state["last_error"] = {
        "task": name,
        "signature": signature,
        "at_utc": iso_now(),
        "log": task_state["last_log"],
    }
    if task_state["consecutive_same_signature"] >= 3:
        state["quarantined_tasks"] = sorted(
            set(state.get("quarantined_tasks", [])) | {name}
        )
        append_log(f"QUARANTINE task={name} signature={signature}")
    else:
        append_log(f"FAIL task={name} signature={signature}")
    save_state(state)
    return False


def should_run_capture(policy: dict[str, Any], state: dict[str, Any]) -> bool:
    last = state["track_b"].get("last_capture_at_utc")
    if not last:
        return True
    elapsed = (utc_now() - datetime.fromisoformat(last)).total_seconds()
    return elapsed >= policy["track_b"]["capture_cadence_seconds"]


def run_track_a(policy: dict[str, Any], state: dict[str, Any]) -> None:
    for session_date in eligible_track_a_dates(policy, state):
        attempt = state["track_a"]["attempts"].setdefault(
            session_date, {"completed_steps": []}
        )
        complete = True
        for index, step in enumerate(policy["track_a"]["steps"]):
            step_name = step["name"]
            if step_name in attempt["completed_steps"]:
                continue
            command = expanded_command(step["command"], {"DATE": session_date})
            task_name = f"track_a_{step_name}"
            if not run_task(task_name, command, policy, state):
                complete = False
                break
            attempt["completed_steps"].append(step_name)
            attempt["updated_at_utc"] = iso_now()
            save_state(state)
        if complete:
            state["track_a"]["processed_session_dates"] = sorted(
                set(state["track_a"].get("processed_session_dates", []))
                | {session_date}
            )
            attempt["completed_at_utc"] = iso_now()
            save_state(state)


def run_track_b(policy: dict[str, Any], state: dict[str, Any]) -> None:
    research = policy["track_b"].get("one_time_research")
    if research and research["name"] not in state["track_b"]["research_completed"]:
        report = ROOT / research["completion_report"]
        manifest = ROOT / research["completion_manifest"]
        if report.is_file() and manifest.is_file():
            state["track_b"]["research_completed"].append(research["name"])
            append_log(
                f"ADOPT preexisting research={research['name']} "
                f"report_sha={sha256_file(report)} manifest_sha={sha256_file(manifest)}"
            )
            save_state(state)
        elif report.exists() or manifest.exists():
            raise RuntimeError(
                f"partial research completion artifacts: report={report.exists()} "
                f"manifest={manifest.exists()}"
            )
        elif run_task(
            f"track_b_{research['name']}", research["command"], policy, state
        ):
            if not report.is_file() or not manifest.is_file():
                raise RuntimeError(
                    f"research command passed without report+manifest: {report}, {manifest}"
                )
            state["track_b"]["research_completed"].append(research["name"])
            save_state(state)
    if should_run_capture(policy, state):
        capture = policy["track_b"]["capture"]
        if run_task("track_b_news_capture", capture["command"], policy, state):
            state["track_b"]["last_capture_at_utc"] = iso_now()
            state["track_b"]["capture_runs"] += 1
            save_state(state)


def minimum_complete(state: dict[str, Any]) -> bool:
    return (
        float(state.get("wall_runtime_seconds", 0.0))
        >= float(state["minimum_wall_runtime_seconds"])
        and float(state.get("useful_runtime_seconds", 0.0))
        >= float(state["minimum_useful_runtime_seconds"])
    )


def write_final_report(state: dict[str, Any]) -> None:
    track_a = load_json(TRACK_A_STATE_PATH) if TRACK_A_STATE_PATH.is_file() else {}
    news_capture_path = ROOT / "state" / "NEWS_PROSPECTIVE_CAPTURE_STATE.json"
    news_capture = load_json(news_capture_path) if news_capture_path.is_file() else {}
    news_research = (
        load_json(TRACK_B_RESEARCH_STATE_PATH)
        if TRACK_B_RESEARCH_STATE_PATH.is_file()
        else {}
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": iso_now(),
        "status": state["status"],
        "run_id": state["run_id"],
        "active_wall_runtime_seconds": state.get("wall_runtime_seconds", 0.0),
        "credited_useful_runtime_seconds": state.get("useful_runtime_seconds", 0.0),
        "useful_runtime_excludes_sleep_and_reporting": True,
        "successful_task_runs": state.get("successful_task_runs", 0),
        "failed_task_runs": state.get("failed_task_runs", 0),
        "quarantined_tasks": state.get("quarantined_tasks", []),
        "track_a": track_a,
        "track_b_prospective_capture": news_capture,
        "track_b_research": news_research,
        "frozen_integrity": state.get("integrity", {}),
        "v224_protocol_changed": False,
        "research_seal_opened": False,
        "final_meta_opened": False,
        "promotion_performed": False,
    }
    atomic_json(FINAL_JSON_PATH, report)
    lines = [
        "# V224 dual-track 8h+ final runtime report",
        "",
        f"- Status: `{report['status']}`",
        f"- Run ID: `{report['run_id']}`",
        f"- Active wall runtime: `{report['active_wall_runtime_seconds']:.1f}` seconds",
        f"- Credited useful runtime: `{report['credited_useful_runtime_seconds']:.1f}` seconds",
        f"- Successful task runs: `{report['successful_task_runs']}`",
        f"- Failed task runs: `{report['failed_task_runs']}`",
        f"- Quarantined tasks: `{', '.join(report['quarantined_tasks']) or 'none'}`",
        "",
        "Useful runtime excludes scheduler sleep, integrity hashing, and report rendering.",
        "V224 remained frozen; Research Seal and Final Meta were not opened.",
        "See the companion JSON for the full Track A and Track B state snapshots.",
        "",
    ]
    FINAL_MD_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--once", action="store_true", help="run one scheduler iteration")
    args = parser.parse_args()
    policy = load_json(POLICY_PATH)
    try:
        acquire_lock()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 12
    state: dict[str, Any] | None = None
    try:
        state = load_or_create_state(policy, resume=args.resume)
        verify_integrity(policy, state)
        save_state(state)
        append_log(f"START pid={os.getpid()} run_id={state['run_id']}")
        while True:
            verify_integrity(policy, state)
            run_track_a(policy, state)
            run_track_b(policy, state)
            save_state(state)
            if minimum_complete(state):
                state["status"] = "MINIMUM_8H_PRODUCTIVE_RUNTIME_COMPLETE"
                state["completed_at_utc"] = iso_now()
                save_state(state)
                write_final_report(state)
                append_log("COMPLETE minimum wall and useful runtime satisfied")
                return 0
            if args.once:
                append_log("ONCE iteration complete")
                return 0
            time.sleep(policy["scheduler_poll_seconds"])
    except KeyboardInterrupt:
        if state is not None:
            state["status"] = "INTERRUPTED"
            save_state(state)
        append_log("INTERRUPTED")
        return 130
    except Exception as exc:
        if state is not None:
            state["status"] = "INTEGRITY_FAILURE" if "integrity" in str(exc) else "CRASHED"
            state["last_error"] = {
                "at_utc": iso_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            save_state(state)
        append_log(f"FATAL {type(exc).__name__}: {exc}")
        return 10 if "integrity" in str(exc) else 1
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
