"""Long-running V71+ acquisition/research orchestrator.

The process deliberately separates raw provider acquisition from role/label
finalization.  Massive/Alpaca/KIS workers and one material model job may run in
parallel.  Role assignment, DEV label projection, timing audit, and data-epoch
readiness are refreshed only after all provider writers are idle.

Credentials are inherited from the process environment.  This file records
only PRESENT/MISSING and never serializes values or child stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_limits
from provider_env_runtime import (
    ALPACA_ENV_NAMES,
    import_user_environment_if_missing,
    refresh_user_environment,
)


ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
RESEARCH_DIR = ROOT / "research"
STATE_PATH = STATE_DIR / "AUTONOMOUS_V71PLUS_STATE.json"
JOURNAL_PATH = STATE_DIR / "AUTONOMOUS_V71PLUS_JOURNAL.jsonl"
LOCK_PATH = STATE_DIR / "AUTONOMOUS_V71PLUS.lock"
FINAL_MARKER = STATE_DIR / "FINAL_PASS"
LEGACY_STATE = STATE_DIR / "AUTONOMOUS_LOOP_STATE.json"
LEGACY_CHAMPION = RESEARCH_DIR / "CURRENT_CHAMPION.json"
STATE_CHAMPION = STATE_DIR / "CURRENT_CHAMPION.json"
DATA_EPOCH_STATE = STATE_DIR / "CURRENT_DATA_EPOCH.json"
REGISTRY_PATH = RESEARCH_DIR / "EXPERIMENT_REGISTRY.json"
HYPOTHESIS_REGISTRY_PATH = RESEARCH_DIR / "HYPOTHESIS_REGISTRY.json"
CONTROLLER_LOCK_PATH = STATE_DIR / "AUTONOMOUS_CONTROLLER.lock"
READINESS_PATH = RESEARCH_DIR / "V70_NEW_DATA_EPOCH_READINESS.json"
US_AUDIT_PATH = ROOT / "data" / "MASSIVE_COVERAGE_AUDIT.json"
KR_AUDIT_PATH = ROOT / "data" / "KIS_HISTORICAL_COVERAGE_AUDIT.json"
US_QUEUE_PATH = ROOT / "data" / "PRICE_ACQUISITION_QUEUE_US.parquet"
KR_QUEUE_PATH = ROOT / "data" / "PRICE_ACQUISITION_QUEUE_KR.parquet"
PROVIDER = ROOT / "v59_provider_acquisition.py"
ALPACA_BACKFILL = ROOT / "alpaca_backfill_v221.py"
ALPACA_QUEUE_PATH = ROOT / "data" / "ALPACA_BACKFILL_QUEUE.parquet"
ALPACA_PARITY_PATH = RESEARCH_DIR / "PROVIDER_PARITY_SUMMARY.json"
CONTROLLER = ROOT / "autonomous_v37plus.py"
READINESS = ROOT / "audit_v70_data_epoch_readiness.py"
EXPECTED_DEV_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


@contextmanager
def exclusive_lock():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        old = read_json(LOCK_PATH, {}) or {}
        pid = int(old.get("pid", -1))
        if process_alive(pid):
            raise RuntimeError(f"V71+ orchestrator already active: pid={pid}")
        os.replace(LOCK_PATH, LOCK_PATH.with_name(f"{LOCK_PATH.name}.stale.{int(time.time())}"))
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()}).encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()


def credential_presence() -> dict[str, str]:
    massive = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    return {
        "MASSIVE_API_KEY": "PRESENT" if massive else "MISSING",
        "POLYGON_API_KEY": "PRESENT" if os.getenv("POLYGON_API_KEY") else "MISSING",
        "KIS_APP_KEY": "PRESENT" if os.getenv("KIS_APP_KEY") else "MISSING",
        "KIS_APP_SECRET": "PRESENT" if os.getenv("KIS_APP_SECRET") else "MISSING",
        "ALPACA_API_KEY": "PRESENT" if (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")) else "MISSING",
        "ALPACA_API_SECRET": "PRESENT" if (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")) else "MISSING",
    }


def pending_count(path: Path) -> int:
    """Read the queue itself; coverage JSON may predate a later queue rebuild."""
    if not path.is_file():
        return 0
    import pandas as pd

    status = pd.read_parquet(path, columns=["status"])["status"].astype(str)
    return int(status.eq("PENDING").sum())


def ready_model_job() -> bool:
    registry = read_json(REGISTRY_PATH, {}) or {}
    for job in registry.get("jobs", []):
        if job.get("status") not in {"READY", "RETRY_WAIT"}:
            continue
        runner = ROOT / str(job.get("runner", ""))
        if runner.is_file():
            return True
    return False


def new_data_epoch_ready() -> bool:
    readiness = read_json(READINESS_PATH, {}) or {}
    return (
        readiness.get("status") == "READY"
        and readiness.get("activation_contract_written") is True
        and readiness.get("seal_or_final_labels_opened") is False
    )


def alpaca_backfill_command(max_requests: int) -> list[str] | None:
    """Schedule only genuinely unresolved, parity-authorized SIP work."""
    if not ALPACA_QUEUE_PATH.is_file() or not ALPACA_PARITY_PATH.is_file():
        return None
    parity = read_json(ALPACA_PARITY_PATH, {}) or {}
    result = parity.get("result", {}) or {}
    if not (
        parity.get("backfill_authorized") is True
        and parity.get("sealed_role_labels_opened") is False
        and result.get("status") == "PASS"
        and str(result.get("feed", "")).lower() == "sip"
    ):
        return None
    import pandas as pd
    import alpaca_backfill_v221 as worker

    queue = pd.read_parquet(ALPACA_QUEUE_PATH)
    targets = worker.role_targets(queue)
    exact = queue.loc[
        queue.exact_contract_status.astype(str).eq("EXACT_TIMING_ELIGIBLE"), "role"
    ].astype(str).value_counts().to_dict()
    needed_roles = {
        role for role, target in targets.items() if int(exact.get(role, 0)) < target
    }
    if not needed_roles:
        return None
    candidates = queue.loc[queue.role.astype(str).isin(needed_roles)]
    status = candidates.alpaca_sip_status.astype(str)
    settled = {
        "CACHED_VALID", "FETCHED_VALID", "PROVIDER_FAILED",
        "DEFERRED_OUTSIDE_ALPACA_COVERAGE_ANCHOR",
    }
    if (~status.isin(settled)).any():
        return [sys.executable, str(ALPACA_BACKFILL), "--feed", "sip",
                "--max-requests", str(max_requests)]
    retryable = status.eq("PROVIDER_FAILED") & candidates.last_error.astype(str).eq("NO_BAR_OR_QA_FAIL")
    if retryable.any():
        return [sys.executable, str(ALPACA_BACKFILL), "--feed", "sip", "--retry-asof",
                "--max-requests", str(max_requests)]
    return None


def process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def recover_stale_model_jobs() -> None:
    """Release controller claims after an audited child failure/crash.

    The model controller normally repairs CLAIMED/RUNNING jobs on its next
    initialization.  The outer scheduler used to launch that next controller
    only when a READY job already existed, creating a deadlock after a runner
    exception.  An actually live controller lock remains authoritative; only
    when no live controller exists are stale claims returned to READY.
    """
    lock = read_json(CONTROLLER_LOCK_PATH, {}) or {}
    if CONTROLLER_LOCK_PATH.is_file() and process_alive(lock.get("pid")):
        return
    registry = read_json(REGISTRY_PATH, {}) or {}
    changed = False
    for job in registry.get("jobs", []):
        if job.get("status") not in {"CLAIMED", "RUNNING", "COMMITTING"}:
            continue
        attempt = int(job.get("attempt_count", 0)) + 1
        job["attempt_count"] = attempt
        job["status"] = "READY" if attempt < 3 else "QUARANTINED"
        job["recovered_at"] = utc_now()
        changed = True
    if changed:
        atomic_json(registry, REGISTRY_PATH)


def sync_hypothesis_registry() -> None:
    """Project controller job/result state into the richer V71+ registry."""
    hypotheses = read_json(HYPOTHESIS_REGISTRY_PATH, {}) or {}
    experiments = read_json(REGISTRY_PATH, {}) or {}
    if not isinstance(hypotheses.get("hypotheses"), list):
        return
    jobs = {str(row.get("job_id")): row for row in experiments.get("jobs", [])}
    changed = False
    for hypothesis in hypotheses["hypotheses"]:
        job = jobs.get(str(hypothesis.get("id")))
        if not job:
            continue
        status = str(job.get("status", hypothesis.get("status", "READY")))
        if hypothesis.get("status") != status:
            hypothesis["status"] = status
            changed = True
        version = job.get("assigned_version")
        report_path = ROOT / f"output_V{version}" / "DEV_ROBUSTNESS_REPORT.json" if version is not None else None
        if status == "COMMITTED" and report_path is not None and report_path.is_file():
            report = read_json(report_path, {}) or {}
            model_report = read_json(
                report_path.parent / "MODEL_COMPARISON.json", {}
            ) or {}
            selected = report.get("selected", {}) or {}
            metrics = selected.get("metrics", {}) or {}
            robustness = selected.get("robustness", {}) or {}
            material = selected.get("material_gate", {}) or {}
            report_status = str(
                report.get("status")
                or (model_report.get("evaluation", {}) or {}).get("status")
                or ""
            )
            # A fail-closed report deliberately selects the older champion,
            # whose own historical material_gate may be true.  That inherited
            # gate is not evidence that the current hypothesis passed.
            current_material_pass = material.get("material_pass")
            if "FAIL" in report_status.upper() or any(
                report.get(name) is True
                for name in (
                    "fallback_is_exact_v58", "fallback_is_exact_v69",
                    "fallback_is_exact_champion",
                )
            ):
                current_material_pass = False
            result = {
                "version": int(version),
                "status": report_status or None,
                "material_pass": current_material_pass,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "auc": metrics.get("auc"),
                "edge": metrics.get("edge_vs_naive"),
                "highconf_accuracy": metrics.get("highconf_accuracy"),
                "highconf_net": metrics.get("strategy_mean_signed_net"),
                "bootstrap_highconf_net_lower95": (
                    robustness.get("bootstrap", {}) or {}
                ).get("highconf_strategy_net_lower95"),
                "canonical_gate_count": (selected.get("research_gate", {}) or {}).get("passed"),
                "robust_survivor": (selected.get("research_gate", {}) or {}).get("robust_survivor"),
                "report_sha256": sha256(report_path),
            }
            if hypothesis.get("last_result") != result:
                hypothesis["last_result"] = result
                changed = True
    if changed:
        hypotheses["updated_at"] = utc_now()
        atomic_json(hypotheses, HYPOTHESIS_REGISTRY_PATH)


def start_child(name: str, command: list[str]) -> tuple[str, subprocess.Popen[bytes], float]:
    # Child stdout is discarded.  Provider/model artifacts are the audit
    # authority and this prevents credentials embedded in unexpected library
    # exceptions from ever entering a persistent orchestrator log.
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return name, process, time.monotonic()


def wait_children(children: list[tuple[str, subprocess.Popen[bytes], float]]) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    useful_start = min((started for _, _, started in children), default=time.monotonic())
    for name, process, started in children:
        code = process.wait()
        results.append({
            "name": name,
            "pid": process.pid,
            "exit_code": code,
            "duration_seconds": round(time.monotonic() - started, 3),
        })
    useful_wall = max(0.0, time.monotonic() - useful_start) if children else 0.0
    return results, useful_wall


def run_quiet(name: str, command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {"name": name, "exit_code": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3)}


def verify_immutable_dev() -> None:
    path = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
    if not path.is_file() or sha256(path) != EXPECTED_DEV_SHA256:
        raise RuntimeError("immutable V36 DEV authority changed")
    if (ROOT / "cert" / "V36").exists():
        raise RuntimeError("cert/V36 unexpectedly exists")


def refresh_state_projections(state: dict[str, Any]) -> None:
    sync_hypothesis_registry()
    champion = read_json(LEGACY_CHAMPION, {}) or {}
    if champion:
        atomic_json({**champion, "source": str(LEGACY_CHAMPION.relative_to(ROOT))}, STATE_CHAMPION)
    readiness = read_json(READINESS_PATH, {}) or {}
    atomic_json({
        "updated_at_utc": utc_now(),
        "status": readiness.get("status", "UNKNOWN"),
        "activation_contract_written": bool(readiness.get("activation_contract_written", False)),
        "seal_or_final_labels_opened": bool(readiness.get("seal_or_final_labels_opened", False)),
        "counts": readiness.get("counts", {}),
        "checks": readiness.get("checks", {}),
        "blocking_reasons": readiness.get("blocking_reasons", []),
        "source": str(READINESS_PATH.relative_to(ROOT)),
    }, DATA_EPOCH_STATE)
    legacy = read_json(LEGACY_STATE, {}) or {}
    state["current_version"] = legacy.get("current_version")
    state["last_completed_version"] = legacy.get("last_completed_version")
    state["current_champion"] = champion
    state["data_epoch"] = read_json(DATA_EPOCH_STATE, {})


def initialize(min_useful_hours: float) -> dict[str, Any]:
    state = read_json(STATE_PATH, {}) or {}
    state.setdefault("schema_version", 1)
    state.setdefault("session_start_utc", utc_now())
    state.setdefault("min_useful_runtime_hours", float(min_useful_hours))
    state.setdefault("useful_runtime_seconds", 0.0)
    state.setdefault("cycles", 0)
    state.setdefault("phase", "INITIALIZING")
    state.setdefault("final_pass", False)
    state.setdefault("last_actions", [])
    state.setdefault("last_progress_at_utc", state["session_start_utc"])
    return state


def checkpoint(state: dict[str, Any], event: str, detail: Any = None) -> None:
    refresh_state_projections(state)
    state["updated_at_utc"] = utc_now()
    state["credentials"] = credential_presence()
    state["runtime"] = runtime_limits.status()
    state["useful_runtime_hours"] = float(state.get("useful_runtime_seconds", 0.0)) / 3600.0
    state["minimum_useful_runtime_met"] = (
        state["useful_runtime_hours"] >= float(state["min_useful_runtime_hours"])
    )
    atomic_json(state, STATE_PATH)
    append_jsonl({"timestamp_utc": utc_now(), "event": event, "detail": detail or {},
                  "phase": state.get("phase"), "useful_runtime_hours": state["useful_runtime_hours"]},
                 JOURNAL_PATH)


def finalize_acquisition() -> list[dict[str, Any]]:
    commands = [
        ("queue_and_role_refresh", [sys.executable, str(PROVIDER), "--build-queues", "--assign-roles"]),
        ("dev_and_timing_refresh", [sys.executable, str(PROVIDER), "--build-dev-labels", "--audit-timing"]),
        ("data_epoch_readiness", [sys.executable, str(READINESS)]),
    ]
    results = []
    for name, command in commands:
        result = run_quiet(name, command)
        results.append(result)
        if result["exit_code"] != 0:
            break
    return results


def run_cycle(
    state: dict[str, Any], massive_batch: int, alpaca_batch: int, kis_batch: int
) -> dict[str, Any]:
    verify_immutable_dev()
    recover_stale_model_jobs()
    presence = credential_presence()
    children: list[tuple[str, subprocess.Popen[bytes], float]] = []
    acquisition_started = False

    if presence["MASSIVE_API_KEY"] == "PRESENT" and pending_count(US_QUEUE_PATH) > 0:
        children.append(start_child("massive", [sys.executable, str(PROVIDER), "--fetch-massive",
                                                  "--max-requests", str(massive_batch)]))
        acquisition_started = True
    if (presence["KIS_APP_KEY"] == "PRESENT" and presence["KIS_APP_SECRET"] == "PRESENT"
            and pending_count(KR_QUEUE_PATH) > 0):
        children.append(start_child("kis", [sys.executable, str(PROVIDER), "--fetch-kis",
                                              "--max-requests", str(kis_batch)]))
        acquisition_started = True
    if presence["ALPACA_API_KEY"] == "PRESENT" and presence["ALPACA_API_SECRET"] == "PRESENT":
        alpaca_command = alpaca_backfill_command(alpaca_batch)
        if alpaca_command is not None:
            children.append(start_child("alpaca", alpaca_command))
            acquisition_started = True
    # The prompt's data-first contract forbids OLD-DEV model churn until the
    # fresh epoch is fully activated.
    if new_data_epoch_ready() and ready_model_job():
        children.append(start_child("model", [sys.executable, str(CONTROLLER), "--resume",
                                                "--continue-until-pass", "--max-completed-versions", "1"]))

    state["phase"] = "PARALLEL_WORK" if children else "WAITING_EXTERNAL_OR_NEW_HYPOTHESIS"
    state["active_children"] = [{"name": name, "pid": process.pid} for name, process, _ in children]
    checkpoint(state, "cycle_started", {"children": state["active_children"]})
    results, useful_wall = wait_children(children)
    state["active_children"] = []
    state["useful_runtime_seconds"] = float(state.get("useful_runtime_seconds", 0.0)) + useful_wall
    finalization: list[dict[str, Any]] = []
    if acquisition_started:
        state["phase"] = "DATA_FINALIZATION"
        checkpoint(state, "provider_workers_idle", {"results": results})
        finalization = finalize_acquisition()
        state["useful_runtime_seconds"] += sum(float(row["duration_seconds"]) for row in finalization)
    state["cycles"] = int(state.get("cycles", 0)) + 1
    state["last_actions"] = results + finalization
    if children or finalization:
        state["last_progress_at_utc"] = utc_now()
    state["phase"] = "CHECKPOINTED"
    checkpoint(state, "cycle_completed", {"actions": state["last_actions"]})
    return {"children": results, "finalization": finalization, "useful_wall_seconds": useful_wall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-until-pass", action="store_true")
    parser.add_argument("--min-useful-hours", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--massive-batch", type=int, default=300)
    parser.add_argument("--alpaca-batch", type=int, default=300)
    parser.add_argument("--kis-batch", type=int, default=500)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import_user_environment_if_missing()
    # User-scope Alpaca values are authoritative after credential rotation.
    refresh_user_environment(ALPACA_ENV_NAMES)
    runtime_limits.configure()
    with exclusive_lock():
        state = initialize(args.min_useful_hours)
        checkpoint(state, "session_started")
        while True:
            result = run_cycle(state, args.massive_batch, args.alpaca_batch, args.kis_batch)
            if FINAL_MARKER.is_file():
                verifier = subprocess.run(
                    [sys.executable, "-c", "import autonomous_v37plus as a,sys;sys.exit(0 if a.verify_final_marker() else 1)"],
                    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if verifier.returncode == 0:
                    state["phase"] = "FINAL_PASS"
                    state["final_pass"] = True
                    checkpoint(state, "final_pass_verified")
                    return 0
            if args.once or not args.continue_until_pass:
                return 0
            if not result["children"] and not result["finalization"]:
                state["phase"] = "WAITING_EXTERNAL_OR_NEW_HYPOTHESIS"
                checkpoint(state, "no_material_action_available")
            time.sleep(max(10.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
