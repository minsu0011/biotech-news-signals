"""Resume-safe data-only orchestrator for historical high-confidence news.

This module intentionally has no model/training entry points.  It only invokes
the three allow-listed historical-news data programs in this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "state/HIST_NEWS_BACKFILL_STATE.json"
LOCK_PATH = ROOT / "state/HIST_NEWS_BACKFILL.lock.json"
STOP_PATH = ROOT / "state/HIST_NEWS_BACKFILL.STOP"
POLICY_PATH = ROOT / "research/HIST_NEWS_BACKFILL_POLICY_V1.json"
LOG_PATH = ROOT / "logs/historical_news_10hplus.log"
CHECKPOINT_DIR = ROOT / "reports/HIST_NEWS_BACKFILL_CHECKPOINTS"
BACKFILL_DIR = ROOT / "data/historical_news_backfill"
FROZEN_DIR = ROOT / "data/HIST_HIGH_CONF_NEWS_FROZEN_V1"

ALLOWLIST = {
    "inventory": "historical_news_event_inventory.py",
    "articles": "historical_news_article_backfill.py",
    "embeddings": "historical_news_embedding_cache.py",
    "freeze": "historical_news_build_freeze.py",
}
CODE_FILES = (
    "historical_news_event_inventory.py",
    "historical_news_article_backfill.py",
    "historical_news_build_freeze.py",
    "historical_news_embedding_cache.py",
    "historical_news_10h_orchestrator.py",
    "run_historical_news_10hplus.ps1",
    "runtime_limits.py",
)
FORBIDDEN_COMMAND_FRAGMENTS = (
    "model",
    "train",
    "oof",
    "hyperparameter",
    "research_seal",
    "final_meta",
    "v224",
    "assign-roles",
    "build-dev-labels",
    "refresh_current_data_authority",
)
PROTECTED_FILES = (
    "state/CURRENT_DATA_EPOCH.json",
    "data/V59_DATA_ROLE_ASSIGNMENT.parquet",
    "data/PRICE_ACQUISITION_QUEUE_US.parquet",
)
PROTECTED_TREES = (
    "cert",
    "research/V224_PROSPECTIVE_CONFIRMATION",
    "research/V224_PROSPECTIVE_CAUSAL_CONFIRMATION",
    "research/v224_news_confidence",
)
WORK_SUFFIXES = (
    ".parquet",
    ".csv",
    ".csv.gz",
    ".jsonl",
    ".jsonl.gz",
    ".html.gz",
    ".json.gz",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def protected_snapshot() -> dict[str, str]:
    files: set[Path] = set()
    for relative in PROTECTED_FILES:
        candidate = ROOT / relative
        if candidate.is_file():
            files.add(candidate)
    for relative in PROTECTED_TREES:
        directory = ROOT / relative
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file())
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(files, key=lambda item: item.as_posix().lower())
    }


def initialize_policy() -> dict[str, Any]:
    if POLICY_PATH.exists():
        return load_json(POLICY_PATH, {})
    snapshot = protected_snapshot()
    policy = {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "purpose": "HISTORICAL_NEWS_DATA_ONLY_NO_MODEL_BUILDING",
        "allowed_scripts": sorted(ALLOWLIST.values()),
        "allowed_code_sha256": {
            name: sha256(ROOT / name) for name in CODE_FILES if (ROOT / name).is_file()
        },
        "forbidden_command_fragments": list(FORBIDDEN_COMMAND_FRAGMENTS),
        "protected_sha256": snapshot,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "label_membership_allowed_before_source_freeze": False,
    }
    atomic_json(policy, POLICY_PATH)
    return policy


def validate_policy() -> tuple[bool, list[str]]:
    policy = load_json(POLICY_PATH, {})
    expected = policy.get("protected_sha256", {})
    if not isinstance(expected, dict) or not expected:
        return False, ["PROTECTED_BASELINE_MISSING"]
    current = protected_snapshot()
    failures: list[str] = []
    for relative, expected_hash in expected.items():
        observed = current.get(str(relative))
        if observed != expected_hash:
            failures.append(f"PROTECTED_HASH_MISMATCH:{relative}")
    unexpected = sorted(set(current) - set(expected))
    failures.extend(f"PROTECTED_TREE_NEW_FILE:{relative}" for relative in unexpected)
    expected_code = policy.get("allowed_code_sha256", {})
    if not isinstance(expected_code, dict) or set(expected_code) != set(CODE_FILES):
        failures.append("ALLOWED_CODE_BASELINE_MISSING_OR_INCOMPLETE")
    else:
        for name, expected_hash in expected_code.items():
            path = ROOT / str(name)
            if not path.is_file() or sha256(path) != expected_hash:
                failures.append(f"ALLOWED_CODE_HASH_MISMATCH:{name}")
    return not failures, failures


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = load_json(LOCK_PATH, {}) if LOCK_PATH.exists() else {}
    previous_pid = int(previous.get("pid", -1)) if isinstance(previous, dict) else -1
    if pid_alive(previous_pid):
        raise SystemExit(12)
    atomic_json({"pid": os.getpid(), "started_at_utc": now_utc()}, LOCK_PATH)


def release_lock() -> None:
    current = load_json(LOCK_PATH, {})
    if isinstance(current, dict) and int(current.get("pid", -1)) == os.getpid():
        try:
            LOCK_PATH.unlink()
        except OSError:
            pass


def default_state(target_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_start": now_utc(),
        "wall_start_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "state": "INITIALIZING",
        "pid": os.getpid(),
        "productive_seconds": 0.0,
        "target_productive_seconds": float(target_seconds),
        "current_phase": "WORKSPACE_AUDIT",
        "events_total": 0,
        "events_processed": 0,
        "articles_discovered": 0,
        "articles_downloaded": 0,
        "articles_parsed": 0,
        "causal_tier_a": 0,
        "causal_tier_b": 0,
        "aux_tier_c": 0,
        "pending_sources": [],
        "failed_sources": [],
        "last_checkpoint": "CHECKPOINT_00",
        "completed_checkpoints": [],
        "active_task": None,
        "task_history": [],
        "consecutive_no_progress_cycles": 0,
        "source_frozen": False,
        "label_joined_after_source_freeze": False,
        "model_building_forbidden": True,
        "model_building_performed": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "protected_hash_audit": "PENDING",
    }


def read_state(target_seconds: float) -> dict[str, Any]:
    existing = load_json(STATE_PATH, {})
    state = default_state(target_seconds)
    if isinstance(existing, dict):
        state.update(existing)
    state.setdefault("session_start", now_utc())
    state.setdefault("productive_seconds", 0.0)
    state["target_productive_seconds"] = float(target_seconds)
    state["pid"] = os.getpid()
    state["updated_at_utc"] = now_utc()
    state["model_building_forbidden"] = True
    state["model_building_performed"] = False
    state["research_seal_touched"] = False
    state["final_meta_touched"] = False
    state["v224_touched"] = False
    return state


def save_state(state: dict[str, Any]) -> None:
    updated = now_utc()
    state["updated_at_utc"] = updated
    try:
        start = datetime.fromisoformat(str(state.get("session_start", "")).replace("Z", "+00:00"))
        end = datetime.fromisoformat(updated)
        state["actual_wall_seconds"] = max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        state["actual_wall_seconds"] = None
    atomic_json(state, STATE_PATH)


def work_signature(paths: Iterable[Path] = (BACKFILL_DIR, FROZEN_DIR)) -> str:
    records: list[str] = []
    for directory in paths:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not any(name.endswith(suffix) for suffix in WORK_SUFFIXES):
                continue
            stat = path.stat()
            records.append(f"{path.relative_to(ROOT).as_posix()}|{stat.st_size}")
    return hashlib.sha256("\n".join(sorted(records)).encode("utf-8")).hexdigest()


def source_input_signature() -> str:
    digest = hashlib.sha256()
    for path in (
        BACKFILL_DIR / "EVENT_MASTER.parquet",
        BACKFILL_DIR / "ARTICLES_NORMALIZED.parquet",
        BACKFILL_DIR / "embeddings/EMBEDDING_INDEX.parquet",
        BACKFILL_DIR / "embeddings/EMBEDDING_MODEL_FREEZE.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((sha256(path) if path.is_file() else "MISSING").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_allowed(arguments: list[str]) -> None:
    if not arguments or Path(arguments[0]).name not in ALLOWLIST.values():
        raise RuntimeError("COMMAND_NOT_ALLOWLISTED")
    joined = " ".join(arguments).lower()
    bad = [fragment for fragment in FORBIDDEN_COMMAND_FRAGMENTS if fragment in joined]
    if bad:
        raise RuntimeError("FORBIDDEN_COMMAND_FRAGMENT:" + ",".join(bad))


def run_task(
    name: str,
    arguments: list[str],
    state: dict[str, Any],
    *,
    timeout: int,
    count_if_input_changed: bool = False,
) -> tuple[bool, bool]:
    assert_allowed(arguments)
    ok, failures = validate_policy()
    if not ok:
        state["state"] = "FAIL_CLOSED_PROTECTED_HASH_MISMATCH"
        state["protected_hash_audit"] = failures
        save_state(state)
        raise RuntimeError(";".join(failures))
    before = "" if name in {"articles", "embeddings"} or count_if_input_changed else work_signature()
    input_signature = source_input_signature() if count_if_input_changed else ""
    productive_before = float(state.get("productive_seconds", 0.0))
    state["state"] = "RUNNING_DATA_ONLY"
    state["current_phase"] = name.upper()
    state["active_task"] = {"name": name, "arguments": arguments, "started_at_utc": now_utc()}
    state["protected_hash_audit"] = "PASS"
    save_state(state)
    append_log("RUN " + " ".join(arguments))
    started = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    completed = subprocess.run(
        [sys.executable, "-B", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    elapsed = max(0.0, time.monotonic() - started)
    if completed.stdout:
        append_log(completed.stdout.rstrip())
    if completed.stderr:
        append_log("STDERR " + completed.stderr.rstrip())
    after = "" if name in {"articles", "embeddings"} or count_if_input_changed else work_signature()
    changed = before != after
    summary: dict[str, Any] = {}
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            summary = candidate
            break
    # The worker atomically merges only its namespaced subsection into the
    # canonical state.  Reload it before writing root-owned accounting fields.
    worker_state = load_json(STATE_PATH, {})
    if isinstance(worker_state, dict):
        state.update(worker_state)
    if count_if_input_changed:
        signature_key = "last_preview_input_signature" if name == "preview" else "last_freeze_input_signature"
        changed = state.get(signature_key) != input_signature
    if name == "articles":
        # Snapshot/status files can be rewritten during a legitimate no-op.
        # Count time only when the bounded worker reports a useful unit.
        useful_keys = (
            "local_items_processed",
            "new_canonical_articles",
            "articles_downloaded",
            "articles_parsed_operations",
            "requests_used",
            "discovery_queries_attempted",
            "discovery_items_added",
        )
        changed = any(int(summary.get(key, 0) or 0) > 0 for key in useful_keys)
        if str(summary.get("status", "")).upper() == "NO_PROGRESS":
            changed = False
    if name == "embeddings":
        changed = any(
            int(summary.get(key, 0) or 0) > 0
            for key in ("items_processed", "new_embeddings")
        )
        if str(summary.get("status", "")).upper() == "NO_PROGRESS":
            changed = False
    successful = completed.returncode == 0
    productive = bool(successful and changed)
    state["productive_seconds"] = productive_before + (elapsed if productive else 0.0)
    history = list(state.get("task_history", []))[-199:]
    history.append(
        {
            "task": name,
            "returncode": int(completed.returncode),
            "elapsed_seconds": round(elapsed, 3),
            "productive_counted": productive,
            "artifact_signature_changed": changed,
            "finished_at_utc": now_utc(),
        }
    )
    state["task_history"] = history
    state["active_task"] = None
    if name in {"freeze", "preview"} and successful:
        signature_key = "last_preview_input_signature" if name == "preview" else "last_freeze_input_signature"
        state[signature_key] = input_signature
        if name == "preview":
            state["last_preview_productive_seconds"] = float(state.get("productive_seconds", 0.0))
    state["state"] = "RUNNING_DATA_ONLY" if successful else "TASK_FAILED_WILL_RETRY"
    if not successful:
        failures_list = list(state.get("failures", []))[-99:]
        failures_list.append({"task": name, "returncode": int(completed.returncode), "at_utc": now_utc()})
        state["failures"] = failures_list
    save_state(state)
    append_log(
        f"EXIT task={name} rc={completed.returncode} elapsed={elapsed:.3f} "
        f"productive={productive} changed={changed}"
    )
    return successful, productive


def refresh_top_level_counters(state: dict[str, Any]) -> None:
    """Project worker counters into the prompt-mandated top-level fields."""
    article = state.get("article_backfill", {})
    worker = load_json(BACKFILL_DIR / "state/ARTICLE_BACKFILL_WORKER_STATE.json", {})
    if isinstance(worker, dict) and isinstance(article, dict):
        article = {**worker, **article}
        state["article_backfill"] = article
    inventory = state.get("event_inventory", {})
    if isinstance(inventory, dict):
        state["events_total"] = int(
            inventory.get("events_total", inventory.get("canonical_events", state.get("events_total", 0))) or 0
        )
    if isinstance(article, dict):
        mappings = {
            "events_processed": ("events_processed", "local_items_processed_total"),
            "articles_discovered": ("articles_discovered",),
            "articles_downloaded": ("articles_downloaded",),
            "articles_parsed": ("articles_parsed",),
            "causal_tier_a": ("causal_tier_a",),
            "causal_tier_b": ("causal_tier_b",),
            "aux_tier_c": ("aux_tier_c",),
        }
        for target, sources in mappings.items():
            for source in sources:
                if source in article:
                    state[target] = int(article.get(source, 0) or 0)
                    break
        if "pending_sources" in article:
            state["pending_sources"] = article.get("pending_sources", [])
        if "failed_sources" in article:
            state["failed_sources"] = article.get("failed_sources", [])
    embedding = load_json(BACKFILL_DIR / "embeddings/EMBEDDING_WORKER_STATE.json", {})
    if isinstance(embedding, dict):
        state["embedding_cache"] = embedding
        state["embeddings_current"] = int(embedding.get("embeddings_current", 0) or 0)
        state["embedding_backlog"] = int(embedding.get("embedding_backlog", 0) or 0)
        state["embedding_gpu_used"] = bool(embedding.get("gpu_used", False))


def write_checkpoint(state: dict[str, Any], checkpoint_number: int) -> None:
    label = f"CHECKPOINT_{checkpoint_number:02d}"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Historical News Data Session Checkpoint",
        "",
        f"- Checkpoint: `{label}`",
        f"- Updated UTC: `{now_utc()}`",
        f"- Productive runtime seconds: `{float(state.get('productive_seconds', 0.0)):.1f}`",
        f"- Events total / processed: `{state.get('events_total', 0)} / {state.get('events_processed', 0)}`",
        f"- Articles discovered / downloaded / parsed: `{state.get('articles_discovered', 0)} / {state.get('articles_downloaded', 0)} / {state.get('articles_parsed', 0)}`",
        f"- Tier A / B / C: `{state.get('causal_tier_a', 0)} / {state.get('causal_tier_b', 0)} / {state.get('aux_tier_c', 0)}`",
        "- Model building performed: `FALSE`",
        "- Research Seal touched: `FALSE`",
        "- Final Meta touched: `FALSE`",
        "- V224 touched: `FALSE`",
        "",
        "CONTINUING DATA SESSION",
        "",
    ]
    (CHECKPOINT_DIR / f"{label}.md").write_text("\n".join(lines), encoding="utf-8")
    completed = list(state.get("completed_checkpoints", []))
    if label not in completed:
        completed.append(label)
    state["completed_checkpoints"] = completed
    state["last_checkpoint"] = label
    save_state(state)
    append_log(f"===== HIST NEWS DATA SESSION {label} =====")


def maybe_checkpoints(state: dict[str, Any], interval_seconds: float) -> None:
    refresh_top_level_counters(state)
    reached = int(float(state.get("productive_seconds", 0.0)) // interval_seconds)
    for checkpoint_number in range(1, reached + 1):
        label = f"CHECKPOINT_{checkpoint_number:02d}"
        if label not in state.get("completed_checkpoints", []):
            write_checkpoint(state, checkpoint_number)


def freeze_exists() -> bool:
    required = (
        "EVENT_MASTER.parquet",
        "ARTICLES_NORMALIZED.parquet",
        "HIST_NEWS_SOURCE_FREEZE.json",
        "SHA256_MANIFEST.csv",
    )
    return all((FROZEN_DIR / name).is_file() for name in required)


def label_join_exists() -> bool:
    return all(
        (FROZEN_DIR / name).is_file()
        for name in ("HIST_HIGH_CONF_NEWS_DEV_LABELED.parquet", "POST_FREEZE_LABEL_JOIN_STATUS.json")
    )


def create_lightweight_handoff(state: dict[str, Any]) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package = ROOT / "handoff" / f"historical-news-results-{stamp}"
    package.mkdir(parents=True, exist_ok=False)
    status = dict(state)
    status["workspace_root"] = str(ROOT)
    status["frozen_dataset_path"] = str(FROZEN_DIR)
    atomic_json(status, package / "CURRENT_DATA_STATUS.json")
    copies = {
        FROZEN_DIR / "HIST_NEWS_COVERAGE_REPORT.md": "HIST_NEWS_COVERAGE_REPORT.md",
        FROZEN_DIR / "HIST_NEWS_SOURCE_FREEZE.json": "HIST_NEWS_SOURCE_FREEZE.json",
        FROZEN_DIR / "HIST_HIGH_CONF_NEWS_FEATURE_CATALOG.csv": "HIST_HIGH_CONF_NEWS_FEATURE_CATALOG.csv",
        FROZEN_DIR / "SHA256_MANIFEST.csv": "SHA256_MANIFEST.csv",
        FROZEN_DIR / "POST_FREEZE_LABEL_JOIN_STATUS.json": "POST_FREEZE_LABEL_JOIN_STATUS.json",
    }
    for source, name in copies.items():
        if source.is_file():
            shutil.copy2(source, package / name)
    start = (
        "# START HERE\n\n"
        "The historical high-confidence news data-only run is complete. No model was built.\n\n"
        f"- Workspace: `{ROOT}`\n"
        f"- Frozen dataset: `{FROZEN_DIR}`\n"
        f"- Productive seconds: `{float(state.get('productive_seconds', 0.0)):.1f}`\n"
        "- Source selection was outcome-blind and frozen before the DEV label join.\n"
        "- Research Seal, Final Meta, V224 prospective namespaces: untouched.\n"
        "- The raw article corpus is intentionally not duplicated in this handoff.\n"
    )
    (package / "00_START_HERE.md").write_text(start, encoding="utf-8")
    next_prompt = (
        "# NEXT MODEL SESSION PROMPT\n\n"
        "Use only the frozen HIST_HIGH_CONF_NEWS_FROZEN_V1 membership. Build P(correct) "
        "with strict nested/grouped chronological OOF. Any publisher/topic reliability "
        "must be past-only and train-fold-only with Bayesian shrinkage. Never alter source "
        "membership based on outcomes. Keep Research Seal and Final Meta closed. Compare "
        "STRICT_A, A_PLUS_B, MULTISOURCE_ONLY, and STRUCTURED_CONFIRM_ONLY views.\n"
    )
    (package / "NEXT_MODEL_SESSION_PROMPT.md").write_text(next_prompt, encoding="utf-8")
    zip_path = package.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    return package, zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-policy", action="store_true")
    parser.add_argument("--one-cycle", action="store_true")
    parser.add_argument("--target-productive-seconds", type=float, default=10 * 3600)
    parser.add_argument("--checkpoint-seconds", type=float, default=2 * 3600)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--max-local-items", type=int, default=2500)
    parser.add_argument("--max-discovery-queries", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--preview-seconds", type=float, default=30 * 60)
    parser.add_argument("--embedding-max-items", type=int, default=5000)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--idle-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.target_productive_seconds <= 0 or args.checkpoint_seconds <= 0:
        raise SystemExit("runtime targets must be positive")
    if (
        args.max_requests < 0
        or args.max_local_items < 0
        or args.max_discovery_queries < 0
        or args.workers < 1
        or args.preview_seconds <= 0
        or args.embedding_max_items < 1
        or args.embedding_batch_size < 1
        or args.idle_seconds < 1
    ):
        raise SystemExit("invalid request/idle limits")

    # Importing configures all logical CPUs and exposes GPU 0 as requested by
    # the workspace runtime policy.  Child processes inherit the environment.
    import runtime_limits  # noqa: F401,PLC0415

    if args.initialize_policy:
        policy = initialize_policy()
        print(json.dumps({"policy": str(POLICY_PATH), "protected_files": len(policy.get("protected_sha256", {}))}))
        if not args.resume and not args.one_cycle:
            return 0
    if not POLICY_PATH.exists():
        print("policy missing; run with --initialize-policy", file=sys.stderr)
        return 10

    acquire_lock()
    state = read_state(args.target_productive_seconds)
    try:
        ok, failures = validate_policy()
        if not ok:
            state["state"] = "FAIL_CLOSED_PROTECTED_HASH_MISMATCH"
            state["protected_hash_audit"] = failures
            save_state(state)
            return 10
        state["protected_hash_audit"] = "PASS"
        state["state"] = "RUNNING_DATA_ONLY"
        save_state(state)
        append_log(f"START pid={os.getpid()} target={args.target_productive_seconds}")

        inventory = ROOT / ALLOWLIST["inventory"]
        if not inventory.is_file():
            state["state"] = "WAITING_FOR_REQUIRED_IMPLEMENTATION"
            state["missing_script"] = inventory.name
            save_state(state)
            return 20
        if not (BACKFILL_DIR / "EVENT_MASTER.parquet").is_file():
            run_task("inventory", [inventory.name, "--resume"], state, timeout=3600)

        while float(state.get("productive_seconds", 0.0)) < args.target_productive_seconds:
            if STOP_PATH.exists():
                state["state"] = "STOPPED_BY_REQUEST"
                state["current_phase"] = "STOPPED"
                save_state(state)
                append_log("STOP sentinel observed")
                return 0
            ok, failures = validate_policy()
            if not ok:
                state["state"] = "FAIL_CLOSED_PROTECTED_HASH_MISMATCH"
                state["protected_hash_audit"] = failures
                save_state(state)
                return 10

            article_script = ROOT / ALLOWLIST["articles"]
            embedding_script = ROOT / ALLOWLIST["embeddings"]
            freeze_script = ROOT / ALLOWLIST["freeze"]
            if not article_script.is_file() or not embedding_script.is_file() or not freeze_script.is_file():
                state["state"] = "WAITING_FOR_REQUIRED_IMPLEMENTATION"
                state["missing_script"] = [
                    path.name for path in (article_script, embedding_script, freeze_script) if not path.is_file()
                ]
                save_state(state)
                return 20

            article_ok, article_progress = run_task(
                "articles",
                [
                    article_script.name,
                    "--once",
                    "--resume",
                    "--max-requests",
                    str(args.max_requests),
                    "--max-local-items",
                    str(args.max_local_items),
                    "--max-discovery-queries",
                    str(args.max_discovery_queries),
                    "--workers",
                    str(args.workers),
                ],
                state,
                timeout=7200,
            )
            if article_progress:
                state["consecutive_no_progress_cycles"] = 0
            embedding_ok, embedding_progress = run_task(
                "embeddings",
                [
                    embedding_script.name,
                    "--once",
                    "--resume",
                    "--max-items",
                    str(args.embedding_max_items),
                    "--batch-size",
                    str(args.embedding_batch_size),
                    "--device",
                    args.embedding_device,
                ],
                state,
                timeout=7200,
            )
            overall_progress = article_progress or embedding_progress
            if overall_progress:
                state["consecutive_no_progress_cycles"] = 0
            else:
                state["consecutive_no_progress_cycles"] = int(state.get("consecutive_no_progress_cycles", 0)) + 1
            save_state(state)

            preview_status = BACKFILL_DIR / "derived_preview/PREVIEW_STATUS.json"
            preview_due = (
                not preview_status.is_file()
                or float(state.get("productive_seconds", 0.0))
                - float(state.get("last_preview_productive_seconds", 0.0))
                >= args.preview_seconds
            )
            preview_ok = True
            if overall_progress and preview_due:
                preview_ok, _ = run_task(
                    "preview",
                    [freeze_script.name, "--preview", "--resume"],
                    state,
                    timeout=7200,
                    count_if_input_changed=True,
                )

            maybe_checkpoints(state, args.checkpoint_seconds)
            if args.one_cycle:
                state["state"] = "ONE_CYCLE_COMPLETE"
                save_state(state)
                return 0 if article_ok and embedding_ok and preview_ok else 20
            if not overall_progress:
                state["state"] = "WAITING_FOR_NEW_USEFUL_WORK_NOT_COUNTED"
                save_state(state)
                time.sleep(args.idle_seconds)

        # Drain any embedding lag before the immutable source freeze.
        embedding_script = ROOT / ALLOWLIST["embeddings"]
        for _ in range(10000):
            embedding_ok, embedding_progress = run_task(
                "embeddings",
                [
                    embedding_script.name,
                    "--once",
                    "--resume",
                    "--max-items",
                    str(args.embedding_max_items),
                    "--batch-size",
                    str(args.embedding_batch_size),
                    "--device",
                    args.embedding_device,
                ],
                state,
                timeout=7200,
            )
            if not embedding_ok:
                return 20
            if not embedding_progress:
                break
        else:
            state["state"] = "EMBEDDING_DRAIN_LIMIT_REACHED"
            save_state(state)
            return 20

        # The final directory is immutable, so it is created only after the
        # productive target.  Earlier batches normalize into the working area.
        freeze_script = ROOT / ALLOWLIST["freeze"]
        run_task(
            "freeze",
            [freeze_script.name, "--resume"],
            state,
            timeout=7200,
            count_if_input_changed=True,
        )
        state["source_frozen"] = freeze_exists()
        if freeze_exists():
            labels_path = ROOT / "data/dev_contract_v36_labeled.csv.gz"
            run_task(
                "label_join",
                [
                    freeze_script.name,
                    "--join-dev-labels-after-freeze",
                    "--labels-path",
                    str(labels_path),
                ],
                state,
                timeout=7200,
            )
        state["label_joined_after_source_freeze"] = label_join_exists()
        state["model_ready"] = bool(freeze_exists() and label_join_exists())
        state["state"] = "MODEL_READY_DATA_ONLY_COMPLETE" if state["model_ready"] else "POST_FREEZE_LABEL_JOIN_INCOMPLETE"
        state["current_phase"] = "DATA_ONLY_COMPLETE" if state["model_ready"] else "POST_FREEZE_QA_FAILED"
        maybe_checkpoints(state, args.checkpoint_seconds)
        if state["model_ready"] and not state.get("handoff_zip"):
            package, zip_path = create_lightweight_handoff(state)
            state["handoff_directory"] = str(package)
            state["handoff_zip"] = str(zip_path)
            state["handoff_zip_sha256"] = sha256(zip_path)
        save_state(state)
        return 0 if state["model_ready"] else 20
    except subprocess.TimeoutExpired as error:
        state["state"] = "TASK_TIMEOUT_WILL_RESUME"
        state["last_error"] = str(error)
        save_state(state)
        append_log(f"TIMEOUT {error}")
        return 20
    except Exception as error:  # fail closed; the PowerShell guardian resumes transient failures
        state["state"] = "UNEXPECTED_FAILURE_WILL_RESUME"
        state["last_error"] = f"{type(error).__name__}:{error}"
        save_state(state)
        append_log(f"ERROR {type(error).__name__}:{error}")
        return 20
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
