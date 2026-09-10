"""Schema, controlled factory, and atomic V71 registry registration.

This module does not run an experiment.  It records material research
questions and can propose only a small untried information-family combination
for the currently declared source/bottleneck.  It never enumerates arbitrary
feature/threshold/weight products.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research"
HYPOTHESIS_PATH = RESEARCH / "HYPOTHESIS_REGISTRY.json"
EXPERIMENT_PATH = RESEARCH / "EXPERIMENT_REGISTRY.json"
LOCK_PATH = RESEARCH / ".V71_REGISTRY.lock"
SCHEMA_VERSION = 1

V71_HYPOTHESIS = {
    "id": "US_SEC_CAUSAL_RESPONSE_DIRECTION_V1",
    "family": "CAUSAL_PRIOR+SEC_STRUCTURED+SECTOR_CONTEXT",
    "source": "US_SEC",
    "target": "DIRECTION_AUC",
    "preconditions": [
        "V69 atomic COMMIT and ARTIFACT_MANIFEST verify",
        "immutable V36 DEV hash matches",
        "chronological nested OOF with 35-minute embargo",
        "research seal and final meta remain unopened",
    ],
    "material_change": (
        "Add strictly past-only issuer/form/event/taxonomy response priors and combine them with "
        "structured SEC and causal pre-event market/sector context for direction ranking."
    ),
    "dependencies": [
        "output_V69/V69_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        "data/dev_contract_v36_labeled.csv.gz",
        "experiment_v59_sec_structured_v1.py",
    ],
    "runner": "experiment_v71_us_sec_causal_response_direction.py",
    "status": "READY",
    "last_result": None,
    "stagnation_count": 0,
}

V71_JOB = {
    "job_id": V71_HYPOTHESIS["id"],
    "hypothesis": V71_HYPOTHESIS["id"],
    "family": "US_SEC_CAUSAL_RESPONSE_DIRECTION",
    "runner": V71_HYPOTHESIS["runner"],
    "description": V71_HYPOTHESIS["material_change"],
    "priority": 145,
    "status": "READY",
    "attempt_count": 0,
    "assigned_version": 71,
}

# Controlled material family graph.  A factory proposal is a research
# question, not a combinatorial parameter search.
ALLOWED_FACTORY = {
    ("US_SEC", "DIRECTION_AUC"): [
        {
            "family": "CAUSAL_PRIOR+SEC_STRUCTURED+SECTOR_CONTEXT",
            "question": "Do past-only issuer/event responses add SEC direction ranking beyond V69?",
            "dependencies": ["SEC_STRUCTURED", "CAUSAL_PRIOR", "SECTOR_CONTEXT"],
        },
        {
            "family": "SEC_SEMANTIC_RETRIEVAL+CAUSAL_PRIOR",
            "question": "Does material-segment retrieval transfer when conditioned on past-only response priors?",
            "dependencies": ["SEC_SEMANTIC", "CAUSAL_PRIOR"],
        },
        {
            "family": "SOURCE_EXPERT_ENSEMBLE",
            "question": "Can independently useful SEC numeric/structured/semantic experts improve AUC by stable consensus?",
            "dependencies": ["SEC_STRUCTURED", "SEC_SEMANTIC", "NUMERIC"],
        },
    ],
    ("US_NEWS", "CONFIDENCE_OPPORTUNITY"): [
        {
            "family": "CROSSFIT_PCORRECT+OPPORTUNITY",
            "question": "Can cross-fitted P(correct) and P(material move) jointly stabilize US_NEWS HC selection?",
            "dependencies": ["DIRECTION_OOF", "OPPORTUNITY"],
        }
    ],
    ("KR_NEWS", "NEW_INFORMATION"): [
        {
            "family": "KOREAN_EVENT_TEXT+MARKET_CONTEXT",
            "question": "Does separated Korean event text plus causal market context add non-random direction information?",
            "dependencies": ["KR_TEXT", "KIS_CONTEXT"],
        }
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hypothesis(row: dict[str, Any]) -> None:
    required = {
        "id", "family", "source", "target", "preconditions", "material_change",
        "dependencies", "status", "last_result", "stagnation_count",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"hypothesis missing fields: {missing}")
    if row["status"] not in {"READY", "RUNNING", "COMMITTED", "FAILED", "STAGNATED", "BLOCKED"}:
        raise RuntimeError(f"invalid hypothesis status: {row['status']}")
    if not isinstance(row["preconditions"], list) or not isinstance(row["dependencies"], list):
        raise RuntimeError("preconditions/dependencies must be lists")
    if int(row["stagnation_count"]) < 0:
        raise RuntimeError("negative stagnation_count")


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported hypothesis registry schema")
    rows = registry.get("hypotheses")
    if not isinstance(rows, list):
        raise RuntimeError("hypotheses must be a list")
    identifiers = []
    for row in rows:
        validate_hypothesis(row)
        identifiers.append(row["id"])
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("duplicate hypothesis id")


def choose_material_hypothesis(
    registry: dict[str, Any], source: str, bottleneck: str
) -> dict[str, Any] | None:
    """Return the first untried controlled family for a source bottleneck."""
    validate_registry(registry)
    attempted = {
        row["family"]
        for row in registry["hypotheses"]
        if row["source"] == source and row["target"] == bottleneck
    }
    for proposal in ALLOWED_FACTORY.get((source, bottleneck), []):
        if proposal["family"] not in attempted:
            return {
                "source": source,
                "target": bottleneck,
                **proposal,
                "factory_rule": "first untried controlled material family; no parameter-product expansion",
            }
    return None


def atomic_write(payload: dict[str, Any], path: Path) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def register() -> dict[str, Any]:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    # All V71 registry writers use this lock.  Both files are prepared from the
    # latest bytes while held; on a second-file failure, the first is restored.
    with LOCK_PATH.open("a+b") as lock_handle:
        if os.name == "nt":
            import msvcrt

            lock_handle.seek(0)
            if lock_handle.tell() == 0:
                lock_handle.write(b"0")
                lock_handle.flush()
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            hypothesis_before = HYPOTHESIS_PATH.read_bytes() if HYPOTHESIS_PATH.exists() else None
            experiment_before = EXPERIMENT_PATH.read_bytes() if EXPERIMENT_PATH.exists() else None
            hypotheses = read_json(
                HYPOTHESIS_PATH,
                {"schema_version": SCHEMA_VERSION, "created_at": now(), "updated_at": now(), "hypotheses": []},
            )
            experiments = read_json(EXPERIMENT_PATH, {"schema_version": 1, "created_at": now(), "jobs": []})
            rows = [row for row in hypotheses.get("hypotheses", []) if row.get("id") != V71_HYPOTHESIS["id"]]
            rows.append(copy.deepcopy(V71_HYPOTHESIS))
            hypotheses["schema_version"] = SCHEMA_VERSION
            hypotheses["updated_at"] = now()
            hypotheses["factory"] = {
                "policy": "controlled material family selection; no combinatorial parameter search",
                "stagnation_rule": "three consecutive material experiments with delta BA <0.003, delta AUC <0.003, and no HC robustness gain",
                "available_routes": [f"{source}:{target}" for source, target in ALLOWED_FACTORY],
            }
            hypotheses["hypotheses"] = rows
            validate_registry(hypotheses)
            jobs = [job for job in experiments.get("jobs", []) if job.get("job_id") != V71_JOB["job_id"]]
            jobs.append(copy.deepcopy(V71_JOB))
            experiments["jobs"] = jobs
            try:
                atomic_write(hypotheses, HYPOTHESIS_PATH)
                atomic_write(experiments, EXPERIMENT_PATH)
            except Exception:
                if hypothesis_before is None:
                    HYPOTHESIS_PATH.unlink(missing_ok=True)
                else:
                    temporary = HYPOTHESIS_PATH.with_name(HYPOTHESIS_PATH.name + f".rollback.{os.getpid()}")
                    temporary.write_bytes(hypothesis_before)
                    os.replace(temporary, HYPOTHESIS_PATH)
                if experiment_before is None:
                    EXPERIMENT_PATH.unlink(missing_ok=True)
                else:
                    temporary = EXPERIMENT_PATH.with_name(EXPERIMENT_PATH.name + f".rollback.{os.getpid()}")
                    temporary.write_bytes(experiment_before)
                    os.replace(temporary, EXPERIMENT_PATH)
                raise
        finally:
            if os.name == "nt":
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
    return {
        "status": "REGISTERED",
        "hypothesis_path": str(HYPOTHESIS_PATH.relative_to(ROOT)),
        "experiment_path": str(EXPERIMENT_PATH.relative_to(ROOT)),
        "hypothesis_id": V71_HYPOTHESIS["id"],
        "job_status": V71_JOB["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--factory", action="store_true")
    parser.add_argument("--source", default="US_SEC")
    parser.add_argument("--bottleneck", default="DIRECTION_AUC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        result = register()
    else:
        registry = read_json(HYPOTHESIS_PATH, {"schema_version": SCHEMA_VERSION, "hypotheses": []})
        validate_registry(registry)
        if args.factory:
            result = {"status": "OK", "proposal": choose_material_hypothesis(registry, args.source, args.bottleneck)}
        else:
            result = {"status": "VALID", "hypotheses": len(registry["hypotheses"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
