from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from provider_env_runtime import import_user_environment_if_missing


ROOT = Path(__file__).resolve().parent
V1 = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
V2_RESEARCH = ROOT / "research" / "multisource_v2"
V2_WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
V2_QUEUE_DIR = ROOT / "research" / "hist_news_multisource_v2"
QUEUE = V2_QUEUE_DIR / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
QUEUE_SUMMARY = V2_QUEUE_DIR / "QUEUE_SUMMARY.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"
V224 = ROOT / "cert_research" / "V224_CONFIRMATION"
PROMPT = ROOT / "research" / "collection-policy.md"

COLLECTION_POLICY = V2_RESEARCH / "COLLECTION_POLICY_FREEZE.json"
EVENT_TIME_POLICY = V2_RESEARCH / "EVENT_TIME_RESOLUTION_POLICY_FREEZE.json"
POLICY_MANIFEST = V2_RESEARCH / "POLICY_FREEZE_MANIFEST.json"
PROVIDER_AUDIT = V2_RESEARCH / "PROVIDER_CAPABILITY_AUDIT.json"
QUEUE_AUDIT = V2_RESEARCH / "QUEUE_AUDIT.json"

EXPECTED_V1_SOURCE_FREEZE_SHA256 = "1233b2f048bf67e609d0a31c4508a8ff9b91c47f20cb934e7072c37ed5a4354a"
EXPECTED_QUEUE_SHA256 = "8a00e5f8d703ff21c84e82270ecded105193b42136b16b7c925d31e4e9a54ae1"

ALLOWED_PRIORITY_FIELDS = (
    "market", "event_year", "ticker", "issuer", "event_time_utc",
    "article_count", "publisher_count", "publisher_class", "timestamp_quality",
    "exact_price_available", "body_available", "source_coverage", "topic_coverage",
)
FORBIDDEN_SELECTION_FIELDS = (
    "y", "up_down", "future_return", "gross_return", "net_return", "profit",
    "direction_correctness", "confidence_correctness", "signed_net",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_once(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable_existing = {key: value for key, value in existing.items() if key != "created_at_utc"}
        immutable_new = {key: value for key, value in payload.items() if key != "created_at_utc"}
        if immutable_existing != immutable_new:
            raise RuntimeError(f"immutable policy mismatch: {path.name}")
        return existing
    atomic_json(path, payload)
    return payload


def build_collection_policy(prompt_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_COLLECTION_POLICY_V1",
        "created_at_utc": utc_now(),
        "epoch": "HIST_NEWS_MULTISOURCE_V2",
        "append_only": True,
        "base_v1_membership_mutable": False,
        "outcome_blind_collection": True,
        "allowed_priority_fields": list(ALLOWED_PRIORITY_FIELDS),
        "forbidden_selection_fields": list(FORBIDDEN_SELECTION_FIELDS),
        "outcome_fields_read": [],
        "unit_of_collection": "CANONICAL_MARKET_EVENT",
        "decision_cutoff_rule": "event_time_utc + 2 minutes",
        "discovery_window": "event_time_utc - 24h through event_time_utc + 24h",
        "causal_article_rule": "published_at_utc <= decision_cutoff_utc; minute precision or better; HIGH event match",
        "historical_first_seen_rule": "NULL_UNLESS_GENUINE_HISTORICAL_PROVENANCE",
        "current_crawl_as_historical_first_seen": False,
        "independence_unit": "independent_reporting_root_id",
        "host_equals_origin": False,
        "pr_distribution_copies_are_one_root": True,
        "syndicated_copies_are_independent": False,
        "paywall_login_captcha_robots_bypass": False,
        "same_terminal_url_retry": False,
        "allowed_discovery_routes": [
            "EXISTING_RAW_CACHE", "PUBLIC_RSS", "PUBLISHER_ARCHIVE", "ISSUER_PR_ARCHIVE",
            "REGULATORY_PUBLIC_SOURCE", "PUBLIC_API", "EXISTING_PROVIDER",
        ],
        "provider_error_policy": "429/403 cooldown; terminal after provider exhaustion; move to alternate legitimate source",
        "no_progress_policy": "after 3 zero-add cycles, mark family exhausted and switch source family",
        "new_embedding_model_must_match_v1": True,
        "price_feature_cutoff": "decision_cutoff_utc",
        "post_entry_prices_namespace": "LABEL_ONLY",
        "price_contract": {
            "entry_target_seconds_after_event": 120,
            "entry_slippage_max_seconds": 60,
            "exit_seconds_after_actual_entry": 1800,
            "exit_slippage_max_seconds": 60,
            "five_minute_fallback": False,
            "daily_fallback": False,
            "synthetic_interpolation": False,
        },
        "source_freeze_before_label_join": True,
        "role_split_before_label_join": True,
        "hist_confirm_retuning_allowed": False,
        "research_seal_read": False,
        "final_meta_read": False,
        "v224_mutable": False,
        "base_v1_source_freeze_sha256": EXPECTED_V1_SOURCE_FREEZE_SHA256,
        "initial_queue_sha256": EXPECTED_QUEUE_SHA256,
        "task_prompt_sha256": prompt_sha256,
    }


def build_event_time_policy(prompt_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_EVENT_TIME_POLICY_V1",
        "created_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "base_v1_event_time_mutable": False,
        "v2_resolved_event_time_column": "v2_resolved_event_time_utc",
        "resolution_precedence": [
            "EARLIEST_HIGH_CONFIDENCE_OFFICIAL_DISCLOSURE",
            "EARLIEST_HIGH_CONFIDENCE_ORIGINAL_PUBLICATION",
            "PRESERVE_BASE_V1_EVENT_TIME",
        ],
        "candidate_must_match_same_canonical_event": True,
        "ambiguous_resolution": "EVENT_TIME_AMBIGUOUS_AND_PRESERVE_BASE_V1",
        "event_time_freeze_before_price_label": True,
        "label_or_return_use_for_resolution": False,
        "task_prompt_sha256": prompt_sha256,
    }


def verify_protected() -> dict[str, Any]:
    if sha256_file(V1 / "SOURCE_FREEZE.json") != EXPECTED_V1_SOURCE_FREEZE_SHA256:
        raise RuntimeError("BASE V1 source freeze hash mismatch")
    if sha256_file(QUEUE) != EXPECTED_QUEUE_SHA256:
        raise RuntimeError("V2 initial queue hash mismatch")
    v1_manifest = pd.read_csv(V1 / "SHA256_MANIFEST.csv")
    v1_failures = [
        str(row.file) for row in v1_manifest.itertuples(index=False)
        if sha256_file(V1 / str(row.file)) != str(row.sha256)
    ]
    v224_manifest = json.loads((V224 / "FROZEN_V224_SHA256.json").read_text(encoding="utf-8"))
    v224_failures = [
        name for name, record in v224_manifest["files"].items()
        if sha256_file(V224 / name) != record["sha256"]
    ]
    if v1_failures or v224_failures:
        raise RuntimeError(f"protected manifest mismatch: V1={v1_failures}, V224={v224_failures}")
    return {
        "base_v1_manifest_files": int(len(v1_manifest)),
        "base_v1_manifest_failures": 0,
        "v224_manifest_files": int(len(v224_manifest["files"])),
        "v224_manifest_failures": 0,
        "research_seal_read": False,
        "final_meta_read": False,
    }


def hardware_audit() -> dict[str, Any]:
    gpu = {"status": "UNKNOWN"}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False, encoding="utf-8", errors="replace",
        )
        if completed.returncode == 0 and completed.stdout.strip():
            name, memory = [part.strip() for part in completed.stdout.strip().split(",", 1)]
            gpu = {"status": "PRESENT", "name": name, "memory_mib": int(memory)}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "gpu": gpu,
        "runtime_authority": "ACTUAL_MACHINE_NOT_PROMPT_EXAMPLE",
    }


def provider_audit() -> dict[str, Any]:
    presence = import_user_environment_if_missing()
    groups = {
        "MASSIVE_POLYGON": "PRESENT" if presence["MASSIVE_API_KEY"] == "PRESENT" or presence["POLYGON_API_KEY"] == "PRESENT" else "MISSING",
        "ALPACA": "PRESENT" if all(presence[name] == "PRESENT" for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")) else "MISSING",
        "KIS": "PRESENT" if all(presence[name] == "PRESENT" for name in ("KIS_APP_KEY", "KIS_APP_SECRET")) else "MISSING",
    }
    local_sources = {
        "BASE_V1_ARTICLES": V1 / "ARTICLES_NORMALIZED.parquet",
        "BASE_V1_EMBEDDINGS": V1 / "EMBEDDING_INDEX.parquet",
        "GOOGLE_US_RAW": ROOT / "data" / "events_google_us_v33_raw.csv",
        "GOOGLE_KR_RAW": ROOT / "data" / "events_google_news_v33_raw.csv",
        "GOOGLE_KR_V34_RAW": ROOT / "data" / "events_google_kr_v34_raw.csv",
        "NAVER_CACHE": ROOT / "cache" / "naver_news_by_ticker",
        "REUTERS_QIU_CACHE": ROOT / "cache" / "sources" / "QiuStockNews" / "json_files" / "news.json",
    }
    return {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_PROVIDER_AUDIT_V1",
        "audited_at_utc": utc_now(),
        "credential_values_serialized": False,
        "credential_groups": groups,
        "credential_variables": presence,
        "local_sources": {name: "PRESENT" if path.exists() else "MISSING" for name, path in local_sources.items()},
        "public_routes_allowed": ["PUBLIC_RSS", "PUBLISHER_ARCHIVE", "ISSUER_PR_ARCHIVE", "REGULATORY_PUBLIC_SOURCE"],
        "hardware": hardware_audit(),
        "research_seal_read": False,
        "final_meta_read": False,
        "v224_touched": False,
    }


def queue_audit() -> dict[str, Any]:
    queue = pd.read_parquet(QUEUE)
    forbidden = sorted(set(queue.columns) & set(FORBIDDEN_SELECTION_FIELDS))
    if forbidden:
        raise RuntimeError(f"outcome columns present in collection queue: {forbidden}")
    event_time = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    return {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_QUEUE_AUDIT_V1",
        "audited_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_columns_present": [],
        "queue_rows": int(len(queue)),
        "queue_sha256": sha256_file(QUEUE),
        "by_market": {str(key): int(value) for key, value in queue["market"].value_counts().items()},
        "by_year": {str(int(key)): int(value) for key, value in event_time.dt.year.value_counts().sort_index().items()},
        "independent_source_count": {
            str(int(key)): int(value) for key, value in queue["independent_source_count_t2"].value_counts().items()
        },
        "current_publishers": {str(key): int(value) for key, value in queue["current_publishers_t2"].value_counts().items()},
        "base_v1_membership_mutable": False,
    }


def main() -> int:
    V2_RESEARCH.mkdir(parents=True, exist_ok=True)
    V2_WORKING.mkdir(parents=True, exist_ok=True)
    prompt_sha = sha256_file(PROMPT)
    protected = verify_protected()
    collection = write_once(COLLECTION_POLICY, build_collection_policy(prompt_sha))
    event_time = write_once(EVENT_TIME_POLICY, build_event_time_policy(prompt_sha))
    atomic_json(PROVIDER_AUDIT, provider_audit())
    queue = queue_audit()
    atomic_json(QUEUE_AUDIT, queue)
    manifest = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_POLICY_FREEZE_MANIFEST_V1",
        "created_at_utc": utc_now(),
        "collection_policy_sha256": sha256_file(COLLECTION_POLICY),
        "event_time_policy_sha256": sha256_file(EVENT_TIME_POLICY),
        "task_prompt_sha256": prompt_sha,
        "initial_queue_sha256": sha256_file(QUEUE),
        "base_v1_source_freeze_sha256": sha256_file(V1 / "SOURCE_FREEZE.json"),
        "outcome_blind": True,
        "label_join_allowed": False,
        "protected_audit": protected,
    }
    if POLICY_MANIFEST.is_file():
        old = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
        stable_keys = set(manifest) - {"created_at_utc"}
        if any(old.get(key) != manifest.get(key) for key in stable_keys):
            raise RuntimeError("policy freeze manifest changed")
        manifest = old
    else:
        atomic_json(POLICY_MANIFEST, manifest)

    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    state.update(
        {
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_STATE_V2",
            "updated_at_utc": utc_now(),
            "status": "POLICY_FROZEN_COLLECTION_READY",
            "phase": "OUTCOME_BLIND_COLLECTION",
            "base_v1_membership_mutable": False,
            "collection_policy_sha256": sha256_file(COLLECTION_POLICY),
            "event_time_policy_sha256": sha256_file(EVENT_TIME_POLICY),
            "policy_manifest_sha256": sha256_file(POLICY_MANIFEST),
            "queue_rows": queue["queue_rows"],
            "queue_sha256": queue["queue_sha256"],
            "collection_network_requests": int(state.get("collection_network_requests", 0)),
            "labels_read": False,
            "source_frozen": False,
            "split_frozen": False,
            "model_training_allowed": False,
            "research_seal_touched": False,
            "final_meta_touched": False,
            "v224_touched": False,
        }
    )
    atomic_json(STATE, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "queue_rows": queue["queue_rows"],
                "policy_manifest_sha256": state["policy_manifest_sha256"],
                "provider_groups": json.loads(PROVIDER_AUDIT.read_text(encoding="utf-8"))["credential_groups"],
                "protected_audit": protected,
                "outcome_fields_read": collection["outcome_fields_read"],
                "event_time_outcome_fields_read": event_time["outcome_fields_read"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
