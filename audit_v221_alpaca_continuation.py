"""Final value-free audit for the V221+ Alpaca continuation attempt."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from provider_env_runtime import import_user_environment_if_missing, presence


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "V221_ALPACA_CONTINUATION_STATUS.json"
V36_SHA = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
V69_OOF_SHA = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
V69_COMMIT_SHA = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
V69_MANIFEST_SHA = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    environment = import_user_environment_if_missing()
    preflight = load("data/ALPACA_PREFLIGHT_STATUS.json")
    coverage = load("data/ALPACA_HISTORICAL_COVERAGE_AUDIT.json")
    policy = load("data/ALPACA_PROVIDER_PARITY_POLICY_V1.json")
    parity = load("research/PROVIDER_PARITY_SUMMARY.json")
    backfill = load("data/ALPACA_BACKFILL_QUEUE_AUDIT.json")
    backfill_state = load("state/ALPACA_BACKFILL_STATE.json")
    readiness = load("research/V70_NEW_DATA_EPOCH_READINESS.json")
    sample = pd.read_csv(ROOT / "research" / "PROVIDER_PARITY_OVERLAP_SAMPLE.csv")
    queue = pd.read_parquet(ROOT / "data" / "ALPACA_BACKFILL_QUEUE.parquet")
    checks = {
        "V36_DEV_IMMUTABLE": sha256(ROOT / "data" / "dev_contract_v36_labeled.csv.gz") == V36_SHA,
        "V69_OOF_IMMUTABLE": sha256(ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz") == V69_OOF_SHA,
        "V69_COMMIT_IMMUTABLE": sha256(ROOT / "output_V69" / "COMMIT.json") == V69_COMMIT_SHA,
        "V69_MANIFEST_IMMUTABLE": sha256(ROOT / "output_V69" / "ARTIFACT_MANIFEST.json") == V69_MANIFEST_SHA,
        "ALPACA_ENV_PRESENT": all(environment[name] == "PRESENT" for name in (
            "ALPACA_API_KEY", "ALPACA_API_SECRET", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"
        )),
        "ALPACA_AUTHENTICATION_BLOCK_CONFIRMED": preflight.get("blocker_category") == "AUTHENTICATION",
        "PARITY_POLICY_FROZEN_BEFORE_RESULTS": policy.get("frozen_before_alpaca_overlap_results") is True,
        "PARITY_SAMPLE_DEV_ONLY": set(sample.role.astype(str)) == {"DEV_EXTENSION"},
        "PARITY_SAMPLE_OUTCOME_FREE": not any(name in sample.columns for name in ("y", "fwd_ret_30m", "entry_price", "exit_price")),
        "SEALED_ROLE_LABELS_UNOPENED": parity.get("sealed_role_labels_opened") is False and readiness.get("seal_or_final_labels_opened") is False,
        "PARITY_FAILED_CLOSED": parity.get("backfill_authorized") is False,
        "BACKFILL_NETWORK_REQUESTS_ZERO": backfill_state.get("network_requests") == 0,
        "BACKFILL_ROLES_PRESERVED": backfill.get("existing_frozen_roles_preserved") is True,
        "BACKFILL_QUEUE_EXACT_4221": len(queue) == 4221,
        "NEW_DATA_EPOCH_NOT_READY": readiness.get("status") == "NOT_READY",
        "OUTPUT_V221_ABSENT": not (ROOT / "output_V221").exists(),
        "STAGING_V221_ABSENT": not (ROOT / "staging" / "V221").exists(),
        "CERT_V36_ABSENT": not (ROOT / "cert" / "V36").exists(),
    }
    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "values_serialized": False,
        "status": "BLOCKED_EXTERNAL_ALPACA_AUTHENTICATION" if all(checks.values()) else "INTERNAL_AUDIT_FAIL",
        "credentials": presence(),
        "alpaca_preflight": {
            "status": preflight.get("status"),
            "blocker_category": preflight.get("blocker_category"),
            "recent_iex_http_status": preflight.get("recent_1m_preflight", {}).get("http_status"),
            "historical_sip_http_status": preflight.get("historical_pre_2024_09_sip", {}).get("http_status"),
            "historical_iex_http_status": preflight.get("historical_pre_2024_09_iex", {}).get("http_status"),
        },
        "parity": {
            "policy_sha256": sha256(ROOT / "data" / "ALPACA_PROVIDER_PARITY_POLICY_V1.json"),
            "sample_rows": len(sample),
            "sample_unique_tickers": int(sample.ticker.nunique()),
            "sample_by_source": {str(k): int(v) for k, v in sample.source_family.value_counts().items()},
            "sample_by_liquidity": {str(k): int(v) for k, v in sample.liquidity_bucket.value_counts().items()},
            "status": parity.get("result", {}).get("status"),
            "backfill_authorized": parity.get("backfill_authorized"),
        },
        "backfill": {
            "rows": len(queue),
            "inventory": coverage.get("deferred_candidate_inventory"),
            "queue_sha256": sha256(ROOT / "data" / "ALPACA_BACKFILL_QUEUE.parquet"),
            "state": backfill_state,
        },
        "new_data_epoch": {
            "status": readiness.get("status"),
            "blocking_reasons": readiness.get("blocking_reasons"),
            "counts": readiness.get("counts"),
        },
        "checks": checks,
        "next_action": {
            "external": "Replace or regenerate valid Alpaca market-data API credentials in Windows User environment.",
            "then_run": [
                "python alpaca_provider_acquisition.py --preflight",
                "python provider_parity_v221.py --run-parity sip --max-events 120",
                "python alpaca_backfill_v221.py --feed sip",
            ],
            "sip_denied_path": [
                "python provider_parity_v221.py --run-parity iex --max-events 120",
                "python alpaca_backfill_v221.py --feed iex",
            ],
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for name in environment:
        value = os.environ.get(name)
        if value and value in serialized:
            raise RuntimeError("PROVIDER_SECRET_SERIALIZATION_REJECTED")
    temporary = OUTPUT.with_name(OUTPUT.name + f".tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "status": report["status"], "all_checks_pass": all(checks.values()), "values_serialized": False}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
