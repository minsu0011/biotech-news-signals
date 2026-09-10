"""Fail-closed Tiingo IEX fallback preparation for V221+.

The adapter can normalize credentialed one-minute responses, but this module
never authorizes exact labels. Massive/Tiingo overlap parity must pass first.
Credential values are neither returned nor serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from provider_env_runtime import TIINGO_ENV_NAMES, import_user_environment_if_missing


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
PREFLIGHT_PATH = DATA / "TIINGO_PREFLIGHT_STATUS.json"
PLAN_PATH = RESEARCH / "TIINGO_FALLBACK_PLAN.json"
PARITY_POLICY_PATH = DATA / "ALPACA_PROVIDER_PARITY_POLICY_V1.json"
ENDPOINT_TEMPLATE = "https://api.tiingo.com/iex/{symbol}/prices"
OFFICIAL_DOCUMENTATION = (
    "https://www.tiingo.com/documentation/iex",
    "https://www.tiingo.com/documentation/general/connecting",
)
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def credential_status() -> dict[str, str]:
    """Return markers only; never return the token."""
    return import_user_environment_if_missing(TIINGO_ENV_NAMES)


def fetch_1m_bars(
    symbol: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch/normalize Tiingo IEX bars for parity research only.

    A successful response is not exact-contract authorization. Callers must
    enforce the frozen parity policy recorded by :func:`write_preflight`.
    """
    markers = credential_status()
    if markers[TIINGO_ENV_NAMES[0]] != "PRESENT":
        raise RuntimeError("TIINGO_CREDENTIAL_MISSING")
    normalized_symbol = str(symbol).strip().upper()
    if not SYMBOL.fullmatch(normalized_symbol):
        raise ValueError("INVALID_TIINGO_SYMBOL")
    start_utc = pd.Timestamp(start)
    end_utc = pd.Timestamp(end)
    start_utc = start_utc.tz_localize("UTC") if start_utc.tzinfo is None else start_utc.tz_convert("UTC")
    end_utc = end_utc.tz_localize("UTC") if end_utc.tzinfo is None else end_utc.tz_convert("UTC")
    if end_utc <= start_utc:
        raise ValueError("INVALID_TIINGO_INTERVAL")
    token = os.environ[TIINGO_ENV_NAMES[0]]
    response = requests.get(
        ENDPOINT_TEMPLATE.format(symbol=quote(normalized_symbol, safe="")),
        headers={"Authorization": f"Token {token}"},
        params={
            "startDate": start_utc.isoformat(),
            "endDate": end_utc.isoformat(),
            "resampleFreq": "1min",
            "format": "json",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"TIINGO_HTTP_{response.status_code}")
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("TIINGO_RESPONSE_SCHEMA_MISMATCH")
    frame = pd.DataFrame(payload)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise RuntimeError("TIINGO_RESPONSE_MISSING_BAR_FIELDS")
    frame = frame.rename(columns={"date": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].sort_values(
        "timestamp"
    ).reset_index(drop=True)


def write_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    markers = credential_status()
    present = markers[TIINGO_ENV_NAMES[0]] == "PRESENT"
    status = "CREDENTIAL_PRESENT_PARITY_REQUIRED" if present else "TIINGO_CREDENTIAL_MISSING"
    timestamp = now()
    frozen_policy = json.loads(PARITY_POLICY_PATH.read_text(encoding="utf-8"))
    preflight = {
        "schema_version": 1,
        "audited_at": timestamp,
        "status": status,
        "provider": "TIINGO_IEX",
        "credentials": markers,
        "credential_present": present,
        "values_serialized": False,
        "network_attempted": False,
        "read_only_market_data_calls": True,
        "official_documentation": list(OFFICIAL_DOCUMENTATION),
        "adapter": {
            "function": "fetch_1m_bars",
            "endpoint_template": ENDPOINT_TEMPLATE,
            "requested_resample": "1min",
            "response_fields": ["timestamp", "open", "high", "low", "close", "volume"],
            "bar_semantics_status": "UNVERIFIED_UNTIL_CREDENTIALED_PREFLIGHT",
        },
        "promotion": {
            "exact_contract_backfill_authorized": False,
            "reason": "MASSIVE_TIINGO_OVERLAP_PARITY_NOT_RUN",
        },
    }
    plan = {
        "schema_version": 1,
        "updated_at": timestamp,
        "status": status,
        "provider_priority": ["MASSIVE_SIP", "ALPACA_SIP", "TIINGO_IEX_PARITY_ELIGIBLE_ONLY"],
        "selection_uses_labels": False,
        "outcome_based_provider_selection_forbidden": True,
        "resolved_event_date_ticker_required": True,
        "synthetic_price_generation_forbidden": True,
        "forbidden_fills": ["5m interpolation", "daily interpolation", "linear price fabrication", "other-event proxy"],
        "parity": {
            "required": True,
            "executed": False,
            "reference_provider": "MASSIVE_SIP",
            "candidate_provider": "TIINGO_IEX",
            "provisional_threshold_source": str(PARITY_POLICY_PATH.relative_to(ROOT)),
            "provisional_threshold_source_sha256": hashlib.sha256(PARITY_POLICY_PATH.read_bytes()).hexdigest(),
            "provisional_thresholds": frozen_policy["thresholds"],
            "frozen_metrics": [
                "entry_timestamp_agreement",
                "exit_timestamp_agreement",
                "timing_contract_agreement",
                "up_down_label_agreement",
                "return_correlation",
                "absolute_entry_price_relative_difference",
                "bar_coverage",
            ],
            "thresholds_must_not_be_tuned_per_provider": True,
            "provider_disagreement_action": "PROVIDER_DISAGREEMENT_QUARANTINE",
        },
        "exact_contract": {
            "entry_target": "event_time+2m",
            "entry_slippage_max_seconds": 60,
            "exit_target": "actual_entry+30m",
            "exit_slippage_max_seconds": 60,
            "hold_seconds": [1800, 1860],
            "relaxation_forbidden": True,
        },
        "backfill_authorized": False,
        "next_action": (
            "RUN_CREDENTIALED_READ_ONLY_PREFLIGHT_THEN_FROZEN_PARITY"
            if present else "CONTINUE_NON_TIINGO_RECOVERY_UNTIL_CREDENTIAL_AVAILABLE"
        ),
    }
    atomic_json(preflight, PREFLIGHT_PATH)
    atomic_json(plan, PLAN_PATH)
    return preflight, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-preflight", action="store_true")
    args = parser.parse_args()
    if not args.write_preflight:
        parser.error("--write-preflight is required; network preflight is intentionally separate")
    preflight, plan = write_preflight()
    print(json.dumps({
        "status": preflight["status"],
        "credential_present": preflight["credential_present"],
        "network_attempted": preflight["network_attempted"],
        "backfill_authorized": plan["backfill_authorized"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
