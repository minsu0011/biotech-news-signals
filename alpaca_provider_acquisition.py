"""Read-only Alpaca one-minute market-data acquisition for V221+.

Credentials are imported from the Windows User environment only when absent
from the process.  Secret values are never printed or serialized.  This module
contains no trading, order, position, or account endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRICE = ROOT / "price_data"
ALPACA_BASE = "https://data.alpaca.markets"
ALPACA_BARS_PATH = "/v2/stocks/{symbol}/bars"
ALPACA_CACHE = PRICE / "alpaca_1m"
PREFLIGHT_PATH = DATA / "ALPACA_PREFLIGHT_STATUS.json"
SEMANTICS_PATH = DATA / "BAR_SEMANTICS_ALPACA.json"
COVERAGE_PATH = DATA / "ALPACA_HISTORICAL_COVERAGE_AUDIT.json"
OFFICIAL_BARS_DOC = "https://docs.alpaca.markets/us/reference/stockbars"
OFFICIAL_MARKET_DATA_FAQ = "https://docs.alpaca.markets/us/docs/market-data-faq"
OFFICIAL_HISTORICAL_DOC = "https://docs.alpaca.markets/us/docs/historical-stock-data-1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    for secret in alpaca_secret_values(required=False):
        if secret and secret in serialized:
            raise RuntimeError("ALPACA_SECRET_SERIALIZATION_REJECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def alpaca_pair_status() -> dict[str, Any]:
    standard = (os.getenv("ALPACA_API_KEY") or "", os.getenv("ALPACA_API_SECRET") or "")
    compatibility = (os.getenv("APCA_API_KEY_ID") or "", os.getenv("APCA_API_SECRET_KEY") or "")
    standard_complete = all(standard)
    compatibility_complete = all(compatibility)
    selected_source = "STANDARD" if standard_complete else "COMPATIBILITY" if compatibility_complete else "NONE"
    return {
        "standard_pair_complete": standard_complete,
        "compatibility_pair_complete": compatibility_complete,
        "complete_pairs_identical": bool(standard_complete and compatibility_complete and standard == compatibility),
        "selected_pair_source": selected_source,
        "selected_pair_is_complete": selected_source != "NONE",
        "selected_key_secret_distinct": (
            standard[0] != standard[1]
            if selected_source == "STANDARD"
            else compatibility[0] != compatibility[1]
            if selected_source == "COMPATIBILITY"
            else False
        ),
        "outer_whitespace_absent": all(
            value == value.strip()
            for value in (*standard, *compatibility)
            if value
        ),
        "mixed_pair_forbidden": True,
    }


def alpaca_secret_values(required: bool = True) -> tuple[str, ...]:
    standard = (os.getenv("ALPACA_API_KEY") or "", os.getenv("ALPACA_API_SECRET") or "")
    compatibility = (os.getenv("APCA_API_KEY_ID") or "", os.getenv("APCA_API_SECRET_KEY") or "")
    if all(standard):
        key, secret = standard
    elif all(compatibility):
        key, secret = compatibility
    else:
        key, secret = "", ""
    if required and (not key or not secret):
        raise RuntimeError("CREDENTIAL_NOT_VISIBLE_TO_PROCESS:ALPACA")
    return key, secret


def credential_presence() -> dict[str, str]:
    key, secret = alpaca_secret_values(required=False)
    return {
        "ALPACA_API_KEY": "PRESENT" if key else "MISSING",
        "ALPACA_API_SECRET": "PRESENT" if secret else "MISSING",
    }


def safe_error(response: requests.Response | None, error: Exception | None = None) -> str:
    if response is None:
        return type(error).__name__ if error is not None else "UNKNOWN_ERROR"
    message = ""
    try:
        payload = response.json()
        for name in ("code", "status", "error", "message"):
            value = payload.get(name) if isinstance(payload, dict) else None
            if value not in (None, ""):
                message += ("|" if message else "") + str(value)[:240]
    except Exception:
        pass
    return f"HTTP_{response.status_code}" + (f":{message}" if message else "")


def bar_qa(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"pass": False, "rows": 0, "reason": "EMPTY"}
    ordered = frame.sort_values("timestamp")
    deltas = ordered.timestamp.diff().dt.total_seconds().dropna()
    prices = ordered[["open", "high", "low", "close"]]
    invalid_price = (prices <= 0).any(axis=1) | prices.isna().any(axis=1)
    bad_ohlc = (
        (ordered.high < ordered[["open", "close", "low"]].max(axis=1))
        | (ordered.low > ordered[["open", "close", "high"]].min(axis=1))
    )
    result = {
        "rows": int(len(ordered)),
        "timezone": str(ordered.timestamp.dt.tz),
        "duplicates": int(ordered.timestamp.duplicated().sum()),
        "non_monotonic": int((deltas <= 0).sum()),
        "invalid_price": int(invalid_price.sum()),
        "bad_ohlc": int(bad_ohlc.sum()),
        "negative_volume": int((ordered.volume < 0).sum()),
        "median_cadence_sec": float(deltas.median()) if len(deltas) else None,
        "missing_minute_gaps": int((deltas > 60).sum()),
        "timestamp_min": str(ordered.timestamp.min()),
        "timestamp_max": str(ordered.timestamp.max()),
    }
    result["pass"] = not any(
        result[name]
        for name in ("duplicates", "non_monotonic", "invalid_price", "bad_ohlc", "negative_volume")
    )
    return result


class AlpacaClient:
    """Strict read-only client for the single-symbol historical bars endpoint."""

    def __init__(self, min_interval: float = 0.35):
        if min_interval < 0:
            raise ValueError("ALPACA_MIN_INTERVAL_MUST_BE_NONNEGATIVE")
        self.key, self.secret = alpaca_secret_values()
        self.session = requests.Session()
        self.min_interval = float(min_interval)
        self.last_request = 0.0

    def get_bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        feed: str,
        *,
        limit: int = 10_000,
        asof: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        symbol = str(symbol).upper().strip()
        feed = str(feed).lower().strip()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("ALPACA_SYMBOL_REJECTED")
        if feed not in {"sip", "iex"}:
            raise ValueError("ALPACA_FEED_REJECTED")
        path = ALPACA_BARS_PATH.format(symbol=symbol)
        parsed = urlparse(ALPACA_BASE + path)
        if parsed.scheme != "https" or parsed.netloc != "data.alpaca.markets":
            raise RuntimeError("ALPACA_HOST_REJECTED")
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError("ALPACA_INTERVAL_REJECTED")
        headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        }
        params: dict[str, Any] = {
            "timeframe": "1Min",
            "start": start.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "end": end.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
            "limit": min(max(int(limit), 1), 10_000),
        }
        if asof is not None:
            parsed_asof = pd.Timestamp(asof)
            params["asof"] = parsed_asof.date().isoformat()
        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        request_count = 0
        rate_limited = 0
        response: requests.Response | None = None
        while True:
            if page_token:
                params["page_token"] = page_token
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            for attempt in range(5):
                try:
                    response = self.session.get(
                        ALPACA_BASE + path,
                        headers=headers,
                        params=params,
                        timeout=60,
                    )
                    request_count += 1
                    self.last_request = time.monotonic()
                    if response.status_code == 429:
                        rate_limited += 1
                        delay = float(response.headers.get("Retry-After") or min(60, 2 ** (attempt + 1)))
                        time.sleep(delay + random.random())
                        continue
                    if response.status_code >= 400:
                        return pd.DataFrame(), {
                            "ok": False,
                            "error": safe_error(response),
                            "http_status": int(response.status_code),
                            "feed": feed,
                            "requests": request_count,
                            "rate_limited": rate_limited,
                        }
                    payload = response.json()
                    page_rows = payload.get("bars") or []
                    if not isinstance(page_rows, list):
                        return pd.DataFrame(), {
                            "ok": False,
                            "error": "MALFORMED_BARS_RESPONSE",
                            "feed": feed,
                            "requests": request_count,
                            "rate_limited": rate_limited,
                        }
                    rows.extend(page_rows)
                    page_token = payload.get("next_page_token")
                    break
                except Exception as error:
                    self.last_request = time.monotonic()
                    if attempt == 4:
                        return pd.DataFrame(), {
                            "ok": False,
                            "error": safe_error(None, error),
                            "feed": feed,
                            "requests": request_count,
                            "rate_limited": rate_limited,
                        }
                    time.sleep(min(30, 2 ** attempt) + random.random())
            else:
                return pd.DataFrame(), {
                    "ok": False,
                    "error": "RETRY_EXHAUSTED",
                    "feed": feed,
                    "requests": request_count,
                    "rate_limited": rate_limited,
                }
            if not page_token:
                break
        if not rows:
            return pd.DataFrame(), {
                "ok": True,
                "http_status": 200,
                "feed": feed,
                "rows": 0,
                "requests": request_count,
                "rate_limited": rate_limited,
            }
        raw = pd.DataFrame(rows)
        required = {"t", "o", "h", "l", "c", "v"}
        if not required.issubset(raw.columns):
            return pd.DataFrame(), {
                "ok": False,
                "error": "MALFORMED_BAR_COLUMNS",
                "feed": feed,
                "requests": request_count,
                "rate_limited": rate_limited,
            }
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw.t, utc=True),
                "open": pd.to_numeric(raw.o, errors="coerce"),
                "high": pd.to_numeric(raw.h, errors="coerce"),
                "low": pd.to_numeric(raw.l, errors="coerce"),
                "close": pd.to_numeric(raw.c, errors="coerce"),
                "volume": pd.to_numeric(raw.v, errors="coerce"),
                "transactions": pd.to_numeric(raw.get("n"), errors="coerce"),
                "vwap": pd.to_numeric(raw.get("vw"), errors="coerce"),
            }
        ).sort_values("timestamp").drop_duplicates("timestamp")
        return frame, {
            "ok": True,
            "http_status": 200,
            "feed": feed,
            "rows": int(len(frame)),
            "requests": request_count,
            "rate_limited": rate_limited,
            "adjustment": "raw",
            "asof_used": asof is not None,
        }


def probe(
    client: AlpacaClient,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feed: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars, status = client.get_bars(symbol, start, end, feed)
    qa = bar_qa(bars)
    report = {
        **status,
        "symbol": symbol,
        "interval_utc": [str(start), str(end)],
        "qa": qa,
        "http_success": bool(status.get("ok")),
        "usable_1m": bool(status.get("ok") and qa.get("pass") and not bars.empty),
    }
    return bars, report


def run_preflight() -> dict[str, Any]:
    refresh_user_environment(ALPACA_ENV_NAMES)
    pair_status = alpaca_pair_status()
    recent_start = pd.Timestamp("2026-08-25T13:30:00Z")
    recent_end = recent_start + pd.Timedelta(minutes=60)
    historical_start = pd.Timestamp("2023-06-15T13:30:00Z")
    historical_end = historical_start + pd.Timedelta(minutes=60)

    recent_attempts = []
    recent_bars = pd.DataFrame()
    recent: dict[str, Any] = {}
    for attempt in range(1, 4):
        # A credential refresh must never reuse an old HTTP session/object.
        recent_bars, recent = probe(AlpacaClient(min_interval=0), "AAPL", recent_start, recent_end, "iex")
        recent["fresh_process_session_attempt"] = attempt
        recent_attempts.append(recent)
        if recent.get("usable_1m") or recent.get("http_status") != 401:
            break

    sip_bars = pd.DataFrame()
    sip: dict[str, Any] = {"attempted": False, "reason": "RECENT_IEX_AUTH_GATE_FAILED"}
    iex_bars = pd.DataFrame()
    iex: dict[str, Any] = {"attempted": False, "reason": "RECENT_IEX_AUTH_GATE_FAILED"}
    if recent.get("usable_1m"):
        historical_client = AlpacaClient(min_interval=0)
        sip_bars, sip = probe(historical_client, "AAPL", historical_start, historical_end, "sip")
        sip["attempted"] = True
        if sip.get("http_status") == 403:
            iex_bars, iex = probe(AlpacaClient(min_interval=0), "AAPL", historical_start, historical_end, "iex")
            iex["attempted"] = True
        else:
            iex = {"attempted": False, "reason": "SIP_DID_NOT_RETURN_ENTITLEMENT_403"}

    semantics = {
        "schema_version": 1,
        "provider": "ALPACA",
        "official_documentation": [
            OFFICIAL_BARS_DOC,
            OFFICIAL_MARKET_DATA_FAQ,
            OFFICIAL_HISTORICAL_DOC,
        ],
        "timestamp_field": "bars[].t",
        "timestamp_semantics": "BAR_OPEN_LEFT_EDGE",
        "official_explanation": "trade timestamp is truncated to the minute; interval includes left edge and excludes right edge",
        "cadence_seconds": 60,
        "adjustment": "raw",
        "timezone_parse": "RFC3339_UTC",
        "missing_bar_policy": "NO_FILL_NO_INTERPOLATION",
        "feed_contract": {"sip": "ALL_US_EXCHANGES", "iex": "IEX_ONLY"},
        "exact_timing_semantics_eligible": True,
    }
    report = {
        "schema_version": 1,
        "audited_at": now(),
        "read_only_market_data_calls": True,
        "values_serialized": False,
        "credentials": credential_presence(),
        "credential_pair_consistency": pair_status,
        "fresh_user_environment_refresh": True,
        "recent_1m_auth_attempts": recent_attempts,
        "recent_1m_preflight": recent,
        "historical_pre_2024_09_sip": sip,
        "historical_pre_2024_09_iex": iex,
        "bar_semantics": semantics,
        "historical_sip_usable": bool(sip.get("usable_1m")),
        "historical_iex_usable": bool(iex.get("usable_1m")),
        "status": (
            "ALPACA_HISTORICAL_SIP_PASS"
            if sip.get("usable_1m")
            else "ALPACA_IEX_FALLBACK"
            if iex.get("usable_1m")
            else "ALPACA_HARD_BLOCKED"
        ),
        "blocker_category": (
            "NONE"
            if sip.get("usable_1m") or iex.get("usable_1m")
            else "AUTHENTICATION"
            if any(
                probe_result.get("http_status") == 401
                for probe_result in (recent, sip, iex)
            )
            else "ENTITLEMENT"
            if any(
                probe_result.get("http_status") == 403
                for probe_result in (recent, sip, iex)
            )
            else "NO_BAR_OR_NETWORK"
        ),
        "promotion": {
            "sip": "ELIGIBLE_FOR_PARITY_TEST_ONLY" if sip.get("usable_1m") else "INELIGIBLE",
            "iex": "ELIGIBLE_FOR_PARITY_TEST_ONLY" if iex.get("usable_1m") else "INELIGIBLE",
            "exact_contract_backfill_authorized": False,
            "reason": "Massive overlap provider parity must pass before any Alpaca-backed label row is authorized",
        },
    }
    atomic_json(semantics, SEMANTICS_PATH)
    atomic_json(report, PREFLIGHT_PATH)
    atomic_json(
        {
            "schema_version": 1,
            "audited_at": report["audited_at"],
            "representative_probe_only": True,
            "pre_2024_09": {
                "symbol": "AAPL",
                "date": "2023-06-15",
                "sip": sip,
                "iex": iex,
            },
            "status": report["status"],
            "blocker_category": report["blocker_category"],
            "full_deferred_candidate_backfill_started": False,
        },
        COVERAGE_PATH,
    )
    # Keep in-memory data from being mistaken for an authorized cache.
    assert not recent_bars.empty or not recent.get("usable_1m")
    assert not sip_bars.empty or not sip.get("usable_1m")
    assert not iex_bars.empty or not iex.get("usable_1m")
    return report


def run_historical_coverage(feed: str) -> dict[str, Any]:
    refresh_user_environment(ALPACA_ENV_NAMES)
    dates = (
        "2015-06-15",
        "2016-06-15",
        "2017-06-15",
        "2018-06-15",
        "2019-06-14",
        "2020-06-15",
        "2021-06-15",
        "2022-06-15",
        "2023-06-15",
        "2024-06-14",
    )
    rows = []
    for date in dates:
        start = pd.Timestamp(f"{date} 09:30", tz="America/New_York").tz_convert("UTC")
        end = start + pd.Timedelta(minutes=60)
        bars, result = probe(AlpacaClient(min_interval=0), "AAPL", start, end, feed)
        rows.append(
            {
                "feed": feed.upper(),
                "year": int(date[:4]),
                "date": date,
                "http_status": result.get("http_status"),
                "ok": bool(result.get("ok")),
                "usable_1m": bool(result.get("usable_1m")),
                "bars": int(len(bars)),
                "cadence_sec": result.get("qa", {}).get("median_cadence_sec"),
                "qa_pass": bool(result.get("qa", {}).get("pass")),
                "error": result.get("error"),
            }
        )
    successful = [row for row in rows if row["usable_1m"]]
    previous = json.loads(COVERAGE_PATH.read_text(encoding="utf-8")) if COVERAGE_PATH.is_file() else {}
    previous.update(
        {
            "schema_version": 2,
            "audited_at": now(),
            "fresh_network_requests": len(rows),
            "representative_symbol": "AAPL",
            "representative_selection_uses_labels": False,
            "coverage_feed": feed.upper(),
            "annual_probes": rows,
            "earliest_success": min((row["date"] for row in successful), default=None),
            "latest_success": max((row["date"] for row in successful), default=None),
            "successful_years": [row["year"] for row in successful],
            "status": "PASS" if successful else "FAIL",
            "full_deferred_candidate_backfill_started": False,
            "values_serialized": False,
        }
    )
    atomic_json(previous, COVERAGE_PATH)
    return previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--coverage-feed", choices=("sip", "iex"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.coverage_feed:
        report = run_historical_coverage(args.coverage_feed)
        print(
            json.dumps(
                {
                    "output": str(COVERAGE_PATH),
                    "status": report["status"],
                    "feed": report["coverage_feed"],
                    "successful_years": report["successful_years"],
                    "earliest_success": report["earliest_success"],
                    "values_serialized": False,
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "PASS" else 3
    if not args.preflight:
        raise SystemExit("choose --preflight")
    report = run_preflight()
    print(
        json.dumps(
            {
                "output": str(PREFLIGHT_PATH),
                "status": report["status"],
                "recent_1m_ok": report["recent_1m_preflight"].get("usable_1m"),
                "historical_sip_ok": report["historical_sip_usable"],
                "historical_iex_ok": report["historical_iex_usable"],
                "values_serialized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] != "ALPACA_HARD_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
