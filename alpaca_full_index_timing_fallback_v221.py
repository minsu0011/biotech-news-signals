"""Run the frozen Alpaca SIP fallback only for the full-index V2 candidates."""
from __future__ import annotations

import alpaca_recent_timing_fallback_v221 as fallback


fallback.POLICY_PATH = fallback.DATA / "ALPACA_FULL_INDEX_MASSIVE_TIMING_FALLBACK_POLICY_V1.json"
fallback.COLLECTION_STATUS_PATH = (
    fallback.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_STATUS_V2.json"
)
fallback.REPORT_PATH = (
    fallback.RESEARCH / "ALPACA_FULL_INDEX_MASSIVE_TIMING_FALLBACK_REPORT.json"
)


if __name__ == "__main__":
    raise SystemExit(fallback.main())
