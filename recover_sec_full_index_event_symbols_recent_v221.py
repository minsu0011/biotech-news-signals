"""Run the frozen recent-range V2 policy through the shared full-index engine."""
from __future__ import annotations

import recover_sec_full_index_event_symbols_v221 as recovery


recovery.POLICY_PATH = recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_POLICY_V2.json"
recovery.OUTPUT_PATH = recovery.DATA / "events_sec_v221_legacy_gap_full_index_20250601_20260827.csv"
recovery.IDENTITY_AUDIT_PATH = (
    recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_IDENTITY_AUDIT_V2.parquet"
)
recovery.REPORT_PATH = recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_STATUS_V2.json"


if __name__ == "__main__":
    raise SystemExit(recovery.main())
