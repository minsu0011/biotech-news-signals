"""Run the frozen V3 full-index policy over the expanded official-SIC universe."""
from __future__ import annotations

import recover_sec_full_index_event_symbols_v221 as recovery


recovery.UNIVERSE_PATH = recovery.DATA / "universe_v221_cached_official_sic_expanded.csv"
recovery.POLICY_PATH = recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_POLICY_V3.json"
recovery.OUTPUT_PATH = recovery.DATA / "events_sec_v221_legacy_gap_full_index_expanded_20240903_20260827.csv"
recovery.IDENTITY_AUDIT_PATH = (
    recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_IDENTITY_AUDIT_V3.parquet"
)
recovery.REPORT_PATH = recovery.DATA / "V221_SEC_FULL_INDEX_EVENT_SYMBOL_RECOVERY_STATUS_V3.json"


if __name__ == "__main__":
    raise SystemExit(recovery.main())
