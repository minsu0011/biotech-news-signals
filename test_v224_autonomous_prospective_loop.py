"""Guards for the isolated V224 future collector loop."""
from __future__ import annotations

from datetime import datetime

import pytest

import collect_v224_prospective_sec as collector
import v224_autonomous_prospective_loop as loop


def test_collection_contract_is_outcome_blind_and_body_complete() -> None:
    policy, role, digest = collector.load_contract()
    assert len(digest) == 64
    assert policy["selection_uses_labels"] is False
    assert policy["selection_uses_prices"] is False
    assert policy["exact_accession_body_required"] is True
    assert role["role_contract"]["new_role"] == "V224_PROSPECTIVE_CONFIRMATION"


def test_loop_rejects_shared_authority_commands() -> None:
    for command in (
        ["v59_provider_acquisition.py", "--assign-roles"],
        ["refresh_current_data_authority.py"],
        ["activate_v221_new_data_epoch.py"],
    ):
        with pytest.raises(RuntimeError, match="FORBIDDEN_COMMAND"):
            loop.assert_isolated_command(command)
    loop.assert_isolated_command(["v224_prospective_role_freeze.py"])


def test_schedule_is_weekday_and_after_close() -> None:
    before = datetime(2026, 8, 31, 16, 19, tzinfo=loop.ET)
    after = datetime(2026, 8, 31, 16, 21, tzinfo=loop.ET)
    assert loop.due_business_days(before, set()) == []
    assert loop.due_business_days(after, set()) == ["2026-08-31"]
    assert loop.due_business_days(after, {"2026-08-31"}) == []

