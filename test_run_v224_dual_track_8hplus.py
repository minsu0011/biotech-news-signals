from datetime import date

import pytest

import run_v224_dual_track_8hplus as runner


def test_nyse_calendar_rejects_weekend_and_known_holidays() -> None:
    assert runner.is_nyse_session(date(2026, 8, 31))
    assert not runner.is_nyse_session(date(2026, 8, 29))
    assert not runner.is_nyse_session(date(2026, 9, 7))
    assert not runner.is_nyse_session(date(2026, 7, 3))


def test_python_placeholder_is_not_shell_expanded() -> None:
    command = runner.expanded_command(
        ["{PYTHON}", "collector.py", "--date", "{DATE}"],
        {"DATE": "2026-08-31"},
    )
    assert command == ["{PYTHON}", "collector.py", "--date", "2026-08-31"]


def test_forbidden_shared_authority_command_fails_closed() -> None:
    policy = {"forbidden_command_fragments": ["assign-roles", "RESEARCH_SEAL"]}
    with pytest.raises(ValueError, match="forbidden command fragment"):
        runner.assert_allowed_command(
            ["python", "worker.py", "--assign-roles"], policy
        )


def test_error_signature_is_stable_without_exposing_error_text() -> None:
    first = runner.error_signature(2, "private provider error", "")
    second = runner.error_signature(2, "private provider error", "")
    assert first == second
    assert "private provider error" not in first
