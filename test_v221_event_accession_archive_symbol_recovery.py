from __future__ import annotations

import event_accession_archive_symbol_recovery_v221 as target


POLICY = {
    "ambiguous_event_accession_symbol_evidence_forbidden": True,
    "contradictory_event_accession_symbol_evidence_forbidden": True,
    "evidence_file_sha256_required": True,
}


def record(symbol: str) -> dict[str, str]:
    return {"symbol": symbol, "evidence_sha256": "a" * 64}


def test_one_exact_accession_symbol_passes() -> None:
    symbol, reason, evidence = target.evaluate_records([record("BHST")], 0, POLICY)
    assert (symbol, reason, len(evidence)) == ("BHST", "PASS", 1)


def test_multiple_documents_may_agree() -> None:
    symbol, reason, evidence = target.evaluate_records(
        [record("GLPG"), record("GLPG")], 0, POLICY
    )
    assert (symbol, reason, len(evidence)) == ("GLPG", "PASS", 2)


def test_conflicting_event_documents_fail_closed() -> None:
    symbol, reason, _ = target.evaluate_records(
        [record("AAA"), record("BBB")], 0, POLICY
    )
    assert symbol == ""
    assert reason == "CONTRADICTORY_EVENT_ACCESSION_ARCHIVE_SYMBOLS"
