from __future__ import annotations

import sixk_exhibit_consensus_recovery_v221 as target


POLICY = {
    "minimum_distinct_prior_filings": 2,
    "maximum_nearest_prior_days": 90,
    "maximum_oldest_prior_days": 550,
    "ambiguous_symbol_evidence_forbidden": True,
    "contradictory_symbol_evidence_forbidden": True,
    "evidence_file_sha256_required": True,
}


def row(accession: str, symbol: str, days: int) -> dict[str, object]:
    return {
        "accession": accession,
        "symbol": symbol,
        "days_before_event": days,
        "evidence_sha256": "a" * 64,
    }


def test_two_distinct_filings_agree() -> None:
    symbol, reason, evidence = target.evaluate_records(
        [row("a", "CYTO", 10), row("b", "CYTO", 80)], 0, POLICY
    )
    assert (symbol, reason, len(evidence)) == ("CYTO", "PASS", 2)


def test_same_accession_cannot_be_counted_twice() -> None:
    symbol, reason, _ = target.evaluate_records(
        [row("a", "CYTO", 10), row("a", "CYTO", 20)], 0, POLICY
    )
    assert symbol == ""
    assert reason == "INSUFFICIENT_DISTINCT_PRIOR_6K_EXHIBIT_FILINGS"


def test_conflicting_filings_fail_closed() -> None:
    symbol, reason, _ = target.evaluate_records(
        [row("a", "OLD", 10), row("b", "NEW", 20)], 0, POLICY
    )
    assert symbol == ""
    assert reason == "CONTRADICTORY_PRIOR_6K_EXHIBIT_EVIDENCE"
