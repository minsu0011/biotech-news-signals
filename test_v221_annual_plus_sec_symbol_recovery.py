from __future__ import annotations

import annual_plus_sec_symbol_recovery_v221 as target


POLICY = {
    "minimum_prior_annual_filings": 1,
    "minimum_distinct_nonannual_sec_filings": 1,
    "ambiguous_symbol_evidence_forbidden": True,
    "contradictory_symbol_evidence_forbidden": True,
    "maximum_nearest_prior_days": 550,
    "maximum_oldest_prior_days": 1500,
    "evidence_file_sha256_required": True,
}


def record(accession: str, symbol: str, days: float) -> dict[str, object]:
    return {
        "accession": accession,
        "symbol": symbol,
        "days_before_event": days,
        "evidence_sha256": "a" * 64,
    }


def test_cross_form_agreement_passes() -> None:
    symbol, reason, evidence = target.evaluate_records(
        [record("annual", "GLPG", 220)],
        [record("ownership", "GLPG", 15)],
        0,
        POLICY,
    )
    assert (symbol, reason, len(evidence)) == ("GLPG", "PASS", 2)


def test_single_annual_cannot_promote_without_independent_filing() -> None:
    symbol, reason, _ = target.evaluate_records(
        [record("annual", "BHST", 80)], [], 0, POLICY
    )
    assert symbol == ""
    assert reason == "INDEPENDENT_PRIOR_SEC_EXPLICIT_EVIDENCE_MISSING"


def test_cross_form_symbol_conflict_is_rejected() -> None:
    symbol, reason, _ = target.evaluate_records(
        [record("annual", "OLD", 300)],
        [record("ownership", "NEW", 20)],
        0,
        POLICY,
    )
    assert symbol == ""
    assert reason == "CONTRADICTORY_PRIOR_SYMBOL_EVIDENCE"


def test_ambiguous_filing_is_rejected() -> None:
    symbol, reason, _ = target.evaluate_records(
        [record("annual", "GLPG", 220)],
        [record("ownership", "GLPG", 15)],
        1,
        POLICY,
    )
    assert symbol == ""
    assert reason == "AMBIGUOUS_PRIOR_SYMBOL_EVIDENCE"


def test_event_id_digest_is_order_independent_and_newline_delimited() -> None:
    expected = target.event_ids_sha256(["b", "a"])
    assert expected == target.event_ids_sha256(["a", "b"])
    assert expected != target.event_ids_sha256(["ab"])
