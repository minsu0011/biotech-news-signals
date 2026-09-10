import annual_report_symbol_recovery_v221 as annual


def policy():
    value, _ = annual.load_policy()
    return value


def test_annual_consensus_accepts_two_prior_filings_with_same_symbol():
    records = [
        {"accession": "a1", "symbol": "GSK", "days_before_event": 200,
         "evidence_sha256": "1" * 64},
        {"accession": "a2", "symbol": "GSK", "days_before_event": 560,
         "evidence_sha256": "2" * 64},
    ]
    symbol, reason, matching = annual.evaluate_records(records, 0, policy())
    assert (symbol, reason, len(matching)) == ("GSK", "PASS", 2)


def test_annual_consensus_rejects_conflicting_symbol():
    records = [
        {"accession": "a1", "symbol": "OLD", "days_before_event": 200,
         "evidence_sha256": "1" * 64},
        {"accession": "a2", "symbol": "NEW", "days_before_event": 560,
         "evidence_sha256": "2" * 64},
    ]
    assert annual.evaluate_records(records, 0, policy())[1] == (
        "CONTRADICTORY_ANNUAL_SYMBOL_EVIDENCE"
    )


def test_annual_consensus_rejects_single_filing():
    records = [
        {"accession": "a1", "symbol": "NVO", "days_before_event": 100,
         "evidence_sha256": "1" * 64},
    ]
    assert annual.evaluate_records(records, 0, policy())[1] == (
        "INSUFFICIENT_INDEPENDENT_PRIOR_ANNUAL_FILINGS"
    )
