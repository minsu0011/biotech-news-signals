from historical_news_multisource_v2_official_link import event_match_metrics, informative_tokens


def test_informative_tokens_remove_provider_boilerplate_and_ticker() -> None:
    tokens = informative_tokens("BRIEF-UPDATE Merck MRK CEO Kenneth Frazier to retire", "MRK")
    assert "brief" not in tokens
    assert "update" not in tokens
    assert "mrk" not in tokens
    assert {"merck", "ceo", "kenneth", "frazier", "retire"}.issubset(tokens)


def test_event_match_accepts_dense_local_semantic_evidence() -> None:
    headline = "Merck CEO Kenneth Frazier to retire at the end of June"
    body = (
        "The registrant announced that chief executive officer CEO Kenneth Frazier will retire "
        "at the end of June. Merck named a successor effective July. "
    ) * 4
    metrics = event_match_metrics(headline, body, "MRK")
    assert metrics["best_window_overlap_count"] >= 5
    assert metrics["event_match_high"]


def test_event_match_rejects_same_issuer_unrelated_filing() -> None:
    headline = "Gilead remdesivir gets FDA approval for hospitalized COVID-19 patients"
    body = (
        "Gilead filed this current report concerning debt securities, accounting controls, "
        "share capital and administrative exhibits. "
    ) * 5
    metrics = event_match_metrics(headline, body, "GILD")
    assert not metrics["event_match_high"]
