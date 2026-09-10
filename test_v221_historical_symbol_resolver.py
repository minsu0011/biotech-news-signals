from __future__ import annotations

import pandas as pd

import historical_symbol_resolver as resolver
import v59_provider_acquisition as provider


def test_ixbrl_prefers_common_stock_over_warrant():
    raw = """
    <html><body>
      <ix:nonNumeric name="dei:Security12bTitle" contextRef="common">Common Stock</ix:nonNumeric>
      <ix:nonNumeric name="dei:TradingSymbol" contextRef="common">ABCD</ix:nonNumeric>
      <ix:nonNumeric name="dei:Security12bTitle" contextRef="warrant">Warrants</ix:nonNumeric>
      <ix:nonNumeric name="dei:TradingSymbol" contextRef="warrant">ABCDW</ix:nonNumeric>
    </body></html>
    """
    evidence = resolver.extract_trading_symbols(raw)
    symbol, ambiguous = resolver.choose_symbol(evidence, "CURRENT")
    assert symbol == "ABCD"
    assert ambiguous is False


def test_cover_table_symbol_extraction():
    raw = """
    <table>
      <tr><th>Title of each class</th><th>Trading Symbol(s)</th><th>Name of exchange</th></tr>
      <tr><td>Common Stock, par value $0.01</td><td>XYZ</td><td>Nasdaq</td></tr>
    </table>
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "OTHER") == ("XYZ", False)


def test_vertical_cover_key_value_symbol_extraction():
    raw = """
    <table>
      <tr><td>Trading Symbol(s)</td><td>VERT</td></tr>
      <tr><td>Name of exchange</td><td>Nasdaq</td></tr>
    </table>
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "OTHER") == ("VERT", False)


def test_cover_header_footnote_is_supported():
    raw = """
    <table>
      <tr><th>Title of each class</th><th>Trading Symbol(s) (1)</th><th>Exchange</th></tr>
      <tr><td>Class A Common Stock</td><td>FOOT</td><td>Nasdaq</td></tr>
    </table>
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "OTHER") == ("FOOT", False)


def test_form4_issuer_name_does_not_replace_bracketed_ticker():
    raw = """
    <span>2. Issuer Name and Ticker or Trading Symbol</span>
    <a>PACIFIC HEALTH CARE ORGANIZATION INC</a> [ <span>PFHO</span> ]
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "CURRENT") == ("PFHO", False)


def test_native_form4_issuer_trading_symbol_xml():
    raw = """
    <ownershipDocument>
      <issuer><issuerCik>0000123456</issuerCik>
      <issuerName>Example Bio Inc.</issuerName>
      <issuerTradingSymbol>EXBI</issuerTradingSymbol></issuer>
    </ownershipDocument>
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "CURRENT") == ("EXBI", False)


def test_explicit_otc_ticker_sentence_is_identity_evidence():
    raw = 'Our common stock is quoted on the OTC pink sheets under the ticker symbol “GTHP.”'
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "CURRENT") == ("GTHP", False)


def test_symbol_change_uses_destination_not_preposition():
    raw = 'we changed our trading symbol from “BLKE” to “MODD,” effective June 29, 2017.'
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "CURRENT") == ("MODD", False)


def test_equal_strength_multiple_symbols_are_ambiguous():
    evidence = [
        {"symbol": "AAA", "method": "TEST", "score": 100},
        {"symbol": "BBB", "method": "TEST", "score": 100},
    ]
    assert resolver.choose_symbol(evidence, "ZZZ") == ("", True)


def test_paired_ads_and_warrant_text_prefers_ads_symbol():
    raw = """
    Our ADSs and our public warrants are currently traded on NASDAQ under the
    symbols &ldquo;KTOV&rdquo; and &ldquo;KTOVW&rdquo;, respectively.
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "PPBT") == ("KTOV", False)


def test_warrant_only_under_symbol_text_is_not_common_equity_evidence():
    raw = "Our public warrants are listed on NASDAQ under the symbol KTOVW."
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "PPBT") == ("", False)


def test_excluded_security_title_is_never_selected_alone():
    evidence = [{
        "symbol": "ABCDW", "method": "IXBRL_DEI_TRADING_SYMBOL",
        "security_title": "Warrants", "score": 140,
    }]
    assert resolver.choose_symbol(evidence, "ABCD") == ("", False)


def test_prior_common_symbol_sentence_does_not_pair_with_later_warrants():
    raw = """
    Our common stock is listed on NASDAQ under the symbol CERC. Our Class A
    warrants and Class B warrants are listed on NASDAQ under the symbols
    CERCW and CERCZ, respectively.
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "AVTX") == ("CERC", False)


def test_ordered_unit_common_right_list_selects_common_symbol():
    raw = """
    We plan to list our units, ordinary shares and rights on Nasdaq under the
    symbols TENKU, TENK, and TENKR, respectively.
    Once separate trading begins, the ordinary shares and rights will be
    listed on Nasdaq under the symbols TENK and TENKR, respectively.
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "CTOR") == ("TENK", False)


def test_warrant_queue_ticker_preserves_warrant_share_class():
    raw = """
    Once separate trading begins, the common stock and warrants will be listed
    on Nasdaq under the symbols BCYP and BCYPW, respectively.
    """
    evidence = resolver.extract_trading_symbols(raw)
    assert resolver.choose_symbol(evidence, "SABSW") == ("BCYPW", False)


def test_event_id_parses_cik_and_accession():
    assert resolver.parse_event_id("SEC:1173313:0001213900-19-013446") == (
        "1173313", "0001213900-19-013446"
    )


def test_historical_provider_symbol_drives_cache_lookup(tmp_path, monkeypatch):
    cache = tmp_path / "alpaca"
    expected = provider._cache_path(cache / "sip", "OLD", "2020-01-02")
    expected.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": []}).to_parquet(expected, index=False)
    monkeypatch.setattr(provider, "ALPACA_CACHE", cache)
    monkeypatch.setattr(provider, "authorized_alpaca_feed", lambda: "SIP")
    status = pd.DataFrame(
        [{
            "event_id": "e1",
            "selected_price_provider": "ALPACA_SIP",
            "provider_parity_eligible": True,
            "exact_contract_status": "EXACT_TIMING_ELIGIBLE",
            "provider_symbol": "OLD",
        }]
    ).set_index("event_id", drop=False)
    assert provider.resolve_us_price_cache("e1", "NEW", "2020-01-02", status) == (
        expected, "ALPACA", "SIP", "OLD"
    )


def test_historical_massive_provider_symbol_drives_cache_lookup(tmp_path, monkeypatch):
    expected = provider._cache_path(tmp_path, "OLD", "2020-01-02")
    expected.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": []}).to_parquet(expected, index=False)
    monkeypatch.setattr(provider, "MASSIVE_CACHE", tmp_path)
    status = pd.DataFrame(
        [{
            "event_id": "e1",
            "selected_price_provider": "MASSIVE_SIP",
            "provider_parity_eligible": True,
            "exact_contract_status": "EXACT_TIMING_ELIGIBLE",
            "provider_symbol": "OLD",
        }]
    ).set_index("event_id", drop=False)
    assert provider.resolve_us_price_cache("e1", "NEW", "2020-01-02", status) == (
        expected, "MASSIVE", "SIP", "OLD"
    )


def test_queue_is_label_blind():
    frame = pd.read_parquet(resolver.QUEUE_PATH)
    forbidden = {"y", "fwd_ret_30m", "entry_price", "exit_price"}
    assert forbidden.isdisjoint(frame.columns)
    assert frame.CIK.astype(str).ne("").all()


def test_multi_filing_consensus_contract_passes_two_independent_prior_filings():
    policy, _ = resolver.load_consensus_policy()
    records = [
        {"accession": "a1", "symbol": "OLD", "days_before_event": 30,
         "evidence_sha256": "1" * 64},
        {"accession": "a2", "symbol": "OLD", "days_before_event": 150,
         "evidence_sha256": "2" * 64},
    ]
    assert resolver.evaluate_consensus_records(records, 0, policy) == ("OLD", "PASS")


def test_multi_filing_consensus_rejects_contradictory_symbol():
    policy, _ = resolver.load_consensus_policy()
    records = [
        {"accession": "a1", "symbol": "OLD", "days_before_event": 30,
         "evidence_sha256": "1" * 64},
        {"accession": "a2", "symbol": "NEW", "days_before_event": 60,
         "evidence_sha256": "2" * 64},
    ]
    assert resolver.evaluate_consensus_records(records, 0, policy) == (
        "", "CONTRADICTORY_PRIOR_SYMBOL_EVIDENCE"
    )


def test_multi_filing_consensus_rejects_single_filing():
    policy, _ = resolver.load_consensus_policy()
    records = [
        {"accession": "a1", "symbol": "OLD", "days_before_event": 30,
         "evidence_sha256": "1" * 64},
    ]
    assert resolver.evaluate_consensus_records(records, 0, policy) == (
        "", "INSUFFICIENT_INDEPENDENT_PRIOR_FILINGS"
    )
