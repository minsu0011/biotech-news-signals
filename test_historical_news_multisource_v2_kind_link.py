from historical_news_multisource_v2_kind_link import distinctive_codes, kind_event_match


def test_distinctive_codes_normalize_hyphenated_drug_codes() -> None:
    assert distinctive_codes("CT-P55 and Q702") == {"ctp55", "q702"}


def test_kind_match_accepts_specific_same_disclosure() -> None:
    result = kind_event_match(
        "셀트리온, CT-P55 미국 임상 3상 계획 변경 승인",
        "셀트리온",
        "투자판단 관련 주요경영사항(CTP55 미국 임상 3상 시험계획 변경신청 승인)",
        "수시공시",
    )
    assert result["distinctive_code_consistent"]
    assert result["event_match_high"]


def test_kind_match_rejects_different_drug_code_same_issuer() -> None:
    result = kind_event_match(
        "셀트리온, CT-P44 유럽 임상 3상 계획 변경",
        "셀트리온",
        "투자판단 관련 주요경영사항(CTP51 유럽 임상 3상 조기종료)",
        "수시공시",
    )
    assert not result["distinctive_code_consistent"]
    assert not result["event_match_high"]
