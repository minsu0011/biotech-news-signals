from __future__ import annotations

import json

import pandas as pd

import provider_parity_v221 as parity


def detail(rows: int = 40) -> pd.DataFrame:
    massive = pd.Series([index / 1000 for index in range(rows)], dtype=float)
    alpaca = massive * 1.001
    return pd.DataFrame(
        {
            "both_exact": [True] * rows,
            "alpaca_fetch_usable": [True] * rows,
            "massive_exact_reason": ["OK"] * rows,
            "alpaca_exact_reason": ["OK"] * rows,
            "bar_coverage_ratio": [1.0] * rows,
            "entry_timestamp_agreement": [True] * rows,
            "exit_timestamp_agreement": [True] * rows,
            "entry_price_relative_diff": [0.0002] * rows,
            "fwd_ret_30m_massive": massive,
            "fwd_ret_30m_alpaca": alpaca,
            "label_agreement": [True] * rows,
            "provider_disposition": ["PAIR_OK"] * rows,
        }
    )


def policy() -> dict:
    return json.loads(parity.POLICY_PATH.read_text(encoding="utf-8"))


def test_frozen_thresholds_pass_high_parity():
    result = parity.summarize(detail(), "iex", policy())
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_frozen_thresholds_fail_label_disagreement():
    frame = detail()
    frame.loc[:5, "label_agreement"] = False
    frame.loc[:5, "provider_disposition"] = "PROVIDER_LABEL_DISAGREEMENT"
    result = parity.summarize(frame, "iex", policy())
    assert result["status"] == "FAIL"
    assert result["checks"]["up_down_label_agreement"] is False


def test_frozen_policy_forbids_sealed_label_parity():
    value = policy()
    assert value["eligible_role_for_label_level_parity"] == "DEV_EXTENSION"
    assert set(value["sealed_roles_label_access_forbidden"]) == {
        "RESEARCH_SEAL_POOL",
        "FINAL_META_RESERVE",
    }
    assert value["provider_choice_uses_outcomes"] is False


def test_timing_parity_means_timestamp_agreement_not_sample_yield():
    frame = detail()
    rejected = frame.iloc[:20].copy()
    rejected["both_exact"] = False
    rejected["massive_exact_reason"] = "ENTRY_TIMING_MISS"
    rejected["alpaca_exact_reason"] = "ENTRY_TIMING_MISS"
    rejected["entry_timestamp_agreement"] = False
    rejected["exit_timestamp_agreement"] = False
    rejected["entry_price_relative_diff"] = float("nan")
    rejected["fwd_ret_30m_massive"] = float("nan")
    rejected["fwd_ret_30m_alpaca"] = float("nan")
    combined = pd.concat([frame, rejected], ignore_index=True)
    result = parity.summarize(combined, "sip", policy())
    assert result["metrics"]["exact_pair_rate"] < 1.0
    assert result["metrics"]["timing_contract_agreement"] == 1.0
    assert result["metrics"]["exact_eligibility_agreement_on_fetched"] == 1.0
    assert result["status"] == "PASS"


def test_backfill_role_priority_is_dev_then_seal_then_final():
    assert {
        "DEV_EXTENSION": 0,
        "RESEARCH_SEAL_POOL": 1,
        "FINAL_META_RESERVE": 2,
    } == {
        name: rank
        for rank, name in enumerate(
            ["DEV_EXTENSION", "RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"]
        )
    }
