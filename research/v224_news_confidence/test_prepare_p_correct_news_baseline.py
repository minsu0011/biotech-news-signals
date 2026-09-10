from __future__ import annotations

import numpy as np
import pandas as pd

import prepare_p_correct_news_baseline as baseline


def history_frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(rows)],
        "event_time_utc": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        "correct_target": np.arange(rows) % 2,
        "provider_class": ["P"] * rows,
        "publisher_host_class": ["H"] * rows,
        "event_type_clean": ["E"] * rows,
        "ticker_clean": ["T"] * rows,
        "article_key": [f"a{i}" for i in range(rows)],
    })


def test_prequential_history_never_uses_same_row() -> None:
    frame = history_frame()
    features, audit = baseline.history_feature_frame(frame, frame)
    assert audit["same_event_reference_n"] == 0
    assert audit["same_row_correctness_used"] is False
    assert audit["minimum_used_history_gap_minutes"] > 35.0
    assert features.iloc[0].history_global_count == 0.0
    assert features.iloc[0].history_global_correct_rate == 0.5


def test_validation_features_do_not_depend_on_validation_targets() -> None:
    train = history_frame(8)
    valid = history_frame(4).copy()
    valid["event_id"] = [f"v{i}" for i in range(4)]
    valid["event_time_utc"] = pd.date_range("2026-01-02", periods=4, freq="h", tz="UTC")
    toggled = valid.copy()
    toggled["correct_target"] = 1 - toggled.correct_target
    first, audit_first = baseline.frozen_history_feature_frame(train, valid)
    second, audit_second = baseline.frozen_history_feature_frame(train, toggled)
    pd.testing.assert_frame_equal(first, second)
    assert audit_first == audit_second


def test_article_cluster_is_purged_across_outer_fold() -> None:
    frame = history_frame(600)
    frame["article_key"] = [f"a{i}" for i in range(600)]
    # Duplicate an article from the first block into the second block.
    frame.loc[130, "article_key"] = frame.loc[5, "article_key"]
    folds = baseline.purged_outer_folds(frame)
    assert folds
    for train, valid, _, audit in folds:
        overlap = set(train.article_key) & set(valid.article_key)
        assert not overlap
        assert audit["article_overlap_n"] == 0
        assert train.event_time_utc.max() < valid.event_time_utc.min() - baseline.EMBARGO


def test_source_provenance_is_sanitized() -> None:
    assert baseline.provider_class("NAVER_NEWS_V14_SEAL") == "NAVER"
    assert baseline.provider_class("US_EXACT_V33_SEAL") == "GOOGLE_NEWS"
    assert baseline.provider_class("REUTERS_QIU") == "REUTERS"
    assert "SEAL" not in baseline.provider_class("KR_EXACT_V34_SEAL")


def test_raw_identifiers_and_outcomes_are_forbidden_from_model_schema() -> None:
    assert {"url", "article_id", "article_key"}.issubset(baseline.IDENTIFIER_COLUMNS)
    assert {"correct_target", "y", "fwd_ret_30m"}.issubset(baseline.FORBIDDEN_MODEL_COLUMNS)
    catalog = baseline.feature_catalog(pd.DataFrame({
        "direction_prediction": [True], "prob": [0.7],
        "prob_opportunity_direction": [0.6], "prob_event_form_direction": [0.7],
        "prob_fixed_numeric_direction": [0.8], "confidence_signal": [0.4],
        "opportunity_confidence": [0.5], "confidence_opportunity_direction": [0.6],
        "confidence_event_form_direction": [0.7], "confidence_fixed_numeric_direction": [0.8],
        "url": ["https://example.test/x"], "article_id": ["id"],
        "timestamp_quality": ["EXACT"], "headline": ["headline"], "body": ["body"],
        "event_time_utc": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "provider_class": ["OTHER_NEWS"], "publisher_host_class": ["OTHER_HOST"],
        "event_type_clean": ["NEWS"], "source_family": ["US_NEWS"],
    }))
    excluded = catalog.loc[~catalog.baseline_included, "feature"].tolist()
    assert "url" in excluded and "article_id" in excluded and "correct_target" in excluded
