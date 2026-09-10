from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("V227_news_direction_sample_weighting.py")
SPEC = importlib.util.spec_from_file_location("v227_weighting", MODULE_PATH)
assert SPEC and SPEC.loader
v227 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v227)


def synthetic_frame(rows: int = 600) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(rows)],
        "event_time_utc": pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC"),
        "text_cluster_id": [f"c{i}" for i in range(rows)],
        "correct_target": np.asarray([(i % 3) != 0 for i in range(rows)], dtype=int),
        "y": np.asarray([i % 2 for i in range(rows)], dtype=int),
        "source_family": ["US_NEWS"] * rows,
    })


def test_fixed_continuous_transform_and_neutral() -> None:
    probability = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    multiplier = v227.pcorrect_multiplier(probability)
    np.testing.assert_allclose(multiplier, [0.5, 0.75, 1.0, 1.25, 1.5])
    assert np.all(np.diff(multiplier) > 0)


def test_outer_folds_are_chronological_embargoed_and_cluster_purged() -> None:
    frame = synthetic_frame(750)
    # Force a cluster from the first block to recur in the second validation block.
    frame.loc[140, "text_cluster_id"] = frame.loc[5, "text_cluster_id"]
    folds = v227.outer_folds(frame)
    assert len(folds) == 5
    for train, valid, _, audit in folds:
        assert train.event_time_utc.max() < valid.event_time_utc.min() - v227.EMBARGO
        assert not (set(train.text_cluster_id) & set(valid.text_cluster_id))
        assert audit["text_cluster_overlap_n"] == 0


def test_expanding_pcorrect_never_uses_same_block_labels(monkeypatch) -> None:
    frame = synthetic_frame(240)

    def historical_mean(history, target, seed):
        value = float(history.correct_target.mean())
        return np.repeat(value, len(target)), {
            "raw_structured_feature_n": 1,
            "transformed_numeric_feature_n": 1,
            "text_feature_n": 0,
            "total_feature_n": 1,
            "raw_text_included": False,
            "forbidden_feature_overlap": [],
        }

    monkeypatch.setattr(v227, "fit_pcorrect", historical_mean)
    first, evidence_first, audit = v227.expanding_pcorrect(
        frame, 1, "US_NEWS", blocks=6, min_history=30
    )
    changed = frame.copy()
    changed.loc[200:, "correct_target"] = 1 - changed.loc[200:, "correct_target"]
    second, evidence_second, _ = v227.expanding_pcorrect(
        changed, 1, "US_NEWS", blocks=6, min_history=30
    )
    np.testing.assert_allclose(
        first.loc[200:, "p_correct_news_train"],
        second.loc[200:, "p_correct_news_train"],
    )
    assert not evidence_first.correctness_model_fit_contains_event.any()
    assert all(row["event_overlap_n"] == 0 for row in audit)
    assert all(row["text_cluster_overlap_n"] == 0 for row in audit)
    supported = evidence_second.p_correct_supported
    assert (evidence_second.loc[supported, "minimum_history_gap_minutes"] > 35).all()


def test_forbidden_direction_outcomes_are_declared() -> None:
    assert {"y", "fwd_ret_30m", "correct_target", "signed_net"}.issubset(
        v227.FORBIDDEN_DIRECTION_FEATURES
    )
    assert v227.MATERIAL_BUDGET == 1
    assert v227.WEIGHT_TRANSFORM == "0.5 + P_CORRECT_NEWS"
