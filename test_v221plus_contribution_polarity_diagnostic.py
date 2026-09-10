from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiment_v221plus_contribution_polarity_diagnostic as diagnostic


def _metric(primary: float) -> dict[str, float]:
    return {
        "primary_metric": primary,
        "auc": primary,
        "balanced_accuracy": primary,
        "edge_vs_naive": primary - 0.5,
        "confidence_correctness_auc": primary,
        "hc_accuracy": primary,
        "confidence_monotonicity": primary,
        "hc_net": primary - 0.5,
        "material_move_auc": primary,
        "magnitude_spearman": primary - 0.5,
    }


def test_counterfactual_contribution_states_are_exact() -> None:
    full = np.array([2.0, -1.0, 0.5])
    without = np.array([0.5, 0.25, -0.5])
    np.testing.assert_allclose(diagnostic.counterfactual_score(full, without, 1.0), full)
    np.testing.assert_allclose(diagnostic.counterfactual_score(full, without, 0.0), without)
    np.testing.assert_allclose(
        diagnostic.counterfactual_score(full, without, -1.0), 2.0 * without - full,
    )
    np.testing.assert_allclose(
        diagnostic.counterfactual_score(full, without, 0.5), without + 0.5 * (full - without),
    )


def test_inner_policy_stability_and_dosage_are_inner_only() -> None:
    fold = {
        -1.0: _metric(0.49), 0.0: _metric(0.50), 0.5: _metric(0.54),
        1.0: _metric(0.501), 1.5: _metric(0.48),
    }
    policy = diagnostic.select_inner_policy(
        [fold, fold, fold, fold], "direction", 0.75, 0.002,
    )
    assert policy["dosage_enabled"] is True
    assert policy["chosen_alpha"] == 0.5
    assert policy["stability"] == 1.0
    assert policy["outer_metrics_used"] is False
    assert "outer" not in inspect.signature(diagnostic.select_inner_policy).parameters


def test_unstable_inner_policy_falls_back_to_zero() -> None:
    votes = []
    for winner in (-1.0, 0.0, 1.0, -1.0):
        row = {alpha: _metric(0.50) for alpha in diagnostic.DOSAGE_ALPHAS}
        row[winner] = _metric(0.60)
        votes.append(row)
    policy = diagnostic.select_inner_policy(votes, "confidence", 0.75, 0.002)
    assert policy["stability"] == 0.5
    assert policy["chosen_alpha"] == 0.0
    assert policy["fallback_reason"] == "INNER_STABILITY_BELOW_THRESHOLD"


def test_expanding_split_enforces_embargo_and_group_purge() -> None:
    times = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(120)],
        "event_group_id": [f"g{i}" for i in range(120)],
        "event_time_utc": times,
    })
    # The duplicate is old enough to enter training absent the explicit purge.
    frame.loc[5, "event_group_id"] = frame.loc[40, "event_group_id"]
    splits = diagnostic.expanding_purged_splits(frame, 2, minimum_train=20, minimum_valid=20)
    assert splits
    for split in splits:
        audit = diagnostic.chronology_audit(split)
        assert audit["event_group_overlap"] == 0
        assert pd.Timestamp(audit["train_end"]) < pd.Timestamp(audit["valid_start"]) - diagnostic.EMBARGO


def test_classification_detects_reversal_overweighting_and_extremes() -> None:
    scores = {-1.0: 0.49, 0.0: 0.50, 0.5: 0.54, 1.0: 0.52, 1.5: 0.48}
    assert diagnostic.classify_reversal(-1.0, 1.0, scores, 0.75, False) == "POLARITY_REVERSED"
    assert diagnostic.classify_reversal(0.5, 1.0, scores, 0.75, True) == (
        "OVERWEIGHTED|EXTREME_REVERSAL"
    )
    assert diagnostic.classify_reversal(1.0, 0.5, scores, 0.75, False) == "UNSTABLE"


def test_output_guard_rejects_paths_outside_audit_or_staging(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="research/audits or staging"):
        diagnostic.resolve_output_dir(str(tmp_path), smoke=True)
    accepted = diagnostic.resolve_output_dir(
        str(diagnostic.ROOT / "staging" / "v221_test_guard"), smoke=True,
    )
    assert diagnostic.ROOT / "staging" in accepted.parents


def test_source_contains_no_raw_feature_sign_flip() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8").replace(" ", "")
    assert "*=-1" not in source
    registry, _ = diagnostic.load_registry()
    assert registry["counterfactual"]["raw_feature_sign_flip_allowed"] is False


def test_small_source_is_conservative_zero_without_prediction_rows() -> None:
    folds = pd.DataFrame(diagnostic.unsupported_rows("KR_KIND", 49, ["return_path"]))
    score, classification, stability, aggregate = diagnostic.aggregate_scorecards(
        folds, pd.DataFrame(), ["KR_KIND"], ["return_path"], 0.75, 20, 1,
    )
    assert len(score) == len(diagnostic.ROLE_ORDER)
    assert score.chosen_alpha.eq(0.0).all()
    assert score.support.eq(49).all()
    assert classification.classification.eq("UNSTABLE").all()
    assert stability.stable.eq(False).all()
    assert aggregate["source_polarity_map"]["KR_KIND"]["return_path"]["direction"]["status"] == (
        "NO_STABLE_POLARITY_SIGNAL"
    )
