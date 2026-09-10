from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("V228_news_feature_contribution_polarity.py")
SPEC = importlib.util.spec_from_file_location("v228_polarity", MODULE_PATH)
assert SPEC and SPEC.loader
v228 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v228)


def test_alpha_counterfactual_identity() -> None:
    full = np.asarray([-1.0, 0.0, 2.0])
    ablated = np.asarray([-0.5, 0.5, 1.0])
    np.testing.assert_allclose(
        v228.alpha_probability(full, ablated, 1.0),
        np.clip(v228.expit(full), 1e-6, 1.0 - 1e-6),
    )
    np.testing.assert_allclose(
        v228.alpha_probability(full, ablated, 0.0),
        np.clip(v228.expit(ablated), 1e-6, 1.0 - 1e-6),
    )
    np.testing.assert_allclose(
        v228.alpha_probability(full, ablated, -1.0),
        np.clip(v228.expit(2.0 * ablated - full), 1e-6, 1.0 - 1e-6),
    )


def test_inner_choice_never_reads_outer_metrics() -> None:
    rows = [
        {"alpha": alpha, "inner_auc": 0.50, "inner_balanced_accuracy": 0.50,
         "inner_top20_accuracy": 0.50, "outer_auc": 1.0 if alpha == -1.0 else 0.0}
        for alpha in v228.ALPHAS
    ]
    assert v228.choose_alpha(rows) == 1.0
    for row in rows:
        row["outer_auc"] = 1.0 if row["alpha"] == 1.5 else 0.0
    assert v228.choose_alpha(rows) == 1.0


def test_stability_classification_is_inner_only() -> None:
    import pandas as pd
    rows = []
    choices = [-1.0, -1.0, -1.0, -1.0, 0.5]
    for fold, chosen in enumerate(choices, 1):
        for alpha in v228.ALPHAS:
            rows.append({
                "outer_fold": fold, "alpha": alpha, "inner_auc": 0.51,
                "chosen_alpha": chosen, "chosen_by_inner": alpha == chosen,
            })
    result = v228.classification(pd.DataFrame(rows))
    assert result["classification"] == "POLARITY_REVERSED"
    assert result["modal_inner_alpha"] == -1.0
    assert result["stability"] == 0.8


def test_budget_blocks_and_grid_are_fixed() -> None:
    assert v228.MATERIAL_BUDGET == 1
    assert v228.ALPHAS == (-1.0, 0.0, 0.5, 1.0, 1.5)
    assert set(v228.BLOCKS) == {
        "PUBLISHER", "TOPIC", "SOURCE_CONSENSUS", "SOURCE_DIVERSITY", "NOVELTY",
        "STRUCTURED_CONFIRMATION", "MODEL_AGREEMENT", "MARKET_CONTEXT", "TEXT_CERTAINTY",
    }
