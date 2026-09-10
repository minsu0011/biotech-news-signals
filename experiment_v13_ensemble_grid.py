#!/usr/bin/env python
"""DEV-only cutoff search on repeated cross-fitted V13 predictions."""

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def main() -> None:
    data = pd.read_parquet(app.CACHE / "v13_repeated_oof.parquet")
    candidates = []
    for kr_cutoff in np.arange(0.30, 0.551, 0.01):
        for us_cutoff in np.arange(0.35, 0.601, 0.01):
            probability = data.prob.to_numpy(float).copy()
            for market, cutoff in (("KR", kr_cutoff), ("US", us_cutoff)):
                positions = data.market.eq(market).to_numpy()
                probability[positions] = app.shift_probability(probability[positions], cutoff)
            candidate = data.copy()
            candidate["prob"] = probability
            for highconf in np.arange(0.70, 0.951, 0.01):
                metrics = app.evaluate(candidate, float(highconf), 0.002)
                selected, score = app.dev_selection_score(metrics, app.Config())
                candidates.append((selected, score, kr_cutoff, us_cutoff, highconf, metrics))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    for selected, score, kr_cutoff, us_cutoff, highconf, metrics in candidates[:20]:
        print({
            "passes": selected, "score": score, "kr_cutoff": kr_cutoff,
            "us_cutoff": us_cutoff, "highconf": highconf,
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "auc": metrics["auc"], "edge": metrics["edge_vs_naive"],
            "coverage": metrics["highconf_coverage"], "hc_accuracy": metrics["highconf_accuracy"],
            "hc_net": metrics["strategy_mean_signed_net"],
            "KR_bal": metrics["by_market"]["KR"]["balanced_accuracy"],
            "US_bal": metrics["by_market"]["US"]["balanced_accuracy"],
        })


if __name__ == "__main__":
    main()
