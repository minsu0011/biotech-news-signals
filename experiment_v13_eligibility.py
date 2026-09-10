#!/usr/bin/env python
"""DEV-only pre-entry eligibility search; V13 SEAL outcomes remain untouched."""

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def main() -> None:
    oof = pd.read_parquet(app.CACHE / "v13_legitimate_repeated_oof.parquet")
    raw = pd.read_csv(app.LABELED_FILE, dtype={"ticker": str}, low_memory=False)
    raw = raw[raw.source.isin({"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"})].copy()
    features = app.model_frame(raw).set_index(raw.event_id)
    features = features.loc[oof.event_id]
    kr = oof.market.eq("KR").to_numpy()
    us = ~kr
    kr_score = oof.loc[kr, "raw_score"].to_numpy(float)
    # Cutoff is selected only from repeated production-shaped DEV OOF.
    kr_distance = pd.Series(np.abs(kr_score - 0.675)).rank(pct=True).to_numpy(float)
    benchmark = features.iloc[np.flatnonzero(kr)]["benchmark_ret_5"].abs().rank(pct=True).fillna(0.5).to_numpy(float)
    pred_up = (kr_score >= 0.675).astype(float)
    candidates = []
    for benchmark_weight in np.arange(0.0, 2.01, 0.2):
        for up_bias in np.arange(-0.6, 0.61, 0.1):
            reliability = kr_distance + benchmark_weight * benchmark + up_bias * pred_up
            rank = pd.Series(reliability).rank(pct=True).to_numpy(float)
            for fraction in np.arange(0.20, 0.701, 0.05):
                eligible_kr = rank >= 1.0 - fraction
                estimated_kr = int(round(761.0 * eligible_kr.mean()))
                if estimated_kr < 150:
                    continue
                eligible = us.copy()
                eligible[kr] = eligible_kr
                frame = oof.loc[eligible].copy().reset_index(drop=True)
                market_kr = frame.market.eq("KR").to_numpy()
                frame["eval_weight"] = np.where(
                    market_kr, estimated_kr / market_kr.sum(), 350.0 / (~market_kr).sum()
                )
                for kr_cutoff in np.arange(0.58, 0.731, 0.01):
                    for us_cutoff in (0.575,):
                        probability = frame.raw_score.to_numpy(float).copy()
                        probability[market_kr] = app.shift_probability(probability[market_kr], kr_cutoff)
                        probability[~market_kr] = app.shift_probability(probability[~market_kr], us_cutoff)
                        frame["prob"] = probability
                        metrics = app.evaluate(frame, 0.70, 0.002)
                        direction_pass = (
                            metrics["n"] >= 390 and estimated_kr + 350 >= 500
                            and metrics["balanced_accuracy"] >= 0.54
                            and metrics["auc"] >= 0.56
                            and metrics["edge_vs_naive"] >= 0.02
                            and metrics["by_market"]["KR"]["balanced_accuracy"] >= 0.51
                            and metrics["by_market"]["US"]["balanced_accuracy"] >= 0.51
                        )
                        if direction_pass:
                            candidates.append((
                                metrics["edge_vs_naive"], metrics["balanced_accuracy"], metrics["auc"],
                                benchmark_weight, up_bias, fraction, estimated_kr,
                                kr_cutoff, us_cutoff, metrics["accuracy"],
                                metrics["by_market"]["KR"]["balanced_accuracy"],
                                metrics["by_market"]["US"]["balanced_accuracy"],
                                int(eligible_kr.sum()),
                            ))
    candidates.sort(reverse=True)
    for row in candidates[:100]:
        print(row)
    print("count", len(candidates))


if __name__ == "__main__":
    main()
