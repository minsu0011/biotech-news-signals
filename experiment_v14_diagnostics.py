"""Outcome-safe diagnostics on V14 DEV and already-opened prior sources."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


V14_DEV = {"SEC_V14_DEV", "NAVER_NEWS_V14_DEV"}
V13_OPENED = {"SEC_V13_DEV", "SEC_V13_SEAL", "NAVER_NEWS_V13_DEV", "NAVER_NEWS_V13_SEAL"}


def feature_auc(frame: pd.DataFrame, market: str) -> dict[str, float]:
    part = frame[frame.market.eq(market)].copy()
    features = app.model_frame(part)
    output: dict[str, float] = {}
    for column in app.NUMERIC_COLS:
        values = pd.to_numeric(features[column], errors="coerce")
        valid = values.notna()
        if valid.sum() < 40 or values[valid].nunique() < 3 or part.loc[valid, "y"].nunique() < 2:
            continue
        output[column] = float(roc_auc_score(part.loc[valid, "y"], values[valid]))
    return output


def main() -> None:
    frame = app.load_search_frame(app.Config())
    frame = frame[frame.y.notna()].copy()
    sets = {
        "v14_dev": frame[frame.source.isin(V14_DEV)],
        "v13_opened": frame[frame.source.isin(V13_OPENED)],
        "all_prior": frame[~frame.source.isin(V14_DEV)],
    }
    for name, part in sets.items():
        print("SET", name, "n", len(part))
        print(part.groupby("market").y.agg(["size", "mean"]).to_string())
    for market in ("US", "KR"):
        scores = {name: feature_auc(part, market) for name, part in sets.items()}
        common = set.intersection(*(set(values) for values in scores.values()))
        ranking = []
        for column in common:
            current = scores["v14_dev"][column]
            v13 = scores["v13_opened"][column]
            prior = scores["all_prior"][column]
            current_sign = np.sign(current - 0.5)
            consistent = current_sign == np.sign(v13 - 0.5) == np.sign(prior - 0.5)
            strength = min(abs(current - 0.5), abs(v13 - 0.5), abs(prior - 0.5))
            ranking.append((consistent, strength, column, current, v13, prior))
        ranking.sort(reverse=True)
        print("\nMARKET", market, "CONSISTENT FEATURES")
        for row in ranking[:30]:
            print(row)


if __name__ == "__main__":
    main()
