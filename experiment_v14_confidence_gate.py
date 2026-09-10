"""Evaluate a confidence gate that preserves model-margin ranking off the top 15%."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, na_option="bottom").to_numpy(float)


def main() -> None:
    predictions = pd.read_parquet(app.CACHE / "v14_repeated_oof.parquet")
    selected = pd.concat([
        predictions[predictions.market.eq("US") & ~predictions.augment_v13 & predictions.model.eq("blend")],
        predictions[predictions.market.eq("KR") & ~predictions.augment_v13 & predictions.model.eq("cat3")],
    ], ignore_index=True)
    data = app.load_search_frame(app.Config())
    data = data[data.source.isin({"SEC_V14_DEV", "NAVER_NEWS_V14_DEV"}) & data.y.notna()].copy()
    features = app.model_frame(data)
    features["event_id"] = data.event_id.to_numpy()
    selected = selected.merge(features[["event_id", "range_30m", "pre_ret_45"]], on="event_id")
    output = []
    for market, cutoff, column, sign, target in (
        ("US", .490, "range_30m", 1.0, 305.0),
        ("KR", .545, "pre_ret_45", -1.0, 1046.0),
    ):
        part = selected[selected.market.eq(market)].copy().reset_index(drop=True)
        raw = part.prob.to_numpy(float)
        direction = raw >= cutoff
        margin = rank(np.abs(raw - cutoff))
        opportunity = rank(sign * pd.to_numeric(part[column], errors="coerce").fillna(0).to_numpy(float))
        opportunity_score = rank(.25 * margin + .75 * opportunity)
        high = opportunity_score >= .80
        confidence = np.where(high, .86 + .14 * margin, .84 * margin)
        part["prob"] = np.where(direction, .5 + .5 * confidence, .5 - .5 * confidence)
        part["eval_weight"] = target / len(part)
        part["opportunity_high"] = high
        output.append(part)
        print(market, "opportunity_high", int(high.sum()), float(high.mean()))
    combined = pd.concat(output, ignore_index=True)
    metrics = app.evaluate(combined, .925, .002)
    bootstrap = app.bootstrap_by_day(combined, .925, .002, nboot=200)
    print(json.dumps({"metrics": metrics, "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
