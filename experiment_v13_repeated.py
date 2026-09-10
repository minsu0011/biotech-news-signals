#!/usr/bin/env python
"""Repeated grouped DEV-only validation of the frozen-candidate V13 rule."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


def market_oof(source: pd.DataFrame, market: str, seed: int) -> pd.DataFrame:
    source = source[source.market.eq(market)].copy().sort_values("event_time_utc").reset_index(drop=True)
    timestamp = pd.to_datetime(source.event_time_utc, utc=True)
    if market == "KR":
        groups = source.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    else:
        groups = source.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    prediction = np.full(len(source), np.nan)
    direction = np.full(len(source), np.nan)
    fold = np.zeros(len(source), dtype=int)
    for fold_number, (train_index, valid_index) in enumerate(splitter.split(source, source.y, groups), 1):
        model = app.make_model(app.model_specs(market)[0])
        model.fit(app.model_frame(source.iloc[train_index]), source.iloc[train_index].y)
        valid_frame = app.model_frame(source.iloc[valid_index])
        prediction[valid_index] = model.predict_proba(valid_frame)[:, 1]
        direction[valid_index] = model.direction_score(valid_frame)
        fold[valid_index] = fold_number
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"missing {market} OOF")
    output = source[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
    output["prob"] = prediction
    output["direction_raw"] = direction
    output["fold"] = fold
    target = 761.0 if market == "KR" else 350.0
    output["eval_weight"] = target / len(output)
    return output


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE)
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    dev = raw[raw.source.isin(allowed)].copy()
    del raw
    results = []
    prediction_sum = None
    vote_sum = None
    direction_sum = None
    prediction_count = 0
    reference = None
    for seed in (260825, 101, 202, 303, 404, 505, 606):
        combined = pd.concat([
            market_oof(dev, "KR", seed),
            market_oof(dev, "US", seed),
        ], ignore_index=True)
        combined = combined.sort_values("event_id").reset_index(drop=True)
        if reference is None:
            reference = combined.drop(columns=["prob", "direction_raw"]).copy()
            prediction_sum = combined.prob.to_numpy(float)
            vote_sum = combined.prob.ge(0.5).to_numpy(float)
            direction_sum = combined.direction_raw.to_numpy(float)
        else:
            if not combined.event_id.equals(reference.event_id):
                raise RuntimeError("repeated OOF alignment changed")
            prediction_sum += combined.prob.to_numpy(float)
            vote_sum += combined.prob.ge(0.5).to_numpy(float)
            direction_sum += combined.direction_raw.to_numpy(float)
        prediction_count += 1
        metrics = app.evaluate(combined, 0.89, 0.002)
        results.append({
            "seed": seed,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "auc": metrics["auc"],
            "edge_vs_naive": metrics["edge_vs_naive"],
            "coverage": metrics["highconf_coverage"],
            "highconf_accuracy": metrics["highconf_accuracy"],
            "highconf_net": metrics["strategy_mean_signed_net"],
            "KR_bal": metrics["by_market"]["KR"]["balanced_accuracy"],
            "US_bal": metrics["by_market"]["US"]["balanced_accuracy"],
        })
        print(json.dumps(results[-1], ensure_ascii=False))
    numeric = pd.DataFrame(results).drop(columns="seed")
    print("MIN")
    print(numeric.min().to_string())
    print("MEAN")
    print(numeric.mean().to_string())
    averaged = reference.copy()
    averaged["prob"] = prediction_sum / prediction_count
    averaged["vote_up"] = vote_sum / prediction_count
    averaged["direction_raw"] = direction_sum / prediction_count
    averaged_metrics = app.evaluate(averaged, 0.89, 0.002)
    print("AVERAGED CROSS-FIT")
    print(json.dumps(averaged_metrics, ensure_ascii=False, indent=2))
    averaged.to_parquet(app.CACHE / "v13_repeated_oof.parquet", index=False)


if __name__ == "__main__":
    main()
