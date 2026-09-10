"""DEV-only search for outcome-independent high-confidence opportunity scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, na_option="bottom").to_numpy(float)


def main() -> None:
    predictions = pd.read_parquet(app.CACHE / "v14_repeated_oof.parquet")
    us = predictions[
        predictions.market.eq("US") & predictions.augment_v13 & predictions.model.eq("blend")
    ].copy()
    kr = predictions[
        predictions.market.eq("KR") & ~predictions.augment_v13 & predictions.model.eq("cat3")
    ].copy()
    selected = pd.concat([us, kr], ignore_index=True)
    frame = app.load_search_frame(app.Config())
    frame = frame[frame.source.isin({"SEC_V14_DEV", "NAVER_NEWS_V14_DEV"}) & frame.y.notna()].copy()
    feature_frame = app.model_frame(frame)
    feature_frame["event_id"] = frame.event_id.to_numpy()
    data = selected.merge(feature_frame[["event_id"] + app.NUMERIC_COLS], on="event_id", how="left")
    cutoffs = np.where(data.market.eq("US"), .485, .545)
    raw_probability = data.prob.to_numpy(float)
    direction = raw_probability >= cutoffs
    data["signed_net"] = np.where(direction, 1.0, -1.0) * data.fwd_ret_30m.to_numpy(float) - .002
    data["correct"] = direction == data.y.to_numpy(int)
    rows = []
    for market in ("US", "KR"):
        positions = np.flatnonzero(data.market.eq(market).to_numpy())
        part = data.iloc[positions].reset_index(drop=True)
        margin = rank(np.abs(raw_probability[positions] - cutoffs[positions]))
        candidates = {"model_margin": margin}
        for column in app.NUMERIC_COLS:
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(float)
            if np.isfinite(values).sum() < max(40, len(part) // 2):
                continue
            fill = np.nanmedian(values) if np.isfinite(values).any() else 0.0
            values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
            transforms = {
                f"+{column}": rank(values),
                f"-{column}": rank(-values),
                f"abs_{column}": rank(np.abs(values)),
            }
            for name, opportunity in transforms.items():
                for margin_weight in (0.0, .25, .5, .75):
                    candidates[f"{name}|m{margin_weight}"] = (
                        margin_weight * margin + (1.0 - margin_weight) * opportunity
                    )
        for name, confidence in candidates.items():
            for coverage in (.15, .20, .25, .30):
                threshold = np.quantile(confidence, 1.0 - coverage)
                mask = confidence >= threshold
                rows.append({
                    "market": market, "name": name, "target_coverage": coverage,
                    "actual_coverage": float(mask.mean()), "n": int(mask.sum()),
                    "accuracy": float(part.loc[mask, "correct"].mean()),
                    "mean_signed_net": float(part.loc[mask, "signed_net"].mean()),
                })
    results = pd.DataFrame(rows)
    results["pass_count"] = (
        results.accuracy.ge(.60).astype(int) + results.mean_signed_net.gt(0).astype(int)
    )
    for market in ("US", "KR"):
        print("\nMARKET", market)
        top = results[results.market.eq(market)].sort_values(
            ["pass_count", "mean_signed_net", "accuracy"], ascending=False,
        ).head(40)
        print(top.to_string(index=False))
    results.to_csv(app.CACHE / "v14_confidence_search.csv", index=False)


if __name__ == "__main__":
    main()
