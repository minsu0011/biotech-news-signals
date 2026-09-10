"""V15 DEV-only stability audit for a small predeclared confidence shortlist."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, na_option="bottom").to_numpy(float)


def hash_bit(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), 16) % 2


def candidate_score(part: pd.DataFrame, name: str) -> np.ndarray:
    margin = rank(np.abs(part["prob"].to_numpy(float) - 0.53))
    if name == "margin":
        return margin
    column, weight_text = name.split("|m")
    weight = float(weight_text)
    values = pd.to_numeric(part[column], errors="coerce").to_numpy(float)
    fill = np.nanmedian(values) if np.isfinite(values).any() else 0.0
    values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
    return rank(weight * margin + (1.0 - weight) * rank(values))


def metrics(part: pd.DataFrame, name: str, coverage: float) -> tuple[int, float, float]:
    if len(part) < 20:
        return 0, float("nan"), float("nan")
    score = candidate_score(part, name)
    chosen = score >= 1.0 - coverage
    return (
        int(chosen.sum()),
        float(part.loc[chosen, "correct"].mean()),
        float(part.loc[chosen, "signed_net"].mean()),
    )


def main() -> None:
    predictions = pd.read_parquet(app.CACHE / "v15_repeated_oof.parquet")
    selected = pd.concat([
        predictions[predictions.market.eq("US") & ~predictions.augment & predictions.model.eq("native")],
        predictions[predictions.market.eq("KR") & predictions.augment & predictions.model.eq("logit001")],
    ], ignore_index=True)
    data = app.load_search_frame(app.Config())
    data = data[data.source.isin({"SEC_V15_DEV", "KIND_V15_DEV"}) & data.y.notna()].copy()
    features = app.model_frame(data)
    features["event_id"] = data.event_id.to_numpy()
    merged = selected.merge(features[["event_id"] + app.NUMERIC_COLS], on="event_id", how="left")
    merged["event_time_utc"] = pd.to_datetime(merged.event_time_utc, utc=True)
    merged["correct"] = (merged.prob.ge(.53) == merged.y.astype(int))
    merged["signed_net"] = np.where(merged.prob.ge(.53), 1.0, -1.0) * merged.fwd_ret_30m - .002
    merged["event_hash"] = [hash_bit(x, "event") for x in merged.event_id.astype(str)]
    merged["ticker_hash"] = [hash_bit(x, "ticker") for x in merged.ticker.astype(str)]

    shortlist = {
        "US": ["vwap_distance_30|m0.0", "vwap_distance_30|m0.25", "close_position_10|m0.0", "trend_slope_30|m0.5", "margin"],
        "KR": ["pre_vol_60|m0.0", "trend_slope_30|m0.5", "entry_bar_range|m0.0", "margin"],
    }
    rows = []
    for market, names in shortlist.items():
        part = merged[merged.market.eq(market)].sort_values("event_time_utc").reset_index(drop=True)
        midpoint = part.event_time_utc.quantile(.5)
        groups = {
            "all": np.ones(len(part), dtype=bool),
            "time_early": part.event_time_utc.le(midpoint).to_numpy(),
            "time_late": part.event_time_utc.gt(midpoint).to_numpy(),
            "event_hash_0": part.event_hash.eq(0).to_numpy(),
            "event_hash_1": part.event_hash.eq(1).to_numpy(),
            "ticker_hash_0": part.ticker_hash.eq(0).to_numpy(),
            "ticker_hash_1": part.ticker_hash.eq(1).to_numpy(),
        }
        for year in sorted(part.event_time_utc.dt.year.unique()):
            mask = part.event_time_utc.dt.year.eq(year).to_numpy()
            if mask.sum() >= 50:
                groups[f"year_{year}"] = mask
        for name in names:
            for coverage in (.15, .18, .20, .25):
                for group, mask in groups.items():
                    n, accuracy, net = metrics(part.loc[mask].reset_index(drop=True), name, coverage)
                    rows.append({
                        "market": market, "name": name, "coverage": coverage,
                        "group": group, "group_n": int(mask.sum()), "selected_n": n,
                        "accuracy": accuracy, "mean_signed_net": net,
                    })
    result = pd.DataFrame(rows)
    result.to_csv(app.CACHE / "v15_confidence_stability.csv", index=False)
    for market in ("US", "KR"):
        print(f"\n{market} SUMMARY")
        z = result[result.market.eq(market)].groupby(["name", "coverage"]).agg(
            all_accuracy=("accuracy", lambda x: float(x.iloc[0])),
            min_accuracy=("accuracy", "min"), median_accuracy=("accuracy", "median"),
            all_net=("mean_signed_net", lambda x: float(x.iloc[0])),
            min_net=("mean_signed_net", "min"), positive_net_groups=("mean_signed_net", lambda x: int((x > 0).sum())),
            groups=("group", "count"),
        ).reset_index()
        print(z.sort_values(["all_accuracy", "min_accuracy", "all_net"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
