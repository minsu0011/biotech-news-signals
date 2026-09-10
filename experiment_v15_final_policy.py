"""Evaluate the exact V15 probability/confidence transform on DEV OOF only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, na_option="bottom").to_numpy(float)


def transform(part: pd.DataFrame, market: str, gate_quantile: float) -> pd.DataFrame:
    raw = part.prob.to_numpy(float)
    margin = rank(np.abs(raw - .53))
    if market == "US":
        opportunity = rank(pd.to_numeric(part.vwap_distance_30, errors="coerce").fillna(0).to_numpy(float))
    else:
        trend = rank(pd.to_numeric(part.trend_slope_30, errors="coerce").fillna(0).to_numpy(float))
        opportunity = rank(.5 * margin + .5 * trend)
    high = opportunity >= gate_quantile
    confidence = np.where(high, .86 + .14 * margin, .84 * margin)
    probability = np.where(raw >= .53, .5 + .5 * confidence, .5 - .5 * confidence)
    out = part.copy()
    out["prob"] = probability
    return out


def fast_boot_lower(frame: pd.DataFrame, nboot: int = 3000) -> float:
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    codes, unique = pd.factorize(day)
    labels = frame.y.astype(int).to_numpy()
    pred = frame.prob.ge(.5).astype(int).to_numpy()
    base_weight = frame.eval_weight.to_numpy(float)
    rng = np.random.default_rng(app.SEED + 99)
    values = []
    for _ in range(nboot):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = base_weight * counts[codes]
        pos = labels == 1
        neg = ~pos
        if weight[pos].sum() == 0 or weight[neg].sum() == 0:
            continue
        tpr = weight[pos & (pred == 1)].sum() / weight[pos].sum()
        tnr = weight[neg & (pred == 0)].sum() / weight[neg].sum()
        values.append(.5 * (tpr + tnr))
    return float(np.quantile(values, .025))


def main() -> None:
    pred = pd.read_parquet(app.CACHE / "v15_repeated_oof.parquet")
    selected = pd.concat([
        pred[pred.market.eq("US") & ~pred.augment & pred.model.eq("native")],
        pred[pred.market.eq("KR") & pred.augment & pred.model.eq("logit001")],
    ], ignore_index=True)
    data = app.load_search_frame(app.Config())
    data = data[data.source.isin({"SEC_V15_DEV", "KIND_V15_DEV"}) & data.y.notna()].copy()
    features = app.model_frame(data)
    features["event_id"] = data.event_id.to_numpy()
    merged = selected.merge(features[["event_id", "vwap_distance_30", "trend_slope_30"]], on="event_id", how="left")
    for us_gate in (.80, .81, .82, .83, .84):
        for kr_gate in (.83, .84, .85, .86):
            us = transform(merged[merged.market.eq("US")].copy(), "US", us_gate)
            kr = transform(merged[merged.market.eq("KR")].copy(), "KR", kr_gate)
            us["eval_weight"] = 448.0 / len(us)
            kr["eval_weight"] = 1521.0 / len(kr)
            combined = pd.concat([us, kr], ignore_index=True)
            metrics = app.evaluate(combined, .925, .002)
            boot_lower = fast_boot_lower(combined)
            print(json.dumps({
                "us_gate": us_gate, "kr_gate": kr_gate,
                "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                "auc": metrics["auc"], "edge": metrics["edge_vs_naive"],
                "hc_coverage": metrics["highconf_coverage"],
                "hc_accuracy": metrics["highconf_accuracy"],
                "hc_net": metrics["strategy_mean_signed_net"],
                "boot_lower": boot_lower,
                "US_bal": metrics["by_market"]["US"]["balanced_accuracy"],
                "KR_bal": metrics["by_market"]["KR"]["balanced_accuracy"],
            }), flush=True)


if __name__ == "__main__":
    main()
