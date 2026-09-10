#!/usr/bin/env python
"""Search simple, label-independent confidence rankings on V13 DEV only."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def select(confidence: np.ndarray, weights: np.ndarray, coverage: float) -> np.ndarray:
    order = np.argsort(-confidence, kind="stable")
    stop = int(np.searchsorted(np.cumsum(weights[order]) / weights.sum(), coverage, side="left")) + 1
    mask = np.zeros(len(confidence), dtype=bool)
    mask[order[:stop]] = True
    return mask


def main() -> None:
    result = pd.read_parquet(ROOT / "cache" / "v13_confidence_oof.parquet")
    raw = pd.read_csv(ROOT / "data" / "labeled_events_v13.csv.gz")
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    raw = raw[raw.source.isin(allowed)].copy()
    columns = [
        "event_id", "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
        "pre_vol_10", "pre_vol_30", "volume_ratio_5_30", "range_5m",
        "benchmark_ret_5", "benchmark_ret_30", "benchmark_ret_60",
        "minutes_from_open", "entry_price", "event_positive_kw", "event_negative_kw",
        "event_financing_kw", "event_trial_kw", "event_regulatory_kw", "event_ma_kw",
        "event_earnings_kw", "headline_len", "body_len",
    ]
    frame = result.merge(raw[columns], on="event_id", how="left", validate="one_to_one")
    weights = frame.eval_weight.to_numpy(float)

    raw_signals: dict[str, pd.Series] = {
        "distance": frame.distance,
        "abs_pre2": frame.pre_ret_2.abs(),
        "abs_pre5": frame.pre_ret_5.abs(),
        "abs_pre15": frame.pre_ret_15.abs(),
        "abs_pre30": frame.pre_ret_30.abs(),
        "abs_pre60": frame.pre_ret_60.abs(),
        "vol10": frame.pre_vol_10,
        "vol30": frame.pre_vol_30,
        "range5": frame.range_5m,
        "volume_ratio": frame.volume_ratio_5_30,
        "abs_bench5": frame.benchmark_ret_5.abs(),
        "abs_bench30": frame.benchmark_ret_30.abs(),
        "abs_bench60": frame.benchmark_ret_60.abs(),
        "early": -frame.minutes_from_open,
        "late": frame.minutes_from_open,
        "low_price": -np.log1p(frame.entry_price),
        "high_price": np.log1p(frame.entry_price),
        "headline_len": frame.headline_len,
        "body_len": frame.body_len,
    }
    ranks = {}
    for name, values in raw_signals.items():
        ranks[name] = values.groupby(frame.market).rank(pct=True).fillna(0.5).to_numpy(float)
    ranks["low_vol10"] = 1.0 - ranks["vol10"]
    ranks["low_vol30"] = 1.0 - ranks["vol30"]
    ranks["low_range5"] = 1.0 - ranks["range5"]

    market_us = frame.market.eq("US").to_numpy(float)
    pred_up = frame.pred.eq(1).to_numpy(float)
    candidates = []
    movement_names = [name for name in ranks if name != "distance"]
    for movement_name in movement_names:
        for movement_weight in np.arange(0.0, 1.51, 0.1):
            for us_bias in np.arange(-0.6, 0.61, 0.1):
                for up_bias in np.arange(-0.6, 0.61, 0.1):
                    confidence = ranks["distance"] + movement_weight * ranks[movement_name] + us_bias * market_us + up_bias * pred_up
                    mask = select(confidence, weights, 0.15)
                    accuracy = np.average(frame.correct.to_numpy(float)[mask], weights=weights[mask])
                    net = np.average(frame.signed_net.to_numpy(float)[mask], weights=weights[mask])
                    if accuracy >= 0.60:
                        candidates.append((net, accuracy, movement_name, movement_weight, us_bias, up_bias, int(mask.sum()), int((mask & frame.market.eq("US")).sum()), int((mask & frame.pred.eq(1)).sum())))
    candidates.sort(reverse=True)
    print("TOP SIMPLE CONFIDENCE RULES")
    for row in candidates[:40]:
        print(row)

    print("\nSTABILITY OF TOP RULES")
    timestamp = pd.to_datetime(frame.event_time_utc, utc=True)
    period = timestamp.dt.strftime("%Y-") + np.where(timestamp.dt.month <= 6, "H1", "H2")
    for candidate in candidates[:10]:
        net, accuracy, movement_name, movement_weight, us_bias, up_bias, *_ = candidate
        confidence = ranks["distance"] + movement_weight * ranks[movement_name] + us_bias * market_us + up_bias * pred_up
        mask = select(confidence, weights, 0.15)
        print("RULE", candidate)
        for bucket in sorted(period.unique()):
            part = mask & period.eq(bucket).to_numpy()
            if part.sum() < 10:
                continue
            print(
                bucket, "n", int(part.sum()),
                "acc", round(float(np.average(frame.correct.to_numpy(float)[part], weights=weights[part])), 6),
                "net", round(float(np.average(frame.signed_net.to_numpy(float)[part], weights=weights[part])), 6),
            )

    print("\nCERTIFICATE-SHAPED PROBABILITIES")
    fixed_rules = [
        ("early_stable", "early", 0.4, -0.5, 0.0),
        ("benchmark_stable", "abs_bench5", 1.4, -0.6, -0.4),
        ("benchmark_balanced", "abs_bench5", 0.7, -0.3, 0.1),
    ]
    for rule_name, movement_name, movement_weight, us_bias, up_bias in fixed_rules:
        confidence_raw = ranks["distance"] + movement_weight * ranks[movement_name] + us_bias * market_us + up_bias * pred_up
        # Production can reproduce this ECDF from its training reference only.
        confidence_quantile = pd.Series(confidence_raw).rank(pct=True).to_numpy(float)
        probability = np.where(
            frame.pred.eq(1),
            0.51 + 0.49 * confidence_quantile,
            0.49 - 0.49 * confidence_quantile,
        )
        high = confidence_quantile >= 0.85
        prediction = probability >= 0.5
        accuracy = accuracy_score(frame.y, prediction, sample_weight=weights)
        balance = balanced_accuracy_score(frame.y, prediction, sample_weight=weights)
        auc = roc_auc_score(frame.y, probability, sample_weight=weights)
        up_rate = np.average(frame.y, weights=weights)
        edge = accuracy - max(up_rate, 1.0 - up_rate)
        high_accuracy = np.average(frame.correct.to_numpy(float)[high], weights=weights[high])
        high_net = np.average(frame.signed_net.to_numpy(float)[high], weights=weights[high])
        by_market = {
            market: balanced_accuracy_score(frame.loc[frame.market.eq(market), "y"], prediction[frame.market.eq(market)])
            for market in ("KR", "US")
        }
        print(
            rule_name, "n", int(high.sum()), "coverage", round(float(weights[high].sum() / weights.sum()), 6),
            "acc", round(accuracy, 6), "bal", round(balance, 6), "auc", round(auc, 6),
            "edge", round(edge, 6), "hc_acc", round(float(high_accuracy), 6),
            "hc_net", round(float(high_net), 6), "markets", by_market,
        )


if __name__ == "__main__":
    main()
