#!/usr/bin/env python
"""DEV-only confidence search for the legitimate fixed-rule V13 candidate."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app
from experiment_v13_fixed_rules import RULES, oof_rule


def select(confidence: np.ndarray, weights: np.ndarray, coverage: float = 0.15) -> np.ndarray:
    order = np.argsort(-confidence, kind="stable")
    stop = int(np.searchsorted(np.cumsum(weights[order]) / weights.sum(), coverage, side="left")) + 1
    mask = np.zeros(len(confidence), dtype=bool)
    mask[order[:stop]] = True
    return mask


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE, dtype={"ticker": str}, low_memory=False)
    raw = raw[raw.source.isin({"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"})].copy().reset_index(drop=True)
    features = app.model_frame(raw)
    kr_mask = raw.market.eq("KR").to_numpy()
    kr = raw.loc[kr_mask].reset_index(drop=True)
    us = raw.loc[~kr_mask].reset_index(drop=True)
    kr_rule = oof_rule(kr, features.loc[kr_mask].reset_index(drop=True), "KR", RULES["KR"]["kr_robust5"])
    us_rule = oof_rule(us, features.loc[~kr_mask].reset_index(drop=True), "US", RULES["US"]["us_positive5"])
    text = pd.read_parquet(app.CACHE / "v13_kr_text_oof.parquet", columns=["event_id", "text_word_1"])
    inverse_text = (1.0 - kr[["event_id"]].merge(text, on="event_id").text_word_1).rank(pct=True).to_numpy(float)
    score = np.empty(len(raw))
    score[kr_mask] = 0.6 * kr_rule + 0.4 * inverse_text
    score[~kr_mask] = us_rule
    cutoff = np.where(kr_mask, 0.69, 0.58)
    prediction = score >= cutoff
    correct = prediction == raw.y.to_numpy(int)
    signed_net = np.where(prediction, 1.0, -1.0) * raw.fwd_ret_30m.to_numpy(float) - 0.002
    weights = np.where(kr_mask, 761.0 / kr_mask.sum(), 350.0 / (~kr_mask).sum())
    distance = pd.Series(np.abs(score - cutoff)).groupby(raw.market).rank(pct=True).to_numpy(float)
    market_us = (~kr_mask).astype(float)
    pred_up = prediction.astype(float)

    movement_raw = {
        "abs_bench2": features.benchmark_ret_2.abs(),
        "abs_bench5": features.benchmark_ret_5.abs(),
        "abs_bench15": features.benchmark_ret_15.abs(),
        "abs_bench30": features.benchmark_ret_30.abs(),
        "abs_pre5": features.pre_ret_5.abs(),
        "abs_pre15": features.pre_ret_15.abs(),
        "abs_pre30": features.pre_ret_30.abs(),
        "abs_pre60": features.pre_ret_60.abs(),
        "vol10": features.pre_vol_10,
        "vol30": features.pre_vol_30,
        "range10": features.range_10m,
        "range30": features.range_30m,
        "early": -features.minutes_from_open,
        "late": features.minutes_from_open,
        "volume": features.volume_ratio_5_30,
    }
    movement = {
        name: values.groupby(raw.market).rank(pct=True).fillna(0.5).to_numpy(float)
        for name, values in movement_raw.items()
    }
    candidates = []
    for name, values in movement.items():
        for factor in np.arange(0.0, 1.51, 0.1):
            for us_bias in np.arange(-0.6, 0.61, 0.1):
                for up_bias in np.arange(-0.6, 0.61, 0.1):
                    confidence = distance + factor * values + us_bias * market_us + up_bias * pred_up
                    chosen = select(confidence, weights)
                    accuracy = np.average(correct[chosen], weights=weights[chosen])
                    net = np.average(signed_net[chosen], weights=weights[chosen])
                    if accuracy >= 0.60:
                        candidates.append((net, accuracy, name, factor, us_bias, up_bias, int(chosen.sum()), int((chosen & ~kr_mask).sum()), int((chosen & prediction).sum())))
    candidates.sort(reverse=True)
    print("TOP")
    for row in candidates[:50]:
        print(row)
    timestamp = pd.to_datetime(raw.event_time_utc, utc=True)
    period = timestamp.dt.strftime("%Y-") + np.where(timestamp.dt.month <= 6, "H1", "H2")
    print("STABILITY")
    for row in candidates[:10]:
        _, _, name, factor, us_bias, up_bias, *_ = row
        confidence = distance + factor * movement[name] + us_bias * market_us + up_bias * pred_up
        chosen = select(confidence, weights)
        print("RULE", row)
        for bucket in sorted(period.unique()):
            part = chosen & period.eq(bucket).to_numpy()
            if part.sum() >= 10:
                print(bucket, int(part.sum()), float(np.average(correct[part], weights=weights[part])), float(np.average(signed_net[part], weights=weights[part])))
    # Save enough to reconstruct the selected candidate without labels.
    output = raw[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
    output["raw_score"] = score
    output["pred"] = prediction.astype(int)
    output["eval_weight"] = weights
    output.to_parquet(app.CACHE / "v13_legitimate_rule_oof.parquet", index=False)


if __name__ == "__main__":
    main()
