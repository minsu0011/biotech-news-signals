#!/usr/bin/env python
"""Fast DEV direction-cutoff grid for fixed label-free V13 rules."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app
from experiment_v13_fixed_rules import RULES, oof_rule


def balance(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    positive = y == 1
    sensitivity = weight[positive & prediction].sum() / weight[positive].sum()
    specificity = weight[(~positive) & (~prediction)].sum() / weight[~positive].sum()
    return float((sensitivity + specificity) / 2.0)


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE, dtype={"ticker": str}, low_memory=False)
    raw = raw[raw.source.isin({"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"})].copy().reset_index(drop=True)
    features = app.model_frame(raw)
    predictions = {}
    for market in ("KR", "US"):
        mask = raw.market.eq(market).to_numpy()
        part = raw.loc[mask].reset_index(drop=True)
        x = features.loc[mask].reset_index(drop=True)
        for name, rule in RULES[market].items():
            predictions[(market, name)] = oof_rule(part, x, market, rule)
    kr_rows = raw[raw.market.eq("KR")][["event_id"]].copy().reset_index(drop=True)
    text = pd.read_parquet(app.CACHE / "v13_kr_text_oof.parquet", columns=["event_id", "text_word_1"])
    text = kr_rows.merge(text, on="event_id", how="left", validate="one_to_one")
    inverse_text = (1.0 - text.text_word_1).rank(pct=True).fillna(0.5).to_numpy(float)
    kr_names = list(RULES["KR"])
    for base_name in ("kr_robust2", "kr_robust5", "kr_auc5", "kr_auc4"):
        for text_weight in (0.2, 0.4, 0.6):
            name = f"{base_name}_text{text_weight:.1f}"
            predictions[("KR", name)] = (
                (1.0 - text_weight) * predictions[("KR", base_name)]
                + text_weight * inverse_text
            )
            kr_names.append(name)
    market_kr = raw.market.eq("KR").to_numpy()
    y = raw.y.to_numpy(int)
    weight = np.where(market_kr, 761.0 / market_kr.sum(), 350.0 / (~market_kr).sum())
    up_rate = np.average(y, weights=weight)
    candidates = []
    for kr_name in kr_names:
        for us_name in RULES["US"]:
            score = np.empty(len(raw))
            score[market_kr] = predictions[("KR", kr_name)]
            score[~market_kr] = predictions[("US", us_name)]
            for kr_cutoff in np.arange(0.40, 0.901, 0.01):
                kr_prediction = score[market_kr] >= kr_cutoff
                kr_balance = balance(y[market_kr], kr_prediction, np.ones(market_kr.sum()))
                if kr_balance < 0.51:
                    continue
                for us_cutoff in np.arange(0.40, 0.901, 0.01):
                    prediction = np.empty(len(raw), dtype=bool)
                    prediction[market_kr] = kr_prediction
                    prediction[~market_kr] = score[~market_kr] >= us_cutoff
                    us_balance = balance(y[~market_kr], prediction[~market_kr], np.ones((~market_kr).sum()))
                    if us_balance < 0.51:
                        continue
                    accuracy = float(np.average(prediction == y, weights=weight))
                    balanced = balance(y, prediction, weight)
                    edge = accuracy - max(up_rate, 1.0 - up_rate)
                    if balanced < 0.54:
                        continue
                    probability = score.copy()
                    probability[market_kr] = app.shift_probability(probability[market_kr], kr_cutoff)
                    probability[~market_kr] = app.shift_probability(probability[~market_kr], us_cutoff)
                    auc = roc_auc_score(y, probability, sample_weight=weight)
                    candidates.append((edge, balanced, auc, accuracy, kr_name, us_name, kr_cutoff, us_cutoff, kr_balance, us_balance))
    candidates.sort(reverse=True)
    print("TOP EDGE")
    for row in candidates[:50]:
        print(row)
    print("DIRECTION PASS")
    passed = [row for row in candidates if row[0] >= 0.02 and row[2] >= 0.56]
    for row in passed[:50]:
        print(row)
    print("count", len(passed))


if __name__ == "__main__":
    main()
