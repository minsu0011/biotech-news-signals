#!/usr/bin/env python
"""Repeated, production-shaped DEV validation of the legitimate V13 rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app
from experiment_v13_fixed_rules import RULES, rank


def groups_for(part: pd.DataFrame, market: str) -> pd.Series:
    timestamp = pd.to_datetime(part.event_time_utc, utc=True)
    if market == "KR":
        return part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    return part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")


def market_oof(part: pd.DataFrame, features: pd.DataFrame, market: str, seed: int) -> np.ndarray:
    groups = groups_for(part, market)
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(part, part.y, groups)
    output = np.full(len(part), np.nan)
    rule = RULES[market]["kr_robust5" if market == "KR" else "us_positive5"]
    for train_index, valid_index in folds:
        technical = []
        for column, sign in rule:
            train_value = sign * pd.to_numeric(features.iloc[train_index][column], errors="coerce").to_numpy(float)
            valid_value = sign * pd.to_numeric(features.iloc[valid_index][column], errors="coerce").to_numpy(float)
            technical.append(rank(train_value, valid_value))
        technical_score = np.mean(technical, axis=0)
        if market == "KR":
            vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=2,
                max_features=50000, sublinear_tf=True,
            )
            train_text = vectorizer.fit_transform(part.iloc[train_index].text.fillna(""))
            model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=260825,
            ).fit(train_text, part.iloc[train_index].y)
            valid_text_score = 1.0 - model.predict_proba(
                vectorizer.transform(part.iloc[valid_index].text.fillna(""))
            )[:, 1]
            output[valid_index] = 0.6 * technical_score + 0.4 * valid_text_score
        else:
            output[valid_index] = technical_score
    return output


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE, dtype={"ticker": str}, low_memory=False)
    raw = raw[raw.source.isin({"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"})].copy().reset_index(drop=True)
    features = app.model_frame(raw)
    kr_mask = raw.market.eq("KR").to_numpy()
    probabilities = []
    direction_scores = []
    for seed in (260825, 101, 202, 303, 404, 505, 606):
        score = np.empty(len(raw))
        for market, mask in (("KR", kr_mask), ("US", ~kr_mask)):
            score[mask] = market_oof(
                raw.loc[mask].reset_index(drop=True),
                features.loc[mask].reset_index(drop=True), market, seed,
            )
        probability = score.copy()
        probability[kr_mask] = app.shift_probability(probability[kr_mask], 0.69)
        probability[~kr_mask] = app.shift_probability(probability[~kr_mask], 0.58)
        frame = raw[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
        frame["prob"] = probability
        frame["eval_weight"] = np.where(kr_mask, 761.0 / kr_mask.sum(), 350.0 / (~kr_mask).sum())
        metrics = app.evaluate(frame, 0.70, 0.002)
        print(json.dumps({
            "seed": seed, "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"], "auc": metrics["auc"],
            "edge": metrics["edge_vs_naive"],
            "KR_bal": metrics["by_market"]["KR"]["balanced_accuracy"],
            "US_bal": metrics["by_market"]["US"]["balanced_accuracy"],
        }))
        probabilities.append(probability)
        direction_scores.append(score)
    averaged_score = np.mean(direction_scores, axis=0)
    output = raw[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
    output["raw_score"] = averaged_score
    output["prob"] = np.mean(probabilities, axis=0)
    output["eval_weight"] = np.where(kr_mask, 761.0 / kr_mask.sum(), 350.0 / (~kr_mask).sum())
    print("AVERAGE", json.dumps(app.evaluate(output, 0.70, 0.002), indent=2))
    output.to_parquet(app.CACHE / "v13_legitimate_repeated_oof.parquet", index=False)


if __name__ == "__main__":
    main()
