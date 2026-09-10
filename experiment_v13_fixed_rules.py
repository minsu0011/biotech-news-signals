#!/usr/bin/env python
"""Fold-safe DEV evaluation of fixed label-free V13 rank rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


RULES = {
    "KR": {
        "kr_robust2": (("entry_bar_ret", -1.0), ("pre_ret_15", -1.0)),
        "kr_robust5": (
            ("pre_ret_20", -1.0), ("pre_ret_5", -1.0), ("entry_bar_ret", -1.0),
            ("benchmark_ret_60", -1.0), ("pre_ret_15", -1.0),
        ),
        "kr_auc5": (
            ("pre_ret_20", -1.0), ("pre_ret_5", -1.0), ("benchmark_ret_60", -1.0),
            ("pre_ret_120", -1.0), ("volume_ratio_10_60", 1.0),
        ),
        "kr_auc4": (
            ("close_position_10", -1.0), ("benchmark_ret_60", -1.0),
            ("pre_ret_120", -1.0), ("volume_ratio_10_60", 1.0),
        ),
    },
    "US": {
        "us_robust5": (
            ("pre_ret_90", -1.0), ("benchmark_ret_30", 1.0), ("trend_slope_30", 1.0),
            ("volume_ratio_5_30", 1.0), ("event_ma_kw", 1.0),
        ),
        "us_positive5": (
            ("pre_ret_90", -1.0), ("benchmark_ret_30", 1.0), ("trend_slope_30", 1.0),
            ("volume_ratio_5_30", 1.0), ("event_positive_kw", 1.0),
        ),
        "us_simple3": (
            ("pre_ret_90", -1.0), ("benchmark_ret_30", 1.0), ("trend_slope_30", 1.0),
        ),
    },
}


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), 0.5)
    valid = np.isfinite(values)
    output[valid] = np.searchsorted(reference, values[valid], side="right") / len(reference)
    return output


def oof_rule(part: pd.DataFrame, features: pd.DataFrame, market: str, rule: tuple) -> np.ndarray:
    timestamp = pd.to_datetime(part.event_time_utc, utc=True)
    if market == "KR":
        groups = part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    else:
        groups = part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=260825).split(part, part.y, groups)
    prediction = np.full(len(part), np.nan)
    for train_index, valid_index in folds:
        components = []
        for column, sign in rule:
            train = sign * pd.to_numeric(features.iloc[train_index][column], errors="coerce").to_numpy(float)
            valid = sign * pd.to_numeric(features.iloc[valid_index][column], errors="coerce").to_numpy(float)
            components.append(rank(train, valid))
        prediction[valid_index] = np.mean(components, axis=0)
    return prediction


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

    candidates = []
    cfg = app.Config(fixed_highconf_threshold=None)
    for kr_name in RULES["KR"]:
        for us_name in RULES["US"]:
            score = np.full(len(raw), np.nan)
            score[raw.market.eq("KR")] = predictions[("KR", kr_name)]
            score[raw.market.eq("US")] = predictions[("US", us_name)]
            for kr_cutoff in np.arange(0.48, 0.751, 0.02):
                for us_cutoff in np.arange(0.45, 0.751, 0.02):
                    probability = score.copy()
                    for market, cutoff in (("KR", kr_cutoff), ("US", us_cutoff)):
                        positions = raw.market.eq(market).to_numpy()
                        probability[positions] = app.shift_probability(probability[positions], cutoff)
                    frame = raw[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
                    frame["prob"] = probability
                    frame["eval_weight"] = np.where(
                        frame.market.eq("KR"), 761.0 / frame.market.eq("KR").sum(),
                        350.0 / frame.market.eq("US").sum(),
                    )
                    # Fixed, small threshold set avoids an expensive/noisy continuum search.
                    for highconf in (0.60, 0.65, 0.70, 0.75, 0.80):
                        metrics = app.evaluate(frame, highconf, 0.002)
                        passed, selection_score = app.dev_selection_score(metrics, cfg)
                        candidates.append((passed, selection_score, kr_name, us_name, kr_cutoff, us_cutoff, highconf, metrics))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    for row in candidates[:30]:
        passed, selection_score, kr_name, us_name, kr_cutoff, us_cutoff, highconf, metrics = row
        print(json.dumps({
            "passes": passed, "score": selection_score, "KR": kr_name, "US": us_name,
            "kr_cutoff": kr_cutoff, "us_cutoff": us_cutoff, "highconf": highconf,
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "auc": metrics["auc"], "edge": metrics["edge_vs_naive"],
            "coverage": metrics["highconf_coverage"], "hc_accuracy": metrics["highconf_accuracy"],
            "hc_net": metrics["strategy_mean_signed_net"],
            "KR_bal": metrics["by_market"]["KR"]["balanced_accuracy"],
            "US_bal": metrics["by_market"]["US"]["balanced_accuracy"],
        }))
    best = candidates[0]
    _, _, kr_name, us_name, kr_cutoff, us_cutoff, highconf, _ = best
    output = raw[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
    output["raw_score"] = np.where(
        raw.market.eq("KR"), predictions[("KR", kr_name)], predictions[("US", us_name)]
    )
    output["prob"] = output.raw_score
    output.loc[output.market.eq("KR"), "prob"] = app.shift_probability(output.loc[output.market.eq("KR"), "prob"], kr_cutoff)
    output.loc[output.market.eq("US"), "prob"] = app.shift_probability(output.loc[output.market.eq("US"), "prob"], us_cutoff)
    output["eval_weight"] = np.where(output.market.eq("KR"), 761.0 / output.market.eq("KR").sum(), 350.0 / output.market.eq("US").sum())
    output.to_parquet(app.CACHE / "v13_fixed_rule_oof.parquet", index=False)


if __name__ == "__main__":
    main()
