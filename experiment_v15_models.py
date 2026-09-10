"""Grouped OOF model comparison using only V15 DEV outcomes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


SOURCES = {"US": "SEC_V15_DEV", "KR": "KIND_V15_DEV"}
SPECS = {
    "logit_001_bal": {"kind": "logit", "C": 0.01, "class_weight": "balanced"},
    "logit_003_bal": {"kind": "logit", "C": 0.03, "class_weight": "balanced"},
    "logit_010_bal": {"kind": "logit", "C": 0.10, "class_weight": "balanced"},
    "logit_030_bal": {"kind": "logit", "C": 0.30, "class_weight": "balanced"},
    "logit_100_bal": {"kind": "logit", "C": 1.00, "class_weight": "balanced"},
    "logit_003": {"kind": "logit", "C": 0.03, "class_weight": None},
    "lgb_3": {"kind": "lgb", "n_estimators": 80, "num_leaves": 3},
    "lgb_7": {"kind": "lgb", "n_estimators": 120, "num_leaves": 7},
    "cat_3": {"kind": "cat", "iterations": 120, "depth": 3},
    "native_cat_3": {"kind": "native_cat", "iterations": 150, "depth": 3,
                     "cat_cols": ["event_type", "ticker"]},
}


def groups_for(frame: pd.DataFrame, market: str) -> pd.Series:
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str) + "|" + day


def main() -> None:
    raw = app.load_search_frame(app.Config())
    raw = raw[raw.source.isin(set(SOURCES.values())) & raw.y.notna()].copy()
    rows = []
    for market, source in SOURCES.items():
        part = raw[raw.source.eq(source)].sort_values("event_time_utc").reset_index(drop=True)
        features = app.model_frame(part)
        groups = groups_for(part, market)
        folds = list(StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=app.SEED,
        ).split(part, part.y.astype(int), groups))
        for name, spec in SPECS.items():
            prediction = np.full(len(part), np.nan)
            for train_index, valid_index in folds:
                model = app.make_model({**spec, "market": market})
                model.fit(features.iloc[train_index], part.iloc[train_index].y.astype(int))
                prediction[valid_index] = model.predict_proba(features.iloc[valid_index])[:, 1]
            output = part[[
                "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
            ]].copy()
            output["prob"] = prediction
            output["model"] = name
            output["eval_weight"] = 1.0
            metrics = app.evaluate(output, 0.75, 0.002)
            print(json.dumps({
                "market": market, "model": name,
                "n": len(output), "balanced_accuracy": metrics["balanced_accuracy"],
                "auc": metrics["auc"], "accuracy": metrics["accuracy"],
                "edge": metrics["edge_vs_naive"],
                "hc_coverage": metrics["highconf_coverage"],
                "hc_accuracy": metrics["highconf_accuracy"],
                "hc_net": metrics["strategy_mean_signed_net"],
            }, allow_nan=True))
            rows.append(output)
    pd.concat(rows, ignore_index=True).to_parquet(
        app.CACHE / "v15_dev_model_oof.parquet", index=False,
    )


if __name__ == "__main__":
    main()
