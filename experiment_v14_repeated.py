"""Repeated grouped V14 DEV OOF with optional already-opened V13 augmentation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


SEEDS = (260825, 101, 202, 303, 404)
V14 = {"US": "SEC_V14_DEV", "KR": "NAVER_NEWS_V14_DEV"}
V13 = {
    "US": {"SEC_V13_DEV", "SEC_V13_SEAL"},
    "KR": {"NAVER_NEWS_V13_DEV", "NAVER_NEWS_V13_SEAL"},
}
SPECS = {
    "US": {
        "native": {"kind": "native_cat", "iterations": 150, "depth": 3,
                   "cat_cols": ["event_type", "ticker"]},
        "lgb7": {"kind": "lgb", "n_estimators": 120, "num_leaves": 7},
        "lgb3": {"kind": "lgb", "n_estimators": 80, "num_leaves": 3},
        "logit": {"kind": "logit", "C": 1.0, "class_weight": "balanced"},
    },
    "KR": {
        "cat3": {"kind": "cat", "iterations": 120, "depth": 3},
        "lgb3": {"kind": "lgb", "n_estimators": 80, "num_leaves": 3},
    },
}


def groups_for(frame: pd.DataFrame, market: str) -> pd.Series:
    if market == "KR":
        return frame.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str) + "|" + day


def best_cutoff(frame: pd.DataFrame) -> tuple[float, dict]:
    best = None
    for cutoff in np.arange(.35, .651, .005):
        shifted = frame.copy()
        shifted["prob"] = app.shift_probability(frame.prob.to_numpy(float), float(cutoff))
        metrics = app.evaluate(shifted, .75, .002)
        objective = (
            metrics["edge_vs_naive"] >= .02,
            metrics["balanced_accuracy"] >= .51,
            metrics["accuracy"] + metrics["balanced_accuracy"],
        )
        if best is None or objective > best[0]:
            best = (objective, float(cutoff), metrics)
    assert best is not None
    return best[1], best[2]


def main() -> None:
    frame = app.load_search_frame(app.Config())
    frame = frame[frame.y.notna()].copy()
    output_rows = []
    for market in ("US", "KR"):
        validation = frame[frame.source.eq(V14[market])].sort_values("event_time_utc").reset_index(drop=True)
        prior = frame[frame.source.isin(V13[market])].copy().reset_index(drop=True)
        valid_features = app.model_frame(validation)
        prior_features = app.model_frame(prior)
        groups = groups_for(validation, market)
        for augment in (False, True):
            by_model = {name: [] for name in SPECS[market]}
            by_seed_blend = []
            for seed in SEEDS:
                folds = list(StratifiedGroupKFold(
                    n_splits=4, shuffle=True, random_state=seed,
                ).split(validation, validation.y.astype(int), groups))
                seed_predictions = {}
                for name, spec in SPECS[market].items():
                    prediction = np.full(len(validation), np.nan)
                    for train_index, valid_index in folds:
                        train_features = valid_features.iloc[train_index]
                        train_y = validation.iloc[train_index].y.astype(int)
                        if augment:
                            train_features = pd.concat([prior_features, train_features], ignore_index=True)
                            train_y = pd.concat([prior.y.astype(int), train_y], ignore_index=True)
                        model = app.make_model({**spec, "market": market})
                        model.fit(train_features, train_y)
                        prediction[valid_index] = model.predict_proba(valid_features.iloc[valid_index])[:, 1]
                    by_model[name].append(prediction)
                    seed_predictions[name] = prediction
                blend = np.mean(list(seed_predictions.values()), axis=0) if market == "US" else seed_predictions["cat3"]
                by_seed_blend.append(blend)
            candidates = {name: np.mean(values, axis=0) for name, values in by_model.items()}
            candidates["blend"] = np.mean(by_seed_blend, axis=0)
            for name, probability in candidates.items():
                result = validation[[
                    "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
                ]].copy()
                result["prob"] = probability
                result["eval_weight"] = 1.0
                cutoff, metrics = best_cutoff(result)
                print(json.dumps({
                    "market": market, "augment_v13": augment, "model": name,
                    "cutoff": cutoff, "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "auc": metrics["auc"],
                    "edge": metrics["edge_vs_naive"],
                }))
                result["augment_v13"] = augment
                result["model"] = name
                result["cutoff"] = cutoff
                output_rows.append(result)
    pd.concat(output_rows, ignore_index=True).to_parquet(
        app.CACHE / "v14_repeated_oof.parquet", index=False,
    )


if __name__ == "__main__":
    main()
