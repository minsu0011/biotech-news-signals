#!/usr/bin/env python
"""DEV-only tests of leakage-free enhanced entry-time features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import bio_news_30m_v3 as app


SEED = 260825


def groups_for(frame: pd.DataFrame, market: str) -> pd.Series:
    timestamp = pd.to_datetime(frame.event_time_utc, utc=True)
    if market == "KR":
        return frame.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    return frame.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")


def model_factory(name: str, market: str):
    minimum = 45 if market == "KR" else 16
    if name == "logit_l2":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=SEED)),
        ])
    if name == "logit_sparse":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.02, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000, random_state=SEED)),
        ])
    if name.startswith("lgb"):
        leaves, estimators = {
            "lgb3": (3, 80), "lgb7": (7, 100), "lgb15": (15, 120),
        }[name]
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", LGBMClassifier(
                n_estimators=estimators, num_leaves=leaves, learning_rate=0.025,
                min_child_samples=minimum, subsample=0.8, colsample_bytree=0.75,
                reg_lambda=8.0, class_weight="balanced", verbosity=-1,
                random_state=SEED, n_jobs=-1,
            )),
        ])
    if name == "extra":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesClassifier(
                n_estimators=350, max_depth=5, min_samples_leaf=minimum,
                max_features=0.7, class_weight="balanced", random_state=SEED, n_jobs=-1,
            )),
        ])
    if name == "forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=350, max_depth=5, min_samples_leaf=minimum,
                max_features=0.7, class_weight="balanced", random_state=SEED, n_jobs=-1,
            )),
        ])
    if name == "hist":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.035, max_leaf_nodes=7,
                min_samples_leaf=minimum, l2_regularization=8.0, random_state=SEED,
            )),
        ])
    raise KeyError(name)


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE)
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    raw = raw[raw.source.isin(allowed)].copy()
    features = app.model_frame(raw)
    numeric = features[app.NUMERIC_COLS].replace([np.inf, -np.inf], np.nan)
    print("INDIVIDUAL FEATURES")
    for market in ("KR", "US"):
        mask = raw.market.eq(market).to_numpy()
        rankings = []
        for column in app.NUMERIC_COLS:
            valid = mask & numeric[column].notna().to_numpy()
            if valid.sum() < 40 or raw.loc[valid, "y"].nunique() < 2 or numeric.loc[valid, column].nunique() < 3:
                continue
            auc = roc_auc_score(raw.loc[valid, "y"], numeric.loc[valid, column])
            rankings.append((max(auc, 1.0 - auc), "+" if auc >= 0.5 else "-", column, int(valid.sum())))
        rankings.sort(reverse=True)
        print(market)
        for row in rankings[:25]:
            print(row)

    names = ("logit_l2", "logit_sparse", "lgb3", "lgb7", "lgb15", "extra", "forest", "hist")
    predictions = {}
    for market in ("KR", "US"):
        positions = np.flatnonzero(raw.market.eq(market).to_numpy())
        part = raw.iloc[positions].copy().reset_index(drop=True)
        x = numeric.iloc[positions].reset_index(drop=True)
        groups = groups_for(part, market)
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(part, part.y, groups))
        for name in names:
            prediction = np.full(len(part), np.nan)
            fold_auc = []
            for train_index, valid_index in folds:
                model = model_factory(name, market)
                model.fit(x.iloc[train_index], part.iloc[train_index].y)
                prediction[valid_index] = model.predict_proba(x.iloc[valid_index])[:, 1]
                fold_auc.append(roc_auc_score(part.iloc[valid_index].y, prediction[valid_index]))
            auc = roc_auc_score(part.y, prediction)
            balance = balanced_accuracy_score(part.y, prediction >= 0.5)
            print(json.dumps({
                "market": market, "model": name, "auc": auc,
                "balanced_accuracy": balance, "fold_auc": fold_auc,
            }))
            output = part[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
            output["prob"] = prediction
            predictions[(market, name)] = output

    rows = []
    for key, frame in predictions.items():
        candidate = frame.copy()
        candidate["model"] = key[1]
        rows.append(candidate)
    pd.concat(rows, ignore_index=True).to_parquet(app.CACHE / "v13_enhanced_model_oof.parquet", index=False)


if __name__ == "__main__":
    main()
