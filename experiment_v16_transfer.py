"""Opened V15 DEV/SEAL transfer audit for V16 model selection."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


SPECS = [
    {"name":"native2","kind":"native_cat","iterations":150,"depth":2,"cat_cols":["event_type","ticker"]},
    {"name":"native3","kind":"native_cat","iterations":150,"depth":3,"cat_cols":["event_type","ticker"]},
    {"name":"native4","kind":"native_cat","iterations":150,"depth":4,"cat_cols":["event_type","ticker"]},
    {"name":"native3_noticker","kind":"native_cat","iterations":150,"depth":3,"cat_cols":["event_type"]},
    {"name":"lgb3","kind":"lgb","n_estimators":120,"num_leaves":3},
    {"name":"lgb7","kind":"lgb","n_estimators":120,"num_leaves":7},
    {"name":"lgb15","kind":"lgb","n_estimators":120,"num_leaves":15},
    {"name":"cat2","kind":"cat","iterations":150,"depth":2},
    {"name":"cat3","kind":"cat","iterations":150,"depth":3},
    {"name":"cat4","kind":"cat","iterations":150,"depth":4},
    {"name":"logit001","kind":"logit","C":.001,"class_weight":"balanced"},
    {"name":"logit003","kind":"logit","C":.003,"class_weight":"balanced"},
    {"name":"logit01","kind":"logit","C":.01,"class_weight":"balanced"},
    {"name":"logit03","kind":"logit","C":.03,"class_weight":"balanced"},
    {"name":"logit10","kind":"logit","C":.10,"class_weight":"balanced"},
    {"name":"logit03_unbal","kind":"logit","C":.03,"class_weight":None},
]


def evaluate(y: np.ndarray, prob: np.ndarray) -> dict:
    best = None
    for cutoff in np.arange(.30, .701, .005):
        pred = prob >= cutoff
        bal = balanced_accuracy_score(y, pred)
        acc = accuracy_score(y, pred)
        objective = bal + .25 * acc
        if best is None or objective > best[0]:
            best = (objective, float(cutoff), float(acc), float(bal))
    return {
        "auc": float(roc_auc_score(y, prob)),
        "cutoff": best[1], "accuracy": best[2], "balanced_accuracy": best[3],
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, raw_spec: dict) -> np.ndarray:
    spec = {key:value for key,value in raw_spec.items() if key != "name"}
    model = app.make_model(spec)
    model.fit(app.model_frame(train), train.y.astype(int))
    return model.predict_proba(app.model_frame(test))[:, 1]


def main() -> None:
    data = pd.read_csv(
        app.DATA / "labeled_events_v15.csv.gz", compression="gzip",
        dtype={"ticker":str}, parse_dates=["event_time_utc"], low_memory=False,
    )
    rows = []
    for market, dev_source, seal_source in (
        ("US", "SEC_V15_DEV", "SEC_V15_SEAL"),
        ("KR", "KIND_V15_DEV", "KIND_V15_SEAL"),
    ):
        dev = data[data.source.eq(dev_source)].sort_values("event_time_utc").reset_index(drop=True)
        seal = data[data.source.eq(seal_source)].sort_values("event_time_utc").reset_index(drop=True)
        for spec in SPECS:
            dev_to_seal = fit_predict(dev, seal, spec)
            seal_to_dev = fit_predict(seal, dev, spec)
            left = evaluate(seal.y.astype(int).to_numpy(), dev_to_seal)
            right = evaluate(dev.y.astype(int).to_numpy(), seal_to_dev)
            row = {
                "market":market, "name":spec["name"],
                "dev_to_seal_auc":left["auc"], "dev_to_seal_bal_opt":left["balanced_accuracy"],
                "dev_to_seal_acc_opt":left["accuracy"], "dev_to_seal_cutoff_opt":left["cutoff"],
                "seal_to_dev_auc":right["auc"], "seal_to_dev_bal_opt":right["balanced_accuracy"],
                "seal_to_dev_acc_opt":right["accuracy"], "seal_to_dev_cutoff_opt":right["cutoff"],
                "min_auc":min(left["auc"], right["auc"]),
                "mean_auc":.5*(left["auc"]+right["auc"]),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(app.CACHE / "v16_transfer.csv", index=False)
    for market in ("US", "KR"):
        print(f"\n{market} RANK")
        print(result[result.market.eq(market)].sort_values(
            ["min_auc","mean_auc"],ascending=False,
        ).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
