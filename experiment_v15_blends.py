"""V15 DEV-only cutoff and ensemble diagnostics."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def metrics(y: np.ndarray, score: np.ndarray, cutoff: float) -> tuple[float, ...]:
    pred = score >= cutoff
    accuracy = accuracy_score(y, pred)
    balance = balanced_accuracy_score(y, pred)
    return accuracy, balance, roc_auc_score(y, score), accuracy - max(y.mean(), 1-y.mean())


def main() -> None:
    raw = pd.read_parquet(app.CACHE / "v15_dev_model_oof.parquet")
    keys = ["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]
    wide = raw.pivot(index=keys, columns="model", values="prob").reset_index()
    selected = {
        "US": ["native_cat_3", "lgb_7", "cat_3", "lgb_3", "logit_010_bal"],
        "KR": ["logit_001_bal", "logit_003_bal", "logit_003", "logit_010_bal", "logit_030_bal"],
    }
    for market, names in selected.items():
        part = wide[wide.market.eq(market)].copy().reset_index(drop=True)
        y = part.y.to_numpy(int)
        candidates: dict[str, np.ndarray] = {name: part[name].to_numpy(float) for name in names}
        candidates["mean4"] = np.mean([candidates[name] for name in names[:4]], axis=0)
        candidates["rank4"] = np.mean([rank(candidates[name]) for name in names[:4]], axis=0)
        for left, right in itertools.combinations(names[:4], 2):
            candidates[f"{left}+{right}"] = 0.5*candidates[left]+0.5*candidates[right]
        rows=[]
        for name, score in candidates.items():
            best=None
            for cutoff in np.arange(.35,.701,.005):
                result=metrics(y,score,float(cutoff))
                objective=(result[3]>=.02,result[1]>=.51,result[1]+result[0])
                if best is None or objective>best[0]:
                    best=(objective,cutoff,result)
            rows.append((market,name,best[1],*best[2]))
        print("\n",market)
        for row in sorted(rows,key=lambda x:x[4]+x[3],reverse=True):
            print(row)


if __name__=="__main__":
    main()
