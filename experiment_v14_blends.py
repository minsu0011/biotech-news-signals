"""DEV-only decision cutoff and simple ensemble diagnostics."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


def rank(values: np.ndarray) -> np.ndarray:
    order = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    return order


def metrics(y: np.ndarray, score: np.ndarray, cutoff: float) -> tuple[float, ...]:
    pred = score >= cutoff
    accuracy = accuracy_score(y, pred)
    balance = balanced_accuracy_score(y, pred)
    return accuracy, balance, roc_auc_score(y, score), accuracy - max(y.mean(), 1-y.mean())


def main() -> None:
    raw = pd.read_parquet(app.CACHE / "v14_dev_model_oof.parquet")
    keys = ["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]
    wide = raw.pivot(index=keys, columns="model", values="prob").reset_index()
    rows = []
    selected = {
        "US": ["native_cat_3", "lgb_7", "lgb_3", "logit_100_bal", "logit_030_bal"],
        "KR": ["cat_3", "native_cat_3", "lgb_7", "lgb_3", "logit_100_bal"],
    }
    for market, names in selected.items():
        part = wide[wide.market.eq(market)].copy().reset_index(drop=True)
        y = part.y.to_numpy(int)
        candidates: dict[str, np.ndarray] = {name: part[name].to_numpy(float) for name in names}
        candidates["tree_mean"] = np.mean([candidates[name] for name in names[:4]], axis=0)
        candidates["tree_rank_mean"] = np.mean([rank(candidates[name]) for name in names[:4]], axis=0)
        for left, right in itertools.combinations(names[:4], 2):
            candidates[f"{left}+{right}"] = 0.5 * candidates[left] + 0.5 * candidates[right]
        for name, score in candidates.items():
            best = None
            for cutoff in np.arange(0.35, 0.651, 0.005):
                result = metrics(y, score, float(cutoff))
                objective = (result[3] >= .02, result[1] >= .51, result[1] + result[0])
                if best is None or objective > best[0]:
                    best = (objective, cutoff, result)
            assert best is not None
            rows.append((market, name, best[1], *best[2]))
        print("\n", market)
        for row in sorted([r for r in rows if r[0] == market], key=lambda x: x[5]+x[4], reverse=True):
            print(row)


if __name__ == "__main__":
    main()
