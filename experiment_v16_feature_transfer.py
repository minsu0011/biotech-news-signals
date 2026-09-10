"""Opened V15 univariate and categorical transfer diagnostics for V16."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


def auc(y: pd.Series, value: pd.Series) -> float:
    x = pd.to_numeric(value, errors="coerce")
    x = x.fillna(x.median() if x.notna().any() else 0.0)
    return float(roc_auc_score(y.astype(int), x))


def target_encode(train: pd.DataFrame, test: pd.DataFrame, column: str, smoothing: float) -> np.ndarray:
    base = float(train.y.mean())
    stats = train.groupby(column, dropna=False).y.agg(["mean", "count"])
    encoded = (stats["mean"] * stats["count"] + base * smoothing) / (stats["count"] + smoothing)
    return test[column].map(encoded).fillna(base).to_numpy(float)


def main() -> None:
    raw = pd.read_csv(
        app.DATA / "labeled_events_v15.csv.gz", compression="gzip", dtype={"ticker":str},
        parse_dates=["event_time_utc"], low_memory=False,
    )
    data = app.model_frame(raw)
    data["y"] = raw.y.to_numpy()
    data["source"] = raw.source.to_numpy()
    for market, dev_source, seal_source in (
        ("US", "SEC_V15_DEV", "SEC_V15_SEAL"),
        ("KR", "KIND_V15_DEV", "KIND_V15_SEAL"),
    ):
        dev = data[data.source.eq(dev_source)].copy()
        seal = data[data.source.eq(seal_source)].copy()
        rows = []
        for column in app.NUMERIC_COLS:
            left = auc(dev.y, dev[column]); right = auc(seal.y, seal[column])
            orientation = 1.0 if left >= .5 else -1.0
            rows.append({
                "kind":"numeric", "column":column, "parameter":"",
                "dev_auc":left if orientation > 0 else 1-left,
                "seal_auc":right if orientation > 0 else 1-right,
            })
        for column in ("ticker","event_type","event_hour","event_weekday","event_month"):
            for smoothing in (2.0, 5.0, 10.0, 25.0, 50.0):
                p_seal = target_encode(dev, seal, column, smoothing)
                p_dev = target_encode(seal, dev, column, smoothing)
                rows.append({
                    "kind":"target_encode", "column":column, "parameter":smoothing,
                    "dev_auc":float(roc_auc_score(dev.y.astype(int),p_dev)),
                    "seal_auc":float(roc_auc_score(seal.y.astype(int),p_seal)),
                })
        result = pd.DataFrame(rows)
        result["min_auc"] = result[["dev_auc","seal_auc"]].min(axis=1)
        result["mean_auc"] = result[["dev_auc","seal_auc"]].mean(axis=1)
        result.to_csv(app.CACHE / f"v16_feature_transfer_{market}.csv", index=False)
        print(f"\n{market} STABLE FEATURES")
        print(result.sort_values(["min_auc","mean_auc"],ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
