"""Evaluate pre-screened three-way stable rank blends on V17 DEV only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


def rank(series: pd.Series) -> np.ndarray:
    value=pd.to_numeric(series,errors="coerce")
    value=value.fillna(value.median() if value.notna().any() else 0.0)
    return value.rank(method="average",pct=True).to_numpy(float)


def main() -> None:
    data=app.load_search_frame(app.Config())
    for market,source in (("US","SEC_V17_DEV"),("KR","KIND_V17_DEV")):
        part=data[data.source.eq(source)&data.y.notna()].copy().reset_index(drop=True)
        features=app.model_frame(part)
        candidates=pd.read_csv(app.CACHE/f"v17_threeway_blend_{market}.csv")
        # Limit the fourth-way check to models that were already stable before
        # V17 labels were available.
        candidates=candidates[candidates.min_auc.ge(.54)].copy()
        auc4=[]
        for row in candidates.itertuples(index=False):
            columns=str(row.features).split("|")
            signs=[float(x) for x in str(row.signs).split("|")]
            score=np.mean([sign*rank(features[column]) for column,sign in zip(columns,signs)],axis=0)
            auc4.append(float(roc_auc_score(part.y.astype(int),score)))
        candidates["auc_v17_dev"]=auc4
        candidates["min_auc_4way"]=candidates[["auc_1","auc_2","auc_3","auc_v17_dev"]].min(axis=1)
        candidates["mean_auc_4way"]=candidates[["auc_1","auc_2","auc_3","auc_v17_dev"]].mean(axis=1)
        candidates=candidates.sort_values(["min_auc_4way","mean_auc_4way"],ascending=False)
        candidates.to_csv(app.CACHE/f"v17_dev_rank_{market}.csv",index=False)
        print(f"\n{market} FOUR-WAY")
        print(candidates.head(40).to_string(index=False))


if __name__=="__main__":
    main()
