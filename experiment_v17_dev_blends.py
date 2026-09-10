"""V17 DEV-only hash-half stable entry-time rank blends."""
from __future__ import annotations

import hashlib
import itertools

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
        part=data[data.source.eq(source)&data.y.notna()].reset_index(drop=True)
        frame=app.model_frame(part);y=part.y.astype(int).to_numpy()
        half=part.event_id.astype(str).map(
            lambda value:int(hashlib.sha256(f"audit|{value}".encode()).hexdigest(),16)%2
        ).to_numpy()
        scores={};singles=[]
        for column in app.NUMERIC_COLS:
            value=rank(frame[column]);raw_auc=roc_auc_score(y,value)
            sign=1.0 if raw_auc>=.5 else -1.0;value=sign*value
            auc_all=float(roc_auc_score(y,value));a0=float(roc_auc_score(y[half==0],value[half==0]));a1=float(roc_auc_score(y[half==1],value[half==1]))
            scores[column]=(sign,value)
            singles.append((min(a0,a1),auc_all,a0,a1,column,sign))
        singles.sort(reverse=True)
        top=[row[4] for row in singles[:14]]
        rows=[]
        for size in (1,2,3,4):
            for combo in itertools.combinations(top,size):
                value=np.mean([scores[column][1] for column in combo],axis=0)
                a=float(roc_auc_score(y,value));a0=float(roc_auc_score(y[half==0],value[half==0]));a1=float(roc_auc_score(y[half==1],value[half==1]))
                rows.append({
                    "features":"|".join(combo),"signs":"|".join(str(scores[c][0]) for c in combo),
                    "auc":a,"auc_half0":a0,"auc_half1":a1,"min_half":min(a0,a1),
                })
        result=pd.DataFrame(rows).sort_values(["min_half","auc"],ascending=False)
        result.to_csv(app.CACHE/f"v17_dev_blends_{market}.csv",index=False)
        print(f"\n{market} TOP")
        print(result.head(40).to_string(index=False))


if __name__=="__main__":
    main()
