"""Joint V17 DEV direction-cutoff search on hash-half stable rank blends."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":[("benchmark_ret_30",-1.0,1.0),("up_fraction_10",-1.0,1.0),("pre_ret_45",1.0,1.0)],
    "KR":[("benchmark_ret_5",1.0,1.0),("pre_ret_90",-1.0,1.0),
          ("entry_bar_ret",-1.0,1.0),("up_fraction_10",1.0,1.0)],
}
TARGET={"US":354.0,"KR":419.0}


def rank(series: pd.Series) -> np.ndarray:
    value=pd.to_numeric(series,errors="coerce")
    value=value.fillna(value.median() if value.notna().any() else 0.0)
    return value.rank(method="average",pct=True).to_numpy(float)


def score(frame: pd.DataFrame, market: str) -> np.ndarray:
    pieces=[];weights=[]
    for column,sign,weight in POLICY[market]:
        component=rank(frame[column])
        pieces.append(weight*(component if sign>0 else 1.0-component));weights.append(weight)
    return np.sum(pieces,axis=0)/sum(weights)


def main() -> None:
    data=app.load_search_frame(app.Config())
    parts={}
    for market,source in (("US","SEC_V17_DEV"),("KR","KIND_V17_DEV")):
        raw=data[data.source.eq(source)&data.y.notna()].copy().reset_index(drop=True)
        frame=app.model_frame(raw)
        for column in raw.columns:
            if column not in frame.columns:frame[column]=raw[column].to_numpy()
        frame["raw_score"]=score(frame,market)
        frame["audit_half"]=raw.event_id.astype(str).map(
            lambda value:int(hashlib.sha256(f"audit|{value}".encode()).hexdigest(),16)%2
        ).to_numpy()
        parts[market]=frame
    rows=[]
    for us_cutoff in np.arange(.30,.701,.01):
        for kr_cutoff in np.arange(.30,.701,.01):
            cutoffs={"US":float(us_cutoff),"KR":float(kr_cutoff)};metrics={}
            for subset in ("ALL",0,1):
                pieces=[]
                for market,part in parts.items():
                    chosen=part if subset=="ALL" else part[part.audit_half.eq(subset)].copy()
                    chosen=chosen.copy()
                    chosen["prob"]=app.shift_probability(chosen.raw_score.to_numpy(float),cutoffs[market])
                    chosen["eval_weight"]=TARGET[market]/len(chosen);pieces.append(chosen)
                metrics[str(subset)]=app.evaluate(pd.concat(pieces,ignore_index=True),.75,.002)
            required=[]
            for value in metrics.values():
                required.extend([
                    value["balanced_accuracy"]>=.54,value["auc"]>=.56,
                    value["edge_vs_naive"]>=.02,
                    value["by_market"]["US"]["balanced_accuracy"]>=.51,
                    value["by_market"]["KR"]["balanced_accuracy"]>=.51,
                ])
            objective=sum(v["accuracy"]+v["balanced_accuracy"]+v["auc"]+min(v["edge_vs_naive"],.08) for v in metrics.values())
            rows.append((sum(required),objective,cutoffs,metrics))
    rows.sort(key=lambda x:(x[0],x[1]),reverse=True)
    for passes,objective,cutoffs,metrics in rows[:30]:
        print(json.dumps({
            "passes_of_15":passes,"objective":objective,"cutoffs":cutoffs,
            "metrics":{key:{
                "accuracy":v["accuracy"],"balanced_accuracy":v["balanced_accuracy"],
                "auc":v["auc"],"edge":v["edge_vs_naive"],
                "US_bal":v["by_market"]["US"]["balanced_accuracy"],
                "KR_bal":v["by_market"]["KR"]["balanced_accuracy"],
            } for key,v in metrics.items()},
        }))


if __name__=="__main__":
    main()
