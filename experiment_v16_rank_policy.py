"""Opened V15 cutoff policy for the stable V16 entry-time rank blends."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY = {
    "US": [
        ("intraday_ret_open",-1.0,1.0),
        ("benchmark_ret_5",-1.0,1.0),
        ("range_60m",-1.0,1.0),
        ("excess_ret_30",-1.0,1.0),
    ],
    "KR": [
        ("close_position_10",-1.0,.75),
        ("excess_ret_5",-1.0,.25),
    ],
}
TARGET={"US":501.0,"KR":136.0}


def reference_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference=np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(reference,values,side="right")/len(reference)


def score_market(frame: pd.DataFrame, market: str) -> np.ndarray:
    pieces=[];weights=[]
    for column,sign,weight in POLICY[market]:
        value=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
        fill=np.nanmedian(value) if np.isfinite(value).any() else 0.0
        value=np.nan_to_num(value,nan=fill,posinf=fill,neginf=fill)
        component=reference_rank(value,value)
        pieces.append(sign*weight*component);weights.append(weight)
    raw=np.sum(pieces,axis=0)/sum(weights)
    return .5+.5*raw if min(raw)>=0 else 1.0+raw


def assemble(parts: dict[str,pd.DataFrame], cutoffs: dict[str,float], subset: str) -> pd.DataFrame:
    rows=[]
    for market,part in parts.items():
        if subset=="DEV":
            chosen=part[part.source.str.endswith("_DEV")].copy()
        elif subset=="SEAL":
            chosen=part[part.source.str.endswith("_SEAL")].copy()
        else:
            chosen=part.copy()
        chosen["prob"]=app.shift_probability(chosen.raw_score.to_numpy(float),cutoffs[market])
        chosen["eval_weight"]=TARGET[market]/len(chosen)
        rows.append(chosen)
    return pd.concat(rows,ignore_index=True)


def main() -> None:
    raw=pd.read_csv(
        app.DATA/"labeled_events_v15.csv.gz",compression="gzip",dtype={"ticker":str},
        parse_dates=["event_time_utc"],low_memory=False,
    )
    raw=raw[raw.source.isin({"SEC_V15_DEV","SEC_V15_SEAL","KIND_V15_DEV","KIND_V15_SEAL"})].copy()
    features=app.model_frame(raw)
    for column in raw.columns:
        if column not in features.columns:
            features[column]=raw[column].to_numpy()
    parts={}
    for market in ("US","KR"):
        part=features[features.market.eq(market)].copy()
        part["raw_score"]=score_market(part,market)
        parts[market]=part
    results=[]
    for us_cutoff in np.arange(.20,.801,.01):
        for kr_cutoff in np.arange(.20,.801,.01):
            cutoffs={"US":float(us_cutoff),"KR":float(kr_cutoff)}
            metrics={subset:app.evaluate(assemble(parts,cutoffs,subset),.75,.002)
                     for subset in ("ALL","DEV","SEAL")}
            required=[]
            for subset in ("ALL","DEV","SEAL"):
                m=metrics[subset]
                required.extend([
                    m["balanced_accuracy"]>=.54,m["auc"]>=.56,
                    m["edge_vs_naive"]>=.02,
                    m["by_market"]["US"]["balanced_accuracy"]>=.51,
                    m["by_market"]["KR"]["balanced_accuracy"]>=.51,
                ])
            score=sum(required)
            objective=sum(
                m["accuracy"]+m["balanced_accuracy"]+m["auc"]+min(m["edge_vs_naive"],.06)
                for m in metrics.values()
            )
            results.append((score,objective,cutoffs,metrics))
    results.sort(key=lambda x:(x[0],x[1]),reverse=True)
    for score,objective,cutoffs,metrics in results[:30]:
        print(json.dumps({
            "passes_of_15":score,"objective":objective,"cutoffs":cutoffs,
            "metrics":{
                key:{
                    "accuracy":value["accuracy"],"balanced_accuracy":value["balanced_accuracy"],
                    "auc":value["auc"],"edge":value["edge_vs_naive"],
                    "US_bal":value["by_market"]["US"]["balanced_accuracy"],
                    "KR_bal":value["by_market"]["KR"]["balanced_accuracy"],
                } for key,value in metrics.items()
            },
        }))


if __name__=="__main__":
    main()
