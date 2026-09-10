"""Leakage-safe robust direction search for the untouched V35 seal.

Only opened outcomes through V34 are parsed.  V35 rows are asserted to have
masked outcomes and are used only in a separate feature-only coverage audit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app
import runtime_limits


ROOT=Path(__file__).resolve().parent
SEALED={"US_EXACT_V35_SEAL","KR_EXACT_V35_SEAL"}
PRIMARY={"US":{"US_EXACT_V34_SEAL"},"KR":{"KR_EXACT_V34_SEAL"}}
ANALOG={
    "US":{"US_EXACT_V31_DEV","US_EXACT_V31_SEAL"},
    "KR":{"KR_EXACT_V30_DEV","KR_EXACT_V30_SEAL",
          "KR_EXACT_V31_DEV","KR_EXACT_V31_SEAL"},
}
EXCLUDED={
    "year","month","weekday","hour","minute","minutes_from_open",
    "minutes_to_close","event_hour","event_weekday","event_month",
    "event_hour_sin","event_hour_cos","headline_len","body_len",
}
FEATURES=tuple(column for column in app.NUMERIC_COLS if column not in EXCLUDED)


def rank(values:np.ndarray)->np.ndarray:
    values=np.asarray(values,float);finite=np.isfinite(values)
    fill=float(np.median(values[finite])) if finite.any() else 0.0
    values=np.nan_to_num(values,nan=fill,posinf=fill,neginf=fill)
    reference=np.sort(values)
    left=np.searchsorted(reference,values,side="left")
    right=np.searchsorted(reference,values,side="right")
    return (left+right+1.0)/(2.0*len(reference))


def metrics(y:np.ndarray,probability:np.ndarray,cutoff:float,auc:float|None=None)->dict:
    prediction=probability>=cutoff;positive=y==1;negative=~positive
    accuracy=float((prediction==y).mean());naive=max(float(y.mean()),1.0-float(y.mean()))
    return {
        "n":int(len(y)),"balanced_accuracy":float(0.5*(prediction[positive].mean()+
                                                            (~prediction[negative]).mean())),
        "auc":float(roc_auc_score(y,probability) if auc is None else auc),"accuracy":accuracy,
        "edge":accuracy-naive,"pred_up":float(prediction.mean()),
    }


def best_candidate(market:str,components:list,scores:dict[str,dict[str,np.ndarray]],
                   labels:dict[str,np.ndarray],partitions:dict[str,np.ndarray])->dict:
    probability={name:sum(float(weight)*(scores[name][column] if sign>0 else
                                          1.0-scores[name][column])
                          for column,sign,weight in components)/
                      sum(float(item[2]) for item in components)
                 for name in scores}
    current=probability["primary"];y=labels["primary"]
    primary_auc=float(roc_auc_score(y,current))
    analog_auc=float(roc_auc_score(labels["analog"],probability["analog"]))
    fold_aucs={name:float(roc_auc_score(y[mask],current[mask]))
               for name,mask in partitions.items()
               if mask.sum()>=40 and np.unique(y[mask]).size>=2}
    best=None
    for cutoff in np.arange(0.30,0.701,0.01):
        primary=metrics(y,current,float(cutoff),primary_auc)
        analog=metrics(labels["analog"],probability["analog"],float(cutoff),analog_auc)
        folds=[]
        for name,mask in partitions.items():
            if mask.sum()<40 or np.unique(y[mask]).size<2:continue
            folds.append((name,metrics(y[mask],current[mask],float(cutoff),fold_aucs[name])))
        min_bal=min(item[1]["balanced_accuracy"] for item in folds)
        min_auc=min(item[1]["auc"] for item in folds)
        checks=sum((primary["balanced_accuracy"]>=0.54,primary["auc"]>=0.56,
                    primary["edge"]>=0.02,analog["balanced_accuracy"]>=0.50,
                    analog["auc"]>=0.50,min_bal>=0.49,min_auc>=0.49))
        score=(2.0*primary["balanced_accuracy"]+2.0*primary["auc"]+primary["edge"]+
               0.65*analog["balanced_accuracy"]+0.65*analog["auc"]+
               0.65*min_bal+0.65*min_auc)
        row={"market":market,"components":components,"cutoff":float(cutoff),
             "primary":primary,"analog":analog,"min_fold_balanced_accuracy":float(min_bal),
             "min_fold_auc":float(min_auc),"folds":{name:value for name,value in folds},
             "checks":int(checks),"score":float(score)}
        if best is None or (row["checks"],row["score"])>(best["checks"],best["score"]):best=row
    return best


def main()->None:
    runtime_limits.configure();data=app.load_search_frame(app.Config())
    sealed=data.source.isin(SEALED)
    if data.loc[sealed,["y","fwd_ret_30m"]].notna().any().any():
        raise RuntimeError("V35 seal outcomes exposed")
    opened=data[data.y.notna()].copy();output={}
    for market in ("US","KR"):
        frames={
            "primary":opened[opened.market.eq(market)&opened.source.isin(PRIMARY[market])]
                         .sort_values("event_time_utc").reset_index(drop=True),
            "analog":opened[opened.market.eq(market)&opened.source.isin(ANALOG[market])]
                        .sort_values("event_time_utc").reset_index(drop=True),
        }
        if any(frame.empty or frame.y.nunique()<2 for frame in frames.values()):
            raise RuntimeError(f"insufficient opened cohort: {market}")
        features={name:app.model_frame(frame) for name,frame in frames.items()}
        labels={name:frame.y.astype(int).to_numpy() for name,frame in frames.items()}
        scores={name:{column:rank(pd.to_numeric(feature[column],errors="coerce").to_numpy(float))
                      for column in FEATURES} for name,feature in features.items()}
        primary=frames["primary"];n=len(primary);order=np.arange(n)
        hashes=primary.event_id.astype(str).map(
            lambda value:int(hashlib.sha256(value.encode()).hexdigest(),16)%3).to_numpy()
        partitions={"time_first":order<n//2,"time_second":order>=n//2}
        partitions.update({f"hash_{bucket}":hashes==bucket for bucket in range(3)})
        singles=[best_candidate(market,[(column,sign,1.0)],scores,labels,partitions)
                 for column,sign in itertools.product(FEATURES,(-1.0,1.0))]
        singles.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
        selected=[]
        for row in singles:
            column,sign,_=row["components"][0]
            if column not in {item[0] for item in selected}:selected.append((column,sign))
            if len(selected)==24:break
        rows=list(singles)
        for (first,first_sign),(second,second_sign) in itertools.combinations(selected,2):
            for weight in (0.20,0.35,0.50,0.65,0.80):
                rows.append(best_candidate(market,[(first,first_sign,weight),
                    (second,second_sign,1.0-weight)],scores,labels,partitions))
        for combination in itertools.combinations(selected[:14],3):
            rows.append(best_candidate(market,[(column,sign,1.0/3.0)
                                                for column,sign in combination],
                                       scores,labels,partitions))
        rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
        output[market]=rows[:300]
        print(f"DATA {market} primary={len(frames['primary'])} analog={len(frames['analog'])}",flush=True)
        for row in rows[:30]:print("TOP",market,json.dumps(row,ensure_ascii=False),flush=True)
    path=ROOT/"cache"/"v35_robust_direction.json"
    path.write_text(json.dumps({"v35_seal_outcomes_loaded":False,"features":FEATURES,
                                "rows":output},ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path),flush=True)


if __name__=="__main__":main()
