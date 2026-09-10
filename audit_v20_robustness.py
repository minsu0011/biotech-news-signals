"""Read-only robustness views of the frozen-candidate V20 DEV predictions."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score


def bucket(value:str,modulo:int=2)->int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16],16)%modulo


def metric(frame:pd.DataFrame)->dict:
    y=frame.y.astype(int).to_numpy();p=frame.prob.to_numpy(float);pred=(p>=0.5).astype(int)
    return {"n":len(frame),"up_rate":float(y.mean()),"accuracy":float((pred==y).mean()),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "auc":float(roc_auc_score(y,p))}


def main()->None:
    frame=pd.read_parquet("cache/v20_production_cv.parquet")
    frame["year"]=pd.to_datetime(frame.event_time_utc,utc=True).dt.year
    frame["ticker_half"]=frame.ticker.astype(str).map(bucket)
    frame["article_key"]=frame.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    frame["article_half"]=frame.article_key.map(bucket)
    output={"overall":metric(frame),"markets":{}}
    for market,part in frame.groupby("market"):
        output["markets"][market]={
            "overall":metric(part),
            "year":{str(key):metric(value) for key,value in part.groupby("year") if value.y.nunique()>1},
            "ticker_half":{str(key):metric(value) for key,value in part.groupby("ticker_half")},
        }
        if market=="KR":
            output["markets"][market]["article_half"]={
                str(key):metric(value) for key,value in part.groupby("article_half")
            }
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
