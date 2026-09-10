"""External prior-to-V20 DEV model comparison; V20 SEAL outcomes stay unparsed."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


def bucket(value: str, modulo: int) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16) % modulo


def audits(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict:
    y=frame.y.astype(int).to_numpy(); pred=(probability>=cutoff).astype(int)
    time=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    h=frame.event_id.astype(str).map(lambda value:bucket(value,2)).to_numpy()
    masks={"all":np.ones(len(frame),dtype=bool),"h0":h==0,"h1":h==1,
           "early":time<=0.5,"late":time>0.5}
    out={}
    for name,mask in masks.items():
        yy=y[mask];pp=probability[mask];pr=pred[mask]
        out[name]={
            "n":int(mask.sum()),"bal":float(balanced_accuracy_score(yy,pr)),
            "auc":float(roc_auc_score(yy,pp)),"acc":float((yy==pr).mean()),
        }
    return out


def best_cutoff(frame:pd.DataFrame,probability:np.ndarray)->tuple[float,dict,float]:
    rows=[]
    for cutoff in np.arange(0.30,0.701,0.025):
        result=audits(frame,probability,float(cutoff))
        core=[result[name] for name in ("all","h0","h1","early","late")]
        score=(2*min(value["bal"] for value in core)+min(value["auc"] for value in core)
               +result["all"]["bal"]+result["all"]["auc"])
        rows.append((score,float(cutoff),result))
    score,cutoff,result=max(rows,key=lambda value:value[0])
    return cutoff,result,float(score)


def model_candidates(market:str)->list[dict]:
    out=[]
    if market=="KR":
        for c in (0.003,0.01,0.03,0.1,0.3):
            out.append({"kind":"logit","C":c,"class_weight":"balanced"})
    else:
        for c in (0.003,0.01,0.03,0.1):
            out.append({"kind":"logit","C":c,"class_weight":"balanced"})
    for depth in (2,3,4):
        out.append({"kind":"native_cat","iterations":180,"depth":depth,
                    "cat_cols":["event_type","ticker"]})
    for leaves in (5,9,15):
        out.append({"kind":"lgb","n_estimators":180,"num_leaves":leaves,
                    "cat_cols":["event_type","ticker"]})
    return out


def prior_pools(data:pd.DataFrame,target:pd.DataFrame,market:str)->dict[str,pd.DataFrame]:
    opened=data[data.market.eq(market)&data.y.notna()&~data.event_id.isin(target.event_id)].copy()
    tickers=set(target.ticker.astype(str))
    same=opened[opened.ticker.astype(str).isin(tickers)].copy()
    if market=="US":
        return {
            "v19_same":opened[opened.source.isin(["SEC_V19_DEV","SEC_V19_SEAL"])&
                              opened.ticker.astype(str).isin(tickers)].copy(),
            "all_sec_same":same[same.source.astype(str).str.startswith("SEC")].copy(),
        }
    naver=opened[opened.source.astype(str).str.startswith("NAVER_NEWS")].copy()
    return {
        "naver_same":naver[naver.ticker.astype(str).isin(tickers)].copy(),
        "naver_all":naver,
        "naver_recent":naver[naver.source.isin([
            "NAVER_NEWS_V12_DEV","NAVER_NEWS_V12_SEAL","NAVER_NEWS_V13_DEV",
            "NAVER_NEWS_V13_SEAL","NAVER_NEWS_V14_DEV","NAVER_NEWS_V14_SEAL",
        ])].copy(),
    }


def univariate(target:pd.DataFrame,top:int=25)->list[dict]:
    features=app.model_frame(target)
    rows=[]
    for column in app.NUMERIC_COLS:
        values=pd.to_numeric(features[column],errors="coerce")
        fill=float(values.median()) if values.notna().any() else 0.0
        rank=values.fillna(fill).rank(method="average",pct=True).to_numpy(float)
        for sign in (-1.0,1.0):
            probability=rank if sign>0 else 1.0-rank
            cutoff,result,score=best_cutoff(target,probability)
            rows.append({"column":column,"sign":sign,"cutoff":cutoff,"score":score,
                         "audits":result})
    return sorted(rows,key=lambda value:value["score"],reverse=True)[:top]


def main()->None:
    cfg=app.Config();data=app.load_search_frame(cfg)
    seals={"SEC_V20_SEAL","NAVER_NEWS_V20_SEAL"}
    assert data.loc[data.source.isin(seals),"y"].isna().all()
    output={"seal_outcomes_loaded":False,"markets":{}}
    for market,source in (("US","SEC_V20_DEV"),("KR","NAVER_NEWS_V20_DEV")):
        target=data[data.source.eq(source)&data.y.notna()].copy().reset_index(drop=True)
        pools=prior_pools(data,target,market)
        rows=[]
        for pool_name,prior in pools.items():
            if len(prior)<100 or prior.y.nunique()<2:continue
            for spec in model_candidates(market):
                model=app.make_model(spec)
                model.fit(app.model_frame(prior),prior.y.astype(int))
                probability=model.predict_proba(app.model_frame(target))[:,1]
                cutoff,result,score=best_cutoff(target,probability)
                rows.append({"pool":pool_name,"prior_n":len(prior),"spec":spec,
                             "cutoff":cutoff,"score":score,"audits":result})
        output["markets"][market]={
            "target_n":len(target),"up_rate":float(target.y.mean()),
            "univariate":univariate(target),
            "models":sorted(rows,key=lambda value:value["score"],reverse=True),
        }
    print(json.dumps(output,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
