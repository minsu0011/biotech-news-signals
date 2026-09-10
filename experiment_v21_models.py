"""Grouped V21 DEV model comparison with V21 SEAL outcomes unparsed."""
from __future__ import annotations

import runtime_limits

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


def hb(value:str,modulo:int=2)->int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16],16)%modulo


def audit(frame:pd.DataFrame,probability:np.ndarray,cutoff:float)->dict:
    y=frame.y.astype(int).to_numpy();pred=(probability>=cutoff).astype(int)
    event=frame.event_id.astype(str);time=pd.to_datetime(frame.event_time_utc,utc=True)
    ticker_half=frame.ticker.astype(str).map(hb).to_numpy();hash_half=event.map(hb).to_numpy()
    order=time.rank(method="first",pct=True).to_numpy()
    masks={"all":np.ones(len(frame),dtype=bool),"h0":hash_half==0,"h1":hash_half==1,
           "t0":order<=.5,"t1":order>.5,"ticker0":ticker_half==0,"ticker1":ticker_half==1}
    out={}
    for name,mask in masks.items():
        yy=y[mask];pp=probability[mask];pr=pred[mask]
        out[name]={"n":int(mask.sum()),"bal":float(balanced_accuracy_score(yy,pr)),
                   "auc":float(roc_auc_score(yy,pp)),"acc":float((yy==pr).mean())}
    return out


def select_cutoff(frame:pd.DataFrame,probability:np.ndarray)->tuple[float,dict,float]:
    rows=[]
    for cutoff in np.arange(.30,.701,.025):
        audits=audit(frame,probability,float(cutoff));core=list(audits.values())
        score=2*min(x["bal"] for x in core)+min(x["auc"] for x in core)+audits["all"]["bal"]+audits["all"]["auc"]
        rows.append((score,float(cutoff),audits))
    return max(rows,key=lambda value:value[0])


def candidates()->list[dict]:
    out=[]
    for c in (.001,.003,.01,.03,.1,.3):
        out.append({"kind":"logit","C":c,"class_weight":"balanced"})
    for depth in (2,3,4,5):
        out.append({"kind":"native_cat","iterations":220,"depth":depth,
                    "cat_cols":["event_type","ticker"]})
    for leaves in (5,9,15,23):
        out.append({"kind":"lgb","n_estimators":220,"num_leaves":leaves,
                    "cat_cols":["event_type","ticker"]})
    return out


def crossfit(target:pd.DataFrame,spec:dict,market:str)->np.ndarray:
    if market=="KR":groups=target.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    else:
        day=pd.to_datetime(target.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
        groups=target.ticker.astype(str)+"|"+day
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED)
    probability=np.full(len(target),np.nan)
    for train_index,valid_index in splitter.split(target,target.y.astype(int),groups):
        model=app.make_model(spec);model.fit(app.model_frame(target.iloc[train_index]),target.y.iloc[train_index].astype(int))
        probability[valid_index]=model.predict_proba(app.model_frame(target.iloc[valid_index]))[:,1]
    if not np.isfinite(probability).all():raise RuntimeError("OOF missing")
    return probability


def univariate(data:pd.DataFrame,target:pd.DataFrame,market:str,top:int=40)->list[dict]:
    sources=(['SEC_V20_DEV','SEC_V20_SEAL'] if market=='US'
             else ['NAVER_NEWS_V20_DEV','NAVER_NEWS_V20_SEAL'])
    audits_raw={"v21_dev":target}
    for source in sources:
        part=data[data.source.eq(source)&data.y.notna()].copy()
        if len(part)>=30:audits_raw[source]=part
    rows=[]
    for column in app.NUMERIC_COLS:
        parts={}
        for name,raw in audits_raw.items():
            features=app.model_frame(raw);value=pd.to_numeric(features[column],errors="coerce")
            fill=float(value.median()) if value.notna().any() else 0.0
            parts[name]=(raw,value.fillna(fill).rank(method="average",pct=True).to_numpy(float))
        for sign in (-1.,1.):
            best=None
            for cutoff in np.arange(.4,.601,.025):
                result={name:audit(raw,(score if sign>0 else 1-score),float(cutoff))["all"]
                        for name,(raw,score) in parts.items()}
                core=list(result.values());value=2*min(x['bal'] for x in core)+min(x['auc'] for x in core)+result['v21_dev']['bal']+result['v21_dev']['auc']
                if best is None or value>best[0]:best=(value,float(cutoff),result)
            rows.append({"column":column,"sign":sign,"score":best[0],"cutoff":best[1],"audits":best[2]})
    return sorted(rows,key=lambda value:value['score'],reverse=True)[:top]


def main()->None:
    cfg=app.Config();data=app.load_search_frame(cfg)
    assert data.loc[data.source.isin({'SEC_V21_SEAL','NAVER_NEWS_V21_SEAL'}),'y'].isna().all()
    output={"seal_outcomes_loaded":False,"markets":{}}
    for market,source in (("US","SEC_V21_DEV"),("KR","NAVER_NEWS_V21_DEV")):
        target=data[data.source.eq(source)&data.y.notna()].copy().reset_index(drop=True)
        rows=[]
        for spec in candidates():
            probability=crossfit(target,spec,market);score,cutoff,audits=select_cutoff(target,probability)
            rows.append({"spec":spec,"score":score,"cutoff":cutoff,"audits":audits})
            print(f"[{market}] {spec} bal={audits['all']['bal']:.4f} auc={audits['all']['auc']:.4f}")
        output['markets'][market]={"n":len(target),"up_rate":float(target.y.mean()),
                                  "models":sorted(rows,key=lambda value:value['score'],reverse=True),
                                  "univariate":univariate(data,target,market)}
    print("RESULT_JSON="+json.dumps(output,ensure_ascii=False,default=str))


if __name__=="__main__":main()
