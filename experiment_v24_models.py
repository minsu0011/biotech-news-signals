"""V24 grouped OOF direction scan; V24 seal outcomes stay unparsed."""
from __future__ import annotations

import runtime_limits

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent
TARGET_SOURCE={"US":"SEC_V24_DEV","KR":"NAVER_NEWS_V24_DEV"}
PRIOR_SOURCES={
    "US":{
        "v23":{"SEC_V23_DEV","SEC_V23_SEAL"},
        "v22v23":{"SEC_V22_DEV","SEC_V22_SEAL","SEC_V23_DEV","SEC_V23_SEAL"},
    },
    "KR":{
        "v23":{"NAVER_NEWS_V23_DEV","NAVER_NEWS_V23_SEAL"},
        "v22v23":{"NAVER_NEWS_V22_DEV","NAVER_NEWS_V22_SEAL",
                  "NAVER_NEWS_V23_DEV","NAVER_NEWS_V23_SEAL"},
    },
}


def specs()->dict[str,dict]:
    rows={
        f"cat{depth}":{"kind":"native_cat","iterations":300,"depth":depth,
                        "cat_cols":["event_type","ticker"]}
        for depth in (2,4,6)
    }
    rows.update({
        f"lgb{leaves}":{"kind":"lgb","n_estimators":300,"num_leaves":leaves,
                         "cat_cols":["event_type","ticker"]}
        for leaves in (5,9,15)
    })
    rows.update({
        "logit005":{"kind":"logit","C":.005,"class_weight":"balanced"},
        "logit02":{"kind":"logit","C":.02,"class_weight":"balanced"},
    })
    return rows


def groups(frame:pd.DataFrame,market:str)->pd.Series:
    if market=="KR":return frame.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str)+"|"+day


def predict(target:pd.DataFrame,prior:pd.DataFrame,market:str,spec:dict,repeat:int)->np.ndarray:
    if repeat==0:
        model=app.make_model(spec);model.fit(app.model_frame(prior),prior.y.astype(int))
        return model.predict_proba(app.model_frame(target))[:,1]
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED)
    probability=np.full(len(target),np.nan)
    for train_index,valid_index in splitter.split(target,target.y.astype(int),groups(target,market)):
        target_train=target.iloc[train_index]
        train=pd.concat([prior]+[target_train]*repeat,ignore_index=True,sort=False)
        model=app.make_model(spec);model.fit(app.model_frame(train),train.y.astype(int))
        probability[valid_index]=model.predict_proba(app.model_frame(target.iloc[valid_index]))[:,1]
    if not np.isfinite(probability).all():raise RuntimeError("incomplete V24 OOF")
    return probability


def choices(frame:pd.DataFrame,probability:np.ndarray)->list[dict]:
    y=frame.y.astype(int).to_numpy();up=float(y.mean());auc=float(roc_auc_score(y,probability));rows=[]
    for cutoff in np.arange(.20,.801,.005):
        pred=probability>=cutoff;acc=float((pred==y).mean());bal=float(balanced_accuracy_score(y,pred))
        edge=acc-max(up,1-up)
        rows.append({"cutoff":float(cutoff),"accuracy":acc,"balanced_accuracy":bal,
                     "auc":auc,"edge":edge,"pred_up":float(pred.mean()),
                     "direction_passes":int(bal>=.54)+int(edge>=.02),
                     "score":float(8*bal+5*edge)})
    return sorted(rows,key=lambda row:(row["direction_passes"],row["score"]),reverse=True)


def main()->None:
    data=app.load_search_frame(app.Config())
    seals={"SEC_V24_SEAL","NAVER_NEWS_V24_SEAL"}
    if data.loc[data.source.isin(seals),"y"].notna().any():
        raise RuntimeError("V24 seal outcome exposure detected")
    output={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"markets":{}}
    cache=[]
    for market in ("US","KR"):
        target=(data[data.source.eq(TARGET_SOURCE[market])&data.y.notna()].copy()
                .sort_values("event_time_utc").reset_index(drop=True))
        audits=[]
        for prior_name,source_names in PRIOR_SOURCES[market].items():
            prior=(data[data.market.eq(market)&data.source.isin(source_names)&data.y.notna()].copy()
                   .sort_values("event_time_utc").reset_index(drop=True))
            for model_name,spec in specs().items():
                repeats=(0,1,3) if model_name.startswith("cat") else (0,3)
                for repeat in repeats:
                    probability=predict(target,prior,market,spec,repeat)
                    ranked=choices(target,probability)
                    name=f"{prior_name}_{model_name}_r{repeat}"
                    audits.append({"name":name,"prior":prior_name,"prior_n":len(prior),
                                   "spec":spec,"target_repeat":repeat,"choices":ranked[:25]})
                    z=target[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
                    z["model_name"]=name;z["probability"]=probability;cache.append(z)
                    best=ranked[0]
                    print(f"[{market} {name}] bal={best['balanced_accuracy']:.4f} "
                          f"auc={best['auc']:.4f} edge={best['edge']:.4f} "
                          f"cut={best['cutoff']:.3f}",flush=True)
        audits.sort(key=lambda row:(row["choices"][0]["direction_passes"],
                                    row["choices"][0]["score"]),reverse=True)
        output["markets"][market]={"target_n":len(target),"up_rate":float(target.y.mean()),
                                    "models":audits}
    pd.concat(cache,ignore_index=True).to_parquet(ROOT/"cache"/"v24_model_oof.parquet",index=False)
    path=ROOT/"cache"/"v24_model_audits.json"
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path))


if __name__=="__main__":main()
