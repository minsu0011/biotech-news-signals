"""V23 grouped OOF with opened V22 rows used as prior training data."""
from __future__ import annotations

import runtime_limits

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent


def specs()->dict[str,dict]:
    return {
        "cat2":{"kind":"native_cat","iterations":220,"depth":2,
                "cat_cols":["event_type","ticker"]},
        "cat4":{"kind":"native_cat","iterations":220,"depth":4,
                "cat_cols":["event_type","ticker"]},
        "cat6":{"kind":"native_cat","iterations":220,"depth":6,
                "cat_cols":["event_type","ticker"]},
        "lgb9":{"kind":"lgb","n_estimators":220,"num_leaves":9,
                "cat_cols":["event_type","ticker"]},
        "text":{"kind":"logit","C":.01,"class_weight":"balanced"},
    }


def groups(frame:pd.DataFrame,market:str)->pd.Series:
    if market=="KR":return frame.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str)+"|"+day


def predict(target:pd.DataFrame,prior:pd.DataFrame,market:str,spec:dict,ratio:int)->np.ndarray:
    if ratio==0:
        model=app.make_model(spec)
        model.fit(app.model_frame(prior),prior.y.astype(int))
        return model.predict_proba(app.model_frame(target))[:,1]
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED)
    probability=np.full(len(target),np.nan)
    for train_index,valid_index in splitter.split(target,target.y.astype(int),groups(target,market)):
        target_train=target.iloc[train_index]
        train=pd.concat([prior]+[target_train]*ratio,ignore_index=True,sort=False)
        model=app.make_model(spec)
        model.fit(app.model_frame(train),train.y.astype(int))
        probability[valid_index]=model.predict_proba(app.model_frame(target.iloc[valid_index]))[:,1]
    return probability


def best_cutoffs(frame:pd.DataFrame,probability:np.ndarray)->list[dict]:
    y=frame.y.astype(int).to_numpy();up=float(y.mean());auc=float(roc_auc_score(y,probability));rows=[]
    for cutoff in np.arange(.30,.801,.005):
        pred=probability>=cutoff;acc=float((pred==y).mean());bal=float(balanced_accuracy_score(y,pred))
        edge=acc-max(up,1-up)
        passes=int(bal>=.54)+int(edge>=.02)
        rows.append({"cutoff":float(cutoff),"balanced_accuracy":bal,"auc":auc,
                     "accuracy":acc,"edge":edge,"up_prediction_rate":float(pred.mean()),
                     "pass_count":passes,"score":float(8*bal+5*edge)})
    return sorted(rows,key=lambda row:(row["pass_count"],row["score"]),reverse=True)


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V23_SEAL","NAVER_NEWS_V23_SEAL"}),"y"].notna().any():
        raise RuntimeError("V23 seal outcome exposure detected")
    definitions={
        "US":("SEC_V23_DEV",{"SEC_V22_DEV","SEC_V22_SEAL"}),
        "KR":("NAVER_NEWS_V23_DEV",{"NAVER_NEWS_V22_DEV","NAVER_NEWS_V22_SEAL"}),
    }
    output={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"markets":{}}
    cache=[]
    for market,(target_source,prior_sources) in definitions.items():
        target=(data[data.source.eq(target_source)&data.y.notna()].copy()
                .sort_values("event_time_utc").reset_index(drop=True))
        prior=(data[data.market.eq(market)&data.source.isin(prior_sources)&data.y.notna()].copy()
               .sort_values("event_time_utc").reset_index(drop=True))
        rows=[]
        for name,spec in specs().items():
            for ratio in (0,1,3):
                probability=predict(target,prior,market,spec,ratio)
                choices=best_cutoffs(target,probability)
                model_name=f"{name}_r{ratio}"
                rows.append({"name":model_name,"spec":spec,"target_repeat":ratio,
                             "choices":choices[:20]})
                print(f"[{market} {model_name}] "+json.dumps(choices[0]),flush=True)
                z=target[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
                z["model_name"]=model_name;z["probability"]=probability;cache.append(z)
        rows.sort(key=lambda row:(row["choices"][0]["pass_count"],row["choices"][0]["score"]),reverse=True)
        output["markets"][market]={"target_n":len(target),"prior_n":len(prior),
                                    "target_up_rate":float(target.y.mean()),"models":rows}
    pd.concat(cache,ignore_index=True).to_parquet(ROOT/"cache"/"v23_augmented_oof.parquet",index=False)
    path=ROOT/"cache"/"v23_augmented_model_audits.json"
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path))


if __name__=="__main__":main()
