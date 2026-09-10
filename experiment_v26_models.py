"""V26 grouped OOF models; V26 seal outcomes remain unparsed."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import runtime_limits
import bio_news_30m_v3 as app

ROOT=Path(__file__).resolve().parent
TARGET={"US":"SEC_V26_DEV","KR":"NAVER_NEWS_V26_DEV"}
PRIOR={
 "US":{
   "v25":{"SEC_V25_DEV","SEC_V25_SEAL"},
   "v24v25":{"SEC_V24_DEV","SEC_V24_SEAL","SEC_V25_DEV","SEC_V25_SEAL"},
 },
 "KR":{
   "v25":{"NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"},
   "v24v25":{"NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL","NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"},
 },
}


def specs():
    rows={f"cat{d}":{"kind":"native_cat","iterations":350,"depth":d,
                       "cat_cols":["event_type","ticker"]} for d in (2,4,6)}
    rows.update({f"lgb{l}":{"kind":"lgb","n_estimators":350,"num_leaves":l,
                             "cat_cols":["event_type","ticker"]} for l in (3,5,9,15)})
    rows.update({"logit005":{"kind":"logit","C":.005,"class_weight":"balanced"},
                 "logit02":{"kind":"logit","C":.02,"class_weight":"balanced"}})
    return rows


def groups(frame,market):
    if market=="KR":return frame.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str)+"|"+day


def predict(target,prior,market,spec,repeat):
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED)
    probability=np.full(len(target),np.nan)
    for train_index,valid_index in splitter.split(target,target.y.astype(int),groups(target,market)):
        frames=[];labels=[]
        if len(prior):frames.append(prior);labels.append(prior.y.astype(int).to_numpy())
        copies=max(1,repeat)
        frames += [target.iloc[train_index]]*copies
        labels += [target.iloc[train_index].y.astype(int).to_numpy()]*copies
        train=pd.concat(frames,ignore_index=True,sort=False);y=np.concatenate(labels)
        model=app.make_model(spec);model.fit(app.model_frame(train),y)
        probability[valid_index]=model.predict_proba(app.model_frame(target.iloc[valid_index]))[:,1]
    if not np.isfinite(probability).all():raise RuntimeError("incomplete V26 OOF")
    return probability


def audit(frame,probability):
    y=frame.y.astype(int).to_numpy();up=float(y.mean());auc=float(roc_auc_score(y,probability));rows=[]
    for cutoff in np.arange(.20,.801,.005):
        pred=probability>=cutoff;acc=float((pred==y).mean());bal=float(balanced_accuracy_score(y,pred))
        edge=acc-max(up,1-up)
        rows.append({"cutoff":float(cutoff),"accuracy":acc,"balanced_accuracy":bal,"auc":auc,
                     "edge":edge,"pred_up":float(pred.mean()),
                     "passes":int(bal>=.54)+int(auc>=.56)+int(edge>=.02),
                     "score":float(8*bal+3*auc+5*edge)})
    return sorted(rows,key=lambda r:(r["passes"],r["score"]),reverse=True)


def main():
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V26_SEAL","NAVER_NEWS_V26_SEAL"}),"y"].notna().any():
        raise RuntimeError("V26 seal outcome exposure detected")
    output={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"markets":{}};cache=[]
    for market in ("US","KR"):
        target=data[data.source.eq(TARGET[market])&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
        rows=[]
        for prior_name,sources in PRIOR[market].items():
            prior=data[data.market.eq(market)&data.source.isin(sources)&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
            for model_name,spec in specs().items():
                repeats=(1,3,6) if model_name.startswith(("cat","lgb")) else (1,3)
                for repeat in repeats:
                    probability=predict(target,prior,market,spec,repeat);choices=audit(target,probability)
                    name=f"{prior_name}_{model_name}_r{repeat}";best=choices[0]
                    rows.append({"name":name,"prior_n":len(prior),"spec":spec,"target_repeat":repeat,"choices":choices[:30]})
                    z=target[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
                    z["model_name"]=name;z["probability"]=probability;cache.append(z)
                    print(f"[{market} {name}] pass={best['passes']}/3 bal={best['balanced_accuracy']:.4f} auc={best['auc']:.4f} edge={best['edge']:.4f} cut={best['cutoff']:.3f}",flush=True)
        rows.sort(key=lambda r:(r["choices"][0]["passes"],r["choices"][0]["score"]),reverse=True)
        output["markets"][market]={"n":len(target),"up_rate":float(target.y.mean()),"models":rows}
    pd.concat(cache,ignore_index=True).to_parquet(ROOT/"cache"/"v26_model_oof.parquet",index=False)
    path=ROOT/"cache"/"v26_model_audits.json";path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path))

if __name__=="__main__":main()
