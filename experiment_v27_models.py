"""Rolling V27 direction-model search with the V27 seal outcomes hidden."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent
RECENT={
    "US":{"SEC_V24_DEV","SEC_V24_SEAL","SEC_V25_DEV","SEC_V25_SEAL",
          "SEC_V26_DEV","SEC_V26_SEAL"},
    "KR":{"NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL",
          "NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL",
          "NAVER_NEWS_V26_DEV","NAVER_NEWS_V26_SEAL"},
}
AUDIT_PRIOR={
    "US":{"SEC_V24_DEV","SEC_V24_SEAL","SEC_V25_DEV","SEC_V25_SEAL"},
    "KR":{"NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL",
          "NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"},
}
AUDIT_TARGET={
    "US":{"SEC_V26_DEV","SEC_V26_SEAL"},
    "KR":{"NAVER_NEWS_V26_DEV","NAVER_NEWS_V26_SEAL"},
}
CURRENT_TARGET={"US":{"SEC_V27_DEV"},"KR":{"NAVER_NEWS_V27_DEV"}}


def groups(frame:pd.DataFrame,market:str)->pd.Series:
    if market=="KR":return frame.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str)+"|"+day


def fit_predict(prior:pd.DataFrame,target:pd.DataFrame,market:str,
                spec:dict,target_repeat:int)->np.ndarray:
    px=app.model_frame(prior);py=prior.y.astype(int).to_numpy()
    tx=app.model_frame(target);ty=target.y.astype(int).to_numpy()
    if target_repeat==0:
        model=app.make_model(spec);model.fit(px,py)
        return model.predict_proba(tx)[:,1]
    raw=np.full(len(target),np.nan,float)
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED)
    for train_index,valid_index in splitter.split(tx,ty,groups(target,market)):
        train=pd.concat([px]+[tx.iloc[train_index]]*target_repeat,ignore_index=True,sort=False)
        labels=np.concatenate([py]+[ty[train_index]]*target_repeat)
        model=app.make_model(spec);model.fit(train,labels)
        raw[valid_index]=model.predict_proba(tx.iloc[valid_index])[:,1]
    if not np.isfinite(raw).all():raise RuntimeError("incomplete grouped OOF")
    return raw


def metrics(y:np.ndarray,p:np.ndarray,cutoff:float)->dict:
    pred=p>=cutoff;up=float(y.mean());accuracy=float((pred==y).mean())
    return {
        "n":len(y),"accuracy":accuracy,
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "auc":float(roc_auc_score(y,p)),"edge":accuracy-max(up,1-up),
        "up_rate":up,"predicted_up":float(pred.mean()),
    }


def candidates(market:str):
    cats=["event_type","ticker"]
    for leaves in (3,7,15,31):
        spec={"kind":"lgb","n_estimators":350,"num_leaves":leaves,"cat_cols":cats}
        for repeat in (0,1,3):yield f"lgb{leaves}_r{repeat}",spec,repeat
    for depth in (2,4,6):
        spec={"kind":"cat","iterations":400,"depth":depth,"cat_cols":cats}
        for repeat in (0,1):yield f"cat{depth}_r{repeat}",spec,repeat
    for C in (.1,.7,3.0):
        spec={"kind":"logit","C":C,"class_weight":"balanced"}
        for repeat in (0,1,3):yield f"logit{C}_r{repeat}",spec,repeat


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V27_SEAL","NAVER_NEWS_V27_SEAL"}),"y"].notna().any():
        raise RuntimeError("V27 seal exposed")
    opened=data[data.y.notna()].copy()
    rows=[];predictions=[]
    for market in ("US","KR"):
        current_target=opened[opened.market.eq(market)&opened.source.isin(CURRENT_TARGET[market])].copy()
        current_prior=opened[opened.market.eq(market)&opened.source.isin(RECENT[market])].copy()
        audit_target=opened[opened.market.eq(market)&opened.source.isin(AUDIT_TARGET[market])].copy()
        audit_prior=opened[opened.market.eq(market)&opened.source.isin(AUDIT_PRIOR[market])].copy()
        print("DATA",market,len(audit_prior),len(audit_target),len(current_prior),len(current_target),flush=True)
        for name,spec,repeat in candidates(market):
            sets={}
            for label,prior,target in (("audit",audit_prior,audit_target),
                                       ("current",current_prior,current_target)):
                probability=fit_predict(prior,target,market,spec,repeat)
                sets[label]={"frame":target,"probability":probability}
                predictions.extend({"event_id":event_id,"market":market,"set":label,
                                    "model":name,"probability":float(probability[index])}
                                   for index,event_id in enumerate(target.event_id.astype(str)))
            best=None
            for cutoff in np.arange(.30,.701,.01):
                result={label:metrics(part["frame"].y.astype(int).to_numpy(),
                                      part["probability"],float(cutoff))
                        for label,part in sets.items()}
                checks=sum(value["balanced_accuracy"]>=.54 for value in result.values())+sum(
                    value["edge"]>=.02 for value in result.values())+sum(
                    value["auc"]>=.56 for value in result.values())
                score=min(value["balanced_accuracy"] for value in result.values())+min(
                    value["edge"] for value in result.values())
                row={"market":market,"model":name,"spec":spec,"target_repeat":repeat,
                     "cutoff":float(cutoff),"sets":result,"checks":checks,"score":float(score)}
                if best is None or (row["checks"],row["score"])>(best["checks"],best["score"]):best=row
            rows.append(best);print(json.dumps(best,ensure_ascii=False),flush=True)
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
    path=ROOT/"cache"/"v27_model_search.json"
    path.write_text(json.dumps({"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
                                "rows":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(predictions).to_parquet(ROOT/"cache"/"v27_model_oof.parquet",index=False)
    print("TOP")
    for row in rows[:20]:print(json.dumps(row,ensure_ascii=False))
    print("RESULT="+str(path))


if __name__=="__main__":main()
