"""Repeated V15 DEV OOF with optional opened same-format augmentation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


SEEDS=(260825,101,202,303,404)
V15={"US":"SEC_V15_DEV","KR":"KIND_V15_DEV"}
SPECS={
    "US":{
        "native":{"kind":"native_cat","iterations":150,"depth":3,
                  "cat_cols":["event_type","ticker"]},
        "lgb7":{"kind":"lgb","n_estimators":120,"num_leaves":7},
        "cat3":{"kind":"cat","iterations":120,"depth":3},
    },
    "KR":{
        "logit001":{"kind":"logit","C":.01,"class_weight":"balanced"},
        "logit003":{"kind":"logit","C":.03,"class_weight":"balanced"},
        "logit_unbal":{"kind":"logit","C":.03,"class_weight":None},
    },
}


def groups_for(frame:pd.DataFrame)->pd.Series:
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str)+"|"+day


def best_cutoff(frame:pd.DataFrame)->tuple[float,dict]:
    best=None
    for cutoff in np.arange(.35,.751,.005):
        candidate=frame.copy()
        candidate["prob"]=app.shift_probability(frame.prob.to_numpy(float),float(cutoff))
        metrics=app.evaluate(candidate,.75,.002)
        objective=(
            metrics["edge_vs_naive"]>=.02,
            metrics["balanced_accuracy"]>=.51,
            metrics["accuracy"]+metrics["balanced_accuracy"],
        )
        if best is None or objective>best[0]:
            best=(objective,float(cutoff),metrics)
    return best[1],best[2]


def main()->None:
    data=app.load_search_frame(app.Config())
    data=data[data.y.notna()].copy()
    rows=[]
    for market in ("US","KR"):
        validation=data[data.source.eq(V15[market])].sort_values("event_time_utc").reset_index(drop=True)
        if market=="KR":
            prior=data[data.source.eq("KIND")].copy().reset_index(drop=True)
        else:
            prior=data[data.market.eq("US")&data.source.astype(str).str.startswith("SEC")&
                       ~data.source.eq("SEC_V15_DEV")].copy().reset_index(drop=True)
        valid_features=app.model_frame(validation);prior_features=app.model_frame(prior)
        groups=groups_for(validation)
        for augment in (False,True):
            predictions={name:[] for name in SPECS[market]}
            seed_blends=[]
            for seed in SEEDS:
                folds=list(StratifiedGroupKFold(
                    n_splits=4,shuffle=True,random_state=seed,
                ).split(validation,validation.y.astype(int),groups))
                seed_output={}
                for name,spec in SPECS[market].items():
                    probability=np.full(len(validation),np.nan)
                    for train_index,valid_index in folds:
                        train_x=valid_features.iloc[train_index]
                        train_y=validation.iloc[train_index].y.astype(int)
                        if augment:
                            train_x=pd.concat([prior_features,train_x],ignore_index=True)
                            train_y=pd.concat([prior.y.astype(int),train_y],ignore_index=True)
                        model=app.make_model(spec)
                        model.fit(train_x,train_y)
                        probability[valid_index]=model.predict_proba(valid_features.iloc[valid_index])[:,1]
                    predictions[name].append(probability);seed_output[name]=probability
                seed_blends.append(np.mean(list(seed_output.values()),axis=0))
            candidates={name:np.mean(values,axis=0) for name,values in predictions.items()}
            candidates["blend"]=np.mean(seed_blends,axis=0)
            for name,probability in candidates.items():
                result=validation[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
                result["prob"]=probability;result["eval_weight"]=1.0
                cutoff,metrics=best_cutoff(result)
                print(json.dumps({
                    "market":market,"augment":augment,"model":name,"cutoff":cutoff,
                    "accuracy":metrics["accuracy"],"balanced_accuracy":metrics["balanced_accuracy"],
                    "auc":metrics["auc"],"edge":metrics["edge_vs_naive"],
                }))
                result["augment"]=augment;result["model"]=name;result["cutoff"]=cutoff
                rows.append(result)
    pd.concat(rows,ignore_index=True).to_parquet(app.CACHE/"v15_repeated_oof.parquet",index=False)


if __name__=="__main__":
    main()
