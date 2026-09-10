"""V44 material hypothesis: exact completed-bar microstructure signatures."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V44")))
CACHE=ROOT/"cache"
COST=.002


def feature_frame(frame:pd.DataFrame,micro_columns:list[str])->pd.DataFrame:
    output=v36.feature_frame(frame)
    micro=frame[micro_columns].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
    return pd.concat([output,micro],axis=1)


def make_model(family:str,micro_columns:list[str])->Pipeline:
    numeric=list(app.NUMERIC_COLS)+micro_columns
    numeric_pipe=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True))])
    categories=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=3,sparse_output=False))])
    processor=ColumnTransformer([("numeric",numeric_pipe,numeric),
        ("category",categories,list(v36.CAT_COLUMNS))],remainder="drop",sparse_threshold=0.)
    if family=="lightgbm_micro":
        model=LGBMClassifier(n_estimators=400,num_leaves=15,learning_rate=.025,min_child_samples=50,
            objective="binary",random_state=v36.SEED,n_jobs=runtime_limits.THREAD_COUNT,subsample=.85,
            colsample_bytree=.80,reg_lambda=4.,verbosity=-1,deterministic=True,force_col_wise=True)
    elif family=="catboost_gpu_micro":
        model=CatBoostClassifier(iterations=350,depth=5,learning_rate=.025,l2_leaf_reg=12.,
            loss_function="Logloss",eval_metric="AUC",random_seed=v36.SEED,task_type="GPU",devices="0",
            verbose=False,allow_writing_files=False,random_strength=.5)
    else:raise ValueError(family)
    return Pipeline([("features",processor),("model",model)])


def target_fit(train,target,label,family,micro_columns):
    model=make_model(family,micro_columns);model.fit(feature_frame(train,micro_columns),np.asarray(label,int),
        model__sample_weight=train.cluster_weight)
    return model.predict_proba(feature_frame(target,micro_columns))[:,1]


def fit_by_threshold(train,target,thresholds,family,micro_columns):
    direction=target_fit(train,target,train.y,family,micro_columns)
    up=target_fit(train,target,train.fwd_ret_30m>COST,family,micro_columns)
    down=target_fit(train,target,train.fwd_ret_30m<-COST,family,micro_columns);output={}
    for threshold in thresholds:
        opportunity=target_fit(train,target,train.fwd_ret_30m.abs()>=threshold,family,micro_columns)
        output[threshold]={"direction":direction,"up_payoff":up,"down_payoff":down,"opportunity":opportunity}
    return output


def market_oof(frame:pd.DataFrame,market:str,micro_columns:list[str]):
    selected=[];direction_frames=[];audits=[];model_errors=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);thresholds=(.004,.006,.010);policies=[];inner_by_family={}
        for family in ("lightgbm_micro","catboost_gpu_micro"):
            try:
                parts=fit_by_threshold(inner_train,inner_valid,thresholds,family,micro_columns)
                policy=v38.choose_policy(inner_valid.reset_index(drop=True),parts);policy["model_family"]=family
                policies.append(policy);inner_by_family[family]=parts
            except Exception as error:
                model_errors.append({"market":market,"fold":fold,"stage":"inner","family":family,
                                     "error":repr(error)})
                print(f"[V44 MICRO] {market} fold={fold} {family} failed: {error}",flush=True)
        if not policies:raise RuntimeError(f"V44 all models failed {market} fold={fold}")
        policies.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);policy=policies[0]
        family=policy["model_family"];threshold=float(policy["magnitude_threshold"])
        outer=fit_by_threshold(train,valid,(threshold,),family,micro_columns)[threshold]
        probability,high,confidence=v38.apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(v38.prediction_frame(valid,probability,high,confidence,fold,
                                              "V44_INNER_SELECTED",seen))
        raw=outer["direction"];direction_high=v38.rank_against(
            np.abs(inner_by_family[family][threshold]["direction"]-.5),np.abs(raw-.5))>=.80
        direction_frames.append(v38.prediction_frame(valid,raw,direction_high,np.abs(raw-.5),fold,
                                                       f"{family}_direction",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V44 MICRO] {market} fold={fold} family={family} mode={policy['mode']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(direction_frames,ignore_index=True),audits,model_errors


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz";feature_path=CACHE/"v44_microstructure_features.csv.gz"
    if not feature_path.exists():raise RuntimeError("run prepare_v44_microstructure.py first")
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    micro=pd.read_csv(feature_path,compression="gzip",low_memory=False);micro_columns=[c for c in micro if c!="event_id"]
    data=data.merge(micro,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];directions=[];selection={};errors=[]
    for market in ("US","KR"):
        first,second,audits,failures=market_oof(data[data.market.eq(market)].copy(),market,micro_columns)
        selected.append(first);directions.append(second);selection[market]=audits;errors.extend(failures)
    selected_frame=pd.concat(selected,ignore_index=True);direction_frame=pd.concat(directions,ignore_index=True)
    result=summarize(selected_frame);direction_result=summarize(direction_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    feature_status=json.loads((CACHE/"V44_MICROSTRUCTURE_STATUS.json").read_text(encoding="utf-8"))
    comparison={"version":"V44","hypothesis":"MICROSTRUCTURE_SIGNATURE","dev_sha256":v37.sha256(path),
        "micro_feature_sha256":v37.sha256(feature_path),"micro_feature_status":feature_status,
        "models":{"V44_INNER_SELECTED":result,"micro_direction":direction_result},
        "selection_audits":selection,"model_errors":errors}
    robustness={"version":"V44","status":status,"hypothesis":"MICROSTRUCTURE_SIGNATURE",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "direction_component":direction_result,"selection_audits":selection,"model_errors":errors,
        "seal_authorized":False}
    transfer={"version":"V44","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"completed exact 1m bars before actual entry; label-independent features; inner-only model/policy"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v44_microstructure_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "model_errors":errors},indent=2,default=str))


if __name__=="__main__":main()
