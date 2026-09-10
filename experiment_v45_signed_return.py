"""V45 material hypothesis: robust signed-return and magnitude regression."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V45")))
CACHE=ROOT/"cache"
COST=.002


def make_regressor(family:str,micro_columns:list[str]):
    pipeline=v44.make_model("lightgbm_micro" if family=="lightgbm_huber" else "catboost_gpu_micro",
                            micro_columns)
    if family=="lightgbm_huber":
        model=LGBMRegressor(n_estimators=450,num_leaves=15,learning_rate=.02,min_child_samples=50,
            objective="huber",alpha=.9,random_state=v36.SEED,n_jobs=runtime_limits.THREAD_COUNT,
            subsample=.85,colsample_bytree=.8,reg_lambda=5.,verbosity=-1,deterministic=True,force_col_wise=True)
    elif family=="catboost_gpu_rmse":
        model=CatBoostRegressor(iterations=400,depth=5,learning_rate=.025,l2_leaf_reg=15.,
            loss_function="RMSE",random_seed=v36.SEED,task_type="GPU",devices="0",verbose=False,
            allow_writing_files=False,random_strength=.5)
    else:raise ValueError(family)
    pipeline.set_params(model=model);return pipeline


def fit_parts(train:pd.DataFrame,target:pd.DataFrame,family:str,micro_columns:list[str]):
    lower,upper=train.fwd_ret_30m.quantile([.01,.99]);signed=train.fwd_ret_30m.clip(lower,upper)
    magnitude=train.fwd_ret_30m.abs().clip(upper=train.fwd_ret_30m.abs().quantile(.99))
    signed_model=make_regressor(family,micro_columns);magnitude_model=make_regressor(family,micro_columns)
    signed_model.fit(v44.feature_frame(train,micro_columns),signed,
                     model__sample_weight=train.cluster_weight)
    magnitude_model.fit(v44.feature_frame(train,micro_columns),magnitude,
                        model__sample_weight=train.cluster_weight)
    signed_prediction=signed_model.predict(v44.feature_frame(target,micro_columns))
    magnitude_prediction=np.maximum(0.,magnitude_model.predict(v44.feature_frame(target,micro_columns)))
    scale=max(.002,float(np.nanmedian(np.abs(signed-signed.median())))*2.)
    regression_probability=1./(1.+np.exp(-np.clip(signed_prediction/scale,-20,20)))
    classifier=v44.make_model("lightgbm_micro" if family=="lightgbm_huber" else "catboost_gpu_micro",
                              micro_columns)
    classifier.fit(v44.feature_frame(train,micro_columns),train.y.astype(int),
                   model__sample_weight=train.cluster_weight)
    direction=classifier.predict_proba(v44.feature_frame(target,micro_columns))[:,1]
    return {"regression":regression_probability,"direction":direction,"magnitude":magnitude_prediction,
            "signed_score":signed_prediction,"scale":scale}


def specs():
    return [
        {"name":"regression","weights":{"regression":1.}},
        {"name":"classifier","weights":{"direction":1.}},
        {"name":"blend25reg","weights":{"regression":.25,"direction":.75}},
        {"name":"blend50reg","weights":{"regression":.50,"direction":.50}},
        {"name":"blend75reg","weights":{"regression":.75,"direction":.25}},
    ]


def blend(parts,weights):return sum(parts[name]*value for name,value in weights.items())/sum(weights.values())


def choose_policy(frame,by_family):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for family,parts in by_family.items():
        for spec in specs():
            raw=blend(parts,spec["weights"])
            for cutoff in np.arange(.44,.561,.02):
                margin=v38.rank_self(np.abs(raw-cutoff));magnitude=v38.rank_self(parts["magnitude"])
                score=v38.rank_self(np.abs(parts["signed_score"]));confidence=.35*margin+.4*magnitude+.25*score
                metrics=v37.metric(frame,raw,float(cutoff),True);floor=v37.source_floor(frame,raw,float(cutoff))
                for quantile in (.75,.80,.85):
                    threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold
                    prediction=raw>=cutoff;weight=sample_weight[high]
                    accuracy=float(np.average(prediction[high]==y[high],weights=weight))
                    net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,weights=weight))
                    checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                                metrics["edge"]>=0,accuracy>=.55,net>0,floor>=.49))
                    objective=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                               .4*floor+.35*accuracy+8*max(-.01,min(.01,net)))
                    rows.append({"model_family":family,"candidate":spec,"direction_cutoff":float(cutoff),
                        "confidence_quantile":quantile,"confidence_threshold":threshold,"checks":checks,
                        "score":float(objective),"inner_metrics":metrics,"source_floor":floor,
                        "inner_hc_accuracy":accuracy,"inner_hc_net":net,
                        "references":{"margin":np.abs(raw-cutoff),"magnitude":parts["magnitude"],
                                      "signed_score":np.abs(parts["signed_score"])}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,parts):
    cutoff=float(policy["direction_cutoff"]);raw=blend(parts,policy["candidate"]["weights"]);ref=policy["references"]
    confidence=(.35*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .4*v38.rank_against(ref["magnitude"],parts["magnitude"])+
                .25*v38.rank_against(ref["signed_score"],np.abs(parts["signed_score"])))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame,market,micro_columns):
    selected=[];regression_frames=[];audits=[];errors=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);inner={}
        for family in ("lightgbm_huber","catboost_gpu_rmse"):
            try:inner[family]=fit_parts(inner_train,inner_valid,family,micro_columns)
            except Exception as error:
                errors.append({"market":market,"fold":fold,"stage":"inner","family":family,"error":repr(error)})
        if not inner:raise RuntimeError(f"V45 all inner models failed {market} fold={fold}")
        policy=choose_policy(inner_valid.reset_index(drop=True),inner);family=policy["model_family"]
        outer=fit_parts(train,valid,family,micro_columns);prob,high,confidence=apply_policy(policy,outer)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V45_INNER_SELECTED",seen))
        raw=outer["regression"];component_high=v38.rank_against(
            inner[family]["magnitude"],outer["magnitude"])>=.80
        regression_frames.append(v38.prediction_frame(valid,raw,component_high,outer["magnitude"],fold,
                                                        f"{family}_regression",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V45 RETURN] {market} fold={fold} family={family} "
              f"candidate={policy['candidate']['name']} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(regression_frames,ignore_index=True),audits,errors


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz";feature_path=CACHE/"v44_microstructure_features.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    micro=pd.read_csv(feature_path,compression="gzip",low_memory=False);micro_columns=[c for c in micro if c!="event_id"]
    data=data.merge(micro,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];components=[];selection={};errors=[]
    for market in ("US","KR"):
        first,second,audits,failures=market_oof(data[data.market.eq(market)].copy(),market,micro_columns)
        selected.append(first);components.append(second);selection[market]=audits;errors.extend(failures)
    selected_frame=pd.concat(selected,ignore_index=True);component_frame=pd.concat(components,ignore_index=True)
    result=summarize(selected_frame);component=summarize(component_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V45","hypothesis":"SIGNED_RETURN_REGRESSION","dev_sha256":v37.sha256(path),
        "micro_feature_sha256":v37.sha256(feature_path),
        "models":{"V45_INNER_SELECTED":result,"signed_return_component":component},
        "selection_audits":selection,"model_errors":errors}
    robustness={"version":"V45","status":status,"hypothesis":"SIGNED_RETURN_REGRESSION",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "signed_return_component":component,"selection_audits":selection,"model_errors":errors,
        "seal_authorized":False}
    transfer={"version":"V45","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"train-fold winsorized signed return and absolute magnitude; inner-only family/blend/cutoff"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v45_signed_return_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "model_errors":errors},indent=2,default=str))


if __name__=="__main__":main()
