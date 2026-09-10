"""V43 material hypothesis: time-decayed source experts with pooled shrinkage."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V43")))
CACHE=ROOT/"cache"
COST=.002
PARAMETERS={"n_estimators":350,"num_leaves":15,"learning_rate":.025,"min_child_samples":50}


def weights(frame:pd.DataFrame,half_life_days:float|None)->np.ndarray:
    base=frame.cluster_weight.to_numpy(float)
    if half_life_days is None:return base
    age=(frame.event_time_utc.max()-frame.event_time_utc).dt.total_seconds().to_numpy()/86400.
    decay=np.maximum(.03,np.power(.5,age/half_life_days));return base*decay/np.mean(decay)


def fit_direction(train:pd.DataFrame,target:pd.DataFrame,half_life_days:float|None)->np.ndarray:
    model=v36.make_model("lightgbm",PARAMETERS)
    model.fit(v36.feature_frame(train),train.y.astype(int),model__sample_weight=weights(train,half_life_days))
    return model.predict_proba(v36.feature_frame(target))[:,1]


def source_prediction(train:pd.DataFrame,target:pd.DataFrame,pooled:np.ndarray,
                      half_life_days:float|None)->np.ndarray:
    output=pooled.copy()
    for source,target_source in target.groupby("source_family"):
        train_source=train[train.source_family.eq(source)].copy();mask=target.index.isin(target_source.index)
        if len(train_source)<300 or train_source.y.nunique()<2:continue
        expert=fit_direction(train_source,target_source,half_life_days)
        shrink=len(train_source)/(len(train_source)+500.)
        output[mask]=(1.-shrink)*pooled[mask]+shrink*expert
    return output


def fit_parts(train:pd.DataFrame,target:pd.DataFrame,half_life_days:float)->dict[str,np.ndarray]:
    base=v38.fit_components(train,target,.006);base["payoff_action"]=v38.action_probability(base)
    decayed=fit_direction(train,target,half_life_days)
    base["decayed_direction"]=decayed
    base["source_direction"]=source_prediction(train,target,base["direction"],None)
    base["source_decayed"]=source_prediction(train,target,decayed,half_life_days)
    return base


def specs():
    return [
        {"name":"pooled","weights":{"direction":1.}},
        {"name":"pooled_action","weights":{"direction":.5,"payoff_action":.5}},
        {"name":"decayed","weights":{"decayed_direction":1.}},
        {"name":"source","weights":{"source_direction":1.}},
        {"name":"source_decayed","weights":{"source_decayed":1.}},
        {"name":"source_action","weights":{"source_direction":.5,"payoff_action":.5}},
        {"name":"regime_action","weights":{"source_decayed":.5,"payoff_action":.5}},
        {"name":"three_way","weights":{"direction":.25,"source_decayed":.5,"payoff_action":.25}},
    ]


def blend(parts,weight):return sum(parts[name]*value for name,value in weight.items())/sum(weight.values())


def choose_policy(frame:pd.DataFrame,parts:dict[str,np.ndarray])->dict[str,Any]:
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for spec in specs():
        raw=blend(parts,spec["weights"])
        for cutoff in np.arange(.44,.561,.02):
            margin=v38.rank_self(np.abs(raw-cutoff));opportunity=v38.rank_self(parts["opportunity"])
            payoff=v38.rank_self(np.maximum(parts["up_payoff"],parts["down_payoff"]))
            confidence=.4*margin+.3*opportunity+.3*payoff
            metrics=v37.metric(frame,raw,float(cutoff),True);floor=v37.source_floor(frame,raw,float(cutoff))
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold
                prediction=raw>=cutoff;weight=sample_weight[high]
                accuracy=float(np.average(prediction[high]==y[high],weights=weight))
                net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,weights=weight))
                checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                            metrics["edge"]>=0,accuracy>=.55,net>0,floor>=.49))
                score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                       .4*floor+.35*accuracy+8*max(-.01,min(.01,net)))
                rows.append({"candidate":spec,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,"checks":checks,
                    "score":float(score),"inner_metrics":metrics,"source_floor":floor,
                    "inner_hc_accuracy":accuracy,"inner_hc_net":net,
                    "references":{"margin":np.abs(raw-cutoff),"opportunity":parts["opportunity"],
                                  "payoff":np.maximum(parts["up_payoff"],parts["down_payoff"])}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,parts):
    cutoff=float(policy["direction_cutoff"]);raw=blend(parts,policy["candidate"]["weights"]);ref=policy["references"]
    confidence=(.4*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .3*v38.rank_against(ref["opportunity"],parts["opportunity"])+
                .3*v38.rank_against(ref["payoff"],np.maximum(parts["up_payoff"],parts["down_payoff"])))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame:pd.DataFrame,market:str):
    half_life=1095. if market=="US" else 10.;selected=[];source_only=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);inner=fit_parts(inner_train,inner_valid,half_life)
        policy=choose_policy(inner_valid.reset_index(drop=True),inner);outer=fit_parts(train,valid,half_life)
        probability,high,confidence=apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(v38.prediction_frame(valid,probability,high,confidence,fold,
                                              "V43_INNER_SELECTED",seen))
        raw=outer["source_decayed"];component_high=v38.rank_against(
            np.abs(inner["source_decayed"]-.5),np.abs(raw-.5))>=.80
        source_only.append(v38.prediction_frame(valid,raw,component_high,np.abs(raw-.5),fold,
                                                 "source_decayed",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"half_life_days":half_life,"train_n":len(train),
                      "inner_train_n":len(inner_train),"inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "source_train_counts":train.source_family.value_counts().to_dict(),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V43 SOURCE] {market} fold={fold} candidate={policy['candidate']['name']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(source_only,ignore_index=True),audits


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];components=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected.append(first);components.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);component_frame=pd.concat(components,ignore_index=True)
    result=summarize(selected_frame);component=summarize(component_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V43","hypothesis":"SOURCE_REGIME_EXPERT","dev_sha256":v37.sha256(path),
        "issuer_features_used":False,"models":{"V43_INNER_SELECTED":result,"source_decayed":component},
        "selection_audits":selection}
    robustness={"version":"V43","status":status,"hypothesis":"SOURCE_REGIME_EXPERT",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "source_decayed_component":component,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V43","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"past-only source experts; n/(n+500) shrinkage; US 3y and KR 10d half-life; ticker excluded"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v43_source_regime_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
