"""V48 material hypothesis: GPU XGBoost event/form specialists."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V48")))
CACHE=ROOT/"cache"
COST=.002


def make_model(features):
    pipeline=v44.make_model("lightgbm_micro",features)
    model=XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.025,min_child_weight=20,
        subsample=.85,colsample_bytree=.80,reg_lambda=5.,reg_alpha=.1,objective="binary:logistic",
        eval_metric="auc",tree_method="hist",device="cuda",random_state=v36.SEED,
        n_jobs=runtime_limits.THREAD_COUNT)
    pipeline.set_params(model=model);return pipeline


def fit_binary(train,target,label,features):
    model=make_model(features);model.fit(v44.feature_frame(train,features),np.asarray(label,int),
        model__sample_weight=train.cluster_weight)
    return model.predict_proba(v44.feature_frame(target,features))[:,1]


def expert_prediction(train,target,pooled,key,features):
    output=pooled.copy();audit={}
    for value,target_group in target.groupby(key):
        train_group=train[train[key].eq(value)].copy();mask=target.index.isin(target_group.index)
        audit[str(value)]={"train_n":len(train_group),"target_n":len(target_group),"fitted":False}
        if len(train_group)<600 or train_group.y.nunique()<2:continue
        prediction=fit_binary(train_group,target_group,train_group.y,features)
        shrink=len(train_group)/(len(train_group)+800.)
        output[mask]=(1.-shrink)*pooled[mask]+shrink*prediction;audit[str(value)]["fitted"]=True
        audit[str(value)]["shrinkage_weight"]=shrink
    return output,audit


def fit_parts(train,target,features):
    direction=fit_binary(train,target,train.y,features)
    opportunity=fit_binary(train,target,train.fwd_ret_30m.abs()>=.006,features)
    up=fit_binary(train,target,train.fwd_ret_30m>COST,features)
    down=fit_binary(train,target,train.fwd_ret_30m<-COST,features)
    form,form_audit=expert_prediction(train,target,direction,"source_form_key",features)
    event,event_audit=expert_prediction(train,target,direction,"event_key",features)
    return {"direction":direction,"payoff_action":np.clip(.5+.5*(up-down),0,1),
            "opportunity":opportunity,"payoff":np.maximum(up,down),"form_expert":form,"event_expert":event},\
           {"form":form_audit,"event":event_audit}


def specs():
    return [
        {"name":"pooled","weights":{"direction":1.}},
        {"name":"pooled_action","weights":{"direction":.5,"payoff_action":.5}},
        {"name":"form","weights":{"form_expert":1.}},
        {"name":"event","weights":{"event_expert":1.}},
        {"name":"form_event","weights":{"form_expert":.5,"event_expert":.5}},
        {"name":"form_action","weights":{"form_expert":.5,"payoff_action":.5}},
        {"name":"all_experts","weights":{"direction":.2,"form_expert":.3,"event_expert":.3,
                                              "payoff_action":.2}},
    ]


def blend(parts,weights):return sum(parts[name]*value for name,value in weights.items())/sum(weights.values())


def choose_policy(frame,parts):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for spec in specs():
        raw=blend(parts,spec["weights"])
        for cutoff in np.arange(.44,.561,.02):
            margin=v38.rank_self(np.abs(raw-cutoff));opportunity=v38.rank_self(parts["opportunity"])
            payoff=v38.rank_self(parts["payoff"]);confidence=.4*margin+.3*opportunity+.3*payoff
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
                                  "payoff":parts["payoff"]}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,parts):
    cutoff=float(policy["direction_cutoff"]);raw=blend(parts,policy["candidate"]["weights"]);ref=policy["references"]
    confidence=(.4*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .3*v38.rank_against(ref["opportunity"],parts["opportunity"])+
                .3*v38.rank_against(ref["payoff"],parts["payoff"]))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame,market,features):
    selected=[];experts=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);inner,inner_expert=fit_parts(inner_train,inner_valid,features)
        policy=choose_policy(inner_valid.reset_index(drop=True),inner);outer,outer_expert=fit_parts(train,valid,features)
        prob,high,confidence=apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(v38.prediction_frame(valid,prob,high,confidence,fold,"V48_INNER_SELECTED",seen))
        raw=.5*outer["form_expert"]+.5*outer["event_expert"]
        component_high=v38.rank_against(np.abs(.5*inner["form_expert"]+.5*inner["event_expert"]-.5),
                                         np.abs(raw-.5))>=.80
        experts.append(v38.prediction_frame(valid,raw,component_high,np.abs(raw-.5),fold,"xgb_experts",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "inner_experts":inner_expert,"outer_experts":outer_expert,
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V48 XGB] {market} fold={fold} candidate={policy['candidate']['name']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(experts,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz"
    micro_path=CACHE/"v44_microstructure_features.csv.gz";article_path=CACHE/"v46_article_features.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    micro=pd.read_csv(micro_path,compression="gzip",low_memory=False);article=pd.read_csv(article_path,compression="gzip")
    features=[c for c in micro if c!="event_id"]+[c for c in article if c!="event_id"]
    data=data.merge(micro,on="event_id",how="left",validate="one_to_one").merge(
        article,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    data["source_form_key"]=data.source_family.fillna("").astype(str)+"|"+data.form.fillna("").astype(str)
    data["event_key"]=data.event_type.fillna("").astype(str)
    selected=[];experts=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market,features)
        selected.append(first);experts.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);expert_frame=pd.concat(experts,ignore_index=True)
    result=v44.summarize(selected_frame);expert_result=v44.summarize(expert_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V48","hypothesis":"EVENT_FORM_XGB_EXPERT","dev_sha256":v37.sha256(path),
        "models":{"V48_INNER_SELECTED":result,"xgb_experts":expert_result},"selection_audits":selection}
    robustness={"version":"V48","status":status,"hypothesis":"EVENT_FORM_XGB_EXPERT",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "expert_component":expert_result,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V48","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"GPU XGBoost; experts require past n>=600 and shrink n/(n+800); issuer excluded"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v48_xgb_expert_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
