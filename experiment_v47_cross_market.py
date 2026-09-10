"""V47 material hypothesis: past-US shared representation for small KR DEV."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V47")))
CACHE=ROOT/"cache"
COST=.002


def balanced_weight(frame:pd.DataFrame)->np.ndarray:
    weight=frame.cluster_weight.to_numpy(float);market=frame.market.to_numpy(str)
    for value in np.unique(market):
        mask=market==value;total=weight[mask].sum()
        if total>0:weight[mask]*=1./total
    return weight*len(frame)/weight.sum()


def target_fit(train,target,label,family,feature_columns,balance=False):
    model=v44.make_model(family,feature_columns);sample=balanced_weight(train) if balance else train.cluster_weight
    model.fit(v44.feature_frame(train,feature_columns),np.asarray(label,int),model__sample_weight=sample)
    return model.predict_proba(v44.feature_frame(target,feature_columns))[:,1]


def four_parts(train,target,family,features,balance=False):
    return {"direction":target_fit(train,target,train.y,family,features,balance),
        "opportunity":target_fit(train,target,train.fwd_ret_30m.abs()>=.006,family,features,balance),
        "up_payoff":target_fit(train,target,train.fwd_ret_30m>COST,family,features,balance),
        "down_payoff":target_fit(train,target,train.fwd_ret_30m<-COST,family,features,balance)}


def fit_parts(local_train,us_history,target,family,features):
    local=four_parts(local_train,target,family,features,False)
    pool=pd.concat([us_history,local_train],ignore_index=True)
    cross=four_parts(pool,target,family,features,True)
    return {"local_direction":local["direction"],"cross_direction":cross["direction"],
        "local_action":v38.action_probability(local),"cross_action":v38.action_probability(cross),
        "opportunity":.5*local["opportunity"]+.5*cross["opportunity"],
        "payoff":.5*np.maximum(local["up_payoff"],local["down_payoff"])+
                 .5*np.maximum(cross["up_payoff"],cross["down_payoff"])}


def specs():
    return [
        {"name":"local","weights":{"local_direction":1.}},
        {"name":"cross","weights":{"cross_direction":1.}},
        {"name":"blend_direction","weights":{"local_direction":.5,"cross_direction":.5}},
        {"name":"local_action","weights":{"local_direction":.5,"local_action":.5}},
        {"name":"cross_action","weights":{"cross_direction":.5,"cross_action":.5}},
        {"name":"shared_blend","weights":{"local_direction":.25,"cross_direction":.25,
                                               "local_action":.25,"cross_action":.25}},
    ]


def blend(parts,weights):return sum(parts[name]*value for name,value in weights.items())/sum(weights.values())


def choose_policy(frame,by_family):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for family,parts in by_family.items():
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
                    rows.append({"model_family":family,"candidate":spec,"direction_cutoff":float(cutoff),
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


def kr_oof(kr:pd.DataFrame,us_history:pd.DataFrame,features:list[str]):
    if us_history.event_time_utc.max()>=kr.event_time_utc.min():
        raise RuntimeError("US supplement is not strictly historical to KR")
    selected=[];cross_frames=[];audits=[];errors=[]
    for train,valid,fold in v37.chronological_folds(kr):
        inner_train,inner_valid=v37.split_inner(train);inner={}
        for family in ("lightgbm_micro","catboost_gpu_micro"):
            try:inner[family]=fit_parts(inner_train,us_history,inner_valid,family,features)
            except Exception as error:errors.append({"fold":fold,"stage":"inner","family":family,"error":repr(error)})
        if not inner:raise RuntimeError(f"V47 all models failed fold={fold}")
        policy=choose_policy(inner_valid.reset_index(drop=True),inner);family=policy["model_family"]
        outer=fit_parts(train,us_history,valid,family,features);prob,high,confidence=apply_policy(policy,outer)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V47_KR_TRANSFER",seen))
        raw=outer["cross_direction"];component_high=v38.rank_against(
            np.abs(inner[family]["cross_direction"]-.5),np.abs(raw-.5))>=.80
        cross_frames.append(v38.prediction_frame(valid,raw,component_high,np.abs(raw-.5),fold,
                                                  f"{family}_cross",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":"KR","US_history_n":len(us_history),"KR_train_n":len(train),
                      "inner_train_n":len(inner_train),"inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "US_history_end":us_history.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V47 TRANSFER] KR fold={fold} family={family} candidate={policy['candidate']['name']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(cross_frames,ignore_index=True),audits,errors


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
    us_history=data[data.market.eq("US")].copy();kr=data[data.market.eq("KR")].copy()
    kr_selected,cross,audits,errors=kr_oof(kr,us_history,features)
    us_cached=pd.read_csv(CACHE/"v46_article_oof.csv.gz",compression="gzip",low_memory=False,
                          parse_dates=["event_time_utc"],dtype={"ticker":str})
    us_cached=us_cached[us_cached.market.eq("US")].copy();us_cached["family"]="V46_US_FIXED_REUSE"
    selected_frame=pd.concat([us_cached,kr_selected],ignore_index=True);result=v44.summarize(selected_frame)
    cross_result=v44.summarize(cross)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V47","hypothesis":"CROSS_MARKET_TRANSFER","dev_sha256":v37.sha256(path),
        "US_fixed_cache_sha256":v37.sha256(CACHE/"v46_article_oof.csv.gz"),
        "models":{"V47_COMBINED":result,"KR_cross_direction":cross_result},
        "selection_audits":{"KR":audits},"model_errors":errors}
    robustness={"version":"V47","status":status,"hypothesis":"CROSS_MARKET_TRANSFER",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "KR_cross_component":cross_result,"selection_audits":{"KR":audits},"model_errors":errors,
        "seal_authorized":False}
    transfer={"version":"V47","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"US exact DEV ends before KR begins; pooled weights balance markets; KR inner-only blend"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v47_cross_market_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "model_errors":errors},indent=2,default=str))


if __name__=="__main__":main()
