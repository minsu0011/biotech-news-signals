"""V38 material hypothesis: separate direction from trade opportunity/magnitude."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V38")))
CACHE=ROOT/"cache"
COST=.002
PARAMETERS={"n_estimators":350,"num_leaves":15,"learning_rate":.025,"min_child_samples":50}


def target_fit(train:pd.DataFrame,target:pd.DataFrame,column:str):
    model=v36.make_model("lightgbm",PARAMETERS)
    model.fit(v36.feature_frame(train),train[column].astype(int),
              model__sample_weight=train.cluster_weight)
    return model.predict_proba(v36.feature_frame(target))[:,1]


def fit_components(train:pd.DataFrame,target:pd.DataFrame,threshold:float)->dict[str,np.ndarray]:
    working=train.copy()
    working["opportunity_target"]=(working.fwd_ret_30m.abs()>=threshold).astype(int)
    working["up_payoff_target"]=(working.fwd_ret_30m>COST).astype(int)
    working["down_payoff_target"]=(working.fwd_ret_30m<-COST).astype(int)
    return {
        "direction":target_fit(working,target,"y"),
        "opportunity":target_fit(working,target,"opportunity_target"),
        "up_payoff":target_fit(working,target,"up_payoff_target"),
        "down_payoff":target_fit(working,target,"down_payoff_target"),
    }


def rank_self(values:np.ndarray)->np.ndarray:
    order=np.argsort(np.argsort(np.asarray(values,float),kind="stable"),kind="stable")
    return (order+1.)/len(order)


def rank_against(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.sort(np.asarray(reference,float))
    return np.searchsorted(reference,np.asarray(values,float),side="right")/max(1,len(reference))


def action_probability(parts:dict[str,np.ndarray])->np.ndarray:
    return np.clip(.5+.5*(parts["up_payoff"]-parts["down_payoff"]),0,1)


def candidate_values(parts:dict[str,np.ndarray],mode:str,cutoff:float,
                     references:dict[str,np.ndarray]|None=None)->tuple[np.ndarray,np.ndarray]:
    direction=parts["direction"];action=action_probability(parts)
    raw=direction if mode!="payoff_pair" else action
    if mode=="consensus":raw=.5*direction+.5*action
    margin=np.abs(raw-cutoff)
    opportunity=parts["opportunity"]
    payoff_strength=np.maximum(parts["up_payoff"],parts["down_payoff"])
    if references is None:
        ranks={"margin":rank_self(margin),"opportunity":rank_self(opportunity),
               "payoff":rank_self(payoff_strength)}
    else:
        ranks={"margin":rank_against(references["margin"],margin),
               "opportunity":rank_against(references["opportunity"],opportunity),
               "payoff":rank_against(references["payoff"],payoff_strength)}
    if mode=="direction":confidence=ranks["margin"]
    elif mode=="direction_opportunity":confidence=.45*ranks["margin"]+.55*ranks["opportunity"]
    elif mode=="payoff_pair":confidence=.35*ranks["margin"]+.65*ranks["payoff"]
    elif mode=="consensus":confidence=.35*ranks["margin"]+.30*ranks["opportunity"]+.35*ranks["payoff"]
    else:raise ValueError(mode)
    return raw,confidence


def choose_policy(frame:pd.DataFrame,threshold_parts:dict[float,dict[str,np.ndarray]])->dict[str,Any]:
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    weights=frame.cluster_weight.to_numpy(float)
    for threshold,parts in threshold_parts.items():
        for mode in ("direction","direction_opportunity","payoff_pair","consensus"):
            cutoffs=(.5,) if mode=="payoff_pair" else tuple(np.arange(.44,.561,.02))
            for cutoff in cutoffs:
                raw,confidence=candidate_values(parts,mode,float(cutoff))
                metrics=v37.metric(frame,raw,float(cutoff),weighted=True)
                floor=v37.source_floor(frame,raw,float(cutoff))
                for quantile in (.75,.80,.85):
                    confidence_threshold=float(np.quantile(confidence,quantile));high=confidence>=confidence_threshold
                    prediction=raw>=cutoff;high_weights=weights[high]
                    hc_accuracy=float(np.average(prediction[high]==y[high],weights=high_weights))
                    net=np.where(prediction[high],1.,-1.)*returns[high]-COST
                    hc_net=float(np.average(net,weights=high_weights))
                    checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                        metrics["edge"]>=0,hc_accuracy>=.55,hc_net>0,floor>=.49))
                    score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                           .4*floor+.35*hc_accuracy+8*max(-.01,min(.01,hc_net)))
                    rows.append({"magnitude_threshold":threshold,"mode":mode,
                        "direction_cutoff":float(cutoff),"confidence_quantile":quantile,
                        "confidence_threshold":confidence_threshold,"checks":checks,
                        "score":float(score),"inner_metrics":metrics,"source_floor":floor,
                        "inner_hc_accuracy":hc_accuracy,"inner_hc_net":hc_net,
                        "references":{"margin":np.abs(raw-cutoff),
                            "opportunity":parts["opportunity"],
                            "payoff":np.maximum(parts["up_payoff"],parts["down_payoff"])}})
    rows.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
    return rows[0]


def apply_policy(policy:dict[str,Any],parts:dict[str,np.ndarray])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    cutoff=float(policy["direction_cutoff"])
    raw,confidence=candidate_values(parts,policy["mode"],cutoff,policy["references"])
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def prediction_frame(valid:pd.DataFrame,probability:np.ndarray,high:np.ndarray,confidence:np.ndarray,
                     fold:int,name:str,seen:set[str])->pd.DataFrame:
    output=valid[["event_id","event_group_id","event_time_utc","market","ticker","source_family",
                  "form","y","fwd_ret_30m","text_cluster_id","cluster_weight"]].copy()
    output["prob"]=probability;output["high_conf"]=high;output["confidence_signal"]=confidence
    output["fold"]=fold;output["family"]=name;output["new_issuer"]=~output.ticker.astype(str).isin(seen)
    return output


def market_oof(frame:pd.DataFrame,market:str)->tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    selected=[];direction_only=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train)
        thresholds=(.004,.006,.010)
        inner={threshold:fit_components(inner_train,inner_valid,threshold) for threshold in thresholds}
        policy=choose_policy(inner_valid.reset_index(drop=True),inner)
        threshold=float(policy["magnitude_threshold"])
        outer=fit_components(train,valid,threshold)
        probability,high,confidence=apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(prediction_frame(valid,probability,high,confidence,fold,
                                          "V38_INNER_SELECTED",seen))
        direction=outer["direction"];direction_high=rank_against(
            np.abs(inner[threshold]["direction"]-.5),np.abs(direction-.5))>=.80
        direction_only.append(prediction_frame(valid,direction,direction_high,
                                                np.abs(direction-.5),fold,"direction",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V38 OPPORTUNITY] {market} fold={fold} mode={policy['mode']} "
              f"magnitude={threshold:.3f} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(direction_only,ignore_index=True),audits


def summarize(frame:pd.DataFrame)->dict[str,Any]:
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(
        frame,frame.prob.to_numpy(float),.5,weighted=True),"raw_row_metrics":v37.metric(
        frame,frame.prob.to_numpy(float),.5,weighted=False),"robustness":robustness,
        "research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},
                     parse_dates=["event_time_utc"])
    if data.event_group_id.duplicated().any():raise RuntimeError("V38 event-group duplication")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];direction=[];selection={}
    for market in ("US","KR"):
        market_selected,market_direction,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected.append(market_selected);direction.append(market_direction);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);direction_frame=pd.concat(direction,ignore_index=True)
    result=summarize(selected_frame);direction_result=summarize(direction_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    majority=Counter(audit["mode"] for audits in selection.values() for audit in audits).most_common(1)[0][0]
    comparison={"version":"V38","hypothesis":"OPPORTUNITY_MAGNITUDE","dev_sha256":v37.sha256(path),
                "models":{"V38_INNER_SELECTED":result,"direction_only":direction_result},
                "selection_audits":selection,"majority_mode":majority}
    robustness={"version":"V38","status":status,"hypothesis":"OPPORTUNITY_MAGNITUDE",
                "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
                "direction_baseline":direction_result,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V38","seal_outcomes_loaded":False,
              "selected_oof_by_source":result["metrics"]["by_source_family"],
              "selected_oof_by_market":result["metrics"]["by_market"],
              "selected_oof_new_issuer":result["metrics"]["new_issuer"],
              "method":"chronological outer OOF; opportunity is DEV auxiliary only and never a source filter"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v38_opportunity_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "majority_mode":majority},indent=2,default=str))


if __name__=="__main__":main()
