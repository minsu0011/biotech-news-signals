"""V39 material hypothesis: SEC item/form/EX-99 structured semantics."""
from __future__ import annotations

import json
import os
import re
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
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V39")))
CACHE=ROOT/"cache"
COST=.002
ITEM_RE=re.compile(r"(?i)\bitem\s+(1\.01|2\.02|2\.03|3\.02|5\.02|7\.01|8\.01|9\.01)\b")
ANCHOR_RE=re.compile(r"(?i)(phase\s*[123]|clinical|trial|endpoint|fda|approv|complete response|"
                     r"safety|adverse|recall|offering|financ|dilut|warrant|convertible|license|"
                     r"collaborat|milestone|merger|acquisition|earnings|guidance|patent|exhibit\s*99)")


def structured_sec(row:pd.Series)->str:
    if str(row.get("source_family"))!="US_SEC":return "NON_SEC_FALLBACK"
    headline=str(row.get("headline","") or "");body=str(row.get("body","") or "")
    items=[]
    for value in ITEM_RE.findall(headline+" "+body[:16000]):
        token=value.replace(".","_");items.append(f"item_{token}")
    tokens=[f"form_{str(row.get('form','')).lower().replace('-','_')}",
            f"event_{str(row.get('event_type','')).lower()}",f"body_present_{int(bool(body.strip()))}"]
    tokens.extend(sorted(set(items)))
    snippets=[]
    for match in list(ITEM_RE.finditer(body))[:12]:
        snippets.append(body[match.start():min(len(body),match.start()+1800)])
    for match in list(ANCHOR_RE.finditer(body))[:20]:
        snippets.append(body[max(0,match.start()-180):min(len(body),match.end()+520)])
    if not snippets and body:
        snippets=[body[800:3200],body[-1000:]]
    text=" ".join(tokens+[headline]+snippets)
    text=re.sub(r"\b\d{5,}\b"," LONGNUM ",text)
    text=re.sub(r"\s+"," ",text)
    return text[:16000]


def fit_parts(train:pd.DataFrame,target:pd.DataFrame)->dict[str,np.ndarray]:
    base=v38.fit_components(train,target,.006)
    structured=base["direction"].copy();sec_train=train[train.source_family.eq("US_SEC")].copy()
    target_mask=target.source_family.eq("US_SEC").to_numpy()
    if len(sec_train)>=300 and sec_train.y.nunique()==2 and target_mask.any():
        model=v37.TextModel("sec_structured","word").fit(sec_train)
        structured[target_mask]=model.predict(target.loc[target_mask])
    base["sec_structured"]=structured
    base["payoff_action"]=v38.action_probability(base)
    return base


def rank_self(values:np.ndarray)->np.ndarray:
    return v38.rank_self(values)


def rank_against(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    return v38.rank_against(reference,values)


def candidate_specs(market:str)->list[dict[str,Any]]:
    base=[{"name":"numeric_action","weights":{"direction":.5,"payoff_action":.5}},
          {"name":"numeric_only","weights":{"direction":1.}}]
    if market=="US":
        base.extend([
            {"name":"sec_direction","weights":{"sec_structured":1.}},
            {"name":"sec25_action","weights":{"direction":.375,"payoff_action":.375,
                                                   "sec_structured":.25}},
            {"name":"sec50_action","weights":{"direction":.25,"payoff_action":.25,
                                                   "sec_structured":.50}},
            {"name":"sec75_direction","weights":{"direction":.25,"sec_structured":.75}},
        ])
    return base


def blend(parts:dict[str,np.ndarray],weights:dict[str,float])->np.ndarray:
    return sum(parts[name]*weight for name,weight in weights.items())/sum(weights.values())


def choose_policy(frame:pd.DataFrame,parts:dict[str,np.ndarray],market:str)->dict[str,Any]:
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for spec in candidate_specs(market):
        raw=blend(parts,spec["weights"])
        for cutoff in np.arange(.44,.561,.02):
            margin=rank_self(np.abs(raw-cutoff));opportunity=rank_self(parts["opportunity"])
            payoff=rank_self(np.maximum(parts["up_payoff"],parts["down_payoff"]))
            confidence=.40*margin+.30*opportunity+.30*payoff
            metrics=v37.metric(frame,raw,float(cutoff),weighted=True)
            floor=v37.source_floor(frame,raw,float(cutoff))
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold
                prediction=raw>=cutoff;weight=sample_weight[high]
                hc_acc=float(np.average(prediction[high]==y[high],weights=weight))
                net=np.where(prediction[high],1.,-1.)*returns[high]-COST
                hc_net=float(np.average(net,weights=weight))
                checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                    metrics["edge"]>=0,hc_acc>=.55,hc_net>0,floor>=.49))
                score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                       .4*floor+.35*hc_acc+8*max(-.01,min(.01,hc_net)))
                rows.append({"candidate":spec,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,
                    "checks":checks,"score":float(score),"inner_metrics":metrics,
                    "source_floor":floor,"inner_hc_accuracy":hc_acc,"inner_hc_net":hc_net,
                    "references":{"margin":np.abs(raw-cutoff),"opportunity":parts["opportunity"],
                                  "payoff":np.maximum(parts["up_payoff"],parts["down_payoff"])}})
    rows.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
    return rows[0]


def apply_policy(policy:dict[str,Any],parts:dict[str,np.ndarray]):
    cutoff=float(policy["direction_cutoff"]);raw=blend(parts,policy["candidate"]["weights"])
    references=policy["references"]
    margin=rank_against(references["margin"],np.abs(raw-cutoff))
    opportunity=rank_against(references["opportunity"],parts["opportunity"])
    payoff=rank_against(references["payoff"],np.maximum(parts["up_payoff"],parts["down_payoff"]))
    confidence=.40*margin+.30*opportunity+.30*payoff
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def prediction_frame(valid:pd.DataFrame,probability,high,confidence,fold,name,seen):
    return v38.prediction_frame(valid,probability,high,confidence,fold,name,seen)


def market_oof(frame:pd.DataFrame,market:str):
    selected=[];structured=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);inner=fit_parts(inner_train,inner_valid)
        policy=choose_policy(inner_valid.reset_index(drop=True),inner,market);outer=fit_parts(train,valid)
        probability,high,confidence=apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(prediction_frame(valid,probability,high,confidence,fold,
                                          "V39_INNER_SELECTED",seen))
        raw=outer["sec_structured"]
        baseline_high=rank_against(np.abs(inner["sec_structured"]-.5),np.abs(raw-.5))>=.80
        structured.append(prediction_frame(valid,raw,baseline_high,np.abs(raw-.5),fold,
                                            "sec_structured",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V39 SEC] {market} fold={fold} candidate={policy['candidate']['name']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(structured,ignore_index=True),audits


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},
                     parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["sec_structured"]=data.apply(structured_sec,axis=1)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];structured=[];selection={}
    for market in ("US","KR"):
        a,b,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected.append(a);structured.append(b);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);structured_frame=pd.concat(structured,ignore_index=True)
    result=summarize(selected_frame);structured_result=summarize(structured_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V39","hypothesis":"SEC_STRUCTURED_SEMANTICS","dev_sha256":v37.sha256(path),
                "models":{"V39_INNER_SELECTED":result,"sec_structured_component":structured_result},
                "selection_audits":selection}
    robustness={"version":"V39","status":status,"hypothesis":"SEC_STRUCTURED_SEMANTICS",
                "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
                "component":structured_result,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V39","seal_outcomes_loaded":False,
              "selected_oof_by_source":result["metrics"]["by_source_family"],
              "selected_oof_by_market":result["metrics"]["by_market"],
              "selected_oof_new_issuer":result["metrics"]["new_issuer"],
              "body_missing_policy":"form/item/headline fallback; no literal nan token",
              "method":"SEC expert fitted only on past SEC rows inside each fold; non-SEC numeric fallback"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v39_sec_semantics_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},
                     indent=2,default=str))


if __name__=="__main__":main()
