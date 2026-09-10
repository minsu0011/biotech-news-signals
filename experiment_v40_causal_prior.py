"""V40 material hypothesis: hierarchical causal event priors without issuer IDs."""
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
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V40")))
CACHE=ROOT/"cache"
COST=.002
ITEM_RE=re.compile(r"(?i)\b(?:item\s*)?(1\.01|2\.02|2\.03|3\.02|5\.02|7\.01|8\.01|9\.01)\b")
TAXONOMY=(
    ("regulatory_positive",re.compile(r"(?i)(fda.{0,30}approv|approval|authorized|허가|승인)")),
    ("clinical_negative",re.compile(r"(?i)(clinical hold|failed.{0,30}(trial|endpoint)|adverse|recall|반려|중단)")),
    ("clinical",re.compile(r"(?i)(phase\s*[123]|clinical|trial|endpoint|임상|시험)")),
    ("financing",re.compile(r"(?i)(offering|financ|dilut|warrant|convertible|유상증자|전환사채)")),
    ("partnership",re.compile(r"(?i)(license|collaborat|partner|milestone|기술수출|계약)")),
    ("ma",re.compile(r"(?i)(merger|acquisition|합병|인수)")),
    ("earnings",re.compile(r"(?i)(earnings|guidance|results|실적|전망)")),
    ("patent",re.compile(r"(?i)(patent|특허)")),
)


def taxonomy(row:pd.Series)->str:
    text=f"{str(row.get('headline','') or '')} {str(row.get('body','') or '')[:6000]}"
    labels=[name for name,pattern in TAXONOMY if pattern.search(text)]
    return "+".join(labels[:3]) if labels else "other"


def item_bucket(row:pd.Series)->str:
    values=sorted(set(ITEM_RE.findall(str(row.get("headline","") or ""))))
    return "+".join(value.replace(".","_") for value in values) if values else "none"


KEY_LEVELS=(
    ("market",),
    ("market","source_family"),
    ("market","event_type"),
    ("market","source_family","event_type"),
    ("market","source_family","form"),
    ("market","source_family","taxonomy"),
    ("market","source_family","form","item_bucket"),
)


class HierarchicalPrior:
    def __init__(self,alpha:float):self.alpha=float(alpha)

    @staticmethod
    def key(frame:pd.DataFrame,columns:tuple[str,...])->pd.Series:
        return frame[list(columns)].fillna("").astype(str).agg("|".join,axis=1)

    def fit(self,frame:pd.DataFrame)->"HierarchicalPrior":
        weight=frame.cluster_weight.to_numpy(float);y=frame.y.to_numpy(float)
        self.global_mean=float(np.average(y,weights=weight));self.tables=[]
        for columns in KEY_LEVELS:
            key=self.key(frame,columns)
            stats=pd.DataFrame({"key":key,"wy":weight*y,"w":weight}).groupby("key").sum()
            stats["prob"]=(stats.wy+self.alpha*self.global_mean)/(stats.w+self.alpha)
            self.tables.append((columns,stats[["prob","w"]].to_dict("index")))
        return self

    def predict(self,frame:pd.DataFrame)->np.ndarray:
        numerator=np.full(len(frame),self.global_mean*.35);denominator=np.full(len(frame),.35)
        for level,(columns,table) in enumerate(self.tables):
            keys=self.key(frame,columns);values=[];reliability=[]
            for key in keys:
                row=table.get(key)
                if row is None:values.append(self.global_mean);reliability.append(0.)
                else:
                    values.append(float(row["prob"]));reliability.append(float(row["w"])/(float(row["w"])+self.alpha))
            reliability=np.asarray(reliability)*(1.+.12*level)
            numerator+=np.asarray(values)*reliability;denominator+=reliability
        return np.clip(numerator/denominator,.02,.98)


def fit_parts(train:pd.DataFrame,target:pd.DataFrame,alpha:float)->dict[str,np.ndarray]:
    parts=v38.fit_components(train,target,.006)
    parts["payoff_action"]=v38.action_probability(parts)
    parts["causal_prior"]=HierarchicalPrior(alpha).fit(train).predict(target)
    return parts


def specs()->list[dict[str,Any]]:
    return [
        {"name":"numeric","weights":{"direction":1.}},
        {"name":"prior","weights":{"causal_prior":1.}},
        {"name":"prior25_numeric","weights":{"causal_prior":.25,"direction":.75}},
        {"name":"prior50_numeric","weights":{"causal_prior":.50,"direction":.50}},
        {"name":"prior25_action","weights":{"causal_prior":.25,"direction":.375,"payoff_action":.375}},
        {"name":"prior50_action","weights":{"causal_prior":.50,"direction":.25,"payoff_action":.25}},
    ]


def blend(parts,weights):
    return sum(parts[name]*weight for name,weight in weights.items())/sum(weights.values())


def choose_policy(frame:pd.DataFrame,by_alpha:dict[float,dict[str,np.ndarray]])->dict[str,Any]:
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for alpha,parts in by_alpha.items():
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
                    rows.append({"alpha":alpha,"candidate":spec,"direction_cutoff":float(cutoff),
                        "confidence_quantile":quantile,"confidence_threshold":threshold,
                        "checks":checks,"score":float(score),"inner_metrics":metrics,"source_floor":floor,
                        "inner_hc_accuracy":accuracy,"inner_hc_net":net,
                        "references":{"margin":np.abs(raw-cutoff),"opportunity":parts["opportunity"],
                                      "payoff":np.maximum(parts["up_payoff"],parts["down_payoff"])}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,parts):
    cutoff=float(policy["direction_cutoff"]);raw=blend(parts,policy["candidate"]["weights"])
    ref=policy["references"]
    confidence=(.4*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .3*v38.rank_against(ref["opportunity"],parts["opportunity"])+
                .3*v38.rank_against(ref["payoff"],np.maximum(parts["up_payoff"],parts["down_payoff"])))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame:pd.DataFrame,market:str):
    selected=[];prior_frames=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train)
        inner={alpha:fit_parts(inner_train,inner_valid,alpha) for alpha in (20.,50.,100.)}
        policy=choose_policy(inner_valid.reset_index(drop=True),inner)
        outer=fit_parts(train,valid,float(policy["alpha"]));prob,high,confidence=apply_policy(policy,outer)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V40_INNER_SELECTED",seen))
        prior=outer["causal_prior"];prior_high=v38.rank_against(
            np.abs(inner[float(policy["alpha"])]["causal_prior"]-.5),np.abs(prior-.5))>=.80
        prior_frames.append(v38.prediction_frame(valid,prior,prior_high,np.abs(prior-.5),fold,
                                                  "causal_prior",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V40 PRIOR] {market} fold={fold} alpha={policy['alpha']:.0f} "
              f"candidate={policy['candidate']['name']} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(prior_frames,ignore_index=True),audits


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
    data["taxonomy"]=data.apply(taxonomy,axis=1);data["item_bucket"]=data.apply(item_bucket,axis=1)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];priors=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected.append(first);priors.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);prior_frame=pd.concat(priors,ignore_index=True)
    result=summarize(selected_frame);prior_result=summarize(prior_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V40","hypothesis":"CAUSAL_EVENT_PRIOR","dev_sha256":v37.sha256(path),
                "issuer_features_used":False,"key_levels":[list(value) for value in KEY_LEVELS],
                "models":{"V40_INNER_SELECTED":result,"causal_prior_component":prior_result},
                "selection_audits":selection}
    robustness={"version":"V40","status":status,"hypothesis":"CAUSAL_EVENT_PRIOR",
                "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
                "prior_component":prior_result,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V40","seal_outcomes_loaded":False,
              "selected_oof_by_source":result["metrics"]["by_source_family"],
              "selected_oof_by_market":result["metrics"]["by_market"],
              "selected_oof_new_issuer":result["metrics"]["new_issuer"],
              "method":"past-only hierarchical event priors; no ticker/issuer key; unseen groups shrink to market/global"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v40_causal_prior_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
