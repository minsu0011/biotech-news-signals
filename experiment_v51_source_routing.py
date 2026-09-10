"""V51 material hypothesis: prequential source routing of V46 vs E5 experts."""
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
import experiment_v50_embeddings as v50
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V51")))
CACHE=ROOT/"cache"
COST=.002


def candidates(frame):
    base=frame.base_prob.to_numpy(float);e5=frame.e5_prob.to_numpy(float)
    return {"base":base,"e5":e5,"e525":.75*base+.25*e5,
            "e5_50":.5*base+.5*e5,"e5_75":.25*base+.75*e5}


def choose_policy(frame):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    weight=frame.cluster_weight.to_numpy(float);agreement=1.-np.abs(frame.base_prob-frame.e5_prob).to_numpy(float)
    for name,raw in candidates(frame).items():
        for cutoff in np.arange(.44,.561,.02):
            margin=v38.rank_self(np.abs(raw-cutoff));agree=v38.rank_self(agreement);confidence=.75*margin+.25*agree
            metrics=v37.metric(frame,raw,float(cutoff),True)
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold;prediction=raw>=cutoff
                accuracy=float(np.average(prediction[high]==y[high],weights=weight[high]))
                net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,weights=weight[high]))
                checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                            metrics["edge"]>=0,accuracy>=.55,net>0))
                score=2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+.4*accuracy+8*max(-.01,min(.01,net))
                rows.append({"candidate":name,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,"checks":checks,
                    "score":float(score),"past_metrics":metrics,"past_hc_accuracy":accuracy,"past_hc_net":net,
                    "references":{"margin":np.abs(raw-cutoff),"agreement":agreement}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(frame,policy):
    raw=candidates(frame)[policy["candidate"]];cutoff=float(policy["direction_cutoff"]);ref=policy["references"]
    agreement=1.-np.abs(frame.base_prob-frame.e5_prob).to_numpy(float)
    confidence=(.75*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .25*v38.rank_against(ref["agreement"],agreement))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def route(frame):
    outputs=[];audits=[]
    for market in ("US","KR"):
        market_frame=frame[frame.market.eq(market)].copy()
        for fold in sorted(market_frame.fold.unique()):
            fold_frame=market_frame[market_frame.fold.eq(fold)].copy()
            for source,current in fold_frame.groupby("source_family",sort=True):
                past=market_frame[market_frame.fold.lt(fold)&market_frame.source_family.eq(source)].copy()
                past=past[~past.text_cluster_id.isin(set(current.text_cluster_id))]
                if len(past)<30:
                    probability=current.base_prob.to_numpy(float);high=current.base_high.to_numpy(bool)
                    confidence=current.base_confidence.to_numpy(float)
                    policy={"candidate":"base","selection":"FIXED_FALLBACK","past_n":len(past)}
                else:
                    policy=choose_policy(past);probability,high,confidence=apply_policy(current,policy)
                    policy["selection"]="PAST_OOF_ONLY";policy["past_n"]=len(past)
                output=current[["event_id","event_group_id","event_time_utc","market","ticker",
                    "source_family","form","y","fwd_ret_30m","text_cluster_id","cluster_weight",
                    "fold","new_issuer"]].copy()
                output["prob"]=probability;output["high_conf"]=high;output["confidence_signal"]=confidence
                output["family"]="V51_PREQUENTIAL_SOURCE_ROUTE";outputs.append(output)
                audit={key:value for key,value in policy.items() if key!="references"}
                audit.update({"market":market,"fold":int(fold),"source_family":source,
                              "valid_n":len(current),"valid_start":current.event_time_utc.min(),
                              "past_end":past.event_time_utc.max() if len(past) else None})
                audits.append(audit)
                print(f"[V51 ROUTE] {market} fold={fold} source={source} "
                      f"candidate={policy['candidate']} past={len(past)}",flush=True)
    return pd.concat(outputs,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz"
    embedding_path=CACHE/"v50_multilingual_e5_embeddings.npz"
    status=json.loads((CACHE/"V50_EMBEDDING_STATUS.json").read_text(encoding="utf-8"))
    if v37.sha256(embedding_path)!=status["embedding_sha256"]:raise RuntimeError("V51 embedding hash mismatch")
    loaded=np.load(embedding_path,allow_pickle=True);ids=loaded["event_id"].astype(str)
    headline=loaded["headline"].astype(np.float32);semantic=loaded["semantic"].astype(np.float32)
    mapping={event_id:index for index,event_id in enumerate(ids)}
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    data["embedding_index"]=data.event_id.astype(str).map(mapping).astype(int)
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    e5_components=[]
    for market in ("US","KR"):
        _,component,_=v50.market_oof(data[data.market.eq(market)].copy(),market,headline,semantic)
        e5_components.append(component)
    e5=pd.concat(e5_components,ignore_index=True)
    base=pd.read_csv(CACHE/"v46_article_oof.csv.gz",compression="gzip",low_memory=False,
                     parse_dates=["event_time_utc"],dtype={"ticker":str})
    base=base.rename(columns={"prob":"base_prob","high_conf":"base_high",
                              "confidence_signal":"base_confidence"})
    e5=e5[["event_id","prob"]].rename(columns={"prob":"e5_prob"})
    combined=base.merge(e5,on="event_id",how="inner",validate="one_to_one")
    if len(combined)!=len(base):raise RuntimeError("V51 base/E5 mismatch")
    selected_frame,audits=route(combined);result=v44.summarize(selected_frame)
    status_name="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V51","hypothesis":"PREQUENTIAL_SOURCE_ROUTING","dev_sha256":v37.sha256(path),
        "base_cache_sha256":v37.sha256(CACHE/"v46_article_oof.csv.gz"),
        "embedding_sha256":v37.sha256(embedding_path),"models":{"V51_SOURCE_ROUTE":result},
        "selection_audits":audits,"validation":"source routes use only strictly earlier OOF folds"}
    robustness={"version":"V51","status":status_name,"hypothesis":"PREQUENTIAL_SOURCE_ROUTING",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "selection_audits":audits,"seal_authorized":False}
    transfer={"version":"V51","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"base/E5 source route selected only from prior purged OOF; n<30 fixed base fallback"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v51_source_route_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status_name,"metrics":result["metrics"],"gate":result["research_gate"]},
                     indent=2,default=str))


if __name__=="__main__":main()
