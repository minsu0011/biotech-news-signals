"""V50 material hypothesis: frozen multilingual E5 embedding late fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V50")))
CACHE=ROOT/"cache"
COST=.002


def matrix_for(frame,kind,headline,semantic):
    index=frame.embedding_index.to_numpy(int)
    if kind=="headline":return headline[index]
    if kind=="semantic":return semantic[index]
    if kind=="both":return np.concatenate([headline[index],semantic[index]],axis=1)
    raise ValueError(kind)


def fit_embedding(train,target,kind,c,headline,semantic):
    model=LogisticRegression(C=c,solver="liblinear",max_iter=900,random_state=v36.SEED)
    model.fit(matrix_for(train,kind,headline,semantic),train.y.astype(int),
              sample_weight=train.cluster_weight)
    return model.predict_proba(matrix_for(target,kind,headline,semantic))[:,1]


BLENDS={"embedding":{"embedding":1.},"embedding25_numeric":{"embedding":.25,"numeric":.75},
    "embedding50_numeric":{"embedding":.5,"numeric":.5},
    "embedding75_numeric":{"embedding":.75,"numeric":.25},
    "embedding_action":{"embedding":.5,"payoff_action":.5},
    "numeric_action":{"numeric":.5,"payoff_action":.5}}


def blend(embedding,numeric,weights):
    pieces={"embedding":embedding,"numeric":numeric["direction"],
            "payoff_action":v38.action_probability(numeric)}
    return sum(pieces[name]*value for name,value in weights.items())/sum(weights.values())


def choose_policy(frame,embedding_models,numeric):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float)
    for key,embedding in embedding_models.items():
        for blend_name,weights in BLENDS.items():
            if blend_name=="numeric_action" and key!="headline|0.03":continue
            raw=blend(embedding,numeric,weights)
            for cutoff in np.arange(.44,.561,.02):
                margin=v38.rank_self(np.abs(raw-cutoff));opportunity=v38.rank_self(numeric["opportunity"])
                payoff=v38.rank_self(np.maximum(numeric["up_payoff"],numeric["down_payoff"]))
                confidence=.4*margin+.3*opportunity+.3*payoff
                metrics=v37.metric(frame,raw,float(cutoff),True);floor=v37.source_floor(frame,raw,float(cutoff))
                for quantile in (.75,.80,.85):
                    threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold;prediction=raw>=cutoff
                    accuracy=float(np.average(prediction[high]==y[high],weights=sample_weight[high]))
                    net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,
                                         weights=sample_weight[high]))
                    checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                                metrics["edge"]>=0,accuracy>=.55,net>0,floor>=.49))
                    score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                           .4*floor+.35*accuracy+8*max(-.01,min(.01,net)))
                    rows.append({"embedding_key":key,"blend_name":blend_name,"weights":weights,
                        "direction_cutoff":float(cutoff),"confidence_quantile":quantile,
                        "confidence_threshold":threshold,"checks":checks,"score":float(score),
                        "inner_metrics":metrics,"source_floor":floor,"inner_hc_accuracy":accuracy,
                        "inner_hc_net":net,"references":{"margin":np.abs(raw-cutoff),
                            "opportunity":numeric["opportunity"],
                            "payoff":np.maximum(numeric["up_payoff"],numeric["down_payoff"])}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,embedding,numeric):
    cutoff=float(policy["direction_cutoff"]);raw=blend(embedding,numeric,policy["weights"]);ref=policy["references"]
    confidence=(.4*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .3*v38.rank_against(ref["opportunity"],numeric["opportunity"])+
                .3*v38.rank_against(ref["payoff"],np.maximum(numeric["up_payoff"],numeric["down_payoff"])))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame,market,headline,semantic):
    selected=[];embedding_frames=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);numeric_inner=v38.fit_components(inner_train,inner_valid,.006)
        embedding_models={}
        for kind in ("headline","semantic","both"):
            for c in (.03,.10,.30):
                key=f"{kind}|{c:.2f}";embedding_models[key]=fit_embedding(
                    inner_train,inner_valid,kind,c,headline,semantic)
        policy=choose_policy(inner_valid.reset_index(drop=True),embedding_models,numeric_inner)
        kind,c_text=policy["embedding_key"].split("|");c=float(c_text)
        embedding=fit_embedding(train,valid,kind,c,headline,semantic)
        numeric=v38.fit_components(train,valid,.006);prob,high,confidence=apply_policy(policy,embedding,numeric)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V50_INNER_SELECTED",seen))
        component_high=v38.rank_against(np.abs(embedding_models[policy["embedding_key"]]-.5),
                                         np.abs(embedding-.5))>=.80
        embedding_frames.append(v38.prediction_frame(valid,embedding,component_high,np.abs(embedding-.5),
                                                       fold,f"e5_{kind}",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V50 E5] {market} fold={fold} embedding={policy['embedding_key']} "
              f"blend={policy['blend_name']} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(embedding_frames,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz"
    embedding_path=CACHE/"v50_multilingual_e5_embeddings.npz"
    embedding_status=json.loads((CACHE/"V50_EMBEDDING_STATUS.json").read_text(encoding="utf-8"))
    if v37.sha256(embedding_path)!=embedding_status["embedding_sha256"]:
        raise RuntimeError("V50 embedding cache hash mismatch")
    loaded=np.load(embedding_path,allow_pickle=True)
    ids=loaded["event_id"].astype(str);headline=loaded["headline"].astype(np.float32)
    semantic=loaded["semantic"].astype(np.float32);mapping={event_id:index for index,event_id in enumerate(ids)}
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    data["embedding_index"]=data.event_id.astype(str).map(mapping)
    if data.embedding_index.isna().any():raise RuntimeError("V50 embedding key mismatch")
    data["embedding_index"]=data.embedding_index.astype(int)
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];components=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market,headline,semantic)
        selected.append(first);components.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);component_frame=pd.concat(components,ignore_index=True)
    result=v44.summarize(selected_frame);component=v44.summarize(component_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V50","hypothesis":"FROZEN_MULTILINGUAL_EMBEDDING",
        "dev_sha256":v37.sha256(path),"embedding_sha256":v37.sha256(embedding_path),
        "embedding_status":embedding_status,"models":{"V50_INNER_SELECTED":result,"E5_component":component},
        "selection_audits":selection}
    robustness={"version":"V50","status":status,"hypothesis":"FROZEN_MULTILINGUAL_EMBEDDING",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "embedding_component":component,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V50","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"frozen outcome-free E5 embeddings; only fold-local linear heads/blend/cutoff fit labels"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v50_embedding_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
