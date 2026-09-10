"""V56: past-only semantic case retrieval routed only to US news."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V56")))
CACHE=ROOT/"cache"
EMBEDDING_PATH=CACHE/"v50_multilingual_e5_embeddings.npz"
BASE_PATH=CACHE/"v54_diverse_consensus_oof.csv.gz"
NEIGHBORS=25
TEMPERATURE=.07
PRIOR_STRENGTH=10.
RETRIEVAL_BLEND=.50


def retrieve(query:pd.DataFrame,train:pd.DataFrame,headline:np.ndarray,
             semantic:np.ndarray)->tuple[np.ndarray,dict[str,float|int]]:
    similarity=(.5*(headline[query.embedding_index.to_numpy(int)] @
                    headline[train.embedding_index.to_numpy(int)].T)+
                .5*(semantic[query.embedding_index.to_numpy(int)] @
                    semantic[train.embedding_index.to_numpy(int)].T))
    same_cluster=(query.text_cluster_id.to_numpy()[:,None]==train.text_cluster_id.to_numpy()[None,:])
    similarity[same_cluster]=-10.
    k=min(NEIGHBORS,len(train));selected=np.argpartition(similarity,-k,axis=1)[:,-k:]
    selected_similarity=np.take_along_axis(similarity,selected,axis=1)
    weight=np.exp((selected_similarity-selected_similarity.max(axis=1,keepdims=True))/TEMPERATURE)
    neighbor_y=train.y.to_numpy(int)[selected];prior=float(train.y.mean())
    probability=(np.sum(weight*neighbor_y,axis=1)+PRIOR_STRENGTH*prior)/(
        np.sum(weight,axis=1)+PRIOR_STRENGTH)
    audit={"query_n":len(query),"train_n":len(train),"neighbors":k,
        "cluster_pairs_excluded":int(same_cluster.sum()),
        "mean_top_similarity":float(selected_similarity.max(axis=1).mean()),
        "train_prior":prior}
    return probability,audit


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    loaded=np.load(EMBEDDING_PATH,allow_pickle=True);ids=loaded["event_id"].astype(str)
    headline=loaded["headline"].astype(np.float32);semantic=loaded["semantic"].astype(np.float32)
    embedding_index=pd.Series(np.arange(len(ids)),index=ids)
    data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
                     dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    data["embedding_index"]=data.event_id.map(embedding_index)
    if data.embedding_index.isna().any():raise RuntimeError("V56 embedding alignment")
    base=pd.read_csv(BASE_PATH,compression="gzip",low_memory=False)
    base_index=base.set_index("event_id")
    outputs=[];audits=[]
    for market in ("US","KR"):
        for train,valid,fold in v37.chronological_folds(data[data.market.eq(market)].copy()):
            probability=valid.event_id.map(base_index.prob).to_numpy(float)
            routed=valid.source_family.eq("US_NEWS").to_numpy();query=valid[routed]
            retrieval_used=False
            retrieval_probability=np.full(len(valid),np.nan)
            cases=train[train.source_family.eq("US_NEWS")].copy()
            audit={"market":market,"fold":fold,"outer_train_n":len(train),"valid_n":len(valid),
                   "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min(),
                   "route":"US_NEWS_ONLY","retrieval_used":False}
            if len(query) and len(cases)>=30:
                retrieved,detail=retrieve(query,cases,headline,semantic)
                probability[routed]=(RETRIEVAL_BLEND*retrieved+
                                     (1-RETRIEVAL_BLEND)*probability[routed])
                retrieval_probability[routed]=retrieved;retrieval_used=True;audit.update(detail)
                audit["retrieval_used"]=True
            output=valid[["event_id","event_group_id","event_time_utc","market","ticker",
                          "source_family","form","y","fwd_ret_30m","text_cluster_id",
                          "cluster_weight"]].copy()
            output["prob"]=probability;output["retrieval_prob"]=retrieval_probability
            output["high_conf"]=output.event_id.map(base_index.high_conf).to_numpy(bool)
            output["confidence_signal"]=output.event_id.map(base_index.confidence_signal).to_numpy(float)
            output["fold"]=fold;output["family"]="V56_CAUSAL_SEMANTIC_RETRIEVAL"
            seen=set(train.ticker.astype(str));output["new_issuer"]=~output.ticker.astype(str).isin(seen)
            outputs.append(output);audits.append(audit)
            print(f"[V56 RETRIEVAL] {market} fold={fold} query={len(query)} used={retrieval_used}",flush=True)
    selected=pd.concat(outputs,ignore_index=True);result=v44.summarize(selected)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("same-source historical cases strictly precede outer validation; exact text clusters excluded; "
                "fixed k/temperature/prior/blend; non-US-news uses unchanged V54 consensus")
    comparison={"version":"V56","hypothesis":"CAUSAL_SEMANTIC_RETRIEVAL",
        "dev_sha256":v37.sha256(DATA/"dev_contract_v36_labeled.csv.gz"),
        "embedding_sha256":v37.sha256(EMBEDDING_PATH),"base_sha256":v37.sha256(BASE_PATH),
        "configuration":{"neighbors":NEIGHBORS,"temperature":TEMPERATURE,
            "prior_strength":PRIOR_STRENGTH,"retrieval_blend":RETRIEVAL_BLEND},
        "models":{"V56_CAUSAL_SEMANTIC_RETRIEVAL":result},"fold_audits":audits,
        "validation":validation}
    robustness={"version":"V56","status":status,"hypothesis":"CAUSAL_SEMANTIC_RETRIEVAL",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "fold_audits":audits,"seal_authorized":False}
    transfer={"version":"V56","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v56_semantic_retrieval_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],
                      "gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
