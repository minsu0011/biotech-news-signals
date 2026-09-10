"""V61: fold-internal shrunk issuer/event/source response priors for KR news."""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits

ROOT=Path(__file__).resolve().parent;DATA=ROOT/"data";CACHE=ROOT/"cache"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V61")))
BASE=CACHE/"v58_locked_source_routing_oof.csv.gz"
SHRINK={"ticker":20.,"event_type":50.,"source":75.}

def mapping(train:pd.DataFrame,column:str,shrink:float,global_mean:float)->dict[str,float]:
    stats=train.groupby(column,dropna=False).y.agg(["sum","count"])
    return ((stats["sum"]+shrink*global_mean)/(stats["count"]+shrink)).to_dict()

def prior(train:pd.DataFrame,target:pd.DataFrame)->np.ndarray:
    global_mean=float(train.y.mean());parts=[]
    for column,shrink in SHRINK.items():
        values=mapping(train,column,shrink,global_mean)
        parts.append(target[column].map(values).fillna(global_mean).to_numpy(float))
    logits=np.column_stack([logit(np.clip(value,1e-5,1-1e-5)) for value in parts])
    return expit(logits@np.asarray([.5,.3,.2]))

def main()->None:
    runtime_limits.configure();OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
                     dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    base=pd.read_csv(BASE,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    selected=base.copy();component=base.copy();audits=[];base_map=base.set_index("event_id")
    for train,valid,fold in v37.chronological_folds(data[data.market.eq("KR")].copy()):
        target=valid[valid.source_family.eq("KR_NEWS")].copy()
        if target.empty:continue
        p_prior=prior(train,target);p_base=base_map.loc[target.event_id,"prob"].to_numpy(float)
        combined=expit(.8*logit(np.clip(p_base,1e-5,1-1e-5))+.2*logit(np.clip(p_prior,1e-5,1-1e-5)))
        positions=np.flatnonzero(selected.event_id.isin(set(target.event_id)).to_numpy());lookup=dict(zip(target.event_id,combined))
        selected.loc[positions,"prob"]=selected.iloc[positions].event_id.map(lookup).to_numpy(float)
        selected.loc[positions,"family"]="V61_KR_NEWS_CAUSAL_ISSUER_PRIOR"
        plookup=dict(zip(target.event_id,p_prior));component.loc[positions,"prob"]=component.iloc[positions].event_id.map(plookup).to_numpy(float)
        component.loc[positions,"family"]="V61_PRIOR_ONLY"
        audits.append({"fold":fold,"train_n":len(train),"valid_KR_NEWS_n":len(target),
            "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min(),
            "shrinkage":SHRINK,"blend":{"base":.8,"prior":.2},"current_fold_labels_used":False})
    result=v44.summarize(selected);prior_result=v44.summarize(component)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("ticker/event/source target priors are fitted only on each outer training fold with fixed shrinkage; "
                "current validation/seal labels are never read; V58 probabilities and confidence remain fixed")
    comparison={"version":"V61","hypothesis":"KR_NEWS_CAUSAL_ISSUER_PRIOR_V1","base_sha256":v37.sha256(BASE),
        "fixed_shrinkage":SHRINK,"fixed_blend":{"base":.8,"prior":.2},"models":{"V61_BLEND":result,"PRIOR_ONLY":prior_result},
        "fold_audits":audits,"validation":validation}
    robustness={"version":"V61","status":status,"hypothesis":"KR_NEWS_CAUSAL_ISSUER_PRIOR_V1",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,"seal_authorized":False}
    transfer={"version":"V61","seal_outcomes_loaded":False,"selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],"selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v61_kr_issuer_prior_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))

if __name__=="__main__":main()
