"""V60: fixed US-news opportunity/agreement confidence selector."""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits

ROOT=Path(__file__).resolve().parent;CACHE=ROOT/"cache"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V60")))
FILES={"BASE":"v58_locked_source_routing_oof.csv.gz","DIRECTION":"v50_embedding_oof.csv.gz",
       "RETRIEVAL":"v56_semantic_retrieval_oof.csv.gz","DIVERSE":"v54_diverse_consensus_oof.csv.gz",
       "OPPORTUNITY":"v53_decoupled_oof.csv.gz","ECONOMIC":"v48_xgb_expert_oof.csv.gz"}
FORMULAS={"MARGIN_ONLY":{"margin":1.},
          "OPPORTUNITY_AGREEMENT":{"margin":.40,"opportunity":.35,"agreement":.25},
          "ECONOMIC_OPPORTUNITY_AGREEMENT":{"margin":.30,"opportunity":.30,"agreement":.25,"economic":.15}}
SELECTED="ECONOMIC_OPPORTUNITY_AGREEMENT"

def rank(values):
    order=np.argsort(np.argsort(np.asarray(values,float),kind="stable"),kind="stable")
    return (order+1.)/max(1,len(order))

def main()->None:
    runtime_limits.configure();OUT.mkdir(parents=True,exist_ok=True)
    frames={name:pd.read_csv(CACHE/path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
            for name,path in FILES.items()}
    base=frames["BASE"].copy();aligned={name:frame.set_index("event_id").loc[base.event_id] for name,frame in frames.items()}
    for name,frame in aligned.items():
        if not np.array_equal(frame.fold.to_numpy(),base.fold.to_numpy()):raise RuntimeError(f"V60 fold mismatch {name}")
    candidates={name:base.copy() for name in FORMULAS};audit=[]
    for fold in sorted(base.fold.unique()):
        positions=np.flatnonzero((base.fold.eq(fold)&base.source_family.eq("US_NEWS")).to_numpy())
        if not len(positions):continue
        direction=aligned["DIRECTION"].prob.to_numpy(float)[positions]
        expert=np.column_stack([direction,aligned["RETRIEVAL"].prob.to_numpy(float)[positions],
            aligned["DIVERSE"].prob.to_numpy(float)[positions],aligned["OPPORTUNITY"].prob.to_numpy(float)[positions]])
        signals={"margin":rank(np.abs(direction-.5)),
            "opportunity":rank(aligned["OPPORTUNITY"].opportunity_confidence.to_numpy(float)[positions]),
            "agreement":rank(1.-np.std(expert,axis=1)),
            "economic":rank(aligned["ECONOMIC"].confidence_signal.to_numpy(float)[positions])}
        for name,weights in FORMULAS.items():
            confidence=sum(signals[key]*weight for key,weight in weights.items())/sum(weights.values())
            threshold=float(np.quantile(confidence,.85));high=confidence>=threshold
            candidates[name].loc[positions,"prob"]=direction;candidates[name].loc[positions,"confidence_signal"]=confidence
            candidates[name].loc[positions,"high_conf"]=high;candidates[name].loc[positions,"family"]=f"V60_{name}"
            audit.append({"fold":int(fold),"candidate":name,"US_NEWS_n":len(positions),"confidence_quantile":.85,
                          "threshold":threshold,"weights":weights,"selection_uses_labels":False})
    results={name:v44.summarize(frame) for name,frame in candidates.items()};selected=candidates[SELECTED];result=results[SELECTED]
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("all components are chronological outer OOF; formula and quantile are outcome-blind; "
                "US_NEWS direction is unchanged; no outer-metric selection")
    comparison={"version":"V60","hypothesis":"US_NEWS_OPPORTUNITY_SELECTOR_V1",
        "component_sha256":{name:v37.sha256(CACHE/path) for name,path in FILES.items()},"fixed_selected":SELECTED,
        "formulas":FORMULAS,"models":results,"selection_audit":audit,"validation":validation}
    robustness={"version":"V60","status":status,"hypothesis":"US_NEWS_OPPORTUNITY_SELECTOR_V1",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,"fixed_selected":SELECTED,"seal_authorized":False}
    transfer={"version":"V60","seal_outcomes_loaded":False,"selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],"selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v60_us_news_opportunity_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))

if __name__=="__main__":main()
