"""V53: decouple fixed numeric direction from V46 opportunity confidence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V53")))
CACHE=ROOT/"cache"


def numeric_oof(frame,market):
    pieces=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        model=v36.make_model("lightgbm",v36.FAMILIES["lightgbm"][0])
        model.fit(v36.feature_frame(train),train.y.astype(int),model__sample_weight=train.cluster_weight)
        probability=model.predict_proba(v36.feature_frame(valid))[:,1]
        seen=set(train.ticker.astype(str));dummy=probability>=2
        pieces.append(v38.prediction_frame(valid,probability,dummy,abs(probability-.5),fold,
                                            "V53_FIXED_NUMERIC_DIRECTION",seen))
        audits.append({"market":market,"fold":fold,"train_n":len(train),"valid_n":len(valid),
                       "model":"fixed V36 LightGBM candidate 0","direction_cutoff":.5,
                       "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
    return pd.concat(pieces,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    direction=[];audits=[]
    for market in ("US","KR"):
        frame,market_audits=numeric_oof(data[data.market.eq(market)].copy(),market)
        direction.append(frame);audits.extend(market_audits)
    direction=pd.concat(direction,ignore_index=True)
    opportunity=pd.read_csv(CACHE/"v46_article_oof.csv.gz",compression="gzip",low_memory=False,
                             usecols=["event_id","high_conf","confidence_signal"])
    opportunity=opportunity.rename(columns={"high_conf":"opportunity_high",
                                             "confidence_signal":"opportunity_confidence"})
    selected=direction.merge(opportunity,on="event_id",how="inner",validate="one_to_one")
    if len(selected)!=len(direction):raise RuntimeError("V53 direction/opportunity mismatch")
    selected["high_conf"]=selected.opportunity_high.astype(bool)
    selected["confidence_signal"]=selected.opportunity_confidence.astype(float)
    selected["family"]="V53_DECOUPLED_POLICY"
    result=v44.summarize(selected)
    # The direction component intentionally has no trading-selection mask.  Keep
    # its diagnostic directional metrics separate instead of passing an empty
    # high-confidence slice into the strategy robustness routine.
    direction_result={"metrics":v36.summary(direction.assign(high_conf=False))}
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V53","hypothesis":"DECOUPLED_DIRECTION_OPPORTUNITY",
        "dev_sha256":v37.sha256(path),"opportunity_cache_sha256":v37.sha256(CACHE/"v46_article_oof.csv.gz"),
        "models":{"V53_DECOUPLED":result,"fixed_numeric_direction":direction_result},
        "fold_audits":audits}
    robustness={"version":"V53","status":status,"hypothesis":"DECOUPLED_DIRECTION_OPPORTUNITY",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "direction_component":direction_result,"fold_audits":audits,"seal_authorized":False}
    transfer={"version":"V53","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"direction fixed before validation; high-confidence mask independently selected inside V46 folds"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v53_decoupled_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
