"""V46 material hypothesis: within-article cross-sectional market state."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V46")))
CACHE=ROOT/"cache"


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz";micro_path=CACHE/"v44_microstructure_features.csv.gz"
    article_path=CACHE/"v46_article_features.csv.gz"
    if not article_path.exists():raise RuntimeError("run prepare_v46_article_features.py first")
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    micro=pd.read_csv(micro_path,compression="gzip",low_memory=False)
    article=pd.read_csv(article_path,compression="gzip",low_memory=False)
    feature_columns=[column for column in micro if column!="event_id"]+[column for column in article if column!="event_id"]
    data=data.merge(micro,on="event_id",how="left",validate="one_to_one")
    data=data.merge(article,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];directions=[];selection={};errors=[]
    for market in ("US","KR"):
        first,second,audits,failures=v44.market_oof(data[data.market.eq(market)].copy(),market,feature_columns)
        first["family"]="V46_INNER_SELECTED";selected.append(first);directions.append(second)
        selection[market]=audits;errors.extend(failures)
    selected_frame=pd.concat(selected,ignore_index=True);direction_frame=pd.concat(directions,ignore_index=True)
    result=v44.summarize(selected_frame);direction_result=v44.summarize(direction_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    feature_status=json.loads((CACHE/"V46_ARTICLE_FEATURE_STATUS.json").read_text(encoding="utf-8"))
    comparison={"version":"V46","hypothesis":"ARTICLE_CROSS_SECTION","dev_sha256":v37.sha256(path),
        "micro_feature_sha256":v37.sha256(micro_path),"article_feature_sha256":v37.sha256(article_path),
        "article_feature_status":feature_status,
        "models":{"V46_INNER_SELECTED":result,"article_micro_direction":direction_result},
        "selection_audits":selection,"model_errors":errors}
    robustness={"version":"V46","status":status,"hypothesis":"ARTICLE_CROSS_SECTION",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "direction_component":direction_result,"selection_audits":selection,"model_errors":errors,
        "seal_authorized":False}
    transfer={"version":"V46","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"article-ID/URL clusters omit ticker; pre-entry cross-sectional ranks only; cluster-weighted fit"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v46_article_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "model_errors":errors},indent=2,default=str))


if __name__=="__main__":main()
