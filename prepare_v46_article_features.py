"""Build label-blind within-article cross-sectional pre-entry features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_v37_text as v37
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
CACHE=ROOT/"cache"
BASE_COLUMNS=("pre_ret_1","pre_ret_2","pre_ret_3","pre_ret_5","pre_ret_10","pre_ret_15",
    "pre_ret_30","pre_ret_60","pre_vol_5","pre_vol_10","pre_vol_30","pre_vol_60",
    "volume_ratio_1_30","volume_ratio_2_30","volume_ratio_5_30","volume_ratio_10_60",
    "range_5m","range_10m","range_30m","range_60m","entry_bar_ret","intraday_ret_open",
    "vwap_distance_30","benchmark_ret_5","benchmark_ret_30")


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->None:
    runtime_limits.configure();data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",
        low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    output=data[["event_id"]].copy();groups=data.groupby("text_cluster_id",sort=False)
    size=groups.event_id.transform("size").astype(float);output["article_cluster_size"]=size
    for column in BASE_COLUMNS:
        values=pd.to_numeric(data[column],errors="coerce");median=values.groupby(data.text_cluster_id).transform("median")
        mean=values.groupby(data.text_cluster_id).transform("mean");std=values.groupby(data.text_cluster_id).transform("std")
        rank=values.groupby(data.text_cluster_id).rank(method="average",pct=True)
        output[f"article_diff_{column}"]=(values-median).where(size>1,0.)
        output[f"article_z_{column}"]=((values-mean)/std.replace(0,np.nan)).where(size>1,0.)
        output[f"article_rank_{column}"]=rank.where(size>1,.5)
    path=CACHE/"v46_article_features.csv.gz";output.to_csv(path,index=False,compression="gzip")
    report={"rows":len(output),"features":len(output.columns)-1,"future_values_used":False,
        "outcome_columns_used":False,"multirow_events":int((size>1).sum()),
        "clusters":int(data.text_cluster_id.nunique()),"feature_sha256":sha256(path),
        "rule":"URL/article ID without ticker; pre-entry features only; singleton diff/z=0 rank=.5"}
    (CACHE/"V46_ARTICLE_FEATURE_STATUS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
