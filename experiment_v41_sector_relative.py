"""V41 material hypothesis: point-in-time sector-relative pre-event state."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V41")))
CACHE=ROOT/"cache"
PARAMETERS={"n_estimators":350,"num_leaves":15,"learning_rate":.025,"min_child_samples":50}
SECTOR_COLUMNS=[f"{prefix}_{window}" for window in (2,5,15,30,60) for prefix in (
    "xbi_ret","stock_minus_xbi","xbi_minus_market","ibb_ret","stock_minus_ibb",
    "ibb_minus_market","xlv_ret","stock_minus_xlv","xlv_minus_market")]


def feature_frame(frame:pd.DataFrame)->pd.DataFrame:
    output=v36.feature_frame(frame)
    for column in SECTOR_COLUMNS:output[column]=pd.to_numeric(frame[column],errors="coerce").to_numpy()
    return output


def make_model()->Pipeline:
    numeric=list(app.NUMERIC_COLS)+SECTOR_COLUMNS
    numeric_pipe=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True))])
    categories=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=3,sparse_output=False))])
    processor=ColumnTransformer([("numeric",numeric_pipe,numeric),
        ("category",categories,list(v36.CAT_COLUMNS))],remainder="drop",sparse_threshold=0.)
    model=LGBMClassifier(**PARAMETERS,objective="binary",random_state=v36.SEED,
        n_jobs=runtime_limits.THREAD_COUNT,subsample=.85,colsample_bytree=.85,reg_lambda=3.,
        verbosity=-1,deterministic=True,force_col_wise=True)
    return Pipeline([("features",processor),("model",model)])


def target_fit(train:pd.DataFrame,target:pd.DataFrame,label:np.ndarray)->np.ndarray:
    model=make_model();model.fit(feature_frame(train),np.asarray(label,int),
                                 model__sample_weight=train.cluster_weight)
    return model.predict_proba(feature_frame(target))[:,1]


def fit_by_threshold(train:pd.DataFrame,target:pd.DataFrame,
                     thresholds:tuple[float,...])->dict[float,dict[str,np.ndarray]]:
    direction=target_fit(train,target,train.y);up=target_fit(train,target,train.fwd_ret_30m>COST)
    down=target_fit(train,target,train.fwd_ret_30m<-COST);output={}
    for threshold in thresholds:
        opportunity=target_fit(train,target,train.fwd_ret_30m.abs()>=threshold)
        output[threshold]={"direction":direction,"up_payoff":up,"down_payoff":down,
                           "opportunity":opportunity}
    return output


COST=.002


def market_oof(frame:pd.DataFrame,market:str):
    if market=="KR":
        selected,direction,audits=v38.market_oof(frame,market)
        selected["family"]="V41_KR_V38_FALLBACK"
        return selected,direction,audits
    selected=[];direction_frames=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);thresholds=(.004,.006,.010)
        inner=fit_by_threshold(inner_train,inner_valid,thresholds)
        policy=v38.choose_policy(inner_valid.reset_index(drop=True),inner)
        threshold=float(policy["magnitude_threshold"])
        outer=fit_by_threshold(train,valid,(threshold,))[threshold]
        probability,high,confidence=v38.apply_policy(policy,outer);seen=set(train.ticker.astype(str))
        selected.append(v38.prediction_frame(valid,probability,high,confidence,fold,
                                              "V41_INNER_SELECTED",seen))
        raw=outer["direction"];direction_high=v38.rank_against(
            np.abs(inner[threshold]["direction"]-.5),np.abs(raw-.5))>=.80
        direction_frames.append(v38.prediction_frame(valid,raw,direction_high,np.abs(raw-.5),fold,
                                                       "sector_direction",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),
                      "inner_train_n":len(inner_train),"inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V41 SECTOR] {market} fold={fold} mode={policy['mode']} "
              f"magnitude={threshold:.3f} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(direction_frames,ignore_index=True),audits


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz";feature_path=CACHE/"v41_sector_features.csv.gz"
    if not feature_path.exists():raise RuntimeError("run prepare_v41_sector_features.py first")
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    features=pd.read_csv(feature_path,compression="gzip",low_memory=False)
    data=data.merge(features,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];directions=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected.append(first);directions.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);direction_frame=pd.concat(directions,ignore_index=True)
    result=summarize(selected_frame);direction_result=summarize(direction_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    feature_status=json.loads((CACHE/"V41_SECTOR_FEATURE_STATUS.json").read_text(encoding="utf-8"))
    comparison={"version":"V41","hypothesis":"SECTOR_RELATIVE_STATE","dev_sha256":v37.sha256(path),
        "sector_feature_sha256":v37.sha256(feature_path),"sector_feature_status":feature_status,
        "models":{"V41_INNER_SELECTED":result,"sector_direction":direction_result},
        "selection_audits":selection}
    robustness={"version":"V41","status":status,"hypothesis":"SECTOR_RELATIVE_STATE",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "direction_component":direction_result,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V41","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"US XBI/IBB/XLV completed 1-minute bars strictly before actual entry; KR V38 fallback"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v41_sector_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
