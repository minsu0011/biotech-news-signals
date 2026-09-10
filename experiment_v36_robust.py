"""Nested chronological robustness study on opened, exact-contract V36 DEV data.

This module never reads a V36 seal label.  It compares a deliberately small
set of model families, selects only hyperparameters inside each outer training
fold, and reports transfer/strategy diagnostics rather than promoting a peak
split result.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

import bio_news_30m_v3 as app
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=ROOT/"output_V36"
SEED=20260827
COST=0.002
HC_QUANTILES=(0.75,0.80,0.85)
CAT_COLUMNS=("source_family","event_type","form")
FAMILIES={
    "logit":[
        {"C":0.03,"class_weight":None},{"C":0.10,"class_weight":None},
        {"C":0.30,"class_weight":"balanced"},
    ],
    "lightgbm":[
        {"n_estimators":250,"num_leaves":7,"learning_rate":0.035,"min_child_samples":40},
        {"n_estimators":350,"num_leaves":15,"learning_rate":0.025,"min_child_samples":50},
        {"n_estimators":250,"num_leaves":31,"learning_rate":0.025,"min_child_samples":70},
    ],
    "catboost_gpu":[
        {"iterations":300,"depth":4,"learning_rate":0.035,"l2_leaf_reg":8.0},
        {"iterations":400,"depth":5,"learning_rate":0.025,"l2_leaf_reg":12.0},
        {"iterations":350,"depth":6,"learning_rate":0.025,"l2_leaf_reg":18.0},
    ],
}


def atomic_json(payload:Any,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def feature_frame(frame:pd.DataFrame)->pd.DataFrame:
    output=app.model_frame(frame).reset_index(drop=True)
    for column in CAT_COLUMNS:
        output[column]=frame[column].fillna("").astype(str).reset_index(drop=True)
    return output


def make_model(family:str,parameters:dict[str,Any])->Pipeline:
    numeric=list(app.NUMERIC_COLS)
    if family=="logit":
        numeric_pipe=Pipeline([
            ("impute",SimpleImputer(strategy="median",add_indicator=True)),
            ("scale",StandardScaler()),
        ])
        model=LogisticRegression(
            C=float(parameters["C"]),class_weight=parameters["class_weight"],
            solver="liblinear",max_iter=1200,random_state=SEED,
        )
    else:
        numeric_pipe=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True))])
        if family=="lightgbm":
            model=LGBMClassifier(
                **parameters,objective="binary",random_state=SEED,n_jobs=runtime_limits.THREAD_COUNT,
                subsample=0.85,colsample_bytree=0.85,reg_lambda=3.0,verbosity=-1,
                deterministic=True,force_col_wise=True,
            )
        elif family=="catboost_gpu":
            model=CatBoostClassifier(
                **parameters,loss_function="Logloss",eval_metric="AUC",random_seed=SEED,
                task_type="GPU",devices="0",verbose=False,allow_writing_files=False,
                random_strength=0.5,
            )
        else:
            raise ValueError(family)
    categories=Pipeline([
        ("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=3,sparse_output=False)),
    ])
    processor=ColumnTransformer([
        ("numeric",numeric_pipe,numeric),("category",categories,list(CAT_COLUMNS)),
    ],remainder="drop",sparse_threshold=0.0)
    return Pipeline([("features",processor),("model",model)])


def safe_auc(y:np.ndarray,p:np.ndarray)->float:
    return float(roc_auc_score(y,p)) if np.unique(y).size==2 else float("nan")


def direction_metrics(frame:pd.DataFrame,p:np.ndarray)->dict[str,float|int]:
    y=frame.y.astype(int).to_numpy();prediction=p>=0.5
    accuracy=float(accuracy_score(y,prediction));base=float(y.mean())
    return {
        "n":int(len(frame)),"accuracy":accuracy,
        "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
        "auc":safe_auc(y,p),"edge_vs_naive":accuracy-max(base,1.0-base),
        "pred_up":float(prediction.mean()),
    }


def inner_score(frame:pd.DataFrame,p:np.ndarray)->float:
    metrics=direction_metrics(frame,p);source_scores=[]
    for _,indices in frame.groupby("source_family").groups.items():
        positions=frame.index.get_indexer(indices)
        group=frame.loc[indices]
        if len(group)>=30 and group.y.nunique()==2:
            source_scores.append(float(balanced_accuracy_score(group.y,p[positions]>=0.5)))
    floor=min(source_scores) if source_scores else 0.5
    return float(metrics["balanced_accuracy"]+metrics["auc"]+0.25*floor)


def confidence_rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.sort(np.asarray(reference,float))
    return np.searchsorted(reference,np.asarray(values,float),side="right")/max(1,len(reference))


def choose_inner(train:pd.DataFrame,family:str)->tuple[dict[str,Any],float,np.ndarray,dict[str,Any]]:
    ordered=train.sort_values("event_time_utc").reset_index(drop=True)
    split=max(100,int(0.78*len(ordered)))
    boundary=ordered.iloc[split].event_time_utc
    inner_train=ordered[ordered.event_time_utc<boundary-pd.Timedelta(minutes=35)].copy()
    inner_valid=ordered.iloc[split:].copy()
    if inner_train.y.nunique()<2 or inner_valid.y.nunique()<2:
        raise RuntimeError(f"inner chronological classes insufficient: {family}")
    x_train=feature_frame(inner_train);x_valid=feature_frame(inner_valid)
    candidates=[]
    for parameters in FAMILIES[family]:
        model=make_model(family,parameters);model.fit(x_train,inner_train.y.astype(int))
        probability=model.predict_proba(x_valid)[:,1]
        base_score=inner_score(inner_valid.reset_index(drop=True),probability)
        signed=np.where(probability>=0.5,1.0,-1.0)*inner_valid.fwd_ret_30m.to_numpy(float)-COST
        confidence=np.abs(probability-0.5)
        for quantile in HC_QUANTILES:
            threshold=float(np.quantile(confidence,quantile));selected=confidence>=threshold
            hc_accuracy=float(
                ((probability[selected]>=0.5)==inner_valid.y.to_numpy()[selected]).mean()
            )
            hc_net=float(signed[selected].mean())
            score=base_score+0.20*hc_accuracy+5.0*max(-0.01,min(0.01,hc_net))
            candidates.append({"parameters":parameters,"quantile":quantile,"score":score,
                               "metrics":direction_metrics(inner_valid,probability),
                               "highconf_accuracy":hc_accuracy,"highconf_net":hc_net,
                               "confidence_reference":confidence})
    candidates.sort(key=lambda row:row["score"],reverse=True);best=candidates[0]
    detail={key:value for key,value in best.items() if key!="confidence_reference"}
    return best["parameters"],float(best["quantile"]),best["confidence_reference"],detail


def chronological_folds(frame:pd.DataFrame)->list[tuple[pd.DataFrame,pd.DataFrame,int]]:
    ordered=frame.sort_values("event_time_utc").reset_index(drop=True)
    blocks=np.array_split(np.arange(len(ordered)),5);folds=[]
    for fold in range(1,5):
        valid=ordered.iloc[blocks[fold]].copy();boundary=valid.event_time_utc.min()
        train=ordered[ordered.event_time_utc<boundary-pd.Timedelta(minutes=35)].copy()
        if len(train)>=200 and len(valid)>=100 and train.y.nunique()==2 and valid.y.nunique()==2:
            folds.append((train,valid,fold))
    return folds


def family_oof(frame:pd.DataFrame,market:str,family:str)->tuple[pd.DataFrame,dict[str,Any]]:
    pieces=[];selections=[]
    for train,valid,fold in chronological_folds(frame):
        parameters,quantile,reference,inner=choose_inner(train,family)
        model=make_model(family,parameters);model.fit(feature_frame(train),train.y.astype(int))
        raw=model.predict_proba(feature_frame(valid))[:,1]
        rank=confidence_rank(reference,np.abs(raw-0.5));high=rank>=quantile
        output=valid[["event_id","event_group_id","event_time_utc","market","ticker",
                      "source_family","form","y","fwd_ret_30m"]].copy()
        output["prob"]=raw;output["high_conf"]=high;output["confidence_rank"]=rank
        output["fold"]=fold;output["family"]=family
        seen=set(train.ticker.astype(str));output["new_issuer"]=~output.ticker.astype(str).isin(seen)
        pieces.append(output);selections.append({"fold":fold,"train_n":len(train),"valid_n":len(valid),
            "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min(),
            "parameters":parameters,"highconf_quantile":quantile,"inner_selection":inner})
        print(f"[V36 ROBUST] {market} {family} fold={fold} train={len(train)} valid={len(valid)}",flush=True)
    return pd.concat(pieces,ignore_index=True),{"market":market,"family":family,"folds":selections}


def subgroup_metrics(frame:pd.DataFrame,column:str,minimum:int=30)->dict[str,Any]:
    result={}
    for name,group in frame.groupby(column,dropna=False):
        if len(group)<minimum or group.y.nunique()<2:continue
        result[str(name)]=summary(group,include_subgroups=False)
    return result


def summary(frame:pd.DataFrame,include_subgroups:bool=True)->dict[str,Any]:
    probability=frame.prob.to_numpy(float);y=frame.y.to_numpy(int);prediction=probability>=0.5
    high=frame.high_conf.to_numpy(bool);base=float(y.mean());accuracy=float((prediction==y).mean())
    signed=np.where(prediction,1.0,-1.0)*frame.fwd_ret_30m.to_numpy(float)-COST
    result={
        "n":len(frame),"accuracy":accuracy,"balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
        "auc":safe_auc(y,probability),"edge_vs_naive":accuracy-max(base,1.0-base),
        "highconf_n":int(high.sum()),"highconf_coverage":float(high.mean()),
        "highconf_accuracy":float((prediction[high]==y[high]).mean()) if high.any() else float("nan"),
        "strategy_mean_signed_net":float(signed[high].mean()) if high.any() else float("nan"),
        "all_trade_mean_signed_net":float(signed.mean()),"pred_up":float(prediction.mean()),
    }
    if include_subgroups:
        result["by_market"]=subgroup_metrics(frame,"market",100)
        result["by_source_family"]=subgroup_metrics(frame,"source_family",30)
        result["by_fold"]=subgroup_metrics(frame,"fold",50)
        result["new_issuer"]={
            "n":int(frame.new_issuer.sum()),
            "metrics":summary(frame[frame.new_issuer],False) if frame.new_issuer.sum()>=30 else None,
        }
    return result


def bootstrap(frame:pd.DataFrame,nboot:int=1000)->dict[str,Any]:
    x=frame.copy().reset_index(drop=True)
    x["day"]=pd.to_datetime(x.event_time_utc,utc=True).dt.date
    groups={day:np.asarray(index,dtype=int) for day,index in x.groupby("day").indices.items()}
    days=np.asarray(sorted(groups),dtype=object);rng=np.random.default_rng(SEED+99)
    balanced=[];nets=[]
    for _ in range(nboot):
        positions=np.concatenate([groups[day] for day in rng.choice(days,len(days),replace=True)])
        sample=x.iloc[positions];y=sample.y.to_numpy(int);prediction=sample.prob.to_numpy()>=0.5
        if np.unique(y).size<2:continue
        balanced.append(float(balanced_accuracy_score(y,prediction)))
        selected=sample.high_conf.to_numpy(bool)
        if selected.any():
            signed=np.where(prediction[selected],1.0,-1.0)*sample.fwd_ret_30m.to_numpy()[selected]-COST
            nets.append(float(signed.mean()))
    return {"n_boot":len(balanced),
        "balanced_accuracy_lower95":float(np.quantile(balanced,.025)),
        "balanced_accuracy_upper95":float(np.quantile(balanced,.975)),
        "highconf_strategy_net_lower95":float(np.quantile(nets,.025)),
        "highconf_strategy_net_upper95":float(np.quantile(nets,.975))}


def strategy_robustness(frame:pd.DataFrame)->dict[str,Any]:
    selected=frame[frame.high_conf].copy();prediction=selected.prob.to_numpy()>=0.5
    signed=np.where(prediction,1.0,-1.0)*selected.fwd_ret_30m.to_numpy(float)-COST
    low,high=np.quantile(signed,[.01,.99]);order=np.argsort(np.abs(signed))[::-1]
    def removed(count:int)->float:
        keep=np.ones(len(signed),bool);keep[order[:min(count,len(signed))]]=False
        return float(signed[keep].mean()) if keep.any() else float("nan")
    return {"n":len(signed),"mean_net":float(signed.mean()),"median_net":float(np.median(signed)),
        "winsorized_net":float(np.clip(signed,low,high).mean()),"top_1_removed_net":removed(1),
        "top_5_removed_net":removed(5),"bootstrap":bootstrap(frame),
        "market_highconf_net":{str(k):summary(v,False)["strategy_mean_signed_net"]
                               for k,v in frame.groupby("market")},
        "source_highconf_net":{str(k):summary(v,False)["strategy_mean_signed_net"]
                               for k,v in frame.groupby("source_family")},
    }


def gate(metrics:dict[str,Any],robust:dict[str,Any])->dict[str,Any]:
    checks={
        "balanced_accuracy_ge_0_540":metrics["balanced_accuracy"]>=.540,
        "auc_ge_0_560":metrics["auc"]>=.560,
        "edge_ge_0_020":metrics["edge_vs_naive"]>=.020,
        "highconf_coverage_ge_0_15":metrics["highconf_coverage"]>=.15,
        "highconf_accuracy_ge_0_600":metrics["highconf_accuracy"]>=.600,
        "highconf_net_gt_0":metrics["strategy_mean_signed_net"]>0,
        "bootstrap_ba_lower_ge_0_500":robust["bootstrap"]["balanced_accuracy_lower95"]>=.500,
        "bootstrap_highconf_net_lower_gt_0":robust["bootstrap"]["highconf_strategy_net_lower95"]>0,
        "US_ba_ge_0_510":metrics["by_market"].get("US",{}).get("balanced_accuracy",0)>=.510,
        "KR_ba_ge_0_510":metrics["by_market"].get("KR",{}).get("balanced_accuracy",0)>=.510,
        "worst_fold_ba_ge_0_490":min(v["balanced_accuracy"] for v in metrics["by_fold"].values())>=.490,
        "no_major_source_ba_below_0_490":min(v["balanced_accuracy"] for v in metrics["by_source_family"].values())>=.490,
        "winsorized_net_gt_0":robust["winsorized_net"]>0,
        "top_5_removed_net_gt_0":robust["top_5_removed_net"]>0,
    }
    return {"passed":sum(checks.values()),"total":len(checks),"checks":checks,
            "robust_survivor":all(checks.values())}


def source_transfer(data:pd.DataFrame,selected:dict[str,str])->dict[str,Any]:
    report={"selection_uses_v36_seal_labels":False,"tests":{}}
    for market in ("US","KR"):
        part=data[data.market.eq(market)].sort_values("event_time_utc").reset_index(drop=True)
        sources=list(part.source_family.value_counts().index);family=selected[market]
        parameters=FAMILIES[family][0]
        for train_source in sources:
            for target_source in sources:
                if train_source==target_source:continue
                train=part[part.source_family.eq(train_source)];target=part[part.source_family.eq(target_source)]
                key=f"{market}:{train_source}->{target_source}"
                if len(train)<200 or len(target)<40 or train.y.nunique()<2 or target.y.nunique()<2:
                    report["tests"][key]={"status":"INSUFFICIENT","train_n":len(train),"target_n":len(target)};continue
                model=make_model(family,parameters);model.fit(feature_frame(train),train.y.astype(int))
                probability=model.predict_proba(feature_frame(target))[:,1]
                report["tests"][key]={"status":"EVALUATED","train_n":len(train),"target_n":len(target),
                                      "metrics":direction_metrics(target,probability)}
    return report


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},
                     parse_dates=["event_time_utc"])
    if data.event_group_id.duplicated().any():raise RuntimeError("event group leakage in DEV")
    if set(data.contract_version)!={"EXACT_T2_30M_V36"}:raise RuntimeError("mixed price contract")
    if not ((data.entry_slippage_sec.between(0,60))&(data.exit_slippage_sec.between(0,60))&
            (data.actual_hold_seconds.between(1800,1860))).all():
        raise RuntimeError("timing contract violation in DEV")
    market_results={};selection_audits={}
    for market in ("US","KR"):
        part=data[data.market.eq(market)].copy()
        market_results[market]={};selection_audits[market]={}
        for family in FAMILIES:
            oof,audit=family_oof(part,market,family)
            market_results[market][family]=oof;selection_audits[market][family]=audit
    candidates=[]
    for us_family,us_frame in market_results["US"].items():
        for kr_family,kr_frame in market_results["KR"].items():
            combined=pd.concat([us_frame,kr_frame],ignore_index=True)
            metrics=summary(combined);score=(metrics["balanced_accuracy"]+metrics["auc"]+
                metrics["edge_vs_naive"]+metrics["highconf_accuracy"]+
                20.0*metrics["strategy_mean_signed_net"]+
                min(v["balanced_accuracy"] for v in metrics["by_market"].values()))
            candidates.append({"US":us_family,"KR":kr_family,"score":float(score),
                               "metrics":metrics,"frame":combined})
    candidates.sort(key=lambda row:row["score"],reverse=True)
    finalists=[]
    for candidate in candidates[:3]:
        robust=strategy_robustness(candidate["frame"]);research_gate=gate(candidate["metrics"],robust)
        finalists.append({"market_models":{"US":candidate["US"],"KR":candidate["KR"]},
            "selection_score":candidate["score"],"metrics":candidate["metrics"],
            "robustness":robust,"research_gate":research_gate})
    finalists.sort(key=lambda row:(row["research_gate"]["robust_survivor"],
                                   row["research_gate"]["passed"],row["selection_score"]),reverse=True)
    best=finalists[0]
    report={
        "status":"ROBUST_SURVIVOR" if best["research_gate"]["robust_survivor"] else "NO_ROBUST_SURVIVOR",
        "v36_seal_outcomes_loaded":False,"opened_dev_only":True,
        "contract_version":"EXACT_T2_30M_V36","dev_sha256":sha256(path),
        "validation":"5 chronological blocks; 4 expanding outer folds; 35m embargo; inner-only hyperparameter/HC selection",
        "model_families":FAMILIES,"runtime":runtime_limits.status(),
        "selection_audits":selection_audits,"candidate_summaries":[
            {key:value for key,value in row.items() if key!="frame"} for row in candidates],
        "finalists":finalists,"selected":best,
        "seal_authorized":False,
    }
    atomic_json(report,OUT/"DEV_ROBUSTNESS_REPORT.json")
    transfer=source_transfer(data,best["market_models"])
    transfer.update({"selected_market_models":best["market_models"],
        "oof_by_source":best["metrics"]["by_source_family"],
        "new_issuer_oof":best["metrics"]["new_issuer"]})
    atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    print(json.dumps({"status":report["status"],"selected":best["market_models"],
                      "gate":best["research_gate"],"metrics":best["metrics"]},indent=2,default=str))


if __name__=="__main__":main()
