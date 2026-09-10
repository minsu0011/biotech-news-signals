"""V42 material hypothesis: prequential robust stacking of V37-V41 OOF.

For each market/fold, meta rules use only earlier OOF folds.  The first fold
uses a label-blind fixed median seed.  No current-fold outcome selects a rule.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import runtime_limits


ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V42")))
CACHE=ROOT/"cache"
COST=.002
BASE_FILES={
    "text":"v37_text_oof.csv.gz","opportunity":"v38_opportunity_oof.csv.gz",
    "sec":"v39_sec_semantics_oof.csv.gz","prior":"v40_causal_prior_oof.csv.gz",
    "sector":"v41_sector_oof.csv.gz",
}
BASE_COLUMNS=[f"p_{name}" for name in BASE_FILES]


def load_frame()->pd.DataFrame:
    frames={name:pd.read_csv(CACHE/file,compression="gzip",low_memory=False,
                             parse_dates=["event_time_utc"],dtype={"ticker":str})
            for name,file in BASE_FILES.items()}
    authority=frames["sector"].copy()
    if authority.event_id.duplicated().any():raise RuntimeError("V42 duplicate authority event_id")
    for name,frame in frames.items():
        if set(frame.event_id)!=set(authority.event_id):raise RuntimeError(f"V42 event mismatch: {name}")
        probability=frame.set_index("event_id").prob
        authority[f"p_{name}"]=authority.event_id.map(probability)
        high=frame.set_index("event_id").high_conf.astype(bool)
        authority[f"h_{name}"]=authority.event_id.map(high)
    if authority[BASE_COLUMNS].isna().any().any():raise RuntimeError("V42 missing base probability")
    return authority


def logit_features(frame:pd.DataFrame)->np.ndarray:
    probability=np.clip(frame[BASE_COLUMNS].to_numpy(float),1e-5,1-1e-5)
    logits=np.log(probability/(1-probability))
    return np.column_stack([logits,logits.mean(axis=1),logits.std(axis=1)])


def fit_stack(frame:pd.DataFrame)->LogisticRegression:
    model=LogisticRegression(C=.1,solver="liblinear",max_iter=700,random_state=v36.SEED)
    model.fit(logit_features(frame),frame.y.astype(int),sample_weight=frame.cluster_weight)
    return model


def candidates(frame:pd.DataFrame,stack:LogisticRegression|None)->dict[str,np.ndarray]:
    values=frame[BASE_COLUMNS].to_numpy(float);output={
        "mean_all":values.mean(axis=1),"median_all":np.median(values,axis=1),
        "opportunity_sector":.5*frame.p_opportunity.to_numpy()+.5*frame.p_sector.to_numpy(),
        "robust_three":.4*frame.p_opportunity.to_numpy()+.4*frame.p_sector.to_numpy()+.2*frame.p_sec.to_numpy(),
    }
    for column in BASE_COLUMNS:output[column.removeprefix("p_")]=frame[column].to_numpy(float)
    if stack is not None:output["stack_logit"]=stack.predict_proba(logit_features(frame))[:,1]
    return output


def agreement(frame:pd.DataFrame)->np.ndarray:
    return np.clip(1.-2.*frame[BASE_COLUMNS].std(axis=1).to_numpy(float),0,1)


def choose_policy(frame:pd.DataFrame,probabilities:dict[str,np.ndarray])->dict[str,Any]:
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float)
    sample_weight=frame.cluster_weight.to_numpy(float);base_agreement=agreement(frame)
    for name,raw in probabilities.items():
        for cutoff in np.arange(.44,.561,.02):
            margin=v38.rank_self(np.abs(raw-cutoff));agree=v38.rank_self(base_agreement)
            confidence=.7*margin+.3*agree;metrics=v37.metric(frame,raw,float(cutoff),True)
            floor=v37.source_floor(frame,raw,float(cutoff))
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold
                prediction=raw>=cutoff;weight=sample_weight[high]
                accuracy=float(np.average(prediction[high]==y[high],weights=weight))
                net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,weights=weight))
                checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                            metrics["edge"]>=0,accuracy>=.55,net>0,floor>=.49))
                score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                       .4*floor+.35*accuracy+8*max(-.01,min(.01,net)))
                rows.append({"candidate":name,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,
                    "checks":checks,"score":float(score),"inner_metrics":metrics,"source_floor":floor,
                    "inner_hc_accuracy":accuracy,"inner_hc_net":net,
                    "references":{"margin":np.abs(raw-cutoff),"agreement":base_agreement}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(frame:pd.DataFrame,probabilities:dict[str,np.ndarray],policy:dict[str,Any]):
    raw=probabilities[policy["candidate"]];cutoff=float(policy["direction_cutoff"]);ref=policy["references"]
    confidence=(.7*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .3*v38.rank_against(ref["agreement"],agreement(frame)))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def output_frame(valid,probability,high,confidence,name):
    output=valid[["event_id","event_group_id","event_time_utc","market","ticker","source_family",
                  "form","y","fwd_ret_30m","text_cluster_id","cluster_weight","fold","new_issuer"]].copy()
    output["prob"]=probability;output["high_conf"]=high;output["confidence_signal"]=confidence
    output["family"]=name;return output


def prequential_market(frame:pd.DataFrame,market:str):
    outputs=[];audits=[]
    for fold in sorted(frame.fold.unique()):
        valid=frame[frame.fold.eq(fold)].copy();past=frame[frame.fold.lt(fold)].copy()
        past=past[~past.text_cluster_id.isin(set(valid.text_cluster_id))].sort_values("event_time_utc")
        if past.empty:
            raw=np.median(valid[BASE_COLUMNS].to_numpy(float),axis=1);confidence=(
                .7*v38.rank_self(np.abs(raw-.5))+.3*v38.rank_self(agreement(valid)))
            threshold=float(np.quantile(confidence,.80));probability=raw;high=confidence>=threshold
            policy={"candidate":"median_all","direction_cutoff":.5,"confidence_quantile":.8,
                    "confidence_threshold":threshold,"selection":"LABEL_BLIND_FIXED_FIRST_FOLD"}
        else:
            split=max(200,int(.65*len(past)));meta_fit=past.iloc[:split].copy();policy_frame=past.iloc[split:].copy()
            if len(policy_frame)<100:meta_fit=past.iloc[:0].copy();policy_frame=past
            stack=fit_stack(meta_fit) if len(meta_fit)>=200 and meta_fit.y.nunique()==2 else None
            policy_probability=candidates(policy_frame,stack);policy=choose_policy(policy_frame,policy_probability)
            valid_probability=candidates(valid,stack);probability,high,confidence=apply_policy(
                valid,valid_probability,policy)
            policy["meta_fit_n"]=len(meta_fit);policy["policy_n"]=len(policy_frame)
        outputs.append(output_frame(valid,probability,high,confidence,"V42_PREQUENTIAL_ENSEMBLE"))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"market":market,"fold":int(fold),"past_oof_n":len(past),"valid_n":len(valid),
                      "past_end":past.event_time_utc.max() if len(past) else None,
                      "valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V42 ENSEMBLE] {market} fold={fold} candidate={policy['candidate']} "
              f"past={len(past)} valid={len(valid)}",flush=True)
    return pd.concat(outputs,ignore_index=True),audits


def summarize(frame):
    metrics=v36.summary(frame);robustness=v36.strategy_robustness(frame)
    return {"metrics":metrics,"cluster_weighted_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,True),
            "raw_row_metrics":v37.metric(frame,frame.prob.to_numpy(),.5,False),
            "robustness":robustness,"research_gate":v36.gate(metrics,robustness)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);frame=load_frame();outputs=[];selection={}
    for market in ("US","KR"):
        result,audits=prequential_market(frame[frame.market.eq(market)].copy(),market)
        outputs.append(result);selection[market]=audits
    selected_frame=pd.concat(outputs,ignore_index=True);result=summarize(selected_frame)
    components={}
    for name,column in ((name,f"p_{name}") for name in BASE_FILES):
        component=frame.copy();component["prob"]=component[column]
        component["high_conf"]=component[f"h_{name}"].astype(bool)
        components[name]=summarize(component)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V42","hypothesis":"ROBUST_STACKED_ENSEMBLE",
        "base_cache_sha256":{name:v37.sha256(CACHE/file) for name,file in BASE_FILES.items()},
        "models":{"V42_PREQUENTIAL_ENSEMBLE":result,**components},"selection_audits":selection,
        "validation":"current fold labels never used; meta fit/policy use only earlier purged OOF folds"}
    robustness={"version":"V42","status":status,"hypothesis":"ROBUST_STACKED_ENSEMBLE",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V42","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"prequential meta-policy; current event clusters purged from all prior meta evidence"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v42_ensemble_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
