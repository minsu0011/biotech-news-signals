"""V37 material hypothesis: causal text signal and source-specific late fusion.

All vectorizers/models/cutoffs/confidence rules are fitted inside chronological
training folds.  V36 exact DEV is opened evidence; no untouched source or seal
outcome is loaded here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V37")))
CACHE=ROOT/"cache"
SEED=20260827
COST=0.002
MATERIAL_PATTERN=re.compile(
    r"(?i)(item\s+(?:1\.01|2\.02|5\.02|7\.01|8\.01|9\.01)|ex-?99|"
    r"approv|fda|clinical|trial|phase\s*[123]|endpoint|hold|safety|adverse|recall|"
    r"offering|financ|dilut|warrant|convertible|license|partner|milestone|merger|"
    r"acquisition|earnings|guidance|patent|manufactur|"
    r"승인|반려|임상|시험|안전|리콜|유상증자|전환|신주|기술수출|계약|합병|실적|특허)"
)
TOKEN_PATTERN=re.compile(r"[^\w가-힣]+",re.UNICODE)


def atomic_json(payload:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_segment(row:pd.Series)->str:
    headline=str(row.get("headline","") or "");body=str(row.get("body","") or "")
    present=int(bool(body.strip()))
    prefix=(f"source_{row.get('source_family','')} form_{row.get('form','')} "
            f"event_{row.get('event_type','')} body_present_{present} ")
    if str(row.get("source_family"))!="US_SEC":
        return (prefix+headline+" "+body[:5000]).strip()
    snippets=[]
    for match in list(MATERIAL_PATTERN.finditer(body))[:20]:
        start=max(0,match.start()-220);end=min(len(body),match.end()+520)
        snippets.append(body[start:end])
    if not snippets:
        snippets=[body[:2200],body[-600:] if len(body)>2800 else ""]
    return (prefix+headline+" "+" ".join(snippets))[:12000].strip()


def normalized_cluster(row:pd.Series)->str:
    source=str(row.get("source_family",""))
    if source not in {"US_NEWS","KR_NEWS"}:
        return str(row.get("event_group_id",row.get("event_id","")))
    article=str(row.get("article_id","") or "").strip().lower()
    url=str(row.get("url","") or "").strip().lower().split("?",1)[0].rstrip("/")
    external=article or url
    if external:
        return hashlib.sha256(f"{source}|external|{external}".encode("utf-8")).hexdigest()
    headline=TOKEN_PATTERN.sub(" ",str(row.get("headline","")).lower()).strip()
    minute=pd.Timestamp(row.get("event_time_utc")).floor("10min")
    payload=f"{source}|fallback|{minute}|{headline}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TextModel:
    def __init__(self,column:str,kind:str):
        self.column=column;self.kind=kind
        if kind=="word":
            self.vectorizer=TfidfVectorizer(
                ngram_range=(1,2),analyzer="word",min_df=3,max_df=.997,max_features=60000,
                sublinear_tf=True,strip_accents="unicode",dtype=np.float32,
            )
        elif kind=="char":
            self.vectorizer=TfidfVectorizer(
                ngram_range=(3,5),analyzer="char_wb",min_df=3,max_features=70000,
                sublinear_tf=True,dtype=np.float32,
            )
        else:raise ValueError(kind)
        self.model=LogisticRegression(
            C=1.0,solver="liblinear",max_iter=700,random_state=SEED,class_weight=None,
        )

    def fit(self,frame:pd.DataFrame)->"TextModel":
        matrix=self.vectorizer.fit_transform(frame[self.column].fillna("").astype(str))
        self.model.fit(matrix,frame.y.astype(int),sample_weight=frame.cluster_weight);return self

    def predict(self,frame:pd.DataFrame)->np.ndarray:
        return self.model.predict_proba(
            self.vectorizer.transform(frame[self.column].fillna("").astype(str))
        )[:,1]


def fit_components(train:pd.DataFrame,target:pd.DataFrame)->tuple[dict[str,np.ndarray],dict[str,Any]]:
    output={};models={}
    numeric=v36.make_model("lightgbm",v36.FAMILIES["lightgbm"][0])
    numeric.fit(v36.feature_frame(train),train.y.astype(int),model__sample_weight=train.cluster_weight);output["numeric"]=numeric.predict_proba(
        v36.feature_frame(target))[:,1];models["numeric"]=numeric
    specifications={
        "headline_word":("headline","word"),"headline_char":("headline","char"),
        "semantic_word":("semantic_text","word"),
    }
    for name,(column,kind) in specifications.items():
        model=TextModel(column,kind).fit(train);output[name]=model.predict(target);models[name]=model
    source_probability=output["semantic_word"].copy();source_models={}
    for source,source_train in train.groupby("source_family"):
        if len(source_train)<200 or source_train.y.nunique()<2:continue
        model=TextModel("semantic_text","word").fit(source_train)
        mask=target.source_family.eq(source).to_numpy()
        if mask.any():source_probability[mask]=model.predict(target.loc[mask])
        source_models[source]=model
    output["source_semantic"]=source_probability;models["source_semantic"]=source_models
    return output,models


def candidate_specs()->list[dict[str,Any]]:
    candidates=[{"name":name,"weights":{name:1.0}} for name in
                ("numeric","headline_word","headline_char","semantic_word","source_semantic")]
    for text in ("headline_word","headline_char","semantic_word","source_semantic"):
        for numeric_weight in (.25,.50,.75):
            candidates.append({"name":f"numeric_{numeric_weight:.2f}+{text}",
                "weights":{"numeric":numeric_weight,text:1.0-numeric_weight}})
    candidates.append({"name":"three_way","weights":{
        "numeric":.34,"headline_char":.33,"source_semantic":.33}})
    return candidates


def blend(probabilities:dict[str,np.ndarray],weights:dict[str,float])->np.ndarray:
    return sum(float(weight)*probabilities[name] for name,weight in weights.items())/sum(weights.values())


def rank_self(values:np.ndarray)->np.ndarray:
    order=np.argsort(np.argsort(np.asarray(values,float),kind="stable"),kind="stable")
    return (order+1.0)/len(order)


def rank_against(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.sort(np.asarray(reference,float))
    return np.searchsorted(reference,np.asarray(values,float),side="right")/max(1,len(reference))


def agreement(probabilities:dict[str,np.ndarray],weights:dict[str,float],mean:np.ndarray)->np.ndarray:
    pieces=np.vstack([probabilities[name] for name in weights])
    return np.clip(1.0-2.0*np.mean(np.abs(pieces-mean),axis=0),0.0,1.0)


def metric(frame:pd.DataFrame,probability:np.ndarray,cutoff:float,weighted:bool=True)->dict[str,float]:
    y=frame.y.to_numpy(int);prediction=probability>=cutoff
    weight=frame.cluster_weight.to_numpy(float) if weighted and "cluster_weight" in frame else None
    accuracy=float(accuracy_score(y,prediction,sample_weight=weight))
    prevalence=float(np.average(y,weights=weight)) if weight is not None else float(y.mean())
    return {"balanced_accuracy":float(balanced_accuracy_score(y,prediction,sample_weight=weight)),
        "auc":float(roc_auc_score(y,probability,sample_weight=weight)),"accuracy":accuracy,
        "edge":accuracy-max(prevalence,1.0-prevalence)}


def source_floor(frame:pd.DataFrame,probability:np.ndarray,cutoff:float)->float:
    values=[]
    sources=frame.source_family.to_numpy()
    for source in np.unique(sources):
        mask=sources==source;y=frame.y.to_numpy(int)[mask]
        if mask.sum()>=30 and np.unique(y).size==2:
            weight=frame.cluster_weight.to_numpy(float)[mask]
            values.append(float(balanced_accuracy_score(y,probability[mask]>=cutoff,sample_weight=weight)))
    return min(values) if values else .5


def choose_policy(frame:pd.DataFrame,probabilities:dict[str,np.ndarray])->dict[str,Any]:
    rows=[];returns=frame.fwd_ret_30m.to_numpy(float);y=frame.y.to_numpy(int)
    for candidate in candidate_specs():
        probability=blend(probabilities,candidate["weights"])
        for cutoff in np.arange(.40,.601,.02):
            metrics=metric(frame,probability,float(cutoff));floor=source_floor(frame,probability,float(cutoff))
            margin=np.abs(probability-cutoff);margin_rank=rank_self(margin)
            signal=.70*margin_rank+.30*agreement(probabilities,candidate["weights"],probability)
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(signal,quantile));high=signal>=threshold
                prediction=probability>=cutoff
                weights=frame.cluster_weight.to_numpy(float)[high]
                hc_accuracy=float(np.average(prediction[high]==y[high],weights=weights))
                net=np.where(prediction[high],1.0,-1.0)*returns[high]-COST
                hc_net=float(np.average(net,weights=weights));checks=sum((metrics["balanced_accuracy"]>=.52,
                    metrics["auc"]>=.52,metrics["edge"]>=0,hc_accuracy>=.55,hc_net>0,floor>=.49))
                score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                       .40*floor+.35*hc_accuracy+8*max(-.01,min(.01,hc_net)))
                rows.append({"candidate":candidate,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,
                    "margin_reference":margin,"inner_metrics":metrics,"source_floor":floor,
                    "inner_hc_accuracy":hc_accuracy,"inner_hc_net":hc_net,
                    "checks":checks,"score":float(score)})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
    return rows[0]


def apply_policy(policy:dict[str,Any],probabilities:dict[str,np.ndarray])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    weights=policy["candidate"]["weights"];cutoff=float(policy["direction_cutoff"])
    raw=blend(probabilities,weights);margin_rank=rank_against(policy["margin_reference"],np.abs(raw-cutoff))
    signal=.70*margin_rank+.30*agreement(probabilities,weights,raw)
    high=signal>=float(policy["confidence_threshold"])
    shifted=app.shift_probability(raw,cutoff)
    return shifted,high,signal


def split_inner(train:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    ordered=train.sort_values("event_time_utc").reset_index(drop=True);split=max(160,int(.78*len(ordered)))
    valid=ordered.iloc[split:].copy();boundary=valid.event_time_utc.min()
    base=ordered[ordered.event_time_utc<boundary-pd.Timedelta(minutes=35)].copy()
    duplicate=set(valid.loc[valid.source_family.isin(["US_NEWS","KR_NEWS"]),"text_cluster_id"])
    base=base[~base.text_cluster_id.isin(duplicate)].copy()
    return base,valid


def chronological_folds(frame:pd.DataFrame)->list[tuple[pd.DataFrame,pd.DataFrame,int]]:
    ordered=frame.sort_values("event_time_utc").reset_index(drop=True)
    blocks=np.array_split(np.arange(len(ordered)),5);output=[]
    for fold in range(1,5):
        valid=ordered.iloc[blocks[fold]].copy();boundary=valid.event_time_utc.min()
        train=ordered[ordered.event_time_utc<boundary-pd.Timedelta(minutes=35)].copy()
        duplicate=set(valid.loc[valid.source_family.isin(["US_NEWS","KR_NEWS"]),"text_cluster_id"])
        train=train[~train.text_cluster_id.isin(duplicate)].copy()
        if len(train)>=200 and len(valid)>=100 and train.y.nunique()==2 and valid.y.nunique()==2:
            output.append((train,valid,fold))
    return output


def prediction_frame(valid:pd.DataFrame,probability:np.ndarray,high:np.ndarray,confidence:np.ndarray,
                     fold:int,name:str,seen:set[str])->pd.DataFrame:
    output=valid[["event_id","event_group_id","event_time_utc","market","ticker",
                  "source_family","form","y","fwd_ret_30m","text_cluster_id","cluster_weight"]].copy()
    output["prob"]=probability;output["high_conf"]=high;output["confidence_signal"]=confidence
    output["fold"]=fold;output["family"]=name;output["new_issuer"]=~output.ticker.astype(str).isin(seen)
    return output


def market_oof(frame:pd.DataFrame,market:str)->tuple[pd.DataFrame,dict[str,pd.DataFrame],list[dict[str,Any]]]:
    selected=[];components={name:[] for name in
        ("numeric","headline_word","headline_char","semantic_word","source_semantic")};audits=[]
    for train,valid,fold in chronological_folds(frame):
        inner_train,inner_valid=split_inner(train)
        inner_probability,_=fit_components(inner_train,inner_valid)
        policy=choose_policy(inner_valid.reset_index(drop=True),inner_probability)
        outer_probability,_=fit_components(train,valid)
        probability,high,confidence=apply_policy(policy,outer_probability)
        seen=set(train.ticker.astype(str));selected.append(prediction_frame(
            valid,probability,high,confidence,fold,"V37_INNER_SELECTED",seen))
        for name,raw in outer_probability.items():
            inner_margin=np.abs(inner_probability[name]-.5)
            rank=rank_against(inner_margin,np.abs(raw-.5));component_high=rank>=.80
            components[name].append(prediction_frame(valid,raw,component_high,rank,fold,name,seen))
        audit={key:value for key,value in policy.items() if key!="margin_reference"}
        audit.update({"fold":fold,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V37 TEXT] {market} fold={fold} selected={policy['candidate']['name']} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),{
        name:pd.concat(pieces,ignore_index=True) for name,pieces in components.items()},audits


def report_summary(frame:pd.DataFrame)->dict[str,Any]:
    metrics=v36.summary(frame);robust=v36.strategy_robustness(frame);research_gate=v36.gate(metrics,robust)
    weighted=metric(frame,frame.prob.to_numpy(float),.5,weighted=True)
    raw=metric(frame,frame.prob.to_numpy(float),.5,weighted=False)
    return {"metrics":metrics,"cluster_weighted_metrics":weighted,"raw_row_metrics":raw,
            "robustness":robust,"research_gate":research_gate}


def heldout_transfer(data:pd.DataFrame,weights:dict[str,float],cutoff:float)->dict[str,Any]:
    tests={};us=data[data.market.eq("US")]
    for train_source,target_source in (("US_SEC","US_NEWS"),("US_NEWS","US_SEC")):
        source_target=us[us.source_family.eq(target_source)].sort_values("event_time_utc").copy()
        boundary=source_target.event_time_utc.quantile(.75)
        target=source_target[source_target.event_time_utc>=boundary].copy()
        train=us[us.source_family.eq(train_source)&
                 (us.event_time_utc<boundary-pd.Timedelta(minutes=35))].copy()
        if len(train)<200 or len(target)<100:
            tests[f"{train_source}->{target_source}"]={"status":"INSUFFICIENT_PAST_ONLY",
                "train_n":len(train),"target_n":len(target),"boundary":boundary};continue
        probabilities,_=fit_components(train,target)
        usable={name:weight for name,weight in weights.items() if name!="source_semantic"}
        if not usable:usable={"semantic_word":1.0}
        raw=blend(probabilities,usable);tests[f"{train_source}->{target_source}"]={
            "status":"EVALUATED_PAST_ONLY","train_n":len(train),"target_n":len(target),
            "boundary":boundary,"metrics":metric(target,raw,cutoff)}
    return tests


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},
                     parse_dates=["event_time_utc"])
    if data.event_group_id.duplicated().any():raise RuntimeError("V37 event-group duplication")
    if set(data.contract_version)!={"EXACT_T2_30M_V36"}:raise RuntimeError("V37 mixed contract")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    if events.event_id.duplicated().any():raise RuntimeError("V37 metadata event_id duplication")
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["semantic_text"]=data.apply(semantic_segment,axis=1)
    data["text_cluster_id"]=data.apply(normalized_cluster,axis=1)
    data["cluster_weight"]=1.0/data.groupby("text_cluster_id").event_id.transform("size")
    selected_by_market={};components_by_market={};selection={}
    for market in ("US","KR"):
        selected,components,audits=market_oof(data[data.market.eq(market)].copy(),market)
        selected_by_market[market]=selected;components_by_market[market]=components;selection[market]=audits
    combined=pd.concat(list(selected_by_market.values()),ignore_index=True)
    result=report_summary(combined)
    comparisons={"V37_INNER_SELECTED":result}
    for name in components_by_market["US"]:
        frame=pd.concat([components_by_market["US"][name],components_by_market["KR"][name]],ignore_index=True)
        comparisons[name]=report_summary(frame)
    policies=[audit for values in selection.values() for audit in values]
    majority=Counter(audit["candidate"]["name"] for audit in policies).most_common(1)[0][0]
    representative=next(audit for audit in policies if audit["candidate"]["name"]==majority)
    transfer={"v36_seal_outcomes_loaded":False,"selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "heldout_tests":heldout_transfer(data,representative["candidate"]["weights"],
                                          representative["direction_cutoff"]),
        "representative_policy":{"name":majority,"weights":representative["candidate"]["weights"],
                                 "cutoff":representative["direction_cutoff"]}}
    comparison_report={"version":"V37","hypothesis":"TEXT_SIGNAL_SOURCE_SPECIFIC",
        "dev_sha256":sha256(path),"v36_baseline":json.loads(
            (ROOT/"output_V36"/"DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"],
        "models":comparisons,"selection_audits":selection}
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    robustness={"version":"V37","status":status,"hypothesis":"TEXT_SIGNAL_SOURCE_SPECIFIC",
        "v36_seal_outcomes_loaded":False,"opened_dev_only":True,"selected":result,
        "component_comparison":comparisons,"selection_audits":selection,"seal_authorized":False}
    atomic_json(comparison_report,OUT/"MODEL_COMPARISON.json")
    atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    combined.to_csv(CACHE/"v37_text_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"],
                      "majority_policy":transfer["representative_policy"]},indent=2,default=str))


if __name__=="__main__":main()
