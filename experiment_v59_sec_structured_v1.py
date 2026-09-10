"""V59: segment-aware SEC text and explicit structured-semantics direction experts."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import bio_news_30m_v3 as app
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent;DATA=ROOT/"data";CACHE=ROOT/"cache"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V59")))
ITEM_RE=re.compile(r"(?i)\bitem\s+(1\.01|2\.02|2\.03|3\.02|5\.02|7\.01|8\.01|9\.01)\b")
EX99_RE=re.compile(r"(?mi)^\s*EX(?:HIBIT)?[-_ ]?99(?:\.\d+)?\s*$")
SIGNATURE_RE=re.compile(r"(?mi)^\s*SIGNATURES?\s*$")
DISCLAIMER_RE=re.compile(r"(?is)(forward[- ]looking statements?|safe harbor).{0,5000}$")
TAXONOMY={
    "fda_approval":r"\bfda\b.{0,60}\bapprov(?:al|ed)\b|\bapproval\b.{0,60}\bfda\b",
    "crl":r"complete response letter|\bcrl\b|refusal to file",
    "clinical_hold_imposed":r"clinical hold.{0,80}(?:impos|placed|initiated)|(?:impos|placed).{0,80}clinical hold",
    "clinical_hold_lifted":r"clinical hold.{0,80}(?:lift|remov)|(?:lift|remov).{0,80}clinical hold",
    "phase_1":r"\bphase\s*(?:1|i)\b","phase_2":r"\bphase\s*(?:2|ii)\b","phase_3":r"\bphase\s*(?:3|iii)\b",
    "endpoint_met":r"(?:met|achieved)\s+(?:the\s+)?(?:primary|secondary)?\s*endpoint",
    "endpoint_missed":r"(?:missed|did not meet|failed to meet)\s+(?:the\s+)?(?:primary|secondary)?\s*endpoint",
    "trial_positive":r"positive (?:topline|top-line|clinical|trial) results?|statistically significant",
    "trial_negative":r"negative (?:topline|top-line|clinical|trial) results?|not statistically significant",
    "public_offering":r"public offering","registered_direct":r"registered direct",
    "atm":r"at-the-market|\batm (?:program|facility|offering)","convertible":r"convertible (?:note|debt|security)",
    "warrant":r"\bwarrants?\b","offering_priced":r"offering.{0,100}\bpriced\b|priced.{0,100}offering",
    "offering_closed":r"offering.{0,100}\bclos(?:e|ed|ing)\b|clos(?:e|ed|ing).{0,100}offering",
    "license":r"licens(?:e|ing|ed) agreement","partnership":r"partnership|collaboration agreement",
    "upfront":r"upfront payment","milestone":r"milestone payment",
    "ma_definitive":r"definitive (?:merger|acquisition) agreement|agreement and plan of merger",
    "ma_terminated":r"terminat(?:e|ed|ion).{0,100}(?:merger|acquisition) agreement",
    "earnings":r"earnings|results of operations|quarterly results","guidance":r"financial guidance|revis(?:e|ed) guidance",
    "management_appointment":r"appoint(?:ed|ment).{0,100}(?:chief|director|officer|president)",
    "management_resignation":r"resign(?:ed|ation).{0,100}(?:chief|director|officer|president)",
    "safety":r"serious adverse event|safety signal","recall":r"product recall|voluntary recall",
    "manufacturing":r"manufacturing (?:issue|deficien|facility)|cGMP deficien",
}
AMOUNT_RE=re.compile(r"(?i)(?:\$|usd\s*)(\d+(?:\.\d+)?)\s*(million|billion|m|bn)?")
OFFER_PRICE_RE=re.compile(r"(?i)(?:offering|purchase) price.{0,80}?\$\s*(\d+(?:\.\d+)?)")
TRIAL_N_RE=re.compile(r"(?i)\b(?:n\s*=|enroll(?:ed|ment of))\s*(\d{1,5})\b")
PVALUE_RE=re.compile(r"(?i)\bp\s*[=<]\s*(0?\.\d+)")
HR_RE=re.compile(r"(?i)hazard ratio.{0,30}?(0?\.\d+)")


def file_sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_bodies(frame:pd.DataFrame)->tuple[pd.DataFrame,list[dict[str,Any]]]:
    output=frame.copy();manifest=[]
    for index,row in output[output.source_family.eq("US_SEC")&output.body.fillna("").astype(str).str.strip().eq("")].iterrows():
        match=re.match(r"^SEC:(\d+):(.+)$",str(row.event_id))
        if not match:continue
        path=CACHE/"sec_text"/f"{int(match.group(1))}_{match.group(2).replace('-','')}.txt"
        if not path.exists():continue
        text=path.read_text(encoding="utf-8",errors="replace").strip()
        if not text:continue
        output.at[index,"body"]=text
        manifest.append({"event_id":str(row.event_id),"path":str(path.relative_to(ROOT)),
                         "sha256":file_sha(path),"bytes":path.stat().st_size})
    return output,manifest


def segments(row:pd.Series)->dict[str,str]:
    headline=str(row.get("headline","") or "");body=str(row.get("body","") or "")
    body=DISCLAIMER_RE.sub(" ",body);signature=SIGNATURE_RE.search(body)
    if signature:body=body[:signature.start()]
    matches=list(ITEM_RE.finditer(body));item_parts=[]
    for number,match in enumerate(matches[:20]):
        end=matches[number+1].start() if number+1<len(matches) else len(body)
        item_parts.append(body[match.start():end])
    exmatch=EX99_RE.search(body);ex99=body[exmatch.end():] if exmatch else ""
    primary=" ".join(item_parts) if item_parts else body[:12000]
    if exmatch:primary=primary[:max(0,primary.find(exmatch.group(0)))]
    return {"headline":headline,"primary":primary[:16000],"ex99":ex99[:16000],
            "combined":f"HEADLINE {headline} PRIMARY {primary[:12000]} EX99 {ex99[:12000]}"}


def structured_features(row:pd.Series)->dict[str,float]:
    part=segments(row);body=str(row.get("body","") or "");text=(part["headline"]+" "+part["primary"]+" "+part["ex99"]).lower()
    result:dict[str,float]={f"form={str(row.get('form','')).upper()}":1.,
        f"event={str(row.get('event_type','')).upper()}":1.,"body_present":float(bool(body.strip())),
        "ex99_present":float(bool(part["ex99"])),"log_body_len":float(np.log1p(len(body))),
        "log_primary_len":float(np.log1p(len(part["primary"]))),"log_ex99_len":float(np.log1p(len(part["ex99"]))),
        "amendment":float(str(row.get("form","")).endswith("/A")),
        "signature_present":float(bool(SIGNATURE_RE.search(body))),
        "forward_disclaimer_present":float(bool(DISCLAIMER_RE.search(body)))}
    items=sorted(set(ITEM_RE.findall(body)))
    for item in items:result[f"item={item}"]=1.
    for left,right in zip(items,items[1:]):result[f"item_pair={left}|{right}"]=1.
    for name,pattern in TAXONOMY.items():
        regex=re.compile(pattern,re.I|re.S);total=len(regex.findall(text));result[f"tax={name}"]=float(min(total,10))
        result[f"primary_tax={name}"]=float(bool(regex.search(part["primary"])))
        result[f"ex99_tax={name}"]=float(bool(regex.search(part["ex99"])))
    amounts=[]
    for value,scale in AMOUNT_RE.findall(text):
        number=float(value)*({"million":1e6,"m":1e6,"billion":1e9,"bn":1e9}.get(scale.lower(),1.))
        if 0<number<1e13:amounts.append(number)
    result["max_amount_log"]=float(np.log1p(max(amounts))) if amounts else 0.
    offer=OFFER_PRICE_RE.search(text);entry=float(row.get("log_entry_price",np.nan))
    if offer and np.isfinite(entry):result["offering_price_vs_entry"]=float(np.clip(float(offer.group(1))/np.exp(entry)-1,-2,2))
    trial=[int(value) for value in TRIAL_N_RE.findall(text) if int(value)>0]
    result["max_trial_n_log"]=float(np.log1p(max(trial))) if trial else 0.
    pvalues=[float(value) for value in PVALUE_RE.findall(text) if float(value)>0]
    result["min_pvalue_neglog"]=float(-np.log10(min(pvalues))) if pvalues else 0.
    hazards=[float(value) for value in HR_RE.findall(text) if float(value)>0]
    result["hazard_ratio_log"]=float(np.log(hazards[0])) if hazards else 0.
    return result


class TextExpert:
    def __init__(self,c:float):
        self.vector=TfidfVectorizer(lowercase=True,strip_accents="unicode",ngram_range=(1,2),
            min_df=3,max_df=.995,max_features=24000,sublinear_tf=True,dtype=np.float32)
        self.model=LogisticRegression(C=c,class_weight="balanced",max_iter=350,solver="liblinear",random_state=59)
    def fit(self,frame:pd.DataFrame):
        x=self.vector.fit_transform(frame.sec_segment_text);self.model.fit(x,frame.y.astype(int));return self
    def predict(self,frame:pd.DataFrame)->np.ndarray:
        return self.model.predict_proba(self.vector.transform(frame.sec_segment_text))[:,1]


class StructureExpert:
    def __init__(self,c:float):
        self.vector=DictVectorizer(sparse=True,dtype=np.float32)
        self.model=LogisticRegression(C=c,class_weight="balanced",max_iter=350,solver="liblinear",random_state=59)
    def fit(self,frame:pd.DataFrame):
        x=self.vector.fit_transform(frame.sec_structure.tolist());self.model.fit(x,frame.y.astype(int));return self
    def predict(self,frame:pd.DataFrame)->np.ndarray:
        return self.model.predict_proba(self.vector.transform(frame.sec_structure.tolist()))[:,1]


def fit_predict(train:pd.DataFrame,target:pd.DataFrame,c:float)->tuple[np.ndarray,np.ndarray]:
    if len(train)<300 or train.y.nunique()<2:return np.full(len(target),.5),np.full(len(target),.5)
    return TextExpert(c).fit(train).predict(target),StructureExpert(c).fit(train).predict(target)


BLENDS={"N":{"N":1.},"T":{"T":1.},"S":{"S":1.},"N_T":{"N":.5,"T":.5},
        "N_S":{"N":.5,"S":.5},"N_T_S":{"N":.5,"T":.25,"S":.25}}


def blend(parts:dict[str,np.ndarray],weights:dict[str,float])->np.ndarray:
    values=np.zeros(len(next(iter(parts.values()))));total=0.
    for name,weight in weights.items():values+=weight*logit(np.clip(parts[name],1e-5,1-1e-5));total+=weight
    return expit(values/total)


def body_ba(frame:pd.DataFrame,prob:np.ndarray,cutoff:float,present:bool)->float:
    mask=frame.body_present.to_numpy(bool)==present
    if mask.sum()<10 or frame.loc[mask,"y"].nunique()<2:return .5
    return v37.metric(frame.loc[mask].reset_index(drop=True),prob[mask],cutoff,True)["balanced_accuracy"]


def choose(inner:pd.DataFrame,base:np.ndarray)->dict[str,Any]:
    trials=[]
    for c in (.25,.5,1.):
        text,structure=fit_predict(inner.attrs["train_sec"],inner,c);parts={"N":base,"T":text,"S":structure}
        for name,weights in BLENDS.items():
            raw=blend(parts,weights)
            for cutoff in (.46,.48,.50,.52,.54):
                metric=v37.metric(inner,raw,cutoff,True);present=body_ba(inner,raw,cutoff,True);missing=body_ba(inner,raw,cutoff,False)
                floor=min(present,missing);checks=sum((metric["balanced_accuracy"]>=.515,metric["auc"]>=.515,floor>=.49))
                score=(2*metric["balanced_accuracy"]+1.5*metric["auc"]+.5*floor-.25*abs(present-missing))
                trials.append({"C":c,"blend":name,"weights":weights,"cutoff":cutoff,"checks":checks,
                    "score":float(score),"inner_metrics":metric,"body_present_BA":present,"body_missing_BA":missing})
    trials.sort(key=lambda row:(row["checks"],row["score"],-len(row["weights"])),reverse=True)
    return trials[0]


def summarize(frame:pd.DataFrame)->dict[str,Any]:return v44.summarize(frame)


def main()->None:
    runtime_limits.configure();OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    original_present=data.body.str.strip().ne("");data,recovery=recover_bodies(data)
    data["body_present"]=data.body.str.strip().ne("")
    data["sec_segment_text"]="";data["sec_structure"]=None
    sec_mask=data.source_family.eq("US_SEC")
    data.loc[sec_mask,"sec_segment_text"]=data.loc[sec_mask].apply(lambda row:segments(row)["combined"],axis=1)
    structures=data.loc[sec_mask].apply(structured_features,axis=1)
    for index,value in structures.items():data.at[index,"sec_structure"]=value
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    base=pd.read_csv(CACHE/"v58_locked_source_routing_oof.csv.gz",compression="gzip",low_memory=False,
                     dtype={"ticker":str},parse_dates=["event_time_utc"])
    if not set(base.event_id).issubset(set(data.event_id)):raise RuntimeError("V59 V58/data event mismatch")
    selected=base.copy();component_probs={name:base.prob.to_numpy(float).copy() for name in BLENDS}
    selection=[]
    us=data[data.market.eq("US")].copy()
    for train,valid,fold in v37.chronological_folds(us):
        train_sec=train[train.source_family.eq("US_SEC")].copy();valid_sec=valid[valid.source_family.eq("US_SEC")].copy()
        inner_train,inner_valid=v37.split_inner(train);inner_train_sec=inner_train[inner_train.source_family.eq("US_SEC")].copy()
        inner_valid_sec=inner_valid[inner_valid.source_family.eq("US_SEC")].copy()
        overlap=set(inner_train_sec.event_group_id if "event_group_id" in inner_train_sec else [])&set(inner_valid_sec.event_group_id if "event_group_id" in inner_valid_sec else [])
        if overlap:raise RuntimeError("V59 inner event-group overlap")
        inner_base=v38.target_fit(inner_train,inner_valid_sec,"y")
        inner_valid_sec.attrs["train_sec"]=inner_train_sec
        policy=choose(inner_valid_sec,inner_base)
        text,structure=fit_predict(train_sec,valid_sec,float(policy["C"]));parts={"N":v38.target_fit(train,valid_sec,"y"),"T":text,"S":structure}
        valid_positions=selected.index[selected.event_id.isin(set(valid_sec.event_id))]
        chosen=blend(parts,policy["weights"]);shifted=app.shift_probability(chosen,float(policy["cutoff"]))
        mapping=dict(zip(valid_sec.event_id,shifted));selected.loc[valid_positions,"prob"]=selected.loc[valid_positions,"event_id"].map(mapping)
        selected.loc[valid_positions,"family"]="V59_US_SEC_STRUCTURED_SEMANTICS_V1"
        for name,weights in BLENDS.items():
            values=blend(parts,weights);lookup=dict(zip(valid_sec.event_id,values))
            positions=np.flatnonzero(base.event_id.isin(set(valid_sec.event_id)).to_numpy())
            component_probs[name][positions]=base.iloc[positions].event_id.map(lookup).to_numpy(float)
        selection.append({**policy,"fold":fold,"train_n":len(train),"train_sec_n":len(train_sec),
            "inner_train_sec_n":len(inner_train_sec),"inner_valid_sec_n":len(inner_valid_sec),"valid_sec_n":len(valid_sec),
            "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min(),"event_group_overlap":0})
        print(f"[V59 SEC] fold={fold} blend={policy['blend']} C={policy['C']} cutoff={policy['cutoff']} checks={policy['checks']}/3",flush=True)
    selected_result=summarize(selected);models={"V59_INNER_SELECTED":selected_result}
    component_frame=pd.DataFrame({"event_id":base.event_id,"fold":base.fold})
    for name,prob in component_probs.items():
        frame=base.copy();frame["prob"]=prob;models[name]=summarize(frame);component_frame[f"prob_{name}"]=prob
    status="ROBUST_SURVIVOR" if selected_result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    recovery_payload={"files":recovery,"aggregate_sha256":hashlib.sha256(json.dumps(recovery,sort_keys=True).encode()).hexdigest()}
    feature_audit={"version":"V59","selection_uses_labels":False,"US_SEC_rows":int(sec_mask.sum()),
        "original_body_present":int((original_present&sec_mask).sum()),"cache_recovered":len(recovery),
        "effective_body_present":int((data.body_present&sec_mask).sum()),"body_missing":int((~data.body_present&sec_mask).sum()),
        "item_detected":int(data.loc[sec_mask,"body"].str.contains(ITEM_RE).sum()),
        "ex99_detected":int(data.loc[sec_mask,"body"].str.contains(EX99_RE).sum()),
        "taxonomy_counts":{name:int(data.loc[sec_mask,"body"].str.contains(pattern,case=False,regex=True).sum()) for name,pattern in TAXONOMY.items()},
        "cache_recovery":recovery_payload}
    comparison={"version":"V59","hypothesis":"US_SEC_STRUCTURED_SEMANTICS_V1","dev_sha256":v37.sha256(path),
        "v58_oof_sha256":v37.sha256(CACHE/"v58_locked_source_routing_oof.csv.gz"),"models":models,
        "selection_audits":selection,"feature_audit_sha256":hashlib.sha256(json.dumps(feature_audit,sort_keys=True,default=str).encode()).hexdigest()}
    robustness={"version":"V59","status":status,"hypothesis":"US_SEC_STRUCTURED_SEMANTICS_V1",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":selected_result,
        "selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V59","seal_outcomes_loaded":False,
        "selected_oof_by_source":selected_result["metrics"]["by_source_family"],
        "selected_oof_by_market":selected_result["metrics"]["by_market"],
        "selected_oof_new_issuer":selected_result["metrics"]["new_issuer"],
        "method":"V58 fixed routing/confidence; nested past-only SEC segment text and explicit structure replace US_SEC direction only"}
    v37.atomic_json(feature_audit,OUT/"SEC_FEATURE_AUDIT.json");v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json");v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v59_sec_structured_oof.csv.gz",index=False,compression="gzip")
    component_frame.to_csv(CACHE/"v59_sec_components_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":selected_result["metrics"],"gate":selected_result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
