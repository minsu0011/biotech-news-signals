"""Fine joint cutoff search for the cross-version V27 rank directions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent
TARGET={"US":258.0,"KR":325.0}
SETS={
    "v26":{"US":({"SEC_V26_DEV"},{"SEC_V26_SEAL"}),
           "KR":({"NAVER_NEWS_V26_DEV"},{"NAVER_NEWS_V26_SEAL"})},
    "v27":{"US":({"SEC_V27_DEV"},{"SEC_V27_DEV"}),
           "KR":({"NAVER_NEWS_V27_DEV"},{"NAVER_NEWS_V27_DEV"})},
}


def rank(reference,values):
    reference=np.sort(np.asarray(reference,float)[np.isfinite(reference)])
    fill=float(np.median(reference)) if len(reference) else 0.0
    values=np.nan_to_num(np.asarray(values,float),nan=fill,posinf=fill,neginf=fill)
    if not len(reference):return np.full(len(values),.5)
    left=np.searchsorted(reference,values,"left");right=np.searchsorted(reference,values,"right")
    return (left+right+1)/(2*len(reference))


def score(reference,target,components):
    values=[]
    for column,sign,weight in components:
        rr=pd.to_numeric(reference[column],errors="coerce").to_numpy(float)
        tv=pd.to_numeric(target[column],errors="coerce").to_numpy(float)
        value=rank(rr,tv);values.append(float(weight)*(value if float(sign)>0 else 1-value))
    return np.sum(values,axis=0)/sum(float(item[2]) for item in components)


def basic(y,pred,weight):
    total=weight.sum();accuracy=float(np.sum(weight*(pred==y))/total)
    up=float(np.sum(weight*y)/total);positive=weight*y;negative=weight*(1-y)
    tpr=float(np.sum(positive*pred)/positive.sum());tnr=float(np.sum(negative*(~pred))/negative.sum())
    return {"accuracy":accuracy,"balanced_accuracy":.5*(tpr+tnr),
            "up_rate":up,"edge":accuracy-max(up,1-up)}


def marker(components):return json.dumps(components,separators=(",",":"))


def main():
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V27_SEAL","NAVER_NEWS_V27_SEAL"}),"y"].notna().any():
        raise RuntimeError("V27 seal exposed")
    opened=data[data.y.notna()].copy();frames={};features={}
    for label,markets in SETS.items():
        frames[label]={};features[label]={}
        for market,(ref_sources,target_sources) in markets.items():
            reference=opened[opened.market.eq(market)&opened.source.isin(ref_sources)].copy()
            target=opened[opened.market.eq(market)&opened.source.isin(target_sources)].copy()
            frames[label][market]=target;features[label][market]=(app.model_frame(reference),app.model_frame(target))
    joint=json.loads((ROOT/"cache"/"v27_joint_search.json").read_text(encoding="utf-8"))
    components={"US":[],"KR":[]}
    for market in ("US","KR"):
        seen=set()
        for row in joint["short"][market]:
            key=marker(row["components"])
            if key not in seen:seen.add(key);components[market].append(row["components"])
            if len(components[market])>=40:break
    raw={label:{market:{} for market in ("US","KR")} for label in SETS}
    for label in SETS:
        for market in ("US","KR"):
            reference,target=features[label][market]
            for value in components[market]:raw[label][market][marker(value)]=score(reference,target,value)
    short={}
    for market in ("US","KR"):
        rows=[]
        for value in components[market]:
            key=marker(value)
            aucs={label:float(roc_auc_score(frames[label][market].y.astype(int),raw[label][market][key]))
                  for label in SETS}
            if min(aucs.values())<.55:continue
            for cutoff in np.arange(.30,.851,.005):
                sets={}
                for label in SETS:
                    y=frames[label][market].y.astype(int).to_numpy();weight=np.ones(len(y))
                    sets[label]=basic(y,raw[label][market][key]>=cutoff,weight);sets[label]["auc"]=aucs[label]
                if min(item["balanced_accuracy"] for item in sets.values())<.51:continue
                quality=min(item["balanced_accuracy"] for item in sets.values())+min(
                    item["edge"] for item in sets.values())
                rows.append({"components":value,"cutoff":float(cutoff),"sets":sets,"quality":quality})
        rows.sort(key=lambda row:row["quality"],reverse=True);short[market]=rows[:500]
        print("SHORT",market,len(rows),flush=True)
    policies=[]
    for us in short["US"]:
        for kr in short["KR"]:
            sets={};classification_pass=True
            for label in SETS:
                ys=[];ps=[];weights=[];market_metrics={}
                for market,candidate in (("US",us),("KR",kr)):
                    frame=frames[label][market];y=frame.y.astype(int).to_numpy()
                    probability=np.clip(.5+raw[label][market][marker(candidate["components"])]-candidate["cutoff"],1e-6,1-1e-6)
                    weight=np.full(len(y),TARGET[market]/len(y)) if label=="v27" else np.ones(len(y))
                    market_metrics[market]=basic(y,probability>=.5,weight)
                    ys.append(y);ps.append(probability);weights.append(weight)
                y=np.concatenate(ys);p=np.concatenate(ps);weight=np.concatenate(weights)
                result=basic(y,p>=.5,weight);result["auc"]=float(roc_auc_score(y,p,sample_weight=weight))
                result["by_market"]=market_metrics;sets[label]=result
                if not (result["balanced_accuracy"]>=.54 and result["edge"]>=.02 and
                        all(value["balanced_accuracy"]>=.51 for value in market_metrics.values())):
                    classification_pass=False;break
            if not classification_pass:continue
            passes=sum(sets[label]["auc"]>=.56 for label in SETS)+8
            quality=min(sets[label]["balanced_accuracy"] for label in SETS)+min(
                sets[label]["edge"] for label in SETS)+min(sets[label]["auc"] for label in SETS)
            policies.append({"markets":{"US":us,"KR":kr},"sets":sets,
                             "passes":passes,"quality":float(quality)})
    policies.sort(key=lambda row:(row["passes"],row["quality"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
            "short_counts":{market:len(rows) for market,rows in short.items()},"policies":policies[:500]}
    path=ROOT/"cache"/"v27_fine_cutoff.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("POLICIES",len(policies))
    for row in policies[:30]:print(json.dumps(row,ensure_ascii=False),flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
