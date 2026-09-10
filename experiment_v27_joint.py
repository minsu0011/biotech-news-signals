"""Joint US/KR V27 rank-direction policy search on opened rolling audits."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent
TARGET={"US":258.0,"KR":325.0}
SETS={
    "v24":{"US":({"SEC_V24_DEV"},{"SEC_V24_SEAL"}),
           "KR":({"NAVER_NEWS_V24_DEV"},{"NAVER_NEWS_V24_SEAL"})},
    "v25":{"US":({"SEC_V25_DEV"},{"SEC_V25_SEAL"}),
           "KR":({"NAVER_NEWS_V25_DEV"},{"NAVER_NEWS_V25_SEAL"})},
    "v26":{"US":({"SEC_V26_DEV"},{"SEC_V26_SEAL"}),
           "KR":({"NAVER_NEWS_V26_DEV"},{"NAVER_NEWS_V26_SEAL"})},
    "v27":{"US":({"SEC_V27_DEV"},{"SEC_V27_DEV"}),
           "KR":({"NAVER_NEWS_V27_DEV"},{"NAVER_NEWS_V27_DEV"})},
}


def rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.sort(np.asarray(reference,float)[np.isfinite(reference)])
    fill=float(np.median(reference)) if len(reference) else 0.0
    values=np.nan_to_num(np.asarray(values,float),nan=fill,posinf=fill,neginf=fill)
    if not len(reference):return np.full(len(values),.5)
    left=np.searchsorted(reference,values,"left");right=np.searchsorted(reference,values,"right")
    return (left+right+1)/(2*len(reference))


def component_score(reference:pd.DataFrame,target:pd.DataFrame,components)->np.ndarray:
    result=[]
    for column,sign,weight in components:
        rr=pd.to_numeric(reference[column],errors="coerce").to_numpy(float)
        tv=pd.to_numeric(target[column],errors="coerce").to_numpy(float)
        value=rank(rr,tv);result.append(float(weight)*(value if float(sign)>0 else 1-value))
    return np.sum(result,axis=0)/sum(float(item[2]) for item in components)


def part(frame:pd.DataFrame,probability:np.ndarray,market:str,label:str)->pd.DataFrame:
    out=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    out["prob"]=probability
    out["eval_weight"]=(TARGET[market]/len(out)) if label=="v27" else 1.0
    return out


def key(components)->str:
    return json.dumps(components,sort_keys=True,separators=(",",":"))


def main()->None:
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
    rank_rows=json.loads((ROOT/"cache"/"v27_rank_search.json").read_text(encoding="utf-8"))["rows"]
    candidate_components={"US":[],"KR":[]}
    for market in ("US","KR"):
        seen=set()
        for row in [item for item in rank_rows if item["market"]==market]:
            marker=key(row["components"])
            if marker in seen:continue
            seen.add(marker);candidate_components[market].append(row["components"])
            if len(candidate_components[market])>=100:break
    scores={label:{market:{} for market in ("US","KR")} for label in SETS}
    for label in SETS:
        for market in ("US","KR"):
            reference,target=features[label][market]
            for components in candidate_components[market]:
                scores[label][market][key(components)]=component_score(reference,target,components)
    short={}
    for market in ("US","KR"):
        rows=[]
        for components in candidate_components[market]:
            marker=key(components)
            for cutoff in np.arange(.30,.701,.01):
                sets={}
                for label in SETS:
                    probability=np.clip(.5+scores[label][market][marker]-float(cutoff),1e-6,1-1e-6)
                    sets[label]=app.evaluate(part(frames[label][market],probability,market,label),.925,.002)
                latest=(sets["v26"],sets["v27"]);older=(sets["v24"],sets["v25"])
                checks=sum(x["balanced_accuracy"]>=.51 for x in latest)+sum(
                    x["auc"]>=.55 for x in latest)+sum(x["edge_vs_naive"]>=-.01 for x in latest)+sum(
                    x["balanced_accuracy"]>=.50 for x in older)
                score=min(x["balanced_accuracy"] for x in latest)+min(x["auc"] for x in latest)+min(
                    x["edge_vs_naive"] for x in latest)+.15*min(x["balanced_accuracy"] for x in older)
                rows.append({"market":market,"components":components,"cutoff":float(cutoff),
                             "sets":sets,"checks":checks,"score":float(score)})
        rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);short[market]=rows[:120]
        print("SHORT",market,flush=True)
        for row in rows[:10]:print(json.dumps(row,ensure_ascii=False),flush=True)
    policies=[]
    for us in short["US"]:
        for kr in short["KR"]:
            row={"markets":{"US":us,"KR":kr},"sets":{}}
            checks=[]
            for label in SETS:
                parts=[]
                for market,candidate in (("US",us),("KR",kr)):
                    marker=key(candidate["components"])
                    probability=np.clip(.5+scores[label][market][marker]-candidate["cutoff"],1e-6,1-1e-6)
                    parts.append(part(frames[label][market],probability,market,label))
                metrics=app.evaluate(pd.concat(parts,ignore_index=True),.925,.002);row["sets"][label]=metrics
            for label in ("v26","v27"):
                value=row["sets"][label]
                checks.extend([value["balanced_accuracy"]>=.54,value["auc"]>=.56,
                               value["edge_vs_naive"]>=.02,
                               value["by_market"]["US"]["balanced_accuracy"]>=.51,
                               value["by_market"]["KR"]["balanced_accuracy"]>=.51])
            row["passes"]=sum(checks)
            a=row["sets"]["v26"];c=row["sets"]["v27"]
            row["score"]=float(min(a["balanced_accuracy"],c["balanced_accuracy"])+
                               min(a["auc"],c["auc"])+min(a["edge_vs_naive"],c["edge_vs_naive"])+
                               .1*min(row["sets"]["v24"]["balanced_accuracy"],
                                    row["sets"]["v25"]["balanced_accuracy"]))
            policies.append(row)
    policies.sort(key=lambda row:(row["passes"],row["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
            "short":short,"policies":policies[:300]}
    path=ROOT/"cache"/"v27_joint_search.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("POLICIES")
    for row in policies[:30]:print(json.dumps(row,ensure_ascii=False),flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
