"""Outcome-independent low-reliability eligibility search for V27."""
from __future__ import annotations

import itertools,json
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
GATES=tuple(column for column in app.NUMERIC_COLS if column not in {
    "year","month","weekday","hour","minute","minutes_from_open","minutes_to_close",
})


def rank(reference,values):
    reference=np.sort(np.asarray(reference,float)[np.isfinite(reference)])
    fill=float(np.median(reference)) if len(reference) else 0.0
    values=np.nan_to_num(np.asarray(values,float),nan=fill,posinf=fill,neginf=fill)
    if not len(reference):return np.full(len(values),.5)
    left=np.searchsorted(reference,values,"left");right=np.searchsorted(reference,values,"right")
    return (left+right+1)/(2*len(reference))


def direction(reference,target,components):
    values=[]
    for column,sign,weight in components:
        rr=pd.to_numeric(reference[column],errors="coerce").to_numpy(float)
        tv=pd.to_numeric(target[column],errors="coerce").to_numpy(float)
        value=rank(rr,tv);values.append(float(weight)*(value if float(sign)>0 else 1-value))
    return np.sum(values,axis=0)/sum(float(item[2]) for item in components)


def eligibility(reference,target,ref_direction,target_direction,cutoff,column,sign,mw,q):
    ref_margin=np.abs(ref_direction-cutoff);target_margin=np.abs(target_direction-cutoff)
    margin=rank(ref_margin,target_margin);ref_margin_rank=rank(ref_margin,ref_margin)
    rr=float(sign)*pd.to_numeric(reference[column],errors="coerce").to_numpy(float)
    tv=float(sign)*pd.to_numeric(target[column],errors="coerce").to_numpy(float)
    opportunity=rank(rr,tv);ref_opportunity=rank(rr,rr)
    reference_confidence=float(mw)*ref_margin_rank+(1-float(mw))*ref_opportunity
    confidence=float(mw)*margin+(1-float(mw))*opportunity
    return rank(reference_confidence,confidence)>=float(q)


def array_metrics(y,p,weight):
    pred=p>=.5;total=weight.sum();positive=weight*y;negative=weight*(1-y)
    accuracy=float(np.sum(weight*(pred==y))/total)
    up=float(np.sum(weight*y)/total)
    tpr=float(np.sum(positive*pred)/positive.sum())
    tnr=float(np.sum(negative*(~pred))/negative.sum())
    return {"n":len(y),"accuracy":accuracy,"balanced_accuracy":.5*(tpr+tnr),
            "auc":float(roc_auc_score(y,p,sample_weight=weight)),
            "up_rate":up,"naive_accuracy":max(up,1-up),
            "edge_vs_naive":accuracy-max(up,1-up)}


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
    policies=json.loads((ROOT/"cache"/"v27_joint_search.json").read_text(encoding="utf-8"))["policies"][:15]
    rows=[]
    for number,policy in enumerate(policies):
        raw={};probability={}
        for label in SETS:
            raw[label]={};probability[label]={}
            for market in ("US","KR"):
                spec=policy["markets"][market];reference,target=features[label][market]
                raw[label][market]=direction(reference,target,spec["components"])
                probability[label][market]=np.clip(.5+raw[label][market]-spec["cutoff"],1e-6,1-1e-6)
        kr_spec=policy["markets"]["KR"]
        kr_ref_direction={label:direction(features[label]["KR"][0],features[label]["KR"][0],
                                          kr_spec["components"]) for label in SETS}
        for column,sign,mw,q in itertools.product(GATES,(-1,1),(.25,.5,.75,1.0),(.05,.10,.15,.20)):
            eligible={};retention={};expected=0.0
            for label in SETS:
                eligible[label]={"US":np.ones(len(frames[label]["US"]),bool)}
                eligible[label]["KR"]=eligibility(
                    features[label]["KR"][0],features[label]["KR"][1],
                    kr_ref_direction[label],raw[label]["KR"],kr_spec["cutoff"],column,sign,mw,q,
                )
                retention[label]={market:float(eligible[label][market].mean()) for market in ("US","KR")}
            expected=TARGET["US"]*retention["v27"]["US"]+TARGET["KR"]*retention["v27"]["KR"]
            if expected<515 or retention["v26"]["KR"]<.75:continue
            sets={};passes=[]
            for label in SETS:
                ys=[];ps=[];weights=[];by_market={}
                for market in ("US","KR"):
                    mask=eligible[label][market]
                    y=frames[label][market].y.astype(int).to_numpy()[mask]
                    p=probability[label][market][mask]
                    weight=np.full(len(y),TARGET[market]/len(frames[label][market])) if label=="v27" else np.ones(len(y))
                    by_market[market]=array_metrics(y,p,weight)
                    ys.append(y);ps.append(p);weights.append(weight)
                value=array_metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(weights))
                value["by_market"]=by_market;sets[label]=value
                passes.extend([value["balanced_accuracy"]>=.54,value["auc"]>=.56,
                               value["edge_vs_naive"]>=.02,
                               value["by_market"]["US"]["balanced_accuracy"]>=.51,
                               value["by_market"]["KR"]["balanced_accuracy"]>=.51])
            count=sum(passes)
            score=min(sets[label]["balanced_accuracy"] for label in SETS)+min(
                sets[label]["auc"] for label in SETS)+min(sets[label]["edge_vs_naive"] for label in SETS)
            rows.append({"policy_index":number,"markets":policy["markets"],
                         "eligibility":{"market":"KR","column":column,"sign":sign,
                                        "margin_weight":mw,"quantile":q},
                         "retention":retention,"expected_v27_n":expected,
                         "sets":sets,"passes":count,"score":float(score)})
    rows.sort(key=lambda row:(row["passes"],row["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"rows":rows[:500]}
    path=ROOT/"cache"/"v27_eligibility.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("ROWS",len(rows))
    for row in rows[:30]:print(json.dumps(row,ensure_ascii=False),flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
