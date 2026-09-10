"""Cross-version low-dimensional entry-feature rank search for V27."""
from __future__ import annotations

import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent
SOURCES={
    "US":{
        "v24":({"SEC_V24_DEV"},{"SEC_V24_SEAL"}),
        "v25":({"SEC_V25_DEV"},{"SEC_V25_SEAL"}),
        "v26":({"SEC_V26_DEV"},{"SEC_V26_SEAL"}),
        "v27":({"SEC_V27_DEV"},{"SEC_V27_DEV"}),
    },
    "KR":{
        "v24":({"NAVER_NEWS_V24_DEV"},{"NAVER_NEWS_V24_SEAL"}),
        "v25":({"NAVER_NEWS_V25_DEV"},{"NAVER_NEWS_V25_SEAL"}),
        "v26":({"NAVER_NEWS_V26_DEV"},{"NAVER_NEWS_V26_SEAL"}),
        "v27":({"NAVER_NEWS_V27_DEV"},{"NAVER_NEWS_V27_DEV"}),
    },
}
FEATURES=tuple(column for column in app.NUMERIC_COLS if column not in {
    "year","month","weekday","hour","minute","minutes_from_open","minutes_to_close",
})


def rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.sort(np.asarray(reference,float)[np.isfinite(reference)])
    fill=float(np.median(reference)) if len(reference) else 0.0
    values=np.nan_to_num(np.asarray(values,float),nan=fill,posinf=fill,neginf=fill)
    if not len(reference):return np.full(len(values),.5)
    left=np.searchsorted(reference,values,"left")
    right=np.searchsorted(reference,values,"right")
    return (left+right+1)/(2*len(reference))


def metric(y:np.ndarray,p:np.ndarray,cutoff:float)->dict:
    pred=p>=cutoff;up=float(y.mean());accuracy=float((pred==y).mean())
    return {"n":len(y),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "auc":float(roc_auc_score(y,p)),"accuracy":accuracy,
            "edge":accuracy-max(up,1-up),"pred_up":float(pred.mean())}


def best_row(market:str,name:str,components:list[tuple[str,int,float]],
             scores:dict[str,dict[str,np.ndarray]],ys:dict[str,np.ndarray])->dict:
    probabilities={label:sum(weight*(scores[label][column] if sign>0 else 1-scores[label][column])
                              for column,sign,weight in components)/sum(x[2] for x in components)
                   for label in scores}
    best=None
    for cutoff in np.arange(.30,.701,.01):
        sets={label:metric(ys[label],probability,float(cutoff))
              for label,probability in probabilities.items()}
        latest=(sets["v26"],sets["v27"]);older=(sets["v24"],sets["v25"])
        checks=sum(x["balanced_accuracy"]>=.54 for x in latest)+sum(
            x["auc"]>=.56 for x in latest)+sum(x["edge"]>=.02 for x in latest)+sum(
            x["balanced_accuracy"]>=.51 for x in older)+sum(x["auc"]>=.52 for x in older)
        score=(min(x["balanced_accuracy"] for x in latest)+
               min(x["auc"] for x in latest)+min(x["edge"] for x in latest)+
               .25*min(x["balanced_accuracy"] for x in older)+
               .25*min(x["auc"] for x in older))
        row={"market":market,"name":name,"components":components,"cutoff":float(cutoff),
             "sets":sets,"checks":checks,"score":float(score)}
        if best is None or (checks,score)>(best["checks"],best["score"]):best=row
    return best


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V27_SEAL","NAVER_NEWS_V27_SEAL"}),"y"].notna().any():
        raise RuntimeError("V27 seal exposed")
    opened=data[data.y.notna()].copy();all_rows=[]
    for market in ("US","KR"):
        scores={};ys={};sizes={}
        for label,(ref_sources,target_sources) in SOURCES[market].items():
            ref=app.model_frame(opened[opened.market.eq(market)&opened.source.isin(ref_sources)])
            target=opened[opened.market.eq(market)&opened.source.isin(target_sources)].copy()
            tf=app.model_frame(target);ys[label]=target.y.astype(int).to_numpy();sizes[label]=len(target)
            scores[label]={column:rank(pd.to_numeric(ref[column],errors="coerce").to_numpy(float),
                                       pd.to_numeric(tf[column],errors="coerce").to_numpy(float))
                           for column in FEATURES}
        print("DATA",market,sizes,flush=True)
        singles=[]
        for column,sign in itertools.product(FEATURES,(-1,1)):
            singles.append(best_row(market,f"{column}_{sign}",[(column,sign,1.0)],scores,ys))
        singles.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
        chosen=[]
        for row in singles:
            column,sign,_=row["components"][0]
            if column not in {item[0] for item in chosen}:chosen.append((column,sign))
            if len(chosen)>=16:break
        rows=list(singles)
        for (a,sa),(b,sb) in itertools.combinations(chosen,2):
            for weight in (.25,.5,.75):
                rows.append(best_row(market,f"pair_{a}_{b}_{weight}",
                                     [(a,sa,weight),(b,sb,1-weight)],scores,ys))
        for combo in itertools.combinations(chosen[:12],3):
            rows.append(best_row(market,"triple_"+"_".join(item[0] for item in combo),
                                 [(column,sign,1/3) for column,sign in combo],scores,ys))
        rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
        all_rows.extend(rows[:300])
        print("TOP",market,flush=True)
        for row in rows[:20]:print(json.dumps(row,ensure_ascii=False),flush=True)
    all_rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True)
    path=ROOT/"cache"/"v27_rank_search.json"
    path.write_text(json.dumps({"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
                                "rows":all_rows},ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path))


if __name__=="__main__":main()
