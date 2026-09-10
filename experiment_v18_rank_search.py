"""Robust equal-rank direction search on V18 DEV only."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


FEATURES={
    "US":[
        ("excess_ret_30",-1),("pre_ret_30",-1),("pre_ret_10",-1),
        ("entry_bar_ret",-1),("pre_ret_45",-1),("excess_ret_5",-1),
    ],
    "KR":[
        ("close_position_60",1),("benchmark_ret_15",1),("abs_pre_ret_2",-1),
        ("abs_pre_ret_30",1),("log_entry_price",1),("benchmark_ret_30",1),
        ("benchmark_ret_60",1),("event_weekday",1),
    ],
}
TARGET={"US":395.0,"KR":633.0}


def rank(value: np.ndarray) -> np.ndarray:
    x=np.asarray(value,dtype=float);finite=x[np.isfinite(x)]
    fill=float(np.median(finite)) if len(finite) else 0.0
    x=np.nan_to_num(x,nan=fill,posinf=fill,neginf=fill);ref=np.sort(x)
    return np.searchsorted(ref,x,side="right")/len(ref)


def make_part(raw: pd.DataFrame,ranked: dict[str,np.ndarray],
              components: tuple[tuple[str,int],...],cutoff: float) -> pd.DataFrame:
    pieces=[]
    for column,sign in components:
        component=ranked[column]
        pieces.append(component if sign>0 else 1.0-component)
    direction=np.mean(pieces,axis=0);margin=rank(np.abs(direction-cutoff))
    confidence=.20+.60*margin
    probability=np.where(direction>=cutoff,.5+.5*confidence,.5-.5*confidence)
    out=raw.copy();out["prob"]=probability
    out["audit_half"]=out.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V18-AUDIT|{value}".encode()).hexdigest(),16)%2
    )
    return out


def subset_metrics(part: pd.DataFrame) -> dict:
    result={}
    for name,x in (("all",part),("half0",part[part.audit_half.eq(0)]),("half1",part[part.audit_half.eq(1)])):
        result[name]=app.evaluate(x,.925,.002)
    return result


def main() -> None:
    data=app.load_search_frame(app.Config());dev,_,_=app.split_dev_seal(data,app.Config())
    raw={
        market:dev[dev.market.eq(market)&dev.source.eq(source)&dev.y.notna()].reset_index(drop=True)
        for market,source in (("US","SEC_V18_DEV"),("KR","KIND_V18_DEV"))
    }
    ranked={}
    for market in ("US","KR"):
        frame=app.model_frame(raw[market])
        ranked[market]={
            column:rank(pd.to_numeric(frame[column],errors="coerce").to_numpy(float))
            for column,_ in FEATURES[market]
        }
    candidates={}
    for market in ("US","KR"):
        rows=[]
        for size in range(1,4):
            for components in itertools.combinations(FEATURES[market],size):
                for cutoff in np.arange(.35,.681,.02):
                    part=make_part(raw[market],ranked[market],components,float(cutoff));metrics=subset_metrics(part)
                    key=min(metrics[x]["balanced_accuracy"] for x in metrics)
                    auc=min(metrics[x]["auc"] for x in metrics)
                    edge=min(metrics[x]["edge_vs_naive"] for x in metrics)
                    score=3*key+2*auc+edge+metrics["all"]["balanced_accuracy"]
                    rows.append({"components":components,"cutoff":float(cutoff),"score":score,
                                 "min_bal":key,"min_auc":auc,"min_edge":edge,"metrics":metrics})
        rows.sort(key=lambda item:item["score"],reverse=True);candidates[market]=rows[:50]
        print(json.dumps({"market":market,"top":rows[:15]},default=list,indent=2))

    pairs=[]
    for us in candidates["US"]:
        us_part=make_part(raw["US"],ranked["US"],tuple(us["components"]),us["cutoff"])
        for kr in candidates["KR"]:
            kr_part=make_part(raw["KR"],ranked["KR"],tuple(kr["components"]),kr["cutoff"])
            full=pd.concat([us_part,kr_part],ignore_index=True)
            metrics={}
            for name,x in (("all",full),("half0",full[full.audit_half.eq(0)]),("half1",full[full.audit_half.eq(1)])):
                for market in ("US","KR"):
                    mask=x.market.eq(market);x.loc[mask,"eval_weight"]=TARGET[market]/mask.sum()
                metrics[name]=app.evaluate(x,.925,.002)
            checks=[]
            for value in metrics.values():
                checks += [value["balanced_accuracy"]-.54,value["auc"]-.56,
                           value["edge_vs_naive"]-.02,
                           value["by_market"]["US"]["balanced_accuracy"]-.51,
                           value["by_market"]["KR"]["balanced_accuracy"]-.51]
            pairs.append({"us":{"components":us["components"],"cutoff":us["cutoff"]},
                          "kr":{"components":kr["components"],"cutoff":kr["cutoff"]},
                          "min_margin":min(checks),"pass_count":sum(v>=0 for v in checks),
                          "metrics":metrics})
    pairs.sort(key=lambda item:(item["pass_count"],item["min_margin"]),reverse=True)
    print(json.dumps({"pairs":pairs[:25]},default=list,indent=2))


if __name__=="__main__":main()
