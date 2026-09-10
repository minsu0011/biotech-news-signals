"""Exact V17 DEV simulation of direction and high-confidence probabilities."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":{
        "components":[("benchmark_ret_30",-1.0,1.0),("up_fraction_10",-1.0,1.0),("pre_ret_45",1.0,1.0)],
        "cutoff":.61,"opportunity":"abs_pre_ret_15","opportunity_sign":1.0,
        "margin_weight":.75,"gate":.70,
    },
    "KR":{
        "components":[("benchmark_ret_5",1.0,1.0),("pre_ret_90",-1.0,1.0),
                      ("entry_bar_ret",-1.0,1.0),("up_fraction_10",1.0,1.0)],
        "cutoff":.50,"opportunity":"excess_ret_5","opportunity_sign":1.0,
        "margin_weight":.75,"gate":.80,
    },
}
TARGET={"US":354.0,"KR":419.0}


def rank(reference: np.ndarray,values: np.ndarray) -> np.ndarray:
    ref=np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ref,values,side="right")/len(ref)


def numeric(frame: pd.DataFrame,column: str) -> np.ndarray:
    value=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    fill=np.nanmedian(value) if np.isfinite(value).any() else 0.0
    return np.nan_to_num(value,nan=fill,posinf=fill,neginf=fill)


def transform(frame: pd.DataFrame,market: str) -> pd.DataFrame:
    spec=POLICY[market];pieces=[];weights=[]
    for column,sign,weight in spec["components"]:
        value=numeric(frame,column);component=rank(value,value)
        pieces.append(weight*(component if sign>0 else 1.0-component));weights.append(weight)
    direction=np.sum(pieces,axis=0)/sum(weights)
    margin_raw=np.abs(direction-spec["cutoff"]);margin=rank(margin_raw,margin_raw)
    value=spec["opportunity_sign"]*numeric(frame,spec["opportunity"])
    opportunity=rank(value,value)
    confidence_raw=spec["margin_weight"]*margin+(1.0-spec["margin_weight"])*opportunity
    opportunity_score=rank(confidence_raw,confidence_raw)
    high=opportunity_score>=spec["gate"]
    confidence=np.where(high,.86+.14*margin,.84*margin)
    probability=np.where(direction>=spec["cutoff"],.5+.5*confidence,.5-.5*confidence)
    out=frame.copy();out["prob"]=probability
    return out


def fast_boot(frame: pd.DataFrame,nboot: int=3000) -> float:
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    codes,unique=pd.factorize(day);y=frame.y.astype(int).to_numpy();pred=frame.prob.ge(.5).to_numpy();base=frame.eval_weight.to_numpy(float)
    rng=np.random.default_rng(app.SEED+99);values=[]
    for _ in range(nboot):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique));weight=base*counts[codes]
        pos=y==1;neg=~pos;tpr=weight[pos&pred].sum()/weight[pos].sum();tnr=weight[neg&~pred].sum()/weight[neg].sum();values.append(.5*(tpr+tnr))
    return float(np.quantile(values,.025))


def main() -> None:
    data=app.load_search_frame(app.Config());parts=[]
    for market,source in (("US","SEC_V17_DEV"),("KR","KIND_V17_DEV")):
        raw=data[data.source.eq(source)&data.y.notna()].reset_index(drop=True);frame=app.model_frame(raw)
        for column in raw.columns:
            if column not in frame.columns:frame[column]=raw[column].to_numpy()
        part=transform(frame,market);part["audit_half"]=raw.event_id.astype(str).map(lambda value:int(hashlib.sha256(f"audit|{value}".encode()).hexdigest(),16)%2).to_numpy();parts.append(part)
    full=pd.concat(parts,ignore_index=True)
    for subset,part in (("ALL",full),("HALF0",full[full.audit_half.eq(0)].copy()),("HALF1",full[full.audit_half.eq(1)].copy())):
        for market in ("US","KR"):
            mask=part.market.eq(market);part.loc[mask,"eval_weight"]=TARGET[market]/mask.sum()
        metrics=app.evaluate(part,.925,.002)
        print(json.dumps({"subset":subset,"metrics":metrics,"boot_lower":fast_boot(part)},ensure_ascii=False))
    full.to_parquet(app.CACHE/"v17_final_policy.parquet",index=False)


if __name__=="__main__":
    main()
