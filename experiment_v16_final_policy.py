"""Exact opened-V15 simulation of the intended V16 frozen probability policy."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":[("intraday_ret_open",-1.0,1.0),("benchmark_ret_5",-1.0,1.0),
          ("range_60m",-1.0,1.0),("excess_ret_30",-1.0,1.0)],
    "KR":[("close_position_10",-1.0,.75),("excess_ret_5",-1.0,.25)],
}
CUTOFF={"US":.60,"KR":.76}
TARGET={"US":501.0,"KR":136.0}


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref=np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ref,values,side="right")/len(ref)


def values(frame: pd.DataFrame, column: str) -> np.ndarray:
    x=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    fill=np.nanmedian(x) if np.isfinite(x).any() else 0.0
    return np.nan_to_num(x,nan=fill,posinf=fill,neginf=fill)


def transform(part: pd.DataFrame, market: str) -> pd.DataFrame:
    components=[];weights=[]
    for column,sign,weight in POLICY[market]:
        x=values(part,column)
        components.append(sign*weight*rank(x,x));weights.append(weight)
    direction=1.0+np.sum(components,axis=0)/sum(weights)
    margin_raw=np.abs(direction-CUTOFF[market])
    margin=rank(margin_raw,margin_raw)
    if market=="US":
        opportunity=rank(values(part,"range_30m"),values(part,"range_30m"))
        confidence_raw=.5*margin+.5*opportunity
        opportunity_score=rank(confidence_raw,confidence_raw)
        high=opportunity_score>=.75
    else:
        high=np.zeros(len(part),dtype=bool)
    confidence=np.where(high,.86+.14*margin,.84*margin)
    probability=np.where(
        direction>=CUTOFF[market],.5+.5*confidence,.5-.5*confidence
    )
    out=part.copy();out["prob"]=probability
    return out


def fast_boot_lower(frame: pd.DataFrame, nboot: int=3000) -> float:
    day=pd.to_datetime(frame.event_time_utc,utc=True).dt.strftime("%Y-%m-%d")
    codes,unique=pd.factorize(day)
    y=frame.y.astype(int).to_numpy();pred=frame.prob.ge(.5).to_numpy()
    base=frame.eval_weight.to_numpy(float);rng=np.random.default_rng(app.SEED+99);output=[]
    for _ in range(nboot):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        weight=base*counts[codes];pos=y==1;neg=~pos
        tpr=weight[pos&pred].sum()/weight[pos].sum();tnr=weight[neg&~pred].sum()/weight[neg].sum()
        output.append(.5*(tpr+tnr))
    return float(np.quantile(output,.025))


def main() -> None:
    raw=pd.read_csv(
        app.DATA/"labeled_events_v15.csv.gz",compression="gzip",dtype={"ticker":str},
        parse_dates=["event_time_utc"],low_memory=False,
    )
    raw=raw[raw.source.isin({"SEC_V15_DEV","SEC_V15_SEAL","KIND_V15_DEV","KIND_V15_SEAL"})].copy()
    frame=app.model_frame(raw)
    for column in raw.columns:
        if column not in frame.columns:
            frame[column]=raw[column].to_numpy()
    transformed=[]
    for market in ("US","KR"):
        part=transform(frame[frame.market.eq(market)].copy(),market)
        part["eval_weight"]=TARGET[market]/len(part)
        transformed.append(part)
    full=pd.concat(transformed,ignore_index=True)
    for subset,part in (
        ("ALL",full),
        ("DEV",full[full.source.str.endswith("_DEV")].copy()),
        ("SEAL",full[full.source.str.endswith("_SEAL")].copy()),
    ):
        # Restore the target market mixture inside each historical split.
        for market in ("US","KR"):
            mask=part.market.eq(market)
            part.loc[mask,"eval_weight"]=TARGET[market]/mask.sum()
        metrics=app.evaluate(part,.925,.002)
        print(json.dumps({
            "subset":subset,"metrics":metrics,"boot_lower":fast_boot_lower(part),
        },ensure_ascii=False))
    full.to_parquet(app.CACHE/"v16_final_policy.parquet",index=False)


if __name__=="__main__":
    main()
