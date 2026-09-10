"""Opened V15 confidence-gate transfer search for the frozen V16 rank directions."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":[("intraday_ret_open",-1.0,1.0),("benchmark_ret_5",-1.0,1.0),
          ("range_60m",-1.0,1.0),("excess_ret_30",-1.0,1.0)],
    "KR":[("close_position_10",-1.0,.75),("excess_ret_5",-1.0,.25)],
}
CUTOFF={"US":.60,"KR":.76}


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref=np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ref,values,side="right")/len(ref)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    value=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    fill=np.nanmedian(value) if np.isfinite(value).any() else 0.0
    return np.nan_to_num(value,nan=fill,posinf=fill,neginf=fill)


def direction_score(frame: pd.DataFrame, market: str) -> np.ndarray:
    pieces=[];weights=[]
    for column,sign,weight in POLICY[market]:
        value=numeric(frame,column)
        pieces.append(sign*weight*rank(value,value));weights.append(weight)
    return 1.0+np.sum(pieces,axis=0)/sum(weights)


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
    rows=[]
    for market in ("US","KR"):
        part=frame[frame.market.eq(market)].reset_index(drop=True)
        direction=direction_score(part,market)
        margin_raw=np.abs(direction-CUTOFF[market])
        margin=rank(margin_raw,margin_raw)
        correct=(direction>=CUTOFF[market])==part.y.astype(int).to_numpy()
        signed_net=np.where(direction>=CUTOFF[market],1.0,-1.0)*part.fwd_ret_30m.to_numpy(float)-.002
        candidates={"margin":margin}
        for column in app.NUMERIC_COLS:
            value=numeric(part,column)
            transforms={f"+{column}":rank(value,value),f"-{column}":rank(-value,-value),
                        f"abs_{column}":rank(np.abs(value),np.abs(value))}
            for name,opportunity in transforms.items():
                for weight in (0.0,.25,.5,.75):
                    candidates[f"{name}|m{weight}"]=weight*margin+(1.0-weight)*opportunity
        dev=part.source.str.endswith("_DEV").to_numpy();seal=~dev
        for name,score_raw in candidates.items():
            score=rank(score_raw,score_raw)
            for coverage in (.15,.20,.25,.30,.35,.40):
                chosen=score>=1.0-coverage
                record={"market":market,"name":name,"coverage_target":coverage}
                for subset,mask in (("all",np.ones(len(part),dtype=bool)),("dev",dev),("seal",seal)):
                    selected=chosen&mask
                    record[f"{subset}_coverage"]=float(selected.sum()/mask.sum())
                    record[f"{subset}_n"]=int(selected.sum())
                    record[f"{subset}_accuracy"]=float(correct[selected].mean())
                    record[f"{subset}_net"]=float(signed_net[selected].mean())
                record["min_accuracy"]=min(record["dev_accuracy"],record["seal_accuracy"])
                record["min_net"]=min(record["dev_net"],record["seal_net"])
                rows.append(record)
    result=pd.DataFrame(rows)
    result.to_csv(app.CACHE/"v16_confidence.csv",index=False)
    for market in ("US","KR"):
        stable=result[result.market.eq(market)&result.dev_net.gt(0)&result.seal_net.gt(0)].copy()
        stable=stable.sort_values(["min_accuracy","min_net","all_accuracy"],ascending=False)
        print(f"\n{market} STABLE CONFIDENCE")
        print(stable.head(40).to_string(index=False))


if __name__=="__main__":
    main()
