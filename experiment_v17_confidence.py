"""V17 DEV-only high-confidence search with independent hash-half checks."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":[("benchmark_ret_30",-1.0,1.0),("up_fraction_10",-1.0,1.0),("pre_ret_45",1.0,1.0)],
    "KR":[("benchmark_ret_5",1.0,1.0),("pre_ret_90",-1.0,1.0),
          ("entry_bar_ret",-1.0,1.0),("up_fraction_10",1.0,1.0)],
}
CUTOFF={"US":.61,"KR":.50}


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average",pct=True,na_option="bottom").to_numpy(float)


def numeric(frame: pd.DataFrame,column: str) -> np.ndarray:
    value=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    fill=np.nanmedian(value) if np.isfinite(value).any() else 0.0
    return np.nan_to_num(value,nan=fill,posinf=fill,neginf=fill)


def direction(frame: pd.DataFrame,market: str) -> np.ndarray:
    pieces=[];weights=[]
    for column,sign,weight in POLICY[market]:
        component=rank(numeric(frame,column))
        pieces.append(weight*(component if sign>0 else 1.0-component));weights.append(weight)
    return np.sum(pieces,axis=0)/sum(weights)


def main() -> None:
    data=app.load_search_frame(app.Config());rows=[]
    for market,source in (("US","SEC_V17_DEV"),("KR","KIND_V17_DEV")):
        raw=data[data.source.eq(source)&data.y.notna()].reset_index(drop=True)
        frame=app.model_frame(raw)
        score=direction(frame,market);margin=rank(np.abs(score-CUTOFF[market]))
        correct=(score>=CUTOFF[market])==raw.y.astype(int).to_numpy()
        signed_net=np.where(score>=CUTOFF[market],1.0,-1.0)*raw.fwd_ret_30m.to_numpy(float)-.002
        half=raw.event_id.astype(str).map(
            lambda value:int(hashlib.sha256(f"audit|{value}".encode()).hexdigest(),16)%2
        ).to_numpy()
        candidates={"margin":margin}
        for column in app.NUMERIC_COLS:
            value=numeric(frame,column)
            transforms={f"+{column}":rank(value),f"-{column}":rank(-value),
                        f"abs_{column}":rank(np.abs(value))}
            for name,opportunity in transforms.items():
                for weight in (0.0,.25,.5,.75):
                    candidates[f"{name}|m{weight}"]=weight*margin+(1.0-weight)*opportunity
        for name,raw_confidence in candidates.items():
            confidence=rank(raw_confidence)
            for coverage in (.15,.20,.25,.30,.35,.40,.45,.50):
                chosen=confidence>=1.0-coverage
                row={"market":market,"name":name,"coverage_target":coverage}
                for subset,mask in (("all",np.ones(len(raw),dtype=bool)),("half0",half==0),("half1",half==1)):
                    selected=chosen&mask
                    row[f"{subset}_coverage"]=float(selected.sum()/mask.sum())
                    row[f"{subset}_n"]=int(selected.sum())
                    row[f"{subset}_accuracy"]=float(correct[selected].mean())
                    row[f"{subset}_net"]=float(signed_net[selected].mean())
                row["min_accuracy"]=min(row["half0_accuracy"],row["half1_accuracy"])
                row["min_net"]=min(row["half0_net"],row["half1_net"])
                rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(app.CACHE/"v17_confidence.csv",index=False)
    for market in ("US","KR"):
        stable=result[result.market.eq(market)&result.half0_net.gt(0)&result.half1_net.gt(0)]
        stable=stable.sort_values(["min_accuracy","min_net","all_accuracy"],ascending=False)
        print(f"\n{market} STABLE")
        print(stable.head(50).to_string(index=False))


if __name__=="__main__":
    main()
