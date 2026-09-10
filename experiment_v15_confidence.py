"""V15 DEV-only entry-time high-confidence opportunity search."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def rank(values:np.ndarray)->np.ndarray:
    return pd.Series(values).rank(method="average",pct=True,na_option="bottom").to_numpy(float)


def main()->None:
    predictions=pd.read_parquet(app.CACHE/"v15_repeated_oof.parquet")
    selected=pd.concat([
        predictions[predictions.market.eq("US")&~predictions.augment&predictions.model.eq("native")],
        predictions[predictions.market.eq("KR")&predictions.augment&predictions.model.eq("logit001")],
    ],ignore_index=True)
    data=app.load_search_frame(app.Config())
    data=data[data.source.isin({"SEC_V15_DEV","KIND_V15_DEV"})&data.y.notna()].copy()
    features=app.model_frame(data);features["event_id"]=data.event_id.to_numpy()
    merged=selected.merge(features[["event_id"]+app.NUMERIC_COLS],on="event_id",how="left")
    cutoff=.53
    raw=merged.prob.to_numpy(float);direction=raw>=cutoff
    merged["signed_net"]=np.where(direction,1.0,-1.0)*merged.fwd_ret_30m.to_numpy(float)-.002
    merged["correct"]=direction==merged.y.to_numpy(int)
    rows=[]
    for market in ("US","KR"):
        part=merged[merged.market.eq(market)].reset_index(drop=True)
        part_raw=part.prob.to_numpy(float)
        margin=rank(np.abs(part_raw-cutoff))
        candidates={"model_margin":margin}
        for column in app.NUMERIC_COLS:
            values=pd.to_numeric(part[column],errors="coerce").to_numpy(float)
            if np.isfinite(values).sum()<max(50,len(part)//2):continue
            fill=np.nanmedian(values) if np.isfinite(values).any() else 0.0
            values=np.nan_to_num(values,nan=fill,posinf=fill,neginf=fill)
            transforms={f"+{column}":rank(values),f"-{column}":rank(-values),f"abs_{column}":rank(np.abs(values))}
            for name,opportunity in transforms.items():
                for weight in (0.0,.25,.5,.75):
                    candidates[f"{name}|m{weight}"]=weight*margin+(1-weight)*opportunity
        for name,confidence in candidates.items():
            confidence=rank(confidence)
            for coverage in (.15,.20,.25,.30):
                mask=confidence>=1.0-coverage
                rows.append({
                    "market":market,"name":name,"target_coverage":coverage,
                    "actual_coverage":float(mask.mean()),"n":int(mask.sum()),
                    "accuracy":float(part.loc[mask,"correct"].mean()),
                    "mean_signed_net":float(part.loc[mask,"signed_net"].mean()),
                })
    results=pd.DataFrame(rows)
    results["pass_count"]=(results.accuracy.ge(.60).astype(int)+results.mean_signed_net.gt(0).astype(int))
    for market in ("US","KR"):
        print("\nMARKET",market)
        print(results[results.market.eq(market)].sort_values(
            ["pass_count","mean_signed_net","accuracy"],ascending=False,
        ).head(50).to_string(index=False))
    results.to_csv(app.CACHE/"v15_confidence_search.csv",index=False)


if __name__=="__main__":
    main()
