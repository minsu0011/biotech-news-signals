"""Build point-in-time microstructure features from completed exact 1-minute bars."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
CACHE=ROOT/"cache"
WINDOWS=(3,5,10,20,30,60,120)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def finite_moment(values:np.ndarray,order:int)->float:
    values=values[np.isfinite(values)]
    if len(values)<max(4,order+1):return np.nan
    std=float(values.std())
    if std<=0:return 0.
    centered=(values-values.mean())/std
    return float(np.mean(centered**order))


def window_features(times:np.ndarray,open_:np.ndarray,high:np.ndarray,low:np.ndarray,
                    close:np.ndarray,volume:np.ndarray,entry_ns:int,window:int)->dict[str,float]:
    lower=entry_ns-int(pd.Timedelta(minutes=window).value)
    mask=(times+60_000_000_000>lower)&(times+60_000_000_000<=entry_ns)
    index=np.flatnonzero(mask);prefix=f"micro_{window}"
    names=("obs","rv","ret_mean","ret_std","ret_skew","ret_kurt","up_fraction","zero_fraction",
           "range_mean","range_max","signed_volume","volume_recent_ratio","drawdown","drawup","jump_count")
    if len(index)<2:return {f"{prefix}_{name}":np.nan for name in names}
    t=times[index];o=open_[index];h=high[index];l=low[index];c=close[index];v=volume[index]
    gaps=np.diff(t);raw=np.diff(c)/c[:-1];returns=raw[(gaps<=60_000_000_000)&np.isfinite(raw)]
    ranges=np.divide(h-l,o,out=np.full(len(o),np.nan),where=o>0)
    signed=np.sign(np.r_[0.,raw])*np.log1p(np.maximum(v,0.))
    recent=max(1,min(5,len(v)//2));older=v[:-recent];recent_volume=float(np.nanmean(v[-recent:]))
    old_volume=float(np.nanmedian(older)) if len(older) else np.nan
    cumulative=c/c[0]-1. if c[0]>0 else np.full(len(c),np.nan)
    std=float(np.nanstd(returns)) if len(returns) else np.nan
    return {f"{prefix}_obs":float(len(index)),
        f"{prefix}_rv":float(np.sqrt(np.nansum(returns**2))) if len(returns) else np.nan,
        f"{prefix}_ret_mean":float(np.nanmean(returns)) if len(returns) else np.nan,
        f"{prefix}_ret_std":std,f"{prefix}_ret_skew":finite_moment(returns,3),
        f"{prefix}_ret_kurt":finite_moment(returns,4),
        f"{prefix}_up_fraction":float(np.mean(returns>0)) if len(returns) else np.nan,
        f"{prefix}_zero_fraction":float(np.mean(returns==0)) if len(returns) else np.nan,
        f"{prefix}_range_mean":float(np.nanmean(ranges)),f"{prefix}_range_max":float(np.nanmax(ranges)),
        f"{prefix}_signed_volume":float(np.nanmean(signed)),
        f"{prefix}_volume_recent_ratio":recent_volume/old_volume if old_volume>0 else np.nan,
        f"{prefix}_drawdown":float(np.nanmin(cumulative)),f"{prefix}_drawup":float(np.nanmax(cumulative)),
        f"{prefix}_jump_count":float(np.sum(np.abs(returns)>2.5*std)) if std>0 else 0.}


def event_features(bars:pd.DataFrame,entry:pd.Timestamp)->dict[str,float]:
    times=pd.to_datetime(bars.datetime,utc=True).astype("int64").to_numpy();entry_ns=int(entry.value)
    right=np.searchsorted(times+60_000_000_000,entry_ns,side="right")
    left=np.searchsorted(times,entry_ns-int(pd.Timedelta(minutes=125).value),side="left")
    times=times[left:right];subset=bars.iloc[left:right]
    if not len(times):return {"micro_staleness_sec":np.nan}
    open_=pd.to_numeric(subset.open,errors="coerce").to_numpy(float)
    high=pd.to_numeric(subset.high,errors="coerce").to_numpy(float)
    low=pd.to_numeric(subset.low,errors="coerce").to_numpy(float)
    close=pd.to_numeric(subset.close,errors="coerce").to_numpy(float)
    volume=pd.to_numeric(subset.volume,errors="coerce").to_numpy(float)
    output={"micro_staleness_sec":float((entry_ns-(times[-1]+60_000_000_000))/1e9)}
    for window in WINDOWS:output.update(window_features(times,open_,high,low,close,volume,entry_ns,window))
    valid_return=np.diff(close)/close[:-1];valid_gap=np.diff(times)<=60_000_000_000
    returns=valid_return[valid_gap&np.isfinite(valid_return)]
    for lag in (1,2,5):
        if len(returns)>lag and np.std(returns[:-lag])>0 and np.std(returns[lag:])>0:
            output[f"micro_autocorr_{lag}"]=float(np.corrcoef(returns[:-lag],returns[lag:])[0,1])
        else:output[f"micro_autocorr_{lag}"]=np.nan
    return output


def main()->None:
    runtime_limits.configure();labeled=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",
        low_memory=False,dtype={"ticker":str},usecols=["event_id","market","ticker","actual_entry_time_utc"])
    labeled["actual_entry_time_utc"]=pd.to_datetime(labeled.actual_entry_time_utc,utc=True)
    cfg=replace(app.Config(),official_exact_contract=True,us_price_dir="price_data/US",
                kr_price_dir="price_data/KR_V36_1M")
    store=app.PriceStore(cfg);records=[]
    grouped=list(labeled.groupby(["market","ticker"],sort=True))
    for done,((market,ticker),events) in enumerate(grouped,1):
        bars=store.load(str(market),str(ticker))
        if bars is None or bars.empty:raise RuntimeError(f"missing exact bars {market}/{ticker}")
        bars=bars.sort_values("datetime").reset_index(drop=True)
        metadata=dict(bars.attrs)
        if (not metadata.get("official_exact_eligible",False) or
                str(metadata.get("bar_timestamp_semantics","")).upper()!="OPEN" or
                int(metadata.get("price_cadence_sec",0))>60):
            raise RuntimeError(f"ineligible exact metadata {market}/{ticker}: {metadata}")
        for row in events.itertuples(index=False):
            record={"event_id":row.event_id};record.update(event_features(bars,row.actual_entry_time_utc));records.append(record)
        if done%50==0 or done==len(grouped):print(f"[V44 MICRO] {done}/{len(grouped)} rows={len(records)}",flush=True)
    output=pd.DataFrame(records)
    if len(output)!=len(labeled) or output.event_id.duplicated().any():raise RuntimeError("V44 feature key failure")
    path=CACHE/"v44_microstructure_features.csv.gz";output.to_csv(path,index=False,compression="gzip")
    report={"rows":len(output),"features":len(output.columns)-1,"future_prices_used":False,
        "bar_rule":"OPEN timestamp + 60s <= actual_entry; clock windows; no interpolation",
        "feature_sha256":sha256(path),"coverage":{column:int(output[column].notna().sum())
                                                  for column in output if column!="event_id"}}
    (CACHE/"V44_MICROSTRUCTURE_STATUS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
