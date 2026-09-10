"""Extract pinned 1-minute sector ETFs and build entry-time-only V41 features."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_real_data as prep
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
CACHE=ROOT/"cache"
PRICE=ROOT/"price_data"/"US"
TICKERS=("XBI","IBB","XLV")
WINDOWS=(2,5,15,30,60)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def extract()->dict[str,Path]:
    months=sorted(path.stem.removeprefix("ohlcv_") for path in prep.HF_RAW_CACHE.glob("ohlcv_*.parquet")
                  if path.stat().st_size>1_000_000)
    selected={}
    with ThreadPoolExecutor(max_workers=min(8,runtime_limits.THREAD_COUNT)) as executor:
        futures={executor.submit(prep._filter_hf_month,month,set(TICKERS),False):month for month in months}
        for done,future in enumerate(as_completed(futures),1):
            selected[futures[future]]=future.result()
            if done%12==0 or done==len(futures):print(f"[V41 ETF] scan {done}/{len(futures)}",flush=True)
    frames=[pd.read_parquet(selected[month]) for month in months]
    combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    outputs={};PRICE.mkdir(parents=True,exist_ok=True)
    for ticker in TICKERS:
        frame=(combined[combined.ticker.astype(str).eq(ticker)].drop(columns="ticker")
               .sort_values("timestamp").drop_duplicates("timestamp"))
        path=PRICE/f"{ticker}.parquet";frame.to_parquet(path,index=False,compression="zstd");outputs[ticker]=path
        print(f"[V41 ETF] {ticker} rows={len(frame):,}",flush=True)
    return outputs


def returns_asof(entries:pd.Series,bars:pd.DataFrame,window:int)->np.ndarray:
    timestamp=pd.to_datetime(bars.timestamp,utc=True).astype("int64").to_numpy()
    bar_end=timestamp+60*1_000_000_000;close=pd.to_numeric(bars.close,errors="coerce").to_numpy(float)
    event=pd.to_datetime(entries,utc=True).astype("int64").to_numpy();current=np.searchsorted(bar_end,event,"right")-1
    target=event-int(pd.Timedelta(minutes=window).value);previous=np.searchsorted(bar_end,target,"right")-1
    values=np.full(len(entries),np.nan);safe_current=np.maximum(current,0);safe_previous=np.maximum(previous,0)
    valid=(current>=0)&(previous>=0)&((event-bar_end[safe_current])<=60*1_000_000_000)&(
        (target-bar_end[safe_previous])<=60*1_000_000_000)&(close[safe_current]>0)&(close[safe_previous]>0)
    values[valid]=close[safe_current[valid]]/close[safe_previous[valid]]-1.
    return values


def build_features(paths:dict[str,Path])->Path:
    labeled=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
                        usecols=["event_id","market","actual_entry_time_utc","pre_ret_2","pre_ret_5",
                                 "pre_ret_15","pre_ret_30","pre_ret_60","benchmark_ret_2",
                                 "benchmark_ret_5","benchmark_ret_15","benchmark_ret_30","benchmark_ret_60"])
    output=labeled[["event_id"]].copy();us=labeled.market.eq("US").to_numpy()
    for ticker,path in paths.items():
        bars=pd.read_parquet(path,columns=["timestamp","close"])
        for window in WINDOWS:
            values=np.full(len(labeled),np.nan)
            values[us]=returns_asof(labeled.loc[us,"actual_entry_time_utc"],bars,window)
            output[f"{ticker.lower()}_ret_{window}"]=values
            output[f"stock_minus_{ticker.lower()}_{window}"]=pd.to_numeric(
                labeled[f"pre_ret_{window}"],errors="coerce")-values
            output[f"{ticker.lower()}_minus_market_{window}"]=values-pd.to_numeric(
                labeled[f"benchmark_ret_{window}"],errors="coerce")
    path=CACHE/"v41_sector_features.csv.gz";output.to_csv(path,index=False,compression="gzip")
    report={"generated_from_open_stamped_pinned_1m":True,"future_prices_used":False,"rows":len(output),
            "feature_path":str(path.relative_to(ROOT)),"feature_sha256":sha256(path),
            "ETF_files":{ticker:{"path":str(file.relative_to(ROOT)),"sha256":sha256(file)}
                         for ticker,file in paths.items()},
            "coverage":{column:int(output[column].notna().sum()) for column in output if column!="event_id"}}
    (CACHE/"V41_SECTOR_FEATURE_STATUS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2));return path


def main()->None:
    runtime_limits.configure();paths=extract();build_features(paths)


if __name__=="__main__":main()
