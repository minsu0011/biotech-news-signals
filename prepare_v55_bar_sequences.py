"""Build causal 120-minute raw-bar tensors for the V55 temporal model."""
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
STEPS=120
FEATURES=("close_to_close_pct","bar_return_pct","range_pct","close_position",
          "log_volume_z","observed")
MINUTE_NS=60_000_000_000


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def sequence(bars:pd.DataFrame,entry:pd.Timestamp)->tuple[np.ndarray,dict[str,int|float]]:
    times=pd.to_datetime(bars.datetime,utc=True).astype("int64").to_numpy()
    entry_ns=int(entry.value);bar_end=times+MINUTE_NS
    right=np.searchsorted(bar_end,entry_ns,side="right")
    left=np.searchsorted(bar_end,entry_ns-STEPS*MINUTE_NS,side="left")
    index=np.arange(left,right,dtype=int)
    tensor=np.zeros((STEPS,len(FEATURES)),dtype=np.float32)
    if not len(index):
        return tensor,{"observed":0,"future_bar_violations":0,"staleness_sec":float("nan")}
    lag=((entry_ns-bar_end[index])//MINUTE_NS).astype(int)
    keep=(lag>=0)&(lag<STEPS);index=index[keep];lag=lag[keep]
    if not len(index):
        return tensor,{"observed":0,"future_bar_violations":0,"staleness_sec":float("nan")}
    o=pd.to_numeric(bars.open,errors="coerce").to_numpy(float)
    h=pd.to_numeric(bars.high,errors="coerce").to_numpy(float)
    l=pd.to_numeric(bars.low,errors="coerce").to_numpy(float)
    c=pd.to_numeric(bars.close,errors="coerce").to_numpy(float)
    v=pd.to_numeric(bars.volume,errors="coerce").to_numpy(float)
    previous=np.maximum(index-1,0);continuous=(times[index]-times[previous])<=MINUTE_NS
    cc=np.divide(c[index],c[previous],out=np.ones(len(index)),where=c[previous]>0)-1.
    cc[~continuous]=0.
    bar=np.divide(c[index],o[index],out=np.ones(len(index)),where=o[index]>0)-1.
    span=np.divide(h[index]-l[index],o[index],out=np.zeros(len(index)),where=o[index]>0)
    position=np.divide(c[index]-l[index],h[index]-l[index],out=np.full(len(index),.5),
                       where=(h[index]-l[index])>0)-.5
    log_volume=np.log1p(np.maximum(v[index],0.));finite=np.isfinite(log_volume)
    if finite.sum()>=3 and float(np.nanstd(log_volume[finite]))>0:
        log_volume=(log_volume-float(np.nanmean(log_volume[finite])))/float(np.nanstd(log_volume[finite]))
    else:
        log_volume=np.zeros(len(index))
    values=np.column_stack([100*cc,100*bar,100*span,position,log_volume,np.ones(len(index))])
    values=np.nan_to_num(values,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    positions=STEPS-1-lag
    # If duplicated timestamps exist, the later row wins deterministically.
    for destination,value in zip(positions,values):tensor[int(destination)]=value
    return tensor,{"observed":int(tensor[:,-1].sum()),
                   "future_bar_violations":int(np.sum(bar_end[index]>entry_ns)),
                   "staleness_sec":float((entry_ns-bar_end[index[-1]])/1e9)}


def main()->None:
    runtime_limits.configure();CACHE.mkdir(exist_ok=True)
    labeled=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
        dtype={"ticker":str},usecols=["event_id","market","ticker","actual_entry_time_utc"])
    labeled["actual_entry_time_utc"]=pd.to_datetime(labeled.actual_entry_time_utc,utc=True)
    cfg=replace(app.Config(),official_exact_contract=True,us_price_dir="price_data/US",
                kr_price_dir="price_data/KR_V36_1M")
    store=app.PriceStore(cfg);tensors=np.zeros((len(labeled),STEPS,len(FEATURES)),dtype=np.float32)
    audit=[];groups=list(labeled.groupby(["market","ticker"],sort=True))
    for done,((market,ticker),events) in enumerate(groups,1):
        bars=store.load(str(market),str(ticker))
        if bars is None or bars.empty:raise RuntimeError(f"V55 missing exact bars {market}/{ticker}")
        bars=bars.sort_values("datetime").reset_index(drop=True);metadata=dict(bars.attrs)
        if (not metadata.get("official_exact_eligible",False) or
                str(metadata.get("bar_timestamp_semantics","")).upper()!="OPEN" or
                int(metadata.get("price_cadence_sec",0))>60):
            raise RuntimeError(f"V55 ineligible exact metadata {market}/{ticker}: {metadata}")
        for row in events.itertuples():
            tensor,record=sequence(bars,row.actual_entry_time_utc);tensors[row.Index]=tensor
            record.update({"event_id":row.event_id,"market":market});audit.append(record)
        if done%50==0 or done==len(groups):
            print(f"[V55 SEQUENCE] {done}/{len(groups)} events={len(audit)}",flush=True)
    audit_frame=pd.DataFrame(audit)
    if len(audit_frame)!=len(labeled) or audit_frame.event_id.duplicated().any():
        raise RuntimeError("V55 sequence key failure")
    if int(audit_frame.future_bar_violations.sum())!=0:raise RuntimeError("V55 future bar violation")
    max_length=max(len(str(value)) for value in labeled.event_id)
    path=CACHE/"v55_bar_sequences.npz"
    np.savez_compressed(path,event_id=labeled.event_id.astype(str).to_numpy(dtype=f"U{max_length}"),
                        sequence=tensors)
    report={"rows":len(labeled),"steps":STEPS,"features":list(FEATURES),
        "future_prices_used":False,"future_bar_violations":0,
        "bar_rule":"OPEN timestamp + 60 seconds <= actual_entry; exact clock lag; no interpolation",
        "observed_min":int(audit_frame.observed.min()),
        "observed_median":float(audit_frame.observed.median()),
        "observed_full":int(audit_frame.observed.eq(STEPS).sum()),
        "staleness_max_sec":float(audit_frame.staleness_sec.max()),"cache_sha256":sha256(path)}
    (CACHE/"V55_BAR_SEQUENCE_STATUS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
