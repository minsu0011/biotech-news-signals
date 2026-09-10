"""Label-blind size audit for previously unused V26 candidate events."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import bio_news_30m_v3 as app

ROOT=Path(__file__).resolve().parent;DATA=ROOT/"data";PRICE=ROOT/"price_data"


def outside(frame:pd.DataFrame,prior:pd.DataFrame,market:str)->pd.DataFrame:
    x=frame.copy();x["event_time_utc"]=pd.to_datetime(x.event_time_utc,utc=True,errors="coerce")
    x=x[x.event_time_utc.notna()&~x.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    times={(str(ticker)):np.sort(g.event_time_utc.astype("int64").to_numpy())
           for ticker,g in prior[prior.market.eq(market)].groupby("ticker")}
    keep=[];limit=int(pd.Timedelta(minutes=35).value)
    for row in x.itertuples():
        a=times.get(str(row.ticker));ok=True
        if a is not None and len(a):
            target=int(pd.Timestamp(row.event_time_utc).value);i=int(np.searchsorted(a,target))
            near=[]
            if i<len(a):near.append(abs(int(a[i])-target))
            if i:near.append(abs(int(a[i-1])-target))
            ok=not near or min(near)>limit
        keep.append(ok)
    return x.iloc[np.flatnonzero(keep)].copy()


def space(frame:pd.DataFrame)->pd.DataFrame:
    accepted=[]
    for _,g in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
        last=None
        for i,t in zip(g.index,g.event_time_utc):
            if last is None or t-last>pd.Timedelta(minutes=35):accepted.append(i);last=t
    return frame.loc[accepted].copy()


def price_bounds(tickers)->dict:
    out={}
    for ticker in tickers:
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        pf=pq.ParquetFile(path)
        if "timestamp" not in pf.schema_arrow.names:continue
        ci=pf.schema_arrow.names.index("timestamp");lo=[];hi=[]
        for rg in range(pf.num_row_groups):
            s=pf.metadata.row_group(rg).column(ci).statistics
            if s is not None and s.has_min_max:lo.append(pd.Timestamp(s.min));hi.append(pd.Timestamp(s.max))
        if lo:
            a=min(lo);b=max(hi);a=a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
            b=b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC");out[str(ticker)]=(a,b)
    return out


def main():
    prior=pd.read_csv(DATA/"events_exact_v25.csv.gz",compression="gzip",usecols=["event_id","market","ticker","event_time_utc","article_id"],dtype={"ticker":str})
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    us_files=["events_sec_v14_unused_raw.csv","events_sec_v14_2009_2010_raw.csv",
              "events_sec_v14_2009_2012_raw.csv","events_sec_v13_2006_2007_raw.csv"]
    us=pd.concat([pd.read_csv(DATA/f,dtype={"ticker":str}) for f in us_files],ignore_index=True).drop_duplicates("event_id")
    us=outside(us,prior,"US");local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&(local.dt.time<=pd.Timestamp("15:27").time())])
    bounds=price_bounds(us.ticker.astype(str).unique())
    covered=np.asarray([str(r.ticker) in bounds and bounds[str(r.ticker)][0]<=r.event_time_utc and
                        bounds[str(r.ticker)][1]>=r.event_time_utc+pd.Timedelta(minutes=32) for r in us.itertuples()])
    us=us.iloc[np.flatnonzero(covered)].copy()
    kr=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str}).drop_duplicates("event_id")
    kr=outside(kr,prior,"KR")
    articles=set(prior.loc[prior.market.eq("KR")&prior.article_id.notna(),"article_id"].astype(str))
    kr=kr[~kr.article_id.astype(str).isin(articles)].copy();local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=space(kr[(local.dt.time>=pd.Timestamp("09:00").time())&(local.dt.time<=pd.Timestamp("14:57").time())])
    print("US",len(us),"tickers",us.ticker.nunique(),"range",us.event_time_utc.min(),us.event_time_utc.max())
    print("KR",len(kr),"tickers",kr.ticker.nunique(),"articles",kr.article_id.nunique(),"range",kr.event_time_utc.min(),kr.event_time_utc.max())
    print("overlap",len(set(us.event_id)&set(prior.event_id)),len(set(kr.event_id)&set(prior.event_id)),
          len(set(kr.article_id.astype(str))&articles))

if __name__=="__main__":main()
