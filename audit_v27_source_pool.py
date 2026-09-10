"""Label-blind audit of exact-time events unused through immutable V26."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import bio_news_30m_v3 as app

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
PRICE=ROOT/"price_data"


def outside(frame:pd.DataFrame,prior:pd.DataFrame,market:str)->pd.DataFrame:
    x=frame.copy()
    x["event_time_utc"]=pd.to_datetime(x.event_time_utc,utc=True,errors="coerce")
    prior_ids=set(prior.event_id.astype(str))
    x=x[x.event_time_utc.notna()&~x.event_id.astype(str).isin(prior_ids)].copy()
    times={str(ticker):np.sort(group.event_time_utc.astype("int64").to_numpy())
           for ticker,group in prior[prior.market.eq(market)].groupby("ticker")}
    keep=[]
    limit=int(pd.Timedelta(minutes=35).value)
    for row in x.itertuples(index=False):
        values=times.get(str(row.ticker));ok=True
        if values is not None and len(values):
            target=int(pd.Timestamp(row.event_time_utc).value)
            position=int(np.searchsorted(values,target));near=[]
            if position<len(values):near.append(abs(int(values[position])-target))
            if position:near.append(abs(int(values[position-1])-target))
            ok=not near or min(near)>limit
        keep.append(ok)
    return x.iloc[np.flatnonzero(keep)].copy()


def space(frame:pd.DataFrame)->pd.DataFrame:
    accepted=[]
    for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
        last=None
        for index,timestamp in zip(group.index,group.event_time_utc):
            if last is None or timestamp-last>pd.Timedelta(minutes=35):
                accepted.append(index);last=timestamp
    return frame.loc[accepted].copy()


def price_bounds(tickers)->dict[str,tuple[pd.Timestamp,pd.Timestamp]]:
    out={}
    for ticker in tickers:
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        pf=pq.ParquetFile(path)
        if "timestamp" not in pf.schema_arrow.names:continue
        column=pf.schema_arrow.names.index("timestamp");low=[];high=[]
        for group in range(pf.num_row_groups):
            stats=pf.metadata.row_group(group).column(column).statistics
            if stats is not None and stats.has_min_max:
                low.append(pd.Timestamp(stats.min));high.append(pd.Timestamp(stats.max))
        if low:
            lower=min(low);upper=max(high)
            lower=lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
            upper=upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
            out[str(ticker)]=(lower,upper)
    return out


def main()->None:
    prior=pd.read_csv(DATA/"events_exact_v26.csv.gz",dtype={"ticker":str},
                      usecols=["event_id","market","ticker","event_time_utc","article_id"])
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    us_files=sorted(DATA.glob("events_sec*_raw.csv"))
    us=pd.concat([pd.read_csv(path,dtype={"ticker":str}) for path in us_files],
                 ignore_index=True,sort=False).drop_duplicates("event_id")
    us=outside(us,prior,"US")
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    bounds=price_bounds(us.ticker.astype(str).unique())
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in us.itertuples(index=False)
    ],bool)
    us=us.iloc[np.flatnonzero(covered)].copy()

    kr=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str}).drop_duplicates("event_id")
    kr=outside(kr,prior,"KR")
    articles=set(prior.loc[prior.market.eq("KR")&prior.article_id.notna(),"article_id"].astype(str))
    kr=kr[~kr.article_id.astype(str).isin(articles)].copy()
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=space(kr[(local.dt.time>=pd.Timestamp("09:00").time())&
                (local.dt.time<=pd.Timestamp("14:57").time())].copy())
    print("US",len(us),"tickers",us.ticker.nunique(),"raw_files",len(us_files),
          "range",us.event_time_utc.min(),us.event_time_utc.max())
    print("KR",len(kr),"tickers",kr.ticker.nunique(),"articles",kr.article_id.nunique(),
          "range",kr.event_time_utc.min(),kr.event_time_utc.max())
    print("overlap",len(set(us.event_id)&set(prior.event_id)),
          len(set(kr.event_id)&set(prior.event_id)),
          len(set(kr.article_id.astype(str))&articles))


if __name__=="__main__":main()
