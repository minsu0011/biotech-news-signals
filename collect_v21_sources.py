"""Collect label-blind V21 event candidates from new US/KR ticker pools."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import prepare_real_data as prep


def collect_kr(start:str="2025-08-25",end:str="2026-08-24",workers:int=8)->pd.DataFrame:
    universe=(pd.concat([
        pd.read_csv(prep.DATA/"universe_v15_kr.csv",dtype={"ticker":str}),
        pd.read_csv(prep.DATA/"universe_v17_kr.csv",dtype={"ticker":str}),
    ],ignore_index=True,sort=False).drop_duplicates("ticker"))
    universe["ticker"]=universe.ticker.astype(str).str.zfill(6)
    price_tickers={path.stem for path in (prep.PRICE/"KR").glob("*.parquet")}
    tickers=sorted(set(universe.loc[universe.ticker.isin(price_tickers),"ticker"]))
    rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prep._naver_ticker_news,ticker,start,False):ticker for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            ticker=futures[future]
            try:rows.extend(future.result())
            except Exception as exc:failures.append({"ticker":ticker,"error":str(exc)})
            if done%10==0 or done==len(futures):
                print(f"[V21 NAVER] {done}/{len(futures)} rows={len(rows)} failures={len(failures)}")
    if failures:
        prep._atomic_json(failures,prep.DATA/"V21_NAVER_FAILURES.json")
        raise RuntimeError(f"V21 Naver failures={len(failures)}")
    events=pd.DataFrame(rows).drop_duplicates("event_id")
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    upper=pd.Timestamp(end,tz=app.KR_TZ)+pd.Timedelta(days=1)
    events=events[events.event_time_utc<upper.tz_convert("UTC")].copy()
    prior=pd.read_csv(
        prep.DATA/"events_exact_v20.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior=prior[prior.market.eq("KR")].copy();prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    events=events[~events.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    local=events.event_time_utc.dt.tz_convert(app.KR_TZ)
    events=events[(local.dt.time>=pd.Timestamp("09:00").time())&
                  (local.dt.time<=pd.Timestamp("14:57").time())&
                  events.headline.fillna("").str.len().ge(3)].copy()
    prior_times={ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for ticker,group in prior.groupby("ticker")}
    embargo=int(pd.Timedelta(minutes=35).value);keep=[]
    for row in events.itertuples(index=False):
        values=prior_times.get(str(row.ticker),np.empty(0,dtype=np.int64));target=int(row.event_time_utc.value)
        position=int(np.searchsorted(values,target));near=[]
        if position<len(values):near.append(abs(int(values[position])-target))
        if position:near.append(abs(int(values[position-1])-target))
        keep.append(not near or min(near)>embargo)
    events=events.loc[np.asarray(keep)].sort_values(["ticker","event_time_utc","event_id"])
    accepted=[]
    for _,group in events.groupby("ticker"):
        last=None
        for index,timestamp in zip(group.index,group.event_time_utc):
            if last is None or timestamp-last>pd.Timedelta(minutes=35):
                accepted.append(index);last=timestamp
    events=events.loc[accepted].sort_values("event_time_utc").copy()
    events["source"]="NAVER_NEWS_V21_RAW"
    events["event_time_utc"]=events.event_time_utc.astype(str)
    prep._atomic_csv(events,prep.DATA/"events_naver_v21_new_tickers_raw.csv")
    prep._atomic_json({"selection_uses_labels":False,"eligible_tickers":len(tickers),
                       "events":len(events),"event_tickers":events.ticker.nunique(),
                       "prior_event_overlap":0,"outcome_embargo_minutes":35},
                      prep.DATA/"V21_KR_RAW_STATUS.json")
    print(f"[V21 NAVER RAW] events={len(events)} tickers={events.ticker.nunique()}")
    return events


if __name__=="__main__":collect_kr()
