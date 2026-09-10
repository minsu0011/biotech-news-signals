"""Label-blind inventory of fresh V20 candidate events."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import prepare_real_data as prep


def kr_candidates() -> pd.DataFrame:
    prior = pd.read_csv(
        prep.DATA / "events_exact_v19.csv.gz", dtype={"ticker": str},
        usecols=["event_id", "market", "ticker", "event_time_utc"],
    )
    prior["event_time_utc"] = pd.to_datetime(prior.event_time_utc, utc=True)
    prior_kr = prior[prior.market.eq("KR")].copy()
    tickers = set(prior_kr.ticker.astype(str))
    frames = []
    for path in sorted(prep.KIND_DAY_CACHE.glob("*.csv")):
        raw_date = path.name[:10]
        if raw_date < "2024-03-01" or raw_date > "2026-08-24":
            continue
        try:
            frame = pd.read_csv(path, dtype={"ticker": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        frame = frame[frame.ticker.astype(str).isin(tickers)]
        if not frame.empty:
            frames.append(frame)
    events = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("event_id")
    events = events[~events.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
    local = events.event_time_utc.dt.tz_convert(app.KR_TZ)
    events = events[(local.dt.time >= pd.Timestamp("09:00").time()) &
                    (local.dt.time <= pd.Timestamp("14:57").time()) &
                    events.headline.fillna("").str.len().ge(3)].copy()

    prior_times = {ticker: np.sort(group.event_time_utc.astype("int64").to_numpy())
                   for ticker, group in prior_kr.groupby("ticker")}
    embargo = int(pd.Timedelta(minutes=35).value)
    keep = []
    for row in events.itertuples(index=False):
        values = prior_times.get(str(row.ticker), np.empty(0, dtype=np.int64))
        target = int(pd.Timestamp(row.event_time_utc).value)
        index = int(np.searchsorted(values, target))
        nearby = []
        if index < len(values):
            nearby.append(abs(int(values[index]) - target))
        if index:
            nearby.append(abs(int(values[index - 1]) - target))
        keep.append(not nearby or min(nearby) > embargo)
    events = events.loc[np.asarray(keep)].sort_values(["ticker", "event_time_utc", "event_id"])

    # Greedily make candidate outcome windows mutually disjoint as well.
    accepted = []
    for _, group in events.groupby("ticker", sort=True):
        last = None
        for index, timestamp in zip(group.index, group.event_time_utc):
            if last is None or timestamp - last > pd.Timedelta(minutes=35):
                accepted.append(index)
                last = timestamp
    return events.loc[accepted].sort_values("event_time_utc")


def naver_candidates() -> pd.DataFrame:
    prior = pd.read_csv(
        prep.DATA / "events_exact_v19.csv.gz", dtype={"ticker": str},
        usecols=["event_id", "market", "ticker", "event_time_utc"],
    )
    prior["event_time_utc"] = pd.to_datetime(prior.event_time_utc, utc=True)
    prior_kr = prior[prior.market.eq("KR")].copy()
    paths = (
        "events_naver_news_real.csv", "events_naver_v7_h2_2025.csv",
        "events_naver_v8_unused.csv", "events_naver_v9_reserved.csv",
        "events_naver_v10_fresh.csv", "events_naver_v11_reserved.csv",
        "events_naver_v12_dev.csv", "events_naver_v12_seal.csv",
        "events_naver_v13_dev.csv", "events_naver_v13_seal.csv",
        "events_naver_v14_dev.csv", "events_naver_v14_seal.csv",
    )
    events = (pd.concat([pd.read_csv(prep.DATA / name, dtype={"ticker": str})
                         for name in paths if (prep.DATA / name).exists()],
                        ignore_index=True, sort=False)
              .drop_duplicates("event_id"))
    events = events[~events.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
    local = events.event_time_utc.dt.tz_convert(app.KR_TZ)
    events = events[(local.dt.time >= pd.Timestamp("09:00").time()) &
                    (local.dt.time <= pd.Timestamp("14:57").time()) &
                    events.headline.fillna("").str.len().ge(3)].copy()
    prior_times = {ticker: np.sort(group.event_time_utc.astype("int64").to_numpy())
                   for ticker, group in prior_kr.groupby("ticker")}
    embargo = int(pd.Timedelta(minutes=35).value)
    keep = []
    for row in events.itertuples(index=False):
        values = prior_times.get(str(row.ticker), np.empty(0, dtype=np.int64))
        target = int(pd.Timestamp(row.event_time_utc).value)
        index = int(np.searchsorted(values, target))
        nearby = []
        if index < len(values): nearby.append(abs(int(values[index]) - target))
        if index: nearby.append(abs(int(values[index - 1]) - target))
        keep.append(not nearby or min(nearby) > embargo)
    events = events.loc[np.asarray(keep)].sort_values(["ticker", "event_time_utc", "event_id"])
    accepted = []
    for _, group in events.groupby("ticker", sort=True):
        last = None
        for index, timestamp in zip(group.index, group.event_time_utc):
            if last is None or timestamp - last > pd.Timedelta(minutes=35):
                accepted.append(index); last = timestamp
    return events.loc[accepted].sort_values("event_time_utc")


def main() -> None:
    kr = kr_candidates()
    naver = naver_candidates()
    available = {path.stem for path in (prep.PRICE / "KR").glob("*.parquet")}
    print({
        "fresh_kr_candidates": len(kr),
        "tickers": kr.ticker.nunique(),
        "with_price_file": int(kr.ticker.astype(str).isin(available).sum()),
        "price_tickers": int(kr.loc[kr.ticker.astype(str).isin(available), "ticker"].nunique()),
        "start": str(kr.event_time_utc.min()),
        "end": str(kr.event_time_utc.max()),
    })
    print({
        "fresh_naver_candidates": len(naver),
        "tickers": naver.ticker.nunique(),
        "with_price_file": int(naver.ticker.astype(str).isin(available).sum()),
        "price_tickers": int(naver.loc[naver.ticker.astype(str).isin(available), "ticker"].nunique()),
        "start": str(naver.event_time_utc.min()), "end": str(naver.event_time_utc.max()),
    })


if __name__ == "__main__":
    main()
