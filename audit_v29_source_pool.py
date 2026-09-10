"""Outcome-free residual source inventory after the opened V30 audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def article_key(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def main() -> None:
    prior = pd.read_csv(
        DATA / "events_exact_v30.csv.gz", compression="gzip", dtype={"ticker": str, "article_id": str},
        usecols=lambda column: column in {
            "event_id", "market", "ticker", "event_time_utc", "article_id",
        },
    )
    prior["event_time_utc"] = pd.to_datetime(prior.event_time_utc, utc=True)
    prior_ids = set(prior.event_id.astype(str))
    prior_articles = set(prior.loc[
        prior.market.eq("KR") & prior.article_id.notna(), "article_id"
    ].map(article_key)) - {""}
    prior_times = {
        (market, str(ticker)): np.sort(group.event_time_utc.astype("int64").to_numpy())
        for (market, ticker), group in prior.groupby(["market", "ticker"])
    }
    embargo = int(pd.Timedelta(minutes=35).value)

    def read(paths: list[Path], market: str) -> pd.DataFrame:
        frames = []
        for path in paths:
            try:
                frame = pd.read_csv(path, dtype={"ticker": str, "article_id": str}, low_memory=False)
            except Exception as exc:
                if type(exc).__name__ != "EmptyDataError":
                    print("SKIP", path.name, type(exc).__name__, flush=True)
                continue
            required = {"event_id", "ticker", "event_time_utc"}
            if not required.issubset(frame.columns):
                continue
            frame = frame.copy()
            frame["origin_file"] = path.name
            if "article_id" not in frame:
                frame["article_id"] = ""
            frames.append(frame)
        output = pd.concat(frames, ignore_index=True, sort=False)
        output["market"] = market
        output["event_time_utc"] = pd.to_datetime(output.event_time_utc, utc=True, errors="coerce")
        output = output[output.event_time_utc.notna()].copy()
        output = output.drop_duplicates("event_id")
        return output

    def outside(frame: pd.DataFrame, market: str) -> pd.DataFrame:
        frame = frame[~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep = np.ones(len(frame), dtype=bool)
        for position, row in enumerate(frame.itertuples(index=False)):
            times = prior_times.get((market, str(row.ticker)))
            if times is None or not len(times):
                continue
            target = int(pd.Timestamp(row.event_time_utc).value)
            index = int(np.searchsorted(times, target))
            distances = []
            if index < len(times):
                distances.append(abs(int(times[index]) - target))
            if index:
                distances.append(abs(int(times[index - 1]) - target))
            if distances and min(distances) <= embargo:
                keep[position] = False
        return frame.iloc[np.flatnonzero(keep)].copy()

    all_files = list(DATA.glob("events_*.csv"))
    kr_paths = sorted(path for path in all_files if "naver" in path.name or "kind" in path.name)
    kr_paths += sorted((ROOT / "cache" / "kind_days").glob("*.csv"))
    us_paths = sorted(path for path in all_files if "sec" in path.name or path.name == "events_us_news_real.csv")
    kr = outside(read(kr_paths, "KR"), "KR")
    article = kr.article_id.map(article_key)
    kr = kr[~article.isin(prior_articles)].copy()
    local = kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr = kr[(local.dt.time >= pd.Timestamp("09:00").time()) &
            (local.dt.time <= pd.Timestamp("14:57").time())].copy()
    prior_kr_tickers = set(prior.loc[prior.market.eq("KR"), "ticker"].astype(str))
    kr_bio = kr[kr.ticker.astype(str).isin(prior_kr_tickers)].copy()

    us = outside(read(us_paths, "US"), "US")
    local = us.event_time_utc.dt.tz_convert(app.US_TZ)
    us = us[(local.dt.time >= pd.Timestamp("09:30").time()) &
            (local.dt.time <= pd.Timestamp("15:27").time())].copy()
    us["price_file"] = us.ticker.astype(str).map(
        lambda ticker: (ROOT / "price_data" / "US" / f"{ticker}.parquet").exists()
    )

    report = {
        "selection_uses_labels": False,
        "prior_version": "V30",
        "prior_event_ids": len(prior_ids),
        "prior_kr_articles": len(prior_articles),
        "kr_residual_regular": len(kr),
        "kr_residual_tickers": kr.ticker.astype(str).nunique(),
        "kr_prior_bio_tickers": len(prior_kr_tickers),
        "kr_residual_prior_bio": len(kr_bio),
        "kr_residual_prior_bio_tickers": kr_bio.ticker.astype(str).nunique(),
        "kr_prior_bio_by_year": kr_bio.event_time_utc.dt.year.value_counts().sort_index().to_dict(),
        "kr_by_origin": kr.groupby("origin_file").size().sort_values(ascending=False).head(30).to_dict(),
        "kr_by_year": kr.event_time_utc.dt.year.value_counts().sort_index().to_dict(),
        "us_residual_regular": len(us),
        "us_residual_with_price_file": int(us.price_file.sum()),
        "us_residual_tickers_with_price_file": us.loc[us.price_file, "ticker"].astype(str).nunique(),
        "us_by_origin": us.loc[us.price_file].groupby("origin_file").size().sort_values(ascending=False).head(30).to_dict(),
        "us_by_year_with_price": us.loc[us.price_file].event_time_utc.dt.year.value_counts().sort_index().to_dict(),
    }
    path = DATA / "V31_POOL_AUDIT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
