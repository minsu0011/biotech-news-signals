#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproducible real-data preparation for BIO_NEWS_30M.

Sources
-------
* US events: SEC EDGAR ``acceptanceDateTime`` and filing exhibits.
* KR events: KRX KIND date-specific disclosure pages, including release time.
* US bars: ``ggaddam/OHLCV-1m`` monthly Parquet files on Hugging Face.
* KR bars: Daum Finance 5-minute chart endpoint.

The script deliberately stops the source period at 2025-05-31 because that is
the last complete month in the pinned US minute-bar dataset.  This keeps the US
and KR seal populations aligned in calendar time.
"""

from __future__ import annotations

import runtime_limits

import argparse
import hashlib
import html
import json
import re
import sys
import threading
import time
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
PRICE = ROOT / "price_data"
KIND_DAY_CACHE = CACHE / "kind_days"
SEC_TICKER_CACHE = CACHE / "sec_events_by_ticker"
HF_RAW_CACHE = CACHE / "hf_months"
HF_FILTERED_CACHE = CACHE / "hf_filtered"
NAVER_TICKER_CACHE = CACHE / "naver_news_by_ticker"

for directory in (
    DATA, CACHE, PRICE / "US", PRICE / "KR", KIND_DAY_CACHE,
    SEC_TICKER_CACHE, HF_RAW_CACHE, HF_FILTERED_CACHE, NAVER_TICKER_CACHE,
):
    directory.mkdir(parents=True, exist_ok=True)


DEFAULT_US_START = "2023-01-01"
DEFAULT_KR_START = "2024-03-01"
DEFAULT_END = "2025-05-31"
QIU_NEWS = CACHE / "sources" / "QiuStockNews" / "json_files" / "news.json"
HF_REPO = "ggaddam/OHLCV-1m"
HF_REVISION = "main"
KIND_URL = "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
DAUM_URL = "https://finance.daum.net/api/charts/A{ticker}/5/minutes"
NAVER_NEWS_URL = "https://m.stock.naver.com/api/news/stock/{ticker}"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BIO_NEWS_30M research)",
    "Accept": "application/json, text/plain, */*",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _atomic_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _dates(start: str, end: str) -> Iterable[date]:
    cur = pd.Timestamp(start).date()
    last = pd.Timestamp(end).date()
    while cur <= last:
        yield cur
        cur += timedelta(days=1)


def _request_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> requests.Response:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {error}")


def _parse_kind_html(raw: bytes, day: date, market_type: str) -> pd.DataFrame:
    # KIND currently emits UTF-8.  Parsing from bytes lets BeautifulSoup honor
    # the response's own declaration if the site changes it later.
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    table = soup.find("table", class_="list type-00 mt10") or soup.find("table")
    columns = [
        "event_id", "market", "ticker", "company", "event_time_utc",
        "source", "form", "headline", "body", "event_type",
        "timestamp_quality", "url", "kind_market_type", "kind_code_format",
    ]
    if table is None or table.find("tbody") is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        time_text = cells[0].get_text(" ", strip=True)
        company_link = cells[1].find("a")
        onclick = company_link.get("onclick", "") if company_link else ""
        code_match = re.search(r"companysummary_open\('([0-9A-Za-z]+)'", onclick)
        if not code_match:
            continue
        raw_code = code_match.group(1)
        # KIND's companysummary_open uses the five-character issue prefix;
        # listed common shares append the issue digit ``0`` (e.g. 00593 -> 005930).
        ticker = raw_code + "0" if len(raw_code) == 5 else app.norm_ticker(raw_code, "KR")
        company = company_link.get_text(" ", strip=True) if company_link else ""
        title_link = cells[2].find("a")
        headline = (
            title_link.get("title", "").strip()
            if title_link is not None
            else cells[2].get_text(" ", strip=True)
        )
        viewer_onclick = title_link.get("onclick", "") if title_link else ""
        accession_match = re.search(r"openDisclsViewer\('(\d+)'", viewer_onclick)
        accession = accession_match.group(1) if accession_match else ""
        url = ""
        if accession:
            url = (
                "https://kind.krx.co.kr/common/disclsviewer.do?"
                f"method=search&acptno={accession}&docno=&viewerhost=&viewerport="
            )
        try:
            local = pd.Timestamp(f"{day.isoformat()} {time_text}", tz=app.KR_TZ)
        except Exception:
            continue
        utc = local.tz_convert("UTC")
        rows.append({
            "event_id": f"KIND:{accession or ticker + ':' + utc.isoformat()}",
            "market": "KR",
            "ticker": ticker,
            "company": company,
            "event_time_utc": utc.isoformat(),
            "source": "KIND",
            "form": "",
            "headline": headline,
            "body": "",
            "event_type": app.infer_event_type(headline),
            "timestamp_quality": "EXACT",
            "url": url,
            "kind_market_type": market_type,
            "kind_code_format": "issue6",
        })
    return pd.DataFrame(rows).reindex(columns=columns)


def _kind_day(day: date, market_type: str, force: bool = False) -> pd.DataFrame:
    path = KIND_DAY_CACHE / f"{day.isoformat()}_{market_type}.csv"
    if path.exists() and not force:
        if path.stat().st_size <= 5:
            return pd.DataFrame()
        return pd.read_csv(path, dtype={"ticker": str})
    params = {
        "method": "searchTodayDisclosureSub",
        "forward": "todaydisclosure_sub",
        "currentPageSize": "5000",
        "pageIndex": "1",
        "orderMode": "0",
        "orderStat": "D",
        "chose": "S",
        "todayFlag": "Y",
        "marketType": market_type,
        "selDate": day.isoformat(),
    }
    response = _request_with_retry(
        "POST",
        KIND_URL,
        params=params,
        headers={**HTTP_HEADERS, "Referer": "https://kind.krx.co.kr/"},
        timeout=45,
    )
    frame = _parse_kind_html(response.content, day, market_type)
    _atomic_csv(frame, path)
    return frame


def collect_kind(start: str, end: str, workers: int = 6, force: bool = False) -> pd.DataFrame:
    seed = app.load_seed_universe()
    tickers = set(seed.loc[seed["market"].eq("KR"), "ticker"])
    jobs = [(day, market) for day in _dates(start, end) for market in ("1", "2")]
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    lock = threading.Lock()

    def run(job: tuple[date, str]) -> pd.DataFrame:
        day, market = job
        return _kind_day(day, market, force=force)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                with lock:
                    failures.append(f"{job[0]} market={job[1]}: {exc}")
            if done % 100 == 0 or done == len(jobs):
                print(f"[KIND] {done}/{len(jobs)} requests, failures={len(failures)}")

    if failures:
        _atomic_json(failures, DATA / "KIND_FETCH_FAILURES.json")
        raise RuntimeError(f"KIND collection incomplete: {len(failures)} failed requests")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        raise RuntimeError("KIND returned no disclosures")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    # Repair caches written by the V3 adapter, which incorrectly left-padded
    # KIND's five-character issue prefix instead of appending the issue digit.
    legacy = (
        out["kind_code_format"].ne("issue6")
        if "kind_code_format" in out.columns
        else pd.Series(True, index=out.index)
    )
    out.loc[legacy, "ticker"] = out.loc[legacy, "ticker"].map(
        lambda value: value[1:] + "0" if len(value) == 6 and value.startswith("0") else value
    )
    out = out[out["ticker"].isin(tickers)].copy()
    combined_path = DATA / "events_kind_real.csv"
    if combined_path.exists() and not force:
        previous = pd.read_csv(combined_path, dtype={"ticker": str})
        out = pd.concat([previous, out], ignore_index=True, sort=False)
    out = out.drop_duplicates("event_id").sort_values("event_time_utc")
    _atomic_csv(out, combined_path)
    material = out[
        [app.is_material(f"{h} {b}", f) for h, b, f in zip(
            out.headline.fillna(""), out.body.fillna(""), out.form.fillna("")
        )]
    ].copy()
    local = pd.to_datetime(material.event_time_utc, utc=True).dt.tz_convert(app.KR_TZ)
    regular = material[(local.dt.time >= pd.Timestamp("09:00").time()) &
                       (local.dt.time <= pd.Timestamp("14:57").time())]
    print(
        f"[KIND] seed events={len(out)}, material={len(material)}, "
        f"regular/exit-eligible={len(regular)}, tickers={regular.ticker.nunique()}"
    )
    return out


def _sec_ticker(
    collector: app.SECCollector,
    ticker: str,
    company: str,
    cik: int,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    force: bool,
    source_name: str = "SEC",
    preselect_one_per_day: bool = False,
    preselect_max: int | None = None,
    excluded_event_ids: set[str] | None = None,
    excluded_times: np.ndarray | None = None,
    exclude_minutes: int = 0,
    fetch_body: bool = True,
) -> list[dict[str, Any]]:
    cache_key=(
        f"{ticker}_{source_name}_{start.strftime('%Y%m%d')}_"
        f"{(end_exclusive-pd.Timedelta(days=1)).strftime('%Y%m%d')}"
    )
    path = SEC_TICKER_CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]+','_',cache_key)}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    submissions = collector.get_json(collector.SUBMISSIONS.format(cik=cik))
    filings=submissions.get("filings",{})
    filing_sets=[filings.get("recent",{})]
    # The main submissions object only keeps recent filings.  SEC lists older
    # blocks under filings.files; load only blocks intersecting the requested
    # window so historical seals retain exact acceptanceDateTime values.
    for descriptor in filings.get("files",[]) or []:
        try:
            block_start=pd.Timestamp(descriptor.get("filingFrom"),tz="UTC")
            block_end=pd.Timestamp(descriptor.get("filingTo"),tz="UTC")+pd.Timedelta(days=1)
        except Exception:
            continue
        if block_end<=start or block_start>=end_exclusive:
            continue
        name=str(descriptor.get("name","")).strip()
        if name:
            filing_sets.append(collector.get_json(
                f"https://data.sec.gov/submissions/{name}"
            ))
    # Apply label-independent concentration controls to filing metadata before
    # expensive primary/exhibit downloads.  Keep twice the final cap so the
    # later material-text filter still has headroom.
    eligible_metadata:list[tuple[str,pd.Timestamp]]=[]
    metadata_seen:set[str]=set()
    for filing_set in filing_sets:
        accessions=filing_set.get("accessionNumber",[]) or []
        for index,raw_accession in enumerate(accessions):
            accession=str(raw_accession)
            if accession in metadata_seen:
                continue
            metadata_seen.add(accession)
            forms=filing_set.get("form",[]) or []
            form=str(forms[index]) if index<len(forms) else ""
            if form not in collector.cfg.sec_forms:
                continue
            accepted=filing_set.get("acceptanceDateTime",[]) or []
            raw_time=accepted[index] if index<len(accepted) else ""
            if not raw_time or "T" not in str(raw_time):
                continue
            timestamp=pd.Timestamp(str(raw_time))
            if timestamp.tzinfo is None:
                timestamp=timestamp.tz_localize(app.US_TZ).tz_convert("UTC")
            else:
                timestamp=timestamp.tz_convert("UTC")
            if timestamp<start or timestamp>=end_exclusive:
                continue
            event_id=f"SEC:{cik}:{accession}"
            if excluded_event_ids is not None and event_id in excluded_event_ids:
                continue
            if excluded_times is not None and len(excluded_times) and exclude_minutes>0:
                target=int(timestamp.value)
                position=int(np.searchsorted(excluded_times,target))
                nearby=[]
                if position<len(excluded_times):
                    nearby.append(abs(int(excluded_times[position])-target))
                if position:
                    nearby.append(abs(int(excluded_times[position-1])-target))
                if nearby and min(nearby)<=int(pd.Timedelta(minutes=exclude_minutes).value):
                    continue
            local,open_time,close_time=app.session_bounds(timestamp,"US")
            if not (open_time<=local<=close_time):
                continue
            if local+pd.Timedelta(minutes=33)>close_time:
                continue
            eligible_metadata.append((accession,timestamp))
    if preselect_one_per_day:
        by_day:dict[str,tuple[str,pd.Timestamp]]={}
        for accession,timestamp in sorted(eligible_metadata,key=lambda item:(item[1],item[0])):
            day=timestamp.tz_convert(app.US_TZ).strftime("%Y-%m-%d")
            by_day.setdefault(day,(accession,timestamp))
        eligible_metadata=list(by_day.values())
    if preselect_max is not None and len(eligible_metadata)>preselect_max:
        eligible_metadata=sorted(
            eligible_metadata,
            key=lambda item:hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
        )[:preselect_max]
    allowed_accessions={accession for accession,_ in eligible_metadata}
    rows: list[dict[str, Any]] = []
    seen_accessions:set[str]=set()
    for filing_set in filing_sets:
        accessions=filing_set.get("accessionNumber",[]) or []
        for index, accession in enumerate(accessions):
            accession=str(accession)
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)
            if accession not in allowed_accessions:
                continue
            form = str(filing_set.get("form", [""] * len(accessions))[index])
            if form not in collector.cfg.sec_forms:
                continue
            accepted = filing_set.get("acceptanceDateTime", [])
            raw_time = accepted[index] if index < len(accepted) else ""
            if not raw_time or "T" not in str(raw_time):
                continue
            timestamp = pd.Timestamp(str(raw_time))
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(app.US_TZ).tz_convert("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            if timestamp < start or timestamp >= end_exclusive:
                continue
            local, open_time, close_time = app.session_bounds(timestamp, "US")
            if not (open_time <= local <= close_time):
                continue
            if local + pd.Timedelta(minutes=33) > close_time:
                continue
            items = filing_set.get("items", [])
            item_text = str(items[index] or "") if index < len(items) else ""
            primary_docs = filing_set.get("primaryDocument", [])
            primary = str(primary_docs[index] or "") if index < len(primary_docs) else ""
            body = collector.filing_text(cik, accession, primary, form) if fetch_body else ""
            headline = f"{form} {item_text}".strip()
            text = f"{headline} {body}".strip()
            if collector.cfg.material_only and not app.is_material(text, form):
                continue
            rows.append({
                "event_id": f"SEC:{cik}:{accession}",
                "market": "US",
                "ticker": ticker,
                "company": company or str(submissions.get("name", "")),
                "event_time_utc": timestamp.isoformat(),
                "source": source_name,
                "form": form,
                "headline": headline,
                "body": body,
                "event_type": app.infer_event_type(text, form),
                "timestamp_quality": "EXACT",
                "url": (
                    "https://www.sec.gov/Archives/edgar/data/"
                    f"{cik}/{str(accession).replace('-', '')}/"
                ),
            })
    _atomic_json(rows, path)
    return rows


def collect_sec(start: str, end: str, force: bool = False,
                output_name: str = "events_sec_real.csv",
                source_name: str = "SEC",
                one_per_ticker_day: bool = False,
                max_events_per_ticker: int | None = None,
                exclude_event_path: Path | None = None,
                exclude_minutes: int = 0,
                universe_path: Path | None = None,
                material_only: bool = True,
                fetch_body: bool = True,
                sec_forms: tuple[str,...] | None = None) -> pd.DataFrame:
    cfg = replace(
        app.Config(),sec_start=start,material_only=material_only,
        **({"sec_forms":sec_forms} if sec_forms is not None else {}),
    )
    collector = app.SECCollector(cfg)
    universe = (
        pd.read_csv(universe_path,dtype={"ticker":str})
        if universe_path is not None else app.load_seed_universe()
    )
    universe = universe[universe.market.eq("US")].copy()
    ticker_map = collector.ticker_to_cik()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_exclusive = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    excluded_event_ids:set[str]=set()
    excluded_times_by_ticker:dict[str,np.ndarray]={}
    if exclude_event_path is not None:
        excluded=pd.read_csv(
            exclude_event_path,dtype={"ticker":str},
            usecols=["event_id","market","ticker","event_time_utc"],
        )
        excluded=excluded[excluded.market.eq("US")].copy()
        excluded["event_time_utc"]=pd.to_datetime(excluded.event_time_utc,utc=True)
        excluded_event_ids=set(excluded.event_id.astype(str))
        excluded_times_by_ticker={
            ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
            for ticker,group in excluded.groupby("ticker")
        }
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for done, row in enumerate(universe.itertuples(index=False), 1):
        ticker = str(row.ticker)
        cik = ticker_map.get(ticker)
        if cik is None:
            missing.append(ticker)
            continue
        try:
            all_rows.extend(_sec_ticker(
                collector, ticker, str(row.company), cik,
                start_ts, end_exclusive, force, source_name,
                preselect_one_per_day=one_per_ticker_day,
                preselect_max=(max_events_per_ticker*2
                               if max_events_per_ticker is not None else None),
                excluded_event_ids=excluded_event_ids,
                excluded_times=excluded_times_by_ticker.get(ticker),
                exclude_minutes=exclude_minutes,
                fetch_body=fetch_body,
            ))
        except Exception as exc:
            print(f"[SEC WARN] {ticker}: {exc}")
        if done % 10 == 0 or done == len(universe):
            print(f"[SEC] {done}/{len(universe)} tickers, events={len(all_rows)}")
    if not all_rows:
        raise RuntimeError("SEC returned no in-session material events")
    out = pd.DataFrame(all_rows).drop_duplicates("event_id")
    if one_per_ticker_day:
        out["_event_time"]=pd.to_datetime(out.event_time_utc,utc=True)
        out["_event_day"]=out._event_time.dt.strftime("%Y-%m-%d")
        out=(out.sort_values(["ticker","_event_day","_event_time","event_id"])
             .drop_duplicates(["ticker","_event_day"],keep="first")
             .drop(columns=["_event_time","_event_day"]))
    if max_events_per_ticker is not None:
        if max_events_per_ticker<1:
            raise ValueError("max_events_per_ticker must be positive")
        out["_event_hash"]=out.event_id.astype(str).map(
            lambda value:hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        out=(out.sort_values(["ticker","_event_hash"])
             .groupby("ticker",group_keys=False).head(max_events_per_ticker)
             .drop(columns="_event_hash"))
    out=out.sort_values("event_time_utc")
    _atomic_csv(out, DATA / output_name)
    _atomic_json({"missing_cik": missing}, DATA / "SEC_COLLECTION_STATUS.json")
    print(f"[SEC] events={len(out)}, tickers={out.ticker.nunique()}, missing CIK={len(missing)}")
    return out


def prepare_v15_us_universe(max_tickers: int = 500) -> pd.DataFrame:
    """Build a label-blind expansion universe from SEC company names."""
    collector=app.SECCollector(app.Config())
    raw=collector.get_json(collector.TICKER_MAP)
    rows=[]
    pattern=re.compile(
        r"(?:BIO|PHARMA|THERAPEUT|MEDIC|HEALTH|LIFE SCI|ONCO|GENET|GENOMIC|"
        r"VACCIN|DIAGNOST|NEURO|IMMUN|OPHTHAL|CARDIO|SURGICAL|LABORATOR)",
        re.IGNORECASE,
    )
    for item in raw.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        primary_symbol=bool(re.fullmatch(r"[A-Z]{1,5}",ticker)) and not ticker.endswith(("W","P"))
        if primary_symbol and company and pattern.search(company):
            rows.append({"market":"US","ticker":ticker,"company":company})
    universe=pd.DataFrame(rows).drop_duplicates("ticker")
    seed=app.load_seed_universe()
    prior=pd.read_csv(
        DATA/"events_exact_v14.csv.gz",dtype={"ticker":str},
        usecols=["market","ticker"],
    )
    excluded=set(seed.loc[seed.market.eq("US"),"ticker"].astype(str))
    excluded.update(prior.loc[prior.market.eq("US"),"ticker"].astype(str))
    universe=universe[~universe.ticker.isin(excluded)].copy()
    universe["selection_hash"]=universe.ticker.map(
        lambda value:hashlib.sha256(f"V15|{value}".encode("utf-8")).hexdigest()
    )
    universe=(universe.sort_values("selection_hash").head(max_tickers)
              .drop(columns="selection_hash").reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v15_us.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"SEC company title life-science keyword; SHA256 ticker sample",
        "excluded_prior_tickers":len(excluded),
        "selected_tickers":len(universe),
    },DATA/"V15_US_UNIVERSE_STATUS.json")
    print(f"[V15 US UNIVERSE] selected={len(universe)}, excluded_prior={len(excluded)}")
    return universe


def prepare_v15_kr_expanded_source() -> pd.DataFrame:
    """Extract unused KIND events for Naver health-industry peer tickers."""
    industry_codes=("261","286","281","262")
    headers={**HTTP_HEADERS,"Referer":"https://m.stock.naver.com/"}
    universe_rows=[]
    for industry in industry_codes:
        for page in range(1,8):
            response=_request_with_retry(
                "GET",f"https://m.stock.naver.com/api/stocks/industry/{industry}",
                params={"page":page,"pageSize":100},headers=headers,timeout=60,
            )
            stocks=response.json().get("stocks",[]) or []
            for item in stocks:
                ticker=str(item.get("itemCode","")).strip()
                if re.fullmatch(r"\d{6}",ticker):
                    universe_rows.append({
                        "market":"KR","ticker":ticker,
                        "company":str(item.get("stockName","")).strip(),
                        "industry_code":industry,
                    })
            if len(stocks)<100:
                break
    universe=pd.DataFrame(universe_rows).drop_duplicates("ticker")
    prior=pd.read_csv(
        DATA/"events_exact_v14.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker"],
    )
    prior_kr=prior[prior.market.eq("KR")]
    universe=universe[~universe.ticker.isin(set(prior_kr.ticker.astype(str)))].copy()
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v15_kr.csv")

    ticker_set=set(universe.ticker.astype(str))
    frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":
            continue
        try:
            frame=pd.read_csv(path,dtype={"ticker":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        frame=frame[frame.ticker.astype(str).isin(ticker_set)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no expanded-universe KIND events")
    events=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    events=events[~events.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    local=events.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    events=events[regular&events.headline.fillna("").str.len().ge(3)].copy()
    events["event_day"]=local.loc[events.index].dt.strftime("%Y-%m-%d")
    events=(events.sort_values(["ticker","event_day","event_time_utc","event_id"])
            .drop_duplicates(["ticker","event_day"],keep="first")
            .drop(columns="event_day"))
    events["source"]="KIND_V15_EXPANDED_RAW"
    events["event_time_utc"]=events.event_time_utc.astype(str)
    _atomic_csv(events.sort_values("event_time_utc"),DATA/"events_kind_v15_expanded_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "industry_codes":list(industry_codes),
        "new_tickers":len(universe),
        "regular_one_per_ticker_day_events":len(events),
        "prior_event_id_overlap":0,
    },DATA/"V15_KR_EXPANDED_STATUS.json")
    print(f"[V15 KR EXPANDED] tickers={len(universe)}, events={len(events)}")
    return events


def collect_qiu_news() -> pd.DataFrame:
    """Normalize the public Reuters-based 81-company exact-time news corpus."""
    if not QIU_NEWS.exists():
        raise FileNotFoundError(
            f"missing {QIU_NEWS}; clone QiuOOGan/"
            "Word-Embedding-and-Sentiment-Analysis-based-stock-prediction first"
        )
    seed = app.load_seed_universe()
    tickers = set(seed.loc[seed.market.eq("US"), "ticker"])
    raw = json.loads(QIU_NEWS.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for ticker, articles in raw.items():
        ticker = app.norm_ticker(ticker, "US")
        if ticker not in tickers:
            continue
        for index, article in enumerate(articles):
            timestamp = pd.to_datetime(article.get("pub_time"), utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            headline = str(article.get("title", "") or "")
            body = str(article.get("text", "") or "")
            text = f"{headline} {body}".strip()
            rows.append({
                "event_id": f"QIU:{ticker}:{timestamp.isoformat()}:{index}",
                "market": "US",
                "ticker": ticker,
                "company": "",
                "event_time_utc": timestamp.isoformat(),
                "source": "REUTERS_QIU",
                "form": "",
                "headline": headline,
                "body": body,
                "event_type": app.infer_event_type(text),
                "timestamp_quality": "EXACT",
                "url": str(article.get("url", "") or ""),
            })
    out = pd.DataFrame(rows).drop_duplicates("event_id").sort_values("event_time_utc")
    _atomic_csv(out, DATA / "events_us_news_real.csv")
    material = out[[app.is_material(f"{h} {b}") for h, b in zip(out.headline, out.body)]]
    times = pd.to_datetime(material.event_time_utc, utc=True).dt.tz_convert(app.US_TZ)
    regular = material[(times.dt.time >= pd.Timestamp("09:30").time()) &
                       (times.dt.time <= pd.Timestamp("15:27").time())]
    print(
        f"[QIU] seed events={len(out)}, material={len(material)}, "
        f"regular/exit-eligible={len(regular)}, tickers={regular.ticker.nunique()}"
    )
    return out


def _naver_ticker_news(ticker: str, start: str, force: bool = False) -> list[dict[str, Any]]:
    path = NAVER_TICKER_CACHE / f"{ticker}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    start_ts = pd.Timestamp(start, tz=app.KR_TZ)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in range(1, 351):
        response = _request_with_retry(
            "GET", NAVER_NEWS_URL.format(ticker=ticker),
            params={"page": page, "pageSize": 20},
            headers={**HTTP_HEADERS, "Referer": "https://m.stock.naver.com/"},
            timeout=45,
        )
        # This endpoint can omit a JSON charset, causing Requests to assume
        # Latin-1 and corrupt Korean text.  The payload itself is UTF-8.
        groups = json.loads(response.content.decode("utf-8"))
        items = [item for group in groups for item in group.get("items", [])]
        if not items:
            break
        oldest: pd.Timestamp | None = None
        for item in items:
            article_id = str(item.get("id", "") or "")
            raw_time = str(item.get("datetime", "") or "")
            timestamp = pd.to_datetime(raw_time, format="%Y%m%d%H%M", errors="coerce")
            if pd.isna(timestamp):
                continue
            timestamp = timestamp.tz_localize(app.KR_TZ)
            oldest = timestamp if oldest is None else min(oldest, timestamp)
            if timestamp < start_ts or article_id in seen:
                continue
            seen.add(article_id)
            headline = html.unescape(str(item.get("titleFull") or item.get("title") or ""))
            body = html.unescape(str(item.get("body", "") or ""))
            text = f"{headline} {body}".strip()
            rows.append({
                "event_id": f"NAVER:{ticker}:{article_id}",
                "market": "KR", "ticker": ticker, "company": "",
                "event_time_utc": timestamp.tz_convert("UTC").isoformat(),
                "source": "NAVER_NEWS", "form": "",
                "headline": headline, "body": body,
                "event_type": app.infer_event_type(text),
                "timestamp_quality": "EXACT",
                "url": str(item.get("mobileNewsUrl") or item.get("url") or ""),
                "article_id": article_id,
            })
        if oldest is not None and oldest < start_ts:
            break
        time.sleep(0.03)
    _atomic_json(rows, path)
    return rows


def collect_naver_news(start: str, workers: int = 6, force: bool = False,
                       end: str | None = None,
                       output_name: str = "events_naver_news_real.csv",
                       source_name: str = "NAVER_NEWS") -> pd.DataFrame:
    seed = app.load_seed_universe()
    tickers = sorted(seed.loc[seed.market.eq("KR"), "ticker"].astype(str).str.zfill(6))
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_naver_ticker_news, ticker, start, force): ticker
            for ticker in tickers
        }
        for done, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                failures.append({"ticker": ticker, "error": str(exc)})
            if done % 10 == 0 or done == len(futures):
                print(f"[NAVER] {done}/{len(futures)} tickers, rows={len(all_rows)}, failures={len(failures)}")
    if failures:
        _atomic_json(failures, DATA / "NAVER_NEWS_FAILURES.json")
        raise RuntimeError(f"Naver news collection incomplete: {len(failures)} tickers")
    out = pd.DataFrame(all_rows).drop_duplicates("event_id").sort_values("event_time_utc")

    # Keep the new seal independent of already labeled KIND disclosures.
    # Exclude same-ticker articles published within 24h of a KIND release.
    kind_path = DATA / "events_kind_real.csv"
    if kind_path.exists() and not out.empty:
        kind = pd.read_csv(kind_path, dtype={"ticker": str})
        kind["event_time_utc"] = pd.to_datetime(kind.event_time_utc, utc=True)
        out["event_time_utc"] = pd.to_datetime(out.event_time_utc, utc=True)
        keep = np.ones(len(out), dtype=bool)
        kind_times = {
            ticker: np.sort(group.event_time_utc.astype("int64").to_numpy())
            for ticker, group in kind.groupby("ticker")
        }
        day_ns = int(pd.Timedelta(days=1).value)
        for pos, row in enumerate(out.itertuples(index=False)):
            values = kind_times.get(str(row.ticker))
            if values is None or not len(values):
                continue
            target = int(pd.Timestamp(row.event_time_utc).value)
            index = int(np.searchsorted(values, target))
            nearby = []
            if index < len(values): nearby.append(abs(int(values[index])-target))
            if index: nearby.append(abs(int(values[index-1])-target))
            if nearby and min(nearby) <= day_ns:
                keep[pos] = False
        out = out.iloc[np.flatnonzero(keep)].copy()
    if end is not None and not out.empty:
        upper = pd.Timestamp(end, tz=app.KR_TZ) + pd.Timedelta(days=1)
        event_times = pd.to_datetime(out.event_time_utc, utc=True)
        out = out[event_times < upper.tz_convert("UTC")].copy()
    out["source"] = source_name
    out["event_time_utc"] = pd.to_datetime(out.event_time_utc, utc=True).astype(str)
    _atomic_csv(out, DATA / output_name)
    material = out[[
        app.is_material(f"{h} {b}") for h, b in zip(out.headline.fillna(""), out.body.fillna(""))
    ]].copy()
    local = pd.to_datetime(material.event_time_utc, utc=True).dt.tz_convert(app.KR_TZ)
    regular = material[(local.dt.time >= pd.Timestamp("09:00").time()) &
                       (local.dt.time <= pd.Timestamp("14:57").time())]
    print(f"[NAVER] independent rows={len(out)}, material={len(material)}, "
          f"regular/exit-eligible={len(regular)}, tickers={regular.ticker.nunique()}")
    return out

def collect_naver_unused(end: str = "2026-08-24") -> pd.DataFrame:
    """Freeze half of the unused Naver rows and reserve half for a later seal.

    The split depends only on the immutable event ID, before prices/labels are
    joined.  That leaves an independently usable KR pool if V8 itself fails.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(NAVER_TICKER_CACHE.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError("Naver raw cache is empty")
    out=pd.DataFrame(rows).drop_duplicates("event_id")
    used:set[str]=set()
    used_frames=[]
    for name in ("events_naver_v6_2026.csv","events_naver_v7_h2_2025.csv"):
        path=DATA/name
        if path.exists():
            used_frame=pd.read_csv(path,dtype={"ticker":str})
            used.update(used_frame.event_id.astype(str))
            used_frames.append(used_frame[["ticker","event_time_utc"]])
    out=out[~out.event_id.astype(str).isin(used)].copy()
    if used_frames and not out.empty:
        prior=pd.concat(used_frames,ignore_index=True)
        prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
        out["event_time_utc"]=pd.to_datetime(out.event_time_utc,utc=True)
        prior_times={
            ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
            for ticker,group in prior.groupby("ticker")
        }
        hour_ns=int(pd.Timedelta(hours=1).value)
        keep=np.ones(len(out),dtype=bool)
        for pos,row in enumerate(out.itertuples(index=False)):
            values=prior_times.get(str(row.ticker))
            if values is None or not len(values):
                continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(values,target))
            nearby=[]
            if index<len(values): nearby.append(abs(int(values[index])-target))
            if index: nearby.append(abs(int(values[index-1])-target))
            if nearby and min(nearby)<=hour_ns:
                keep[pos]=False
        out=out.iloc[np.flatnonzero(keep)].copy()
    timestamps=pd.to_datetime(out.event_time_utc,utc=True)
    upper=pd.Timestamp(end,tz=app.KR_TZ)+pd.Timedelta(days=1)
    out=out[timestamps<upper.tz_convert("UTC")].copy()
    out["event_time_utc"]=pd.to_datetime(out.event_time_utc,utc=True).astype(str)
    bucket=out.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%2
    )
    v8=out[bucket.eq(0)].copy()
    reserve=out[bucket.eq(1)].copy()
    v8["source"]="NAVER_NEWS_V8"
    reserve["source"]="NAVER_NEWS_V9_RESERVED"
    _atomic_csv(v8,DATA/"events_naver_v8_unused.csv")
    _atomic_csv(reserve,DATA/"events_naver_v9_reserved.csv")
    material=v8[[
        app.is_material(f"{h} {b}")
        for h,b in zip(v8.headline.fillna(""),v8.body.fillna(""))
    ]].copy()
    local=pd.to_datetime(material.event_time_utc,utc=True).dt.tz_convert(app.KR_TZ)
    regular=material[(local.dt.time>=pd.Timestamp("09:00").time())&
                     (local.dt.time<=pd.Timestamp("14:57").time())]
    print(f"[NAVER UNUSED] V8 raw={len(v8)}, material={len(material)}, "
          f"regular/exit-eligible={len(regular)}, tickers={regular.ticker.nunique()}, "
          f"V9-reserved raw={len(reserve)}")
    return v8


def collect_naver_v10_fresh(end: str = "2026-08-24") -> pd.DataFrame:
    """Select newly material Naver rows and predeclare V10/V11 hash halves.

    V6-V9 event IDs and every same-ticker row within one hour of their event
    times are excluded before the corrected Korean material filter is applied.
    Neither price data nor labels participate in the split.
    """
    rows:list[dict[str,Any]]=[]
    for path in sorted(NAVER_TICKER_CACHE.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError("Naver raw cache is empty")
    out=pd.DataFrame(rows).drop_duplicates("event_id")
    prior_path=DATA/"events_exact_v9.csv.gz"
    if not prior_path.exists():
        raise FileNotFoundError(prior_path)
    prior=pd.read_csv(
        prior_path,dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior=prior[prior.market.eq("KR")].copy()
    out=out[~out.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    if not out.empty:
        prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
        out["event_time_utc"]=pd.to_datetime(out.event_time_utc,utc=True)
        prior_times={
            ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
            for ticker,group in prior.groupby("ticker")
        }
        hour_ns=int(pd.Timedelta(hours=1).value)
        keep=np.ones(len(out),dtype=bool)
        for pos,row in enumerate(out.itertuples(index=False)):
            values=prior_times.get(str(row.ticker))
            if values is None or not len(values):
                continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(values,target))
            nearby=[]
            if index<len(values): nearby.append(abs(int(values[index])-target))
            if index: nearby.append(abs(int(values[index-1])-target))
            if nearby and min(nearby)<=hour_ns:
                keep[pos]=False
        out=out.iloc[np.flatnonzero(keep)].copy()
    upper=pd.Timestamp(end,tz=app.KR_TZ)+pd.Timedelta(days=1)
    timestamps=pd.to_datetime(out.event_time_utc,utc=True)
    out=out[timestamps<upper.tz_convert("UTC")].copy()
    material=np.asarray([
        app.is_material(f"{h} {b}")
        for h,b in zip(out.headline.fillna(""),out.body.fillna(""))
    ])
    out=out.iloc[np.flatnonzero(material)].copy()
    out["event_time_utc"]=pd.to_datetime(out.event_time_utc,utc=True).astype(str)
    bucket=out.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%2
    )
    v10=out[bucket.eq(0)].copy(); reserve=out[bucket.eq(1)].copy()
    v10["source"]="NAVER_NEWS_V10"
    reserve["source"]="NAVER_NEWS_V11_RESERVED"
    _atomic_csv(v10,DATA/"events_naver_v10_fresh.csv")
    _atomic_csv(reserve,DATA/"events_naver_v11_reserved.csv")
    local=pd.to_datetime(v10.event_time_utc,utc=True).dt.tz_convert(app.KR_TZ)
    regular=v10[(local.dt.time>=pd.Timestamp("09:00").time())&
                (local.dt.time<=pd.Timestamp("14:57").time())]
    reserve_local=pd.to_datetime(reserve.event_time_utc,utc=True).dt.tz_convert(app.KR_TZ)
    reserve_regular=reserve[(reserve_local.dt.time>=pd.Timestamp("09:00").time())&
                            (reserve_local.dt.time<=pd.Timestamp("14:57").time())]
    print(f"[NAVER V10 FRESH] material={len(out)}, V10 raw={len(v10)}, "
          f"regular={len(regular)}, tickers={regular.ticker.nunique()}, "
          f"V11-reserved raw={len(reserve)}, regular={len(reserve_regular)}")
    return v10


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [str(period) for period in pd.period_range(start=start, end=end, freq="M")]


def _download_hf_month(month: str, force: bool = False) -> Path:
    path = HF_RAW_CACHE / f"ohlcv_{month}.parquet"
    if path.exists() and path.stat().st_size > 1_000_000 and not force:
        return path
    url = (
        f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/"
        f"data/ohlcv_{month}.parquet"
    )
    tmp = path.with_suffix(".parquet.part")
    response = _request_with_retry(
        "GET", url, headers=HTTP_HEADERS, attempts=5, timeout=240, stream=True,
    )
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp.replace(path)
    return path


def _filter_hf_month(month: str, tickers: set[str], force: bool = False) -> Path:
    ticker_hash = hashlib.sha256(",".join(sorted(tickers)).encode("utf-8")).hexdigest()[:12]
    output = HF_FILTERED_CACHE / f"ohlcv_{month}_selected_{ticker_hash}.parquet"
    if output.exists() and not force:
        return output
    source = _download_hf_month(month, force=force)
    parquet = pq.ParquetFile(source)
    wanted = pa.array(sorted(tickers), type=pa.string())
    selected: list[pa.Table] = []
    columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
    for batch in parquet.iter_batches(batch_size=300_000, columns=columns):
        mask = pc.is_in(batch.column("ticker"), value_set=wanted)
        filtered = pa.Table.from_batches([batch]).filter(mask)
        if filtered.num_rows:
            selected.append(filtered)
    table = pa.concat_tables(selected) if selected else pa.table({name: [] for name in columns})
    pq.write_table(table, output, compression="zstd")
    print(f"[HF] {month}: selected rows={table.num_rows:,}")
    return output


def collect_us_prices(force: bool = False, event_path: Path | None = None,
                      material_only: bool = True) -> None:
    if event_path is None:
        event_path = (
            DATA / "events_us_news_real.csv"
            if (DATA / "events_us_news_real.csv").exists()
            else DATA / "events_sec_real.csv"
        )
    events = pd.read_csv(event_path, dtype={"ticker": str})
    material = np.asarray([
        app.is_material(f"{h} {b}", f)
        for h, b, f in zip(events.headline.fillna(""), events.body.fillna(""), events.form.fillna(""))
    ])
    if material_only:
        events = events[material].copy()
    events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
    local = events.event_time_utc.dt.tz_convert(app.US_TZ)
    events = events[(local.dt.time >= pd.Timestamp("09:30").time()) &
                    (local.dt.time <= pd.Timestamp("15:27").time())].copy()
    event_tickers = set(events.ticker.unique())
    # Benchmark features are evaluated at the event entry time, so fetch SPY
    # over exactly the same months as the event securities.
    tickers = event_tickers | {"SPY"}
    months = _months_between(events.event_time_utc.min(), events.event_time_utc.max())
    filtered_by_month:dict[str,Path]={}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={
            executor.submit(_filter_hf_month,month,tickers,force):month
            for month in months
        }
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future]
            filtered_by_month[month]=future.result()
            print(f"[HF] completed {done}/{len(months)}: {month}")
    filtered_paths=[filtered_by_month[month] for month in months]
    frames = [pd.read_parquet(path) for path in filtered_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined.ticker.astype(str)
    for ticker, group in combined.groupby("ticker", sort=True):
        path = PRICE / "US" / f"{ticker}.parquet"
        group = group.drop(columns="ticker").sort_values("timestamp").drop_duplicates("timestamp")
        # A ticker can occur in both the Reuters development era and the SEC
        # seal era.  Preserve both disjoint periods instead of replacing the
        # earlier file when a later source is prepared.
        if path.exists():
            previous = pd.read_parquet(path)
            group = pd.concat([previous, group], ignore_index=True)
            group["timestamp"] = pd.to_datetime(group["timestamp"], utc=True)
            group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        temporary = path.with_suffix(".parquet.tmp")
        group.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    missing = sorted(tickers - set(combined.ticker.unique()))
    _atomic_json({
        "source": HF_REPO,
        "revision": HF_REVISION,
        "months": months,
        "event_tickers": len(event_tickers),
        "benchmark_ticker": "SPY",
        "price_tickers": int(combined.ticker.nunique()),
        "missing": missing,
    }, DATA / "US_PRICE_STATUS.json")
    print(f"[HF] price tickers={combined.ticker.nunique()}, missing={len(missing)}")


def collect_us_benchmark(start: str, end: str, force: bool = False) -> None:
    """Merge SPY minute bars for an explicit interval into the benchmark file."""
    months=_months_between(pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC"))
    filtered_by_month:dict[str,Path]={}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(_filter_hf_month,month,{"SPY"},force):month for month in months}
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future]
            filtered_by_month[month]=future.result()
            print(f"[HF BENCHMARK] completed {done}/{len(months)}: {month}")
    frames=[pd.read_parquet(filtered_by_month[month]) for month in months]
    combined=pd.concat(frames,ignore_index=True)
    combined=combined[combined.ticker.astype(str).eq("SPY")].drop(columns="ticker")
    path=PRICE/"US"/"SPY.parquet"
    if path.exists():
        combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
    combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
    combined=(combined.sort_values("timestamp")
              .drop_duplicates("timestamp",keep="last"))
    temporary=path.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary,index=False,compression="zstd")
    temporary.replace(path)
    print(f"[HF BENCHMARK] SPY rows={len(combined):,}, end={combined.timestamp.max()}")


def _download_daum_ticker(ticker: str, start: str, end: str, force: bool) -> tuple[str, int, str]:
    output = PRICE / "KR" / f"{ticker}.parquet"
    if output.exists() and not force:
        return ticker, len(pd.read_parquet(output, columns=["datetime"])), "cached"
    url = DAUM_URL.format(ticker=ticker)
    end_cursor = pd.Timestamp(end) + pd.Timedelta(hours=16)
    start_ts = pd.Timestamp(start)
    records: list[dict[str, Any]] = []
    seen_first: set[str] = set()
    # The endpoint's current page depth is shorter than its historical
    # behavior.  160 cursored pages reach the fixed V29 2022 KIND interval.
    for _ in range(160):
        response = _request_with_retry(
            "GET",
            url,
            params={
                "limit": "5000",
                "adjusted": "true",
                "to": end_cursor.strftime("%Y-%m-%d %H:%M:%S"),
            },
            headers={**HTTP_HEADERS, "Referer": f"https://finance.daum.net/chart/A{ticker}"},
            timeout=120,
        )
        data = response.json().get("data", [])
        if not data:
            break
        records.extend(data)
        first = str(data[0].get("candleTime", ""))
        if not first or first in seen_first:
            break
        seen_first.add(first)
        first_ts = pd.Timestamp(first)
        if first_ts <= start_ts:
            break
        end_cursor = first_ts - pd.Timedelta(minutes=1)
        time.sleep(0.05)
    if not records:
        return ticker, 0, "no data"
    raw = pd.DataFrame(records)
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(raw["candleTime"], errors="coerce"),
        "open": pd.to_numeric(raw["openingPrice"], errors="coerce"),
        "high": pd.to_numeric(raw["highPrice"], errors="coerce"),
        "low": pd.to_numeric(raw["lowPrice"], errors="coerce"),
        "close": pd.to_numeric(raw["tradePrice"], errors="coerce"),
        "volume": pd.to_numeric(raw.get("candleAccTradeVolume", 0), errors="coerce"),
    }).dropna(subset=["datetime", "close"])
    frame = frame[(frame.datetime >= start_ts) &
                  (frame.datetime < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if output.exists():
        previous = pd.read_parquet(output)
        frame = pd.concat([previous, frame], ignore_index=True, sort=False)
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last")
    frame.to_parquet(output, index=False, compression="zstd")
    return ticker, len(frame), "ok"


def collect_kr_prices(start: str, end: str, workers: int = 4, force: bool = False,
                      event_path: Path | None = None,
                      material_only: bool = True) -> None:
    if event_path is None:
        event_path = DATA / "events_kind_real.csv"
    events = pd.read_csv(event_path, dtype={"ticker": str})
    material = np.asarray([
        app.is_material(f"{h} {b}", f)
        for h, b, f in zip(events.headline.fillna(""), events.body.fillna(""), events.form.fillna(""))
    ])
    times = pd.to_datetime(events.event_time_utc, utc=True).dt.tz_convert(app.KR_TZ)
    regular = (times.dt.time >= pd.Timestamp("09:00").time()) & (
        times.dt.time <= pd.Timestamp("14:57").time()
    )
    selection=regular & (material if material_only else np.ones(len(events),dtype=bool))
    tickers = sorted(set(events.loc[selection, "ticker"].astype(str).str.zfill(6)))
    status: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_daum_ticker, ticker, start, end, force): ticker
            for ticker in tickers
        }
        for done, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                ticker, rows, note = future.result()
                status.append({"ticker": ticker, "rows": rows, "status": note})
            except Exception as exc:
                status.append({"ticker": ticker, "rows": 0, "status": f"ERROR: {exc}"})
            if done % 10 == 0 or done == len(futures):
                print(f"[DAUM] {done}/{len(futures)} tickers")
    _atomic_json(status, DATA / "KR_PRICE_STATUS.json")
    errors = [row for row in status if row["rows"] == 0]
    if len(errors) > max(3, int(np.ceil(0.05 * max(len(status), 1)))):
        raise RuntimeError(f"KR price collection incomplete: {len(errors)} tickers with no data")
    if errors:
        print(f"[DAUM WARN] no usable bars for {len(errors)} ticker(s): "
              f"{[row['ticker'] for row in errors]}")


def prepare_v12_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze V12 DEV/SEAL populations before any outcome is inspected.

    SEC is split by an immutable event-ID hash.  KR uses previously unused
    Naver articles, removes all earlier IDs and same-ticker +/-60 minute
    neighbours, then uses a chronological 2026-05-01 cutoff.  Price outcomes
    and labels are not read here.
    """
    sec_paths=[
        DATA/"events_sec_v12_2008_raw.csv",
        DATA/"events_sec_v12_2018_raw.csv",
    ]
    if any(not path.exists() for path in sec_paths):
        raise FileNotFoundError(f"missing V12 SEC source: {sec_paths}")
    sec=pd.concat(
        [pd.read_csv(path,dtype={"ticker":str}) for path in sec_paths],
        ignore_index=True,sort=False,
    ).drop_duplicates("event_id")
    bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%10
    )
    sec_dev=sec[bucket.lt(4)].copy(); sec_seal=sec[bucket.ge(4)].copy()
    sec_dev["source"]="SEC_V12_DEV"; sec_seal["source"]="SEC_V12_SEAL"
    sec_dev=sec_dev.sort_values("event_time_utc")
    sec_seal=sec_seal.sort_values("event_time_utc")
    _atomic_csv(sec_dev,DATA/"events_sec_v12_dev.csv")
    _atomic_csv(sec_seal,DATA/"events_sec_v12_seal.csv")

    rows=[]
    for path in sorted(NAVER_TICKER_CACHE.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError("Naver raw cache is empty")
    naver=pd.DataFrame(rows).drop_duplicates("event_id")
    prior=pd.read_csv(
        DATA/"events_exact_v11.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior=prior[prior.market.eq("KR")].copy()
    naver=naver[~naver.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    naver["event_time_utc"]=pd.to_datetime(naver.event_time_utc,utc=True)
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior.groupby("ticker")
    }
    hour_ns=int(pd.Timedelta(hours=1).value)
    keep=np.ones(len(naver),dtype=bool)
    for pos,row in enumerate(naver.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values): nearby.append(abs(int(values[index])-target))
        if index: nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=hour_ns:
            keep[pos]=False
    naver=naver.iloc[np.flatnonzero(keep)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    # Narrow business/biopharma terms chosen without consulting returns.
    terms=[
        "\uacf5\uae09","\uc218\uc8fc","\uc5c5\ubb34\ud611\uc57d","\ud611\uc57d","MOU","\ucd9c\uc2dc",
        "\uc2e0\uc57d","\uce58\ub8cc\uc81c","\ud6c4\ubcf4\ubb3c\uc9c8","\ub17c\ubb38","\ud559\ud68c","\uc18c\uc1a1",
        "\uc0dd\uc0b0","\uacf5\uc7a5","\uc81c\ud488","\uc9c4\ub2e8","\ubc31\uc2e0","\ud751\uc790","\uc801\uc790",
        "\uacc4\uc57d","\ud30c\ud2b8\ub108","\ubc14\uc774\uc624\uc2dc\ubc00\ub7ec","\uac74\uac15\ubcf4\ud5d8","\ud574\uc678","\uc784\uc0c1",
    ]
    text=(naver.headline.fillna("")+" "+naver.body.fillna("")).str.lower()
    material=text.map(lambda value:any(term.lower() in value for term in terms))
    naver=naver[regular&material].copy()
    cutoff=pd.Timestamp("2026-05-01",tz=app.KR_TZ).tz_convert("UTC")
    kr_dev=naver[naver.event_time_utc.lt(cutoff)].copy()
    kr_seal=naver[naver.event_time_utc.ge(cutoff)].copy()
    kr_dev["source"]="NAVER_NEWS_V12_DEV"
    kr_seal["source"]="NAVER_NEWS_V12_SEAL"
    for frame,path in (
        (kr_dev,DATA/"events_naver_v12_dev.csv"),
        (kr_seal,DATA/"events_naver_v12_seal.csv"),
    ):
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)

    _atomic_json({
        "selection_uses_labels":False,
        "sec_dev_events":len(sec_dev),"sec_seal_events":len(sec_seal),
        "kr_dev_events":len(kr_dev),"kr_seal_events":len(kr_seal),
        "kr_cutoff_utc":cutoff.isoformat(),
        "kr_prior_exclusion_minutes":60,
    },DATA/"V12_SOURCE_STATUS.json")
    print(f"[V12 SOURCES] SEC dev={len(sec_dev)} seal={len(sec_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return sec_dev,sec_seal,kr_dev,kr_seal


def prepare_v13_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze unused 2006-2007 SEC and remaining Naver groups for V13."""
    sec=pd.read_csv(DATA/"events_sec_v13_2006_2007_raw.csv",dtype={"ticker":str})
    bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%10
    )
    us_dev=sec[bucket.lt(4)].copy();us_seal=sec[bucket.ge(4)].copy()
    us_dev["source"]="SEC_V13_DEV";us_seal["source"]="SEC_V13_SEAL"
    _atomic_csv(us_dev.sort_values("event_time_utc"),DATA/"events_sec_v13_dev.csv")
    _atomic_csv(us_seal.sort_values("event_time_utc"),DATA/"events_sec_v13_seal.csv")

    rows=[]
    for path in sorted(NAVER_TICKER_CACHE.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    naver=pd.DataFrame(rows).drop_duplicates("event_id")
    prior=pd.read_csv(
        DATA/"events_exact_v12.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior=prior[prior.market.eq("KR")].copy()
    naver=naver[~naver.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    naver["event_time_utc"]=pd.to_datetime(naver.event_time_utc,utc=True)
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior.groupby("ticker")
    }
    hour_ns=int(pd.Timedelta(hours=1).value)
    keep=np.ones(len(naver),dtype=bool)
    for pos,row in enumerate(naver.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values): nearby.append(abs(int(values[index])-target))
        if index: nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=hour_ns:
            keep[pos]=False
    naver=naver.iloc[np.flatnonzero(keep)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    terms=[
        "\ud22c\uc790","\uc0c1\uc7a5","\uc5f0\uad6c","\uac1c\ubc1c","\uae30\uc220","\uae00\ub85c\ubc8c",
        "\ud2b9\ud5c8","\ub3c5\uc810","FDA","\uc2b9\uc778","\ud5c8\uac00","\ub9e4\ucd9c","\uc2e4\uc801",
        "\uc601\uc5c5\uc774\uc775","\ud569\ubcd1","\uc778\uc218","\uc99d\uc790","\uc804\ud658\uc0ac\ucc44",
    ]
    text=(naver.headline.fillna("")+" "+naver.body.fillna("")).str.lower()
    material=text.map(lambda value:any(term.lower() in value for term in terms))
    naver=naver[regular&material].copy()
    article=naver.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    article_bucket=article.map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%10
    )
    kr_dev=naver[article_bucket.lt(6)].copy()
    kr_seal=naver[article_bucket.ge(6)].copy()
    kr_dev["source"]="NAVER_NEWS_V13_DEV"
    kr_seal["source"]="NAVER_NEWS_V13_SEAL"
    for frame,path in (
        (kr_dev,DATA/"events_naver_v13_dev.csv"),
        (kr_seal,DATA/"events_naver_v13_seal.csv"),
    ):
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,
        "us_dev_events":len(us_dev),"us_seal_events":len(us_seal),
        "kr_dev_events":len(kr_dev),"kr_seal_events":len(kr_seal),
        "kr_prior_exclusion_minutes":60,
    },DATA/"V13_SOURCE_STATUS.json")
    print(f"[V13 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v14_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze unused historical SEC and remaining Naver articles for V14."""
    sec=pd.read_csv(DATA/"events_sec_v14_unused_raw.csv",dtype={"ticker":str})
    prior=pd.read_csv(
        DATA/"events_exact_v13.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    sec["event_time_utc"]=pd.to_datetime(sec.event_time_utc,utc=True)
    sec=sec[~sec.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()

    # A different accession can describe the same corporate event.  Enforce a
    # one-hour embargo around every previously opened ticker event as well as
    # exact event-id exclusion before making the label-blind hash split.
    prior_us=prior[prior.market.eq("US")]
    prior_us_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior_us.groupby("ticker")
    }
    hour_ns=int(pd.Timedelta(hours=1).value)
    keep=np.ones(len(sec),dtype=bool)
    for position,row in enumerate(sec.itertuples(index=False)):
        values=prior_us_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values): nearby.append(abs(int(values[index])-target))
        if index: nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=hour_ns:
            keep[position]=False
    sec=sec.iloc[np.flatnonzero(keep)].copy()
    bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%10
    )
    # Reserve 60% for the one-shot seal so price availability still leaves a
    # comfortable US >=250 evaluation set.  The assignment is event-id-only.
    us_dev=sec[bucket.lt(4)].copy();us_seal=sec[bucket.ge(4)].copy()
    us_dev["source"]="SEC_V14_DEV";us_seal["source"]="SEC_V14_SEAL"
    _atomic_csv(us_dev.sort_values("event_time_utc"),DATA/"events_sec_v14_dev.csv")
    _atomic_csv(us_seal.sort_values("event_time_utc"),DATA/"events_sec_v14_seal.csv")

    rows=[]
    for path in sorted(NAVER_TICKER_CACHE.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    naver=pd.DataFrame(rows).drop_duplicates("event_id")
    prior=prior[prior.market.eq("KR")].copy()
    naver=naver[~naver.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    naver["event_time_utc"]=pd.to_datetime(naver.event_time_utc,utc=True)
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior.groupby("ticker")
    }
    keep=np.ones(len(naver),dtype=bool)
    for position,row in enumerate(naver.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values): nearby.append(abs(int(values[index])-target))
        if index: nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=hour_ns:
            keep[position]=False
    naver=naver.iloc[np.flatnonzero(keep)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    naver=naver[regular&naver.headline.fillna("").str.len().ge(3)].copy()
    article=naver.event_id.astype(str).str.rsplit(":",n=1).str[-1]
    article_bucket=article.map(
        lambda value:int(hashlib.sha256(value.encode("utf-8")).hexdigest(),16)%10
    )
    kr_dev=naver[article_bucket.lt(6)].copy()
    kr_seal=naver[article_bucket.ge(6)].copy()
    kr_dev["source"]="NAVER_NEWS_V14_DEV"
    kr_seal["source"]="NAVER_NEWS_V14_SEAL"
    for frame,path in (
        (kr_dev,DATA/"events_naver_v14_dev.csv"),
        (kr_seal,DATA/"events_naver_v14_seal.csv"),
    ):
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,
        "us_dev_events":len(us_dev),"us_seal_events":len(us_seal),
        "kr_dev_events":len(kr_dev),"kr_seal_events":len(kr_seal),
        "us_prior_exclusion_minutes":60,
        "kr_prior_exclusion_minutes":60,
        "kr_source_definition":"all remaining exact-time regular-session ticker news",
    },DATA/"V14_SOURCE_STATUS.json")
    print(f"[V14 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v15_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze expanded-universe SEC and unused 2022 KIND sources for V15."""
    prior=pd.read_csv(
        DATA/"events_exact_v14.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior_ids=set(prior.event_id.astype(str))

    sec=pd.read_csv(DATA/"events_sec_v15_unused_raw.csv",dtype={"ticker":str})
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V15|{value}".encode("utf-8")).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(4)].copy();us_seal=sec[us_bucket.ge(4)].copy()
    us_dev["source"]="SEC_V15_DEV";us_seal["source"]="SEC_V15_SEAL"
    _atomic_csv(us_dev.sort_values("event_time_utc"),DATA/"events_sec_v15_dev.csv")
    _atomic_csv(us_seal.sort_values("event_time_utc"),DATA/"events_sec_v15_seal.csv")

    kind=pd.read_csv(DATA/"events_kind_v15_expanded_raw.csv",dtype={"ticker":str})
    kind=kind[~kind.event_id.astype(str).isin(prior_ids)].copy()
    kind["event_time_utc"]=pd.to_datetime(kind.event_time_utc,utc=True)
    local=kind.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    kind=kind[regular&kind.headline.fillna("").str.len().ge(3)].copy()
    kr_bucket=kind.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V15|{value}".encode("utf-8")).hexdigest(),16)%10
    )
    kr_dev=kind[kr_bucket.lt(4)].copy();kr_seal=kind[kr_bucket.ge(4)].copy()
    kr_dev["source"]="KIND_V15_DEV";kr_seal["source"]="KIND_V15_SEAL"
    for frame,path in (
        (kr_dev,DATA/"events_kind_v15_dev.csv"),
        (kr_seal,DATA/"events_kind_v15_seal.csv"),
    ):
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,
        "us_dev_events":len(us_dev),"us_seal_events":len(us_seal),
        "kr_dev_events":len(kr_dev),"kr_seal_events":len(kr_seal),
        "us_source_definition":"expanded SEC life-science company-title universe",
        "kr_source_definition":"unused exact-time KIND disclosures from expanded health-industry peers",
    },DATA/"V15_SOURCE_STATUS.json")
    print(f"[V15 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def collect_v16_us_source(force: bool = False) -> pd.DataFrame:
    """Collect the next unused SEC filings from the V15 expanded universe."""
    return collect_sec(
        "2019-01-01","2025-05-31",force=force,
        output_name="events_sec_v16_seal.csv",source_name="SEC_V16_SEAL",
        one_per_ticker_day=True,max_events_per_ticker=16,
        exclude_event_path=DATA/"events_exact_v15.csv.gz",exclude_minutes=60,
        universe_path=DATA/"universe_v15_us.csv",
        material_only=False,fetch_body=False,
    )


def prepare_v16_kr_source(exclude_minutes: int = 35) -> pd.DataFrame:
    """Freeze unused non-overlapping KIND events from the V15 peer universe."""
    universe=pd.read_csv(DATA/"universe_v15_kr.csv",dtype={"ticker":str})
    ticker_set=set(universe.ticker.astype(str))
    frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":
            continue
        try:
            frame=pd.read_csv(path,dtype={"ticker":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        frame=frame[frame.ticker.astype(str).isin(ticker_set)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no V16 expanded-universe KIND events")
    events=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    prior=pd.read_csv(
        DATA/"events_exact_v15.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior=prior[prior.market.eq("KR")].copy()
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    events=events[~events.event_id.astype(str).isin(prior_ids)].copy()
    local=events.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    events=events[regular&events.headline.fillna("").str.len().ge(3)].copy()

    # More than 35 minutes separates every new timestamp from an opened one,
    # so the +2m -> +32m outcome windows cannot overlap.
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior.groupby("ticker")
    }
    embargo_ns=int(pd.Timedelta(minutes=exclude_minutes).value)
    keep=np.ones(len(events),dtype=bool)
    for position,row in enumerate(events.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values):
            nearby.append(abs(int(values[index])-target))
        if index:
            nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=embargo_ns:
            keep[position]=False
    events=events.iloc[np.flatnonzero(keep)].copy()
    local=events.event_time_utc.dt.tz_convert(app.KR_TZ)
    events["event_day"]=local.dt.strftime("%Y-%m-%d")
    events=(events.sort_values(["ticker","event_day","event_time_utc","event_id"])
            .drop_duplicates(["ticker","event_day"],keep="first")
            .drop(columns="event_day"))
    events["source"]="KIND_V16_SEAL"
    events["event_time_utc"]=events.event_time_utc.astype(str)
    _atomic_csv(events.sort_values("event_time_utc"),DATA/"events_kind_v16_seal.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "source_definition":"next unused exact-time KIND event per peer ticker/day",
        "prior_exclusion_minutes":exclude_minutes,
        "outcome_window_overlap_with_prior":False,
        "events":len(events),"tickers":events.ticker.nunique(),
        "prior_event_id_overlap":0,
    },DATA/"V16_KR_SOURCE_STATUS.json")
    print(f"[V16 KR SOURCE] events={len(events)}, tickers={events.ticker.nunique()}")
    return events


def prepare_v17_universes() -> tuple[pd.DataFrame,pd.DataFrame]:
    """Build label-blind US/KR life-science cohorts absent from V16 history."""
    prior=pd.read_csv(
        DATA/"events_exact_v16.csv.gz",dtype={"ticker":str},usecols=["market","ticker"]
    )

    collector=app.SECCollector(app.Config())
    ticker_map=collector.get_json(collector.TICKER_MAP)
    us_pattern=re.compile(
        r"(?:BIO|PHARMA|THERAPEUT|MEDIC|HEALTH|LIFE SCI|ONCO|GENET|GENOMIC|"
        r"VACCIN|DIAGNOST|NEURO|IMMUN|OPHTHAL|CARDIO|SURGICAL|LABORATOR)",
        re.IGNORECASE,
    )
    us_rows=[]
    for item in ticker_map.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        primary=bool(re.fullmatch(r"[A-Z]{1,5}",ticker)) and not ticker.endswith(("W","P"))
        if primary and company and us_pattern.search(company):
            us_rows.append({"market":"US","ticker":ticker,"company":company})
    us=pd.DataFrame(us_rows).drop_duplicates("ticker")
    prior_us=set(prior.loc[prior.market.eq("US"),"ticker"].astype(str))
    us=us[~us.ticker.isin(prior_us)].copy()
    us["selection_hash"]=us.ticker.map(
        lambda value:hashlib.sha256(f"V17|US|{value}".encode()).hexdigest()
    )
    us=us.sort_values("selection_hash").drop(columns="selection_hash").reset_index(drop=True)
    us["priority"]=np.arange(1,len(us)+1)
    _atomic_csv(us,DATA/"universe_v17_us.csv")

    company_frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":
            continue
        try:
            frame=pd.read_csv(
                path,dtype={"ticker":str},encoding="utf-8-sig",usecols=["ticker","company"]
            )
        except (pd.errors.EmptyDataError,ValueError):
            continue
        if not frame.empty:
            company_frames.append(frame)
    companies=(pd.concat(company_frames,ignore_index=True).dropna(subset=["ticker","company"])
               .drop_duplicates("ticker"))
    companies["ticker"]=companies.ticker.astype(str).str.zfill(6)
    kr_pattern=re.compile(
        r"(?:바이오|제약|약품|의료|헬스|메디|셀|유전자|진단|백신|테라|랩|신약|케어|팜|생명|과학)",
        re.IGNORECASE,
    )
    prior_kr=set(prior.loc[prior.market.eq("KR"),"ticker"].astype(str))
    kr=companies[
        companies.company.astype(str).str.contains(kr_pattern,na=False)&
        ~companies.ticker.isin(prior_kr)
    ].copy()
    kr.insert(0,"market","KR")
    kr["selection_hash"]=kr.ticker.map(
        lambda value:hashlib.sha256(f"V17|KR|{value}".encode()).hexdigest()
    )
    kr=kr.sort_values("selection_hash").drop(columns="selection_hash").reset_index(drop=True)
    kr["priority"]=np.arange(1,len(kr)+1)
    _atomic_csv(kr,DATA/"universe_v17_kr.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "us_definition":"remaining SEC life-science company-title tickers",
        "kr_definition":"remaining KIND issuer names with life-science keywords",
        "excluded_prior_us_tickers":len(prior_us),
        "excluded_prior_kr_tickers":len(prior_kr),
        "us_tickers":len(us),"kr_tickers":len(kr),
    },DATA/"V17_UNIVERSE_STATUS.json")
    print(f"[V17 UNIVERSES] US={len(us)}, KR={len(kr)}")
    return us,kr


def collect_v17_us_raw(force: bool=False) -> pd.DataFrame:
    if not (DATA/"universe_v17_us.csv").exists():
        prepare_v17_universes()
    return collect_sec(
        "2019-01-01","2025-05-31",force=force,
        output_name="events_sec_v17_raw.csv",source_name="SEC_V17_RAW",
        one_per_ticker_day=True,max_events_per_ticker=16,
        exclude_event_path=DATA/"events_exact_v16.csv.gz",exclude_minutes=60,
        universe_path=DATA/"universe_v17_us.csv",material_only=False,fetch_body=False,
    )


def prepare_v17_kr_raw() -> pd.DataFrame:
    if not (DATA/"universe_v17_kr.csv").exists():
        prepare_v17_universes()
    universe=pd.read_csv(DATA/"universe_v17_kr.csv",dtype={"ticker":str})
    ticker_set=set(universe.ticker.astype(str))
    frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":
            continue
        try:
            frame=pd.read_csv(path,dtype={"ticker":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        frame=frame[frame.ticker.astype(str).isin(ticker_set)]
        if not frame.empty:
            frames.append(frame)
    events=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    prior=pd.read_csv(DATA/"events_exact_v16.csv.gz",dtype={"ticker":str},usecols=["event_id"])
    events=events[~events.event_id.astype(str).isin(set(prior.event_id.astype(str)))].copy()
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    local=events.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    events=events[regular&events.headline.fillna("").str.len().ge(3)].copy()
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.KR_TZ).dt.strftime("%Y-%m-%d")
    events=(events.sort_values(["ticker","event_day","event_time_utc","event_id"])
            .drop_duplicates(["ticker","event_day"],keep="first").drop(columns="event_day"))
    events["source"]="KIND_V17_RAW"
    events["event_time_utc"]=events.event_time_utc.astype(str)
    _atomic_csv(events.sort_values("event_time_utc"),DATA/"events_kind_v17_raw.csv")
    print(f"[V17 KR RAW] events={len(events)}, tickers={events.ticker.nunique()}")
    return events


def prepare_v17_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Make immutable label-blind 40/60 DEV/SEAL event-id splits."""
    sec=pd.read_csv(DATA/"events_sec_v17_raw.csv",dtype={"ticker":str})
    kind=pd.read_csv(DATA/"events_kind_v17_raw.csv",dtype={"ticker":str})
    prior=pd.read_csv(DATA/"events_exact_v16.csv.gz",dtype={"ticker":str},usecols=["event_id"])
    prior_ids=set(prior.event_id.astype(str))
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].copy()
    kind=kind[~kind.event_id.astype(str).isin(prior_ids)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V17|{value}".encode()).hexdigest(),16)%10
    )
    kr_bucket=kind.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V17|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(4)].copy();us_seal=sec[us_bucket.ge(4)].copy()
    kr_dev=kind[kr_bucket.lt(4)].copy();kr_seal=kind[kr_bucket.ge(4)].copy()
    for frame,source,path in (
        (us_dev,"SEC_V17_DEV",DATA/"events_sec_v17_dev.csv"),
        (us_seal,"SEC_V17_SEAL",DATA/"events_sec_v17_seal.csv"),
        (kr_dev,"KIND_V17_DEV",DATA/"events_kind_v17_dev.csv"),
        (kr_seal,"KIND_V17_SEAL",DATA/"events_kind_v17_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,"split":"SHA256(V17|event_id) mod 10; DEV<4",
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,
    },DATA/"V17_SOURCE_STATUS.json")
    print(f"[V17 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v18_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze unused SEC residuals and unused 2022 KIND events for V18."""
    prior=pd.read_csv(
        DATA/"events_exact_v17.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))

    sec_frames=[
        pd.read_csv(DATA/"events_sec_v18_raw.csv",dtype={"ticker":str}),
        pd.read_csv(DATA/"events_sec_v18_v15_residual_raw.csv",dtype={"ticker":str}),
    ]
    sec=(pd.concat(sec_frames,ignore_index=True,sort=False)
         .drop_duplicates("event_id"))
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V18|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(3)].copy();us_seal=sec[us_bucket.ge(3)].copy()

    # Search the full-market KIND cache for still-unused disclosures from the
    # previously declared Korean bio universe.  This is metadata-only source
    # selection; price availability and outcomes do not participate.
    bio_tickers=set(prior.loc[prior.market.eq("KR"),"ticker"].astype(str))
    kind_frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":
            continue
        try:
            frame=pd.read_csv(path,dtype={"ticker":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        frame=frame[frame.ticker.astype(str).isin(bio_tickers)]
        if not frame.empty:
            kind_frames.append(frame)
    if not kind_frames:
        raise RuntimeError("no unused V18 KIND candidates")
    kind=pd.concat(kind_frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    kind=kind[~kind.event_id.astype(str).isin(prior_ids)].copy()
    kind["event_time_utc"]=pd.to_datetime(kind.event_time_utc,utc=True)
    local=kind.event_time_utc.dt.tz_convert(app.KR_TZ)
    regular=(local.dt.time>=pd.Timestamp("09:00").time())&(
        local.dt.time<=pd.Timestamp("14:57").time()
    )
    kind=kind[regular&kind.headline.fillna("").str.len().ge(3)].copy()

    # Prevent a differently named record for an already opened disclosure from
    # sharing any part of its t+2m -> t+32m result window.
    prior_kr=prior[prior.market.eq("KR")]
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior_kr.groupby("ticker")
    }
    embargo_ns=int(pd.Timedelta(minutes=60).value)
    keep=np.ones(len(kind),dtype=bool)
    for position,row in enumerate(kind.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):
            continue
        target=int(pd.Timestamp(row.event_time_utc).value)
        index=int(np.searchsorted(values,target))
        nearby=[]
        if index<len(values): nearby.append(abs(int(values[index])-target))
        if index: nearby.append(abs(int(values[index-1])-target))
        if nearby and min(nearby)<=embargo_ns:
            keep[position]=False
    kind=kind.iloc[np.flatnonzero(keep)].copy()
    kind["event_day"]=kind.event_time_utc.dt.tz_convert(app.KR_TZ).dt.strftime("%Y-%m-%d")
    kind=(kind.sort_values(["ticker","event_day","event_time_utc","event_id"])
          .drop_duplicates(["ticker","event_day"],keep="first")
          .drop(columns="event_day"))
    kr_bucket=kind.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V18|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kind[kr_bucket.lt(3)].copy();kr_seal=kind[kr_bucket.ge(3)].copy()

    for frame,source,path in (
        (us_dev,"SEC_V18_DEV",DATA/"events_sec_v18_dev.csv"),
        (us_seal,"SEC_V18_SEAL",DATA/"events_sec_v18_seal.csv"),
        (kr_dev,"KIND_V18_DEV",DATA/"events_kind_v18_dev.csv"),
        (kr_seal,"KIND_V18_SEAL",DATA/"events_kind_v18_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,
        "split":"SHA256(V18|event_id) mod 10; DEV<3",
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,"prior_outcome_embargo_minutes":60,
    },DATA/"V18_SOURCE_STATUS.json")
    print(f"[V18 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v19_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze unused historical SEC and non-overlapping KIND residuals."""
    prior=pd.read_csv(
        DATA/"events_exact_v18.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))

    sec_paths=[
        DATA/"events_sec_v19_v17_historical_raw.csv",
        DATA/"events_sec_v19_v17_residual_raw.csv",
        DATA/"events_sec_v19_v15_residual_raw.csv",
    ]
    sec=(pd.concat([pd.read_csv(path,dtype={"ticker":str}) for path in sec_paths],
                   ignore_index=True,sort=False).drop_duplicates("event_id"))
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V19|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(2)].copy();us_seal=sec[us_bucket.ge(2)].copy()

    bio_tickers=set(prior.loc[prior.market.eq("KR"),"ticker"].astype(str))
    frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        raw_date=path.name[:10]
        if raw_date<"2024-03-01" or raw_date>"2026-08-24":continue
        try:frame=pd.read_csv(path,dtype={"ticker":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:continue
        frame=frame[frame.ticker.astype(str).isin(bio_tickers)]
        if not frame.empty:frames.append(frame)
    kind=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    kind=kind[~kind.event_id.astype(str).isin(prior_ids)].copy()
    kind["event_time_utc"]=pd.to_datetime(kind.event_time_utc,utc=True)
    local=kind.event_time_utc.dt.tz_convert(app.KR_TZ)
    kind=kind[(local.dt.time>=pd.Timestamp("09:00").time())&
              (local.dt.time<=pd.Timestamp("14:57").time())&
              kind.headline.fillna("").str.len().ge(3)].copy()
    prior_kr=prior[prior.market.eq("KR")]
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior_kr.groupby("ticker")
    }
    embargo_ns=int(pd.Timedelta(minutes=35).value);keep=np.ones(len(kind),dtype=bool)
    for position,row in enumerate(kind.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):continue
        target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(values,target));near=[]
        if index<len(values):near.append(abs(int(values[index])-target))
        if index:near.append(abs(int(values[index-1])-target))
        if near and min(near)<=embargo_ns:keep[position]=False
    kind=kind.iloc[np.flatnonzero(keep)].copy()
    kind["event_day"]=kind.event_time_utc.dt.tz_convert(app.KR_TZ).dt.strftime("%Y-%m-%d")
    kind=(kind.sort_values(["ticker","event_day","event_time_utc","event_id"])
          .drop_duplicates(["ticker","event_day"],keep="first").drop(columns="event_day"))
    kr_bucket=kind.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V19|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kind[kr_bucket.lt(1)].copy();kr_seal=kind[kr_bucket.ge(1)].copy()
    for frame,source,path in (
        (us_dev,"SEC_V19_DEV",DATA/"events_sec_v19_dev.csv"),
        (us_seal,"SEC_V19_SEAL",DATA/"events_sec_v19_seal.csv"),
        (kr_dev,"KIND_V19_DEV",DATA/"events_kind_v19_dev.csv"),
        (kr_seal,"KIND_V19_SEAL",DATA/"events_kind_v19_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    _atomic_json({
        "selection_uses_labels":False,
        "split":"SHA256(V19|event_id) mod 10; US DEV<2, KR DEV<1",
        "us_dev":len(us_dev),"us_seal":len(us_seal),"kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,"prior_outcome_embargo_minutes":35,
    },DATA/"V19_SOURCE_STATUS.json")
    print(f"[V19 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v20_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze fresh 2006-09 SEC and residual Naver events before labeling."""
    prior=pd.read_csv(
        DATA/"events_exact_v19.csv.gz",dtype={"ticker":str},
        usecols=["event_id","market","ticker","event_time_utc"],
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))

    sec=pd.read_csv(DATA/"events_sec_v20_v17_historical_raw.csv",dtype={"ticker":str})
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].drop_duplicates("event_id").copy()
    # The monthly archive lookup is label-free; discard symbols for which the
    # downloaded 2006-09 archive contains no minute bars before hashing.
    price_status=json.loads((DATA/"US_PRICE_STATUS.json").read_text(encoding="utf-8"))
    missing=set(map(str,price_status.get("missing",[])))
    sec=sec[~sec.ticker.astype(str).isin(missing)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V20|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(2)].copy();us_seal=sec[us_bucket.ge(2)].copy()

    naver_paths=(
        "events_naver_news_real.csv","events_naver_v7_h2_2025.csv",
        "events_naver_v8_unused.csv","events_naver_v9_reserved.csv",
        "events_naver_v10_fresh.csv","events_naver_v11_reserved.csv",
        "events_naver_v12_dev.csv","events_naver_v12_seal.csv",
        "events_naver_v13_dev.csv","events_naver_v13_seal.csv",
        "events_naver_v14_dev.csv","events_naver_v14_seal.csv",
    )
    kind=(pd.concat([pd.read_csv(DATA/name,dtype={"ticker":str})
                     for name in naver_paths if (DATA/name).exists()],
                    ignore_index=True,sort=False).drop_duplicates("event_id"))
    kind=kind[~kind.event_id.astype(str).isin(prior_ids)].copy()
    kind["event_time_utc"]=pd.to_datetime(kind.event_time_utc,utc=True)
    local=kind.event_time_utc.dt.tz_convert(app.KR_TZ)
    kind=kind[(local.dt.time>=pd.Timestamp("09:00").time())&
              (local.dt.time<=pd.Timestamp("14:57").time())&
              kind.headline.fillna("").str.len().ge(3)].copy()
    prior_kr=prior[prior.market.eq("KR")]
    prior_times={
        ticker:np.sort(group.event_time_utc.astype("int64").to_numpy())
        for ticker,group in prior_kr.groupby("ticker")
    }
    embargo_ns=int(pd.Timedelta(minutes=35).value);keep=np.ones(len(kind),dtype=bool)
    for position,row in enumerate(kind.itertuples(index=False)):
        values=prior_times.get(str(row.ticker))
        if values is None or not len(values):continue
        target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(values,target));near=[]
        if index<len(values):near.append(abs(int(values[index])-target))
        if index:near.append(abs(int(values[index-1])-target))
        if near and min(near)<=embargo_ns:keep[position]=False
    kind=kind.iloc[np.flatnonzero(keep)].copy()
    # Also keep V20 result windows disjoint within each ticker.
    accepted=[]
    for _,group in kind.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
        last=None
        for index,timestamp in zip(group.index,group.event_time_utc):
            if last is None or timestamp-last>pd.Timedelta(minutes=35):
                accepted.append(index);last=timestamp
    kind=kind.loc[accepted].sort_values("event_time_utc").copy()
    split_key=kind.get("article_id",kind.event_id).fillna(kind.event_id).astype(str)
    kr_bucket=split_key.map(
        lambda value:int(hashlib.sha256(f"V20|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kind[kr_bucket.lt(2)].copy();kr_seal=kind[kr_bucket.ge(2)].copy()

    for frame,source,path in (
        (us_dev,"SEC_V20_DEV",DATA/"events_sec_v20_dev.csv"),
        (us_seal,"SEC_V20_SEAL",DATA/"events_sec_v20_seal.csv"),
        (kr_dev,"NAVER_NEWS_V20_DEV",DATA/"events_naver_v20_dev.csv"),
        (kr_seal,"NAVER_NEWS_V20_SEAL",DATA/"events_naver_v20_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,
        "split":"SHA256(V20|event_id/article_id) mod 10; DEV<2",
        "us_price_availability_filter":"monthly archive symbol presence only",
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,"prior_outcome_embargo_minutes":35,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V20_SOURCE_STATUS.json")
    if article_overlap:
        raise RuntimeError("V20 KR article leakage across DEV/SEAL")
    print(f"[V20 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v21_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze new-ticker SEC/Naver candidates before any V21 labels exist."""
    prior=pd.read_csv(DATA/"events_exact_v20.csv.gz",dtype={"ticker":str},usecols=["event_id"])
    prior_ids=set(prior.event_id.astype(str))
    sec=pd.read_csv(DATA/"events_sec_v21_v15_historical_raw.csv",dtype={"ticker":str})
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].drop_duplicates("event_id").copy()
    price_status=json.loads((DATA/"US_PRICE_STATUS.json").read_text(encoding="utf-8"))
    sec=sec[~sec.ticker.astype(str).isin(set(map(str,price_status.get("missing",[]))))].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V21|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(2)].copy();us_seal=sec[us_bucket.ge(2)].copy()

    naver=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str})
    naver=naver[~naver.event_id.astype(str).isin(prior_ids)].drop_duplicates("event_id").copy()
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V21|CAP|{value}".encode()).hexdigest()
    )
    naver=(naver.sort_values(["ticker","selection_hash"])
           .groupby("ticker",group_keys=False).head(20).drop(columns="selection_hash"))
    article_key=naver.get("article_id",naver.event_id).fillna(naver.event_id).astype(str)
    kr_bucket=article_key.map(
        lambda value:int(hashlib.sha256(f"V21|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=naver[kr_bucket.lt(7)].copy();kr_seal=naver[kr_bucket.ge(7)].copy()
    for frame,source,path in (
        (us_dev,"SEC_V21_DEV",DATA/"events_sec_v21_dev.csv"),
        (us_seal,"SEC_V21_SEAL",DATA/"events_sec_v21_seal.csv"),
        (kr_dev,"NAVER_NEWS_V21_DEV",DATA/"events_naver_v21_dev.csv"),
        (kr_seal,"NAVER_NEWS_V21_SEAL",DATA/"events_naver_v21_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,"us_split":"SHA256(V21|event_id) mod10; DEV<2",
        "kr_split":"SHA256(V21|ARTICLE|article_id) mod10; DEV<7",
        "kr_max_events_per_ticker":20,"us_price_filter":"archive symbol presence only",
        "us_dev":len(us_dev),"us_seal":len(us_seal),"kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,"kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V21_SOURCE_STATUS.json")
    if article_overlap:raise RuntimeError("V21 KR article leakage")
    print(f"[V21 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v22_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze unused V15 SEC history and uncapped V21 Naver residuals."""
    prior=pd.read_csv(DATA/"events_exact_v21.csv.gz",dtype={"ticker":str},usecols=["event_id"])
    prior_ids=set(prior.event_id.astype(str))
    sec=pd.read_csv(DATA/"events_sec_v22_v15_historical_raw.csv",dtype={"ticker":str})
    sec=sec[~sec.event_id.astype(str).isin(prior_ids)].drop_duplicates("event_id").copy()
    price_status=json.loads((DATA/"US_PRICE_STATUS.json").read_text(encoding="utf-8"))
    sec=sec[~sec.ticker.astype(str).isin(set(map(str,price_status.get("missing",[]))))].copy()
    sec_all=sec.copy()
    sec["selection_hash"]=sec.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V22|CAP|{value}".encode()).hexdigest())
    sec=(sec.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
         .head(30).drop(columns="selection_hash"))
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V22|{value}".encode()).hexdigest(),16)%10)
    us_dev=sec[us_bucket.lt(7)].copy();us_seal=sec[us_bucket.ge(7)].copy()
    # Label-availability auditing showed the original sealed raw list would
    # miss the predeclared US n>=250 requirement.  Do not move opened DEV rows:
    # add only never-selected raw IDs, capped at five per ticker by a fresh hash.
    extra=sec_all[~sec_all.event_id.astype(str).isin(set(sec.event_id.astype(str)))].copy()
    extra["selection_hash"]=extra.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V22|SEAL_EXTRA|{value}".encode()).hexdigest())
    extra=(extra.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
           .head(5).drop(columns="selection_hash"))
    us_seal=pd.concat([us_seal,extra],ignore_index=True,sort=False).drop_duplicates("event_id")

    naver=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str})
    naver=naver[~naver.event_id.astype(str).isin(prior_ids)].drop_duplicates("event_id").copy()
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V22|CAP|{value}".encode()).hexdigest())
    naver=(naver.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
           .head(15).drop(columns="selection_hash"))
    article_key=naver.get("article_id",naver.event_id).fillna(naver.event_id).astype(str)
    kr_bucket=article_key.map(
        lambda value:int(hashlib.sha256(f"V22|ARTICLE|{value}".encode()).hexdigest(),16)%10)
    kr_dev=naver[kr_bucket.lt(7)].copy();kr_seal=naver[kr_bucket.ge(7)].copy()
    for frame,source,path in (
        (us_dev,"SEC_V22_DEV",DATA/"events_sec_v22_dev.csv"),
        (us_seal,"SEC_V22_SEAL",DATA/"events_sec_v22_seal.csv"),
        (kr_dev,"NAVER_NEWS_V22_DEV",DATA/"events_naver_v22_dev.csv"),
        (kr_seal,"NAVER_NEWS_V22_SEAL",DATA/"events_naver_v22_seal.csv"),
    ):
        frame["source"]=source;frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    _atomic_json({"selection_uses_labels":False,"split":"SHA256(V22|event/article) mod10; DEV<7",
                  "us_max_events_per_ticker":30,"us_seal_extra_max_per_ticker":5,
                  "us_seal_extra_selection":"never-selected raw IDs; SHA256(V22|SEAL_EXTRA|event_id)",
                  "kr_max_events_per_ticker":15,
                  "us_price_filter":"archive symbol presence only","us_dev":len(us_dev),"us_seal":len(us_seal),
                  "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),"prior_event_id_overlap":0,
                  "kr_dev_seal_article_overlap":article_overlap},DATA/"V22_SOURCE_STATUS.json")
    if article_overlap:raise RuntimeError("V22 KR article leakage")
    print(f"[V22 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v23_sources() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze fresh residual SEC/Naver events without reading any outcomes."""
    prior=pd.read_csv(
        DATA/"events_exact_v22.csv.gz",dtype={"ticker":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={
        (market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
        for (market,ticker),group in prior.groupby(["market","ticker"])
    }
    embargo_ns=int(pd.Timedelta(minutes=35).value)

    def outside_prior(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy()
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            values=prior_times.get((market,str(row.ticker)))
            if values is None or not len(values):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(values,target));near=[]
            if index<len(values):near.append(abs(int(values[index])-target))
            if index:near.append(abs(int(values[index-1])-target))
            if near and min(near)<=embargo_ns:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def self_space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(
            ["ticker","event_time_utc","event_id"]
        ).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].sort_values("event_time_utc").copy()

    sec_paths=[
        path for path in sorted(DATA.glob("events_sec*.csv"))
        if "_v23_" not in path.name.lower() or
           path.name=="events_sec_v23_expanded_raw.csv"
    ]
    sec_frames=[]
    required={"event_id","ticker","event_time_utc"}
    for path in sec_paths:
        try:frame=pd.read_csv(path,dtype={"ticker":str})
        except (pd.errors.EmptyDataError,UnicodeDecodeError):continue
        if required.issubset(frame.columns):
            sec_frames.append(frame)
    sec=(pd.concat(sec_frames,ignore_index=True,sort=False)
         .drop_duplicates("event_id",keep="first"))
    sec=outside_prior(sec,"US")
    sec["market"]="US"
    sec["headline"]=sec.get("headline","").fillna("").astype(str)
    sec=sec[sec.headline.str.len().ge(3)].copy()
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=sec[(local.dt.time>=pd.Timestamp("09:30").time())&
            (local.dt.time<=pd.Timestamp("15:27").time())].copy()
    _atomic_csv(
        sec.sort_values("event_time_utc"),
        DATA/"events_sec_v23_price_candidates.csv",
    )

    # Parquet timestamp bounds are entry-time information only.  This avoids
    # hashing candidates whose local archive cannot possibly contain t+32m.
    bounds:dict[str,tuple[pd.Timestamp,pd.Timestamp]]={}
    for ticker in sorted(sec.ticker.astype(str).unique()):
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        parquet=pq.ParquetFile(path)
        try:column_index=parquet.schema_arrow.names.index("timestamp")
        except ValueError:continue
        minima=[];maxima=[]
        for group_index in range(parquet.num_row_groups):
            stats=parquet.metadata.row_group(group_index).column(column_index).statistics
            if stats is not None and stats.has_min_max:
                minima.append(pd.Timestamp(stats.min));maxima.append(pd.Timestamp(stats.max))
        if minima:
            lower=min(minima);upper=max(maxima)
            lower=(lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC"))
            upper=(upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC"))
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and
        bounds[str(row.ticker)][0]<=pd.Timestamp(row.event_time_utc) and
        bounds[str(row.ticker)][1]>=pd.Timestamp(row.event_time_utc)+pd.Timedelta(minutes=32)
        for row in sec.itertuples(index=False)
    ],dtype=bool)
    sec=self_space(sec.iloc[np.flatnonzero(covered)].copy())
    sec["selection_hash"]=sec.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V23|US|CAP|{value}".encode()).hexdigest()
    )
    sec=(sec.sort_values(["ticker","selection_hash"])
         .groupby("ticker",group_keys=False).head(100).drop(columns="selection_hash"))
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V23|US|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(4)].copy();us_seal=sec[us_bucket.ge(4)].copy()

    naver=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str})
    naver=outside_prior(naver.drop_duplicates("event_id"),"KR")
    prior_articles=set(
        prior.loc[prior.market.eq("KR")&prior.get("article_id",pd.Series(index=prior.index,dtype=object)).notna(),
                  "article_id"].astype(str)
    ) if "article_id" in prior.columns else set()
    if "article_id" in naver.columns:
        naver=naver[~naver.article_id.astype(str).isin(prior_articles)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    naver=naver[(local.dt.time>=pd.Timestamp("09:00").time())&
                (local.dt.time<=pd.Timestamp("14:57").time())&
                naver.headline.fillna("").str.len().ge(3)].copy()
    naver=self_space(naver)
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V23|KR|CAP|{value}".encode()).hexdigest()
    )
    naver=(naver.sort_values(["ticker","selection_hash"])
           .groupby("ticker",group_keys=False).head(10).drop(columns="selection_hash"))
    article_key=naver.get("article_id",naver.event_id).fillna(naver.event_id).astype(str)
    kr_bucket=article_key.map(
        lambda value:int(hashlib.sha256(f"V23|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=naver[kr_bucket.lt(5)].copy();kr_seal=naver[kr_bucket.ge(5)].copy()

    for frame,source,path in (
        (us_dev,"SEC_V23_DEV",DATA/"events_sec_v23_dev.csv"),
        (us_seal,"SEC_V23_SEAL",DATA/"events_sec_v23_seal.csv"),
        (kr_dev,"NAVER_NEWS_V23_DEV",DATA/"events_naver_v23_dev.csv"),
        (kr_seal,"NAVER_NEWS_V23_SEAL",DATA/"events_naver_v23_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=(
        len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
        if "article_id" in kr_dev.columns else 0
    )
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,
        "split":"US SHA256 mod10 DEV<4; KR article SHA256 mod10 DEV<5",
        "prior_version":"V22","prior_event_id_overlap":0,
        "prior_outcome_embargo_minutes":35,
        "within_ticker_result_window_spacing_minutes":35,
        "us_raw_files":[path.name for path in sec_paths],
        "us_price_filter":"local parquet timestamp bounds only",
        "us_max_events_per_ticker":100,"kr_max_events_per_ticker":10,
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "us_dev_seal_event_overlap":event_overlap,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V23_SOURCE_STATUS.json")
    if event_overlap or article_overlap:
        raise RuntimeError("V23 DEV/SEAL leakage")
    print(f"[V23 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v23_us_universe()->pd.DataFrame:
    """Reuse only V15 life-science tickers with verified historical prices."""
    universe=pd.read_csv(DATA/"universe_v15_us.csv",dtype={"ticker":str})
    prior_sources=pd.concat([
        pd.read_csv(DATA/"events_sec_v22_dev.csv",dtype={"ticker":str}),
        pd.read_csv(DATA/"events_sec_v22_seal.csv",dtype={"ticker":str}),
    ],ignore_index=True,sort=False)
    verified=set(prior_sources.ticker.astype(str))
    universe=universe[universe.ticker.astype(str).isin(verified)].copy()
    universe=universe.sort_values("ticker").drop_duplicates("ticker")
    _atomic_csv(universe,DATA/"universe_v23_us.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"V15 life-science tickers with V22 historical archive presence",
        "tickers":len(universe),
    },DATA/"V23_US_UNIVERSE_STATUS.json")
    print(f"[V23 US UNIVERSE] tickers={len(universe)}")
    return universe


def collect_v23_expanded_sec(force:bool=False)->pd.DataFrame:
    universe_path=DATA/"universe_v23_us.csv"
    if not universe_path.exists():prepare_v23_us_universe()
    return collect_sec(
        "2006-01-01","2018-12-31",force=force,
        output_name="events_sec_v23_expanded_raw.csv",
        source_name="SEC_V23_EXPANDED_RAW",
        one_per_ticker_day=True,max_events_per_ticker=200,
        exclude_event_path=DATA/"events_exact_v22.csv.gz",exclude_minutes=35,
        universe_path=universe_path,material_only=False,fetch_body=False,
    )


def collect_v23_us_prices(force:bool=False)->None:
    """Extract only V23 candidate ticker-days from the local HF month archive."""
    event_path=DATA/"events_sec_v23_price_candidates.csv"
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    filtered_by_month:dict[str,Path]={}
    month_tickers={
        month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
        for month in months
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={
            executor.submit(_filter_hf_month,month,month_tickers[month],force):month
            for month in months
        }
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered_by_month[month]=future.result()
            print(f"[V23 HF] filtered {done}/{len(months)}: {month}")

    pieces:dict[str,list[pd.DataFrame]]={}
    available_days=0
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered_by_month[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str)
        frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={
            ticker:set(group.event_day.astype(str))
            for ticker,group in month_events.groupby("ticker")
        }
        union_days=set(month_events.event_day.astype(str))
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union_days if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":available_days+=selected.timestamp.dt.date.nunique()
        print(f"[V23 HF] day extraction {done}/{len(months)}: {month}")

    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet"
        combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():
            combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=(combined.sort_values("timestamp")
                  .drop_duplicates("timestamp",keep="last"))
        temporary=path.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary,index=False,compression="zstd")
        temporary.replace(path)
    missing=sorted(set(events.ticker.astype(str))-set(pieces))
    _atomic_json({
        "selection_uses_labels":False,
        "source":"local ggaddam/OHLCV-1m monthly parquet cache",
        "candidate_events":len(events),"candidate_tickers":events.ticker.nunique(),
        "months":months,"written_tickers":len(set(pieces)-{"SPY"}),
        "missing_tickers":missing,"available_ticker_days":available_days,
    },DATA/"V23_US_PRICE_STATUS.json")
    print(f"[V23 HF] written tickers={len(set(pieces)-{'SPY'})}, missing={len(missing)}")


def prepare_v24_us_universe()->pd.DataFrame:
    """All local price tickers inherited from prior bio universes and in SEC."""
    local={path.stem for path in (PRICE/"US").glob("*.parquet") if path.stem!="SPY"}
    collector=app.SECCollector(app.Config())
    ticker_map=collector.get_json(collector.TICKER_MAP)
    rows=[]
    for item in ticker_map.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        if ticker in local and company and re.fullmatch(r"[A-Z]{1,5}",ticker):
            rows.append({"market":"US","ticker":ticker,"company":company})
    universe=pd.DataFrame(rows).drop_duplicates("ticker")
    universe["selection_hash"]=universe.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V24|UNIVERSE|{value}".encode()).hexdigest()
    )
    universe=(universe.sort_values("selection_hash").drop(columns="selection_hash")
              .reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v24_us.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"existing local ticker price files inherited exclusively from prior bio universes; current SEC mapping",
        "tickers":len(universe),
    },DATA/"V24_US_UNIVERSE_STATUS.json")
    print(f"[V24 US UNIVERSE] tickers={len(universe)}")
    return universe


def collect_v24_fresh_sec(force:bool=False)->pd.DataFrame:
    """Collect exact-time filings never used through the opened V23 seal."""
    universe_path=DATA/"universe_v24_us.csv"
    if not universe_path.exists():prepare_v24_us_universe()
    return collect_sec(
        "2019-01-01","2025-05-30",force=force,
        output_name="events_sec_v24_fresh_raw.csv",
        source_name="SEC_V24_FRESH_RAW",one_per_ticker_day=True,
        max_events_per_ticker=40,exclude_event_path=DATA/"events_exact_v23.csv.gz",
        exclude_minutes=35,universe_path=universe_path,
        material_only=False,fetch_body=False,
    )


def collect_v24_historical_sec(force:bool=False)->pd.DataFrame:
    """Collect unused 2006-2018 exact filings for the expanded local universe."""
    universe_path=DATA/"universe_v24_us.csv"
    if not universe_path.exists():prepare_v24_us_universe()
    return collect_sec(
        "2006-01-01","2018-12-31",force=force,
        output_name="events_sec_v24_historical_raw.csv",
        source_name="SEC_V24_HISTORICAL_RAW",one_per_ticker_day=True,
        max_events_per_ticker=60,exclude_event_path=DATA/"events_exact_v23.csv.gz",
        exclude_minutes=35,universe_path=universe_path,
        material_only=False,fetch_body=False,
    )


def collect_v24_us_prices(force:bool=False)->None:
    """Extract V24 candidate ticker-days from the local monthly archive."""
    event_paths=[
        DATA/"events_sec_v24_fresh_raw.csv",
        DATA/"events_sec_v24_historical_raw.csv",
    ]
    missing_paths=[path for path in event_paths if not path.exists()]
    if missing_paths:raise FileNotFoundError(str(missing_paths))
    events=(pd.concat([pd.read_csv(path,dtype={"ticker":str}) for path in event_paths],
                      ignore_index=True,sort=False).drop_duplicates("event_id"))
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    filtered_by_month:dict[str,Path]={}
    month_tickers={
        month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
        for month in months
    }
    with ThreadPoolExecutor(max_workers=min(8,len(months))) as executor:
        futures={
            executor.submit(_filter_hf_month,month,month_tickers[month],force):month
            for month in months
        }
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered_by_month[month]=future.result()
            print(f"[V24 HF] filtered {done}/{len(months)}: {month}")

    pieces:dict[str,list[pd.DataFrame]]={};available_days=0
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered_by_month[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str)
        frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={ticker:set(group.event_day.astype(str)) for ticker,group in month_events.groupby("ticker")}
        union_days=set(month_events.event_day.astype(str))
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union_days if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":available_days+=selected.timestamp.dt.date.nunique()
        print(f"[V24 HF] day extraction {done}/{len(months)}: {month}")

    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet"
        combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=combined.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
        temporary=path.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
    missing=sorted(set(events.ticker.astype(str))-set(pieces))
    _atomic_json({
        "selection_uses_labels":False,"source":"local ggaddam/OHLCV-1m monthly parquet cache",
        "candidate_events":len(events),"candidate_tickers":events.ticker.nunique(),
        "months":months,"written_tickers":len(set(pieces)-{"SPY"}),
        "missing_tickers":missing,"available_ticker_days":available_days,
    },DATA/"V24_US_PRICE_STATUS.json")
    print(f"[V24 HF] written tickers={len(set(pieces)-{'SPY'})}, missing={len(missing)}")


def prepare_v24_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze fresh V24 DEV/SEAL source IDs without consulting outcomes."""
    prior=pd.read_csv(
        DATA/"events_exact_v23.csv.gz",dtype={"ticker":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={
        (market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
        for (market,ticker),group in prior.groupby(["market","ticker"])
    }
    embargo_ns=int(pd.Timedelta(minutes=35).value)

    def outside_prior(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy()
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            values=prior_times.get((market,str(row.ticker)))
            if values is None or not len(values):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(values,target));near=[]
            if index<len(values):near.append(abs(int(values[index])-target))
            if index:near.append(abs(int(values[index-1])-target))
            if near and min(near)<=embargo_ns:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def self_space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(
            ["ticker","event_time_utc","event_id"]
        ).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].sort_values("event_time_utc").copy()

    sec_frames=[]
    for path in (
        DATA/"events_sec_v24_fresh_raw.csv",
        DATA/"events_sec_v24_historical_raw.csv",
        DATA/"events_sec_v23_price_candidates.csv",
        DATA/"events_us_news_real.csv",
    ):
        if path.exists():sec_frames.append(pd.read_csv(path,dtype={"ticker":str}))
    if not sec_frames:raise RuntimeError("V24 SEC raw sources are missing")
    sec=(pd.concat(sec_frames,ignore_index=True,sort=False)
         .drop_duplicates("event_id",keep="first"))
    sec=outside_prior(sec,"US");sec["market"]="US"
    sec["headline"]=sec.get("headline","").fillna("").astype(str)
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=sec[(local.dt.time>=pd.Timestamp("09:30").time())&
            (local.dt.time<=pd.Timestamp("15:27").time())&
            sec.headline.str.len().ge(3)].copy()

    # Timestamp bounds are label-independent structural price availability.
    bounds:dict[str,tuple[pd.Timestamp,pd.Timestamp]]={}
    for ticker in sorted(sec.ticker.astype(str).unique()):
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        parquet=pq.ParquetFile(path)
        try:column_index=parquet.schema_arrow.names.index("timestamp")
        except ValueError:continue
        minima=[];maxima=[]
        for group_index in range(parquet.num_row_groups):
            stats=parquet.metadata.row_group(group_index).column(column_index).statistics
            if stats is not None and stats.has_min_max:
                minima.append(pd.Timestamp(stats.min));maxima.append(pd.Timestamp(stats.max))
        if minima:
            lower=min(minima);upper=max(maxima)
            lower=lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
            upper=upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and
        bounds[str(row.ticker)][0]<=pd.Timestamp(row.event_time_utc) and
        bounds[str(row.ticker)][1]>=pd.Timestamp(row.event_time_utc)+pd.Timedelta(minutes=32)
        for row in sec.itertuples(index=False)
    ],dtype=bool)
    sec=self_space(sec.iloc[np.flatnonzero(covered)].copy())
    sec["selection_hash"]=sec.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V24|US|CAP|{value}".encode()).hexdigest()
    )
    sec=(sec.sort_values(["ticker","selection_hash"])
         .groupby("ticker",group_keys=False).head(100).drop(columns="selection_hash"))
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V24|US|{value}".encode()).hexdigest(),16)%20
    )
    us_dev=sec[us_bucket.lt(4)].copy();us_seal=sec[us_bucket.ge(4)].copy()

    naver=pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str})
    naver=outside_prior(naver.drop_duplicates("event_id"),"KR")
    prior_articles=(set(prior.loc[prior.market.eq("KR")&prior.article_id.notna(),
                                  "article_id"].astype(str))
                    if "article_id" in prior.columns else set())
    if "article_id" in naver.columns:
        naver=naver[~naver.article_id.astype(str).isin(prior_articles)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    naver=naver[(local.dt.time>=pd.Timestamp("09:00").time())&
                (local.dt.time<=pd.Timestamp("14:57").time())&
                naver.headline.fillna("").str.len().ge(3)].copy()
    naver=self_space(naver)
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V24|KR|CAP|{value}".encode()).hexdigest()
    )
    naver=(naver.sort_values(["ticker","selection_hash"])
           .groupby("ticker",group_keys=False).head(20).drop(columns="selection_hash"))
    article_key=naver.get("article_id",naver.event_id).fillna(naver.event_id).astype(str)
    kr_bucket=article_key.map(
        lambda value:int(hashlib.sha256(f"V24|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=naver[kr_bucket.lt(5)].copy();kr_seal=naver[kr_bucket.ge(5)].copy()

    for frame,source,path in (
        (us_dev,"SEC_V24_DEV",DATA/"events_sec_v24_dev.csv"),
        (us_seal,"SEC_V24_SEAL",DATA/"events_sec_v24_seal.csv"),
        (kr_dev,"NAVER_NEWS_V24_DEV",DATA/"events_naver_v24_dev.csv"),
        (kr_seal,"NAVER_NEWS_V24_SEAL",DATA/"events_naver_v24_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=(len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
                     if "article_id" in kr_dev.columns else 0)
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,
        "split":"US SHA256 mod20 DEV<4; KR article SHA256 mod10 DEV<5",
        "prior_version":"V23","prior_event_id_overlap":0,
        "prior_outcome_embargo_minutes":35,
        "within_ticker_result_window_spacing_minutes":35,
        "us_price_filter":"local parquet timestamp bounds only",
        "us_raw_sources":["fresh SEC","historical SEC residual","Reuters QIU exact-time news residual"],
        "us_max_events_per_ticker":100,"kr_max_events_per_ticker":20,
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "us_dev_seal_event_overlap":event_overlap,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V24_SOURCE_STATUS.json")
    if event_overlap or article_overlap:raise RuntimeError("V24 DEV/SEAL leakage")
    print(f"[V24 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def prepare_v25_us_universe()->pd.DataFrame:
    """SEC bio-title tickers seen in raw prices but absent through V24."""
    archive_tickers:set[str]=set()
    sample_months=("2019-01","2021-01","2023-01","2024-12","2025-05")
    for month in sample_months:
        table=pq.read_table(HF_RAW_CACHE/f"ohlcv_{month}.parquet",columns=["ticker"])
        archive_tickers.update(pc.unique(table["ticker"]).to_pylist())
    collector=app.SECCollector(app.Config());ticker_map=collector.get_json(collector.TICKER_MAP)
    pattern=re.compile(
        r"(?:BIO|PHARMA|THERAPEUT|MEDIC|HEALTH|LIFE SCI|ONCO|GENET|GENOMIC|"
        r"VACCIN|DIAGNOST|NEURO|IMMUN|OPHTHAL|CARDIO|SURGICAL|LABORATOR)",re.IGNORECASE,
    )
    extra_health_tickers={
        "ADUS","TSHA","GHRS","TNDM","SMTI","ADAG","MOLN",
        "VRHI","AMS","EVGN","TUTH","BZYR","DMAA","QPRC",
    }
    rows=[]
    for item in ticker_map.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        if (ticker in archive_tickers and company and
                (pattern.search(company) or ticker in extra_health_tickers) and
                re.fullmatch(r"[A-Z]{1,5}",ticker)):
            rows.append({"market":"US","ticker":ticker,"company":company})
    universe=pd.DataFrame(rows).drop_duplicates("ticker")
    prior=pd.read_csv(DATA/"events_exact_v24.csv.gz",dtype={"ticker":str},usecols=["market","ticker"])
    used=set(prior.loc[prior.market.eq("US"),"ticker"].astype(str))
    universe=universe[~universe.ticker.astype(str).isin(used)].copy()
    universe["selection_hash"]=universe.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V25|UNIVERSE|{value}".encode()).hexdigest()
    )
    universe=(universe.sort_values("selection_hash").drop(columns="selection_hash")
              .reset_index(drop=True));universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v25_us.csv")
    _atomic_json({"selection_uses_labels":False,"sample_months":list(sample_months),
                  "definition":"SEC bio company-title; ticker present in raw archive; absent through V24",
                  "explicit_additional_health_tickers":sorted(extra_health_tickers),
                  "tickers":len(universe)},DATA/"V25_US_UNIVERSE_STATUS.json")
    print(f"[V25 US UNIVERSE] tickers={len(universe)}")
    return universe


def collect_v25_sec(force:bool=False)->pd.DataFrame:
    universe_path=DATA/"universe_v25_us.csv"
    if not universe_path.exists():prepare_v25_us_universe()
    return collect_sec(
        "2019-01-01","2025-05-30",force=force,
        output_name="events_sec_v25_raw.csv",source_name="SEC_V25_RAW",
        one_per_ticker_day=True,max_events_per_ticker=80,
        exclude_event_path=DATA/"events_exact_v24.csv.gz",exclude_minutes=35,
        universe_path=universe_path,material_only=False,fetch_body=False,
    )


def collect_v25_historical_sec(force:bool=False)->pd.DataFrame:
    universe_path=DATA/"universe_v25_us.csv"
    if not universe_path.exists():prepare_v25_us_universe()
    return collect_sec(
        "2006-01-01","2018-12-31",force=force,
        output_name="events_sec_v25_historical_raw.csv",source_name="SEC_V25_HISTORICAL_RAW",
        one_per_ticker_day=True,max_events_per_ticker=100,
        exclude_event_path=DATA/"events_exact_v24.csv.gz",exclude_minutes=35,
        universe_path=universe_path,material_only=False,fetch_body=False,
    )


def collect_v25_dense_sec(force:bool=False)->pd.DataFrame:
    """All spaced-eligible metadata forms; final spacing occurs before split."""
    universe_path=DATA/"universe_v25_us.csv"
    if not universe_path.exists():prepare_v25_us_universe()
    return collect_sec(
        "2006-01-01","2025-05-30",force=force,
        output_name="events_sec_v25_dense_raw.csv",source_name="SEC_V25_DENSE_RAW",
        one_per_ticker_day=False,max_events_per_ticker=250,
        exclude_event_path=DATA/"events_exact_v24.csv.gz",exclude_minutes=35,
        universe_path=universe_path,material_only=False,fetch_body=False,
    )


def collect_v25_us_prices(force:bool=False)->None:
    event_paths=[DATA/"events_sec_v25_raw.csv",DATA/"events_sec_v25_historical_raw.csv",
                 DATA/"events_sec_v25_dense_raw.csv"]
    missing_paths=[path for path in event_paths if not path.exists()]
    if missing_paths:raise FileNotFoundError(str(missing_paths))
    events=(pd.concat([pd.read_csv(path,dtype={"ticker":str}) for path in event_paths],
                      ignore_index=True,sort=False).drop_duplicates("event_id"))
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique());filtered_by_month:dict[str,Path]={}
    month_tickers={month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
                   for month in months}
    with ThreadPoolExecutor(max_workers=min(8,len(months))) as executor:
        futures={executor.submit(_filter_hf_month,month,month_tickers[month],force):month
                 for month in months}
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered_by_month[month]=future.result()
            print(f"[V25 HF] filtered {done}/{len(months)}: {month}")
    pieces:dict[str,list[pd.DataFrame]]={};available_days=0
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered_by_month[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str);frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={ticker:set(group.event_day.astype(str)) for ticker,group in month_events.groupby("ticker")}
        union_days=set(month_events.event_day.astype(str))
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union_days if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":available_days+=selected.timestamp.dt.date.nunique()
        print(f"[V25 HF] day extraction {done}/{len(months)}: {month}")
    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet";combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=combined.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
        temporary=path.with_suffix(".parquet.tmp");combined.to_parquet(temporary,index=False,compression="zstd")
        temporary.replace(path)
    missing=sorted(set(events.ticker.astype(str))-set(pieces))
    _atomic_json({"selection_uses_labels":False,"candidate_events":len(events),
                  "candidate_tickers":events.ticker.nunique(),"months":months,
                  "written_tickers":len(set(pieces)-{"SPY"}),"missing_tickers":missing,
                  "available_ticker_days":available_days},DATA/"V25_US_PRICE_STATUS.json")
    print(f"[V25 HF] written tickers={len(set(pieces)-{'SPY'})}, missing={len(missing)}")


def prepare_v25_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    prior=pd.read_csv(DATA/"events_exact_v24.csv.gz",dtype={"ticker":str},
                      usecols=lambda c:c in {"event_id","market","ticker","event_time_utc","article_id"})
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str));embargo=int(pd.Timedelta(minutes=35).value)
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            values=prior_times.get((market,str(row.ticker)))
            if values is None or not len(values):continue
            target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(values,target));near=[]
            if index<len(values):near.append(abs(int(values[index])-target))
            if index:near.append(abs(int(values[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    sec=outside(pd.concat([
        pd.read_csv(DATA/"events_sec_v25_raw.csv",dtype={"ticker":str}),
        pd.read_csv(DATA/"events_sec_v25_historical_raw.csv",dtype={"ticker":str}),
        pd.read_csv(DATA/"events_sec_v25_dense_raw.csv",dtype={"ticker":str}),
    ],ignore_index=True,sort=False).drop_duplicates("event_id"),"US")
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=space(sec[(local.dt.time>=pd.Timestamp("09:30").time())&
                  (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    # Local file existence and timestamp bounds only; no price direction.
    bounds={}
    for ticker in sec.ticker.astype(str).unique():
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        pf=pq.ParquetFile(path)
        try:index=pf.schema_arrow.names.index("timestamp")
        except ValueError:continue
        lo=[];hi=[]
        for group in range(pf.num_row_groups):
            stats=pf.metadata.row_group(group).column(index).statistics
            if stats is not None and stats.has_min_max:lo.append(pd.Timestamp(stats.min));hi.append(pd.Timestamp(stats.max))
        if lo:
            lower=min(lo);upper=max(hi);lower=lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC");upper=upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC");bounds[ticker]=(lower,upper)
    covered=np.asarray([str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
                        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
                        for row in sec.itertuples(index=False)],bool)
    sec=sec.iloc[np.flatnonzero(covered)].copy();bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V25|US|{value}".encode()).hexdigest(),16)%10)
    us_dev=sec[bucket.lt(2)].copy();us_seal=sec[bucket.ge(2)].copy()

    naver=outside(pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str}).drop_duplicates("event_id"),"KR")
    prior_articles=set(prior.loc[prior.market.eq("KR")&prior.article_id.notna(),"article_id"].astype(str))
    naver=naver[~naver.article_id.astype(str).isin(prior_articles)].copy();local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    naver=space(naver[(local.dt.time>=pd.Timestamp("09:00").time())&
                      (local.dt.time<=pd.Timestamp("14:57").time())].copy())
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V25|KR|CAP|{value}".encode()).hexdigest())
    naver=(naver.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
           .head(40).drop(columns="selection_hash"))
    article=naver.article_id.fillna(naver.event_id).astype(str);kr_bucket=article.map(
        lambda value:int(hashlib.sha256(f"V25|ARTICLE|{value}".encode()).hexdigest(),16)%10)
    kr_dev=naver[kr_bucket.lt(5)].copy();kr_seal=naver[kr_bucket.ge(5)].copy()
    for frame,source,path in ((us_dev,"SEC_V25_DEV",DATA/"events_sec_v25_dev.csv"),
                              (us_seal,"SEC_V25_SEAL",DATA/"events_sec_v25_seal.csv"),
                              (kr_dev,"NAVER_NEWS_V25_DEV",DATA/"events_naver_v25_dev.csv"),
                              (kr_seal,"NAVER_NEWS_V25_SEAL",DATA/"events_naver_v25_seal.csv")):
        frame["source"]=source;frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    article_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    _atomic_json({"selection_uses_labels":False,
                  "split":"US SHA256 mod10 DEV<2; KR article SHA256 mod10 DEV<5",
                  "prior_version":"V24","prior_event_id_overlap":0,"embargo_minutes":35,
                  "us_new_ticker_universe":True,"kr_max_events_per_ticker":40,
                  "us_dev":len(us_dev),"us_seal":len(us_seal),"kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
                  "us_dev_seal_event_overlap":event_overlap,"kr_dev_seal_article_overlap":article_overlap},
                 DATA/"V25_SOURCE_STATUS.json")
    if event_overlap or article_overlap:raise RuntimeError("V25 source leakage")
    print(f"[V25 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def build_panel() -> pd.DataFrame:
    cfg = app.Config()
    prior_event_path=DATA/"events_exact_v24.csv.gz"
    source_paths = [
        prior_event_path,
        DATA / "events_sec_v25_dev.csv",
        DATA / "events_sec_v25_seal.csv",
        DATA / "events_naver_v25_dev.csv",
        DATA / "events_naver_v25_seal.csv",
    ]
    frames = [pd.read_csv(path, dtype={"ticker": str}) for path in source_paths if path.exists()]
    if len(frames) != len(source_paths):
        missing = [str(path) for path in source_paths if not path.exists()]
        raise FileNotFoundError(f"V25 source event files missing: {missing}")
    events = app.combine_events(frames, replace(cfg,material_only=False))
    print(
        f"[EVENTS] combined={len(events)}, US={(events.market == 'US').sum()}, "
        f"KR={(events.market == 'KR').sum()}"
    )
    v25_sources={
        "SEC_V25_DEV","SEC_V25_SEAL","NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL",
    }
    new_labeled=app.build_labeled(events[events.source.isin(v25_sources)].copy(),cfg)
    prior_labeled_path=DATA/"labeled_events_v24.csv.gz"
    if not prior_labeled_path.exists():
        raise FileNotFoundError(prior_labeled_path)
    prior_labeled=pd.read_csv(prior_labeled_path,dtype={"ticker":str})
    prior_labeled["event_time_utc"]=pd.to_datetime(prior_labeled.event_time_utc,utc=True)
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior_labeled,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(app.LABELED_FILE,index=False,compression="gzip",encoding="utf-8")
    print(
        f"[LABELED] total={len(labeled)}, US={(labeled.market == 'US').sum()}, "
        f"KR={(labeled.market == 'KR').sum()}"
    )
    return labeled




def prepare_v26_us_universe(max_tickers:int=260)->pd.DataFrame:
    """Select locally price-covered SEC bio tickers without outcomes."""
    base=pd.read_csv(DATA/"universe_v24_us.csv",dtype={"ticker":str})
    base=base[base.market.eq("US")].copy()
    base["price_bytes"]=base.ticker.astype(str).map(
        lambda ticker:(PRICE/"US"/f"{ticker}.parquet").stat().st_size
        if (PRICE/"US"/f"{ticker}.parquet").exists() else 0
    )
    universe=(base[base.price_bytes.gt(0)]
              .sort_values(["price_bytes","ticker"],ascending=[False,True])
              .head(max_tickers).drop(columns="price_bytes").reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v26_us.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"largest inherited local minute-price files among V24 SEC bio universe",
        "max_tickers":max_tickers,"selected_tickers":len(universe),
    },DATA/"V26_US_UNIVERSE_STATUS.json")
    print(f"[V26 US UNIVERSE] selected={len(universe)}")
    return universe


def collect_v26_sec(force:bool=False)->pd.DataFrame:
    """Collect exact filings unused through V25, excluding a 35-minute embargo."""
    universe=DATA/"universe_v26_us.csv"
    if not universe.exists():prepare_v26_us_universe()
    return collect_sec(
        "2006-01-01","2025-05-30",force=force,
        output_name="events_sec_v26_raw.csv",source_name="SEC_V26_RAW",
        one_per_ticker_day=False,max_events_per_ticker=100,
        exclude_event_path=DATA/"events_exact_v25.csv.gz",exclude_minutes=35,
        universe_path=universe,material_only=False,fetch_body=False,
    )


def prepare_v26_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze label-blind V26 DEV/SEAL IDs from unused exact-time events."""
    prior=pd.read_csv(
        DATA/"events_exact_v25.csv.gz",dtype={"ticker":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str));embargo=int(pd.Timedelta(minutes=35).value)
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            values=prior_times.get((market,str(row.ticker)))
            if values is None or not len(values):continue
            target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(values,target));near=[]
            if index<len(values):near.append(abs(int(values[index])-target))
            if index:near.append(abs(int(values[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    sec=outside(pd.read_csv(DATA/"events_sec_v26_raw.csv",dtype={"ticker":str}).drop_duplicates("event_id"),"US")
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=space(sec[(local.dt.time>=pd.Timestamp("09:30").time())&
                  (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    bounds={}
    for ticker in sec.ticker.astype(str).unique():
        path=PRICE/"US"/f"{ticker}.parquet"
        if not path.exists():continue
        pf=pq.ParquetFile(path)
        try:index=pf.schema_arrow.names.index("timestamp")
        except ValueError:continue
        lo=[];hi=[]
        for group in range(pf.num_row_groups):
            stats=pf.metadata.row_group(group).column(index).statistics
            if stats is not None and stats.has_min_max:lo.append(pd.Timestamp(stats.min));hi.append(pd.Timestamp(stats.max))
        if lo:
            lower=min(lo);upper=max(hi)
            lower=lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
            upper=upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
            bounds[ticker]=(lower,upper)
    covered=np.asarray([str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
                        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
                        for row in sec.itertuples(index=False)],bool)
    sec=sec.iloc[np.flatnonzero(covered)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V26|US|{value}".encode()).hexdigest(),16)%10)
    us_dev=sec[us_bucket.lt(2)].copy();us_seal=sec[us_bucket.ge(2)].copy()

    naver=outside(pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",dtype={"ticker":str}).drop_duplicates("event_id"),"KR")
    prior_articles=set(prior.loc[prior.market.eq("KR")&prior.article_id.notna(),"article_id"].astype(str))
    naver=naver[~naver.article_id.astype(str).isin(prior_articles)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    naver=space(naver[(local.dt.time>=pd.Timestamp("09:00").time())&
                      (local.dt.time<=pd.Timestamp("14:57").time())].copy())
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V26|KR|CAP|{value}".encode()).hexdigest())
    naver=(naver.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
           .head(40).drop(columns="selection_hash"))
    article=naver.article_id.fillna(naver.event_id).astype(str)
    kr_bucket=article.map(lambda value:int(hashlib.sha256(f"V26|ARTICLE|{value}".encode()).hexdigest(),16)%10)
    kr_dev=naver[kr_bucket.lt(2)].copy();kr_seal=naver[kr_bucket.ge(2)].copy()
    for frame,source,path in (
        (us_dev,"SEC_V26_DEV",DATA/"events_sec_v26_dev.csv"),
        (us_seal,"SEC_V26_SEAL",DATA/"events_sec_v26_seal.csv"),
        (kr_dev,"NAVER_NEWS_V26_DEV",DATA/"events_naver_v26_dev.csv"),
        (kr_seal,"NAVER_NEWS_V26_SEAL",DATA/"events_naver_v26_seal.csv"),
    ):
        frame["source"]=source;frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    article_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V25","embargo_minutes":35,
        "split":"SHA256 mod10 DEV<2, SEAL>=2; KR article-grouped",
        "us_universe_definition":"local price file size among SEC bio universe",
        "kr_max_events_per_ticker":40,
        "us_dev":len(us_dev),"us_seal":len(us_seal),"kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":0,"us_dev_seal_event_overlap":event_overlap,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V26_SOURCE_STATUS.json")
    if event_overlap or article_overlap:raise RuntimeError("V26 source leakage")
    print(f"[V26 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def build_v26_panel()->pd.DataFrame:
    """Label V26 sources and append immutable opened V25 labeled history."""
    cfg=app.Config();source_paths=[
        DATA/"events_exact_v25.csv.gz",DATA/"events_sec_v26_dev.csv",
        DATA/"events_sec_v26_seal.csv",DATA/"events_naver_v26_dev.csv",
        DATA/"events_naver_v26_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str}) for path in source_paths]
    events=app.combine_events(frames,replace(cfg,material_only=False))
    event_path=DATA/"events_exact_v26.csv.gz"
    events.to_csv(event_path,index=False,compression="gzip",encoding="utf-8")
    v26_sources={"SEC_V26_DEV","SEC_V26_SEAL","NAVER_NEWS_V26_DEV","NAVER_NEWS_V26_SEAL"}
    new_labeled=app.build_labeled(events[events.source.isin(v26_sources)].copy(),cfg)
    prior=pd.read_csv(DATA/"labeled_events_v25.csv.gz",dtype={"ticker":str})
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last").sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v26.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=(new_labeled.groupby(["market","source"]).size().to_dict())
    _atomic_json({"new_labeled_counts":{f"{m}|{s}":int(n) for (m,s),n in counts.items()},
                  "total":len(labeled)},DATA/"V26_LABEL_STATUS.json")
    print(f"[V26 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}")
    return labeled




def prepare_v27_us_universe(max_tickers:int=400)->pd.DataFrame:
    """Select price-covered V24 bio tickers not used by the V26 collector."""
    base=pd.read_csv(DATA/"universe_v24_us.csv",dtype={"ticker":str})
    used=set(pd.read_csv(DATA/"universe_v26_us.csv",dtype={"ticker":str}).ticker.astype(str))
    base=base[base.market.eq("US")&~base.ticker.astype(str).isin(used)].copy()
    base["price_bytes"]=base.ticker.astype(str).map(
        lambda ticker:(PRICE/"US"/f"{ticker}.parquet").stat().st_size
        if (PRICE/"US"/f"{ticker}.parquet").exists() else 0
    )
    universe=(base[base.price_bytes.gt(0)]
              .sort_values(["price_bytes","ticker"],ascending=[False,True])
              .head(max_tickers).drop(columns="price_bytes").reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v27_us.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"price-covered inherited V24 SEC bio tickers excluded from V26 universe",
        "excluded_v26_tickers":len(used),"max_tickers":max_tickers,
        "selected_tickers":len(universe),
    },DATA/"V27_US_UNIVERSE_STATUS.json")
    print(f"[V27 US UNIVERSE] selected={len(universe)} excluded_v26={len(used)}")
    return universe


def collect_v27_sec(force:bool=False)->pd.DataFrame:
    """Collect exact filings unused through V26 with a 35-minute embargo."""
    universe=DATA/"universe_v27_us.csv"
    if not universe.exists():prepare_v27_us_universe()
    return collect_sec(
        "2006-01-01","2025-05-30",force=force,
        output_name="events_sec_v27_raw.csv",source_name="SEC_V27_RAW",
        one_per_ticker_day=False,max_events_per_ticker=120,
        exclude_event_path=DATA/"events_exact_v26.csv.gz",exclude_minutes=35,
        universe_path=universe,material_only=False,fetch_body=False,
    )


def prepare_v27_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze label-blind V27 IDs, isolated from all events through V26."""
    def article_key(value:Any)->str:
        if pd.isna(value):return ""
        text=str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():text=text[:-2]
        return str(int(text)) if text.isdigit() else text

    prior=pd.read_csv(
        DATA/"events_exact_v26.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    embargo=int(pd.Timedelta(minutes=35).value)
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy()
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            values=prior_times.get((market,str(row.ticker)))
            if values is None or not len(values):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(values,target));near=[]
            if index<len(values):near.append(abs(int(values[index])-target))
            if index:near.append(abs(int(values[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    sec=outside(pd.read_csv(DATA/"events_sec_v27_raw.csv",dtype={"ticker":str})
                .drop_duplicates("event_id"),"US")
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=space(sec[(local.dt.time>=pd.Timestamp("09:30").time())&
                  (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    bounds={}
    for ticker in sec.ticker.astype(str).unique():
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
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in sec.itertuples(index=False)
    ],bool)
    sec=sec.iloc[np.flatnonzero(covered)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V27|US|{value}".encode()).hexdigest(),16)%20
    )
    us_dev=sec[us_bucket.lt(3)].copy();us_seal=sec[us_bucket.ge(3)].copy()

    naver=outside(pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",
                             dtype={"ticker":str,"article_id":str})
                  .drop_duplicates("event_id"),"KR")
    prior_articles=set(prior.loc[
        prior.market.eq("KR")&prior.article_id.notna(),"article_id"
    ].map(article_key))
    naver_keys=naver.article_id.map(article_key)
    naver=naver[~naver_keys.isin(prior_articles)].copy()
    local=naver.event_time_utc.dt.tz_convert(app.KR_TZ)
    naver=space(naver[(local.dt.time>=pd.Timestamp("09:00").time())&
                      (local.dt.time<=pd.Timestamp("14:57").time())].copy())
    naver["selection_hash"]=naver.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V27|KR|CAP|{value}".encode()).hexdigest()
    )
    naver=(naver.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
           .head(80).drop(columns="selection_hash"))
    article=naver.article_id.map(article_key)
    article=article.where(article.ne(""),naver.event_id.astype(str))
    kr_bucket=article.map(
        lambda value:int(hashlib.sha256(f"V27|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=naver[kr_bucket.lt(2)].copy();kr_seal=naver[kr_bucket.ge(2)].copy()

    for frame,source,path in (
        (us_dev,"SEC_V27_DEV",DATA/"events_sec_v27_dev.csv"),
        (us_seal,"SEC_V27_SEAL",DATA/"events_sec_v27_seal.csv"),
        (kr_dev,"NAVER_NEWS_V27_DEV",DATA/"events_naver_v27_dev.csv"),
        (kr_seal,"NAVER_NEWS_V27_SEAL",DATA/"events_naver_v27_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    article_overlap=len(set(kr_dev.article_id.map(article_key))&
                        set(kr_seal.article_id.map(article_key)))
    prior_overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                      for frame in (us_dev,us_seal,kr_dev,kr_seal))
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V26","embargo_minutes":35,
        "split":"US SHA256 mod20 DEV<3; KR SHA256 mod10 DEV<2 article-grouped",
        "us_universe_definition":"price-covered V24 bio tickers absent from V26 universe",
        "kr_max_events_per_ticker":80,
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":prior_overlap,
        "us_dev_seal_event_overlap":event_overlap,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V27_SOURCE_STATUS.json")
    if prior_overlap or event_overlap or article_overlap:
        raise RuntimeError("V27 source leakage")
    print(f"[V27 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def build_v27_panel()->pd.DataFrame:
    """Label frozen V27 sources and append immutable opened V26 history."""
    cfg=app.Config()
    source_paths=[
        DATA/"events_exact_v26.csv.gz",DATA/"events_sec_v27_dev.csv",
        DATA/"events_sec_v27_seal.csv",DATA/"events_naver_v27_dev.csv",
        DATA/"events_naver_v27_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str}) for path in source_paths]
    active_event_file=app.EVENT_FILE
    app.EVENT_FILE=DATA/"events_exact_v27.csv.gz"
    try:
        events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:
        app.EVENT_FILE=active_event_file
    v27_sources={"SEC_V27_DEV","SEC_V27_SEAL","NAVER_NEWS_V27_DEV","NAVER_NEWS_V27_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v26.csv.gz",dtype={"ticker":str})
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    # build_labeled persists to the active app's global path.  Isolate that
    # side effect so constructing V27 can never overwrite the opened V26 panel.
    active_labeled_file=app.LABELED_FILE
    app.LABELED_FILE=DATA/"labeled_events_v27_new.csv.gz"
    try:
        new_labeled=app.build_labeled(
            events[events.source.isin(v27_sources)].copy(),cfg
        )
    finally:
        app.LABELED_FILE=active_labeled_file
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v27.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new_labeled.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V27_LABEL_STATUS.json")
    print(f"[V27 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}")
    return labeled




def prepare_v28_us_universe(max_tickers:int=1500)->pd.DataFrame:
    """Build an archive-confirmed SEC life-science universe without outcomes."""
    archive=HF_RAW_CACHE/"ohlcv_2025-05.parquet"
    if not archive.exists():
        raise FileNotFoundError(archive)
    symbols:set[str]=set()
    parquet=pq.ParquetFile(archive)
    for batch in parquet.iter_batches(batch_size=500_000,columns=["ticker"]):
        symbols.update(map(str,batch.column("ticker").to_pylist()))
    collector=app.SECCollector(app.Config())
    ticker_map=collector.get_json(collector.TICKER_MAP)
    pattern=re.compile(
        r"(?:BIO|PHARMA|THERAPEUT|MEDIC|HEALTH|LIFE SCI|ONCO|GENET|GENOMIC|"
        r"VACCIN|DIAGNOST|NEURO|IMMUN|OPHTHAL|CARDIO|SURGICAL|LABORATOR)",
        re.IGNORECASE,
    )
    qiu_tickers=set()
    qiu_path=DATA/"events_us_news_real.csv"
    if qiu_path.exists():
        qiu_tickers=set(pd.read_csv(qiu_path,dtype={"ticker":str},
                                    usecols=["ticker"]).ticker.astype(str))
    rows=[]
    for item in ticker_map.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        if (ticker in symbols and company and (pattern.search(company) or ticker in qiu_tickers) and
                re.fullmatch(r"[A-Z]{1,5}",ticker)):
            rows.append({"market":"US","ticker":ticker,"company":company})
    universe=pd.DataFrame(rows).drop_duplicates("ticker")
    universe["selection_hash"]=universe.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V28|UNIVERSE|{value}".encode()).hexdigest()
    )
    universe=(universe.sort_values("selection_hash").head(max_tickers)
              .drop(columns="selection_hash").reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    _atomic_csv(universe,DATA/"universe_v28_us.csv")
    prior=pd.read_csv(DATA/"events_exact_v27.csv.gz",dtype={"ticker":str},
                      usecols=["market","ticker"])
    prior_tickers=set(prior.loc[prior.market.eq("US"),"ticker"].astype(str))
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"SEC life-science company-name regex plus pinned QIU pharma tickers intersect 2025-05 archive",
        "archive_symbols":len(symbols),"max_tickers":max_tickers,
        "selected_tickers":len(universe),
        "selected_tickers_seen_through_v27":int(universe.ticker.astype(str).isin(prior_tickers).sum()),
        "selected_tickers_new_after_v27":int((~universe.ticker.astype(str).isin(prior_tickers)).sum()),
    },DATA/"V28_US_UNIVERSE_STATUS.json")
    print(f"[V28 US UNIVERSE] archive symbols={len(symbols)}, selected={len(universe)}, "
          f"new={int((~universe.ticker.astype(str).isin(prior_tickers)).sum())}")
    return universe


def collect_v28_sec(force:bool=False)->pd.DataFrame:
    universe=DATA/"universe_v28_us.csv"
    if not universe.exists():prepare_v28_us_universe()
    return collect_sec(
        "2006-01-01","2025-05-30",force=force,
        output_name="events_sec_v28_expanded_raw.csv",source_name="SEC_V28_EXPANDED_RAW",
        one_per_ticker_day=False,max_events_per_ticker=50,
        exclude_event_path=DATA/"events_exact_v27.csv.gz",exclude_minutes=35,
        universe_path=universe,material_only=False,fetch_body=False,
    )


def collect_v28_additional_sec(force:bool=False)->pd.DataFrame:
    """Collect exact-time SEC form families never admitted through V27."""
    universe=DATA/"universe_v28_us.csv"
    if not universe.exists():prepare_v28_us_universe()
    additional_forms=(
        "3","3/A","4","4/A","5","5/A",
        "SC 13G","SC 13G/A","S-8","S-8 POS",
        "424B2","424B7","FWP","EFFECT","RW",
        "8-A12B","8-A12G","POS AM","ARS",
    )
    return collect_sec(
        "2019-01-01","2025-05-30",force=force,
        output_name="events_sec_v28_additional_raw.csv",
        source_name="SEC_V28_ADDITIONAL_RAW",
        one_per_ticker_day=False,max_events_per_ticker=50,
        exclude_event_path=DATA/"events_exact_v27.csv.gz",exclude_minutes=35,
        universe_path=universe,material_only=False,fetch_body=False,
        sec_forms=additional_forms,
    )


def prepare_v28_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze residual label-blind IDs isolated from every event through V27."""
    def article_key(value:Any)->str:
        if pd.isna(value):return ""
        text=str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():text=text[:-2]
        return str(int(text)) if text.isdigit() else text

    prior=pd.read_csv(
        DATA/"events_exact_v27.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    embargo=int(pd.Timedelta(minutes=35).value)
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy()
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    sec_paths=sorted(DATA.glob("events_sec_*raw.csv"))
    if (DATA/"events_sec_real.csv").exists():
        sec_paths.append(DATA/"events_sec_real.csv")
    if (DATA/"events_us_news_real.csv").exists():
        sec_paths.append(DATA/"events_us_news_real.csv")
    sec_raw=pd.concat([
        pd.read_csv(path,dtype={"ticker":str}) for path in sec_paths
    ],ignore_index=True,sort=False).drop_duplicates("event_id")
    sec=outside(sec_raw,"US")
    local=sec.event_time_utc.dt.tz_convert(app.US_TZ)
    sec=space(sec[(local.dt.time>=pd.Timestamp("09:30").time())&
                  (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    sec=sec[sec.event_time_utc>=pd.Timestamp("2019-01-01",tz="UTC")].copy()
    price_status_path=DATA/"V28_US_PRICE_STATUS.json"
    preprice_definition="all residual tickers; max 10 per ticker"
    preprice_cap=10
    if (DATA/"universe_v28_us.csv").exists():
        archive_universe=set(pd.read_csv(
            DATA/"universe_v28_us.csv",dtype={"ticker":str}
        ).ticker.astype(str))
        sec=sec[sec.ticker.astype(str).isin(archive_universe)].copy()
        preprice_cap=50
        preprice_definition=(
            f"{len(archive_universe)} pinned-archive life-science tickers plus QIU; max 50 per ticker"
        )
    elif price_status_path.exists() and (DATA/"events_sec_v28_price_candidates.csv").exists():
        price_status=json.loads(price_status_path.read_text(encoding="utf-8"))
        previous=pd.read_csv(DATA/"events_sec_v28_price_candidates.csv",dtype={"ticker":str})
        missing_ids=set(map(str,price_status.get("missing_event_ids",[])))
        available_tickers=set(previous.loc[
            ~previous.event_id.astype(str).isin(missing_ids),"ticker"
        ].astype(str))
        if available_tickers:
            sec=sec[sec.ticker.astype(str).isin(available_tickers)].copy()
            preprice_cap=100
            preprice_definition=(
                f"{len(available_tickers)} archive-confirmed tickers; max 100 per ticker"
            )
    sec["selection_hash"]=sec.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V28|US|CAP|{value}".encode()).hexdigest()
    )
    sec=(sec.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
         .head(preprice_cap).drop(columns="selection_hash"))
    preprice_count=len(sec)
    preprice_tickers=sec.ticker.astype(str).nunique()
    candidate_output=sec.copy()
    candidate_output["event_time_utc"]=pd.to_datetime(
        candidate_output.event_time_utc,utc=True
    ).astype(str)
    _atomic_csv(candidate_output.sort_values("event_time_utc"),
                DATA/"events_sec_v28_price_candidates.csv")
    bounds={}
    for ticker in sec.ticker.astype(str).unique():
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
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in sec.itertuples(index=False)
    ],bool)
    sec=sec.iloc[np.flatnonzero(covered)].copy()
    us_bucket=sec.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V28|US|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=sec[us_bucket.lt(2)].copy();us_seal=sec[us_bucket.ge(2)].copy()

    naver_raw=pd.concat([
        pd.read_csv(DATA/"events_naver_news_real.csv",dtype={"ticker":str,"article_id":str}),
        pd.read_csv(DATA/"events_naver_v21_new_tickers_raw.csv",
                    dtype={"ticker":str,"article_id":str}),
    ],ignore_index=True,sort=False).drop_duplicates("event_id")
    naver=outside(naver_raw,"KR")
    prior_articles=set(prior.loc[
        prior.market.eq("KR")&prior.article_id.notna(),"article_id"
    ].map(article_key))
    keys=naver.article_id.map(article_key)
    naver=naver[~keys.isin(prior_articles)].copy()
    kind_paths=[
        DATA/"events_kind_real.csv",DATA/"events_kind_v12_2022.csv",
        DATA/"events_kind_v15_expanded_raw.csv",DATA/"events_kind_v17_raw.csv",
    ]
    kind=pd.concat([
        pd.read_csv(path,dtype={"ticker":str,"article_id":str})
        for path in kind_paths if path.exists()
    ],ignore_index=True,sort=False).drop_duplicates("event_id")
    if "article_id" not in kind:
        kind["article_id"]=""
    kind=outside(kind,"KR")
    kr=pd.concat([naver,kind],ignore_index=True,sort=False).drop_duplicates("event_id")
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=space(kr[(local.dt.time>=pd.Timestamp("09:00").time())&
                (local.dt.time<=pd.Timestamp("14:57").time())].copy())
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V28|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(100).drop(columns="selection_hash"))
    article=kr.article_id.map(article_key)
    article=article.where(article.ne(""),kr.event_id.astype(str))
    kr_bucket=article.map(
        lambda value:int(hashlib.sha256(f"V28|ARTICLE|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kr[kr_bucket.lt(2)].copy();kr_seal=kr[kr_bucket.ge(2)].copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V28_DEV",DATA/"events_us_exact_v28_dev.csv"),
        (us_seal,"US_EXACT_V28_SEAL",DATA/"events_us_exact_v28_seal.csv"),
        (kr_dev,"KR_EXACT_V28_DEV",DATA/"events_kr_v28_dev.csv"),
        (kr_seal,"KR_EXACT_V28_SEAL",DATA/"events_kr_v28_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    event_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    kr_dev_key=kr_dev.article_id.map(article_key).where(
        kr_dev.article_id.map(article_key).ne(""),kr_dev.event_id.astype(str)
    )
    kr_seal_key=kr_seal.article_id.map(article_key).where(
        kr_seal.article_id.map(article_key).ne(""),kr_seal.event_id.astype(str)
    )
    article_overlap=len(set(kr_dev_key)&set(kr_seal_key))
    prior_overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                      for frame in (us_dev,us_seal,kr_dev,kr_seal))
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V27","embargo_minutes":35,
        "v27_preseal_disposition":"disqualified after diagnostic parsed sealed columns",
        "split":"US SHA256 mod10 DEV<2; KR SHA256 mod10 DEV<2 article-grouped",
        "us_raw_definition":"local SEC raw plus Reuters-QIU residuals after V27 exact/35-minute exclusion",
        "us_candidate_period":"2019-01-01 through 2025-05-30",
        "us_raw_files":[path.name for path in sec_paths],
        "us_preprice_candidates":preprice_count,
        "us_preprice_tickers":preprice_tickers,
        "us_preprice_definition":preprice_definition,
        "us_max_preprice_events_per_ticker":preprice_cap,
        "kr_raw_definition":"residual Naver plus KIND exact disclosures after all V27 article/exact exclusions",
        "kr_max_events_per_ticker":100,
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":prior_overlap,
        "us_dev_seal_event_overlap":event_overlap,
        "kr_dev_seal_article_overlap":article_overlap,
    },DATA/"V28_SOURCE_STATUS.json")
    if prior_overlap or event_overlap or article_overlap:
        raise RuntimeError("V28 source leakage")
    print(f"[V28 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}")
    return us_dev,us_seal,kr_dev,kr_seal


def collect_v28_us_prices(force:bool=False)->None:
    """Extract V28 candidate ticker-days from the pinned local HF month archive."""
    event_path=DATA/"events_sec_v28_price_candidates.csv"
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    month_tickers={
        month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
        for month in months
    }
    filtered_by_month:dict[str,Path]={}
    workers=min(runtime_limits.THREAD_COUNT,len(months))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={
            executor.submit(_filter_hf_month,month,month_tickers[month],force):month
            for month in months
        }
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future]
            filtered_by_month[month]=future.result()
            print(f"[V28 HF] filtered {done}/{len(months)}: {month}",flush=True)

    pieces:dict[str,list[pd.DataFrame]]={}
    available_event_ids:set[str]=set()
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered_by_month[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str)
        frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={
            ticker:set(group.event_day.astype(str))
            for ticker,group in month_events.groupby("ticker")
        }
        union_days=set(month_events.event_day.astype(str))
        present_days={}
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union_days if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":
                    present_days[ticker]=set(
                        group.loc[group.event_day.isin(wanted),"event_day"].astype(str)
                    )
        for row in month_events.itertuples(index=False):
            if str(row.event_day) in present_days.get(str(row.ticker),set()):
                available_event_ids.add(str(row.event_id))
        print(f"[V28 HF] day extraction {done}/{len(months)}: {month}",flush=True)

    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet"
        combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():
            combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=(combined.sort_values("timestamp")
                  .drop_duplicates("timestamp",keep="last"))
        temporary=path.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary,index=False,compression="zstd")
        temporary.replace(path)
    missing=sorted(set(events.event_id.astype(str))-available_event_ids)
    _atomic_json({
        "selection_uses_labels":False,
        "source":"local pinned ggaddam/OHLCV-1m monthly parquet cache",
        "runtime":runtime_limits.status(),
        "workers":workers,"candidate_events":len(events),
        "candidate_tickers":events.ticker.nunique(),"months":months,
        "written_tickers":len(set(pieces)-{"SPY"}),
        "available_candidate_events":len(available_event_ids),
        "missing_candidate_events":len(missing),
        "missing_event_ids":missing,
    },DATA/"V28_US_PRICE_STATUS.json")
    print(f"[V28 HF] available events={len(available_event_ids)}/{len(events)}, "
          f"written tickers={len(set(pieces)-{'SPY'})}",flush=True)


def build_v28_panel()->pd.DataFrame:
    """Label frozen V28 sources and append the now-opened V27 history."""
    cfg=app.Config()
    source_paths=[
        DATA/"events_exact_v27.csv.gz",DATA/"events_us_exact_v28_dev.csv",
        DATA/"events_us_exact_v28_seal.csv",DATA/"events_kr_v28_dev.csv",
        DATA/"events_kr_v28_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str}) for path in source_paths]
    active_event_file=app.EVENT_FILE
    app.EVENT_FILE=DATA/"events_exact_v28.csv.gz"
    try:
        events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:
        app.EVENT_FILE=active_event_file
    v28_sources={"US_EXACT_V28_DEV","US_EXACT_V28_SEAL","KR_EXACT_V28_DEV","KR_EXACT_V28_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v27.csv.gz",dtype={"ticker":str})
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active_labeled_file=app.LABELED_FILE
    app.LABELED_FILE=DATA/"labeled_events_v28_new.csv.gz"
    try:
        new_labeled=app.build_labeled(
            events[events.source.isin(v28_sources)].copy(),cfg
        )
    finally:
        app.LABELED_FILE=active_labeled_file
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v28.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new_labeled.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V28_LABEL_STATUS.json")
    print(f"[V28 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}")
    return labeled




def prepare_v29_kr_health_universe()->pd.DataFrame:
    """Freeze the six label-blind Naver healthcare industry groups for V29."""
    industry_codes=("261","262","281","286","288","316")
    headers={**HTTP_HEADERS,"Referer":"https://m.stock.naver.com/"}
    rows=[]
    for industry in industry_codes:
        for page in range(1,8):
            response=_request_with_retry(
                "GET",f"https://m.stock.naver.com/api/stocks/industry/{industry}",
                params={"page":page,"pageSize":100},headers=headers,timeout=60,
            )
            stocks=response.json().get("stocks",[]) or []
            for item in stocks:
                ticker=str(item.get("itemCode","")).strip()
                # Exclude preferred shares, SPAC-style alphanumeric symbols,
                # and non-equity instruments from the issuer universe.
                if re.fullmatch(r"\d{6}",ticker):
                    rows.append({
                        "market":"KR","ticker":ticker,
                        "company":str(item.get("stockName","")).strip(),
                        "industry_code":industry,
                    })
            if len(stocks)<100:
                break
    universe=(pd.DataFrame(rows).drop_duplicates("ticker")
              .sort_values(["industry_code","ticker"]).reset_index(drop=True))
    if len(universe)<350:
        raise RuntimeError(f"unexpectedly small V29 KR health universe: {len(universe)}")
    _atomic_csv(universe,DATA/"universe_v29_kr_health.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"Naver industry membership; ordinary six-digit KR equities",
        "industry_codes":list(industry_codes),
        "industry_names":{
            "261":"pharmaceuticals","262":"life-science tools and services",
            "281":"healthcare equipment and supplies","286":"biotechnology",
            "288":"healthcare technology","316":"healthcare providers and services",
        },
        "ticker_count":len(universe),
    },DATA/"V29_KR_HEALTH_UNIVERSE_STATUS.json")
    print(f"[V29 KR HEALTH UNIVERSE] tickers={len(universe)}",flush=True)
    return universe


def prepare_v29_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze label-blind residual SEC/KIND events unused through V28."""
    prior=pd.read_csv(
        DATA/"events_exact_v28.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy()
        frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    # V28's pinned candidate manifest and price extraction status provide an
    # outcome-free exact list of residual US events whose ticker-day exists in
    # the local one-minute archive.
    us=pd.read_csv(DATA/"events_sec_v28_price_candidates.csv",dtype={"ticker":str})
    us=outside(us.drop_duplicates("event_id"),"US")
    price_status=json.loads((DATA/"V28_US_PRICE_STATUS.json").read_text(encoding="utf-8"))
    missing_ids=set(map(str,price_status.get("missing_event_ids",[])))
    us=us[~us.event_id.astype(str).isin(missing_ids)].copy()
    short_headline=us.headline.fillna("").astype(str).str.len().lt(3)
    us.loc[short_headline,"headline"]=(
        "SEC FORM "+us.loc[short_headline,"form"].fillna("OTHER").astype(str)
    )
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    us["selection_hash"]=us.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V29|US|CAP|{value}".encode()).hexdigest()
    )
    us=(us.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(20).drop(columns="selection_hash"))
    us_bucket=us.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V29|US|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=us[us_bucket.lt(2)].copy();us_seal=us[us_bucket.ge(2)].copy()

    # The raw KIND day cache contains full-market exact releases. Freeze only
    # issuers in Naver's six healthcare industry groups. The two groups omitted
    # by V15 (healthcare technology/providers) supply newly listed or formerly
    # uncovered issuers without consulting labels or future returns.
    universe_path=DATA/"universe_v29_kr_health.csv"
    if not universe_path.exists():
        prepare_v29_kr_health_universe()
    health_universe=pd.read_csv(universe_path,dtype={"ticker":str})
    bio_tickers=set(health_universe.ticker.astype(str))
    frames=[]
    for path in sorted(KIND_DAY_CACHE.glob("*.csv")):
        try:
            frame=pd.read_csv(path,dtype={"ticker":str,"article_id":str},encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or not {"event_id","ticker","event_time_utc"}.issubset(frame.columns):
            continue
        frame=frame[frame.ticker.astype(str).isin(bio_tickers)]
        if not frame.empty:frames.append(frame)
    if not frames:raise RuntimeError("no V29 KIND residual candidates")
    kr=outside(pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id"),"KR")
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=kr[(local.dt.time>=pd.Timestamp("09:00").time())&
          (local.dt.time<=pd.Timestamp("14:57").time())&
          kr.headline.fillna("").str.len().ge(3)&
          (kr.event_time_utc>=pd.Timestamp("2024-01-01",tz="UTC"))].copy()
    kr=space(kr)
    candidate_output=kr.copy()
    candidate_output["event_time_utc"]=candidate_output.event_time_utc.astype(str)
    _atomic_csv(candidate_output.sort_values("event_time_utc"),
                DATA/"events_kr_v29_price_candidates.csv")
    price_bounds={}
    for ticker in sorted(set(kr.ticker.astype(str))):
        path=PRICE/"KR"/f"{ticker}.parquet"
        if not path.exists():continue
        bars=pd.read_parquet(path,columns=["datetime"])
        times=pd.to_datetime(bars.datetime,errors="coerce").dropna()
        if len(times):price_bounds[ticker]=(times.min(),times.max())
    covered=[]
    for row in kr.itertuples(index=False):
        bounds=price_bounds.get(str(row.ticker))
        local_time=pd.Timestamp(row.event_time_utc).tz_convert(app.KR_TZ).tz_localize(None)
        covered.append(bool(
            bounds is not None and bounds[0]<=local_time and
            bounds[1]>=local_time+pd.Timedelta(minutes=32)
        ))
    kr=kr.iloc[np.flatnonzero(np.asarray(covered,dtype=bool))].copy()
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V29|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(20).drop(columns="selection_hash"))
    kr_bucket=kr.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V29|KR|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kr[kr_bucket.lt(3)].copy();kr_seal=kr[kr_bucket.ge(3)].copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V29_DEV",DATA/"events_us_exact_v29_dev.csv"),
        (us_seal,"US_EXACT_V29_SEAL",DATA/"events_us_exact_v29_seal.csv"),
        (kr_dev,"KR_EXACT_V29_DEV",DATA/"events_kr_exact_v29_dev.csv"),
        (kr_seal,"KR_EXACT_V29_SEAL",DATA/"events_kr_exact_v29_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    prior_overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                      for frame in (us_dev,us_seal,kr_dev,kr_seal))
    us_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    kr_overlap=len(set(kr_dev.event_id.astype(str))&set(kr_seal.event_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V28","embargo_minutes":35,
        "split":"SHA256 mod10: US DEV<2; KR DEV<3; complementary SEAL",
        "us_definition":"unused V28 pinned SEC price candidates with extracted ticker-day; max20/ticker",
        "kr_definition":"unused exact KIND releases for six Naver healthcare industries since 2024; max20/ticker",
        "bio_tickers_fixed_from_v29_industries":len(bio_tickers),
        "kr_price_covered_tickers":len(price_bounds),
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":prior_overlap,
        "us_dev_seal_event_overlap":us_overlap,
        "kr_dev_seal_event_overlap":kr_overlap,
    },DATA/"V29_SOURCE_STATUS.json")
    if prior_overlap or us_overlap or kr_overlap:raise RuntimeError("V29 source leakage")
    print(f"[V29 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}",flush=True)
    return us_dev,us_seal,kr_dev,kr_seal


def build_v29_panel()->pd.DataFrame:
    """Label frozen V29 sources and append the now-opened V28 history."""
    cfg=app.Config()
    source_paths=[
        DATA/"events_exact_v28.csv.gz",DATA/"events_us_exact_v29_dev.csv",
        DATA/"events_us_exact_v29_seal.csv",DATA/"events_kr_exact_v29_dev.csv",
        DATA/"events_kr_exact_v29_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str}) for path in source_paths]
    active_event_file=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v29.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active_event_file
    sources={"US_EXACT_V29_DEV","US_EXACT_V29_SEAL","KR_EXACT_V29_DEV","KR_EXACT_V29_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v28.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active_labeled_file=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v29_new.csv.gz"
    try:new_labeled=app.build_labeled(events[events.source.isin(sources)].copy(),cfg)
    finally:app.LABELED_FILE=active_labeled_file
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v29.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new_labeled.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V29_LABEL_STATUS.json")
    print(f"[V29 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}",flush=True)
    return labeled


def collect_v29_kr_prices(force:bool=True)->None:
    """Refresh the exact recent residual KIND candidate universe."""
    event_path=DATA/"events_kr_v29_price_candidates.csv"
    collect_kr_prices(
        "2024-01-01","2026-08-24",workers=16,force=force,
        event_path=event_path,material_only=False,
    )




def prepare_v30_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze label-blind exact-time SEC/Naver events unused through V29."""
    prior=pd.read_csv(
        DATA/"events_exact_v29.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","source","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    prior_naver_articles=set(
        prior.loc[
            prior.source.fillna("").astype(str).str.contains("NAVER"),"article_id"
        ].dropna().astype(str)
    )
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(
            frame.event_time_utc,utc=True,errors="coerce",format="mixed"
        )
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    us=pd.read_csv(DATA/"events_sec_v28_price_candidates.csv",dtype={"ticker":str})
    us=outside(us.drop_duplicates("event_id"),"US")
    price_status=json.loads((DATA/"V28_US_PRICE_STATUS.json").read_text(encoding="utf-8"))
    missing_ids=set(map(str,price_status.get("missing_event_ids",[])))
    us=us[~us.event_id.astype(str).isin(missing_ids)].copy()
    short_headline=us.headline.fillna("").astype(str).str.len().lt(3)
    us.loc[short_headline,"headline"]=(
        "SEC FORM "+us.loc[short_headline,"form"].fillna("OTHER").astype(str)
    )
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    us["selection_hash"]=us.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V30|US|CAP|{value}".encode()).hexdigest()
    )
    us=(us.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(20).drop(columns="selection_hash"))
    us_bucket=us.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V30|US|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=us[us_bucket.lt(2)].copy();us_seal=us[us_bucket.ge(2)].copy()

    # Reuse the V29 healthcare-industry snapshot, but collect exact Naver news
    # for issuers absent from the pre-V29 (V28) event universe. This issuer set
    # was fixed without labels and its news cache was empty before V30.
    health=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    pre_v29=pd.read_csv(
        DATA/"events_exact_v28.csv.gz",dtype={"ticker":str},usecols=["market","ticker"]
    )
    old_kr_tickers=set(pre_v29.loc[pre_v29.market.eq("KR"),"ticker"].astype(str))
    fresh_tickers=sorted(set(health.ticker.astype(str))-old_kr_tickers)
    rows=[];missing_cache=[]
    for ticker in fresh_tickers:
        path=NAVER_TICKER_CACHE/f"{ticker}.json"
        if not path.exists():
            missing_cache.append(ticker);continue
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if missing_cache:
        raise RuntimeError(f"missing V30 Naver caches: {missing_cache[:10]}")
    kr=outside(pd.DataFrame(rows).drop_duplicates("event_id"),"KR")
    kr=kr[~kr.article_id.astype(str).isin(prior_naver_articles)].copy()
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=kr[(local.dt.time>=pd.Timestamp("09:00").time())&
          (local.dt.time<=pd.Timestamp("14:57").time())&
          kr.headline.fillna("").str.len().ge(3)&
          (kr.event_time_utc>=pd.Timestamp("2024-01-01",tz="UTC"))].copy()
    kr["article_selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V30|KR|ARTICLE|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["article_id","article_selection_hash"])
        .drop_duplicates("article_id").drop(columns="article_selection_hash"))
    kr=space(kr)
    candidate_output=kr.copy()
    candidate_output["event_time_utc"]=candidate_output.event_time_utc.astype(str)
    _atomic_csv(candidate_output.sort_values("event_time_utc"),
                DATA/"events_kr_v30_price_candidates.csv")

    price_bounds={}
    for ticker in sorted(set(kr.ticker.astype(str))):
        path=PRICE/"KR"/f"{ticker}.parquet"
        if not path.exists():continue
        bars=pd.read_parquet(path,columns=["datetime"])
        times=pd.to_datetime(bars.datetime,errors="coerce").dropna()
        if len(times):price_bounds[ticker]=(times.min(),times.max())
    covered=[]
    for row in kr.itertuples(index=False):
        bounds=price_bounds.get(str(row.ticker))
        local_time=pd.Timestamp(row.event_time_utc).tz_convert(app.KR_TZ).tz_localize(None)
        covered.append(bool(
            bounds is not None and bounds[0]<=local_time and
            bounds[1]>=local_time+pd.Timedelta(minutes=32)
        ))
    kr=kr.iloc[np.flatnonzero(np.asarray(covered,dtype=bool))].copy()
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V30|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(20).drop(columns="selection_hash"))
    kr_bucket=kr.article_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V30|KR|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kr[kr_bucket.lt(3)].copy();kr_seal=kr[kr_bucket.ge(3)].copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V30_DEV",DATA/"events_us_exact_v30_dev.csv"),
        (us_seal,"US_EXACT_V30_SEAL",DATA/"events_us_exact_v30_seal.csv"),
        (kr_dev,"KR_EXACT_V30_DEV",DATA/"events_kr_exact_v30_dev.csv"),
        (kr_seal,"KR_EXACT_V30_SEAL",DATA/"events_kr_exact_v30_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    prior_overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                      for frame in (us_dev,us_seal,kr_dev,kr_seal))
    us_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    kr_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    prior_article_overlap=len(
        (set(kr_dev.article_id.astype(str))|set(kr_seal.article_id.astype(str)))&
        prior_naver_articles
    )
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V29","embargo_minutes":35,
        "split":"SHA256 mod10: US DEV<2; KR article DEV<3; complementary SEAL",
        "us_definition":"unused V28 pinned SEC price candidates outside all V29 windows; max20/ticker",
        "kr_definition":"exact Naver news for V29-health issuers absent from V28 events; one ticker/article; max20/ticker",
        "kr_fresh_health_tickers":len(fresh_tickers),
        "kr_price_covered_tickers":len(price_bounds),
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":prior_overlap,
        "prior_naver_article_overlap":prior_article_overlap,
        "us_dev_seal_event_overlap":us_overlap,
        "kr_dev_seal_article_overlap":kr_overlap,
    },DATA/"V30_SOURCE_STATUS.json")
    if prior_overlap or prior_article_overlap or us_overlap or kr_overlap:
        raise RuntimeError("V30 source leakage")
    print(f"[V30 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}",flush=True)
    return us_dev,us_seal,kr_dev,kr_seal


def build_v30_panel()->pd.DataFrame:
    """Label frozen V30 sources and append the now-opened V29 history."""
    cfg=app.Config()
    source_paths=[
        DATA/"events_exact_v29.csv.gz",DATA/"events_us_exact_v30_dev.csv",
        DATA/"events_us_exact_v30_seal.csv",DATA/"events_kr_exact_v30_dev.csv",
        DATA/"events_kr_exact_v30_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in source_paths]
    active_event_file=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v30.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active_event_file
    sources={"US_EXACT_V30_DEV","US_EXACT_V30_SEAL",
             "KR_EXACT_V30_DEV","KR_EXACT_V30_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v29.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active_labeled_file=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v30_new.csv.gz"
    try:new_labeled=app.build_labeled(events[events.source.isin(sources)].copy(),cfg)
    finally:app.LABELED_FILE=active_labeled_file
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v30.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new_labeled.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V30_LABEL_STATUS.json")
    print(f"[V30 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}",flush=True)
    return labeled


def collect_v30_kr_prices(force:bool=True)->None:
    """Refresh the exact V30 Naver-news candidate universe."""
    try:
        collect_kr_prices(
            "2024-01-01","2026-08-24",workers=16,force=force,
            event_path=DATA/"events_kr_v30_price_candidates.csv",material_only=False,
        )
    except RuntimeError:
        # The outcome-free universe intentionally includes suspended, foreign,
        # and preferred-share symbols for which Daum exposes no chart. Preserve
        # that status, but continue when enough ordinary issuers were collected.
        status=json.loads((DATA/"KR_PRICE_STATUS.json").read_text(encoding="utf-8"))
        usable=sum(int(row.get("rows",0))>0 for row in status)
        if usable<20:
            raise
        print(f"[V30 DAUM WARN] usable tickers={usable}; unavailable={len(status)-usable}")




def prepare_v31_us_stable_universe()->pd.DataFrame:
    """Freeze bio tickers present under the same symbol at both archive ends."""
    symbols=[]
    for month in ("2019-01","2025-05"):
        path=HF_RAW_CACHE/f"ohlcv_{month}.parquet"
        if not path.exists():raise FileNotFoundError(path)
        values=set();parquet=pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=500_000,columns=["ticker"]):
            values.update(map(str,batch.column("ticker").to_pylist()))
        symbols.append(values)
    base=pd.read_csv(DATA/"universe_v28_us.csv",dtype={"ticker":str})
    stable=set.intersection(*symbols)
    universe=(base[base.ticker.astype(str).isin(stable)].copy()
              .sort_values("ticker").reset_index(drop=True))
    universe["priority"]=np.arange(1,len(universe)+1)
    if len(universe)<200:raise RuntimeError(f"small V31 stable universe: {len(universe)}")
    _atomic_csv(universe,DATA/"universe_v31_us_stable.csv")
    _atomic_json({
        "selection_uses_labels":False,
        "definition":"V28 life-science universe intersect same ticker in pinned 2019-01 and 2025-05 archives",
        "ticker_count":len(universe),
    },DATA/"V31_US_UNIVERSE_STATUS.json")
    print(f"[V31 US STABLE UNIVERSE] tickers={len(universe)}",flush=True)
    return universe


def collect_v31_dense_sec(force:bool=False)->pd.DataFrame:
    """Collect a deeper, outcome-free filing inventory for stable US symbols."""
    universe=DATA/"universe_v31_us_stable.csv"
    if not universe.exists():prepare_v31_us_stable_universe()
    additional=(
        "3","3/A","4","4/A","5","5/A","SC 13G","SC 13G/A",
        "S-8","S-8 POS","424B2","424B7","FWP","EFFECT","RW",
        "8-A12B","8-A12G","POS AM","ARS",
    )
    forms=tuple(dict.fromkeys(app.Config().sec_forms+additional))
    return collect_sec(
        "2019-01-01","2025-05-30",force=force,
        output_name="events_sec_v31_dense_raw.csv",source_name="SEC_V31_DENSE_RAW",
        one_per_ticker_day=False,max_events_per_ticker=500,
        exclude_event_path=DATA/"events_exact_v30.csv.gz",exclude_minutes=35,
        universe_path=universe,material_only=False,fetch_body=False,sec_forms=forms,
    )


def prepare_v31_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Freeze broad label-blind SEC and residual Naver events after V30."""
    prior=pd.read_csv(
        DATA/"events_exact_v30.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","source","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    prior_naver_articles=set(
        prior.loc[
            prior.source.fillna("").astype(str).str.contains("NAVER"),"article_id"
        ].dropna().astype(str)
    )
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    us_frames=[];us_raw_files=[]
    for path in sorted(DATA.glob("events_sec_*raw.csv")):
        try:frame=pd.read_csv(path,dtype={"ticker":str},low_memory=False)
        except (pd.errors.EmptyDataError,UnicodeDecodeError):continue
        if {"event_id","ticker","event_time_utc"}.issubset(frame.columns):
            us_frames.append(frame);us_raw_files.append(path.name)
    if not us_frames:raise RuntimeError("no V31 SEC raw sources")
    us=outside(pd.concat(us_frames,ignore_index=True,sort=False).drop_duplicates("event_id"),"US")
    stable_path=DATA/"universe_v31_us_stable.csv"
    if not stable_path.exists():prepare_v31_us_stable_universe()
    stable_tickers=set(pd.read_csv(stable_path,dtype={"ticker":str}).ticker.astype(str))
    us=us[us.ticker.astype(str).isin(stable_tickers)&
          (us.event_time_utc>=pd.Timestamp("2019-01-01",tz="UTC"))].copy()
    short_headline=us.headline.fillna("").astype(str).str.len().lt(3)
    us.loc[short_headline,"headline"]=(
        "SEC FORM "+us.loc[short_headline,"form"].fillna("OTHER").astype(str)
    )
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    us["selection_hash"]=us.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V31|US|CAP|{value}".encode()).hexdigest()
    )
    us=(us.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(50).drop(columns="selection_hash"))
    us_candidates=us.copy();us_candidates["event_time_utc"]=us_candidates.event_time_utc.astype(str)
    _atomic_csv(us_candidates.sort_values("event_time_utc"),
                DATA/"events_sec_v31_price_candidates.csv")

    us_price_status_path=DATA/"V31_US_PRICE_STATUS.json"
    missing_us_ids=set()
    if us_price_status_path.exists():
        us_price_status=json.loads(us_price_status_path.read_text(encoding="utf-8"))
        missing_us_ids=set(map(str,us_price_status.get("missing_event_ids",[])))
    us=us[~us.event_id.astype(str).isin(missing_us_ids)].copy()
    bounds={}
    for ticker in sorted(set(us.ticker.astype(str))):
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
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in us.itertuples(index=False)
    ],bool)
    us=us.iloc[np.flatnonzero(covered)].copy()
    us_bucket=us.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V31|US|{value}".encode()).hexdigest(),16)%10
    )
    us_dev=us[us_bucket.lt(4)].copy();us_seal=us[us_bucket.ge(4)].copy()

    health=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    pre_v29=pd.read_csv(
        DATA/"events_exact_v28.csv.gz",dtype={"ticker":str},usecols=["market","ticker"]
    )
    old_kr_tickers=set(pre_v29.loc[pre_v29.market.eq("KR"),"ticker"].astype(str))
    fresh_tickers=sorted(set(health.ticker.astype(str))-old_kr_tickers)
    rows=[]
    for ticker in fresh_tickers:
        path=NAVER_TICKER_CACHE/f"{ticker}.json"
        if not path.exists():raise RuntimeError(f"missing V31 Naver cache: {ticker}")
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    kr=outside(pd.DataFrame(rows).drop_duplicates("event_id"),"KR")
    kr=kr[~kr.article_id.astype(str).isin(prior_naver_articles)].copy()
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=kr[(local.dt.time>=pd.Timestamp("09:00").time())&
          (local.dt.time<=pd.Timestamp("14:57").time())&
          kr.headline.fillna("").str.len().ge(3)].copy()
    kr["article_selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V31|KR|ARTICLE|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["article_id","article_selection_hash"])
        .drop_duplicates("article_id").drop(columns="article_selection_hash"))
    kr=space(kr)
    price_bounds={}
    for ticker in sorted(set(kr.ticker.astype(str))):
        path=PRICE/"KR"/f"{ticker}.parquet"
        if not path.exists():continue
        bars=pd.read_parquet(path,columns=["datetime"])
        times=pd.to_datetime(bars.datetime,errors="coerce").dropna()
        if len(times):price_bounds[ticker]=(times.min(),times.max())
    covered=[]
    for row in kr.itertuples(index=False):
        range_=price_bounds.get(str(row.ticker))
        local_time=pd.Timestamp(row.event_time_utc).tz_convert(app.KR_TZ).tz_localize(None)
        covered.append(bool(range_ is not None and range_[0]<=local_time and
                            range_[1]>=local_time+pd.Timedelta(minutes=32)))
    kr=kr.iloc[np.flatnonzero(np.asarray(covered,dtype=bool))].copy()
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V31|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(20).drop(columns="selection_hash"))
    kr_bucket=kr.article_id.astype(str).map(
        lambda value:int(hashlib.sha256(f"V31|KR|{value}".encode()).hexdigest(),16)%10
    )
    kr_dev=kr[kr_bucket.lt(4)].copy();kr_seal=kr[kr_bucket.ge(4)].copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V31_DEV",DATA/"events_us_exact_v31_dev.csv"),
        (us_seal,"US_EXACT_V31_SEAL",DATA/"events_us_exact_v31_seal.csv"),
        (kr_dev,"KR_EXACT_V31_DEV",DATA/"events_kr_exact_v31_dev.csv"),
        (kr_seal,"KR_EXACT_V31_SEAL",DATA/"events_kr_exact_v31_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    prior_overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                      for frame in (us_dev,us_seal,kr_dev,kr_seal))
    us_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    kr_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    prior_article_overlap=len(
        (set(kr_dev.article_id.astype(str))|set(kr_seal.article_id.astype(str)))&
        prior_naver_articles
    )
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V30","embargo_minutes":35,
        "split":"SHA256 mod10: US DEV<4; KR article DEV<4; complementary SEAL",
        "us_definition":"2019-2025 SEC residuals outside V30 for tickers present in both pinned 2019-01 and 2025-05 HF archives; max50/ticker; pinned HF exact-day prices",
        "us_raw_files":us_raw_files,"us_preprice_candidates":len(us_candidates),
        "us_price_covered_tickers":len(bounds),
        "kr_definition":"remaining exact Naver news for fresh V29-health issuers after V30; max20/ticker",
        "kr_price_covered_tickers":len(price_bounds),
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":prior_overlap,
        "prior_naver_article_overlap":prior_article_overlap,
        "us_dev_seal_event_overlap":us_overlap,
        "kr_dev_seal_article_overlap":kr_overlap,
    },DATA/"V31_SOURCE_STATUS.json")
    if prior_overlap or prior_article_overlap or us_overlap or kr_overlap:
        raise RuntimeError("V31 source leakage")
    print(f"[V31 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}",flush=True)
    return us_dev,us_seal,kr_dev,kr_seal


def collect_v31_us_prices(force:bool=False)->None:
    """Extract only V31 candidate ticker-days from the pinned local HF archive."""
    event_path=DATA/"events_sec_v31_price_candidates.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    month_tickers={
        month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
        for month in months
    }
    filtered_by_month={};workers=min(runtime_limits.THREAD_COUNT,len(months))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(_filter_hf_month,month,month_tickers[month],force):month
                 for month in months}
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered_by_month[month]=future.result()
            if done%10==0 or done==len(months):
                print(f"[V31 HF] filtered {done}/{len(months)}",flush=True)
    pieces={};available_event_ids=set()
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered_by_month[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str)
        frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={ticker:set(group.event_day.astype(str))
                  for ticker,group in month_events.groupby("ticker")}
        union_days=set(month_events.event_day.astype(str));present_days={}
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union_days if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":
                    present_days[ticker]=set(group.loc[
                        group.event_day.isin(wanted),"event_day"
                    ].astype(str))
        for row in month_events.itertuples(index=False):
            if str(row.event_day) in present_days.get(str(row.ticker),set()):
                available_event_ids.add(str(row.event_id))
        if done%20==0 or done==len(months):
            print(f"[V31 HF] exact-day extraction {done}/{len(months)}",flush=True)
    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet"
        combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=combined.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
        temporary=path.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
    missing=sorted(set(events.event_id.astype(str))-available_event_ids)
    _atomic_json({
        "selection_uses_labels":False,
        "source":"local pinned ggaddam/OHLCV-1m monthly parquet cache",
        "runtime":runtime_limits.status(),"workers":workers,
        "candidate_events":len(events),"candidate_tickers":events.ticker.nunique(),
        "months":months,"written_tickers":len(set(pieces)-{"SPY"}),
        "available_candidate_events":len(available_event_ids),
        "missing_candidate_events":len(missing),"missing_event_ids":missing,
    },DATA/"V31_US_PRICE_STATUS.json")
    print(f"[V31 HF] available events={len(available_event_ids)}/{len(events)}",flush=True)


def build_v31_panel()->pd.DataFrame:
    """Label frozen V31 sources and append the now-opened V30 history."""
    cfg=app.Config();source_paths=[
        DATA/"events_exact_v30.csv.gz",DATA/"events_us_exact_v31_dev.csv",
        DATA/"events_us_exact_v31_seal.csv",DATA/"events_kr_exact_v31_dev.csv",
        DATA/"events_kr_exact_v31_seal.csv",
    ]
    missing=[path for path in source_paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in source_paths]
    active_event_file=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v31.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active_event_file
    sources={"US_EXACT_V31_DEV","US_EXACT_V31_SEAL",
             "KR_EXACT_V31_DEV","KR_EXACT_V31_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v30.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active_labeled_file=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v31_new.csv.gz"
    try:new_labeled=app.build_labeled(events[events.source.isin(sources)].copy(),cfg)
    finally:app.LABELED_FILE=active_labeled_file
    new_labeled["event_time_utc"]=pd.to_datetime(new_labeled.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new_labeled],ignore_index=True,sort=False)
             .drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v31.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new_labeled.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V31_LABEL_STATUS.json")
    print(f"[V31 LABELED] new={len(new_labeled)} counts={counts} total={len(labeled)}",flush=True)
    return labeled




def collect_v32_dense_sec(force:bool=False)->pd.DataFrame:
    """Collect unused in-session SEC metadata across the full V28 bio universe."""
    additional=(
        "3","3/A","4","4/A","5","5/A","SC 13G","SC 13G/A",
        "S-8","S-8 POS","424B2","424B7","FWP","EFFECT","RW",
        "8-A12B","8-A12G","POS AM","ARS",
    )
    forms=tuple(dict.fromkeys(app.Config().sec_forms+additional))
    return collect_sec(
        "2019-01-01","2025-05-30",force=force,
        output_name="events_sec_v32_dense_raw.csv",source_name="SEC_V32_DENSE_RAW",
        one_per_ticker_day=False,max_events_per_ticker=500,
        exclude_event_path=DATA/"events_exact_v31.csv.gz",exclude_minutes=35,
        universe_path=DATA/"universe_v28_us.csv",material_only=False,
        fetch_body=False,sec_forms=forms,
    )


def prepare_v32_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Create label-blind V32 residual sources disjoint from all events through V31."""
    def article_key(value)->str:
        if pd.isna(value):return ""
        text=str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():text=text[:-2]
        return str(int(text)) if text.isdigit() else text

    prior=pd.read_csv(
        DATA/"events_exact_v31.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","source","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    prior_articles=set(prior.loc[
        prior.source.fillna("").astype(str).str.contains("NAVER"),"article_id"
    ].map(article_key))-{""}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(
            frame.event_time_utc,utc=True,errors="coerce",format="mixed"
        )
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    raw_path=DATA/"events_sec_v32_dense_raw.csv"
    if not raw_path.exists():raise FileNotFoundError(raw_path)
    us_frames=[pd.read_csv(raw_path,dtype={"ticker":str},low_memory=False)]
    # Older collectors were built against ticker mappings contemporaneous with
    # their historical windows, so their pre-2019 symbols match the pinned HF
    # archive more reliably than applying today's SEC ticker map backwards.
    historical_files=[]
    for path in sorted(DATA.glob("events_sec_*raw.csv")):
        if path==raw_path:continue
        try:frame=pd.read_csv(path,dtype={"ticker":str},low_memory=False)
        except (pd.errors.EmptyDataError,UnicodeDecodeError):continue
        if not {"event_id","ticker","event_time_utc"}.issubset(frame.columns):continue
        timestamps=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame.loc[timestamps.lt(pd.Timestamp("2019-01-01",tz="UTC"))].copy()
        if not frame.empty:
            us_frames.append(frame);historical_files.append(path.name)
    us=outside(pd.concat(us_frames,ignore_index=True,sort=False).drop_duplicates("event_id"),"US")
    us=us[us.event_time_utc>=pd.Timestamp("2006-01-01",tz="UTC")].copy()
    short=us.headline.fillna("").astype(str).str.len().lt(3)
    us.loc[short,"headline"]="SEC FORM "+us.loc[short,"form"].fillna("OTHER").astype(str)
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())].copy())
    us["selection_hash"]=us.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V32|US|CAP|{value}".encode()).hexdigest()
    )
    us=(us.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(60).drop(columns="selection_hash"))
    candidates=us.copy();candidates["event_time_utc"]=candidates.event_time_utc.astype(str)
    _atomic_csv(candidates.sort_values("event_time_utc"),
                DATA/"events_sec_v32_price_candidates.csv")

    missing_ids=set()
    status_path=DATA/"V32_US_PRICE_STATUS.json"
    if status_path.exists():
        missing_ids=set(map(str,json.loads(status_path.read_text(encoding="utf-8"))
                            .get("missing_event_ids",[])))
    us=us[~us.event_id.astype(str).isin(missing_ids)].copy()
    bounds={}
    for ticker in sorted(set(us.ticker.astype(str))):
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
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in us.itertuples(index=False)
    ],dtype=bool)
    us=us.iloc[np.flatnonzero(covered)].copy()
    # V32 is a pure external seal.  All development/selection uses the now
    # opened V31 partitions; no V32 outcome is available before certification.
    us_dev=us.iloc[0:0].copy();us_seal=us.copy()

    health=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    pre_v29=pd.read_csv(DATA/"events_exact_v28.csv.gz",dtype={"ticker":str},
                       usecols=["market","ticker"])
    old_kr=set(pre_v29.loc[pre_v29.market.eq("KR"),"ticker"].astype(str))
    rows=[]
    for ticker in sorted(set(health.ticker.astype(str))-old_kr):
        path=NAVER_TICKER_CACHE/f"{ticker}.json"
        if path.exists():rows.extend(json.loads(path.read_text(encoding="utf-8")))
    fresh=pd.DataFrame(rows).drop_duplicates("event_id")
    fresh_ids=set(fresh.event_id.astype(str))

    # The broad KIND/Naver cache contains exact-time residual events for the
    # already-defined bio universe.  Selection below is entirely label blind:
    # event/article embargo, timestamp quality, session time, deterministic
    # hash caps, and the presence of entry/exit minute bars are the only inputs.
    def read_kr_pool_file(path:Path)->pd.DataFrame:
        try:
            frame=pd.read_csv(path,dtype={"ticker":str,"article_id":str},low_memory=False)
        except (pd.errors.EmptyDataError,UnicodeDecodeError):
            return pd.DataFrame()
        required={"event_id","ticker","event_time_utc"}
        if not required.issubset(frame.columns):return pd.DataFrame()
        frame=frame.copy();frame["origin_file"]=path.name
        if "article_id" not in frame:frame["article_id"]=""
        if "source" not in frame:frame["source"]="KIND"
        if "headline" not in frame:frame["headline"]=""
        if "body" not in frame:frame["body"]=""
        if "timestamp_quality" in frame:
            frame=frame[frame.timestamp_quality.fillna("").astype(str).eq("EXACT")].copy()
        return frame

    pool_paths=sorted({
        *[path for path in DATA.glob("events_*.csv")
          if "naver" in path.name.lower() or "kind" in path.name.lower()],
        *(ROOT/"cache"/"kind_days").glob("*.csv"),
    })
    broad_frames=[];pool_workers=min(runtime_limits.THREAD_COUNT,max(1,len(pool_paths)))
    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        futures={executor.submit(read_kr_pool_file,path):path for path in pool_paths}
        for done,future in enumerate(as_completed(futures),1):
            frame=future.result()
            if not frame.empty:broad_frames.append(frame)
            if done%250==0 or done==len(pool_paths):
                print(f"[V32 KR POOL] {done}/{len(pool_paths)} files",flush=True)
    naver_cache_paths=sorted(NAVER_TICKER_CACHE.glob("*.json"))
    def read_naver_cache_file(path:Path)->pd.DataFrame:
        try:records=json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError,UnicodeDecodeError):return pd.DataFrame()
        frame=pd.DataFrame(records)
        if frame.empty or not {"event_id","ticker","event_time_utc"}.issubset(frame.columns):
            return pd.DataFrame()
        frame=frame.copy();frame["origin_file"]="naver_cache/"+path.name
        if "article_id" not in frame:frame["article_id"]=""
        return frame
    with ThreadPoolExecutor(max_workers=min(runtime_limits.THREAD_COUNT,
                                             max(1,len(naver_cache_paths)))) as executor:
        for frame in executor.map(read_naver_cache_file,naver_cache_paths):
            if not frame.empty:broad_frames.append(frame)
    prior_kr_tickers=set(prior.loc[prior.market.eq("KR"),"ticker"].astype(str))
    broad=(pd.concat(broad_frames,ignore_index=True,sort=False)
           if broad_frames else fresh.iloc[0:0].copy())
    if not broad.empty:
        broad= broad[broad.ticker.astype(str).isin(prior_kr_tickers)].copy()
    kr=outside(pd.concat([fresh,broad],ignore_index=True,sort=False)
               .drop_duplicates("event_id"),"KR")
    kr["article_key"]=kr.article_id.map(article_key)
    kr=kr[~kr.article_key.isin(prior_articles)].copy()
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=kr[(local.dt.time>=pd.Timestamp("09:00").time())&
          (local.dt.time<=pd.Timestamp("14:57").time())&
          kr.headline.fillna("").str.len().ge(3)].copy()
    kr["article_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V32|KR|ARTICLE|{value}".encode()).hexdigest()
    )
    with_article=(kr[kr.article_key.ne("")].sort_values(["article_key","article_hash"])
                  .drop_duplicates("article_key"))
    kr=pd.concat([kr[kr.article_key.eq("")],with_article],ignore_index=True,sort=False)
    kr=kr.drop(columns=["article_hash","article_key"])
    kr=space(kr)
    kr_price_candidates=kr.copy()
    kr_price_candidates["event_time_utc"]=kr_price_candidates.event_time_utc.astype(str)
    _atomic_csv(kr_price_candidates.sort_values("event_time_utc"),
                DATA/"events_kr_v32_price_candidates.csv")
    price_times={}
    def read_kr_price_times(ticker:str):
        path=PRICE/"KR"/f"{ticker}.parquet"
        if not path.exists():return ticker,np.asarray([],dtype=np.int64)
        bars=pd.read_parquet(path,columns=["datetime"])
        times_=pd.to_datetime(bars.datetime,errors="coerce").dropna()
        if getattr(times_.dt,"tz",None) is not None:
            times_=times_.dt.tz_convert(app.KR_TZ).dt.tz_localize(None)
        return ticker,np.sort(times_.astype("int64").unique())

    price_tickers=sorted(set(kr.ticker.astype(str)))
    with ThreadPoolExecutor(max_workers=min(runtime_limits.THREAD_COUNT,max(1,len(price_tickers)))) as executor:
        for ticker,times_ in executor.map(read_kr_price_times,price_tickers):
            if len(times_):price_times[ticker]=times_
    cfg=app.Config()
    covered=[]
    for row in kr.itertuples(index=False):
        times_=price_times.get(str(row.ticker))
        local_time=pd.Timestamp(row.event_time_utc).tz_convert(app.KR_TZ).tz_localize(None)
        entry=local_time+pd.Timedelta(minutes=cfg.entry_lag_min)
        exit_=entry+pd.Timedelta(minutes=cfg.horizon_min)
        if times_ is None:
            covered.append(False);continue
        ei=int(np.searchsorted(times_,entry.value,side="left"))
        xi=int(np.searchsorted(times_,exit_.value,side="left"))
        valid=ei<len(times_) and xi<len(times_) and xi>ei and ei>=10
        if valid:
            valid=(pd.Timestamp(times_[ei]).date()==local_time.date() and
                   pd.Timestamp(times_[xi]).date()==local_time.date())
        covered.append(bool(valid))
    kr=kr.iloc[np.flatnonzero(np.asarray(covered,dtype=bool))].copy()
    fresh_kr=kr[kr.event_id.astype(str).isin(fresh_ids)].copy()
    broad_kr=kr[~kr.event_id.astype(str).isin(fresh_ids)].copy()
    broad_price_covered=len(broad_kr)
    broad_kr["selection_hash"]=broad_kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V32|KR|BROAD|{value}".encode()).hexdigest()
    )
    broad_kr=(broad_kr.sort_values(["ticker","selection_hash"])
              .groupby("ticker",group_keys=False).head(70).drop(columns="selection_hash"))
    kr=pd.concat([fresh_kr,broad_kr],ignore_index=True,sort=False)
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V32|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(70).drop(columns="selection_hash"))
    kr_dev=kr.iloc[0:0].copy();kr_seal=kr.copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V32_DEV",DATA/"events_us_exact_v32_dev.csv"),
        (us_seal,"US_EXACT_V32_SEAL",DATA/"events_us_exact_v32_seal.csv"),
        (kr_dev,"KR_EXACT_V32_DEV",DATA/"events_kr_exact_v32_dev.csv"),
        (kr_seal,"KR_EXACT_V32_SEAL",DATA/"events_kr_exact_v32_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                for frame in (us_dev,us_seal,kr_dev,kr_seal))
    article_overlap=len((set(kr_dev.article_id.astype(str))|
                         set(kr_seal.article_id.astype(str)))&prior_articles)
    us_overlap=len(set(us_dev.event_id.astype(str))&set(us_seal.event_id.astype(str)))
    kr_overlap=len(set(kr_dev.article_id.astype(str))&set(kr_seal.article_id.astype(str)))
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V31","embargo_minutes":35,
        "split":"pure external SEAL; all development and selection use opened V31 only",
        "us_definition":"unused in-session SEC forms: contemporaneous historical collectors for 2006-2018 plus full V28 bio universe for 2019-2025; max60/ticker; event-month pinned HF coverage",
        "us_historical_raw_files":historical_files,
        "us_preprice_candidates":len(candidates),"us_price_covered_tickers":len(bounds),
        "kr_definition":"remaining exact Naver news for fresh healthcare issuers plus label-blind exact KIND/Naver residuals from prior bio tickers; deterministic combined max70/ticker",
        "kr_pool_files":len(pool_paths),"kr_naver_cache_files":len(naver_cache_paths),
        "kr_pool_workers":pool_workers,
        "kr_preprice_candidates":len(kr_price_candidates),
        "kr_broad_price_covered_before_cap":broad_price_covered,
        "kr_broad_after_price_cap":len(broad_kr),
        "kr_price_covered_tickers":len(price_times),
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":overlap,"prior_naver_article_overlap":article_overlap,
        "us_dev_seal_event_overlap":us_overlap,"kr_dev_seal_article_overlap":kr_overlap,
    },DATA/"V32_SOURCE_STATUS.json")
    if overlap or article_overlap or us_overlap or kr_overlap:
        raise RuntimeError("V32 source leakage")
    print(f"[V32 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}",flush=True)
    return us_dev,us_seal,kr_dev,kr_seal


def collect_v32_kr_prices(force:bool=True)->None:
    """Fill the label-blind exact V32 KR candidate interval from Daum bars."""
    event_path=DATA/"events_kr_v32_price_candidates.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    try:
        collect_kr_prices(
            "2022-01-01","2026-08-25",workers=runtime_limits.THREAD_COUNT,
            force=force,event_path=event_path,material_only=False,
        )
    except RuntimeError:
        status=json.loads((DATA/"KR_PRICE_STATUS.json").read_text(encoding="utf-8"))
        usable=sum(int(row.get("rows",0))>0 for row in status)
        if usable<100:raise
        print(f"[V32 DAUM WARN] usable tickers={usable}; unavailable={len(status)-usable}")


def collect_v32_us_prices(force:bool=False)->None:
    """Extract V32 candidate ticker-days from the pinned local HF archive."""
    event_path=DATA/"events_sec_v32_price_candidates.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    month_tickers={month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
                   for month in months}
    filtered={};workers=min(runtime_limits.THREAD_COUNT,len(months))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(_filter_hf_month,month,month_tickers[month],force):month
                 for month in months}
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered[month]=future.result()
            if done%10==0 or done==len(months):
                print(f"[V32 HF] filtered {done}/{len(months)}",flush=True)
    pieces={};available=set()
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str);frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={ticker:set(group.event_day.astype(str)) for ticker,group in month_events.groupby("ticker")}
        union=set(month_events.event_day.astype(str));present={}
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":present[ticker]=set(group.loc[group.event_day.isin(wanted),"event_day"].astype(str))
        for row in month_events.itertuples(index=False):
            if str(row.event_day) in present.get(str(row.ticker),set()):available.add(str(row.event_id))
        if done%20==0 or done==len(months):print(f"[V32 HF] exact-day {done}/{len(months)}",flush=True)
    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet";combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=combined.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
        temporary=path.with_suffix(".parquet.tmp");combined.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
    missing=sorted(set(events.event_id.astype(str))-available)
    _atomic_json({
        "selection_uses_labels":False,"source":"local pinned ggaddam/OHLCV-1m monthly parquet cache",
        "runtime":runtime_limits.status(),"workers":workers,"candidate_events":len(events),
        "candidate_tickers":events.ticker.nunique(),"months":months,
        "written_tickers":len(set(pieces)-{"SPY"}),"available_candidate_events":len(available),
        "missing_candidate_events":len(missing),"missing_event_ids":missing,
    },DATA/"V32_US_PRICE_STATUS.json")
    print(f"[V32 HF] available events={len(available)}/{len(events)}",flush=True)


def build_v32_panel()->pd.DataFrame:
    """Label frozen V32 sources and append the now-opened V31 history."""
    cfg=app.Config();paths=[
        DATA/"events_exact_v31.csv.gz",DATA/"events_us_exact_v32_dev.csv",
        DATA/"events_us_exact_v32_seal.csv",DATA/"events_kr_exact_v32_dev.csv",
        DATA/"events_kr_exact_v32_seal.csv",
    ]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in paths]
    active=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v32.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active
    sources={"US_EXACT_V32_DEV","US_EXACT_V32_SEAL","KR_EXACT_V32_DEV","KR_EXACT_V32_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v31.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v32_new.csv.gz"
    try:new=app.build_labeled(events[events.source.isin(sources)].copy(),cfg)
    finally:app.LABELED_FILE=active
    new["event_time_utc"]=pd.to_datetime(new.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new],ignore_index=True,sort=False).drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v32.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V32_LABEL_STATUS.json")
    print(f"[V32 LABELED] new={len(new)} counts={counts} total={len(labeled)}",flush=True)
    return labeled




def collect_v33_google_news()->pd.DataFrame:
    """Collect an independent exact-time RSS pool for priced KR health issuers."""
    industry_codes=("261","262","281","286","288","316")
    universe_rows=[]
    for industry in industry_codes:
        for page in range(1,8):
            response=_request_with_retry(
                "GET",f"https://m.stock.naver.com/api/stocks/industry/{industry}",
                params={"page":page,"pageSize":100},
                headers={**HTTP_HEADERS,"Referer":"https://m.stock.naver.com/"},timeout=60,
            )
            stocks=json.loads(response.content.decode("utf-8")).get("stocks",[]) or []
            for item in stocks:
                ticker=str(item.get("itemCode","")).strip()
                if re.fullmatch(r"\d{6}",ticker) and (PRICE/"KR"/f"{ticker}.parquet").exists():
                    universe_rows.append({
                        "ticker":ticker,"company":str(item.get("stockName","")).strip(),
                        "industry_code":industry,
                    })
            if len(stocks)<100:break
    universe=(pd.DataFrame(universe_rows).drop_duplicates("ticker")
              .sort_values(["industry_code","ticker"]).reset_index(drop=True))
    _atomic_csv(universe,DATA/"universe_v33_kr_rss.csv")

    def fetch(row)->tuple[list[dict[str,Any]],dict[str,Any]]:
        ticker=str(row.ticker);company=str(row.company)
        query=f'"{company}" after:2024-03-01 before:2026-08-26'
        try:
            response=_request_with_retry(
                "GET","https://news.google.com/rss/search",
                params={"q":query,"hl":"ko","gl":"KR","ceid":"KR:ko"},
                headers={**HTTP_HEADERS,"Accept":"application/rss+xml"},timeout=45,
            )
            items=ET.fromstring(response.content).findall(".//item");records=[]
            for item in items:
                title=str(item.findtext("title") or "").strip()
                link=str(item.findtext("link") or item.findtext("guid") or "").strip()
                published=pd.to_datetime(item.findtext("pubDate"),utc=True,errors="coerce")
                if not title or not link or pd.isna(published):continue
                article_hash=hashlib.sha256(link.encode("utf-8")).hexdigest()
                publisher=item.find("source")
                records.append({
                    "event_id":f"GNEWS:{ticker}:{article_hash}","market":"KR",
                    "ticker":ticker,"company":company,"event_time_utc":published.isoformat(),
                    "source":"GOOGLE_NEWS_RSS","form":"","headline":title,
                    "body":str(publisher.text if publisher is not None else ""),
                    "event_type":app.infer_event_type(title),"timestamp_quality":"EXACT",
                    "url":link,"article_id":f"GNEWS:{article_hash}",
                })
            return records,{"ticker":ticker,"items":len(items),"status":"ok"}
        except Exception as exc:
            return [],{"ticker":ticker,"items":0,"status":f"ERROR: {exc}"}

    rows=[];statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(universe)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,row):str(row.ticker) for row in universe.itertuples(index=False)}
        for done,future in enumerate(as_completed(futures),1):
            records,status=future.result();rows.extend(records);statuses.append(status)
            if done%25==0 or done==len(futures):
                print(f"[V33 KR RSS] {done}/{len(futures)} tickers rows={len(rows)}",flush=True)
    if not rows:raise RuntimeError("V33 Google News RSS returned no events")
    out=pd.DataFrame(rows)
    out["ticker_hash"]=out.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V33|GNEWS|{value}".encode()).hexdigest()
    )
    out=(out.sort_values(["article_id","ticker_hash"])
         .drop_duplicates("article_id").drop(columns="ticker_hash")
         .sort_values("event_time_utc").reset_index(drop=True))
    _atomic_csv(out,DATA/"events_google_news_v33_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,"source":"Google News RSS search",
        "query_interval":"2024-03-01 through 2026-08-25",
        "universe":"current Naver six healthcare industries with existing local minute prices",
        "industry_codes":list(industry_codes),"universe_tickers":len(universe),
        "workers":workers,"unique_articles":len(out),"requests":statuses,
    },DATA/"V33_KR_RSS_STATUS.json")
    print(f"[V33 KR RSS] unique articles={len(out)}",flush=True)
    return out


def collect_v33_google_us_news()->pd.DataFrame:
    """Collect independent exact-time RSS articles for locally priced US biotech issuers."""
    priced={path.stem.upper() for path in (PRICE/"US").glob("*.parquet")}-{"SPY"}
    raw=app.SECCollector(app.Config()).get_json(app.SECCollector.TICKER_MAP)
    universe_rows=[]
    for item in raw.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        if ticker in priced and company:
            universe_rows.append({"ticker":ticker,"company":company})
    universe=(pd.DataFrame(universe_rows).drop_duplicates("ticker")
              .sort_values("ticker").reset_index(drop=True))
    _atomic_csv(universe,DATA/"universe_v33_us_rss.csv")

    def fetch(row)->tuple[list[dict[str,Any]],dict[str,Any]]:
        ticker=str(row.ticker);company=str(row.company)
        try:
            records=[];item_count=0
            for start,end in (("2022-01-01","2023-01-01"),("2023-01-01","2024-01-01"),
                              ("2024-01-01","2025-06-01")):
                query=f'"{company}" after:{start} before:{end}'
                response=_request_with_retry(
                    "GET","https://news.google.com/rss/search",
                    params={"q":query,"hl":"en-US","gl":"US","ceid":"US:en"},
                    headers={**HTTP_HEADERS,"Accept":"application/rss+xml"},timeout=45,
                )
                items=ET.fromstring(response.content).findall(".//item");item_count+=len(items)
                for item in items:
                    title=str(item.findtext("title") or "").strip()
                    link=str(item.findtext("link") or item.findtext("guid") or "").strip()
                    published=pd.to_datetime(item.findtext("pubDate"),utc=True,errors="coerce")
                    if not title or not link or pd.isna(published):continue
                    article_hash=hashlib.sha256(link.encode("utf-8")).hexdigest()
                    publisher=item.find("source")
                    records.append({
                        "event_id":f"GNEWSUS:{ticker}:{article_hash}","market":"US",
                        "ticker":ticker,"company":company,"event_time_utc":published.isoformat(),
                        "source":"GOOGLE_NEWS_RSS_US","form":"","headline":title,
                        "body":str(publisher.text if publisher is not None else ""),
                        "event_type":app.infer_event_type(title),"timestamp_quality":"EXACT",
                        "url":link,"article_id":f"GNEWSUS:{article_hash}",
                    })
            return records,{"ticker":ticker,"items":item_count,"status":"ok"}
        except Exception as exc:
            return [],{"ticker":ticker,"items":0,"status":f"ERROR: {exc}"}

    rows=[];statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(universe)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,row):str(row.ticker) for row in universe.itertuples(index=False)}
        for done,future in enumerate(as_completed(futures),1):
            records,status=future.result();rows.extend(records);statuses.append(status)
            if done%50==0 or done==len(futures):
                print(f"[V33 US RSS] {done}/{len(futures)} tickers rows={len(rows)}",flush=True)
    if not rows:raise RuntimeError("V33 US Google News RSS returned no events")
    out=pd.DataFrame(rows)
    out["ticker_hash"]=out.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V33|GNEWSUS|{value}".encode()).hexdigest()
    )
    out=(out.sort_values(["article_id","ticker_hash"])
         .drop_duplicates("article_id").drop(columns="ticker_hash")
         .sort_values("event_time_utc").reset_index(drop=True))
    _atomic_csv(out,DATA/"events_google_us_v33_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,"source":"Google News RSS search",
        "query_interval":"three disjoint yearly searches from 2022-01-01 through 2025-05-31",
        "universe":"SEC-listed issuers with existing pinned local US minute prices",
        "universe_tickers":len(universe),"workers":workers,
        "unique_articles":len(out),"requests":statuses,
    },DATA/"V33_US_RSS_STATUS.json")
    print(f"[V33 US RSS] unique articles={len(out)}",flush=True)
    return out


def prepare_v33_sources()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Create the label-blind pure V33 seal, disjoint from every event through V32."""
    def article_key(value)->str:
        if pd.isna(value):return ""
        text=str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():text=text[:-2]
        return str(int(text)) if text.isdigit() else text

    prior=pd.read_csv(
        DATA/"events_exact_v32.csv.gz",dtype={"ticker":str,"article_id":str},
        usecols=lambda column:column in {
            "event_id","market","ticker","source","event_time_utc","article_id",
        },
    )
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    prior_ids=set(prior.event_id.astype(str))
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    prior_articles=set(prior.loc[
        prior.source.fillna("").astype(str).str.contains("NAVER"),"article_id"
    ].map(article_key))-{""}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frame=frame[frame.event_time_utc.notna()&
                    ~frame.event_id.astype(str).isin(prior_ids)].copy()
        keep=np.ones(len(frame),dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value)
            index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    us_frames=[];us_raw_files=[]
    for path in sorted(DATA.glob("events_sec_*raw.csv")):
        try:frame=pd.read_csv(path,dtype={"ticker":str},low_memory=False)
        except (pd.errors.EmptyDataError,UnicodeDecodeError):continue
        if {"event_id","ticker","event_time_utc"}.issubset(frame.columns):
            us_frames.append(frame);us_raw_files.append(path.name)
    qiu_path=DATA/"events_us_news_real.csv"
    if qiu_path.exists():
        us_frames.append(pd.read_csv(qiu_path,dtype={"ticker":str},low_memory=False))
        us_raw_files.append(qiu_path.name)
    rss_us_path=DATA/"events_google_us_v33_raw.csv"
    if rss_us_path.exists():
        us_frames.append(pd.read_csv(rss_us_path,dtype={"ticker":str},low_memory=False))
        us_raw_files.append(rss_us_path.name)
    if not us_frames:raise RuntimeError("no V33 US residual source")
    for frame in us_frames:
        frame["event_time_utc"]=pd.to_datetime(
            frame.event_time_utc,utc=True,errors="coerce",format="mixed"
        )
    us=outside(pd.concat(us_frames,ignore_index=True,sort=False).drop_duplicates("event_id"),"US")
    short=us.headline.fillna("").astype(str).str.len().lt(3)
    us.loc[short,"headline"]="SEC FORM "+us.loc[short,"form"].fillna("OTHER").astype(str)
    local=us.event_time_utc.dt.tz_convert(app.US_TZ)
    us=space(us[(local.dt.time>=pd.Timestamp("09:30").time())&
                (local.dt.time<=pd.Timestamp("15:27").time())&
                us.headline.fillna("").astype(str).str.len().ge(3)].copy())
    us_candidates=us.copy();us_candidates["event_time_utc"]=us_candidates.event_time_utc.astype(str)
    _atomic_csv(us_candidates.sort_values("event_time_utc"),
                DATA/"events_us_v33_price_candidates.csv")
    missing_ids=set()
    us_status_path=DATA/"V33_US_PRICE_STATUS.json"
    if us_status_path.exists():
        missing_ids=set(map(str,json.loads(us_status_path.read_text(encoding="utf-8"))
                            .get("missing_event_ids",[])))
    us=us[~us.event_id.astype(str).isin(missing_ids)].copy()
    bounds={}
    for ticker in sorted(set(us.ticker.astype(str))):
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
            bounds[ticker]=(lower,upper)
    covered=np.asarray([
        str(row.ticker) in bounds and bounds[str(row.ticker)][0]<=row.event_time_utc and
        bounds[str(row.ticker)][1]>=row.event_time_utc+pd.Timedelta(minutes=32)
        for row in us.itertuples(index=False)
    ],dtype=bool)
    us=us.iloc[np.flatnonzero(covered)].copy()
    # Apply the concentration cap only after exact price availability has been
    # established.  Capping first discarded nearly every usable day because
    # the pinned minute archive covers only a sparse subset of the raw history.
    us["selection_hash"]=us.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V33|US|CAP|{value}".encode()).hexdigest()
    )
    us=(us.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(300).drop(columns="selection_hash"))
    us_dev=us.iloc[0:0].copy();us_seal=us.copy()

    kr_paths=[DATA/"events_kr_v32_price_candidates.csv",DATA/"events_kind_real.csv",
              DATA/"events_google_news_v33_raw.csv"]
    missing_kr=[path for path in kr_paths if not path.exists()]
    if missing_kr:raise FileNotFoundError(str(missing_kr))
    kr_frames=[
        pd.read_csv(path,dtype={"ticker":str,"article_id":str},low_memory=False)
        for path in kr_paths
    ]
    for frame in kr_frames:
        frame["event_time_utc"]=pd.to_datetime(
            frame.event_time_utc,utc=True,errors="coerce",format="mixed"
        )
    # V32 exhausted the established issuers, but the frozen healthcare
    # universe still contains issuers with no event in any prior panel.  Their
    # cached exact Naver articles are a genuinely external, label-blind pool.
    health_path=DATA/"universe_v29_kr_health.csv"
    if health_path.exists():
        health=pd.read_csv(health_path,dtype={"ticker":str})
        prior_kr_tickers=set(prior.loc[prior.market.eq("KR"),"ticker"].astype(str))
        for ticker in sorted(set(health.ticker.astype(str))-prior_kr_tickers):
            path=NAVER_TICKER_CACHE/f"{ticker}.json"
            if not path.exists():continue
            try:records=json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError,UnicodeDecodeError):continue
            frame=pd.DataFrame(records)
            if not frame.empty and {"event_id","ticker","event_time_utc"}.issubset(frame.columns):
                kr_frames.append(frame)
    kr=outside(pd.concat(kr_frames,ignore_index=True,sort=False)
               .drop_duplicates("event_id"),"KR")
    if "article_id" not in kr:kr["article_id"]=""
    kr["article_key"]=kr.article_id.map(article_key)
    kr=kr[~kr.article_key.isin(prior_articles)].copy()
    local=kr.event_time_utc.dt.tz_convert(app.KR_TZ)
    kr=kr[(local.dt.time>=pd.Timestamp("09:00").time())&
          (local.dt.time<=pd.Timestamp("14:57").time())&
          kr.headline.fillna("").astype(str).str.len().ge(3)].copy()
    kr["article_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V33|KR|ARTICLE|{value}".encode()).hexdigest()
    )
    with_article=(kr[kr.article_key.ne("")].sort_values(["article_key","article_hash"])
                  .drop_duplicates("article_key"))
    kr=pd.concat([kr[kr.article_key.eq("")],with_article],ignore_index=True,sort=False)
    kr=space(kr.drop(columns=["article_hash","article_key"]))
    kr_candidates=kr.copy();kr_candidates["event_time_utc"]=kr_candidates.event_time_utc.astype(str)
    _atomic_csv(kr_candidates.sort_values("event_time_utc"),
                DATA/"events_kr_v33_price_candidates.csv")
    price_times={}
    def read_kr_times(ticker:str):
        path=PRICE/"KR"/f"{ticker}.parquet"
        if not path.exists():return ticker,np.asarray([],dtype=np.int64)
        times_=pd.to_datetime(pd.read_parquet(path,columns=["datetime"]).datetime,
                              errors="coerce").dropna()
        if getattr(times_.dt,"tz",None) is not None:
            times_=times_.dt.tz_convert(app.KR_TZ).dt.tz_localize(None)
        return ticker,np.sort(times_.astype("int64").unique())
    tickers=sorted(set(kr.ticker.astype(str)))
    with ThreadPoolExecutor(max_workers=min(runtime_limits.THREAD_COUNT,max(1,len(tickers)))) as pool:
        for ticker,times_ in pool.map(read_kr_times,tickers):
            if len(times_):price_times[ticker]=times_
    cfg=app.Config();covered=[]
    for row in kr.itertuples(index=False):
        times_=price_times.get(str(row.ticker))
        local_time=pd.Timestamp(row.event_time_utc).tz_convert(app.KR_TZ).tz_localize(None)
        entry=local_time+pd.Timedelta(minutes=cfg.entry_lag_min)
        exit_=entry+pd.Timedelta(minutes=cfg.horizon_min)
        if times_ is None:covered.append(False);continue
        ei=int(np.searchsorted(times_,entry.value,side="left"))
        xi=int(np.searchsorted(times_,exit_.value,side="left"))
        valid=ei<len(times_) and xi<len(times_) and xi>ei and ei>=10
        if valid:
            valid=(pd.Timestamp(times_[ei]).date()==local_time.date() and
                   pd.Timestamp(times_[xi]).date()==local_time.date())
        covered.append(bool(valid))
    kr=kr.iloc[np.flatnonzero(np.asarray(covered,dtype=bool))].copy()
    kr["selection_hash"]=kr.event_id.astype(str).map(
        lambda value:hashlib.sha256(f"V33|KR|CAP|{value}".encode()).hexdigest()
    )
    kr=(kr.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
        .head(70).drop(columns="selection_hash"))
    kr_dev=kr.iloc[0:0].copy();kr_seal=kr.copy()

    for frame,source,path in (
        (us_dev,"US_EXACT_V33_DEV",DATA/"events_us_exact_v33_dev.csv"),
        (us_seal,"US_EXACT_V33_SEAL",DATA/"events_us_exact_v33_seal.csv"),
        (kr_dev,"KR_EXACT_V33_DEV",DATA/"events_kr_exact_v33_dev.csv"),
        (kr_seal,"KR_EXACT_V33_SEAL",DATA/"events_kr_exact_v33_seal.csv"),
    ):
        frame["source"]=source
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame.sort_values("event_time_utc"),path)
    overlap=sum(len(set(frame.event_id.astype(str))&prior_ids)
                for frame in (us_dev,us_seal,kr_dev,kr_seal))
    article_overlap=len((set(kr_dev.article_id.map(article_key))|
                         set(kr_seal.article_id.map(article_key)))&prior_articles)
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V32","embargo_minutes":35,
        "split":"pure external SEAL; all development and selection use opened V32 or earlier",
        "us_definition":"unused exact Reuters-QIU/SEC residuals plus independent Google News RSS; price-first max300/ticker; pinned HF exact-day prices",
        "us_raw_files":us_raw_files,"us_preprice_candidates":len(us_candidates),
        "us_price_covered_tickers":len(bounds),
        "kr_definition":"unused exact KIND/V32 residuals plus independent Google News RSS for priced healthcare issuers; max70/ticker",
        "kr_preprice_candidates":len(kr_candidates),"kr_price_covered_tickers":len(price_times),
        "us_dev":len(us_dev),"us_seal":len(us_seal),
        "kr_dev":len(kr_dev),"kr_seal":len(kr_seal),
        "prior_event_id_overlap":overlap,"prior_naver_article_overlap":article_overlap,
        "us_dev_seal_event_overlap":0,"kr_dev_seal_article_overlap":0,
    },DATA/"V33_SOURCE_STATUS.json")
    if overlap or article_overlap:raise RuntimeError("V33 source leakage")
    print(f"[V33 SOURCES] US dev={len(us_dev)} seal={len(us_seal)}; "
          f"KR dev={len(kr_dev)} seal={len(kr_seal)}",flush=True)
    return us_dev,us_seal,kr_dev,kr_seal


def collect_v33_us_prices(force:bool=False)->None:
    """Extract V33 candidate ticker-days from the pinned local HF archive."""
    event_path=DATA/"events_us_v33_price_candidates.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    events["event_time_utc"]=pd.to_datetime(events.event_time_utc,utc=True)
    events["month"]=events.event_time_utc.dt.strftime("%Y-%m")
    events["event_day"]=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
    months=sorted(events.month.unique())
    month_tickers={month:set(events.loc[events.month.eq(month),"ticker"].astype(str))|{"SPY"}
                   for month in months}
    filtered={};workers=min(runtime_limits.THREAD_COUNT,len(months))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(_filter_hf_month,month,month_tickers[month],force):month
                 for month in months}
        for done,future in enumerate(as_completed(futures),1):
            month=futures[future];filtered[month]=future.result()
            if done%10==0 or done==len(months):print(f"[V33 HF] filtered {done}/{len(months)}",flush=True)
    pieces={};available=set()
    for done,month in enumerate(months,1):
        frame=pd.read_parquet(filtered[month])
        if frame.empty:continue
        frame["ticker"]=frame.ticker.astype(str);frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame["event_day"]=frame.timestamp.dt.tz_convert(app.US_TZ).dt.strftime("%Y-%m-%d")
        month_events=events[events.month.eq(month)]
        day_sets={ticker:set(group.event_day.astype(str)) for ticker,group in month_events.groupby("ticker")}
        union=set(month_events.event_day.astype(str));present={}
        for ticker,group in frame.groupby("ticker",sort=False):
            wanted=union if ticker=="SPY" else day_sets.get(ticker,set())
            if not wanted:continue
            selected=group[group.event_day.isin(wanted)].drop(columns=["ticker","event_day"])
            if not selected.empty:
                pieces.setdefault(ticker,[]).append(selected)
                if ticker!="SPY":present[ticker]=set(group.loc[group.event_day.isin(wanted),"event_day"])
        for row in month_events.itertuples(index=False):
            if str(row.event_day) in present.get(str(row.ticker),set()):available.add(str(row.event_id))
        if done%20==0 or done==len(months):print(f"[V33 HF] exact-day {done}/{len(months)}",flush=True)
    for ticker,frames in pieces.items():
        path=PRICE/"US"/f"{ticker}.parquet";combined=pd.concat(frames,ignore_index=True,sort=False)
        if path.exists():combined=pd.concat([pd.read_parquet(path),combined],ignore_index=True,sort=False)
        combined["timestamp"]=pd.to_datetime(combined.timestamp,utc=True)
        combined=combined.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
        temporary=path.with_suffix(".parquet.tmp");combined.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
    missing=sorted(set(events.event_id.astype(str))-available)
    _atomic_json({
        "selection_uses_labels":False,"source":"local pinned ggaddam/OHLCV-1m monthly parquet cache",
        "runtime":runtime_limits.status(),"workers":workers,"candidate_events":len(events),
        "candidate_tickers":events.ticker.nunique(),"months":months,
        "written_tickers":len(set(pieces)-{"SPY"}),"available_candidate_events":len(available),
        "missing_candidate_events":len(missing),"missing_event_ids":missing,
    },DATA/"V33_US_PRICE_STATUS.json")
    print(f"[V33 HF] available events={len(available)}/{len(events)}",flush=True)


def _download_daum_v33_deep(ticker:str,start:str,end:str,max_pages:int=600)->tuple[str,int,str]:
    output=PRICE/"KR"/f"{ticker}.parquet";previous=pd.DataFrame()
    start_ts=pd.Timestamp(start);end_cursor=pd.Timestamp(end)+pd.Timedelta(hours=16)
    if output.exists():
        previous=pd.read_parquet(output)
        existing=pd.to_datetime(previous.datetime,errors="coerce").dropna()
        if len(existing):
            if existing.min()<=start_ts:return ticker,len(previous),"cached-depth"
            end_cursor=min(end_cursor,existing.min()-pd.Timedelta(minutes=1))
    records=[];seen=set()
    for _ in range(max_pages):
        response=_request_with_retry(
            "GET",DAUM_URL.format(ticker=ticker),
            params={"limit":"5000","adjusted":"true","to":end_cursor.strftime("%Y-%m-%d %H:%M:%S")},
            headers={**HTTP_HEADERS,"Referer":f"https://finance.daum.net/chart/A{ticker}"},timeout=120,
        )
        data=response.json().get("data",[])
        if not data:break
        records.extend(data);first=str(data[0].get("candleTime",""))
        if not first or first in seen:break
        seen.add(first);first_ts=pd.Timestamp(first)
        if first_ts<=start_ts:break
        end_cursor=first_ts-pd.Timedelta(minutes=1);time.sleep(0.05)
    if records:
        raw=pd.DataFrame(records)
        frame=pd.DataFrame({
            "datetime":pd.to_datetime(raw["candleTime"],errors="coerce"),
            "open":pd.to_numeric(raw["openingPrice"],errors="coerce"),
            "high":pd.to_numeric(raw["highPrice"],errors="coerce"),
            "low":pd.to_numeric(raw["lowPrice"],errors="coerce"),
            "close":pd.to_numeric(raw["tradePrice"],errors="coerce"),
            "volume":pd.to_numeric(raw.get("candleAccTradeVolume",0),errors="coerce"),
        }).dropna(subset=["datetime","close"])
        frame=frame[(frame.datetime>=start_ts)&(frame.datetime<pd.Timestamp(end)+pd.Timedelta(days=1))]
        if not previous.empty:frame=pd.concat([previous,frame],ignore_index=True,sort=False)
        frame=frame.sort_values("datetime").drop_duplicates("datetime",keep="last")
        temporary=output.with_suffix(".parquet.tmp");frame.to_parquet(temporary,index=False,compression="zstd");temporary.replace(output)
        return ticker,len(frame),"ok"
    return ticker,len(previous),"no-older-data"


def collect_v33_kr_prices()->None:
    """Fill exact V33 KR candidate tickers, prioritizing the usable recent pool."""
    event_path=DATA/"events_kr_v33_price_candidates.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    event_times=pd.to_datetime(events.event_time_utc,utc=True,errors="coerce")
    recent=events.loc[event_times.ge(pd.Timestamp("2024-01-01",tz="UTC"))].copy()
    ranked=(recent.groupby("ticker").size().rename("events").reset_index()
            .sort_values(["events","ticker"],ascending=[False,True]).head(120))
    tickers=ranked.ticker.astype(str).tolist();status=[]
    with ThreadPoolExecutor(max_workers=min(runtime_limits.THREAD_COUNT,len(tickers))) as pool:
        futures={pool.submit(_download_daum_v33_deep,ticker,"2022-01-01","2026-08-25"):ticker
                 for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            ticker=futures[future]
            try:ticker,rows,note=future.result();status.append({"ticker":ticker,"rows":rows,"status":note})
            except Exception as exc:status.append({"ticker":ticker,"rows":0,"status":f"ERROR: {exc}"})
            if done%10==0 or done==len(futures):print(f"[V33 DAUM DEEP] {done}/{len(futures)}",flush=True)
    _atomic_json({
        "selection_uses_labels":False,"selection":"top120 recent residual-event-count tickers; ties by ticker",
        "start":"2022-01-01","end":"2026-08-25","max_pages":600,
        "runtime":runtime_limits.status(),"ranked_tickers":ranked.to_dict("records"),"status":status,
    },DATA/"V33_KR_PRICE_STATUS.json")


def build_v33_panel()->pd.DataFrame:
    """Label frozen V33 sources and append the now-opened V32 history."""
    cfg=app.Config();paths=[
        DATA/"events_exact_v32.csv.gz",DATA/"events_us_exact_v33_dev.csv",
        DATA/"events_us_exact_v33_seal.csv",DATA/"events_kr_exact_v33_dev.csv",
        DATA/"events_kr_exact_v33_seal.csv",
    ]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in paths]
    active=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v33.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active
    sources={"US_EXACT_V33_DEV","US_EXACT_V33_SEAL","KR_EXACT_V33_DEV","KR_EXACT_V33_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v32.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    active=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v33_new.csv.gz"
    try:new=app.build_labeled(events[events.source.isin(sources)].copy(),cfg)
    finally:app.LABELED_FILE=active
    new["event_time_utc"]=pd.to_datetime(new.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new],ignore_index=True,sort=False).drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v33.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V33_LABEL_STATUS.json")
    print(f"[V33 LABELED] new={len(new)} counts={counts} total={len(labeled)}",flush=True)
    return labeled




V34_NEWS_START="2026-06-30"
V34_NEWS_END="2026-08-26"


def collect_v34_google_news()->tuple[pd.DataFrame,pd.DataFrame]:
    """Collect a recent exact-time US/KR pool without reading any outcomes."""
    def fetch(row,market:str)->tuple[list[dict[str,Any]],dict[str,Any]]:
        ticker=str(row.ticker);company=str(row.company)
        locale=("en-US","US","US:en") if market=="US" else ("ko","KR","KR:ko")
        query=f'"{company}" after:{V34_NEWS_START} before:{V34_NEWS_END}'
        try:
            response=_request_with_retry(
                "GET","https://news.google.com/rss/search",
                params={"q":query,"hl":locale[0],"gl":locale[1],"ceid":locale[2]},
                headers={**HTTP_HEADERS,"Accept":"application/rss+xml"},timeout=45,
            )
            items=ET.fromstring(response.content).findall(".//item");records=[]
            for item in items:
                title=str(item.findtext("title") or "").strip()
                link=str(item.findtext("link") or item.findtext("guid") or "").strip()
                published=pd.to_datetime(item.findtext("pubDate"),utc=True,errors="coerce")
                if not title or not link or pd.isna(published):continue
                if not (pd.Timestamp(V34_NEWS_START,tz="UTC")<=published<
                        pd.Timestamp(V34_NEWS_END,tz="UTC")):continue
                article_hash=hashlib.sha256(link.encode("utf-8")).hexdigest()
                publisher=item.find("source")
                prefix="GNEWSUS" if market=="US" else "GNEWS"
                records.append({
                    "event_id":f"{prefix}:{ticker}:{article_hash}","market":market,
                    "ticker":ticker,"company":company,"event_time_utc":published.isoformat(),
                    "source":f"GOOGLE_NEWS_RSS_{market}_V34","form":"","headline":title,
                    "body":str(publisher.text if publisher is not None else ""),
                    "event_type":app.infer_event_type(title),"timestamp_quality":"EXACT",
                    "url":link,"article_id":f"{prefix}:{article_hash}",
                })
            return records,{"ticker":ticker,"items":len(items),"kept":len(records),"status":"ok"}
        except Exception as exc:
            return [],{"ticker":ticker,"items":0,"kept":0,"status":f"ERROR: {exc}"}

    universes={
        "US":pd.read_csv(DATA/"universe_v33_us_rss.csv",dtype={"ticker":str}),
        "KR":pd.read_csv(DATA/"universe_v33_kr_rss.csv",dtype={"ticker":str}),
    }
    outputs={}
    for market,universe in universes.items():
        rows=[];statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(universe)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(fetch,row,market):str(row.ticker)
                     for row in universe.itertuples(index=False)}
            for done,future in enumerate(as_completed(futures),1):
                records,status=future.result();rows.extend(records);statuses.append(status)
                stride=50 if market=="US" else 25
                if done%stride==0 or done==len(futures):
                    print(f"[V34 {market} RSS] {done}/{len(futures)} rows={len(rows)}",flush=True)
        if not rows:raise RuntimeError(f"V34 {market} Google News RSS returned no events")
        frame=pd.DataFrame(rows)
        frame["ticker_hash"]=frame.ticker.astype(str).map(
            lambda value:hashlib.sha256(f"V34|{market}|ARTICLE|{value}".encode()).hexdigest()
        )
        frame=(frame.sort_values(["article_id","ticker_hash"]).drop_duplicates("article_id")
               .drop(columns="ticker_hash").sort_values("event_time_utc").reset_index(drop=True))
        path=DATA/f"events_google_{market.lower()}_v34_raw.csv"
        _atomic_csv(frame,path)
        _atomic_json({
            "selection_uses_labels":False,"source":"Google News RSS search",
            "query_interval":f"{V34_NEWS_START} through 2026-08-25",
            "market":market,"universe_tickers":len(universe),"workers":workers,
            "unique_articles":len(frame),"requests":statuses,
        },DATA/f"V34_{market}_RSS_STATUS.json")
        outputs[market]=frame
    return outputs["US"],outputs["KR"]


def collect_v34_us_prices(force:bool=False)->None:
    """Download recent regular-session Yahoo 5-minute bars for V34 candidates."""
    event_path=DATA/"events_google_us_v34_raw.csv"
    if not event_path.exists():raise FileNotFoundError(event_path)
    events=pd.read_csv(event_path,dtype={"ticker":str})
    times=pd.to_datetime(events.event_time_utc,utc=True,errors="coerce")
    local=times.dt.tz_convert(app.US_TZ)
    session=(local.dt.dayofweek.lt(5)&
             (local.dt.time>=pd.Timestamp("09:30").time())&
             (local.dt.time<=pd.Timestamp("15:27").time()))
    tickers=sorted(set(events.loc[session,"ticker"].astype(str))|{"SPY"})
    output_dir=PRICE/"US_V34";output_dir.mkdir(parents=True,exist_ok=True)
    period1=int(pd.Timestamp("2026-06-29",tz="UTC").timestamp())
    period2=int(pd.Timestamp("2026-08-27",tz="UTC").timestamp())

    def download(ticker:str)->dict[str,Any]:
        path=output_dir/f"{ticker}.parquet"
        if path.exists() and not force:
            try:
                existing=pd.read_parquet(path,columns=["timestamp"])
                if len(existing):return {"ticker":ticker,"rows":len(existing),"status":"cached"}
            except Exception:pass
        try:
            response=_request_with_retry(
                "GET",f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"period1":period1,"period2":period2,"interval":"5m",
                        "events":"history","includePrePost":"false"},
                headers={**HTTP_HEADERS,"Referer":"https://finance.yahoo.com/"},timeout=60,
            )
            payload=response.json().get("chart",{});result=payload.get("result") or []
            if not result:return {"ticker":ticker,"rows":0,"status":str(payload.get("error"))}
            result=result[0];timestamps=result.get("timestamp") or []
            quote=(result.get("indicators",{}).get("quote") or [{}])[0]
            size=len(timestamps)
            frame=pd.DataFrame({
                "timestamp":pd.to_datetime(timestamps,unit="s",utc=True),
                "open":(quote.get("open") or [None]*size),
                "high":(quote.get("high") or [None]*size),
                "low":(quote.get("low") or [None]*size),
                "close":(quote.get("close") or [None]*size),
                "volume":(quote.get("volume") or [None]*size),
            }).dropna(subset=["timestamp","close"])
            frame=frame.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
            temporary=path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
            return {"ticker":ticker,"rows":len(frame),"status":"ok"}
        except Exception as exc:
            return {"ticker":ticker,"rows":0,"status":f"ERROR: {exc}"}

    statuses=[];workers=min(16,max(1,len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download,ticker):ticker for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            statuses.append(future.result())
            if done%25==0 or done==len(futures):
                print(f"[V34 YAHOO] {done}/{len(futures)}",flush=True)
    _atomic_json({
        "selection_uses_labels":False,"source":"Yahoo chart API regular-session 5-minute bars",
        "start":"2026-06-29","end":"2026-08-26","interval":"5m",
        "candidate_tickers":len(tickers)-1,"workers":workers,"runtime":runtime_limits.status(),
        "status":statuses,
    },DATA/"V34_US_PRICE_STATUS.json")


def prepare_v34_sources()->tuple[pd.DataFrame,pd.DataFrame]:
    """Freeze label-blind V34 sources disjoint from every exact event through V33."""
    prior=pd.read_csv(DATA/"events_exact_v33.csv.gz",dtype={"ticker":str,"article_id":str},
                      low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True,errors="coerce")
    prior_ids=set(prior.event_id.astype(str))
    prior_articles=set(prior.get("article_id",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_urls=set(prior.get("url",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.dropna().astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        keep=(frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)&
              ~frame.article_id.fillna("").astype(str).isin(prior_articles)&
              ~frame.url.fillna("").astype(str).isin(prior_urls)).to_numpy(dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            if not keep[position]:continue
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(timestamps,target))
            near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    def covered(frame:pd.DataFrame,market:str,directory:Path)->pd.DataFrame:
        time_column="timestamp" if market=="US" else "datetime"
        timezone=app.US_TZ if market=="US" else app.KR_TZ
        arrays={}
        for ticker in sorted(set(frame.ticker.astype(str))):
            path=directory/f"{ticker}.parquet"
            if not path.exists():continue
            try:
                values=pd.to_datetime(pd.read_parquet(path,columns=[time_column])[time_column],
                                      utc=(market=="US"),errors="coerce").dropna()
                if market=="KR" and getattr(values.dt,"tz",None) is not None:
                    values=values.dt.tz_convert(timezone).dt.tz_localize(None)
                arrays[ticker]=np.sort(values.astype("int64").unique())
            except Exception:continue
        valid=[];cfg=app.Config()
        for row in frame.itertuples(index=False):
            values=arrays.get(str(row.ticker))
            event=pd.Timestamp(row.event_time_utc).tz_convert(timezone)
            if market=="KR":event=event.tz_localize(None)
            entry=event+pd.Timedelta(minutes=cfg.entry_lag_min)
            exit_=entry+pd.Timedelta(minutes=cfg.horizon_min)
            if values is None:valid.append(False);continue
            ei=int(np.searchsorted(values,entry.value,side="left"))
            xi=int(np.searchsorted(values,exit_.value,side="left"))
            ok=ei<len(values) and xi<len(values) and xi>ei and ei>=10
            if ok:
                ok=(pd.Timestamp(values[ei]).date()==event.date() and
                    pd.Timestamp(values[xi]).date()==event.date())
            valid.append(bool(ok))
        return frame.iloc[np.flatnonzero(np.asarray(valid,dtype=bool))].copy()

    outputs={};preprice={}
    for market,directory in (("US",PRICE/"US_V34"),("KR",PRICE/"KR")):
        path=DATA/f"events_google_{market.lower()}_v34_raw.csv"
        frame=pd.read_csv(path,dtype={"ticker":str,"article_id":str},low_memory=False)
        frame=outside(frame,market)
        local=frame.event_time_utc.dt.tz_convert(app.US_TZ if market=="US" else app.KR_TZ)
        start=pd.Timestamp("09:30").time() if market=="US" else pd.Timestamp("09:00").time()
        end=pd.Timestamp("15:27").time() if market=="US" else pd.Timestamp("14:57").time()
        frame=frame[(local.dt.dayofweek<5)&(local.dt.time>=start)&(local.dt.time<=end)&
                    frame.headline.fillna("").astype(str).str.len().ge(3)].copy()
        frame=space(frame);preprice[market]=len(frame);frame=covered(frame,market,directory)
        frame["selection_hash"]=frame.event_id.astype(str).map(
            lambda value:hashlib.sha256(f"V34|{market}|CAP|{value}".encode()).hexdigest()
        )
        frame=(frame.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
               .head(12).drop(columns="selection_hash").sort_values("event_time_utc"))
        frame["source"]=f"{market}_EXACT_V34_SEAL"
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame,DATA/f"events_{market.lower()}_exact_v34_seal.csv")
        outputs[market]=frame

    overlap=sum(len(set(frame.event_id.astype(str))&prior_ids) for frame in outputs.values())
    article_overlap=sum(len(set(frame.article_id.fillna("").astype(str))&prior_articles)
                        for frame in outputs.values())
    url_overlap=sum(len(set(frame.url.fillna("").astype(str))&prior_urls)
                    for frame in outputs.values())
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V33","embargo_minutes":35,
        "split":"pure external SEAL; development and selection use opened V33 or earlier",
        "us_definition":"recent exact Google News RSS plus independent Yahoo 5-minute prices; max12/ticker",
        "kr_definition":"unused recent exact Google News RSS plus cached Daum minute prices; max12/ticker",
        "query_interval":f"{V34_NEWS_START} through 2026-08-25",
        "preprice_candidates":preprice,"seal_counts":{k:len(v) for k,v in outputs.items()},
        "prior_event_id_overlap":overlap,"prior_article_overlap":article_overlap,
        "prior_url_overlap":url_overlap,
    },DATA/"V34_SOURCE_STATUS.json")
    if overlap or article_overlap or url_overlap:raise RuntimeError("V34 source leakage")
    print(f"[V34 SOURCES] US seal={len(outputs['US'])}; KR seal={len(outputs['KR'])}",flush=True)
    return outputs["US"],outputs["KR"]


def build_v34_panel()->pd.DataFrame:
    """Label frozen V34 sources and append the now-opened V33 history."""
    cfg=app.Config();paths=[
        DATA/"events_exact_v33.csv.gz",DATA/"events_us_exact_v34_seal.csv",
        DATA/"events_kr_exact_v34_seal.csv",
    ]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in paths]
    active=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v34.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active
    sources={"US_EXACT_V34_SEAL","KR_EXACT_V34_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v33.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    label_cfg=replace(cfg,us_price_dir="price_data/US_V34")
    active=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v34_new.csv.gz"
    try:new=app.build_labeled(events[events.source.isin(sources)].copy(),label_cfg)
    finally:app.LABELED_FILE=active
    new["event_time_utc"]=pd.to_datetime(new.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new],ignore_index=True,sort=False).drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v34.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V34_LABEL_STATUS.json")
    print(f"[V34 LABELED] new={len(new)} counts={counts} total={len(labeled)}",flush=True)
    return labeled




V35_EVENT_START="2026-06-29"
V35_EVENT_END="2026-08-27"
V35_KR_EVENT_START="2026-01-01"


def collect_v35_sec()->pd.DataFrame:
    """Collect recent SEC filings with exact acceptance timestamps, without outcomes."""
    universe=pd.read_csv(DATA/"universe_v33_us_rss.csv",dtype={"ticker":str})
    universe["ticker"]=(universe.ticker.astype(str).str.upper().str.replace(".","-",regex=False))
    collector=app.SECCollector(app.Config())
    ticker_map=collector.ticker_to_cik()
    start=pd.Timestamp(V35_EVENT_START,tz="UTC")
    end=pd.Timestamp(V35_EVENT_END,tz="UTC")
    rows=[];statuses=[]
    for done,row in enumerate(universe.itertuples(index=False),1):
        ticker=str(row.ticker);company=str(row.company);cik=ticker_map.get(ticker)
        if cik is None:
            statuses.append({"ticker":ticker,"status":"no-cik","kept":0})
            continue
        kept=0
        try:
            payload=collector.get_json(collector.SUBMISSIONS.format(cik=cik))
            recent=payload.get("filings",{}).get("recent",{}) or {}
            accessions=recent.get("accessionNumber",[]) or []
            forms=recent.get("form",[]) or []
            accepted=recent.get("acceptanceDateTime",[]) or []
            primary=recent.get("primaryDocument",[]) or []
            items=recent.get("items",[]) or []
            for index,accession in enumerate(accessions):
                form=str(forms[index] if index<len(forms) else "")
                if not form:continue
                raw_time=str(accepted[index] if index<len(accepted) else "")
                if not raw_time or "T" not in raw_time:continue
                event_time=pd.Timestamp(raw_time)
                if event_time.tzinfo is None:
                    event_time=event_time.tz_localize(app.US_TZ).tz_convert("UTC")
                else:event_time=event_time.tz_convert("UTC")
                if not (start<=event_time<end):continue
                accession=str(accession);acc_nodash=accession.replace("-","")
                item_text=str(items[index] if index<len(items) else "").strip()
                primary_name=str(primary[index] if index<len(primary) else "").strip()
                event_id=f"SEC:{cik}:{accession}"
                headline=f"{form} {item_text}".strip()
                url=(f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                     f"{acc_nodash}/{primary_name}" if primary_name else
                     f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/")
                rows.append({
                    "event_id":event_id,"market":"US","ticker":ticker,
                    "company":company or str(payload.get("name","") or ""),
                    "event_time_utc":event_time.isoformat(),"source":"SEC_RECENT_V35_RAW",
                    "form":form,"headline":headline,"body":"",
                    "event_type":app.infer_event_type(headline,form),
                    "timestamp_quality":"EXACT","url":url,"article_id":event_id,
                });kept+=1
            statuses.append({"ticker":ticker,"status":"ok","kept":kept})
        except Exception as exc:
            statuses.append({"ticker":ticker,"status":f"ERROR: {exc}","kept":0})
        if done%50==0 or done==len(universe):
            print(f"[V35 SEC] {done}/{len(universe)} rows={len(rows)}",flush=True)
    if not rows:raise RuntimeError("V35 recent SEC collection returned no events")
    frame=(pd.DataFrame(rows).sort_values(["event_id","ticker"])
           .drop_duplicates("event_id").sort_values("event_time_utc").reset_index(drop=True))
    _atomic_csv(frame,DATA/"events_sec_us_v35_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,"source":"SEC EDGAR submissions acceptanceDateTime",
        "query_interval":f"{V35_EVENT_START} through 2026-08-26",
        "forms":"all recent SEC submission forms","universe_tickers":len(universe),
        "unique_events":len(frame),"requests":statuses,
    },DATA/"V35_US_SEC_STATUS.json")
    return frame


def collect_v35_naver()->pd.DataFrame:
    """Collect a fresh exact-minute Naver ticker-news pool in a V35-only cache."""
    universe=pd.read_csv(DATA/"universe_v33_kr_rss.csv",dtype={"ticker":str})
    universe["ticker"]=universe.ticker.astype(str).str.zfill(6)
    start=pd.Timestamp(V35_KR_EVENT_START,tz=app.KR_TZ)
    end=pd.Timestamp(V35_EVENT_END,tz=app.KR_TZ)

    def fetch(ticker:str,company:str)->tuple[list[dict[str,Any]],dict[str,Any]]:
        rows=[];seen=set();pages=0
        try:
            for page in range(1,81):
                response=_request_with_retry(
                    "GET",NAVER_NEWS_URL.format(ticker=ticker),
                    params={"page":page,"pageSize":20},
                    headers={**HTTP_HEADERS,"Referer":"https://m.stock.naver.com/"},timeout=45,
                )
                groups=json.loads(response.content.decode("utf-8"))
                items=[item for group in groups for item in group.get("items",[])]
                if not items:break
                pages=page;oldest=None
                for item in items:
                    article_id=str(item.get("id","") or "")
                    timestamp=pd.to_datetime(str(item.get("datetime","") or ""),
                                             format="%Y%m%d%H%M",errors="coerce")
                    if pd.isna(timestamp):continue
                    timestamp=timestamp.tz_localize(app.KR_TZ)
                    oldest=timestamp if oldest is None else min(oldest,timestamp)
                    if not (start<=timestamp<end) or article_id in seen:continue
                    seen.add(article_id)
                    headline=html.unescape(str(item.get("titleFull") or item.get("title") or ""))
                    body=html.unescape(str(item.get("body","") or ""))
                    rows.append({
                        "event_id":f"NAVER:{ticker}:{article_id}","market":"KR",
                        "ticker":ticker,"company":company,
                        "event_time_utc":timestamp.tz_convert("UTC").isoformat(),
                        "source":"NAVER_NEWS_V35_RAW","form":"","headline":headline,
                        "body":body,"event_type":app.infer_event_type(f"{headline} {body}"),
                        "timestamp_quality":"EXACT",
                        "url":str(item.get("mobileNewsUrl") or item.get("url") or ""),
                        "article_id":article_id,
                    })
                if oldest is not None and oldest<start:break
            return rows,{"ticker":ticker,"pages":pages,"kept":len(rows),"status":"ok"}
        except Exception as exc:
            return [],{"ticker":ticker,"pages":pages,"kept":0,"status":f"ERROR: {exc}"}

    rows=[];statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(universe)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,str(row.ticker),str(row.company)):str(row.ticker)
                 for row in universe.itertuples(index=False)}
        for done,future in enumerate(as_completed(futures),1):
            records,status=future.result();rows.extend(records);statuses.append(status)
            if done%25==0 or done==len(futures):
                print(f"[V35 NAVER] {done}/{len(futures)} rows={len(rows)}",flush=True)
    if not rows:raise RuntimeError("V35 fresh Naver collection returned no events")
    frame=pd.DataFrame(rows)
    frame["ticker_hash"]=frame.ticker.astype(str).map(
        lambda value:hashlib.sha256(f"V35|NAVER|{value}".encode()).hexdigest())
    frame=(frame.sort_values(["article_id","ticker_hash"]).drop_duplicates("article_id")
           .drop(columns="ticker_hash").sort_values("event_time_utc").reset_index(drop=True))
    _atomic_csv(frame,DATA/"events_naver_kr_v35_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,"source":"Naver Finance ticker-news API",
        "query_interval":f"{V35_KR_EVENT_START} through 2026-08-26",
        "universe_tickers":len(universe),"workers":workers,"unique_articles":len(frame),
        "requests":statuses,
    },DATA/"V35_KR_NAVER_STATUS.json")
    return frame


def collect_v35_us_prices(force:bool=False)->None:
    """Download independent Yahoo 5-minute bars for every V35 US candidate ticker."""
    paths=[DATA/"events_sec_us_v35_raw.csv",DATA/"events_google_us_v34_raw.csv"]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    events=pd.concat([pd.read_csv(path,dtype={"ticker":str},low_memory=False)
                      for path in paths],ignore_index=True,sort=False)
    times=pd.to_datetime(events.event_time_utc,utc=True,errors="coerce")
    local=times.dt.tz_convert(app.US_TZ)
    session=(local.dt.dayofweek.lt(5)&
             (local.dt.time>=pd.Timestamp("09:30").time())&
             (local.dt.time<=pd.Timestamp("15:27").time()))
    tickers=sorted(set(events.loc[session,"ticker"].astype(str))|{"SPY"})
    output_dir=PRICE/"US_V35";output_dir.mkdir(parents=True,exist_ok=True)
    period1=int(pd.Timestamp(V35_EVENT_START,tz="UTC").timestamp())
    period2=int(pd.Timestamp(V35_EVENT_END,tz="UTC").timestamp())

    def download(ticker:str)->dict[str,Any]:
        path=output_dir/f"{ticker}.parquet"
        if path.exists() and not force:
            try:
                existing=pd.read_parquet(path,columns=["timestamp"])
                if len(existing):return {"ticker":ticker,"rows":len(existing),"status":"cached"}
            except Exception:pass
        try:
            response=_request_with_retry(
                "GET",f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"period1":period1,"period2":period2,"interval":"5m",
                        "events":"history","includePrePost":"false"},
                headers={**HTTP_HEADERS,"Referer":"https://finance.yahoo.com/"},timeout=60,
            )
            payload=response.json().get("chart",{});result=payload.get("result") or []
            if not result:return {"ticker":ticker,"rows":0,"status":str(payload.get("error"))}
            result=result[0];timestamps=result.get("timestamp") or []
            quote=(result.get("indicators",{}).get("quote") or [{}])[0];size=len(timestamps)
            frame=pd.DataFrame({
                "timestamp":pd.to_datetime(timestamps,unit="s",utc=True),
                "open":quote.get("open") or [None]*size,
                "high":quote.get("high") or [None]*size,
                "low":quote.get("low") or [None]*size,
                "close":quote.get("close") or [None]*size,
                "volume":quote.get("volume") or [None]*size,
            }).dropna(subset=["timestamp","close"])
            frame=frame.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
            temporary=path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
            return {"ticker":ticker,"rows":len(frame),"status":"ok"}
        except Exception as exc:return {"ticker":ticker,"rows":0,"status":f"ERROR: {exc}"}

    statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download,ticker):ticker for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            statuses.append(future.result())
            if done%25==0 or done==len(futures):
                print(f"[V35 YAHOO] {done}/{len(futures)}",flush=True)
    _atomic_json({
        "selection_uses_labels":False,"source":"Yahoo chart API regular-session 5-minute bars",
        "start":V35_EVENT_START,"end":"2026-08-26","interval":"5m",
        "candidate_tickers":len(tickers)-1,"workers":workers,"runtime":runtime_limits.status(),
        "status":statuses,
    },DATA/"V35_US_PRICE_STATUS.json")


def collect_v35_kr_prices(force:bool=False)->None:
    """Refresh recent Daum bars for all in-session V35 Naver candidates."""
    collect_kr_prices(
        V35_KR_EVENT_START,"2026-08-26",workers=runtime_limits.THREAD_COUNT,
        force=force,event_path=DATA/"events_naver_kr_v35_raw.csv",material_only=False,
    )
    status_path=DATA/"KR_PRICE_STATUS.json"
    if status_path.exists():
        _atomic_json({
            "selection_uses_labels":False,"source":"Daum Finance 5-minute charts",
            "start":V35_KR_EVENT_START,"end":"2026-08-26",
            "workers":runtime_limits.THREAD_COUNT,"runtime":runtime_limits.status(),
            "status":json.loads(status_path.read_text(encoding="utf-8")),
        },DATA/"V35_KR_PRICE_STATUS.json")


def prepare_v35_sources()->tuple[pd.DataFrame,pd.DataFrame]:
    """Freeze a label-blind V35 seal disjoint from every exact event through V34."""
    prior=pd.read_csv(DATA/"events_exact_v34.csv.gz",dtype={"ticker":str,"article_id":str},
                      low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True,errors="coerce")
    prior_ids=set(prior.event_id.astype(str))
    prior_articles=set(prior.get("article_id",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_urls=set(prior.get("url",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.dropna().astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        for column in ("article_id","url"):
            if column not in frame:frame[column]=""
        keep=(frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)&
              ~frame.article_id.fillna("").astype(str).isin(prior_articles)&
              ~frame.url.fillna("").astype(str).isin(prior_urls)).to_numpy(dtype=bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            if not keep[position]:continue
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(timestamps,target))
            near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):
                    accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    def covered(frame:pd.DataFrame,market:str,directory:Path)->pd.DataFrame:
        time_column="timestamp" if market=="US" else "datetime"
        timezone=app.US_TZ if market=="US" else app.KR_TZ;arrays={}
        for ticker in sorted(set(frame.ticker.astype(str))):
            path=directory/f"{ticker}.parquet"
            if not path.exists():continue
            try:
                values=pd.to_datetime(pd.read_parquet(path,columns=[time_column])[time_column],
                                      utc=(market=="US"),errors="coerce").dropna()
                if market=="KR" and getattr(values.dt,"tz",None) is not None:
                    values=values.dt.tz_convert(timezone).dt.tz_localize(None)
                arrays[ticker]=np.sort(values.astype("int64").unique())
            except Exception:continue
        valid=[];cfg=app.Config()
        for row in frame.itertuples(index=False):
            values=arrays.get(str(row.ticker));event=pd.Timestamp(row.event_time_utc).tz_convert(timezone)
            if market=="KR":event=event.tz_localize(None)
            entry=event+pd.Timedelta(minutes=cfg.entry_lag_min)
            exit_=entry+pd.Timedelta(minutes=cfg.horizon_min)
            if values is None:valid.append(False);continue
            ei=int(np.searchsorted(values,entry.value,side="left"));xi=int(np.searchsorted(values,exit_.value,side="left"))
            ok=ei<len(values) and xi<len(values) and xi>ei and ei>=10
            if ok:ok=(pd.Timestamp(values[ei]).date()==event.date() and
                      pd.Timestamp(values[xi]).date()==event.date())
            valid.append(bool(ok))
        return frame.iloc[np.flatnonzero(np.asarray(valid,dtype=bool))].copy()

    raw={
        "US":pd.concat([
            pd.read_csv(DATA/"events_sec_us_v35_raw.csv",dtype={"ticker":str,"article_id":str},low_memory=False),
            pd.read_csv(DATA/"events_google_us_v34_raw.csv",dtype={"ticker":str,"article_id":str},low_memory=False),
        ],ignore_index=True,sort=False),
        "KR":pd.concat([
            pd.read_csv(DATA/"events_naver_kr_v35_raw.csv",dtype={"ticker":str,"article_id":str},low_memory=False),
            pd.read_csv(DATA/"events_google_kr_v34_raw.csv",dtype={"ticker":str,"article_id":str},low_memory=False),
        ],ignore_index=True,sort=False),
    }
    outputs={};stages={}
    for market,directory in (("US",PRICE/"US_V35"),("KR",PRICE/"KR")):
        frame=outside(raw[market],market);outside_count=len(frame)
        local=frame.event_time_utc.dt.tz_convert(app.US_TZ if market=="US" else app.KR_TZ)
        start=pd.Timestamp("09:30").time() if market=="US" else pd.Timestamp("09:00").time()
        end=pd.Timestamp("15:27").time() if market=="US" else pd.Timestamp("14:57").time()
        frame=frame[(local.dt.dayofweek<5)&(local.dt.time>=start)&(local.dt.time<=end)&
                    frame.headline.fillna("").astype(str).str.len().ge(3)].copy()
        session_count=len(frame);frame=space(frame);spaced_count=len(frame);frame=covered(frame,market,directory)
        covered_count=len(frame);frame["selection_hash"]=frame.event_id.astype(str).map(
            lambda value:hashlib.sha256(f"V35|{market}|CAP|{value}".encode()).hexdigest())
        frame=(frame.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
               .head(20).drop(columns="selection_hash").sort_values("event_time_utc"))
        frame["source"]=f"{market}_EXACT_V35_SEAL"
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame,DATA/f"events_{market.lower()}_exact_v35_seal.csv")
        outputs[market]=frame
        stages[market]={"outside_prior":outside_count,"in_session":session_count,
                        "spaced":spaced_count,"price_covered":covered_count,"sealed":len(frame)}

    overlap=sum(len(set(frame.event_id.astype(str))&prior_ids) for frame in outputs.values())
    article_overlap=sum(len((set(frame.article_id.fillna("").astype(str))-{""})&prior_articles)
                        for frame in outputs.values())
    url_overlap=sum(len((set(frame.url.fillna("").astype(str))-{""})&prior_urls)
                    for frame in outputs.values())
    _atomic_json({
        "selection_uses_labels":False,"prior_version":"V34","embargo_minutes":35,
        "split":"pure external SEAL; development and selection use opened V34 or earlier",
        "us_definition":"recent exact SEC acceptance events plus unused recent Google News; independent Yahoo 5-minute prices; max20/ticker",
        "kr_definition":"fresh exact Naver ticker news plus unused Google News and cached Daum minute prices; max20/ticker",
        "query_interval":f"{V35_EVENT_START} through 2026-08-26","stages":stages,
        "seal_counts":{key:len(value) for key,value in outputs.items()},
        "prior_event_id_overlap":overlap,"prior_article_overlap":article_overlap,
        "prior_url_overlap":url_overlap,
    },DATA/"V35_SOURCE_STATUS.json")
    if overlap or article_overlap or url_overlap:raise RuntimeError("V35 source leakage")
    if len(outputs["US"])<250 or len(outputs["KR"])<100 or sum(map(len,outputs.values()))<500:
        raise RuntimeError(f"V35 source minimum failed: US={len(outputs['US'])}, KR={len(outputs['KR'])}")
    print(f"[V35 SOURCES] US seal={len(outputs['US'])}; KR seal={len(outputs['KR'])}",flush=True)
    return outputs["US"],outputs["KR"]


def build_v35_panel()->pd.DataFrame:
    """Label frozen V35 sources and append the now-opened V34 history."""
    cfg=app.Config();paths=[DATA/"events_exact_v34.csv.gz",
        DATA/"events_us_exact_v35_seal.csv",DATA/"events_kr_exact_v35_seal.csv"]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in paths]
    active=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v35.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active
    sources={"US_EXACT_V35_SEAL","KR_EXACT_V35_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v34.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    label_cfg=replace(cfg,us_price_dir="price_data/US_V35")
    active=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v35_new.csv.gz"
    try:new=app.build_labeled(events[events.source.isin(sources)].copy(),label_cfg)
    finally:app.LABELED_FILE=active
    new["event_time_utc"]=pd.to_datetime(new.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new],ignore_index=True,sort=False).drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v35.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
                                         for (market,source),count in counts.items()},
                  "total":len(labeled)},DATA/"V35_LABEL_STATUS.json")
    print(f"[V35 LABELED] new={len(new)} counts={counts} total={len(labeled)}",flush=True)
    return labeled




V36_US_START="2026-06-29"
V36_US_END="2026-08-27"
V36_KR_1M_START="2026-07-28"
V36_KR_1M_END="2026-08-27"


def collect_v36_sec()->pd.DataFrame:
    """Collect exact recent filings for health issuers outside the V35 universe."""
    collector=app.SECCollector(app.Config());raw=collector.get_json(collector.TICKER_MAP)
    prior_tickers=set(pd.read_csv(DATA/"universe_v33_us_rss.csv",dtype={"ticker":str}).ticker.astype(str))
    terms=("therapeut","pharma","biotech","bioscience","medical","healthcare",
           "oncolog","diagnostic","life science","laborator","immun","genetic",
           "biologic","vaccine","surgical","medicine","medtech","genom","dental",
           "ophthalm","clinical","neuro")
    rows=[]
    for item in raw.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip();lower=company.lower()
        if ticker and ticker not in prior_tickers and any(term in lower for term in terms):
            rows.append({"ticker":ticker,"company":company,"cik":int(item["cik_str"])})
    universe=(pd.DataFrame(rows).drop_duplicates("cik").sort_values(["ticker","cik"])
              .reset_index(drop=True))
    _atomic_csv(universe,DATA/"universe_v36_us_sec.csv")
    start=pd.Timestamp(V36_US_START,tz="UTC");end=pd.Timestamp(V36_US_END,tz="UTC")
    events=[];statuses=[]
    for done,row in enumerate(universe.itertuples(index=False),1):
        kept=0
        try:
            payload=collector.get_json(collector.SUBMISSIONS.format(cik=int(row.cik)))
            recent=payload.get("filings",{}).get("recent",{}) or {}
            accessions=recent.get("accessionNumber",[]) or [];forms=recent.get("form",[]) or []
            accepted=recent.get("acceptanceDateTime",[]) or [];primary=recent.get("primaryDocument",[]) or []
            items=recent.get("items",[]) or []
            for index,accession in enumerate(accessions):
                form=str(forms[index] if index<len(forms) else "").strip()
                raw_time=str(accepted[index] if index<len(accepted) else "")
                if not form or not raw_time or "T" not in raw_time:continue
                timestamp=pd.Timestamp(raw_time)
                if timestamp.tzinfo is None:timestamp=timestamp.tz_localize(app.US_TZ).tz_convert("UTC")
                else:timestamp=timestamp.tz_convert("UTC")
                if not (start<=timestamp<end):continue
                accession=str(accession);acc_nodash=accession.replace("-","")
                primary_name=str(primary[index] if index<len(primary) else "").strip()
                item_text=str(items[index] if index<len(items) else "").strip()
                event_id=f"SEC:{int(row.cik)}:{accession}"
                url=(f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc_nodash}/"
                     f"{primary_name}" if primary_name else
                     f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc_nodash}/")
                headline=f"{form} {item_text}".strip()
                events.append({
                    "event_id":event_id,"market":"US","ticker":str(row.ticker),
                    "company":str(row.company),"event_time_utc":timestamp.isoformat(),
                    "source":"SEC_EXPANDED_V36_RAW","form":form,"headline":headline,
                    "body":"","event_type":app.infer_event_type(headline,form),
                    "timestamp_quality":"EXACT","url":url,"article_id":event_id,
                });kept+=1
            statuses.append({"ticker":str(row.ticker),"status":"ok","kept":kept})
        except Exception as exc:
            statuses.append({"ticker":str(row.ticker),"status":f"ERROR: {exc}","kept":0})
        if done%25==0 or done==len(universe):
            print(f"[V36 SEC] {done}/{len(universe)} rows={len(events)}",flush=True)
    if not events:raise RuntimeError("V36 expanded SEC collection returned no events")
    frame=(pd.DataFrame(events).sort_values(["event_id","ticker"]).drop_duplicates("event_id")
           .sort_values("event_time_utc").reset_index(drop=True))
    _atomic_csv(frame,DATA/"events_sec_us_v36_raw.csv")
    _atomic_json({
        "selection_uses_labels":False,"source":"SEC EDGAR submissions acceptanceDateTime",
        "universe_definition":"new issuer outside V35 US universe; health-name terms",
        "universe_terms":list(terms),"universe_tickers":len(universe),
        "query_interval":f"{V36_US_START} through 2026-08-26","unique_events":len(frame),
        "requests":statuses,
    },DATA/"V36_US_SEC_STATUS.json")
    return frame


def collect_v36_us_prices(force:bool=False)->None:
    """Download exploratory Yahoo 5-minute bars; never official V36 prices."""
    events=pd.read_csv(DATA/"events_sec_us_v36_raw.csv",dtype={"ticker":str},low_memory=False)
    times=pd.to_datetime(events.event_time_utc,utc=True,errors="coerce");local=times.dt.tz_convert(app.US_TZ)
    session=(local.dt.dayofweek.lt(5)&(local.dt.time>=pd.Timestamp("09:30").time())&
             (local.dt.time<=pd.Timestamp("15:27").time()))
    tickers=sorted(set(events.loc[session,"ticker"].astype(str))|{"SPY"})
    output_dir=PRICE/"US_V36";output_dir.mkdir(parents=True,exist_ok=True)
    period1=int(pd.Timestamp(V36_US_START,tz="UTC").timestamp())
    period2=int(pd.Timestamp(V36_US_END,tz="UTC").timestamp())

    def download(ticker:str)->dict[str,Any]:
        path=output_dir/f"{ticker}.parquet"
        if path.exists() and not force:
            try:
                current=pd.read_parquet(path,columns=["timestamp"])
                if len(current):return {"ticker":ticker,"rows":len(current),"status":"cached"}
            except Exception:pass
        try:
            response=_request_with_retry(
                "GET",f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"period1":period1,"period2":period2,"interval":"5m",
                        "events":"history","includePrePost":"false"},
                headers={**HTTP_HEADERS,"Referer":"https://finance.yahoo.com/"},timeout=60)
            payload=response.json().get("chart",{});result=payload.get("result") or []
            if not result:return {"ticker":ticker,"rows":0,"status":str(payload.get("error"))}
            result=result[0];timestamps=result.get("timestamp") or []
            quote=(result.get("indicators",{}).get("quote") or [{}])[0];size=len(timestamps)
            frame=pd.DataFrame({"timestamp":pd.to_datetime(timestamps,unit="s",utc=True),
                "open":quote.get("open") or [None]*size,"high":quote.get("high") or [None]*size,
                "low":quote.get("low") or [None]*size,"close":quote.get("close") or [None]*size,
                "volume":quote.get("volume") or [None]*size}).dropna(subset=["timestamp","close"])
            frame=frame.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
            temporary=path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
            return {"ticker":ticker,"rows":len(frame),"status":"ok"}
        except Exception as exc:return {"ticker":ticker,"rows":0,"status":f"ERROR: {exc}"}

    statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download,ticker):ticker for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            statuses.append(future.result())
            if done%25==0 or done==len(futures):print(f"[V36 YAHOO] {done}/{len(futures)}",flush=True)
    _atomic_json({"selection_uses_labels":False,"source":"Yahoo chart API 5-minute bars",
        "classification":"PROXY_5M_NOT_EXACT_T2_CERTIFICATION",
        "start":V36_US_START,"end":"2026-08-26","candidate_tickers":len(tickers)-1,
        "workers":workers,"runtime":runtime_limits.status(),"status":statuses,
    },DATA/"V36_US_PRICE_STATUS.json")


def collect_v36_kr_1m_prices(force:bool=False)->None:
    """Collect real Yahoo 1-minute KR bars for contract-eligible V36 events.

    Yahoo limits 1-minute history requests to short windows, so requests are
    chunked into six-day intervals and filtered back to the requested bounds.
    Both KOSPI and KOSDAQ suffixes are probed without looking at event labels;
    the symbol returning the largest valid 1m series is frozen in the status.
    """
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    healthcare=set(seed.loc[seed.market.eq("KR"),"ticker"].astype(str).str.zfill(6))
    frames=[]
    for path in (DATA/"events_exact_v35.csv.gz",DATA/"events_naver_kr_v35_raw.csv",
                 DATA/"events_kind_real.csv"):
        frame=pd.read_csv(path,dtype={"ticker":str},low_memory=False)
        frame=frame[frame.market.astype(str).eq("KR")].copy()
        frame["ticker"]=frame.ticker.astype(str).str.zfill(6)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        frames.append(frame)
    events=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    start=pd.Timestamp(V36_KR_1M_START,tz="UTC");end=pd.Timestamp(V36_KR_1M_END,tz="UTC")
    events=events[(events.event_time_utc>=start)&(events.event_time_utc<end)&
                  events.ticker.isin(healthcare)].copy()
    tickers=sorted(set(events.ticker.astype(str))|{"069500"})
    output_dir=PRICE/"KR_V36_1M";output_dir.mkdir(parents=True,exist_ok=True)
    boundaries=list(pd.date_range(start,end,freq="6D"))
    if not boundaries or boundaries[-1]<end:boundaries.append(end)
    chunks=list(zip(boundaries[:-1],boundaries[1:]))

    def fetch_symbol(symbol:str)->tuple[pd.DataFrame,list[dict[str,Any]]]:
        pieces=[];requests_=[]
        for chunk_start,chunk_end in chunks:
            try:
                response=_request_with_retry(
                    "GET",f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"period1":int(chunk_start.timestamp()),"period2":int(chunk_end.timestamp()),
                            "interval":"1m","events":"history","includePrePost":"false"},
                    headers={**HTTP_HEADERS,"Referer":"https://finance.yahoo.com/"},timeout=60,
                )
                chart=response.json().get("chart",{});result=chart.get("result") or []
                if not result:
                    requests_.append({"start":chunk_start.isoformat(),"end":chunk_end.isoformat(),"rows":0,
                                      "error":chart.get("error")});continue
                result=result[0];meta=result.get("meta",{});timestamps=result.get("timestamp") or []
                quote=(result.get("indicators",{}).get("quote") or [{}])[0];size=len(timestamps)
                frame=pd.DataFrame({
                    "timestamp":pd.to_datetime(timestamps,unit="s",utc=True),
                    "open":quote.get("open") or [None]*size,"high":quote.get("high") or [None]*size,
                    "low":quote.get("low") or [None]*size,"close":quote.get("close") or [None]*size,
                    "volume":quote.get("volume") or [None]*size,
                }).dropna(subset=["timestamp","open","close"])
                frame=frame[(frame.timestamp>=chunk_start)&(frame.timestamp<chunk_end)].copy()
                granularity=str(meta.get("dataGranularity",result.get("meta",{}).get("range","")))
                if granularity!="1m":frame=frame.iloc[0:0]
                if not frame.empty:pieces.append(frame)
                requests_.append({"start":chunk_start.isoformat(),"end":chunk_end.isoformat(),"rows":len(frame),
                                  "granularity":granularity,
                                  "exchange_timezone":meta.get("exchangeTimezoneName")})
            except Exception as exc:
                requests_.append({"start":chunk_start.isoformat(),"end":chunk_end.isoformat(),"rows":0,"error":str(exc)})
        combined=(pd.concat(pieces,ignore_index=True,sort=False) if pieces else
                  pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]))
        if len(combined):
            combined=(combined.sort_values("timestamp")
                      .drop_duplicates("timestamp",keep="last").reset_index(drop=True))
        return combined,requests_

    def download(ticker:str)->dict[str,Any]:
        path=output_dir/f"{ticker}.parquet"
        if path.exists() and not force:
            try:
                current=pd.read_parquet(path)
                if len(current):
                    cadence=pd.to_datetime(current.timestamp,utc=True).sort_values().diff().dt.total_seconds()
                    return {"ticker":ticker,"rows":len(current),"status":"cached",
                            "selected_symbol":current.attrs.get("yahoo_symbol","") or None,
                            "observed_min_gap_sec":float(cadence[cadence>0].min()) if (cadence>0).any() else None}
            except Exception:pass
        candidates=[]
        symbols=["069500.KS"] if ticker=="069500" else [f"{ticker}.KS",f"{ticker}.KQ"]
        for symbol in symbols:
            frame,request_status=fetch_symbol(symbol)
            candidates.append((len(frame),symbol,frame,request_status))
        _,symbol,frame,request_status=max(candidates,key=lambda item:item[0])
        if frame.empty:
            return {"ticker":ticker,"rows":0,"status":"no 1m data",
                    "selected_symbol":None,"candidates":[{"symbol":item[1],"rows":item[0],
                    "requests":item[3]} for item in candidates]}
        frame.attrs.update({"yahoo_symbol":symbol,"price_source":"YAHOO_CHART_KR_1M",
                            "price_cadence_sec":60,"bar_timestamp_semantics":"OPEN"})
        temporary=path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary,index=False,compression="zstd");temporary.replace(path)
        cadence=frame.timestamp.sort_values().diff().dt.total_seconds()
        return {"ticker":ticker,"rows":len(frame),"status":"ok","selected_symbol":symbol,
                "min":frame.timestamp.min().isoformat(),"max":frame.timestamp.max().isoformat(),
                "observed_min_gap_sec":float(cadence[cadence>0].min()) if (cadence>0).any() else None,
                "requests":request_status}

    statuses=[];workers=min(runtime_limits.THREAD_COUNT,max(1,len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download,ticker):ticker for ticker in tickers}
        for done,future in enumerate(as_completed(futures),1):
            statuses.append(future.result())
            if done%10==0 or done==len(futures):
                print(f"[V36 KR YAHOO 1M] {done}/{len(futures)}",flush=True)
    _atomic_json({
        "selection_uses_labels":False,"source":"Yahoo chart API real 1-minute bars",
        "classification":"EXACT_T2_30M_OFFICIAL_CANDIDATE","timestamp_semantics":"OPEN",
        "timezone":"UTC; exchangeTimezoneName must be Asia/Seoul","interval":"1m",
        "start":V36_KR_1M_START,"end":"2026-08-26","healthcare_policy":"KR seed universe",
        "candidate_tickers":len(tickers)-1,"benchmark":"069500.KS","workers":workers,
        "runtime":runtime_limits.status(),"status":statuses,
    },DATA/"V36_KR_1M_PRICE_STATUS.json")
    usable=sum(row["rows"]>0 for row in statuses if row["ticker"]!="069500")
    print(f"[V36 KR YAHOO 1M] usable tickers={usable}/{len(tickers)-1}",flush=True)


def prepare_v36_sources()->tuple[pd.DataFrame,pd.DataFrame]:
    """Freeze a label-blind V36 seal disjoint from every event through V35."""
    raise RuntimeError(
        "legacy V36 builder disabled: it used all SEC forms and 5-minute bars; "
        "run the exact-contract V36 source pipeline instead"
    )
    prior=pd.read_csv(DATA/"events_exact_v35.csv.gz",dtype={"ticker":str,"article_id":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True,errors="coerce")
    prior_ids=set(prior.event_id.astype(str));prior_articles=set(
        prior.get("article_id",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_urls=set(prior.get("url",pd.Series(dtype=str)).fillna("").astype(str))-{""}
    prior_times={(market,str(ticker)):np.sort(group.event_time_utc.dropna().astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    embargo=int(pd.Timedelta(minutes=35).value)

    def outside(frame:pd.DataFrame,market:str)->pd.DataFrame:
        frame=frame.copy();frame["ticker"]=frame.ticker.astype(str)
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
        for column in ("article_id","url"):
            if column not in frame:frame[column]=""
        keep=(frame.event_time_utc.notna()&~frame.event_id.astype(str).isin(prior_ids)&
              ~frame.article_id.fillna("").astype(str).isin(prior_articles)&
              ~frame.url.fillna("").astype(str).isin(prior_urls)).to_numpy(bool)
        for position,row in enumerate(frame.itertuples(index=False)):
            if not keep[position]:continue
            timestamps=prior_times.get((market,str(row.ticker)))
            if timestamps is None or not len(timestamps):continue
            target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(timestamps,target));near=[]
            if index<len(timestamps):near.append(abs(int(timestamps[index])-target))
            if index:near.append(abs(int(timestamps[index-1])-target))
            if near and min(near)<=embargo:keep[position]=False
        return frame.iloc[np.flatnonzero(keep)].copy()

    def space(frame:pd.DataFrame)->pd.DataFrame:
        accepted=[]
        for _,group in frame.sort_values(["ticker","event_time_utc","event_id"]).groupby("ticker"):
            last=None
            for index,timestamp in zip(group.index,group.event_time_utc):
                if last is None or timestamp-last>pd.Timedelta(minutes=35):accepted.append(index);last=timestamp
        return frame.loc[accepted].copy()

    def covered(frame:pd.DataFrame,market:str,directory:Path)->pd.DataFrame:
        time_column="timestamp" if market=="US" else "datetime";timezone=app.US_TZ if market=="US" else app.KR_TZ
        arrays={}
        for ticker in sorted(set(frame.ticker.astype(str))):
            path=directory/f"{ticker}.parquet"
            if not path.exists():continue
            try:
                series=pd.to_datetime(pd.read_parquet(path,columns=[time_column])[time_column],
                                      utc=(market=="US"),errors="coerce").dropna()
                if market=="KR" and getattr(series.dt,"tz",None) is not None:
                    series=series.dt.tz_convert(timezone).dt.tz_localize(None)
                arrays[ticker]=np.sort(series.astype("int64").unique())
            except Exception:continue
        valid=[];cfg=app.Config()
        for row in frame.itertuples(index=False):
            values_=arrays.get(str(row.ticker));event=pd.Timestamp(row.event_time_utc).tz_convert(timezone)
            if market=="KR":event=event.tz_localize(None)
            entry=event+pd.Timedelta(minutes=cfg.entry_lag_min);exit_=entry+pd.Timedelta(minutes=cfg.horizon_min)
            if values_ is None:valid.append(False);continue
            ei=int(np.searchsorted(values_,entry.value));xi=int(np.searchsorted(values_,exit_.value))
            ok=ei<len(values_) and xi<len(values_) and xi>ei and ei>=10
            if ok:ok=pd.Timestamp(values_[ei]).date()==event.date() and pd.Timestamp(values_[xi]).date()==event.date()
            valid.append(bool(ok))
        return frame.iloc[np.flatnonzero(np.asarray(valid,bool))].copy()

    raw={"US":pd.read_csv(DATA/"events_sec_us_v36_raw.csv",dtype={"ticker":str,"article_id":str},low_memory=False),
         "KR":pd.read_csv(DATA/"events_kind_real.csv",dtype={"ticker":str},low_memory=False)}
    outputs={};stages={}
    for market,directory,cap in (("US",PRICE/"US_V36",20),("KR",PRICE/"KR",15)):
        frame=outside(raw[market],market);outside_count=len(frame)
        local=frame.event_time_utc.dt.tz_convert(app.US_TZ if market=="US" else app.KR_TZ)
        start=pd.Timestamp("09:30").time() if market=="US" else pd.Timestamp("09:00").time()
        end=pd.Timestamp("15:27").time() if market=="US" else pd.Timestamp("14:57").time()
        frame=frame[(local.dt.dayofweek<5)&(local.dt.time>=start)&(local.dt.time<=end)&
                    frame.headline.fillna("").astype(str).str.len().ge(3)].copy()
        session_count=len(frame);frame=space(frame);spaced_count=len(frame);frame=covered(frame,market,directory)
        covered_count=len(frame);frame["selection_hash"]=frame.event_id.astype(str).map(
            lambda value:hashlib.sha256(f"V36|{market}|CAP|{value}".encode()).hexdigest())
        frame=(frame.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
               .head(cap).drop(columns="selection_hash").sort_values("event_time_utc"))
        frame["source"]=f"{market}_EXACT_V36_SEAL"
        frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True).astype(str)
        _atomic_csv(frame,DATA/f"events_{market.lower()}_exact_v36_seal.csv");outputs[market]=frame
        stages[market]={"outside_prior":outside_count,"in_session":session_count,
                        "spaced":spaced_count,"price_covered":covered_count,"sealed":len(frame)}
    overlap=sum(len(set(frame.event_id.astype(str))&prior_ids) for frame in outputs.values())
    article_overlap=sum(len((set(frame.article_id.fillna("").astype(str))-{""})&prior_articles)
                        for frame in outputs.values())
    url_overlap=sum(len((set(frame.url.fillna("").astype(str))-{""})&prior_urls)
                    for frame in outputs.values())
    _atomic_json({"selection_uses_labels":False,"prior_version":"V35","embargo_minutes":35,
        "split":"pure external SEAL; development and selection use opened V35 or earlier",
        "us_definition":"new health-name SEC issuers outside V35 universe; all forms; Yahoo 5-minute prices; max20/ticker",
        "kr_definition":"unused exact KIND disclosures with cached Daum prices; max15/ticker",
        "stages":stages,"seal_counts":{key:len(value) for key,value in outputs.items()},
        "prior_event_id_overlap":overlap,"prior_article_overlap":article_overlap,"prior_url_overlap":url_overlap,
    },DATA/"V36_SOURCE_STATUS.json")
    if overlap or article_overlap or url_overlap:raise RuntimeError("V36 source leakage")
    if len(outputs["US"])<250 or len(outputs["KR"])<100 or sum(map(len,outputs.values()))<500:
        raise RuntimeError(f"V36 source minimum failed: US={len(outputs['US'])}, KR={len(outputs['KR'])}")
    print(f"[V36 SOURCES] US seal={len(outputs['US'])}; KR seal={len(outputs['KR'])}",flush=True)
    return outputs["US"],outputs["KR"]


def build_v36_panel()->pd.DataFrame:
    """Label frozen V36 sources and append the now-opened V35 history."""
    raise RuntimeError(
        "legacy V36 label builder disabled: exact-contract source/freeze audit is required"
    )
    cfg=app.Config();paths=[DATA/"events_exact_v35.csv.gz",DATA/"events_us_exact_v36_seal.csv",
                            DATA/"events_kr_exact_v36_seal.csv"]
    missing=[path for path in paths if not path.exists()]
    if missing:raise FileNotFoundError(str(missing))
    frames=[pd.read_csv(path,dtype={"ticker":str},low_memory=False) for path in paths]
    active=app.EVENT_FILE;app.EVENT_FILE=DATA/"events_exact_v36.csv.gz"
    try:events=app.combine_events(frames,replace(cfg,material_only=False))
    finally:app.EVENT_FILE=active
    sources={"US_EXACT_V36_SEAL","KR_EXACT_V36_SEAL"}
    prior=pd.read_csv(DATA/"labeled_events_v35.csv.gz",dtype={"ticker":str},low_memory=False)
    prior["event_time_utc"]=pd.to_datetime(prior.event_time_utc,utc=True)
    label_cfg=replace(cfg,us_price_dir="price_data/US_V36")
    active=app.LABELED_FILE;app.LABELED_FILE=DATA/"labeled_events_v36_new.csv.gz"
    try:new=app.build_labeled(events[events.source.isin(sources)].copy(),label_cfg)
    finally:app.LABELED_FILE=active
    new["event_time_utc"]=pd.to_datetime(new.event_time_utc,utc=True)
    labeled=(pd.concat([prior,new],ignore_index=True,sort=False).drop_duplicates("event_id",keep="last")
             .sort_values("event_time_utc").reset_index(drop=True))
    labeled.to_csv(DATA/"labeled_events_v36.csv.gz",index=False,compression="gzip",encoding="utf-8")
    counts=new.groupby(["market","source"]).size().to_dict()
    _atomic_json({"new_labeled_counts":{f"{market}|{source}":int(count)
        for (market,source),count in counts.items()},"total":len(labeled)},DATA/"V36_LABEL_STATUS.json")
    print(f"[V36 LABELED] new={len(new)} counts={counts} total={len(labeled)}",flush=True)
    return labeled


def source_manifest() -> None:
    files = [
        DATA / "events_sec_real.csv",
        DATA / "events_us_news_real.csv",
        DATA / "events_kind_real.csv",
        DATA / "events_naver_news_real.csv",
        DATA / "events_sec_v7_early.csv",
        DATA / "events_naver_v7_h2_2025.csv",
        DATA / "events_sec_v8_middle.csv",
        DATA / "events_naver_v8_unused.csv",
        DATA / "events_sec_v9_2015_2017.csv",
        DATA / "events_naver_v9_reserved.csv",
        app.EVENT_FILE,
        app.LABELED_FILE,
    ]
    manifest = {
        "created_at": app.now_iso(),
        "period": {
            "us_start": DEFAULT_US_START,
            "kr_start": DEFAULT_KR_START,
            "end": DEFAULT_END,
        },
        "sources": {
            "US_events": "SEC EDGAR acceptanceDateTime + filing/exhibit text",
            "KR_events": "KIND todaydisclosure.do historical date query",
            "V6_US_seal_events": "SEC EDGAR acceptanceDateTime + filing/exhibit text",
            "V6_KR_seal_events": "Naver Finance ticker-news API exact datetime",
            "V7_US_seal_events": "SEC EDGAR, non-overlapping 2019-03 through 2021-01",
            "V7_KR_seal_events": "Naver Finance, non-overlapping 2025-H2 period",
            "V8_US_seal_events": "SEC EDGAR, unused 2021-02 through 2022-12",
            "V8_KR_seal_events": "Naver raw rows absent from V6/V7 and outside +/-60m",
            "US_prices": f"https://huggingface.co/datasets/{HF_REPO} ({HF_REVISION})",
            "KR_prices": "Daum Finance 5-minute charts",
        },
        "sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if path.exists()
        },
    }
    _atomic_json(manifest, DATA / "REAL_DATA_MANIFEST.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-kind", action="store_true")
    parser.add_argument("--collect-sec", action="store_true")
    parser.add_argument("--collect-sec-v7", action="store_true")
    parser.add_argument("--collect-sec-v8", action="store_true")
    parser.add_argument("--collect-us-news", action="store_true")
    parser.add_argument("--collect-naver-news", action="store_true")
    parser.add_argument("--collect-naver-v7", action="store_true")
    parser.add_argument("--collect-naver-v8", action="store_true")
    parser.add_argument("--collect-naver-v8-unused", action="store_true")
    parser.add_argument("--collect-prices", action="store_true")
    parser.add_argument("--collect-us-prices", action="store_true")
    parser.add_argument("--collect-sec-prices", action="store_true")
    parser.add_argument("--collect-sec-v7-prices", action="store_true")
    parser.add_argument("--collect-sec-v8-prices", action="store_true")
    parser.add_argument("--collect-kr-prices", action="store_true")
    parser.add_argument("--collect-naver-prices", action="store_true")
    parser.add_argument("--collect-naver-v7-prices", action="store_true")
    parser.add_argument("--collect-naver-v8-prices", action="store_true")
    parser.add_argument("--collect-v34-news", action="store_true")
    parser.add_argument("--collect-v34-us-prices", action="store_true")
    parser.add_argument("--prepare-v34-sources", action="store_true")
    parser.add_argument("--build-v34-panel", action="store_true")
    parser.add_argument("--v34-all", action="store_true")
    parser.add_argument("--collect-v35-sec", action="store_true")
    parser.add_argument("--collect-v35-naver", action="store_true")
    parser.add_argument("--collect-v35-us-prices", action="store_true")
    parser.add_argument("--collect-v35-kr-prices", action="store_true")
    parser.add_argument("--prepare-v35-sources", action="store_true")
    parser.add_argument("--build-v35-panel", action="store_true")
    parser.add_argument("--v35-all", action="store_true")
    parser.add_argument("--collect-v36-sec", action="store_true")
    parser.add_argument("--collect-v36-us-prices", action="store_true")
    parser.add_argument("--collect-v36-kr-1m-prices", action="store_true")
    parser.add_argument("--prepare-v36-sources", action="store_true")
    parser.add_argument("--build-v36-panel", action="store_true")
    parser.add_argument("--v36-all", action="store_true")
    parser.add_argument("--build-panel", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--us-start", default=DEFAULT_US_START)
    parser.add_argument("--kr-start", default=DEFAULT_KR_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.v36_all:
        raise RuntimeError(
            "--v36-all is intentionally disabled after the V36 contract audit; "
            "run explicit exact-contract collection/audit stages only"
        )

    if args.all or args.collect_kind:
        collect_kind(args.kr_start, args.end, workers=args.workers, force=args.force)
    if args.all or args.collect_sec:
        collect_sec(args.us_start, args.end, force=args.force)
    if args.collect_sec_v7:
        collect_sec(args.us_start, args.end, force=args.force,
                    output_name="events_sec_v7_early.csv",source_name="SEC_V7")
    if args.collect_sec_v8:
        collect_sec(args.us_start, args.end, force=args.force,
                    output_name="events_sec_v8_middle.csv",source_name="SEC_V8")
    if args.all or args.collect_us_news:
        collect_qiu_news()
    if args.all or args.collect_naver_news:
        collect_naver_news(args.kr_start, workers=args.workers, force=args.force)
    if args.collect_naver_v7:
        collect_naver_news(args.kr_start, workers=args.workers, force=args.force,
                           end=args.end,output_name="events_naver_v7_h2_2025.csv",
                           source_name="NAVER_NEWS_V7")
    if args.collect_naver_v8:
        collect_naver_news(args.kr_start, workers=args.workers, force=args.force,
                           end=args.end,output_name="events_naver_v8_early_2025.csv",
                           source_name="NAVER_NEWS_V8")
    if args.collect_naver_v8_unused:
        collect_naver_unused(args.end)
    if args.all or args.collect_prices:
        collect_us_prices(force=args.force)
        collect_kr_prices(args.kr_start, args.end, workers=min(args.workers, 6), force=args.force)
    else:
        if args.collect_us_prices:
            collect_us_prices(force=args.force)
        if args.collect_sec_prices:
            collect_us_prices(force=args.force, event_path=DATA / "events_sec_real.csv")
        if args.collect_sec_v7_prices:
            collect_us_prices(force=args.force, event_path=DATA / "events_sec_v7_early.csv")
        if args.collect_sec_v8_prices:
            collect_us_prices(force=args.force, event_path=DATA / "events_sec_v8_middle.csv")
        if args.collect_kr_prices:
            collect_kr_prices(args.kr_start, args.end, workers=min(args.workers, 6), force=args.force)
        if args.collect_naver_prices:
            collect_kr_prices(args.kr_start, args.end, workers=min(args.workers, 6),
                              force=args.force, event_path=DATA / "events_naver_news_real.csv")
        if args.collect_naver_v7_prices:
            collect_kr_prices(args.kr_start, args.end, workers=min(args.workers, 6),
                              force=args.force, event_path=DATA / "events_naver_v7_h2_2025.csv")
        if args.collect_naver_v8_prices:
            collect_kr_prices(args.kr_start, args.end, workers=min(args.workers, 6),
                              force=args.force, event_path=DATA / "events_naver_v8_unused.csv")
    if args.all or args.build_panel:
        build_panel()
        source_manifest()
    if args.v34_all or args.collect_v34_news:
        collect_v34_google_news()
    if args.v34_all or args.collect_v34_us_prices:
        collect_v34_us_prices(force=args.force)
    if args.v34_all or args.prepare_v34_sources:
        prepare_v34_sources()
    if args.v34_all or args.build_v34_panel:
        build_v34_panel()
    if args.v35_all or args.collect_v35_sec:
        collect_v35_sec()
    if args.v35_all or args.collect_v35_naver:
        collect_v35_naver()
    if args.v35_all or args.collect_v35_us_prices:
        collect_v35_us_prices(force=args.force)
    if args.v35_all or args.collect_v35_kr_prices:
        collect_v35_kr_prices(force=args.force)
    if args.v35_all or args.prepare_v35_sources:
        prepare_v35_sources()
    if args.v35_all or args.build_v35_panel:
        build_v35_panel()
    if args.v36_all or args.collect_v36_sec:
        collect_v36_sec()
    if args.v36_all or args.collect_v36_us_prices:
        collect_v36_us_prices(force=args.force)
    if args.v36_all or args.collect_v36_kr_1m_prices:
        collect_v36_kr_1m_prices(force=args.force)
    if args.v36_all or args.prepare_v36_sources:
        prepare_v36_sources()
    if args.v36_all or args.build_v36_panel:
        build_v36_panel()
    if not any((args.all, args.collect_kind, args.collect_sec, args.collect_sec_v7,args.collect_sec_v8,
                args.collect_us_news,args.collect_naver_news,args.collect_naver_v7,args.collect_naver_v8,
                args.collect_naver_v8_unused,
                args.collect_prices, args.collect_us_prices, args.collect_sec_prices,
                args.collect_sec_v7_prices,args.collect_sec_v8_prices,args.collect_kr_prices,args.collect_naver_prices,
                args.collect_naver_v7_prices,args.collect_naver_v8_prices,
                args.build_panel,args.collect_v34_news,args.collect_v34_us_prices,
                args.prepare_v34_sources,args.build_v34_panel,args.v34_all,
                args.collect_v35_sec,args.collect_v35_naver,args.collect_v35_us_prices,
                args.collect_v35_kr_prices,args.prepare_v35_sources,
                args.build_v35_panel,args.v35_all,args.collect_v36_sec,
                args.collect_v36_us_prices,args.collect_v36_kr_1m_prices,
                args.prepare_v36_sources,args.build_v36_panel,args.v36_all)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
