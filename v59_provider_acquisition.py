"""V59+ read-only Massive/KIS minute-bar acquisition.

Secrets are read only from the process environment, never persisted.  The KIS
client has a small method/path allowlist and intentionally contains no order or
account API implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

import bio_news_30m_v3 as app
import runtime_limits
import v36_exact_pipeline as exact
from audit_v36_static import biomedical_name_evidence
from provider_env_runtime import import_user_environment_if_missing


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data";STATE=ROOT/"state";RESEARCH=ROOT/"research";PRICE=ROOT/"price_data"
MASSIVE_CACHE=PRICE/"US_V59_MASSIVE_1M";KIS_CACHE=PRICE/"KR_V59_KIS_1M"
ALPACA_CACHE=PRICE/"alpaca_1m"
LOCAL_FINNHUB_CACHE=PRICE/"US"
LOCAL_FINNHUB_POLICY_PATH=DATA/"LOCAL_FINNHUB_PROVIDER_PARITY_POLICY_V1.json"
LOCAL_FINNHUB_PARITY_PATH=RESEARCH/"LOCAL_FINNHUB_PROVIDER_PARITY_SUMMARY.json"
LOCAL_FINNHUB_ARCHIVE_CACHE=PRICE/"local_finnhub_archive_1m"
LOCAL_FINNHUB_ARCHIVE_POLICY_PATH=DATA/"LOCAL_FINNHUB_ARCHIVE_PROVIDER_PARITY_POLICY_V1.json"
LOCAL_FINNHUB_ARCHIVE_PARITY_PATH=RESEARCH/"LOCAL_FINNHUB_ARCHIVE_PARITY_SUMMARY.json"
US_QUEUE=DATA/"PRICE_ACQUISITION_QUEUE_US.parquet"
KR_QUEUE=DATA/"PRICE_ACQUISITION_QUEUE_KR.parquet"
ROLE_PATH=DATA/"V59_DATA_ROLE_ASSIGNMENT.parquet"
US_PROVIDER_STATUS_PATH=DATA/"PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
ALPACA_PARITY_PATH=RESEARCH/"PROVIDER_PARITY_SUMMARY.json"
NEW_ROLE_POLICY_PATH=DATA/"V70_NEW_ROLE_POLICY.json"
ACQ_STATE=STATE/"ACQUISITION_STATE.json"
ACQ_LEDGER=RESEARCH/"DATA_ACQUISITION_LEDGER.csv"
MASSIVE_BASE="https://api.massive.com"
KIS_BASE="https://openapi.koreainvestment.com:9443"
KIS_TOKEN_PATH="/oauth2/tokenP"
KIS_MINUTE_PATH="/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
KIS_MINUTE_TR="FHKST03010230"
KIS_TODAY_PATH="/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
KIS_TODAY_TR="FHKST03010200"
KIS_TRADES_PATH="/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion"
KIS_TRADES_TR="FHPST01060000"
MAX_PRE_EVENT_MINUTES=125
PRE_EVENT_SAFETY_MINUTES=5
POST_EVENT_MINUTES=35
CONTRACT_VERSION="EXACT_T2_30M_V36"
ROLE_SEED="MARKET_BIO_V59_ROLE_V1"
NEW_ROLE_POLICY="V59_FRESH_RESERVE_RECOVERY_10_45_45"
# Frozen to the oldest empirically successful KIS probe.  A rolling ``now-365``
# cutoff would silently remove already frozen events on a later resume.
KR_HISTORY_START_UTC=pd.Timestamp("2025-08-20T00:00:00Z")
MASSIVE_DOC="https://massive.com/docs/rest/stocks/aggregates/custom-bars"
KIS_DOC=("https://github.com/koreainvestment/open-trading-api/blob/main/"
         "examples_llm/domestic_stock/inquire_time_dailychartprice/"
         "inquire_time_dailychartprice.py")


def now()->str:return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(value:Any)->bytes:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),
                      default=str).encode("utf-8")


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def atomic_json(value:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=str)
        handle.flush();os.fsync(handle.fileno())
    replace_with_retry(temporary,path)


def read_json_file(path:Path)->dict[str,Any]:
    if not path.exists():return {}
    try:
        value=json.loads(path.read_text(encoding="utf-8"));return value if isinstance(value,dict) else {}
    except Exception:return {}


def replace_with_retry(temporary:Path,path:Path,attempts:int=8)->None:
    """Tolerate brief Windows reader locks without losing the atomic checkpoint."""
    for attempt in range(attempts):
        try:
            os.replace(temporary,path);return
        except PermissionError:
            if attempt+1>=attempts:raise
            time.sleep(min(0.05*(2**attempt),1.0))


@contextmanager
def acquisition_ledger_lock(timeout_sec:float=120.0):
    """Serialize the cross-process ledger read/append/replace transaction."""
    import msvcrt
    lock_path=ACQ_LEDGER.with_suffix(ACQ_LEDGER.suffix+".lock")
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    handle=lock_path.open("a+b")
    if handle.tell()==0:
        handle.write(b"\0");handle.flush()
    deadline=time.monotonic()+timeout_sec;locked=False
    try:
        while not locked:
            try:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);locked=True
            except OSError:
                if time.monotonic()>=deadline:raise TimeoutError("ACQUISITION_LEDGER_LOCK_TIMEOUT")
                time.sleep(.1)
        yield
    finally:
        if locked:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        handle.close()


def atomic_parquet(frame:pd.DataFrame,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    frame.to_parquet(temporary,index=False);replace_with_retry(temporary,path)


def safe_error(response:requests.Response|None,exc:Exception|None=None)->str:
    if response is not None:
        try:
            payload=response.json()
            values=[payload.get(key) for key in ("status","error","message","msg_cd","msg1","rt_cd")]
            message=" | ".join(str(value)[:300] for value in values if value not in (None,""))
            return f"HTTP_{response.status_code}"+(f":{message}" if message else "")
        except Exception:return f"HTTP_{response.status_code}"
    return type(exc).__name__ if exc is not None else "UNKNOWN_ERROR"


def credential_status()->dict[str,str]:
    massive=os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    return {"MASSIVE_API_KEY":"PRESENT" if massive else "ABSENT",
            "KIS_APP_KEY":"PRESENT" if os.getenv("KIS_APP_KEY") else "ABSENT",
            "KIS_APP_SECRET":"PRESENT" if os.getenv("KIS_APP_SECRET") else "ABSENT"}


def massive_key()->str:
    value=os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not value:raise RuntimeError("CREDENTIAL_NOT_VISIBLE_TO_PROCESS:MASSIVE_API_KEY")
    return value


def kis_credentials()->tuple[str,str]:
    key=os.getenv("KIS_APP_KEY");secret=os.getenv("KIS_APP_SECRET")
    if not key or not secret:raise RuntimeError("CREDENTIAL_NOT_VISIBLE_TO_PROCESS:KIS")
    return key,secret


class MassiveClient:
    def __init__(self,min_interval:float=12.5):
        if min_interval<0:raise ValueError("MASSIVE_MIN_INTERVAL_MUST_BE_NONNEGATIVE")
        self.key=massive_key();self.session=requests.Session();self.last=0.;self.min_interval=float(min_interval)

    def get(self,ticker:str,start:pd.Timestamp,end:pd.Timestamp)->tuple[pd.DataFrame,dict[str,Any]]:
        wait=self.min_interval-(time.monotonic()-self.last)
        if wait>0:time.sleep(wait)
        start_ms=int(pd.Timestamp(start).timestamp()*1000);end_ms=int(pd.Timestamp(end).timestamp()*1000)
        path=f"/v2/aggs/ticker/{ticker}/range/1/minute/{start_ms}/{end_ms}"
        url=MASSIVE_BASE+path;params={"adjusted":"false","sort":"asc","limit":50000}
        response=None;rate_limited=False
        for attempt in range(5):
            try:
                response=self.session.get(url,params=params,headers={"Authorization":f"Bearer {self.key}"},timeout=60)
                self.last=time.monotonic()
                if response.status_code==429:
                    rate_limited=True
                    # A fast paid-plan probe may discover a free-plan limit.  One
                    # 429 permanently returns this process to the audited 5/min
                    # cadence, while Retry-After still governs the immediate wait.
                    self.min_interval=max(self.min_interval,12.5)
                    delay=float(response.headers.get("Retry-After") or min(120,15*2**attempt))+random.random()
                    time.sleep(delay);continue
                if response.status_code>=400:return pd.DataFrame(),{"ok":False,"error":safe_error(response,None),"rate_limited":rate_limited}
                payload=response.json();rows=payload.get("results") or []
                frame=pd.DataFrame(rows)
                if frame.empty:return frame,{"ok":True,"request_id":payload.get("request_id"),"rows":0,"rate_limited":rate_limited}
                required={"t","o","h","l","c","v"}
                if not required.issubset(frame.columns):return pd.DataFrame(),{"ok":False,"error":"MALFORMED_RESPONSE","rate_limited":rate_limited}
                frame=pd.DataFrame({"timestamp":pd.to_datetime(frame.t,unit="ms",utc=True),
                    "open":pd.to_numeric(frame.o,errors="coerce"),"high":pd.to_numeric(frame.h,errors="coerce"),
                    "low":pd.to_numeric(frame.l,errors="coerce"),"close":pd.to_numeric(frame.c,errors="coerce"),
                    "volume":pd.to_numeric(frame.v,errors="coerce"),
                    "transactions":pd.to_numeric(frame.get("n"),errors="coerce"),
                    "vwap":pd.to_numeric(frame.get("vw"),errors="coerce")})
                return frame.sort_values("timestamp").drop_duplicates("timestamp"),{
                    "ok":True,"request_id":payload.get("request_id"),"rows":len(frame),"rate_limited":rate_limited,
                    "effective_min_interval_sec":self.min_interval}
            except Exception as error:
                self.last=time.monotonic()
                if attempt==4:return pd.DataFrame(),{"ok":False,"error":safe_error(None,error),"rate_limited":False}
                time.sleep(min(60,2**attempt)+random.random())
        return pd.DataFrame(),{"ok":False,"error":"RETRY_EXHAUSTED","rate_limited":True}


class KISClient:
    """Minimal production quote client.  No trading/account paths are representable."""
    def __init__(self,min_interval:float=.5):
        self.key,self.secret=kis_credentials();self.session=requests.Session()
        self.token="";self.token_expiry=0.;self.last=0.;self.min_interval=min_interval

    def _request(self,method:str,path:str,**kwargs:Any)->requests.Response:
        allowed={("POST",KIS_TOKEN_PATH),("GET",KIS_MINUTE_PATH),("GET",KIS_TODAY_PATH),("GET",KIS_TRADES_PATH)}
        if (method.upper(),path) not in allowed:raise RuntimeError("KIS_READ_ONLY_ALLOWLIST_REJECTED")
        parsed=urlparse(KIS_BASE+path)
        if parsed.scheme!="https" or parsed.netloc!="openapi.koreainvestment.com:9443":
            raise RuntimeError("KIS_HOST_REJECTED")
        wait=self.min_interval-(time.monotonic()-self.last)
        if wait>0:time.sleep(wait)
        response=self.session.request(method,KIS_BASE+path,timeout=60,**kwargs);self.last=time.monotonic()
        return response

    def authenticate(self,force:bool=False)->dict[str,Any]:
        if self.token and time.time()<self.token_expiry-300 and not force:return {"ok":True,"reused":True}
        response=self._request("POST",KIS_TOKEN_PATH,json={"grant_type":"client_credentials",
            "appkey":self.key,"appsecret":self.secret})
        if response.status_code>=400:return {"ok":False,"error":safe_error(response,None)}
        payload=response.json();token=str(payload.get("access_token") or "")
        if not token:return {"ok":False,"error":"TOKEN_MISSING"}
        self.token=token;expires=int(payload.get("expires_in") or 86400);self.token_expiry=time.time()+expires
        return {"ok":True,"reused":False,"expires_in":expires}

    def page(self,ticker:str,date:str,hour:str)->tuple[pd.DataFrame,dict[str,Any]]:
        auth=self.authenticate()
        if not auth.get("ok"):return pd.DataFrame(),auth
        headers={"authorization":f"Bearer {self.token}","appkey":self.key,"appsecret":self.secret,
                 "tr_id":KIS_MINUTE_TR,"custtype":"P","tr_cont":""}
        params={"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":str(ticker).zfill(6),
                "FID_INPUT_HOUR_1":hour,"FID_INPUT_DATE_1":date,
                "FID_PW_DATA_INCU_YN":"Y","FID_FAKE_TICK_INCU_YN":"N"}
        response=self._request("GET",KIS_MINUTE_PATH,headers=headers,params=params)
        if response.status_code==401:
            auth=self.authenticate(force=True)
            if not auth.get("ok"):return pd.DataFrame(),auth
            headers["authorization"]=f"Bearer {self.token}"
            response=self._request("GET",KIS_MINUTE_PATH,headers=headers,params=params)
        try:payload=response.json()
        except Exception:return pd.DataFrame(),{"ok":False,"error":safe_error(response,None)}
        # Provider gateway throttle is transient.  Preserve the token and retry
        # the same read-only quote rather than re-authenticating.
        if str(payload.get("msg_cd") or "")=="EGW00201":
            time.sleep(61)
            response=self._request("GET",KIS_MINUTE_PATH,headers=headers,params=params)
            try:payload=response.json()
            except Exception:return pd.DataFrame(),{"ok":False,"error":safe_error(response,None)}
        if response.status_code>=400 or str(payload.get("rt_cd","0"))!="0":
            return pd.DataFrame(),{"ok":False,"error":safe_error(response,None),
                "msg_cd":str(payload.get("msg_cd") or "")}
        rows=payload.get("output2") or [];frame=pd.DataFrame(rows)
        return frame,{"ok":True,"rows":len(frame),"msg_cd":str(payload.get("msg_cd") or "")}

    def get_day(self,ticker:str,date:str,max_pages:int=8)->tuple[pd.DataFrame,dict[str,Any]]:
        frames=[];cursor="153000";requests_used=0;last_cursor=None
        for _ in range(max_pages):
            raw,status=self.page(ticker,date,cursor);requests_used+=1
            if not status.get("ok"):return pd.DataFrame(),{**status,"api_requests":requests_used}
            if raw.empty:break
            required={"stck_bsop_date","stck_cntg_hour","stck_oprc","stck_hgpr","stck_lwpr","stck_prpr","cntg_vol"}
            if not required.issubset(raw.columns):return pd.DataFrame(),{"ok":False,"error":"MALFORMED_RESPONSE","api_requests":requests_used}
            raw=raw[raw.stck_bsop_date.astype(str).eq(date)].copy()
            if raw.empty:break
            frames.append(raw);earliest=min(raw.stck_cntg_hour.astype(str).str.zfill(6))
            if earliest<="090000" or len(raw)<120 or earliest==last_cursor:break
            parsed=datetime.strptime(earliest,"%H%M%S")-timedelta(seconds=1)
            last_cursor=earliest;cursor=parsed.strftime("%H%M%S")
        if not frames:return pd.DataFrame(),{"ok":True,"rows":0,"api_requests":requests_used}
        raw=pd.concat(frames,ignore_index=True).drop_duplicates(["stck_bsop_date","stck_cntg_hour"])
        local=pd.to_datetime(raw.stck_bsop_date.astype(str)+raw.stck_cntg_hour.astype(str).str.zfill(6),
                             format="%Y%m%d%H%M%S",errors="coerce").dt.tz_localize(app.KR_TZ)
        frame=pd.DataFrame({"timestamp":local.dt.tz_convert("UTC"),
            "open":pd.to_numeric(raw.stck_oprc,errors="coerce"),"high":pd.to_numeric(raw.stck_hgpr,errors="coerce"),
            "low":pd.to_numeric(raw.stck_lwpr,errors="coerce"),"close":pd.to_numeric(raw.stck_prpr,errors="coerce"),
            "volume":pd.to_numeric(raw.cntg_vol,errors="coerce")}).dropna(subset=["timestamp"])
        return frame.sort_values("timestamp").drop_duplicates("timestamp"),{
            "ok":True,"rows":len(frame),"api_requests":requests_used}

    def read_quote(self,path:str,tr_id:str,ticker:str,hour:str)->tuple[pd.DataFrame,dict[str,Any]]:
        auth=self.authenticate()
        if not auth.get("ok"):return pd.DataFrame(),auth
        headers={"authorization":f"Bearer {self.token}","appkey":self.key,"appsecret":self.secret,
                 "tr_id":tr_id,"custtype":"P","tr_cont":""}
        params={"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":str(ticker).zfill(6),"FID_INPUT_HOUR_1":hour}
        if path==KIS_TODAY_PATH:params.update({"FID_PW_DATA_INCU_YN":"Y","FID_ETC_CLS_CODE":""})
        response=self._request("GET",path,headers=headers,params=params)
        try:payload=response.json()
        except Exception:return pd.DataFrame(),{"ok":False,"error":safe_error(response,None)}
        if response.status_code>=400 or str(payload.get("rt_cd","0"))!="0":return pd.DataFrame(),{"ok":False,"error":safe_error(response,None)}
        frame=pd.DataFrame(payload.get("output2") or [])
        return frame,{"ok":True,"rows":len(frame)}


def bar_qa(frame:pd.DataFrame,market:str)->dict[str,Any]:
    if frame.empty:return {"pass":False,"rows":0,"reason":"EMPTY"}
    ordered=frame.sort_values("timestamp");delta=ordered.timestamp.diff().dt.total_seconds().dropna()
    invalid_price=(ordered[["open","high","low","close"]]<=0).any(axis=1)
    bad_ohlc=((ordered.high<ordered[["open","close","low"]].max(axis=1))|
              (ordered.low>ordered[["open","close","high"]].min(axis=1)))
    duplicates=int(ordered.timestamp.duplicated().sum());nonmono=int((delta<=0).sum())
    local=ordered.timestamp.dt.tz_convert(app.US_TZ if market=="US" else app.KR_TZ)
    session_start=pd.Timestamp("09:30").time() if market=="US" else pd.Timestamp("09:00").time()
    session_end=pd.Timestamp("16:00").time() if market=="US" else pd.Timestamp("15:30").time()
    regular=int(((local.dt.time>=session_start)&(local.dt.time<=session_end)).sum())
    result={"rows":len(ordered),"duplicates":duplicates,"non_monotonic":nonmono,
            "invalid_price":int(invalid_price.sum()),"bad_ohlc":int(bad_ohlc.sum()),
            "negative_volume":int((ordered.volume<0).sum()),"regular_session_rows":regular,
            "median_cadence_sec":float(delta.median()) if len(delta) else None,
            "missing_minute_gaps":int((delta>60).sum())}
    result["pass"]=not any(result[key] for key in ("duplicates","non_monotonic","invalid_price","bad_ohlc","negative_volume"))
    return result


def preflight(massive_client:MassiveClient|None=None,kis_client:KISClient|None=None)->dict[str,Any]:
    report={"timestamp":now(),"credentials":credential_status(),"read_only":True}
    massive_semantics={"provider":"Massive","official_documentation":MASSIVE_DOC,
        "timestamp_field":"results[].t","timestamp_semantics":"BAR_OPEN",
        "official_exact_eligible":True,"cadence_seconds":60,"adjusted":False,
        "missing_bar_policy":"NO_FILL_NO_INTERPOLATION","timezone_parse":"UNIX_MS_UTC"}
    kis_semantics={"provider":"KIS","official_documentation":KIS_DOC,
        "timestamp_fields":["stck_bsop_date","stck_cntg_hour"],
        "official_wording":"stock execution time","timestamp_semantics":"UNVERIFIED",
        "official_exact_eligible":False,"promotion_requirement":"trade-to-minute OHLCV alignment audit",
        "timezone_parse":"ASIA_SEOUL_TO_UTC","missing_bar_policy":"NO_FILL_NO_INTERPOLATION"}
    try:
        client=massive_client or MassiveClient(min_interval=0)
        start=pd.Timestamp("2026-08-25T13:30:00Z");bars,status=client.get("AAPL",start,start+pd.Timedelta(minutes=10))
        report["massive"]={**status,"qa":bar_qa(bars,"US"),"sample_interval_utc":[str(start),str(start+pd.Timedelta(minutes=10))],
            "timestamp_min":str(bars.timestamp.min()) if not bars.empty else None,
            "timestamp_max":str(bars.timestamp.max()) if not bars.empty else None}
    except Exception as error:report["massive"]={"ok":False,"error":safe_error(None,error)}
    try:
        client=kis_client or KISClient();auth=client.authenticate();bars,status=client.get_day("005930","20260825",max_pages=1) if auth.get("ok") else (pd.DataFrame(),auth)
        report["kis"]={"token_acquisition":bool(auth.get("ok")),**status,"qa":bar_qa(bars,"KR"),
            "sample_ticker":"005930","sample_date":"20260825",
            "timestamp_min":str(bars.timestamp.min()) if not bars.empty else None,
            "timestamp_max":str(bars.timestamp.max()) if not bars.empty else None}
    except Exception as error:report["kis"]={"ok":False,"error":safe_error(None,error)}
    atomic_json(report,DATA/"API_PREFLIGHT_STATUS.json")
    atomic_json(massive_semantics,DATA/"BAR_SEMANTICS_MASSIVE.json")
    atomic_json(kis_semantics,DATA/"BAR_SEMANTICS_KIS.json")
    return report


def _us_candidates()->pd.DataFrame:
    prior=exact.normalized_events(DATA/"events_exact_v35.csv.gz");frames=[]
    for path in sorted(DATA.glob("events*sec*.csv")):
        try:
            frame=exact.normalized_events(path);frame["raw_file"]=path.name;frames.append(frame)
        except Exception:continue
    raw=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    outside,_=exact.outside_prior(raw,prior);outside=outside[outside.form.fillna("").astype(str).isin(set(app.Config().sec_forms))]
    outside=exact.regular_session(outside,"US")
    # V221 current-provider fresh collections are not required to have a
    # legacy Hugging Face ticker file. Massive/Alpaca supply their exact bars;
    # all older sources retain the original pinned-file availability guard.
    v221_current_provider=outside.raw_file.astype(str).str.startswith(
        ("events_sec_v221_fresh_", "events_sec_v221_expanded_fresh_", "events_sec_v221_sic_fresh_",
         "events_sec_v221_legacy_gap_")
    )
    legacy_local=outside.ticker.map(lambda value:(PRICE/"US"/f"{value}.parquet").exists())
    outside=outside[v221_current_provider|legacy_local].copy()
    outside["cik"]=outside.event_id.map(exact.cik_from_event_id)
    payload=json.loads((DATA/"V36_SEC_ISSUER_METADATA.json").read_text(encoding="utf-8"))
    metadata={str(int(row["cik"])):row for row in payload.get("issuers",[]) if str(row.get("cik","")).isdigit()}
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str});approved=set(seed.loc[seed.market.eq("US"),"ticker"].astype(str))
    def health_indices(frame:pd.DataFrame)->list[int]:
      keep=[]
      for row in frame.itertuples():
        issuer=metadata.get(str(row.cik),{});evidence=biomedical_name_evidence(row.company)
        raw_source_sic=getattr(row,"sic","")
        try:source_sic=str(int(float(raw_source_sic))) if pd.notna(raw_source_sic) else ""
        except (TypeError,ValueError):source_sic=str(raw_source_sic or "")
        if (str(row.ticker) in approved or source_sic in exact.APPROVED_HEALTHCARE_SIC or
                str(issuer.get("sic","") or "") in exact.APPROVED_HEALTHCARE_SIC or
                evidence["name_rule_pass"]):keep.append(row.Index)
      return keep
    historical=outside.loc[health_indices(outside)].copy()
    fresh_frames=[]
    for path in sorted((DATA/"untouched_pool").glob("sec_*.csv.gz")):
        try:fresh_frames.append(exact.normalized_events(path))
        except Exception:continue
    if fresh_frames:
        fresh=pd.concat(fresh_frames,ignore_index=True,sort=False).drop_duplicates("event_id")
        fresh,_=exact.outside_prior(fresh,prior);fresh=fresh[fresh.form.fillna("").astype(str).isin(set(app.Config().sec_forms))]
        fresh=exact.regular_session(fresh,"US");fresh["cik"]=fresh.event_id.map(exact.cik_from_event_id)
        # The immutable fresh snapshot was already selected by
        # audited_prospective_universe() using this same pinned seed/SIC/name
        # policy.  Re-fetching every missing CIK here is redundant and makes a
        # local queue rebuild depend on hundreds of mutable network calls.
        fresh=fresh.loc[health_indices(fresh)].copy()
        historical=pd.concat([historical,fresh],ignore_index=True,sort=False).drop_duplicates("event_id")
    news_frames=[]
    for path in sorted((DATA/"untouched_pool").glob("gnews_us_20*.csv.gz")):
        try:news_frames.append(exact.normalized_events(path))
        except Exception:continue
    if news_frames:
        news=pd.concat(news_frames,ignore_index=True,sort=False).drop_duplicates("event_id")
        news,_=exact.outside_prior(news,prior);news=exact.regular_session(news,"US")
        # V33 is the pinned label-blind US healthcare RSS universe.  Keep only
        # tickers whose event-time price history authority already exists.
        approved_news=set(pd.read_csv(DATA/"universe_v33_us_rss.csv",dtype={"ticker":str}).ticker.astype(str))
        news=news[news.ticker.astype(str).isin(approved_news)]
        news=news[news.ticker.map(lambda value:(PRICE/"US"/f"{value}.parquet").exists())].copy()
        historical=pd.concat([historical,news],ignore_index=True,sort=False).drop_duplicates("event_id")
    return exact.space_events(historical).sort_values(["ticker","event_time_utc","event_id"])


def _kr_candidates()->pd.DataFrame:
    prior=exact.normalized_events(DATA/"events_exact_v35.csv.gz")
    # Naver ticker-news API timestamps are minute-floored, not exact publication
    # seconds.  Raw Naver snapshots are therefore discovery-only and must not
    # enter the strict queue until the article page timestamp is independently
    # verified into a ``naver_verified_kr_*`` snapshot.
    frames=[exact.normalized_events(DATA/"events_kind_real.csv")]
    google=DATA/"events_google_kr_v34_raw.csv"
    if google.exists():frames.append(exact.normalized_events(google))
    for path in sorted((DATA/"untouched_pool").glob("gnews_kr_20*.csv.gz")):
        try:frames.append(exact.normalized_events(path))
        except Exception:continue
    for path in sorted((DATA/"untouched_pool").glob("naver_verified_kr_*.csv.gz")):
        try:frames.append(exact.normalized_events(path))
        except Exception:continue
    for path in sorted((DATA/"untouched_pool").glob("kind_kr_*.csv.gz")):
        try:frames.append(exact.normalized_events(path))
        except Exception:continue
    raw=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    outside,_=exact.outside_prior(raw,prior)
    # V29's six Naver healthcare industries are a prior label-blind audited
    # universe (413 rows), broader than the small deployment seed.  This is the
    # authority for the requested KR historical re-audit.
    universe=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    approved=set(universe.ticker.astype(str).str.zfill(6))
    outside=outside[outside.ticker.isin(approved)&(outside.event_time_utc>=KR_HISTORY_START_UTC)].copy()
    return exact.space_events(exact.regular_session(outside,"KR")).sort_values(["ticker","event_time_utc","event_id"])


def _queue(frame:pd.DataFrame,market:str,provider:str)->pd.DataFrame:
    timezone_name=app.US_TZ if market=="US" else app.KR_TZ
    event=pd.to_datetime(frame.event_time_utc,utc=True)
    source_family=np.where(frame.source.astype(str).str.contains("SEC"),"US_SEC",
                  np.where(frame.source.astype(str).str.contains("KIND"),"KR_KIND",
                           f"{market}_NEWS"))
    return pd.DataFrame({"event_id":frame.event_id.astype(str),"ticker":frame.ticker.astype(str),
        "issuer":frame.company.astype(str),"event_time_utc":event,"event_time_local":event.dt.tz_convert(timezone_name),
        "required_start_utc":event-pd.Timedelta(minutes=MAX_PRE_EVENT_MINUTES+PRE_EVENT_SAFETY_MINUTES),
        "required_end_utc":event+pd.Timedelta(minutes=POST_EVENT_MINUTES),"source_family":source_family,
        "source_event_id":frame.event_id.astype(str),"status":"PENDING","attempt_count":0,
        "last_error":"","provider":provider,"selection_uses_labels":False})


def build_queues()->dict[str,Any]:
    us=_queue(_us_candidates(),"US","MASSIVE");kr=_queue(_kr_candidates(),"KR","KIS")
    for frame,path in ((us,US_QUEUE),(kr,KR_QUEUE)):
        if path.exists():
            old=pd.read_parquet(path).set_index("event_id")
            common=frame.event_id.astype(str).isin(old.index.astype(str))
            for column in ("status","attempt_count","last_error"):
                lookup=old[column].to_dict();frame.loc[common,column]=frame.loc[common,"event_id"].map(lookup)
    atomic_parquet(us,US_QUEUE);atomic_parquet(kr,KR_QUEUE)
    audit={"timestamp":now(),"selection_uses_labels":False,"outcome_columns_loaded":False,
        "max_feature_clock_window_minutes":MAX_PRE_EVENT_MINUTES,"safety_margin_minutes":PRE_EVENT_SAFETY_MINUTES,
        "post_event_minutes":POST_EVENT_MINUTES,"US":{"events":len(us),"ticker_dates":int(us.assign(date=us.event_time_local.dt.date).groupby(["ticker","date"]).ngroups)},
        "KR":{"events":len(kr),"ticker_dates":int(kr.assign(date=kr.event_time_local.dt.date).groupby(["ticker","date"]).ngroups)}}
    atomic_json(audit,DATA/"PRICE_ACQUISITION_QUEUE_AUDIT.json")
    atomic_json({"timestamp":now(),"phase":"BUILD_QUEUE","queues":audit,"last_progress_at":now()},ACQ_STATE)
    return audit


def append_ledger(rows:list[dict[str,Any]])->None:
    columns=["provider","market","ticker_date","events_requested","api_requests","bars_downloaded",
             "exact_eligible","failed","rate_limited","cached","timestamp"]
    with acquisition_ledger_lock():
        previous=pd.read_csv(ACQ_LEDGER) if ACQ_LEDGER.exists() else pd.DataFrame(columns=columns)
        output=pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
        temporary=ACQ_LEDGER.with_name(ACQ_LEDGER.name+f".tmp.{os.getpid()}")
        output.to_csv(temporary,index=False);replace_with_retry(temporary,ACQ_LEDGER)


def _cache_path(base:Path,ticker:str,date:Any)->Path:
    value=pd.Timestamp(date);return base/f"{value.year:04d}"/str(ticker)/f"{value.date().isoformat()}.parquet"


def authorized_alpaca_feed()->str|None:
    report=read_json_file(ALPACA_PARITY_PATH);result=report.get("result",{})
    if (report.get("backfill_authorized") is True and result.get("status")=="PASS" and
            str(result.get("feed","")).upper() in {"SIP","IEX"} and
            report.get("sealed_role_labels_opened") is False):
        return str(result["feed"]).upper()
    return None


def local_finnhub_authorized()->bool:
    """Authorize the pinned local 1m store only under its frozen parity result."""
    if not LOCAL_FINNHUB_POLICY_PATH.exists() or not LOCAL_FINNHUB_PARITY_PATH.exists():return False
    report=read_json_file(LOCAL_FINNHUB_PARITY_PATH)
    if (report.get("backfill_authorized") is not True or
            (report.get("result") or {}).get("status")!="PASS" or
            report.get("sealed_role_labels_opened") is not False):return False
    return report.get("policy_sha256")==sha256(LOCAL_FINNHUB_POLICY_PATH)


def local_finnhub_archive_authorized()->bool:
    if not LOCAL_FINNHUB_ARCHIVE_POLICY_PATH.exists() or not LOCAL_FINNHUB_ARCHIVE_PARITY_PATH.exists():return False
    report=read_json_file(LOCAL_FINNHUB_ARCHIVE_PARITY_PATH)
    if (report.get("backfill_authorized") is not True or
            (report.get("result") or {}).get("status")!="PASS" or
            report.get("sealed_role_labels_opened") is not False):return False
    return report.get("policy_sha256")==sha256(LOCAL_FINNHUB_ARCHIVE_POLICY_PATH)


def load_us_provider_status()->pd.DataFrame:
    if not US_PROVIDER_STATUS_PATH.exists():return pd.DataFrame()
    frame=pd.read_parquet(US_PROVIDER_STATUS_PATH)
    if frame.event_id.astype(str).duplicated().any():raise RuntimeError("DUPLICATE_US_PROVIDER_STATUS_EVENT")
    return frame.set_index(frame.event_id.astype(str),drop=False)


def resolve_us_price_cache(event_id:str,ticker:str,date:Any,status:pd.DataFrame)->tuple[Path,str,str,str]|None:
    candidates=resolve_us_price_cache_candidates(event_id,ticker,date,status)
    return candidates[0] if candidates else None


def resolve_us_price_cache_candidates(
    event_id:str,ticker:str,date:Any,status:pd.DataFrame
)->list[tuple[Path,str,str,str]]:
    """Return authorized caches in frozen P1 Massive, P2 Alpaca, P3 local order."""
    candidates=[]
    massive=_cache_path(MASSIVE_CACHE,ticker,date)
    if massive.exists():candidates.append((massive,"MASSIVE","SIP",str(ticker)))
    feed=authorized_alpaca_feed()
    if status.empty or str(event_id) not in status.index:return candidates
    row=status.loc[str(event_id)];selected=str(row.get("selected_price_provider","")).upper()
    provider_symbol_value=row.get("provider_symbol","")
    provider_symbol=(str(ticker) if pd.isna(provider_symbol_value) or not str(provider_symbol_value).strip()
                     else str(provider_symbol_value))
    exact=(bool(row.get("provider_parity_eligible")) is True and
           str(row.get("exact_contract_status"))=="EXACT_TIMING_ELIGIBLE")
    if selected=="MASSIVE_SIP" and exact and provider_symbol!=str(ticker):
        resolved_massive=_cache_path(MASSIVE_CACHE,provider_symbol,date)
        if resolved_massive.exists():
            candidates.insert(0,(resolved_massive,"MASSIVE","SIP",provider_symbol))
    if (feed is not None and selected==f"ALPACA_{feed}" and exact):
        path=_cache_path(ALPACA_CACHE/feed.lower(),provider_symbol,date)
        if path.exists():candidates.append((path,"ALPACA",feed,provider_symbol))
    if selected=="LOCAL_FINNHUB_1M" and exact and local_finnhub_authorized():
        path=LOCAL_FINNHUB_CACHE/f"{provider_symbol}.parquet"
        if path.exists():candidates.append((path,"LOCAL_FINNHUB","FINNHUB_1M_DATASET",provider_symbol))
    if selected=="LOCAL_FINNHUB_ARCHIVE_1M" and exact and local_finnhub_archive_authorized():
        path=_cache_path(LOCAL_FINNHUB_ARCHIVE_CACHE,provider_symbol,date)
        if path.exists():candidates.append((path,"LOCAL_FINNHUB_ARCHIVE","FINNHUB_1M_DATASET",provider_symbol))
    return candidates


def massive_verified_entitlement_anchor()->pd.Timestamp:
    audit=read_json_file(DATA/"MASSIVE_COVERAGE_AUDIT.json")
    value=audit.get("oldest_successful_anchor")
    probes=audit.get("probes") or []
    if not value or not any(row.get("ok") and row.get("rows") for row in probes):
        raise RuntimeError("MASSIVE_VERIFIED_ENTITLEMENT_ANCHOR_MISSING_RUN_PROBE_FIRST")
    anchor=pd.Timestamp(str(value))
    if anchor.tzinfo is None:anchor=anchor.tz_localize("UTC")
    else:anchor=anchor.tz_convert("UTC")
    return anchor.normalize()


def defer_outside_massive_entitlement(queue:pd.DataFrame,anchor:pd.Timestamp)->tuple[pd.DataFrame,int]:
    result=queue.copy();local=pd.to_datetime(result.event_time_local,utc=True).dt.tz_convert(app.US_TZ)
    trade_dates=local.dt.date.astype(str);deferred=0
    for (ticker,date),indices in result.groupby([result.ticker.astype(str),trade_dates]).groups.items():
        index=list(indices);pending=result.loc[index,"status"].astype(str).eq("PENDING")
        if not pending.any() or pd.Timestamp(date,tz="UTC")>=anchor:continue
        if _cache_path(MASSIVE_CACHE,str(ticker),date).exists():continue
        selected=[idx for idx,flag in zip(index,pending) if flag]
        result.loc[selected,"status"]="DEFERRED_OUTSIDE_VERIFIED_ENTITLEMENT_ANCHOR"
        result.loc[selected,"last_error"]=(
            "DEFERRED_NO_API_CALL:OLDER_THAN_OLDEST_SUCCESSFUL_PROBE:"+anchor.date().isoformat()
        )
        deferred+=len(selected)
    return result,deferred


def massive_pending_group_schedule(queue:pd.DataFrame)->pd.DataFrame:
    """Order outcome-blind work by the frozen epoch bottleneck.

    ``US_SEC`` is the only US family authorized to satisfy the predeclared
    DEV_EXTENSION epoch.  Frozen DEV groups within that family therefore come
    before reserved-role SEC groups and unrelated US_NEWS groups.  No return,
    label, or model score participates in this ordering.
    """
    return (queue[["ticker","trade_date","status","_data_role","source_family"]]
            .assign(is_pending=lambda frame:frame.status.eq("PENDING"),
                    is_required_source=lambda frame:frame.source_family.eq("US_SEC"),
                    is_dev=lambda frame:frame._data_role.eq("DEV_EXTENSION"))
            .groupby(["ticker","trade_date"],as_index=False)
            .agg(has_pending=("is_pending","max"),
                 has_required_source=("is_required_source","max"),
                 has_dev=("is_dev","max"))
            .loc[lambda frame:frame.has_pending]
            .sort_values(["has_required_source","has_dev","trade_date","ticker"],
                         ascending=[False,False,False,True]))


def fetch_massive(max_requests:int|None=None,client:MassiveClient|None=None)->dict[str,Any]:
    queue=pd.read_parquet(US_QUEUE);queue["event_time_local"]=pd.to_datetime(queue.event_time_local,utc=True).dt.tz_convert(app.US_TZ)
    queue["trade_date"]=queue.event_time_local.dt.date.astype(str);client=client or MassiveClient();ledger=[];used=0
    roles=pd.read_parquet(ROLE_PATH,use_nullable_dtypes=False)[["event_id","role"]]
    if roles.event_id.duplicated().any():raise RuntimeError("DUPLICATE_EVENT_ID_IN_FROZEN_ROLE_ASSIGNMENT")
    role_map=roles.set_index("event_id")["role"]
    queue["_data_role"]=queue.event_id.map(role_map)
    if queue._data_role.isna().any():raise RuntimeError("QUEUE_EVENT_MISSING_FROZEN_DATA_ROLE")
    anchor=massive_verified_entitlement_anchor()
    queue_without_helpers=queue.drop(columns=["trade_date","_data_role"])
    queue_without_helpers,deferred=defer_outside_massive_entitlement(queue_without_helpers,anchor)
    queue["status"]=queue_without_helpers["status"]
    queue["last_error"]=queue_without_helpers["last_error"]
    if deferred:atomic_parquet(queue_without_helpers,US_QUEUE)
    # Exploit the provider's accessible recent range before spending calls on
    # known out-of-entitlement history.  Within new PENDING work, prioritize
    # the already-frozen DEV_EXTENSION role so research support can be reached
    # without changing assignments or inspecting outcomes.
    group_keys=massive_pending_group_schedule(queue)
    for ticker,date,_,_,_ in group_keys.itertuples(index=False,name=None):
        group=queue[(queue.ticker.astype(str)==str(ticker))&queue.trade_date.eq(str(date))]
        if max_requests is not None and used>=max_requests:break
        path=_cache_path(MASSIVE_CACHE,ticker,date);cached=path.exists();status={"ok":True};bars=pd.DataFrame()
        if cached:
            try:bars=pd.read_parquet(path)
            except Exception:cached=False
        if not cached:
            start=group.required_start_utc.min();end=group.required_end_utc.max()
            bars,status=client.get(ticker,pd.Timestamp(start),pd.Timestamp(end));used+=1
            if status.get("ok") and not bars.empty and bar_qa(bars,"US")["pass"]:
                path.parent.mkdir(parents=True,exist_ok=True);atomic_parquet(bars,path)
        indices=group.index;ok=bool(status.get("ok") and not bars.empty and bar_qa(bars,"US")["pass"])
        queue.loc[indices,"attempt_count"]=queue.loc[indices,"attempt_count"].astype(int)+(0 if cached else 1)
        queue.loc[indices,"status"]="CACHED_VALID" if cached and ok else ("FETCHED_VALID" if ok else "PROVIDER_FAILED")
        queue.loc[indices,"last_error"]="" if ok else str(status.get("error","EMPTY_OR_QA_FAILED"))[:300]
        ledger.append({"provider":"MASSIVE","market":"US","ticker_date":f"{ticker}/{date}",
            "events_requested":len(group),"api_requests":0 if cached else 1,"bars_downloaded":len(bars),
            "exact_eligible":0,"failed":0 if ok else 1,"rate_limited":int(bool(status.get("rate_limited"))),
            "cached":int(cached),"timestamp":now()})
        atomic_parquet(queue.drop(columns=["trade_date","_data_role"]),US_QUEUE)
        atomic_json({"timestamp":now(),"phase":"FETCH_PRICE","provider":"MASSIVE","api_requests":used,
            "last_ticker_date":f"{ticker}/{date}","pending":int((queue.status=="PENDING").sum())},ACQ_STATE)
    append_ledger(ledger)
    counts=Counter(queue.status.astype(str));audit={"timestamp":now(),"candidate_total":len(queue),
        "status_counts":dict(counts),"api_requests_this_run":used,"successful_dates":sum(row["failed"]==0 for row in ledger),
        "missing_dates":sum(row["failed"] for row in ledger),"rate_limited":sum(row["rate_limited"] for row in ledger),
        "exact_contract_eligible":0,"history_entitlement":"EMPIRICAL_QUEUE_IN_PROGRESS",
        "oldest_successful_probe_anchor":anchor.date().isoformat(),"deferred_without_api_calls":deferred,
        "scheduling_policy":"VERIFIED_ENTITLEMENT_THEN_US_SEC_THEN_FROZEN_DEV_THEN_DATE_DESC_V4",
        "scheduling_uses_labels":False}
    existing=read_json_file(DATA/"MASSIVE_COVERAGE_AUDIT.json")
    if existing.get("probes"):audit["probes"]=existing["probes"];audit["oldest_successful_anchor"]=existing.get("oldest_successful_anchor")
    atomic_json(audit,DATA/"MASSIVE_COVERAGE_AUDIT.json");return audit


def probe_massive_coverage(client:MassiveClient|None=None)->dict[str,Any]:
    client=client or MassiveClient();probes=[]
    for date in ("2026-08-25","2024-09-03","2021-09-01","2016-09-01","2011-09-01"):
        start=pd.Timestamp(f"{date}T13:30:00Z");bars,status=client.get("AAPL",start,start+pd.Timedelta(minutes=10))
        probes.append({"ticker":"AAPL","date":date,"ok":bool(status.get("ok")),"rows":len(bars),
                       "error":status.get("error"),"qa":bar_qa(bars,"US")})
    accessible=[row["date"] for row in probes if row["ok"] and row["rows"]]
    audit={"timestamp":now(),"probes":probes,"oldest_successful_anchor":min(accessible) if accessible else None,
           "candidate_total":len(pd.read_parquet(US_QUEUE)) if US_QUEUE.exists() else None,
           "exact_contract_eligible":0}
    atomic_json(audit,DATA/"MASSIVE_COVERAGE_AUDIT.json");return audit


def probe_kis_coverage(client:KISClient|None=None)->dict[str,Any]:
    client=client or KISClient();probes=[]
    for date in ("20260727","20260227","20250901","20250820"):
        bars,status=client.get_day("005930",date,max_pages=1)
        probes.append({"ticker":"005930","date":date,"ok":bool(status.get("ok")),"rows":len(bars),
                       "error":status.get("error"),"qa":bar_qa(bars,"KR")})
    audit={"timestamp":now(),"official_max_retention":"up to one year","probes":probes,
        "bar_semantics_verified":False,"official_exact_eligible":False}
    atomic_json(audit,DATA/"KIS_HISTORICAL_COVERAGE_AUDIT.json");return audit


def audit_kis_semantics(client:KISClient|None=None)->dict[str,Any]:
    client=client or KISClient();rows=[];same_only=next_only=both=neither=0
    for ticker in ("005930","000660","035420"):
        for hour in ("100000","120000","140000"):
            bars,bar_status=client.read_quote(KIS_TODAY_PATH,KIS_TODAY_TR,ticker,hour)
            trades,trade_status=client.read_quote(KIS_TRADES_PATH,KIS_TRADES_TR,ticker,hour)
            if not bar_status.get("ok") or not trade_status.get("ok") or bars.empty or trades.empty:
                rows.append({"ticker":ticker,"hour":hour,"bars":len(bars),"trades":len(trades),
                             "bar_error":bar_status.get("error"),"trade_error":trade_status.get("error")});continue
            bar_map={str(row.stck_cntg_hour).zfill(6):row for row in bars.itertuples()}
            local_counts=Counter()
            for trade in trades.itertuples():
                stamp=str(getattr(trade,"stck_cntg_hour","")).zfill(6);price=float(getattr(trade,"stck_prpr",getattr(trade,"stck_pbpr",np.nan)))
                if len(stamp)!=6 or not np.isfinite(price):continue
                minute=stamp[:4]+"00";next_minute=(datetime.strptime(minute,"%H%M%S")+timedelta(minutes=1)).strftime("%H%M%S")
                def within(key:str)->bool:
                    bar=bar_map.get(key)
                    if bar is None:return False
                    low=float(getattr(bar,"stck_lwpr",np.nan));high=float(getattr(bar,"stck_hgpr",np.nan))
                    return np.isfinite(low) and np.isfinite(high) and low<=price<=high
                same=within(minute);following=within(next_minute)
                category="both" if same and following else ("same_only" if same else ("next_only" if following else "neither"))
                local_counts[category]+=1
            same_only+=local_counts["same_only"];next_only+=local_counts["next_only"]
            both+=local_counts["both"];neither+=local_counts["neither"]
            rows.append({"ticker":ticker,"hour":hour,"bars":len(bars),"trades":len(trades),"alignment":dict(local_counts)})
    discriminating=same_only+next_only
    passed=discriminating>=10 and same_only/max(1,discriminating)>=.95
    report={"provider":"KIS","timestamp":now(),"official_documentation":KIS_DOC,
        "official_shape_note":"forming minute is first output row; on its first trade the prior minute moves to the second row",
        "same_minute_only":same_only,"next_minute_only":next_only,"both":both,"neither":neither,
        "discriminating":discriminating,"empirical_alignment_pass":passed,
        "timestamp_semantics":"BAR_OPEN" if passed else "UNVERIFIED","official_exact_eligible":passed,"probes":rows}
    atomic_json(report,DATA/"BAR_SEMANTICS_KIS.json");return report


def fetch_kis(max_requests:int|None=None,client:KISClient|None=None)->dict[str,Any]:
    queue=pd.read_parquet(KR_QUEUE);queue["event_time_local"]=pd.to_datetime(queue.event_time_local,utc=True).dt.tz_convert(app.KR_TZ)
    semantics=read_json_file(DATA/"BAR_SEMANTICS_KIS.json");verified=bool(semantics.get("official_exact_eligible"))
    queue["trade_date"]=queue.event_time_local.dt.date.astype(str);client=client or KISClient();ledger=[];used=0
    roles=pd.read_parquet(ROLE_PATH,use_nullable_dtypes=False)[["event_id","role"]]
    if roles.event_id.duplicated().any():raise RuntimeError("DUPLICATE_EVENT_ID_IN_FROZEN_ROLE_ASSIGNMENT")
    role_map=roles.set_index("event_id")["role"]
    queue["_data_role"]=queue.event_id.map(role_map)
    if queue._data_role.isna().any():raise RuntimeError("QUEUE_EVENT_MISSING_FROZEN_DATA_ROLE")
    # KIS retention is short and the new KR source pool is much larger than the
    # old queue.  Spend calls on already-frozen DEV_EXTENSION rows first, with
    # no label/outcome access, so the independent new-data epoch can reach its
    # predeclared support floor before sealed roles consume the quota.
    group_keys=(queue[["ticker","trade_date","status","_data_role"]]
                .assign(is_pending=lambda frame:frame.status.eq("PENDING"),
                        is_dev=lambda frame:frame._data_role.eq("DEV_EXTENSION"))
                .groupby(["ticker","trade_date"],as_index=False)
                .agg(has_pending=("is_pending","max"),has_dev=("is_dev","max"))
                .sort_values(["has_pending","has_dev","trade_date","ticker"],
                             ascending=[False,False,False,True]))
    for ticker,date,_,_ in group_keys.itertuples(index=False,name=None):
        group=queue[(queue.ticker.astype(str)==str(ticker))&queue.trade_date.eq(str(date))]
        if max_requests is not None and used>=max_requests:break
        path=_cache_path(KIS_CACHE,ticker,date);cached=path.exists();status={"ok":True};bars=pd.DataFrame()
        if cached:
            try:bars=pd.read_parquet(path)
            except Exception:cached=False
        if not cached:
            bars,status=client.get_day(ticker,date.replace("-",""));used+=int(status.get("api_requests",1))
            if status.get("ok") and not bars.empty and bar_qa(bars,"KR")["pass"]:
                path.parent.mkdir(parents=True,exist_ok=True);atomic_parquet(bars,path)
        qa=bar_qa(bars,"KR");ok=bool(status.get("ok") and not bars.empty and qa["pass"]);indices=group.index
        queue.loc[indices,"attempt_count"]=queue.loc[indices,"attempt_count"].astype(int)+(0 if cached else 1)
        valid_status="CACHED_VALID" if cached and verified else ("FETCHED_VALID" if verified else
                     ("CACHED_UNVERIFIED_SEMANTICS" if cached else "FETCHED_UNVERIFIED_SEMANTICS"))
        queue.loc[indices,"status"]=valid_status if ok else "PROVIDER_FAILED"
        fallback="EMPTY" if bars.empty else "QA_FAIL:"+json.dumps(qa,sort_keys=True)
        queue.loc[indices,"last_error"]="" if ok else str(status.get("error",fallback))[:300]
        ledger.append({"provider":"KIS","market":"KR","ticker_date":f"{ticker}/{date}","events_requested":len(group),
            "api_requests":0 if cached else int(status.get("api_requests",1)),"bars_downloaded":len(bars),
            "exact_eligible":0,"failed":0 if ok else 1,"rate_limited":int(str(status.get("msg_cd",""))=="EGW00201"),
            "cached":int(cached),"timestamp":now()})
        atomic_parquet(queue.drop(columns=["trade_date","_data_role"]),KR_QUEUE)
        atomic_json({"timestamp":now(),"phase":"FETCH_PRICE","provider":"KIS","api_requests":used,
            "last_ticker_date":f"{ticker}/{date}","pending":int((queue.status=="PENDING").sum())},ACQ_STATE)
    append_ledger(ledger);counts=Counter(queue.status.astype(str));audit={"timestamp":now(),"candidate_total":len(queue),
        "status_counts":dict(counts),"api_requests_this_run":used,"bar_semantics_verified":verified,
        "exact_contract_eligible":0,
        "scheduling_policy":"PENDING_THEN_FROZEN_DEV_EXTENSION_THEN_DATE_DESC_V1",
        "scheduling_uses_labels":False}
    existing=DATA/"KIS_HISTORICAL_COVERAGE_AUDIT.json"
    payload=json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}
    payload["queue_fetch"]=audit;atomic_json(payload,existing);return audit


def assign_roles()->dict[str,Any]:
    previous=pd.read_parquet(ROLE_PATH).set_index("event_id") if ROLE_PATH.exists() else pd.DataFrame()
    previous_sha=sha256(ROLE_PATH) if ROLE_PATH.exists() else None
    frames=[]
    for market,path in (("US",US_QUEUE),("KR",KR_QUEUE)):
        if not path.exists():continue
        frame=pd.read_parquet(path);frame["market"]=market;frames.append(frame)
    all_rows=pd.concat(frames,ignore_index=True).drop_duplicates("event_id").sort_values(["market","ticker","event_time_utc"])
    active_ids=set(all_rows.event_id.astype(str))
    hashes=all_rows.event_id.astype(str).map(lambda value:hashlib.sha256(f"{ROLE_SEED}|{value}".encode()).hexdigest())
    bucket=hashes.str[:8].map(lambda value:int(value,16)%100)
    policy=read_json_file(NEW_ROLE_POLICY_PATH)
    if policy:
        allocation=policy.get("new_row_allocation",{})
        dev_share=int(allocation.get("DEV_EXTENSION",-1));research_share=int(allocation.get("RESEARCH_SEAL_POOL",-1))
        final_share=int(allocation.get("FINAL_META_RESERVE",-1))
        if (policy.get("selection_uses_labels") is not False or
                dev_share<0 or research_share<0 or final_share<0 or
                dev_share+research_share+final_share!=100):
            raise RuntimeError("INVALID_NEW_ROLE_POLICY_ARTIFACT")
        policy_name=str(policy.get("policy_name") or "UNNAMED_NEW_ROLE_POLICY")
    else:
        dev_share,research_share,final_share=10,45,45;policy_name=NEW_ROLE_POLICY
    # This policy is frozen before any labels for these new rows are opened.
    # The larger independent pools are required by the original >=500 sample
    # criterion; all assignments remain deterministic and outcome blind.
    all_rows["role"]=np.select([bucket<dev_share,bucket<dev_share+research_share],
        ["DEV_EXTENSION","RESEARCH_SEAL_POOL"],default="FINAL_META_RESERVE")
    all_rows["role_hash"]=hashes;all_rows["role_frozen_at"]=now();all_rows["labels_opened"]=False
    all_rows["role_status"]="ACTIVE"
    if not previous.empty:
        common=all_rows.event_id.astype(str).isin(previous.index.astype(str))
        common_ids=all_rows.loc[common,"event_id"].astype(str)
        all_rows.loc[common,"role"]=common_ids.map(previous.role.to_dict()).to_numpy()
        all_rows.loc[common,"role_hash"]=common_ids.map(previous.role_hash.to_dict()).to_numpy()
        all_rows.loc[common,"role_frozen_at"]=all_rows.loc[common,"event_id"].map(previous.role_frozen_at.to_dict())
        all_rows.loc[common,"labels_opened"]=all_rows.loc[common,"event_id"].map(previous.labels_opened.to_dict()).fillna(False)
        # A failed source/timestamp audit may remove an event from the active
        # price queue, but it must never erase or recycle its frozen role.
        missing=previous.loc[~previous.index.astype(str).isin(active_ids)].reset_index().copy()
        if not missing.empty:
            missing["role_status"]="QUARANTINED_NOT_IN_ACTIVE_QUEUE"
            all_rows=pd.concat([all_rows,missing],ignore_index=True,sort=False)
        observed=all_rows.set_index("event_id").loc[previous.index,"role"].astype(str)
        if not observed.equals(previous.role.astype(str)):raise RuntimeError("FROZEN_ROLE_CHANGED")
    columns=["event_id","market","ticker","event_time_utc","source_family","provider","role","role_hash","role_frozen_at","labels_opened","role_status"]
    atomic_parquet(all_rows[columns],ROLE_PATH)
    active=all_rows[all_rows.role_status.eq("ACTIVE")]
    counts=active.groupby(["market","role"]).size().unstack(fill_value=0).to_dict(orient="index")
    report={"timestamp":now(),"seed_sha256":hashlib.sha256(ROLE_SEED.encode()).hexdigest(),
        "selection_uses_labels":False,"labels_opened":False,"new_role_policy":policy_name,
        "new_role_policy_artifact":str(NEW_ROLE_POLICY_PATH.relative_to(ROOT)) if policy else None,
        "new_role_policy_artifact_sha256":sha256(NEW_ROLE_POLICY_PATH) if policy else None,
        "policy_activation_previous_assignment_sha256":policy.get("activation_previous_assignment_sha256") if policy else None,
        "new_row_allocation":{"DEV_EXTENSION":dev_share,"RESEARCH_SEAL_POOL":research_share,"FINAL_META_RESERVE":final_share},
        "counts":counts,"active_rows":len(active),"preserved_inactive_rows":int(len(all_rows)-len(active)),
        "rows":len(all_rows),"previous_assignment_sha256":previous_sha,
        "existing_roles_unchanged":True,"new_rows":int(len(active_ids)-len(active_ids&set(previous.index.astype(str)))) if not previous.empty else len(active_ids)}
    atomic_json(report,DATA/"V59_DATA_ROLE_ASSIGNMENT.json");return report


def build_dev_extension_labels()->dict[str,Any]:
    """Open outcomes only for contract-authorized frozen DEV_EXTENSION rows.

    The V70 epoch was predeclared for US_SEC and KR_NEWS.  Other source
    families keep their immutable role assignments, but are not projected into
    this labeled epoch and cannot satisfy either source floor.
    """
    if not ROLE_PATH.exists():raise RuntimeError("ROLE_ASSIGNMENT_REQUIRED_BEFORE_LABELS")
    roles=pd.read_parquet(ROLE_PATH)
    authorized=(roles.role.eq("DEV_EXTENSION")&(
        (roles.market.eq("US")&roles.source_family.eq("US_SEC"))|
        (roles.market.eq("KR")&roles.source_family.eq("KR_NEWS"))))
    excluded_dev=roles.loc[roles.role.eq("DEV_EXTENSION")&~authorized,
                           ["market","source_family","event_id"]].copy()
    dev_ids=set(roles.loc[authorized&roles.market.eq("US"),"event_id"].astype(str))
    events=_us_candidates();events=events[events.event_id.astype(str).isin(dev_ids)].copy()
    cfg=app.Config(official_exact_contract=True);rows=[];reasons=Counter();provenance=[]
    us_status=load_us_provider_status();loaded_us={};provenance_counts=Counter()
    for _,event in events.assign(
            trade_date=events.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)).iterrows():
        authorities=resolve_us_price_cache_candidates(
            str(event.event_id),str(event.ticker),event.trade_date,us_status
        )
        if not authorities:reasons["US:NO_AUTHORIZED_CACHE"]+=1;continue
        selected=None;last_reason="NO_AUTHORIZED_CACHE"
        for path,provider,feed,provider_symbol in authorities:
          try:
            key=str(path)
            if key not in loaded_us:
                raw=pd.read_parquet(path);bars=raw.rename(columns={"timestamp":"datetime"})
                bars["datetime"]=pd.to_datetime(bars.datetime,utc=True);bars=bars.sort_values("datetime").drop_duplicates("datetime")
                bars.attrs.update({"price_source":f"{provider}_{feed}_1M","price_cadence_sec":60,
                    "bar_timestamp_semantics":"OPEN","official_exact_eligible":True})
                loaded_us[key]=(bars,sha256(path))
            bars,raw_sha=loaded_us[key]
            row,last_reason=app.event_to_row_exact_contract(event,bars,cfg)
            if row is not None:
                selected=(row,path,provider,feed,provider_symbol,raw_sha);break
          except Exception as error:
            last_reason=safe_error(None,error)
        if selected is None:reasons[f"US:{last_reason}"]+=1;continue
        row,path,provider,feed,provider_symbol,raw_sha=selected
        reasons["US:OK"]+=1
        row.update({"price_provider":provider,"price_feed":feed,"provider_symbol":provider_symbol,
            "raw_file":str(path.relative_to(ROOT)),"raw_file_sha256":raw_sha,
            "download_timestamp":datetime.fromtimestamp(path.stat().st_mtime,tz=timezone.utc).isoformat(),
            "data_role":"DEV_EXTENSION"});rows.append(row)
        provenance_counts[(str(path.relative_to(ROOT)),raw_sha,provider,feed)]+=1
    provenance.extend({"path":key[0],"sha256":key[1],"provider":key[2],"feed":key[3],"events":count}
                      for key,count in provenance_counts.items())
    kr_ids=set(roles.loc[authorized&roles.market.eq("KR"),"event_id"].astype(str))
    kr_events=_kr_candidates();kr_events=kr_events[kr_events.event_id.astype(str).isin(kr_ids)].copy()
    kis_verified=bool(read_json_file(DATA/"BAR_SEMANTICS_KIS.json").get("official_exact_eligible"))
    for (ticker,date),group in kr_events.assign(
            trade_date=kr_events.event_time_utc.dt.tz_convert(app.KR_TZ).dt.date.astype(str)).groupby(["ticker","trade_date"]):
        path=_cache_path(KIS_CACHE,ticker,date)
        if not kis_verified:reasons["KR:BAR_SEMANTICS_UNVERIFIED"]+=len(group);continue
        if not path.exists():reasons["KR:NO_CACHE"]+=len(group);continue
        try:
            raw=pd.read_parquet(path);bars=raw.rename(columns={"timestamp":"datetime"})
            bars["datetime"]=pd.to_datetime(bars.datetime,utc=True);bars=bars.sort_values("datetime").drop_duplicates("datetime")
            bars.attrs.update({"price_source":"KIS_HISTORICAL_1M","price_cadence_sec":60,
                "bar_timestamp_semantics":"OPEN","official_exact_eligible":True})
            raw_sha=sha256(path)
            for _,event in group.iterrows():
                row,reason=app.event_to_row_exact_contract(event,bars,cfg);reasons[f"KR:{reason}"]+=1
                if row is not None:
                    row.update({"price_provider":"KIS","provider_symbol":str(ticker).zfill(6),"raw_file_sha256":raw_sha,
                        "download_timestamp":datetime.fromtimestamp(path.stat().st_mtime,tz=timezone.utc).isoformat(),
                        "data_role":"DEV_EXTENSION"});rows.append(row)
            provenance.append({"path":str(path.relative_to(ROOT)),"sha256":raw_sha,"events":len(group)})
        except Exception as error:reasons[f"KR:{safe_error(None,error)}"]+=len(group)
    output=pd.DataFrame(rows)
    out_path=DATA/"V59_DEV_EXTENSION_LABELED.csv.gz"
    if not output.empty:
        temporary=out_path.with_name(out_path.name+f".tmp.{os.getpid()}")
        # Freeze both gzip timestamp and embedded filename so identical
        # labeled rows have an identical hash across audit-only rebuilds.
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="",mode="wb",fileobj=raw_handle,mtime=0) as compressed:
                with io.TextIOWrapper(compressed,encoding="utf-8",newline="") as text_handle:
                    output.to_csv(text_handle,index=False)
        os.replace(temporary,out_path)
    manifest={"timestamp":now(),"role_assignment_sha256":sha256(ROLE_PATH),"role":"DEV_EXTENSION",
        "selection_uses_labels":False,"labels_opened_after_role_freeze":True,"rows":len(output),
        "US":int(output.market.eq("US").sum()) if not output.empty else 0,
        "KR":int(output.market.eq("KR").sum()) if not output.empty else 0,"reasons":dict(reasons),"raw_files":provenance,
        "authorized_source_families":["US_SEC","KR_NEWS"],
        "excluded_frozen_dev_rows":int(len(excluded_dev)),
        "excluded_frozen_dev_by_source":({str(key):int(value) for key,value in
            excluded_dev.groupby("source_family").size().to_dict().items()} if not excluded_dev.empty else {}),
        "output":str(out_path.relative_to(ROOT)) if out_path.exists() else None,
        "output_sha256":sha256(out_path) if out_path.exists() else None}
    atomic_json(manifest,DATA/"V59_DEV_EXTENSION_MANIFEST.json")
    # ROLE_PATH is immutable after label-blind assignment.  Opened DEV ids are
    # projected only through this manifest, never by rewriting role rows.
    return manifest


def audit_exact_timing_roles()->dict[str,Any]:
    """Audit authorized provider timestamps without opening any return labels."""
    roles=pd.read_parquet(ROLE_PATH).set_index("event_id");counts=Counter();reasons=Counter();eligible_ids=[]
    us_status=load_us_provider_status()
    def timing_reason(path:Path,row:Any)->str:
        bars=pd.read_parquet(path,columns=["timestamp"]);values=pd.to_datetime(bars.timestamp,utc=True).sort_values().drop_duplicates().array.asi8
        target=pd.Timestamp(row.event_time_utc)+pd.Timedelta(minutes=2);position=int(np.searchsorted(values,int(target.value)))
        if position>=len(values):return "BAR_MISSING"
        entry=pd.Timestamp(values[position],tz="UTC");slip=(entry-target).total_seconds()
        if not 0<=slip<=60:return "ENTRY_TIMING_MISS"
        exit_target=entry+pd.Timedelta(minutes=30);exit_position=int(np.searchsorted(values,int(exit_target.value)))
        if exit_position>=len(values):return "BAR_MISSING"
        exit_time=pd.Timestamp(values[exit_position],tz="UTC");exit_slip=(exit_time-exit_target).total_seconds()
        hold=(exit_time-entry).total_seconds()
        if not 0<=exit_slip<=60 or not 1800<=hold<=1860:return "EXIT_TIMING_MISS"
        return "OK"
    providers=(("US",pd.read_parquet(US_QUEUE),MASSIVE_CACHE,app.US_TZ,True),
               ("KR",pd.read_parquet(KR_QUEUE),KIS_CACHE,app.KR_TZ,
                bool(read_json_file(DATA/"BAR_SEMANTICS_KIS.json").get("official_exact_eligible"))))
    for market,queue,cache,timezone_name,semantics_verified in providers:
      for row in queue.itertuples(index=False):
        role=str(roles.loc[str(row.event_id),"role"]);local=pd.Timestamp(row.event_time_utc).tz_convert(timezone_name)
        if not semantics_verified:reasons[f"{market}:{role}:SEMANTICS_UNVERIFIED"]+=1;continue
        if market=="US":
            authorities=resolve_us_price_cache_candidates(
                str(row.event_id),str(row.ticker),local.date(),us_status
            )
            paths=[authority[0] for authority in authorities]
            if not paths:reasons[f"{role}:NO_AUTHORIZED_CACHE"]+=1;continue
        else:
            path=_cache_path(cache,str(row.ticker),local.date())
            if not path.exists():reasons[f"{role}:NO_CACHE"]+=1;continue
            paths=[path]
        try:
            result="NO_CACHE"
            for candidate_path in paths:
                result=timing_reason(candidate_path,row)
                if result=="OK":break
            if result!="OK":reasons[f"{role}:{result}"]+=1;continue
            counts[f"{market}:{role}"]+=1;eligible_ids.append(str(row.event_id))
        except Exception as error:reasons[f"{role}:{safe_error(None,error)}"]+=1
    report={"timestamp":now(),"providers":["MASSIVE","ALPACA_SIP","LOCAL_FINNHUB_1M","LOCAL_FINNHUB_ARCHIVE_1M","KIS"],"bar_semantics":"OPEN","selection_uses_labels":False,
        "outcome_columns_loaded":False,"eligible_by_role":dict(counts),"eligible_total":sum(counts.values()),
        "reason_counts":dict(reasons),"eligible_id_set_sha256":hashlib.sha256("\n".join(sorted(eligible_ids)).encode()).hexdigest()}
    atomic_json(report,DATA/"V59_EXACT_TIMING_ELIGIBILITY.json")
    massive=read_json_file(DATA/"MASSIVE_COVERAGE_AUDIT.json")
    massive_counts={key.split(":",1)[1]:value for key,value in counts.items() if key.startswith("US:")}
    massive["exact_timing_eligible_by_role"]=massive_counts;massive["exact_contract_eligible"]=sum(massive_counts.values())
    atomic_json(massive,DATA/"MASSIVE_COVERAGE_AUDIT.json")
    kis=read_json_file(DATA/"KIS_HISTORICAL_COVERAGE_AUDIT.json")
    kis_counts={key.split(":",1)[1]:value for key,value in counts.items() if key.startswith("KR:")}
    kis.setdefault("queue_fetch",{})["exact_timing_eligible_by_role"]=kis_counts
    kis["queue_fetch"]["exact_contract_eligible"]=sum(kis_counts.values());atomic_json(kis,DATA/"KIS_HISTORICAL_COVERAGE_AUDIT.json")
    # Keep the controller's acquisition projection synchronized with the
    # authoritative timing-only audit.  No return or direction labels are
    # loaded here, including for the two sealed roles.
    source_path=RESEARCH/"SOURCE_ACQUISITION_STATUS.json"
    source=read_json_file(source_path)
    by_market={market:sum(value for key,value in counts.items() if key.startswith(f"{market}:"))
               for market in ("US","KR")}
    role_exact={role:{market:int(counts.get(f"{market}:{role}",0)) for market in ("US","KR")}
                for role in ("DEV_EXTENSION","RESEARCH_SEAL_POOL","FINAL_META_RESERVE")}
    for role,payload in role_exact.items():payload["total"]=payload["US"]+payload["KR"]
    def minimum_met(role:str)->bool:
        payload=role_exact[role]
        return payload["total"]>=500 and payload["US"]>=250 and payload["KR"]>=100
    source.update({"timestamp":now(),"exact_eligible":{"US":by_market["US"],"KR":by_market["KR"],
        "total":by_market["US"]+by_market["KR"]},"exact_eligible_by_role":role_exact,
        "research_seal_minimum_met":minimum_met("RESEARCH_SEAL_POOL"),
        "final_meta_minimum_met":minimum_met("FINAL_META_RESERVE"),
        "seal_frozen":False,"status":"ACQUIRING_EXACT_PRICE"})
    atomic_json(source,source_path)
    return report


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--build-queues",action="store_true");parser.add_argument("--fetch-massive",action="store_true")
    parser.add_argument("--probe-massive",action="store_true");parser.add_argument("--probe-kis",action="store_true")
    parser.add_argument("--audit-kis-semantics",action="store_true")
    parser.add_argument("--fetch-kis",action="store_true");parser.add_argument("--assign-roles",action="store_true")
    parser.add_argument("--build-dev-labels",action="store_true");parser.add_argument("--audit-timing",action="store_true")
    parser.add_argument("--max-requests",type=int)
    parser.add_argument("--massive-min-interval",type=float,default=12.5,
                        help="Initial seconds between Massive requests; any 429 raises it to at least 12.5.")
    return parser.parse_args()


def main()->int:
    args=parse_args();import_user_environment_if_missing();runtime_limits.configure();results={}
    massive_client=MassiveClient(args.massive_min_interval) if (args.preflight or args.probe_massive or args.fetch_massive) and credential_status()["MASSIVE_API_KEY"]=="PRESENT" else None
    kis_client=KISClient() if (args.preflight or args.probe_kis or args.fetch_kis or args.audit_kis_semantics) and credential_status()["KIS_APP_KEY"]=="PRESENT" else None
    if args.preflight:results["preflight"]=preflight(massive_client,kis_client)
    if args.build_queues:results["queues"]=build_queues()
    if args.probe_massive:results["massive_probe"]=probe_massive_coverage(massive_client)
    if args.probe_kis:results["kis_probe"]=probe_kis_coverage(kis_client)
    if args.audit_kis_semantics:results["kis_semantics"]=audit_kis_semantics(kis_client)
    if args.fetch_massive:results["massive_fetch"]=fetch_massive(args.max_requests,massive_client)
    if args.fetch_kis:results["kis_fetch"]=fetch_kis(args.max_requests,kis_client)
    if args.assign_roles:results["roles"]=assign_roles()
    if args.build_dev_labels:results["dev_labels"]=build_dev_extension_labels()
    if args.audit_timing:results["timing_audit"]=audit_exact_timing_roles()
    print(json.dumps(results,ensure_ascii=False,indent=2,default=str));return 0


if __name__=="__main__":raise SystemExit(main())
