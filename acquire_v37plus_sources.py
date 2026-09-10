"""Outcome-blind V37+ prospective source acquisition and exact-price audit.

This worker may append event metadata, but never computes/opens forward returns
and never writes cert/.  Exact eligibility remains zero until an audited 60-second
OPEN-stamped raw price provider covers the event window.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import bio_news_30m_v3 as app
import prepare_real_data as prep
import runtime_limits
import v36_exact_pipeline as exact
from audit_v36_static import biomedical_name_evidence
from provider_env_runtime import import_user_environment_if_missing


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
POOL=DATA/"untouched_pool"
RESEARCH=ROOT/"research"
STATUS=RESEARCH/"SOURCE_ACQUISITION_STATUS.json"
LEDGER=RESEARCH/"SOURCE_ACQUISITION_LEDGER.jsonl"
# Prospective issuer selection must stay aligned with the audited V36 contract.
# In particular, never re-introduce substring tests such as "dental" in
# "Occidental".  Name evidence is evaluated by word-boundary regular
# expressions in audit_v36_static.biomedical_name_evidence().
EXPLICIT_NON_HEALTH_SIC_RANGES=((100,999),(1000,1499),(4900,4999))


def now()->str:return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def atomic_json(value:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    os.replace(temporary,path)


def append_ledger(value:dict[str,Any])->None:
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with LEDGER.open("a",encoding="utf-8",newline="\n") as handle:
        handle.write(json.dumps(value,ensure_ascii=False,sort_keys=True,default=str)+"\n")
        handle.flush();os.fsync(handle.fileno())


def credential_audit()->dict[str,Any]:
    groups={
        "alpaca":["APCA_API_KEY_ID","APCA_API_SECRET_KEY"],
        "massive":["MASSIVE_API_KEY"],
        "polygon_legacy":["POLYGON_API_KEY"],
        "kis":["KIS_APP_KEY","KIS_APP_SECRET"],
        "kiwoom":["KIWOOM_APP_KEY","KIWOOM_SECRET_KEY"],
    }
    return {provider:{"configured":all(bool(os.environ.get(name)) for name in names),
                      "required_names":names} for provider,names in groups.items()}


def existing_audit()->dict[str,Any]:
    audit=json.loads((DATA/"V36_EXACT_SOURCE_CANDIDATE_AUDIT.json").read_text(encoding="utf-8"))
    us=audit["US"]["stages"];kr=audit["KR"]["stages"]
    return {
        "selection_uses_labels":False,"US_historical_metadata_candidates":us["canonical_35m_spaced"],
        "US_exact_timing_eligible":us["exact_timing_eligible_before_cap"],
        "US_timing_reasons":audit["US"].get("timing_reasons",{}),
        "KR_unused_healthcare_candidates":kr["healthcare_seed_policy"],
        "KR_regular_session_candidates":kr["regular_session"],
        "KR_exact_timing_eligible":kr["exact_timing_eligible_before_cap"],
    }


def audited_prospective_universe(ticker_map:dict[str,Any])->tuple[list[tuple[str,str,int]],dict[str,Any]]:
    """Build a label-blind SEC issuer universe under the audited V36 policy."""
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    approved_seed=set(seed.loc[seed.market.eq("US"),"ticker"].astype(str).str.upper()
                      .str.replace(".","-",regex=False))
    metadata_payload=json.loads((DATA/"V36_SEC_ISSUER_METADATA.json").read_text(encoding="utf-8"))
    metadata={str(int(row["cik"])):row for row in metadata_payload.get("issuers",[])
              if str(row.get("cik","")).strip().isdigit()}
    universe=[];quarantine=[];evidence_counts={"seed":0,"approved_sic":0,"boundary_name":0}
    for item in ticker_map.values():
        ticker=str(item.get("ticker","")).upper().replace(".","-").strip()
        company=str(item.get("title","")).strip()
        if not ticker or not str(item.get("cik_str","")).isdigit():continue
        cik=int(item["cik_str"]);issuer=metadata.get(str(cik),{})
        sic=str(issuer.get("sic","") or "").strip();name=biomedical_name_evidence(company)
        seed_pass=ticker in approved_seed
        sic_pass=sic in exact.APPROVED_HEALTHCARE_SIC
        name_pass=bool(name["name_rule_pass"])
        numeric_sic=int(sic) if sic.isdigit() else None
        explicit_conflict=(numeric_sic is not None and
                           any(low<=numeric_sic<=high for low,high in EXPLICIT_NON_HEALTH_SIC_RANGES))
        if explicit_conflict and not seed_pass and not sic_pass:
            if name_pass:quarantine.append({"ticker":ticker,"cik":cik,"sic":sic,
                "reason":"SIC_CONFLICT_QUARANTINE"})
            continue
        if seed_pass or sic_pass or name_pass:
            universe.append((ticker,company,cik))
            evidence_counts["seed"]+=int(seed_pass)
            evidence_counts["approved_sic"]+=int(sic_pass)
            evidence_counts["boundary_name"]+=int(name_pass)
    return sorted(set(universe),key=lambda value:(value[2],value[0])),{
        "policy":"V36_SEED_OR_APPROVED_SIC_OR_BOUNDARY_SAFE_NAME",
        "selection_uses_labels":False,"substring_matching_forbidden":True,
        "approved_sic_sha256":hashlib.sha256("\n".join(sorted(exact.APPROVED_HEALTHCARE_SIC)).encode()).hexdigest(),
        "evidence_counts":evidence_counts,"sic_conflict_quarantine_count":len(quarantine),
        "sic_conflict_quarantine_sample":quarantine[:20],
    }


def sec_archive_files_for_interval(payload:dict[str,Any],start_time:pd.Timestamp,
                                   end_time:pd.Timestamp)->list[str]:
    """Return official submission shards that can overlap ``[start, end)``.

    SEC's top-level ``filings.recent`` arrays are capped.  High-filing-count
    issuers can therefore hide otherwise eligible filings in the immutable
    ``filings.files`` shards.  Descriptor dates are used only to avoid
    irrelevant network calls; exact acceptance timestamps are filtered again
    after the shard is loaded.
    """
    selected=[]
    for descriptor in payload.get("filings",{}).get("files",[]) or []:
        name=str(descriptor.get("name","") or "").strip()
        if (not re.fullmatch(r"CIK\d+-submissions-\d+\.json",name)
                or Path(name).name!=name):
            continue
        filing_from=pd.to_datetime(descriptor.get("filingFrom"),errors="coerce")
        filing_to=pd.to_datetime(descriptor.get("filingTo"),errors="coerce")
        if pd.isna(filing_from) or pd.isna(filing_to):
            continue
        interval_start=start_time.tz_convert("UTC").tz_localize(None).normalize()
        # ``end_time`` is exclusive while the SEC descriptor end date is
        # inclusive, so compare it with the final included calendar day.
        interval_last=(end_time.tz_convert("UTC")-pd.Timedelta(nanoseconds=1)).tz_localize(None).normalize()
        if filing_to.normalize()>=interval_start and filing_from.normalize()<=interval_last:
            selected.append(name)
    return sorted(set(selected))


def collect_recent_sec(start:str,end:str)->dict[str,Any]:
    """Append an outcome-free SEC acceptance-time snapshot for the interval."""
    collector=app.SECCollector(app.Config());ticker_map=collector.get_json(collector.TICKER_MAP)
    universe,universe_audit=audited_prospective_universe(ticker_map)
    start_time=pd.Timestamp(start,tz="UTC");end_time=pd.Timestamp(end,tz="UTC")
    allowed=set(app.Config().sec_forms);events=[];errors=[]
    archive_considered=archive_fetched=archive_errors=0
    for index,(ticker,company,cik) in enumerate(universe,1):
        try:
            payload=collector.get_json(collector.SUBMISSIONS.format(cik=cik))
            recent=payload.get("filings",{}).get("recent",{}) or {}
            blocks=[("recent",recent)]
            archive_names=sec_archive_files_for_interval(payload,start_time,end_time)
            archive_considered+=len(archive_names)
            for name in archive_names:
                try:
                    blocks.append((name,collector.get_json(f"https://data.sec.gov/submissions/{name}")))
                    archive_fetched+=1
                except Exception as error:
                    archive_errors+=1
                    errors.append({"ticker":ticker,"cik":cik,"scope":name,"error":repr(error)})
            for _,block in blocks:
                accessions=block.get("accessionNumber",[]) or [];forms=block.get("form",[]) or []
                accepted=block.get("acceptanceDateTime",[]) or [];primary=block.get("primaryDocument",[]) or []
                items=block.get("items",[]) or []
                for position,accession in enumerate(accessions):
                    form=str(forms[position] if position<len(forms) else "").strip()
                    raw_time=str(accepted[position] if position<len(accepted) else "")
                    if form not in allowed or not raw_time:continue
                    timestamp=pd.Timestamp(raw_time)
                    if timestamp.tzinfo is None:timestamp=timestamp.tz_localize(app.US_TZ).tz_convert("UTC")
                    else:timestamp=timestamp.tz_convert("UTC")
                    if not start_time<=timestamp<end_time:continue
                    local=timestamp.tz_convert(app.US_TZ)
                    if local.dayofweek>=5 or not pd.Timestamp("09:30").time()<=local.time()<=pd.Timestamp("15:28").time():continue
                    accession=str(accession);primary_name=str(primary[position] if position<len(primary) else "").strip()
                    accession_path=accession.replace("-","")
                    url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{primary_name}" if primary_name else \
                        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/"
                    item_text=str(items[position] if position<len(items) else "").strip();headline=f"{form} {item_text}".strip()
                    events.append({"event_id":f"SEC:{cik}:{accession}","market":"US","ticker":ticker,
                        "company":company,"event_time_utc":timestamp,"source":"SEC_PROSPECTIVE_V37PLUS",
                        "form":form,"headline":headline,"body":"","event_type":app.infer_event_type(headline,form),
                        "timestamp_quality":"EXACT","url":url,"article_id":f"SEC:{cik}:{accession}",
                        "selection_uses_labels":False})
        except Exception as error:errors.append({"ticker":ticker,"cik":cik,"error":repr(error)})
        if index%100==0:print(f"[SOURCE SEC] {index}/{len(universe)} events={len(events)}",flush=True)
    frame=pd.DataFrame(events)
    if not frame.empty:
        frame=frame.drop_duplicates("event_id").sort_values("event_time_utc").reset_index(drop=True)
        prior=exact.normalized_events(DATA/"events_exact_v35.csv.gz")
        prior_ids=set(prior.event_id.astype(str));prior_urls=set(prior.url.fillna("").astype(str))
        frame=frame[~frame.event_id.astype(str).isin(prior_ids)&~frame.url.astype(str).isin(prior_urls)].copy()
        # Apply the canonical ticker-time embargo without reading any outcomes.
        keep=[];prior_times={ticker:group.event_time_utc.sort_values().astype("int64").to_numpy()
                            for ticker,group in prior.groupby("ticker")}
        for row in frame.itertuples(index=False):
            values=prior_times.get(str(row.ticker));stamp=int(pd.Timestamp(row.event_time_utc).value)
            keep.append(values is None or not ((abs(values-stamp)<=35*60*1_000_000_000).any()))
        frame=frame.loc[keep].copy()
        raw=frame.to_csv(index=False).encode("utf-8");content_sha=hashlib.sha256(raw).hexdigest()
        POOL.mkdir(parents=True,exist_ok=True);path=POOL/f"sec_{start}_{end}_{content_sha[:12]}.csv.gz"
        if not path.exists():frame.to_csv(path,index=False,compression="gzip")
        snapshot={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"rows":len(frame)}
    else:snapshot={"path":None,"sha256":None,"rows":0}
    return {"interval":{"start":start,"end":end},"universe":len(universe),
            "universe_audit":universe_audit,"snapshot":snapshot,
            "submission_coverage_class":"RECENT_PLUS_OVERLAPPING_OFFICIAL_ARCHIVE_SHARDS",
            "archive_shards_considered":archive_considered,"archive_shards_fetched":archive_fetched,
            "archive_shard_errors":archive_errors,
            "request_errors":len(errors),"error_sample":errors[:20],"selection_uses_labels":False,
            "exact_price_eligible":0,"seal_frozen":False}


def collect_recent_kr_naver(start:str,end:str,pages:int=5)->dict[str,Any]:
    """Collect a label-blind recent Naver ticker-news snapshot for V29 health issuers."""
    universe=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    universe["ticker"]=universe.ticker.astype(str).str.zfill(6)
    start_time=pd.Timestamp(start,tz=app.KR_TZ);end_time=pd.Timestamp(end,tz=app.KR_TZ)
    headers={"User-Agent":"Mozilla/5.0 MARKET_BIO research","Accept":"application/json",
             "Referer":"https://m.stock.naver.com/"}
    def fetch(ticker:str,company:str)->tuple[list[dict[str,Any]],dict[str,Any]]:
        records=[];seen=set();error=""
        for page in range(1,pages+1):
            response=None
            for attempt in range(3):
                try:
                    response=requests.get(f"https://m.stock.naver.com/api/news/stock/{ticker}",
                        params={"page":page,"pageSize":20},headers=headers,timeout=30)
                    if response.status_code==429:time.sleep(2**attempt);continue
                    response.raise_for_status();break
                except Exception as exc:
                    error=type(exc).__name__;time.sleep(.5*(attempt+1))
            if response is None or response.status_code!=200:break
            try:groups=response.json();items=[item for group in groups for item in group.get("items",[])]
            except Exception:error="MALFORMED_JSON";break
            if not items:break
            oldest=None
            for item in items:
                article_id=str(item.get("id","") or "");stamp=pd.to_datetime(str(item.get("datetime","") or ""),format="%Y%m%d%H%M",errors="coerce")
                if pd.isna(stamp):continue
                stamp=stamp.tz_localize(app.KR_TZ);oldest=stamp if oldest is None else min(oldest,stamp)
                if article_id in seen or not start_time<=stamp<end_time:continue
                seen.add(article_id);headline=html.unescape(str(item.get("titleFull") or item.get("title") or ""))
                body=html.unescape(str(item.get("body","") or ""))
                records.append({"event_id":f"NAVER:{ticker}:{article_id}","market":"KR","ticker":ticker,
                    "company":company,"event_time_utc":stamp.tz_convert("UTC"),"source":"NAVER_NEWS_PROSPECTIVE_V59",
                    "form":"","headline":headline,"body":body,"event_type":app.infer_event_type(f"{headline} {body}"),
                    "timestamp_quality":"EXACT","url":str(item.get("mobileNewsUrl") or item.get("url") or ""),
                    "article_id":article_id,"selection_uses_labels":False})
            if oldest is not None and oldest<start_time:break
        return records,{"ticker":ticker,"kept":len(records),"error":error}
    records=[];statuses=[]
    with ThreadPoolExecutor(max_workers=min(16,len(universe))) as pool:
        futures=[pool.submit(fetch,str(row.ticker),str(row.company)) for row in universe.itertuples(index=False)]
        for done,future in enumerate(as_completed(futures),1):
            rows,status=future.result();records.extend(rows);statuses.append(status)
            if done%50==0:print(f"[SOURCE KR NAVER] {done}/{len(futures)} rows={len(records)}",flush=True)
    frame=pd.DataFrame(records)
    if not frame.empty:
        frame["selection_hash"]=frame.apply(lambda row:hashlib.sha256(f"{row.article_id}|{row.ticker}".encode()).hexdigest(),axis=1)
        frame=(frame.sort_values(["article_id","selection_hash"]).drop_duplicates("article_id")
               .drop(columns="selection_hash").sort_values("event_time_utc"))
        raw=frame.to_csv(index=False).encode("utf-8");content_sha=hashlib.sha256(raw).hexdigest();POOL.mkdir(parents=True,exist_ok=True)
        path=POOL/f"naver_kr_{start}_{end}_{content_sha[:12]}.csv.gz";frame.to_csv(path,index=False,compression="gzip")
        snapshot={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"rows":len(frame)}
    else:snapshot={"path":None,"sha256":None,"rows":0}
    return {"interval":{"start":start,"end":end},"universe":len(universe),"pages":pages,"snapshot":snapshot,
            "request_errors":sum(bool(row["error"]) for row in statuses),
            "coverage_class":f"PARTIAL_LATEST_{pages*20}_PER_TICKER",
            "complete_interval_claimed":False,"selection_uses_labels":False}


def collect_recent_kr_kind(start:str,end:str,workers:int=8)->dict[str,Any]:
    """Collect exact-time KIND disclosures for the audited V29 KR universe.

    ``prepare_real_data._kind_day`` is the already audited parser for KIND's
    date-specific disclosure table.  This wrapper deliberately does not use
    labels, returns partial progress when individual days fail, and writes a
    content-addressed immutable source snapshot before price acquisition.
    """
    universe=pd.read_csv(DATA/"universe_v29_kr_health.csv",dtype={"ticker":str})
    universe["ticker"]=universe.ticker.astype(str).str.zfill(6)
    approved=set(universe.ticker)
    jobs=[(stamp.date(),market) for stamp in pd.date_range(start,end,freq="D") for market in ("1","2")]
    frames=[];failures=[]

    def fetch(job:tuple[Any,str])->tuple[Any,str,pd.DataFrame,str]:
        day,market=job
        try:return day,market,prep._kind_day(day,market,force=False),""
        except Exception as error:return day,market,pd.DataFrame(),type(error).__name__

    with ThreadPoolExecutor(max_workers=min(max(1,workers),16)) as pool:
        futures=[pool.submit(fetch,job) for job in jobs]
        for done,future in enumerate(as_completed(futures),1):
            day,market,frame,error=future.result()
            if error:failures.append({"date":day.isoformat(),"market_type":market,"error":error})
            elif not frame.empty:frames.append(frame)
            if done%100==0 or done==len(futures):
                print(f"[SOURCE KR KIND] {done}/{len(futures)} failures={len(failures)}",flush=True)

    frame=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
    if not frame.empty:
        frame["ticker"]=frame.ticker.astype(str).str.zfill(6)
        frame=frame[frame.ticker.isin(approved)].copy()
        frame["source"]="KIND_PROSPECTIVE_V59"
        frame["selection_uses_labels"]=False
        frame=(frame.drop_duplicates("event_id").sort_values(["event_time_utc","ticker","event_id"]))
        raw=frame.to_csv(index=False).encode("utf-8");content_sha=hashlib.sha256(raw).hexdigest()
        POOL.mkdir(parents=True,exist_ok=True)
        path=POOL/f"kind_kr_{start}_{end}_{content_sha[:12]}.csv.gz"
        frame.to_csv(path,index=False,compression="gzip")
        snapshot={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"rows":len(frame)}
    else:snapshot={"path":None,"sha256":None,"rows":0}
    failure_path=None
    if failures:
        failure_file=DATA/f"KIND_V59_FETCH_FAILURES_{start}_{end}.json"
        atomic_json(failures,failure_file);failure_path=str(failure_file.relative_to(ROOT))
    return {"interval":{"start":start,"end":end},"universe":len(universe),"requests":len(jobs),
            "snapshot":snapshot,"request_errors":len(failures),"failure_report":failure_path,
            "selection_uses_labels":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--collect-sec",action="store_true")
    parser.add_argument("--collect-kr-naver",action="store_true");parser.add_argument("--pages",type=int,default=5)
    parser.add_argument("--collect-kr-kind",action="store_true");parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--start",default=(pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=3)).date().isoformat())
    parser.add_argument("--end",default=(pd.Timestamp.now(tz="UTC")+pd.Timedelta(days=1)).date().isoformat())
    args=parser.parse_args();import_user_environment_if_missing();runtime_limits.configure()
    credentials=credential_audit();local=existing_audit()
    previous=json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    prospective=collect_recent_sec(args.start,args.end) if args.collect_sec else previous.get("prospective_SEC")
    kr_prospective=(collect_recent_kr_naver(args.start,args.end,args.pages) if args.collect_kr_naver
                    else previous.get("prospective_KR_NAVER"))
    if kr_prospective:
        kr_prospective.setdefault("coverage_class",f"PARTIAL_LATEST_{int(kr_prospective.get('pages',0))*20}_PER_TICKER")
        kr_prospective.setdefault("complete_interval_claimed",False)
    kr_kind=(collect_recent_kr_kind(args.start,args.end,args.workers) if args.collect_kr_kind
             else previous.get("prospective_KR_KIND"))
    prior_exact=previous.get("exact_eligible",{}) or {}
    prior_exact={"US":int(prior_exact.get("US",0) or 0),
                 "KR":int(prior_exact.get("KR",0) or 0),
                 "total":int(prior_exact.get("total",0) or 0)}
    exact_provider_ready=(credentials["alpaca"]["configured"] or credentials["massive"]["configured"] or
                          credentials["polygon_legacy"]["configured"] or prior_exact["US"]>0 or
                          bool(previous.get("US_exact_price_provider_ready",False)))
    # Metadata collection must not erase independently audited price eligibility,
    # deterministic roles, or seal-readiness projections.  Start from the latest
    # status and update only the fields owned by this collector.
    status=dict(previous)
    status.update({"timestamp":now(),"selection_uses_labels":False,"outcome_columns_loaded":False,
            "local_candidate_audit":local,"credentials":credentials,"prospective_SEC":prospective,
            "prospective_KR_NAVER":kr_prospective,
            "prospective_KR_KIND":kr_kind,
            "US_exact_price_provider_ready":exact_provider_ready,
            "KR_exact_price_provider_ready":bool(credentials["kis"]["configured"] or prior_exact["KR"]>0 or
                previous.get("KR_exact_price_provider_ready",False)),
            "source_minimum":previous.get("source_minimum",{"US":250,"KR":100,"total":500}),
            "exact_eligible":prior_exact,
            "seal_frozen":bool(previous.get("seal_frozen",False)),
            "status":previous.get("status") if prior_exact["total"]>0 else
                "WAITING_EXACT_PRICE_AND_KR_PROSPECTIVE_SOURCE",
            "next_actions":previous.get("next_actions",[
                "Alpaca 2016+ as-of ticker pilot when credentials exist",
                "KIS last-year DEV expansion when credentials exist",
                "append prospective KIND/Naver regular-session events before labeling"])} )
    atomic_json(status,STATUS);append_ledger(status)
    print(json.dumps(status,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
