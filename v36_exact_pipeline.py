"""Label-blind V36 source construction and exact-contract DEV regeneration."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import bio_news_30m_v3 as app
import runtime_limits
from audit_v36_static import biomedical_name_evidence


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
PRICE=ROOT/"price_data"
OUT=ROOT/"output_V36"
CACHE=ROOT/"cache"/"v36_sec_issuer_meta"
OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
FORBIDDEN_OUTCOME_COLUMNS={"y","fwd_ret_30m","entry_price","exit_price","prediction","probability"}
APPROVED_HEALTHCARE_SIC={
    "2833","2834","2835","2836",  # medicinal chemicals, pharma, diagnostics, biologics
    "3826",                          # analytical laboratory instruments
    "3841","3842","3843","3844","3845","3851",  # medical/dental/ophthalmic devices
    "8011","8021","8031","8041","8042","8049","8051","8052","8059",
    "8062","8063","8069","8071","8082","8090","8092","8093","8099",
}


def atomic_json(payload:Any,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame:pd.DataFrame,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temporary,index=False,encoding="utf-8")
    temporary.replace(path)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_events(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path,dtype={"ticker":str,"article_id":str,"cik":str},low_memory=False)
    for column in ("event_id","market","ticker","company","source","form","headline","body",
                   "event_type","url","article_id","timestamp_quality"):
        if column not in frame:frame[column]=""
    frame["ticker"]=frame.ticker.astype(str)
    frame.loc[frame.market.astype(str).eq("KR"),"ticker"]=(
        frame.loc[frame.market.astype(str).eq("KR"),"ticker"].str.zfill(6)
    )
    frame["event_time_utc"]=pd.to_datetime(frame.event_time_utc,utc=True,errors="coerce")
    return frame.dropna(subset=["event_time_utc"]).copy()


def accession_key(frame:pd.DataFrame)->pd.Series:
    return frame.event_id.astype(str).str.extract(r"^SEC:\d+:(.+)$",expand=False).fillna("")


def article_key(frame:pd.DataFrame)->pd.Series:
    article=frame.article_id.fillna("").astype(str).str.strip()
    return article.where(article.ne(""),frame.event_id.astype(str))


def outside_prior(frame:pd.DataFrame,prior:pd.DataFrame)->tuple[pd.DataFrame,dict[str,int]]:
    candidate=frame.copy()
    prior_ids=set(prior.event_id.astype(str))
    prior_articles=set(prior.article_id.fillna("").astype(str))-{""}
    prior_urls=set(prior.url.fillna("").astype(str))-{""}
    prior_accessions=set(accession_key(prior))-{""}
    mask=(~candidate.event_id.astype(str).isin(prior_ids)&
          ~candidate.article_id.fillna("").astype(str).isin(prior_articles)&
          ~candidate.url.fillna("").astype(str).isin(prior_urls)&
          ~accession_key(candidate).isin(prior_accessions)).to_numpy(bool)
    prior_times={(str(market),str(ticker)):np.sort(group.event_time_utc.astype("int64").to_numpy())
                 for (market,ticker),group in prior.groupby(["market","ticker"])}
    embargo=int(pd.Timedelta(minutes=35).value);time_rejected=0
    for position,row in enumerate(candidate.itertuples(index=False)):
        if not mask[position]:continue
        values=prior_times.get((str(row.market),str(row.ticker)))
        if values is None or not len(values):continue
        target=int(pd.Timestamp(row.event_time_utc).value);index=int(np.searchsorted(values,target))
        distances=[]
        if index<len(values):distances.append(abs(int(values[index])-target))
        if index:distances.append(abs(int(values[index-1])-target))
        if distances and min(distances)<=embargo:
            mask[position]=False;time_rejected+=1
    output=candidate.iloc[np.flatnonzero(mask)].copy()
    return output,{
        "raw":len(candidate),"outside_prior":len(output),"event_article_url_accession_rejected":
        int(len(candidate)-mask.sum()-time_rejected),"ticker_time_embargo_rejected":time_rejected,
    }


def regular_session(frame:pd.DataFrame,market:str)->pd.DataFrame:
    timezone=app.US_TZ if market=="US" else app.KR_TZ
    local=frame.event_time_utc.dt.tz_convert(timezone)
    start=pd.Timestamp("09:30").time() if market=="US" else pd.Timestamp("09:00").time()
    end=pd.Timestamp("15:27").time() if market=="US" else pd.Timestamp("14:57").time()
    return frame[(local.dt.dayofweek<5)&(local.dt.time>=start)&(local.dt.time<=end)].copy()


def space_events(frame:pd.DataFrame)->pd.DataFrame:
    accepted=[]
    group_key=article_key(frame)
    frame=frame.assign(_event_group_key=group_key)
    frame=(frame.sort_values(["ticker","event_time_utc","event_id"])
           .drop_duplicates(["ticker","_event_group_key"],keep="first"))
    for _,group in frame.groupby("ticker"):
        last=None
        for index,timestamp in zip(group.index,group.event_time_utc):
            if last is None or timestamp-last>pd.Timedelta(minutes=35):
                accepted.append(index);last=timestamp
    return frame.loc[accepted].drop(columns="_event_group_key").copy()


def cik_from_event_id(value:str)->str:
    match=re.match(r"^SEC:(\d+):",str(value))
    return str(int(match.group(1))) if match else ""


def fetch_sec_issuer(cik:str)->dict[str,Any]:
    path=CACHE/f"CIK{int(cik):010d}.json"
    if path.exists():
        try:return json.loads(path.read_text(encoding="utf-8"))
        except Exception:pass
    time.sleep(.45)
    try:
        response=requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json",
                              headers=app.SEC_HEADERS,timeout=60)
        response.raise_for_status();payload=response.json()
        result={"cik":str(int(cik)),"name":payload.get("name"),"sic":str(payload.get("sic") or ""),
                "sic_description":payload.get("sicDescription"),"status":"ok"}
    except Exception as exc:
        result={"cik":str(cik),"name":None,"sic":"","sic_description":None,
                "status":f"ERROR: {exc}"}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def collect_issuer_metadata(ciks:set[str])->dict[str,dict[str,Any]]:
    results={};workers=min(4,max(1,len(ciks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch_sec_issuer,cik):cik for cik in sorted(ciks)}
        for done,future in enumerate(as_completed(futures),1):
            result=future.result();results[str(result["cik"])]=result
            if done%50==0 or done==len(futures):
                print(f"[V36 SIC] {done}/{len(futures)}",flush=True)
    atomic_json({"selection_uses_labels":False,"approved_sic":sorted(APPROVED_HEALTHCARE_SIC),
                 "issuers":list(results.values())},DATA/"V36_SEC_ISSUER_METADATA.json")
    return results


def timing_coverage(frame:pd.DataFrame,market:str,directory:Path)->tuple[pd.DataFrame,dict[str,int]]:
    """Check cadence/timing/open existence without reading or returning outcomes."""
    cfg=replace(app.Config(),official_exact_contract=True)
    timezone=app.US_TZ if market=="US" else app.KR_TZ
    time_column="timestamp"
    eligible_rows=[];reasons=Counter()
    for ticker,group in frame.groupby("ticker",sort=True):
        path=directory/f"{ticker}.parquet"
        if not path.exists():
            reasons["NO_PRICE"]+=len(group);continue
        metadata=app.price_contract_metadata(path,market)
        if not metadata["official_exact_eligible"] or metadata["price_cadence_sec"]>60:
            reasons["PRICE_CADENCE_INELIGIBLE"]+=len(group);continue
        try:
            raw=pd.read_parquet(path,columns=[time_column,"open"])
            timestamps=pd.to_datetime(raw[time_column],utc=True,errors="coerce")
            valid=timestamps.notna()&pd.to_numeric(raw["open"],errors="coerce").notna()
            timestamps=timestamps[valid].sort_values().drop_duplicates()
            values=timestamps.astype("int64").to_numpy()
        except Exception:
            reasons["PRICE_READ_ERROR"]+=len(group);continue
        cadence=int(metadata["price_cadence_sec"]);bar_end=values+int(pd.Timedelta(seconds=cadence).value)
        for index,row in group.iterrows():
            event=pd.Timestamp(row.event_time_utc)
            entry_target=event+pd.Timedelta(minutes=cfg.entry_lag_min)
            if (not len(values) or int(entry_target.value)<int(values[0]) or
                    int(entry_target.value)>int(values[-1])):
                reasons["NO_PRICE_DATE_COVERAGE"]+=1;continue
            ei=int(np.searchsorted(values,int(entry_target.value),side="left"))
            if ei>=len(values):reasons["BAR_MISSING"]+=1;continue
            actual_entry=pd.Timestamp(values[ei],tz="UTC")
            entry_slip=float((actual_entry-entry_target).total_seconds())
            if not 0<=entry_slip<=cfg.max_entry_slippage_sec:
                reasons["ENTRY_TIMING_MISS"]+=1;continue
            exit_target=actual_entry+pd.Timedelta(minutes=cfg.horizon_min)
            xi=int(np.searchsorted(values,int(exit_target.value),side="left"))
            if xi>=len(values) or xi<=ei:reasons["BAR_MISSING"]+=1;continue
            actual_exit=pd.Timestamp(values[xi],tz="UTC")
            exit_slip=float((actual_exit-exit_target).total_seconds())
            hold=float((actual_exit-actual_entry).total_seconds())
            if not 0<=exit_slip<=cfg.max_exit_slippage_sec:
                reasons["EXIT_TIMING_MISS"]+=1;continue
            if not cfg.min_hold_seconds<=hold<=cfg.max_hold_seconds:
                reasons["HOLD_TIMING_MISS"]+=1;continue
            local_event=event.tz_convert(timezone);local_entry=actual_entry.tz_convert(timezone)
            local_exit=actual_exit.tz_convert(timezone)
            if len({local_event.date(),local_entry.date(),local_exit.date()})!=1:
                reasons["CROSSES_SESSION"]+=1;continue
            completed_before=int(np.searchsorted(bar_end,int(actual_entry.value),side="right"))
            if completed_before<2:
                reasons["NO_PREHISTORY"]+=1;continue
            record=row.to_dict();record.update({
                "price_source":metadata["price_source"],"price_cadence_sec":cadence,
                "bar_timestamp_semantics":metadata["bar_timestamp_semantics"],
                "entry_target_utc":entry_target,"actual_entry_time_utc":actual_entry,
                "entry_slippage_sec":entry_slip,"exit_target_utc":exit_target,
                "actual_exit_time_utc":actual_exit,"exit_slippage_sec":exit_slip,
                "actual_hold_seconds":hold,"timing_contract_status":"ELIGIBLE",
            })
            eligible_rows.append(record);reasons["ELIGIBLE"]+=1
    return pd.DataFrame(eligible_rows),dict(reasons)


def audited_us_candidates(prior:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    paths=sorted(DATA.glob("events*sec*.csv"))
    frames=[]
    for path in paths:
        try:
            frame=normalized_events(path);frame["raw_file"]=path.name;frames.append(frame)
        except Exception:continue
    raw=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates("event_id")
    outside,stages=outside_prior(raw,prior)
    allowed=set(app.Config().sec_forms)
    outside=outside[outside.form.fillna("").astype(str).isin(allowed)].copy()
    stages["explicit_form_allowlist"]=len(outside)
    outside=regular_session(outside,"US");stages["regular_session"]=len(outside)
    outside=outside[outside.ticker.map(lambda ticker:(PRICE/"US"/f"{ticker}.parquet").exists())].copy()
    stages["local_1m_file_exists"]=len(outside)
    outside["cik"]=outside.event_id.map(cik_from_event_id)
    metadata=collect_issuer_metadata(set(outside.cik)-{""})
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    approved_seed=set(seed.loc[seed.market.eq("US"),"ticker"].astype(str))
    audit_rows=[];health=[]
    for index,row in outside.iterrows():
        issuer=metadata.get(str(row.cik),{})
        evidence=biomedical_name_evidence(row.company)
        seed_pass=str(row.ticker) in approved_seed
        sic=str(issuer.get("sic","") or "")
        sic_pass=sic in APPROVED_HEALTHCARE_SIC
        name_pass=bool(evidence["name_rule_pass"])
        accepted=seed_pass or sic_pass or name_pass
        audit_rows.append({"ticker":str(row.ticker),"company":str(row.company),"cik":str(row.cik),
                           "sic":sic,"sic_description":issuer.get("sic_description"),
                           "seed_pass":seed_pass,"sic_pass":sic_pass,"name_pass":name_pass,
                           "accepted":accepted,"strong_matches":evidence["strong"],
                           "generic_matches":evidence["generic"]})
        if accepted:health.append(index)
    outside=outside.loc[health].copy();stages["healthcare_policy"]=len(outside)
    outside=space_events(outside);stages["canonical_35m_spaced"]=len(outside)
    covered,reasons=timing_coverage(outside,"US",PRICE/"US")
    stages["exact_timing_eligible_before_cap"]=len(covered)
    if not covered.empty:
        covered["selection_hash"]=covered.event_id.astype(str).map(
            lambda value:hashlib.sha256(f"V36|US|CAP|{value}".encode()).hexdigest())
        covered=(covered.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
                 .head(20).drop(columns="selection_hash").sort_values("event_time_utc"))
    stages["sealed_after_cap"]=len(covered)
    unique_audit={row["ticker"]:row for row in audit_rows}
    accepted_issuers=[row for row in unique_audit.values() if row["accepted"]]
    rejected_issuers=[row for row in unique_audit.values() if not row["accepted"]]
    universe_report={
        "selection_uses_labels":False,"policy":"seed OR approved healthcare SIC OR boundary-safe biomedical name",
        "approved_sic":sorted(APPROVED_HEALTHCARE_SIC),"candidate_issuers":len(unique_audit),
        "accepted_issuers":len(accepted_issuers),"rejected_issuers":len(rejected_issuers),
        "industry_counts":dict(Counter(str(row["sic_description"]) for row in accepted_issuers)),
        "random_positive_sample":sorted(accepted_issuers,key=lambda row:hashlib.sha256(row["ticker"].encode()).hexdigest())[:25],
        "suspicious_sample":sorted(rejected_issuers,key=lambda row:hashlib.sha256(row["ticker"].encode()).hexdigest())[:25],
        "oxy_matches":[row for row in unique_audit.values() if row["ticker"]=="OXY"],
        "false_positive_check_pass":not any(row["ticker"]=="OXY" and row["accepted"] for row in unique_audit.values()),
    }
    atomic_json(universe_report,DATA/"V36_UNIVERSE_AUDIT.json")
    return covered,{"raw_files":[path.name for path in paths],"stages":stages,
                    "timing_reasons":reasons,"universe":universe_report}


def audited_kr_candidates(prior:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    raw=pd.concat([normalized_events(DATA/"events_naver_kr_v35_raw.csv"),
                   normalized_events(DATA/"events_kind_real.csv")],ignore_index=True,sort=False)
    raw=raw.drop_duplicates("event_id")
    outside,stages=outside_prior(raw,prior)
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    approved=set(seed.loc[seed.market.eq("KR"),"ticker"].astype(str).str.zfill(6))
    outside=outside[outside.ticker.isin(approved)].copy();stages["healthcare_seed_policy"]=len(outside)
    outside=outside[(outside.event_time_utc>=pd.Timestamp("2026-07-28",tz="UTC"))&
                    (outside.event_time_utc<pd.Timestamp("2026-08-27",tz="UTC"))].copy()
    stages["yahoo_1m_window"]=len(outside)
    outside=regular_session(outside,"KR");stages["regular_session"]=len(outside)
    outside=space_events(outside);stages["canonical_35m_spaced"]=len(outside)
    covered,reasons=timing_coverage(outside,"KR",PRICE/"KR_V36_1M")
    stages["exact_timing_eligible_before_cap"]=len(covered)
    if not covered.empty:
        covered["selection_hash"]=covered.event_id.astype(str).map(
            lambda value:hashlib.sha256(f"V36|KR|CAP|{value}".encode()).hexdigest())
        covered=(covered.sort_values(["ticker","selection_hash"]).groupby("ticker",group_keys=False)
                 .head(15).drop(columns="selection_hash").sort_values("event_time_utc"))
    stages["sealed_after_cap"]=len(covered)
    return covered,{"stages":stages,"timing_reasons":reasons}


def freeze_sources()->dict[str,Any]:
    if (ROOT/"cert"/"V36"/"SEALED_CERTIFICATE.json").exists():
        raise RuntimeError("V36 already opened; source rebuild forbidden")
    prior=normalized_events(DATA/"events_exact_v35.csv.gz")
    us,us_report=audited_us_candidates(prior)
    kr,kr_report=audited_kr_candidates(prior)
    counts={"US":len(us),"KR":len(kr),"total":len(us)+len(kr)}
    audit={
        "selection_uses_labels":False,"contract_version":"EXACT_T2_30M_V36",
        "prior_version":"V35","embargo_minutes":35,"sec_form_policy":"SEC_FORM_POLICY_V36.json",
        "us_definition":"unused exact SEC; explicit form allowlist; healthcare seed/SIC/boundary policy; local pinned 1m; max20/ticker",
        "kr_definition":"unused exact Naver/KIND; KR healthcare seed; Yahoo real 1m; max15/ticker",
        "US":us_report,"KR":kr_report,"seal_counts":counts,
    }
    if counts["US"]<250 or counts["KR"]<100 or counts["total"]<500:
        audit["status"]="BLOCKED_SOURCE_MINIMUM"
        atomic_json(audit,DATA/"V36_EXACT_SOURCE_CANDIDATE_AUDIT.json")
        raise RuntimeError(f"exact source minimum failed: {counts}")
    for market,frame in (("US",us),("KR",kr)):
        forbidden=FORBIDDEN_OUTCOME_COLUMNS&set(frame.columns)
        if forbidden:raise RuntimeError(f"outcome columns entered source selection: {forbidden}")
        output=frame.copy()
        output["source_family"]=output.apply(app.source_family_for_event,axis=1)
        output["event_group_id"]=output.apply(app.canonical_event_group_id,axis=1)
        output["source"]=f"{market}_EXACT_V36_SEAL"
        output["contract_version"]="EXACT_T2_30M_V36"
        path=DATA/f"events_{market.lower()}_exact_v36_seal.csv"
        atomic_csv(output,path)
        audit.setdefault("source_sha256",{})[market]=sha256(path)
    us_ids=set(us.event_id.astype(str));kr_ids=set(kr.event_id.astype(str))
    audit.update({"status":"FROZEN_LABEL_BLIND","prior_event_id_overlap":
                  len((us_ids|kr_ids)&set(prior.event_id.astype(str))),
                  "source_frozen_at":pd.Timestamp.now(tz="UTC")})
    atomic_json(audit,DATA/"V36_SOURCE_STATUS.json")
    print(json.dumps({"status":audit["status"],"counts":counts,
                      "source_sha256":audit["source_sha256"]},indent=2))
    return audit


def build_dev_contract()->pd.DataFrame:
    """Relabel only opened V35-and-earlier metadata under the repaired contract."""
    opened=normalized_events(DATA/"events_exact_v35.csv.gz")
    opened["source_family"]=opened.apply(app.source_family_for_event,axis=1)
    allowed=set(app.Config().sec_forms)
    sec_mask=opened.source_family.eq("US_SEC")
    opened=opened[~sec_mask|opened.form.fillna("").astype(str).isin(allowed)].copy()
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    kr_seed=set(seed.loc[seed.market.eq("KR"),"ticker"].astype(str).str.zfill(6))
    opened=opened[~opened.market.eq("KR")|opened.ticker.isin(kr_seed)].copy()
    opened["event_group_id"]=opened.apply(app.canonical_event_group_id,axis=1)
    opened=opened.sort_values("event_time_utc").drop_duplicates("event_group_id",keep="first")
    cfg=replace(app.Config(),official_exact_contract=True,kr_price_dir="price_data/KR_V36_1M",
                us_price_dir="price_data/US")
    # Process one symbol at a time.  The US 1-minute store is tens of GB, so
    # retaining every symbol in PriceStore.cache would needlessly expand RAM
    # and can make an otherwise deterministic rebuild fail by allocation.
    store=app.PriceStore(cfg);rows=[];reasons=Counter();done=0
    grouped=opened.groupby(["market","ticker"],sort=True)
    for (market,ticker),events in grouped:
        bars=store.load(str(market),str(ticker))
        if bars is None or bars.empty:
            reasons["NO_PRICE"]+=len(events);done+=len(events)
        else:
            for _,event in events.iterrows():
                row,status=app.event_to_row(event,bars,cfg);reasons[status]+=1
                if row is not None:rows.append(row)
                done+=1
        store.cache.pop((str(market),str(ticker)),None)
        if done%1000<len(events):
            print(f"[V36 DEV CONTRACT] {done}/{len(opened)} usable={len(rows)}",flush=True)
    if not rows:raise RuntimeError(f"no exact-contract DEV rows: {dict(reasons)}")
    labeled=pd.DataFrame(rows).sort_values("event_time_utc").reset_index(drop=True)
    labeled=app.add_benchmark_context(labeled,store)
    event_columns=[column for column in opened.columns if column not in FORBIDDEN_OUTCOME_COLUMNS]
    opened[event_columns].to_csv(DATA/"dev_contract_v36_events.csv.gz",index=False,compression="gzip",encoding="utf-8")
    labeled.to_csv(DATA/"dev_contract_v36_labeled.csv.gz",index=False,compression="gzip",encoding="utf-8")
    timing={}
    for (market,source_family),group in labeled.groupby(["market","source_family"]):
        timing[f"{market}|{source_family}"]={
            "n":len(group),"entry_slippage_p50":float(group.entry_slippage_sec.quantile(.5)),
            "entry_slippage_p95":float(group.entry_slippage_sec.quantile(.95)),
            "entry_slippage_max":float(group.entry_slippage_sec.max()),
            "exit_slippage_p95":float(group.exit_slippage_sec.quantile(.95)),
            "hold_seconds_min":float(group.actual_hold_seconds.min()),
            "hold_seconds_max":float(group.actual_hold_seconds.max()),
        }
    status={
        "contract_version":"EXACT_T2_30M_V36","old_labels_reused":False,
        "opened_metadata_source":"events_exact_v35.csv.gz","selection_uses_seal_labels":False,
        "candidate_opened_events":len(opened),"labeled_rows":len(labeled),
        "by_market_source":{f"{market}|{source}":int(count) for (market,source),count in
                            labeled.groupby(["market","source_family"]).size().items()},
        "rejection_reasons":dict(reasons),"timing":timing,
        "events_sha256":sha256(DATA/"dev_contract_v36_events.csv.gz"),
        "labeled_sha256":sha256(DATA/"dev_contract_v36_labeled.csv.gz"),
    }
    atomic_json(status,DATA/"DEV_CONTRACT_V36_STATUS.json")
    print(json.dumps({"labeled_rows":len(labeled),"by_market_source":status["by_market_source"],
                      "rejections":dict(reasons)},indent=2))
    return labeled


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-sources",action="store_true")
    parser.add_argument("--build-dev-contract",action="store_true")
    args=parser.parse_args()
    if args.freeze_sources:freeze_sources()
    if args.build_dev_contract:build_dev_contract()
    if not (args.freeze_sources or args.build_dev_contract):parser.print_help()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
