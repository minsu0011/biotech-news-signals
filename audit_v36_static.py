"""Static, label-safe V36 contract audit and immutable evidence summary."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
PRICE=ROOT/"price_data"
OUT=ROOT/"output_V36"
OUT.mkdir(parents=True,exist_ok=True)

PRE_REPAIR_HASHES={
    "bio_news_30m_v3.py":"3c86886e4d3f58e37e9fa96624abddc1ce5ed2cc8505a87d8973514bd814e5f3",
    "prepare_real_data.py":"866a6ca7a4111ff5ef9b13293be7407d6bea6119e6b2673e54a5ef51c0dd9fa2",
    "runtime_limits.py":"1b5d943cafb981e82a625139a2d57a3132189f0a0191573ce4f7c795f6a2dfab",
    "data/events_sec_us_v36_raw.csv":"c6208a94b9e81aab8d17d9fd8ab036eaee0a534b88c0016bd02aaecbd19e9629",
    "data/universe_v36_us_sec.csv":"bd76080ba7ae003a81da3fbd2354c861777e88b1b1182839a59701a11a018332",
    "data/V36_US_SEC_STATUS.json":"a8b228e9f588262b6f5f3a25eb8b8fa75cde249b5348114c39611e04e001b893",
    "data/events_exact_v35.csv.gz":"a7c489eb5afe0e335f1f7c46e6faae612e7a856d3c77bbb710fcd111d2b29c02",
    "data/labeled_events_v35.csv.gz":"84d7b08b9b3ef7935d528c44352256c01fc8d5547b215cf7466b22798647901b",
    "data/labeled_events_v35_new.csv.gz":"e863b0ec7ecc1ec4cd85e1d4494e99cb7930acba0ad1307175c76d55a589fee7",
    "data/events_kind_real.csv":"db078f4933d5963bde5e316f13a93a19f7913fbef5ddd32e11a65447b9893e96",
}


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def initial_state()->dict[str,Any]:
    current={}
    for relative,baseline in PRE_REPAIR_HASHES.items():
        path=ROOT/relative
        current[relative]={
            "exists":path.exists(),"pre_repair_sha256":baseline,
            "current_sha256":sha256(path) if path.exists() else None,
            "unchanged_from_pre_repair":path.exists() and sha256(path)==baseline,
        }
    v36_paths=[
        "cert/V36","data/V36_SOURCE_STATUS.json","data/V36_LABEL_STATUS.json",
        "data/labeled_events_v36.csv.gz","data/labeled_events_v36_new.csv.gz",
        "data/events_us_exact_v36_seal.csv","data/events_kr_exact_v36_seal.csv",
    ]
    state={
        "v36_opened":(ROOT/"cert/V36"/"SEALED_CERTIFICATE.json").exists(),
        "v36_frozen_spec":(ROOT/"cert/V36"/"FROZEN_SPEC.json").exists(),
        "v36_frozen_model":(ROOT/"cert/V36"/"FROZEN_MODEL.joblib").exists(),
        "path_state":{name:(ROOT/name).exists() for name in v36_paths},
        "hashes":current,
        "immutable_latest":"V35 FAIL",
    }
    atomic_json(state,DATA/"V36_PRE_REPAIR_STATE.json")
    return state


def certification_summary()->pd.DataFrame:
    rows=[]
    candidates=[ROOT/"cert"/"SEALED_CERTIFICATE.json"]
    candidates.extend(sorted((ROOT/"cert").glob("V*/SEALED_CERTIFICATE.json")))
    for path in candidates:
        payload=json.loads(path.read_text(encoding="utf-8"))
        metrics=payload.get("metrics",{});markets=metrics.get("by_market",{})
        bootstrap=payload.get("bootstrap",{})
        version=int(str(payload.get("version","0")).split(".")[0])
        rows.append({
            "version":f"V{version}","status":payload.get("status"),"certificate_path":str(path.relative_to(ROOT)),
            "n":metrics.get("n"),"us_n":markets.get("US",{}).get("n"),"kr_n":markets.get("KR",{}).get("n"),
            "balanced_accuracy":metrics.get("balanced_accuracy"),"auc":metrics.get("auc"),
            "edge_vs_naive":metrics.get("edge_vs_naive"),"highconf_coverage":metrics.get("highconf_coverage"),
            "highconf_accuracy":metrics.get("highconf_accuracy"),
            "highconf_strategy_net":metrics.get("strategy_mean_signed_net"),
            "bootstrap_bal_lower95":bootstrap.get("balanced_accuracy_lower95"),
            "us_balanced_accuracy":markets.get("US",{}).get("balanced_accuracy"),
            "kr_balanced_accuracy":markets.get("KR",{}).get("balanced_accuracy"),
            "certificate_sha256":payload.get("certificate_sha256"),
        })
    frame=pd.DataFrame(rows).sort_values("version",key=lambda x:x.str[1:].astype(int))
    frame.to_csv(OUT/"CERTIFICATION_SUMMARY_V5_V35.csv",index=False,encoding="utf-8-sig")
    return frame


def sample_bar_file(path:Path,market:str)->dict[str,Any]:
    try:
        frame=pd.read_parquet(path)
        column="timestamp" if "timestamp" in frame else "datetime"
        times=pd.to_datetime(frame[column],errors="coerce",utc=(market=="US")).dropna().sort_values()
        differences=times.diff().dt.total_seconds().dropna()
        intraday=differences[(differences>0)&(differences<=3600)]
        return {
            "file":str(path.relative_to(ROOT)),"rows":len(frame),"datetime_column":column,
            "dtype":str(frame[column].dtype),"min":times.min(),"max":times.max(),
            "median_positive_intraday_gap_sec":float(intraday.median()) if len(intraday) else None,
            "p90_positive_intraday_gap_sec":float(intraday.quantile(.90)) if len(intraday) else None,
            "columns":list(frame.columns),
        }
    except Exception as exc:
        return {"file":str(path.relative_to(ROOT)),"error":str(exc)}


def bar_semantics_audit()->dict[str,Any]:
    definitions={
        "price_data/US":{
            "source":"ggaddam/OHLCV-1m; originally Finnhub",
            "documented_cadence_sec":60,"timestamp_semantics":"OPEN","timezone":"UTC",
            "official_exact_eligible":True,
            "evidence":"dataset card defines timestamp as start time of the minute",
            "source_url":"https://huggingface.co/datasets/ggaddam/OHLCV-1m",
        },
        "price_data/US_V34":{
            "source":"Yahoo chart API","documented_cadence_sec":300,"timestamp_semantics":"OPEN",
            "timezone":"UTC","official_exact_eligible":False,"classification":"PROXY_5M",
        },
        "price_data/US_V35":{
            "source":"Yahoo chart API","documented_cadence_sec":300,"timestamp_semantics":"OPEN",
            "timezone":"UTC","official_exact_eligible":False,"classification":"PROXY_5M",
        },
        "price_data/KR":{
            "source":"Daum Finance /5/minutes endpoint","documented_cadence_sec":300,
            "timestamp_semantics":"OPEN","timezone":"Asia/Seoul naive in file",
            "official_exact_eligible":False,"classification":"PROXY_5M",
        },
        "price_data/KR_V36_1M":{
            "source":"Yahoo chart API interval=1m","documented_cadence_sec":60,
            "timestamp_semantics":"OPEN","timezone":"UTC with exchangeTimezoneName=Asia/Seoul",
            "official_exact_eligible":True,"classification":"EXACT_T2_30M_OFFICIAL_CANDIDATE",
            "evidence":"direct chart response interval=1m; conservative execution uses candle open only",
        },
    }
    stores=[]
    for relative,definition in definitions.items():
        directory=ROOT/relative
        files=sorted(directory.glob("*.parquet")) if directory.exists() else []
        sample=[]
        if files:
            indices=sorted(set([0,len(files)//2,len(files)-1]))
            sample=[sample_bar_file(files[index],"KR" if "/KR" in relative else "US") for index in indices]
        stores.append({"directory":relative,"exists":directory.exists(),"file_count":len(files),
                       "definition":definition,"samples":sample})
    audit={
        "contract_version":"EXACT_T2_30M_V36","selection_uses_labels":False,
        "no_interpolation":True,"official_max_cadence_sec":60,
        "entry_price_rule":"OPEN of first audited <=1m bar at/after event+2m; slippage 0..60 sec",
        "exit_price_rule":"OPEN of first audited <=1m bar at/after actual entry+30m; slippage 0..60 sec",
        "stores":stores,
    }
    atomic_json(audit,DATA/"BAR_SEMANTICS_AUDIT_V36.json")
    return audit


def sec_form_policy()->dict[str,Any]:
    allow=list(app.Config().sec_forms)
    raw=pd.read_csv(DATA/"events_sec_us_v36_raw.csv",dtype={"ticker":str},low_memory=False)
    counts=Counter(raw.get("form",pd.Series(dtype=str)).fillna("").astype(str))
    forbidden={form:counts.get(form,0) for form in ("SCHEDULE 13G","SCHEDULE 13G/A","144")}
    policy={
        "version":"SEC_FORM_POLICY_V36","selection_uses_labels":False,
        "basis":"original explicit Config.sec_forms allowlist; no all-forms path",
        "allowed_forms":allow,"allowed_forms_sha256":hashlib.sha256("\n".join(allow).encode()).hexdigest(),
        "explicitly_not_auto_allowed":["SCHEDULE 13G","SCHEDULE 13G/A","144"],
        "v36_exploratory_raw_form_counts":dict(counts.most_common()),
        "v36_exploratory_13g_144_count":sum(forbidden.values()),
        "v36_exploratory_13g_144_breakdown":forbidden,
        "official_rule":"form must be exactly in allowed_forms before price coverage or source freeze",
    }
    atomic_json(policy,DATA/"SEC_FORM_POLICY_V36.json")
    return policy


STRONG_BIOMEDICAL=[
    r"\btherapeutics?\b",r"\bbiotherapeutics?\b",r"\bbiopharma(?:ceuticals?)?\b",
    r"\bpharmaceuticals?\b",r"\bbiotechnology\b",r"\bbiosciences?\b",
    r"\boncology\b",r"\bdiagnostics?\b",r"\bgenomics?\b",r"\bimmunotherap(?:y|ies)\b",
    r"\bmedical devices?\b",r"\blife sciences?\b",
]
GENERIC_BIOMEDICAL=[
    r"\bhealthcare\b",r"\bhealth\b",r"\bmedical\b",r"\bclinical\b",
    r"\blaborator(?:y|ies)\b",r"\bdental\b",r"\bneuro(?:science)?\b",
]


def biomedical_name_evidence(company:str)->dict[str,Any]:
    value=str(company or "").lower()
    strong=[pattern for pattern in STRONG_BIOMEDICAL if re.search(pattern,value,re.I)]
    generic=[pattern for pattern in GENERIC_BIOMEDICAL if re.search(pattern,value,re.I)]
    return {"strong":strong,"generic":generic,"name_rule_pass":bool(strong) or len(generic)>=2}


def universe_audit()->dict[str,Any]:
    exploratory=pd.read_csv(DATA/"universe_v36_us_sec.csv",dtype={"ticker":str,"cik":str})
    seed=pd.read_csv(ROOT/"universe_seed.csv",dtype={"ticker":str})
    approved=set(seed.loc[seed.market.eq("US"),"ticker"].astype(str))
    rows=[]
    for item in exploratory.itertuples(index=False):
        evidence=biomedical_name_evidence(item.company)
        seed_pass=str(item.ticker) in approved
        eligible=seed_pass or evidence["name_rule_pass"]
        rows.append({
            "ticker":str(item.ticker),"company":str(item.company),"cik":str(item.cik),
            "seed_approved":seed_pass,"strong_matches":evidence["strong"],
            "generic_matches":evidence["generic"],"name_rule_pass":evidence["name_rule_pass"],
            "provisional_eligible_without_sic":eligible,
        })
    accepted=[row for row in rows if row["provisional_eligible_without_sic"]]
    rejected=[row for row in rows if not row["provisional_eligible_without_sic"]]
    accepted_sample=sorted(accepted,key=lambda row:hashlib.sha256(row["ticker"].encode()).hexdigest())[:20]
    suspicious=sorted(rejected,key=lambda row:hashlib.sha256(row["ticker"].encode()).hexdigest())[:20]
    oxy=[row for row in rows if row["ticker"]=="OXY"]
    term_counts=Counter()
    for row in rows:
        for pattern in row["strong_matches"]+row["generic_matches"]:
            term_counts[pattern]+=1
    audit={
        "version":"V36_UNIVERSE_AUDIT","selection_uses_labels":False,
        "official_policy":"approved healthcare seed OR approved healthcare SIC OR boundary-safe strong biomedical name; generic names require >=2 independent matches",
        "substring_matching_forbidden":True,"sic_validation_pending":True,
        "exploratory_rows":len(rows),"provisional_eligible_without_sic":len(accepted),
        "provisional_rejected_without_sic":len(rejected),"term_counts":dict(term_counts),
        "random_positive_sample":accepted_sample,"suspicious_sample":suspicious,
        "oxy_check":oxy,"oxy_false_positive_count":sum(row["provisional_eligible_without_sic"] for row in oxy),
        "status":"PASS_PROVISIONAL" if oxy and not oxy[0]["provisional_eligible_without_sic"] else "FAIL",
        "rows":rows,
    }
    atomic_json(audit,DATA/"V36_UNIVERSE_AUDIT.json")
    return audit


def cache_compatibility()->dict[str,Any]:
    groups=[
        {"path":"cache/hf_months","classification":"COMPATIBLE_RAW_US_1M","reason":"raw pinned one-minute bars; relabel/re-feature required"},
        {"path":"cache/hf_filtered","classification":"COMPATIBLE_RAW_US_1M","reason":"ticker-filtered raw one-minute bars; verify source month hash"},
        {"path":"cache/sec_events_by_ticker","classification":"COMPATIBLE_RAW_EVENT_METADATA","reason":"exact SEC metadata; apply V36 source guards"},
        {"path":"cache/sec_text","classification":"COMPATIBLE_RAW_TEXT","reason":"raw filing text only"},
        {"path":"cache/kind_days","classification":"COMPATIBLE_RAW_EVENT_METADATA","reason":"raw exact KIND metadata"},
        {"path":"cache/naver_news_by_ticker","classification":"COMPATIBLE_RAW_EVENT_METADATA","reason":"raw news metadata; overlap guard required"},
        {"path":"price_data/US","classification":"COMPATIBLE_RAW_US_1M","reason":"open-stamped 1m source"},
        {"path":"price_data/KR","classification":"DEV_PROXY_5M_NOT_OFFICIAL","reason":"Daum five-minute candles"},
        {"path":"price_data/US_V34","classification":"DEV_PROXY_5M_NOT_OFFICIAL","reason":"Yahoo five-minute candles"},
        {"path":"price_data/US_V35","classification":"DEV_PROXY_5M_NOT_OFFICIAL","reason":"Yahoo five-minute candles"},
        {"path":"data/labeled_events_v35.csv.gz","classification":"INCOMPATIBLE_LABEL_FEATURE","reason":"legacy timing and row-count features"},
        {"path":"data/labeled_events_v35_new.csv.gz","classification":"INCOMPATIBLE_LABEL_FEATURE","reason":"legacy five-minute timing contract"},
        {"path":"cache/v35_*","classification":"INCOMPATIBLE_MODEL_FEATURE_CACHE","reason":"selected on legacy labels/features"},
    ]
    for group in groups:
        path=ROOT/group["path"]
        if "*" in group["path"]:
            files=list(path.parent.glob(path.name));group["exists"]=bool(files)
            group["file_count"]=len(files);group["bytes"]=sum(item.stat().st_size for item in files)
        elif path.is_dir():
            files=list(path.rglob("*"));files=[item for item in files if item.is_file()]
            group["exists"]=True;group["file_count"]=len(files);group["bytes"]=sum(item.stat().st_size for item in files)
        else:
            group["exists"]=path.exists();group["file_count"]=int(path.exists())
            group["bytes"]=path.stat().st_size if path.exists() else 0
    report={"contract_version":"EXACT_T2_30M_V36","groups":groups,
            "policy":"incompatible caches are preserved but never loaded by the exact-contract pipeline"}
    atomic_json(report,OUT/"CACHE_COMPATIBILITY_REPORT.json")
    return report


def describe_seconds(values:pd.Series)->dict[str,Any]:
    clean=pd.to_numeric(values,errors="coerce").dropna()
    return {"n":len(clean),"p50":float(clean.quantile(.50)) if len(clean) else None,
            "p90":float(clean.quantile(.90)) if len(clean) else None,
            "p95":float(clean.quantile(.95)) if len(clean) else None,
            "max":float(clean.max()) if len(clean) else None}


def legacy_timing_report()->dict[str,Any]:
    frame=pd.read_csv(DATA/"labeled_events_v35_new.csv.gz",dtype={"ticker":str},low_memory=False)
    event=pd.to_datetime(frame["event_time_utc"],utc=True,errors="coerce")
    entry=pd.to_datetime(frame["entry_time_utc"],utc=True,errors="coerce")
    exit_=pd.to_datetime(frame["exit_time_utc"],utc=True,errors="coerce")
    frame["entry_slippage_sec"]=(entry-(event+pd.Timedelta(minutes=2))).dt.total_seconds()
    frame["exit_slippage_sec"]=(exit_-(entry+pd.Timedelta(minutes=30))).dt.total_seconds()
    frame["actual_hold_seconds"]=(exit_-entry).dt.total_seconds()
    groups=[]
    for market,group in frame.groupby("market"):
        source="YAHOO_CHART_US_5M" if market=="US" else "DAUM_FINANCE_KR_5M"
        groups.append({
            "market":market,"price_source":source,"bar_cadence_sec":300,"n":len(group),
            "entry_slippage":describe_seconds(group["entry_slippage_sec"]),
            "exit_slippage":describe_seconds(group["exit_slippage_sec"]),
            "hold_time":describe_seconds(group["actual_hold_seconds"]),
            "official_exact_eligible":0,"rejection_reason_counts":{"PRICE_CADENCE_INELIGIBLE":len(group)},
        })
    report={
        "legacy_dataset":"labeled_events_v35_new.csv.gz","legacy_rows":len(frame),
        "contract_assessment":"PROXY_5M_NOT_EXACT_T2_CERTIFICATION",
        "groups":groups,"no_interpolation":True,
    }
    atomic_json(report,OUT/"TIMING_DISTRIBUTION_REPORT.json")
    return report


def initial_preseal_audit(state:dict[str,Any],universe:dict[str,Any])->dict[str,Any]:
    checks=[
        {"name":"v36_unopened","pass":not state["v36_opened"]},
        {"name":"v36_source_not_frozen","pass":not state["path_state"]["data/V36_SOURCE_STATUS.json"]},
        {"name":"v36_labels_absent","pass":not state["path_state"]["data/V36_LABEL_STATUS.json"]},
        {"name":"five_minute_not_official","pass":True},
        {"name":"sec_explicit_allowlist","pass":True},
        {"name":"oxy_excluded_by_boundary_policy","pass":universe["oxy_false_positive_count"]==0},
        {"name":"exact_source_minimums","pass":False,"detail":"not yet constructed"},
        {"name":"dev_robust_survivor","pass":False,"detail":"not yet evaluated"},
        {"name":"freeze_integrity","pass":False,"detail":"not frozen"},
    ]
    report={"status":"PASS" if all(item["pass"] for item in checks) else "BLOCKED",
            "seal_authorized":False,"checks":checks,
            "blockers":[item["name"] for item in checks if not item["pass"]]}
    atomic_json(report,OUT/"PRESEAL_CONTRACT_AUDIT.json")
    return report


def main()->None:
    state=initial_state()
    summary=certification_summary()
    bars=bar_semantics_audit()
    forms=sec_form_policy()
    universe=universe_audit()
    cache=cache_compatibility()
    timing=legacy_timing_report()
    preseal=initial_preseal_audit(state,universe)
    print(json.dumps({
        "v36_opened":state["v36_opened"],"certified_versions":len(summary),
        "bar_stores":len(bars["stores"]),"sec_allowlist":len(forms["allowed_forms"]),
        "universe_status":universe["status"],"cache_groups":len(cache["groups"]),
        "legacy_timing_rows":timing["legacy_rows"],"preseal":preseal["status"],
    },indent=2))


if __name__=="__main__":
    main()
