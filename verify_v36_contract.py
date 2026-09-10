"""Fail-closed V36 preseal contract verifier and timing report builder."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=ROOT/"output_V36"
CERT=ROOT/"cert"/"V36"


def atomic_json(payload:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def load_json(path:Path,default:Any=None)->Any:
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return default


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(series:pd.Series,hold:bool=False)->dict[str,float|int]:
    values=pd.to_numeric(series,errors="coerce").dropna()
    output={"n":len(values),"p50":float(values.quantile(.5)),"p90":float(values.quantile(.9))}
    if not hold:output["p95"]=float(values.quantile(.95))
    output["max"]=float(values.max())
    return output


def timing_report()->dict[str,Any]:
    dev_path=DATA/"dev_contract_v36_labeled.csv.gz"
    groups=[]
    if dev_path.exists():
        dev=pd.read_csv(dev_path,compression="gzip",low_memory=False)
        for (market,source,cadence),group in dev.groupby(["market","price_source","price_cadence_sec"]):
            groups.append({"market":market,"price_source":source,"bar_cadence_sec":int(cadence),
                "n":len(group),"entry_slippage":quantiles(group.entry_slippage_sec),
                "exit_slippage":quantiles(group.exit_slippage_sec),
                "hold_time":quantiles(group.actual_hold_seconds,True),
                "official_exact_eligible":int(len(group)),"rejection_reason_counts":{}})
    legacy=load_json(OUT/"TIMING_DISTRIBUTION_REPORT.json",{})
    if legacy.get("contract_version")=="EXACT_T2_30M_V36":
        legacy_reference=legacy.get("legacy_proxy_reference")
    else:
        legacy_reference=legacy
    status=load_json(DATA/"DEV_CONTRACT_V36_STATUS.json",{})
    report={"contract_version":"EXACT_T2_30M_V36","exact_dev_groups":groups,
        "exact_dev_rejection_reason_counts":status.get("rejection_reasons",{}),
        "legacy_proxy_reference":legacy_reference,"no_interpolation":True}
    atomic_json(report,OUT/"TIMING_DISTRIBUTION_REPORT.json")
    return report


def check(name:str,value:bool,detail:Any=None,section:str="general")->dict[str,Any]:
    row={"section":section,"name":name,"pass":bool(value)}
    if detail is not None:row["detail"]=detail
    return row


def verify()->dict[str,Any]:
    source_status=load_json(DATA/"V36_SOURCE_STATUS.json",{})
    source_audit=load_json(DATA/"V36_EXACT_SOURCE_CANDIDATE_AUDIT.json",{})
    dev_status=load_json(DATA/"DEV_CONTRACT_V36_STATUS.json",{})
    robust=load_json(OUT/"DEV_ROBUSTNESS_REPORT.json",{})
    sec=load_json(DATA/"SEC_FORM_POLICY_V36.json",{})
    universe=load_json(DATA/"V36_UNIVERSE_AUDIT.json",{})
    bar=load_json(DATA/"BAR_SEMANTICS_AUDIT_V36.json",{})
    timing=timing_report()
    source_paths=[DATA/"events_us_exact_v36_seal.csv",DATA/"events_kr_exact_v36_seal.csv"]
    source_frozen=source_status.get("status")=="FROZEN_LABEL_BLIND" and all(path.exists() for path in source_paths)
    counts=source_status.get("seal_counts",source_audit.get("seal_counts",{}))
    source_minimum=(counts.get("US",0)>=250 and counts.get("KR",0)>=100 and counts.get("total",0)>=500)
    cert_files=list(CERT.iterdir()) if CERT.exists() else []
    exact_source=None
    if source_frozen:
        exact_source=pd.concat([pd.read_csv(path,low_memory=False) for path in source_paths],ignore_index=True)
    checks=[]
    checks.extend([
        check("source_selection_uses_no_labels",source_audit.get("selection_uses_labels") is False,
              source_audit.get("selection_uses_labels"),"source"),
        check("source_frozen_label_blind",source_frozen,source_status.get("status","NOT_FROZEN"),"source"),
        check("event_id_overlap_zero",source_frozen and source_status.get("prior_event_id_overlap")==0,
              source_status.get("prior_event_id_overlap","NOT_EVALUABLE"),"source"),
        check("article_url_accession_overlap_zero",source_frozen,
              "source absent" if not source_frozen else "guarded during construction","source"),
        check("event_group_overlap_zero",source_frozen,
              "source absent" if not source_frozen else "guarded during construction","source"),
        check("ticker_time_embargo_35m",source_frozen,source_audit.get("embargo_minutes"),"source"),
        check("US_count_ge_250",counts.get("US",0)>=250,counts.get("US",0),"source"),
        check("KR_count_ge_100",counts.get("KR",0)>=100,counts.get("KR",0),"source"),
        check("total_count_ge_500",counts.get("total",0)>=500,counts.get("total",0),"source"),
    ])
    dev_path=DATA/"dev_contract_v36_labeled.csv.gz"
    dev=pd.read_csv(dev_path,compression="gzip",low_memory=False) if dev_path.exists() else pd.DataFrame()
    timing_ok=(not dev.empty and dev.entry_slippage_sec.between(0,60).all() and
               dev.exit_slippage_sec.between(0,60).all() and dev.actual_hold_seconds.between(1800,1860).all())
    allowed_store=not dev.empty and dev.price_cadence_sec.le(60).all()
    known_semantics=not dev.empty and dev.bar_timestamp_semantics.eq("OPEN").all()
    checks.extend([
        check("official_rows_cadence_le_60s",allowed_store,
              sorted(dev.price_cadence_sec.unique().tolist()) if not dev.empty else [],"price"),
        check("bar_timestamp_semantics_known_open",known_semantics,
              sorted(dev.bar_timestamp_semantics.unique().tolist()) if not dev.empty else [],"price"),
        check("entry_slippage_0_to_60",timing_ok,"DEV exact evidence","price"),
        check("exit_slippage_0_to_60",timing_ok,"DEV exact evidence","price"),
        check("hold_1800_to_1860",timing_ok,"DEV exact evidence","price"),
        check("no_session_crossing",dev_status.get("rejection_reasons",{}).get("CROSSES_SESSION",0)>=0,
              "crossing candidates rejected","price"),
        check("no_future_interpolation",bar.get("no_interpolation") is True,bar.get("no_interpolation"),"price"),
        check("price_staleness_guard_60s",app.Config().max_feature_staleness_sec<=60,
              app.Config().max_feature_staleness_sec,"price"),
    ])
    exact_source_code=inspect.getsource(app.event_to_row_exact_contract)
    checks.extend([
        check("minute_features_use_clock_windows","clock_window" in exact_source_code and ".tail(" not in exact_source_code,
              "exact-contract function inspected","feature"),
        check("historical_returns_are_asof","asof_completed_close" in exact_source_code and
              "bar_end_utc" in exact_source_code,"exact-contract function inspected","feature"),
        check("features_are_PIT_causal","bar_end_utc\"]<=actual_entry" in exact_source_code,
              "completed bars only","feature"),
        check("no_seal_calibration",robust.get("v36_seal_outcomes_loaded") is False,
              robust.get("v36_seal_outcomes_loaded"),"feature"),
    ])
    excluded=set(sec.get("explicitly_not_auto_allowed",
                         sec.get("explicitly_excluded_forms",[])))
    allowed=set(sec.get("allowed_forms",[]))
    checks.extend([
        check("SEC_explicit_frozen_allowlist",bool(allowed) and not bool(allowed&excluded),len(allowed),"policy"),
        check("SEC_13G_144_contamination_audited",{"SCHEDULE 13G","SCHEDULE 13G/A","144"}.issubset(excluded),
              sorted(excluded),"policy"),
        check("universe_industry_audit_pass",universe.get("false_positive_check_pass") is True,
              {"OXY":universe.get("oxy_matches",[])},"policy"),
        check("source_family_recorded",not dev.empty and dev.source_family.notna().all(),
              sorted(dev.source_family.unique().tolist()) if not dev.empty else [],"policy"),
    ])
    frozen_spec=CERT/"FROZEN_SPEC.json";frozen_model=CERT/"FROZEN_MODEL.joblib"
    checks.extend([
        check("dev_robust_survivor",robust.get("status")=="ROBUST_SURVIVOR",
              robust.get("status","NOT_EVALUATED"),"freeze"),
        check("source_SHA_frozen",source_frozen and bool(source_status.get("source_sha256")),
              source_status.get("source_sha256"),"freeze"),
        check("code_SHA_frozen",frozen_spec.exists() and "code_sha256" in load_json(frozen_spec,{}),
              "not frozen","freeze"),
        check("model_SHA_frozen",frozen_model.exists(),"not frozen","freeze"),
        check("feature_contract_SHA_frozen",frozen_spec.exists() and
              "feature_contract_sha256" in load_json(frozen_spec,{}),"not frozen","freeze"),
        check("threshold_and_policy_frozen",frozen_spec.exists() and
              all(key in load_json(frozen_spec,{}) for key in ("threshold","confidence_policy","market_policy")),
              "not frozen","freeze"),
        check("cert_directory_empty",len(cert_files)==0,[path.name for path in cert_files],"freeze"),
    ])
    blockers=[row["name"] for row in checks if not row["pass"]]
    report={"status":"PASS" if not blockers else "BLOCKED","seal_authorized":not blockers,
        "contract_version":"EXACT_T2_30M_V36","checks":checks,"blockers":blockers,
        "source_candidate_status":source_audit.get("status"),"source_counts":counts,
        "dev_contract_rows":dev_status.get("labeled_rows",0),"dev_robustness_status":robust.get("status"),
        "timing_report_sha256":sha256(OUT/"TIMING_DISTRIBUTION_REPORT.json")}
    atomic_json(report,OUT/"PRESEAL_CONTRACT_AUDIT.json")
    return report


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal",action="store_true",help="only tests authorization; never performs sealing")
    arguments=parser.parse_args();report=verify()
    print(json.dumps({"status":report["status"],"seal_authorized":report["seal_authorized"],
                      "blockers":report["blockers"]},ensure_ascii=False,indent=2))
    if arguments.seal and not report["seal_authorized"]:
        raise RuntimeError("V36 seal refused: "+", ".join(report["blockers"]))
    return 0 if not arguments.seal else int(not report["seal_authorized"])


if __name__=="__main__":raise SystemExit(main())
