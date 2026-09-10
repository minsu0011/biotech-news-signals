"""Fail-closed one-shot V37+ research/final-meta seal worker.

The worker performs label-blind preflight first, creates an exclusive
OPEN_INTENT keyed by the untouched-source SHA, and only then reads outcomes.
Any post-intent crash burns that source permanently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score


ROOT=Path(__file__).resolve().parent
STATE=ROOT/"state"
CERT=ROOT/"cert"
COST=.002
FORBIDDEN_PREDICTION_COLUMNS={"y","label","fwd_ret_30m","entry_price","exit_price","outcome"}


def now()->str:return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read_json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(value:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=str);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def resolve(value:str|Path)->Path:
    path=Path(value);return path if path.is_absolute() else ROOT/path


def open_intent(source_sha:str,payload:dict[str,Any])->Path:
    directory=STATE/"seals"/source_sha
    try:directory.mkdir(parents=True,exist_ok=False)
    except FileExistsError:raise RuntimeError(f"untouched source already opened/burned: {source_sha}")
    path=directory/"OPEN_INTENT.json";flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    descriptor=os.open(path,flags)
    with os.fdopen(descriptor,"w",encoding="utf-8",newline="\n") as handle:
        json.dump(payload,handle,ensure_ascii=False,indent=2,default=str);handle.flush();os.fsync(handle.fileno())
    return directory


def prior_event_ids()->set[str]:
    used=set()
    for path in CERT.rglob("CERTIFICATE.json") if CERT.exists() else []:
        certificate=read_json(path)
        used.update(str(value) for value in certificate.get("event_ids",[]))
    return used


def preflight(args:argparse.Namespace)->dict[str,Any]:
    manifest_path=resolve(args.source_manifest);prediction_path=resolve(args.predictions)
    frozen_path=resolve(args.frozen_package);label_path=resolve(args.labels)
    manifest=read_json(manifest_path);frozen=read_json(frozen_path)
    source_path=resolve(manifest["events_file"])
    actual_source_sha=sha256(source_path);declared=manifest.get("source_sha256")
    if declared!=actual_source_sha:raise RuntimeError("untouched source SHA mismatch")
    if manifest.get("selection_uses_labels") is not False:raise RuntimeError("source selection is not label-blind")
    source=pd.read_csv(source_path,compression="infer",low_memory=False,dtype={"ticker":str})
    if any(column in source for column in FORBIDDEN_PREDICTION_COLUMNS):
        raise RuntimeError("untouched source contains outcome columns")
    if source.event_id.duplicated().any():raise RuntimeError("duplicate untouched event_id")
    overlap=set(source.event_id.astype(str))&prior_event_ids()
    if overlap:raise RuntimeError(f"seal source overlaps prior seals: {len(overlap)}")
    predictions=pd.read_csv(prediction_path,compression="infer",low_memory=False)
    if FORBIDDEN_PREDICTION_COLUMNS&set(predictions.columns):raise RuntimeError("prediction file contains outcomes")
    if not {"event_id","prob","high_conf"}.issubset(predictions.columns):
        raise RuntimeError("prediction schema missing event_id/prob/high_conf")
    if predictions.event_id.duplicated().any() or set(predictions.event_id.astype(str))!=set(source.event_id.astype(str)):
        raise RuntimeError("prediction/source key mismatch")
    if frozen.get("version") not in (args.version,f"V{args.version}"):
        raise RuntimeError("frozen package version mismatch")
    if frozen.get("no_leakage_audit") is not True:raise RuntimeError("frozen package leakage audit not PASS")
    if frozen.get("prediction_sha256")!=sha256(prediction_path):raise RuntimeError("frozen prediction SHA mismatch")
    dev=read_json(ROOT/f"output_V{args.version}"/"DEV_ROBUSTNESS_REPORT.json")
    gate=dev.get("selected",{}).get("research_gate",{})
    if gate.get("robust_survivor") is not True or gate.get("passed")!=gate.get("total"):
        raise RuntimeError("candidate is not a complete robust DEV survivor")
    if args.stage=="final_meta":
        parent=resolve(frozen.get("parent_research_certificate",""))
        if not parent.is_file() or read_json(parent).get("status")!="RESEARCH_SEAL_PASS":
            raise RuntimeError("FINAL_META requires a valid parent research seal PASS")
    return {"manifest_path":manifest_path,"prediction_path":prediction_path,"frozen_path":frozen_path,
        "label_path":label_path,"source_path":source_path,"source_sha":actual_source_sha,
        "source":source,"predictions":predictions,"manifest":manifest,"frozen":frozen,
        "hashes":{"manifest":sha256(manifest_path),"predictions":sha256(prediction_path),
                  "frozen_package":sha256(frozen_path),"labels":sha256(label_path)}}


def contract_check(frame:pd.DataFrame)->tuple[bool,list[str]]:
    errors=[];required={"event_id","event_group_id","market","y","fwd_ret_30m","contract_version",
        "price_cadence_sec","bar_timestamp_semantics","entry_slippage_sec","exit_slippage_sec",
        "actual_hold_seconds","event_time_utc"}
    missing=required-set(frame.columns)
    if missing:errors.append(f"missing columns: {sorted(missing)}");return False,errors
    if frame.event_id.duplicated().any():errors.append("duplicate event_id")
    if frame.event_group_id.duplicated().any():errors.append("duplicate event_group_id")
    if not frame.contract_version.astype(str).str.startswith("EXACT_T2_30M").all():errors.append("contract version")
    if not pd.to_numeric(frame.price_cadence_sec,errors="coerce").le(60).all():errors.append("cadence")
    if not frame.bar_timestamp_semantics.astype(str).str.upper().eq("OPEN").all():errors.append("bar semantics")
    if not pd.to_numeric(frame.entry_slippage_sec,errors="coerce").between(0,60).all():errors.append("entry slippage")
    if not pd.to_numeric(frame.exit_slippage_sec,errors="coerce").between(0,60).all():errors.append("exit slippage")
    if not pd.to_numeric(frame.actual_hold_seconds,errors="coerce").between(1800,1860).all():errors.append("hold")
    if not set(frame.market.astype(str)).issubset({"US","KR"}):errors.append("market")
    if not set(pd.to_numeric(frame.y,errors="coerce").dropna().astype(int)).issubset({0,1}):errors.append("label")
    return not errors,errors


def bootstrap_lower(frame:pd.DataFrame,n_boot:int=4000)->tuple[float,float]:
    working=frame.reset_index(drop=True).copy();working["day"]=pd.to_datetime(
        working.event_time_utc,utc=True).dt.date
    groups={day:np.asarray(index,dtype=int) for day,index in working.groupby("day").indices.items()}
    days=np.asarray(sorted(groups),dtype=object);rng=np.random.default_rng(20260827+777)
    ba=[];net=[]
    for _ in range(n_boot):
        position=np.concatenate([groups[day] for day in rng.choice(days,len(days),replace=True)])
        sample=working.iloc[position];truth=sample.y.to_numpy(int);prediction=sample.prob.to_numpy(float)>=.5
        if np.unique(truth).size<2:continue
        ba.append(float(balanced_accuracy_score(truth,prediction)));high=sample.high_conf.to_numpy(bool)
        if high.any():net.append(float((np.where(prediction[high],1.,-1.)*
            sample.fwd_ret_30m.to_numpy(float)[high]-COST).mean()))
    return float(np.quantile(ba,.025)),float(np.quantile(net,.025))


def evaluate(frame:pd.DataFrame)->dict[str,Any]:
    truth=frame.y.to_numpy(int);probability=frame.prob.to_numpy(float);prediction=probability>=.5
    high=frame.high_conf.to_numpy(bool);accuracy=float(accuracy_score(truth,prediction))
    signed=np.where(prediction,1.,-1.)*frame.fwd_ret_30m.to_numpy(float)-COST
    ba_lower,hc_lower=bootstrap_lower(frame)
    metrics={"n":len(frame),"US_n":int(frame.market.eq("US").sum()),"KR_n":int(frame.market.eq("KR").sum()),
        "balanced_accuracy":float(balanced_accuracy_score(truth,prediction)),
        "roc_auc":float(roc_auc_score(truth,probability)),"accuracy":accuracy,
        "accuracy_edge":accuracy-max(float(truth.mean()),1-float(truth.mean())),
        "highconf_coverage":float(high.mean()),
        "highconf_accuracy":float((prediction[high]==truth[high]).mean()) if high.any() else float("nan"),
        "highconf_strategy_mean_signed_net":float(signed[high].mean()) if high.any() else float("nan"),
        "bootstrap_BA_lower95":ba_lower,"bootstrap_HC_net_lower95":hc_lower,
        "US_BA":float(balanced_accuracy_score(truth[frame.market.eq("US")],prediction[frame.market.eq("US")])),
        "KR_BA":float(balanced_accuracy_score(truth[frame.market.eq("KR")],prediction[frame.market.eq("KR")]))}
    checks=[("total seal events >= 500",metrics["n"]>=500),("US events >= 250",metrics["US_n"]>=250),
        ("KR events >= 100",metrics["KR_n"]>=100),("balanced accuracy >= 0.540",metrics["balanced_accuracy"]>=.540),
        ("ROC-AUC >= 0.560",metrics["roc_auc"]>=.560),("accuracy edge >= +0.020",metrics["accuracy_edge"]>=.020),
        ("high-confidence coverage >= 0.15",metrics["highconf_coverage"]>=.15),
        ("high-confidence accuracy >= 0.600",metrics["highconf_accuracy"]>=.600),
        ("HC strategy mean signed net > 0 after 0.20% cost",metrics["highconf_strategy_mean_signed_net"]>0),
        ("date-block bootstrap BA lower95 >= 0.500",metrics["bootstrap_BA_lower95"]>=.500),
        ("US BA >= 0.510",metrics["US_BA"]>=.510),("KR BA >= 0.510",metrics["KR_BA"]>=.510)]
    return {"metrics":metrics,"checks":[{"name":name,"pass":bool(value)} for name,value in checks],
            "passed":int(sum(value for _,value in checks)),"total":12,"hc_bootstrap_lower_gt_zero":hc_lower>0}


def run(args:argparse.Namespace)->int:
    pre=preflight(args);source_sha=pre["source_sha"]
    intent={"created_at":now(),"stage":args.stage,"version":args.version,"source_sha256":source_sha,
            "preflight_hashes":pre["hashes"],"labels_not_read_before_this_record":True}
    seal_state=open_intent(source_sha,intent)
    try:
        labels=pd.read_csv(pre["label_path"],compression="infer",low_memory=False,dtype={"ticker":str})
        integrity,errors=contract_check(labels)
        if set(labels.event_id.astype(str))!=set(pre["source"].event_id.astype(str)):
            integrity=False;errors.append("label/source key mismatch")
        predictions=pre["predictions"].copy();predictions["event_id"]=predictions.event_id.astype(str)
        labels["event_id"]=labels.event_id.astype(str)
        frame=labels.merge(predictions[["event_id","prob","high_conf"]],on="event_id",validate="one_to_one")
        frame["prob"]=pd.to_numeric(frame.prob,errors="coerce");frame["high_conf"]=frame.high_conf.astype(bool)
        if not frame.prob.between(0,1).all():integrity=False;errors.append("invalid probability")
        evaluation=evaluate(frame) if integrity else {"metrics":{},"checks":[],"passed":0,"total":12}
        success=integrity and evaluation["passed"]==12
        status=("FINAL_META_SEAL_PASS" if args.stage=="final_meta" else "RESEARCH_SEAL_PASS") if success else \
               ("FINAL_META_SEAL_FAIL" if args.stage=="final_meta" else "RESEARCH_SEAL_FAIL")
        certificate={"schema_version":1,"created_at":now(),"stage":args.stage,"version":args.version,
            "status":status,"source_sha256":source_sha,"event_ids":sorted(frame.event_id.astype(str)),
            "event_ids_sha256":hashlib.sha256("\n".join(sorted(frame.event_id.astype(str))).encode()).hexdigest(),
            "artifact_hashes":pre["hashes"],"contract_integrity":bool(integrity),
            "contract_errors":errors,"no_leakage":pre["frozen"].get("no_leakage_audit") is True,
            "metrics":evaluation.get("metrics",{}),"checks":evaluation.get("checks",[]),
            "passed":evaluation.get("passed",0),"total":12,
            "seal_reuse_forbidden":True,"opened_once":True}
        destination=CERT/("FINAL_META" if args.stage=="final_meta" else f"V{args.version}")/source_sha
        if destination.exists():raise RuntimeError("certificate destination already exists")
        staging=CERT/".staging"/f"{args.stage}-V{args.version}-{source_sha}-{os.getpid()}"
        staging.mkdir(parents=True,exist_ok=False);atomic_json(certificate,staging/"CERTIFICATE.json")
        destination.parent.mkdir(parents=True,exist_ok=True);os.replace(staging,destination)
        atomic_json({"completed_at":now(),"status":status,"certificate":str((destination/"CERTIFICATE.json").relative_to(ROOT)),
                     "certificate_sha256":sha256(destination/"CERTIFICATE.json")},seal_state/"RESULT.json")
        if success and args.stage=="final_meta":
            completed=subprocess.run([sys.executable,str(ROOT/"verify_final_meta_seal.py"),"--certificate",
                str(destination/"CERTIFICATE.json"),"--write-marker"],cwd=ROOT,check=False)
            if completed.returncode!=0:raise RuntimeError("independent FINAL_META verifier rejected certificate")
        print(json.dumps({"status":status,"passed":evaluation.get("passed",0),"total":12,
                          "certificate":str(destination/"CERTIFICATE.json")},indent=2));return 0 if success else 30
    except Exception as error:
        atomic_json({"burned_at":now(),"status":"BURNED_AFTER_OPEN_INTENT","error":repr(error),
                     "traceback":traceback.format_exc()},seal_state/"BURNED.json")
        raise


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("research","final_meta"),required=True)
    parser.add_argument("--version",type=int,required=True);parser.add_argument("--source-manifest",required=True)
    parser.add_argument("--predictions",required=True);parser.add_argument("--labels",required=True)
    parser.add_argument("--frozen-package",required=True);args=parser.parse_args()
    try:return run(args)
    except Exception as error:print(json.dumps({"status":"SEAL_WORKER_ERROR","error":repr(error)},indent=2),file=sys.stderr);return 40


if __name__=="__main__":raise SystemExit(main())
