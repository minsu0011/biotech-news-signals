"""Rebuild the global research ledger from atomically committed version outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT/"research"
JSONL=RESEARCH/"AUTONOMOUS_EXPERIMENT_LEDGER.jsonl"
CSV=RESEARCH/"AUTONOMOUS_EXPERIMENT_LEDGER.csv"
FIELDS=("version","timestamp","hypothesis","material_change","data_sha","feature_sha",
        "US_model","KR_model","source_models","BA","AUC","edge","HC_coverage",
        "HC_accuracy","HC_net","bootstrap_BA_lower","bootstrap_HC_net_lower","US_BA",
        "KR_BA","worst_fold","research_gate_passed_count","seal_state","final_status",
        "experiment_id")


def read(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def archive(path:Path)->None:
    if not path.exists():return
    history=RESEARCH/"ledger_history";history.mkdir(parents=True,exist_ok=True)
    target=history/f"{path.stem}.pre_rebuild_{sha256(path)[:16]}{path.suffix}"
    if not target.exists():shutil.copy2(path,target)


def describe_models(comparison:dict[str,Any])->tuple[str,str,str]:
    routes=comparison.get("fixed_routes")
    if isinstance(routes,dict):
        return (str(routes.get("US_SEC","source-routed")),
                str(routes.get("KR_NEWS","source-routed")),json.dumps(routes,sort_keys=True))
    models=comparison.get("models",{})
    names=",".join(str(name) for name in models) if isinstance(models,dict) else "reported"
    return names,names,names


def build(version:int)->dict[str,Any]:
    folder=ROOT/f"output_V{version}";plan=read(folder/"VERSION_PLAN.json")
    material=read(folder/"MATERIAL_CHANGE.json");report=read(folder/"DEV_ROBUSTNESS_REPORT.json")
    run=read(folder/"RUN_STATUS.json");comparison=read(folder/"MODEL_COMPARISON.json")
    selected=report.get("selected",{});metrics=selected.get("metrics",{})
    robust=selected.get("robustness",{});bootstrap=robust.get("bootstrap",{})
    gate=selected.get("research_gate",{});market=metrics.get("by_market",{})
    folds=metrics.get("by_fold",{});worst=min((value.get("balanced_accuracy",1.)
                                               for value in folds.values()),default=None)
    us_model,kr_model,source_models=describe_models(comparison)
    return {"version":version,"timestamp":run.get("completed_at",plan.get("created_at")),
        "hypothesis":material.get("hypothesis",plan.get("hypothesis")),
        "material_change":material.get("description",plan.get("material_change")),
        "data_sha":plan.get("data_sha"),"feature_sha":plan.get("feature_sha"),
        "US_model":us_model,"KR_model":kr_model,"source_models":source_models,
        "BA":metrics.get("balanced_accuracy"),"AUC":metrics.get("auc"),
        "edge":metrics.get("edge_vs_naive",metrics.get("edge")),
        "HC_coverage":metrics.get("highconf_coverage"),"HC_accuracy":metrics.get("highconf_accuracy"),
        "HC_net":metrics.get("strategy_mean_signed_net"),
        "bootstrap_BA_lower":bootstrap.get("balanced_accuracy_lower95"),
        "bootstrap_HC_net_lower":bootstrap.get("highconf_strategy_net_lower95"),
        "US_BA":market.get("US",{}).get("balanced_accuracy"),
        "KR_BA":market.get("KR",{}).get("balanced_accuracy"),"worst_fold":worst,
        "research_gate_passed_count":gate.get("passed"),"seal_state":run.get("seal_state","UNOPENED"),
        "final_status":run.get("final_status","CONTINUE"),"experiment_id":plan.get("experiment_id")}


def atomic_write(rows:list[dict[str,Any]])->None:
    json_tmp=JSONL.with_name(JSONL.name+f".tmp.{os.getpid()}")
    with json_tmp.open("w",encoding="utf-8",newline="\n") as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)+"\n")
        handle.flush();os.fsync(handle.fileno())
    csv_tmp=CSV.with_name(CSV.name+f".tmp.{os.getpid()}")
    with csv_tmp.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(FIELDS));writer.writeheader();writer.writerows(rows)
        handle.flush();os.fsync(handle.fileno())
    os.replace(json_tmp,JSONL);os.replace(csv_tmp,CSV)


def main()->None:
    versions=[]
    for folder in ROOT.glob("output_V*"):
        try:version=int(folder.name.removeprefix("output_V"))
        except ValueError:continue
        if version>=37 and (folder/"COMMIT.json").exists():versions.append(version)
    versions=sorted(versions);expected=list(range(min(versions),max(versions)+1)) if versions else []
    if versions!=expected:raise RuntimeError(f"non-contiguous committed versions: {versions}")
    rows=[build(version) for version in versions]
    if any(row["BA"] is None or row["HC_coverage"] is None or
           row["research_gate_passed_count"] is None for row in rows):
        raise RuntimeError("required ledger metric missing")
    archive(JSONL);archive(CSV);atomic_write(rows)
    print(json.dumps({"rows":len(rows),"versions":[versions[0],versions[-1]],
        "jsonl_sha256":sha256(JSONL),"csv_sha256":sha256(CSV),
        "required_metrics_complete":True},indent=2))


if __name__=="__main__":main()
